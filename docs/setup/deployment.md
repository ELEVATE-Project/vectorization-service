# Deployment

**Purpose.** Read this page to understand how the service is deployed to QA and production: the Ansible playbook, how the `.env` file is produced from Vault, the systemd unit, the release 1.0.0 upgrade procedure, and the places where those artifacts disagree with each other or with the code.

Deployment assets live in `deployment/` (`ansible.yml`, `json2env.sh`, `templates/vectorization-service-uvicorn.j2`) and `release-doc/release-1.0.0.md`.

## 1. Topology

```text
  Ansible control host
        |
        v   (hosts: elevate, become: yes)
  Target VM  /opt/deployment/
        |-- .token                         Vault token (read by playbook)
        |-- releases/<YYYYmmddHHMMSS>/     git clone of the requested branch
        '-- vectorization-service/         "current" deployment (recreated every run)
              |-- .env                     generated from Vault JSON
              |-- .venv/                   uv virtualenv
              |-- deployment/json2env.sh
              '-- vectorization-service.log   stdout+stderr of uvicorn (systemd append)
  systemd: vectorization-service-uvicorn.service
        '-- uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
  External: Qdrant (QA 1.12, production 1.18), Redis, PostgreSQL
```

`ENVIRONMENT` in the generated `.env` must be set to something other than `local` on a server. `app/main.py` then uses `root_path="/vector"`, so the reverse proxy is expected to publish the service under `/vector` (for example `/vector/api/health`).

## 2. Ansible playbook (`deployment/ansible.yml`)

The play targets host group `elevate` with `become: yes`. Required extra variables (not defined in the file): `gitBranch`, `vaultAddress`. Defined `vars`: `project_path=/opt/deployment`, `current_path=/opt/deployment/vectorization-service`, `uvicorn_port=9000`, `uvicorn_workers=4`.

Task order and what each does:

| # | Task | Behaviour |
|---|---|---|
| 1 | Read Vault token | `slurp` of `/opt/deployment/.token`. |
| 2-3 | Release path and directory | `releases/<timestamp>` via `date +%Y%m%d%H%M%S`. |
| 4 | Clone repository | `https://github.com/ELEVATE-Project/vectorization-service.git` at `gitBranch`. |
| 5 | Download secrets | `curl --silent --insecure --location GET {{ vaultAddress }}vectorization-service` with `X-Vault-Token`, piped through `jq '.data.data'` into `<release>/data2.json`. Note `--insecure` disables TLS verification. |
| 6-8 | Swap deployment | Deletes `current_path`, recreates it, `mv release/* current/`. This removes the previous `.venv`, `.env`, and log file on every deploy. Dotfiles in the release (for example `.git`) are not moved because the glob is `*`. |
| 9 | `chmod 0744 deployment/json2env.sh` | |
| 10 | Generate `.env` | `cat data2.json | json2env.sh > .env`. |
| 11-13 | Python | `uv venv` (creates `.venv`), `uv pip install -r requirements.txt`, `python -m spacy download en_core_web_sm`. |
| 14 | Create `logs/` | Directory is created but nothing writes to it. |
| 15 | "Run Django migrations" | Misnamed. The task only runs `export $(cat .env | xargs)` in a shell, which proves the `.env` is parseable by `xargs` and changes nothing else. There are no migrations (no Django, no Alembic; tables are created by `Base.metadata.create_all` at import). |
| 16-18 | systemd | Renders `templates/vectorization-service-uvicorn.j2` to `/etc/systemd/system/`, `daemon-reload` handler, enable, restart. |
| 19-20 | Verification | `systemctl is-active`; on failure dumps `journalctl -u vectorization-service-uvicorn -n 50`. |

Note: `uvicorn_port: 9000` and `uvicorn_workers: 4` are declared but never referenced; the template hardcodes `--port 8000 --workers 4`. Changing the Ansible variables has no effect.

Note: the playbook never installs OS packages (Tesseract, Poppler), never checks Qdrant, and never runs the sparse back-fill. Those are manual prerequisites (see [Developer setup](developer_setup.md) and section 5 below).

Note: the playbook does not wait for the application to become healthy. `systemctl is-active` reports `active` as soon as the processes are forked, even if each worker later crashes while importing (for example PostgreSQL unreachable). With `Restart=always` and `RestartSec=5`, a crash loop appears as `active (auto-restart)` intermittently. Check the log file.

## 3. `.env` generation (`deployment/json2env.sh`)

```sh
tr -d '\n' |
grep -o '"[A-Za-z_][A-Za-z_0-9]*"\s*:\s*\("[^"]*"\|[0-9.]*\|true\|false\|null\)' |
sed 's/"\([^"]*\)"\s*:\s*"\?\([^"]*\)"\?/\1="\2"/'
```

Behaviour and limits:

- Input is the flat Vault secret JSON (`.data.data`). Newlines are stripped, each `"KEY": value` pair is extracted, and rewritten as `KEY="value"`.
- Booleans and numbers are emitted as quoted strings (`HYBRID_SEARCH_ENABLED="true"`), which `python-dotenv` and pydantic accept.
- Only scalar values are matched. Nested objects or arrays are not representable; keys inside nested objects would be flattened into the output.
- A value containing a double quote or backslash is truncated at the first `"`; a value with no match (for example an empty array) is dropped silently.
- `null` becomes `KEY="null"` (the string).

Because the secret store is the source of truth, any setting you add to `app/config.py` that must differ per environment has to be added to the Vault secret `vectorization-service`, not only to `.env.sample`.

## 4. systemd unit (`templates/vectorization-service-uvicorn.j2`)

```ini
[Service]
Type=simple
User=root
Group=root
WorkingDirectory=/opt/deployment/vectorization-service
Environment="PYTHONPATH=/opt/deployment/vectorization-service"
ExecStart=/opt/deployment/vectorization-service/.venv/bin/uvicorn \
    app.main:app --host 0.0.0.0 --port 8000 --workers 4
Restart=always
RestartSec=5
StandardOutput=append:/opt/deployment/vectorization-service/vectorization-service.log
StandardError=append:/opt/deployment/vectorization-service/vectorization-service.log
```

Operational consequences:

- The unit does not load `.env` through `EnvironmentFile`. Settings reach the process because `app/config.py` calls `load_dotenv()` from the working directory (`WorkingDirectory` makes this work).
- The service runs as `root`.
- **Four workers means four copies of every module-level singleton.** Each worker loads the SentenceTransformer model, the spaCy model (lazily), the fastembed BM25 model (lazily), and runs `ensure_collections_exist()` at lifespan startup. Size memory for 4x the model footprint, and expect four near-simultaneous idempotent startup runs against Qdrant.
- Each worker runs `logging.basicConfig` with `FileHandler('app.log')` (from `language_utils.py`), so `app.log` in the working directory is appended to by all four processes in addition to the systemd log file.
- Logs have no rotation configured.
- `/api/health` returns 503 unless `REDIS_CACHE_ENABLED=true` (see [Troubleshooting](../operations/troubleshooting.md)); do not wire a load balancer health probe to it without setting that variable.

## 5. Release 1.0.0 upgrade procedure (`release-doc/release-1.0.0.md`)

Release 1.0.0 introduced hybrid search (dense plus BM25) and the move to `qdrant-client` 1.18. The document's own warning applies: deploy code and client together.

Condensed procedure:

1. Confirm `pip show qdrant-client` reports `>=1.18.0`.
2. Snapshot the collection: `POST /collections/<COLLECTION_NAME>/snapshots` on Qdrant, list, and download.
3. Add the new env keys to the Vault secret with `SPARSE_SEARCH_ENABLED=false` so no half-migrated state is served.
4. `uv pip install -r requirements.txt` and install the spaCy model (the playbook does both).
5. Back-fill BM25 vectors for existing documents with `scripts/migrate_to_sparse_vectors.py` (see [Scripts](../operations/scripts.md)).
6. Set `SPARSE_SEARCH_ENABLED=true` and restart.

Rollback: revert code, pin `qdrant-client[fastembed]>=1.9.0,<1.14.0`, set `SPARSE_SEARCH_ENABLED=false`, restart. Sparse field and prefix indexes are inert and can stay.

### Where the release document disagrees with code or other docs

| Release document says | Code or other source says |
|---|---|
| Qdrant server `>=1.18` is required. | QA runs server 1.12 with client 1.18, and the service is verified on it (`issueOnCompactabilityWithServerVersion1.12.md`, [Qdrant compatibility](../operations/qdrant_compatibility.md)). Only production is on 1.18. |
| Step 4 env block sets `QDRANT_CHECK_COMPATIBILITY=true`. | The config default and `.env.sample` are `false`; setting `true` on QA brings back the blanket `UserWarning`. Use `false` until the server reaches 1.18. |
| Step 4 env block sets `INCLUDE_SCORING_DEBUG=true`. | `.env.sample` and `config.py` comments say keep it `false` in production (responses are large because the default `top_k` is 1,000,000). |
| Candidate limit is `min(top_k x 20, 10000)`. | Code and `.env.sample`: `min(top_k x 8, 2000)` (`SEARCH_CANDIDATE_FANOUT=8`, `SEARCH_CANDIDATE_MAX=2000`). |
| Dense-only weights are 0.36/0.27/0.14/0.14/0.09. | `SEARCH_PRIORITY_WEIGHTS` default is 0.34/0.26/0.20/0.12/0.08. |
| "the service uses `documents1`" and the migration defaults to `documents`; commands pass `COLLECTION_NAME=documents1`. | Code default for `COLLECTION_NAME` is `documents` in both. `documents1` is a deployment-specific value that must be present in the Vault secret. |
| Step 7 runs the in-place back-fill while the service has `SPARSE_SEARCH_ENABLED=false`. | In-place mode needs the `bm25` sparse field to already exist on the collection. With the service flag off, startup never adds it (`ensure_collections_exist` only calls `_ensure_sparse_vector_field` when the flag is true). The script's own docstring states that sparse fields cannot be added to an existing collection and recommends blue-green mode (`--new-collection`) for collections without the field. Follow the script docstring: use `--new-collection` for any collection created before sparse support. |
| The inline `SPARSE_SEARCH_ENABLED=true` "applies only to this command". | The script never reads `SPARSE_SEARCH_ENABLED` (it appears only in its docstring), and it does not load `.env`; see [Scripts](../operations/scripts.md). |

## 6. Pre-deploy checklist (derived from code)

- Vault secret contains: `QDRANT_HOST`, `QDRANT_PORT`, `COLLECTION_NAME`, `POSTGRES_DATABASE_URI`, `REDIS_HOST`, `REDIS_PORT`, `ENVIRONMENT` (not `local`), and feature flags (`SPARSE_SEARCH_ENABLED`, `HYBRID_FUSION_METHOD`).
- PostgreSQL reachable from the VM and the role can create tables (import-time `create_all`).
- Redis reachable; `REDIS_CACHE_ENABLED=true` if `/api/health` is used as a probe.
- OS packages: `tesseract-ocr` and Poppler installed.
- Qdrant server version known (1.12 or 1.18) and `QDRANT_CHECK_COMPATIBILITY` set accordingly.
- Enough RAM for four workers each holding the embedding model.
- After restart: `GET /vector/api/health` (or `/api/health` locally), then a `POST /api/documents/search` smoke query.

## Known issues / gotchas

- `uvicorn_port` and `uvicorn_workers` Ansible vars are dead; the template is hardcoded to 8000 and 4.
- The "Run Django migrations" task is a no-op environment export.
- The playbook wipes `current_path`, including `.venv`, on every deploy and re-installs dependencies each time.
- `--insecure` is used when fetching secrets from Vault.
- `User=root` and no `EnvironmentFile`.
- The release document contradicts the QA compatibility document on server version, and its env block diverges from `.env.sample`.
- Deployment does not verify health after restart.

## Related pages

- [Developer setup](developer_setup.md)
- [Configuration reference](configuration.md)
- [Scripts](../operations/scripts.md)
- [Qdrant compatibility](../operations/qdrant_compatibility.md)
- [Troubleshooting](../operations/troubleshooting.md)
- [App lifecycle](../backend/app_lifecycle.md)
