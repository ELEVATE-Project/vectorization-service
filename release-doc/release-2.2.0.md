# Release 2.2.0 — Ingestion Validation, Metadata Cleanup, Hybrid Score Fixes & Acronym-Aware Search

**Service:** vectorization-service

> ⚠️ **Run the metadata cleanup script during this deploy.** Existing points still carry
> `title`/`summary`/`tags` duplicated inside `payload.metadata`; this release's code
> assumes they live only at the top level. See [Deployment](#deployment).

> ⚠️ **Run `alembic upgrade head` before starting the service.** The acronym cache warms at startup by reading the `acronym_mapping` table. If the table does not exist yet, warm-up fails (logged, non-fatal) and acronym detection finds nothing until the migration has run.

> ℹ️ **About [release 2.1.0](release-2.1.0.md).** That note describes an earlier, unreleased design of acronym search (a ranking bonus, a UUID `code` key). This release is the first to ship acronym search; where the two notes differ, this one applies.

---

## Table of Contents

- [What's New](#whats-new)
- [How Acronym Search Works](#how-acronym-search-works)
- [API](#api)
- [New Configuration](#new-configuration)
- [Dependencies](#dependencies)
- [Deployment](#deployment)
- [Rollback](#rollback)

---

## What's New

**New implementation**

- **New: theme support, end to end** — upload/replace/upsert accept an optional `theme`
  form field (stored top-level, never embedded, so it can't move a search score). A new
  `PATCH /documents/{source_id}/theme` endpoint sets or changes the theme alone, without
  reprocessing the file. Search gets matching `themes` / `exclude_themes` filters, and
  `theme` is now returned on every search result.
- **Ingestion is now validated up front** — `source_id`, `priority`, `title`/`summary`/`tags`,
  and metadata (`company`/`source_id`/`markdown_url`) are checked and normalized before any
  file parsing or embedding. Bad input now fails fast with a 400 instead of being silently
  dropped or overwritten.
- **No more silent partial uploads** — an empty file, or a batch that partially fails to
  write to Qdrant, now fails the request (500/502) instead of returning `201`. Any points
  already written for a failed request are rolled back automatically.
- **Search results flag keyword-injected matches** — `SearchResultItem.match_source` marks
  results injected by the title/summary keyword-match step, so their floor score isn't
  mistaken for a real semantic score.
- **Stricter tag validation** — a blank or non-string entry anywhere in the `tags` array is
  now a 400 instead of being silently stored; duplicate tags are de-duplicated.
- **Postman collection added** — `postman/vectorization-service.postman_collection.json`
  covers every endpoint with a request per success/validation case and a saved example
  response. `README.md` documents how to import it and generate the OpenAPI file.

**Acronym search**

- **Acronym detection in search queries.** A query containing a registered acronym (e.g. `DIET`, `PTM`, `SSC`) is detected and expanded before retrieval, so a search for the acronym also finds documents that only spell out its full meaning. Gated by `ACRONYM_SEARCH_ENABLED` (default `true`).
- **Acronym-aware ranking.** Acronym queries are searched with the acronym written out (`ACRONYM_SEARCH_METHOD`: `inline`, the default, or `dual_query`), then get the same title and summary boosts as any other query (the query as typed). A title that has the query with the acronym written out ("PTM Doc" → "Parent Teacher Meeting Doc") gets an extra boost on top: x1.7 when the title equals it, x1.4 when the title contains it. The summary gets the same rule (x1.4 / x1.2).
- **Acronym dictionary.** New Postgres table `acronym_mapping`, seeded with 486 acronyms. One acronym can have several meanings (`SSC` → "Staff Selection Commission" / "Sainik School Society"). Lookups go through a Redis cache.
- **Bulk upload endpoint.** `POST /api/acronyms/bulkUpload` creates, updates, enables or disables acronyms from a CSV, and reports which acronyms were created, updated and deactivated.
- **`acronym_info` in the search response**, showing whether acronym search is enabled, the search method in use, whether an acronym was detected, and its expansions.
- **Scoring debug output** for acronym queries (with `include_scoring_debug=true`), so the final score can be rebuilt from its parts.

**New scripts**

- **Metadata cleanup script** — `scripts/release_2.2.0/cleanup_metadata_duplicates.py`
  removes the now-redundant `title`/`summary`/`tags`/`theme` copies from `payload.metadata`
  on existing points (backfilling the top level first for any legacy point that only had
  the value there). Idempotent — safe to re-run.
- **New: vector-coverage audit script** — `scripts/release_2.2.0/analyze_missing_vectors.py`
  is a read-only report of how many points are missing their `title`/`summary`/`tags` named
  vectors (documents uploaded without those fields never get them). With `--theme`, it also
  reports theme coverage per document/point. Talks to Qdrant directly; does not start the
  service or load the embedding model.

**Fixes on current**

- **`title`/`summary`/`tags` live only at the payload top level now** — no longer duplicated
  inside `metadata`. Existing points need a one-time cleanup script run (see Deployment).
- **Fixed a hybrid-search scoring bug** — a BM25 query that matched nothing was still
  capping dense scores at 70% of scale (the configured `0.7×dense` weight). Dense now gets
  full weight for that query when sparse contributes nothing.
- **Replace (`PUT`/upsert) is now safe against failed uploads** — previously the old
  version was deleted *before* the new one was uploaded, so a failed upload could lose the
  document entirely. It now uploads the new version first and only deletes the old chunks
  (by id) once that succeeds; a failed upload leaves the existing document untouched.
- **Replace no longer drops `title`/`summary`/`tags`/`theme`** — `PUT`/upsert previously
  forwarded only the file and metadata to the upload step, silently losing these fields on
  every update. They're now accepted as form fields and carried through like `POST`.
- **`PATCH /documents/{source_id}/metadata` no longer scrolls and read-modify-writes** —
  the old implementation fetched every chunk in batches of 100, merged client-side, and
  wrote each one back (not safe under concurrent updates). It's now a single server-side
  `set_payload` merge over a filter, independent of chunk count.

---

## How Acronym Search Works

A query that contains a registered acronym is searched with the acronym written out: `inline` (the default) uses one text, "PTM (Parent Teacher Meeting) documents"; `dual_query` searches the query and each meaning separately. Documents whose title or summary has the written-out query get an extra boost. Queries without an acronym are searched exactly as before. Acronyms come from Postgres through a Redis cache; if Redis or Postgres is down, search still works.

For the details (detection rules, search texts, boosts, cache, upload checks, known limits), see the [acronym design notes](acronym-design-notes.md).

---

## API

<details>
<summary><code>POST /api/documents/search</code> — <code>acronym_info</code></summary>

Always an object for a request with query text (`null` only without query text):

| Situation | Value |
|---|---|
| `ACRONYM_SEARCH_ENABLED=false` | `{"enabled": false, "detected": false}` |
| Enabled, no acronym in the query | `{"enabled": true, "detected": false, "method": "inline", "method_description": {"semantic": "...", "keyword": "...", "boosting": "..."}}` |
| Acronym detected | `{"enabled": true, "detected": true, "method": "inline", "method_description": {...}, "mapping": {"DIET": ["District Institute of Education and Training"]}}`, plus `"ambiguous": ["SSC"]` for acronyms with several meanings |

`method` is the `ACRONYM_SEARCH_METHOD` in use (`inline` or `dual_query`). `method_description` explains it with three keys, `semantic`, `keyword` and `boosting` (with the configured boost values):

| Key | `inline` | `dual_query` |
|---|---|---|
| `semantic` | One search text with each acronym's meaning written next to it, e.g. "PTM (Parent Teacher Meeting) documents". | The query and the query with each acronym written out ("Parent Teacher Meeting documents") are searched separately; each field keeps the higher score. |
| `keyword` | One search with the words of the query and of every meaning; with `ACRONYM_KEYWORD_OR=true`, one search per meaning as for `dual_query`. | One search for the query and one for each meaning; each chunk keeps the higher score. |
| `boosting` | A title that equals or contains the query with the acronym written out gets x1.7 or x1.4, a summary x1.4 or x1.2, on top of the normal title and summary boosts (hybrid mode only). | Same |

</details>

<details>
<summary><code>POST /api/documents/search</code> — debug fields (<code>include_scoring_debug=true</code>)</summary>

Acronym queries add `title_expansion_multiplier`, `summary_expansion_multiplier`, `title_expansion_match` and `summary_expansion_match` to each result, and the expansion boosts to `search_config.scoring_context`. The field descriptions are in the API docs (`/docs`); how they are computed is in the [acronym design notes](acronym-design-notes.md).

</details>

<details>
<summary><code>POST /api/acronyms/bulkUpload</code></summary>

Requires the `internal_access_token` header.

| Column | Required | Notes |
|---|---|---|
| `acronym` | Yes | Case-insensitive, stored uppercase, max 32 characters. Letters only, 1 to `ACRONYM_MAX_PHRASE_WORDS` space-separated words (detection drops digits and punctuation, so `G20` or `COVID-19` cannot be registered). |
| `expansions` | Yes, unless `is_active` is `false` (then ignored) | Several meanings separated by `\|`, e.g. `Staff Selection Commission\|Sainik School Society` |
| `description` | No | Free text |
| `is_active` | No | `true` / `false`; missing means active. Any other value is a row error. `false` only turns off an existing acronym; its stored expansions and description stay. An acronym that does not exist yet is a row error (it cannot be added as inactive). |

Column names are trimmed and case-insensitive. A duplicate or unsupported column, or an unnamed column that holds values, rejects the file (400); an empty unnamed column (a trailing comma) is ignored. Each row must have as many fields as the header; a value that contains a comma must be in double quotes. A row that breaks a rule is reported in `errors`.

Response: `received`, `created`, `updated`, `deactivated`, `cache_refreshed`, `errors` (one entry per rejected row; the other rows still save), and `created_acronyms`, `updated_acronyms`, `deactivated_acronyms` in CSV order. Uploads are upserts on `acronym`, so re-uploading a file is safe. A database error returns 503 (unreachable) or 500 and says the outcome could not be confirmed.

`GET /api/acronyms` (header `internal_access_token`) lists the dictionary, with `prefix`, `is_active`, `limit` and `offset` filters. Each item has `acronym`, `expansions`, `description`, `is_active`, `created_by`, `updated_by`, `created_at` and `updated_at`.

</details>

---

## New Configuration

### Ingestion and upload

Add these to `.env` (defaults shown match `app/config.py`, so omitting them is safe too):

| Variable | Default | Description |
|---|---|---|
| `MAX_SOURCE_ID_LENGTH` | `255` | Max length of `source_id` on new uploads |
| `SOURCE_ID_PATTERN` | `^[A-Za-z0-9_\-.:]+$` | Allowed character set for `source_id` on new uploads (must fully match) |
| `PRIORITY_PATTERN` | `^P\d+$` | Required format for `priority` (upper-cased before matching) |
| `UPLOAD_ROLLBACK_MAX_ATTEMPTS` | `3` | In-request rollback retries for a partial upload |
| `UPLOAD_ROLLBACK_RETRY_WAIT_SECONDS` | `0.5` | First in-request retry wait (doubles each attempt) |
| `UPLOAD_ROLLBACK_BACKGROUND_MAX_ATTEMPTS` | `9` | Background rollback retries after the request has already returned a 502 |
| `UPLOAD_ROLLBACK_BACKGROUND_FIRST_WAIT_SECONDS` | `2` | First background retry wait (doubles each attempt) |
| `UPLOAD_ROLLBACK_BACKGROUND_MAX_WAIT_SECONDS` | `60` | Cap on each background wait (~5 min of total retrying by default) |

> `QDRANT_CHECK_COMPATIBILITY` default also flipped from `false` to `true` — every
> environment now runs Qdrant server 1.18.2, which is within the client's allowed
> client/server version gap, so the compatibility check is safe to leave on.

### Acronym search

| Variable | Default | Purpose |
|---|---|---|
| `ACRONYM_SEARCH_ENABLED` | `true` | Master switch for acronym search |
| `INTERNAL_API_TOKEN` | — (required) | Header `internal_access_token` for the acronym endpoints. The service does not start if it is not set. |
| `ACRONYM_BULK_UPLOAD_MAX_SIZE_MB` | `5` | Largest CSV accepted |
| `ACRONYM_BULK_UPLOAD_BATCH_SIZE` | `200` | Rows saved per database statement during an upload (1 to 10000) |
| `ACRONYM_EXPANSION_EXACT_TITLE_BOOST` | `1.7` | Title multiplier when the title equals the query with the acronym written out |
| `ACRONYM_EXPANSION_PARTIAL_TITLE_BOOST` | `1.4` | Title multiplier when the title contains the query with the acronym written out |
| `ACRONYM_EXPANSION_EXACT_SUMMARY_BOOST` | `1.4` | Summary multiplier when the summary equals the query with the acronym written out |
| `ACRONYM_EXPANSION_PARTIAL_SUMMARY_BOOST` | `1.2` | Summary multiplier when the summary contains the query with the acronym written out |
| `ACRONYM_SEARCH_METHOD` | `inline` | `inline` (one text with the meanings written in) or `dual_query` (one text per meaning, higher score per field); empty means `inline` |
| `ACRONYM_KEYWORD_OR` | `false` | One keyword (BM25) search per meaning, each chunk keeping its higher score; always on with `dual_query` |
| `ACRONYM_MAX_DENSE_VARIANTS` | `3` | Dense query variants per acronym query, including the original (dual_query only; keyword OR searches every meaning) |
| `ACRONYM_MAX_PHRASE_WORDS` | `4` | Longest multi-word acronym detected and accepted on upload |
| `ACRONYM_MAX_QUERY_WORDS` | `30` | Words of the query checked for acronyms; the rest is still searched, only not looked up, which bounds Redis/Postgres work for very long queries |
| `ACRONYM_MIN_PREFIX_MATCH_LEN` | `4` | Shortest word stem allowed to prefix-match an expansion word |
| `ACRONYM_PREFIX_SUFFIX_CAP` | `3` | Most extra characters in such a prefix match ("tests" ~ "test") |
| `CACHE_ENABLED` | `true` | Use Redis for acronym lookups |
| `REDIS_DB` | `2` | Redis database for the acronym cache (commons uses db 0 on a shared Redis) |
| `REDIS_CACHE_TTL` / `REDIS_NEGATIVE_CACHE_TTL` / `REDIS_DB_ERROR_CACHE_TTL` | `86400` / `3600` / `30` | Cache lifetimes (seconds) for hits, "not an acronym" and database-outage misses |
| `REDIS_SOCKET_CONNECT_TIMEOUT` / `REDIS_SOCKET_TIMEOUT` | `1` / `1` | Redis timeouts (seconds) |
| `POSTGRES_CONNECT_TIMEOUT` / `POSTGRES_STATEMENT_TIMEOUT_MS` | `3` / `5000` | Postgres connect timeout (seconds) and per-statement timeout |

Existing settings that acronym search now also depends on:

| Variable | Why |
|---|---|
| `POSTGRES_DATABASE_URI` | The `acronym_mapping` table; also used by `alembic upgrade head` |
| `REDIS_HOST` / `REDIS_PORT` / `REDIS_PASSWORD` | The acronym cache (with `CACHE_ENABLED=true`) |
| `SPARSE_SEARCH_ENABLED` | Keyword (BM25) search, which also gets the meanings' words |
| `HYBRID_SEARCH_ENABLED` | Title/summary boosts, including the acronym expansion boosts |

---

## Dependencies

- Still requires `qdrant-client[fastembed]>=1.18.0,<2.0.0` and Qdrant server 1.18.2 (the
  version every environment runs).
- **`alembic`** manages the `acronym_mapping` schema.
- **`POSTGRES_DATABASE_URI` must use the `postgresql://` format** (driver `psycopg2-binary`). A `postgresql+psycopg://` URL no longer works, because psycopg 3 is not installed.
- **`click`** is an explicit dependency: `spacy` imports it, and recent `typer` releases no longer pull it in.
- The stopword list is `spacy`'s built-in list; no extra model download. The `en_core_web_sm` model is installed by the Ansible deploy step.
- **`data/acronyms.csv`** (486 acronyms) is committed; the migration seeds the table from it.

---

## Deployment

### Pre-deploy checklist

1. **Confirm current callers' `priority` and `source_id` values are compliant** — check
   recent ingestion requests (or with upstream teams, e.g. commons-backend) for priorities
   that aren't `P<digits>` or source_ids with characters outside `[A-Za-z0-9_\-.:]`.
   Non-compliant callers will start getting 400s on new uploads.

2. **Back up the collection — take a Qdrant snapshot before doing anything else:**
   ```bash
   # Create a snapshot of the collection
   curl -X POST "http://<QDRANT_HOST>:<QDRANT_PORT>/collections/<COLLECTION_NAME>/snapshots"

   # List snapshots to get the snapshot name and confirm it was created
   curl "http://<QDRANT_HOST>:<QDRANT_PORT>/collections/<COLLECTION_NAME>/snapshots"

   # Download the snapshot locally for safekeeping
   curl -o backup-<COLLECTION_NAME>-$(date +%Y%m%d).snapshot \
     "http://<QDRANT_HOST>:<QDRANT_PORT>/collections/<COLLECTION_NAME>/snapshots/<snapshot_name>"
   ```

3. **Postgres is reachable**, and `POSTGRES_DATABASE_URI` uses the `postgresql://` format (not `postgresql+psycopg://`). `acronym_mapping` is new; the existing `translations` table is untouched.

4. **BM25 coverage** (recommended): add BM25 vectors to any chunks that lack them; a chunk without one gets no keyword score. Already-indexed points are skipped, so it is safe to re-run:
   ```bash
   PYTHONPATH=. COLLECTION_NAME=<collection> SPARSE_SEARCH_ENABLED=true \
     .venv/bin/python3 scripts/migrate_to_sparse_vectors.py --dry-run   # report only
   PYTHONPATH=. COLLECTION_NAME=<collection> SPARSE_SEARCH_ENABLED=true \
     .venv/bin/python3 scripts/migrate_to_sparse_vectors.py
   ```

### Deploy steps

5. Pull the new code and add the new env keys to `.env` (see [New Configuration](#new-configuration) and `.env.sample`):
   ```dotenv
   MAX_SOURCE_ID_LENGTH=255
   SOURCE_ID_PATTERN='^[A-Za-z0-9_\-.:]+$'
   PRIORITY_PATTERN='^P\d+$'
   ACRONYM_SEARCH_ENABLED=true
   SPARSE_SEARCH_ENABLED=true
   INTERNAL_API_TOKEN=<generate-a-real-secret>
   ```
   `INTERNAL_API_TOKEN` is required. The `UPLOAD_ROLLBACK_*` retry/backoff variables and the
   other acronym settings are optional tuning knobs — their defaults already match the code,
   so omitting them is safe.

6. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

7. **Run the migrations** (`deployment/ansible.yml` already does this):
   ```bash
   alembic upgrade head
   ```
   This creates `acronym_mapping` (primary key `id`, an identity column that is `GENERATED ALWAYS`; `acronym` unique) and seeds it from `data/acronyms.csv`. On a database that already has the table from a pre-release build, the second migration (`5b7dea819fa4`) converts it to the same shape and keeps every row ([design notes, Database](acronym-design-notes.md#database)).

8. **Optional — audit current vector/theme coverage** before cleaning up, as a baseline to
   compare against afterwards. Read-only; talks to Qdrant directly and does not start the
   service.
   ```bash
   PYTHONPATH=. .vector-env/bin/python3 scripts/release_2.2.0/analyze_missing_vectors.py --theme
   ```

9. **Run the metadata cleanup script** — removes the now-redundant `title`/`summary`/`tags`/
   `theme` copies from `payload.metadata` on existing points (backfilling the top level first
   for any legacy point that only had the value there). Idempotent — safe to re-run, and safe
   to run before, during, or after the code deploy/restart; it only patches payloads via
   `set_payload` and does not touch vectors, so it can run against a live collection while
   the previous code version is still serving traffic.
   ```bash
   # Dry run first — reports scanned/cleaned/backfilled counts, writes nothing
   PYTHONPATH=. .vector-env/bin/python3 scripts/release_2.2.0/cleanup_metadata_duplicates.py --dry-run

   # Real run
   PYTHONPATH=. .vector-env/bin/python3 scripts/release_2.2.0/cleanup_metadata_duplicates.py
   ```

10. Restart the service. This also creates the new `theme` payload index automatically
    (`create_payload_index` is idempotent and runs on every startup) — no manual indexing
    step needed. It also runs on the first request if the process isn't restarted.
    Check the log line:
    ```
    Acronym cache loaded: 486 active acronym(s)
    ```
    A warning instead ("Acronym cache load failed …") means the service booted with a cold cache; lookups still work through Postgres.

11. **Smoke test:**
    - Upload a document with a valid `source_id`/`priority` → expect `201`.
    - Upload with a non-numeric priority (e.g. `priority=Pxyz`) → expect `400`.
    - Run a search with `include_scoring_debug: true` for a query where BM25 matches
      nothing and confirm `search_config.scoring_context.sparse_has_hits` is `false` and
      scores are no longer capped at 0.7.
    - Spot check a few pre-existing documents via `search` and confirm `title`/`summary`/`tags`
      still return correctly at the top level of each result.
    - Upload with a `theme`, then `PATCH /documents/{source_id}/theme` to change it, and
      confirm `search` with a matching `themes` filter returns it and `exclude_themes` drops it.
    - Replace a document (`PUT`) with a `title`/`summary`/`tags`/`theme` and confirm they
      come back in the response and in a subsequent search (previously dropped on replace).
    - Search for an acronym:
      ```bash
      curl -s -X POST "http://<HOST>:<PORT>/api/documents/search" -H 'Content-Type: application/json' \
        -d '{"query":"DIET","top_k":5,"filter_score":0.35,"include_scoring_debug":true}' \
        | jq '{acronym_info, boost_config: .search_config.scoring_context.boost_config,
               top: [.results[] | {title, score, normalized_dense, normalized_sparse, title_multiplier}]}'
      ```
      Expect `acronym_info` with `"enabled": true, "detected": true` and the DIET expansion, `boost_config` with the configured multipliers (not 1.0), and `title_multiplier` above 1.0 only on titles that contain "diet".

12. **Optional — add or update acronyms:**
    ```bash
    curl -X POST "http://<HOST>:<PORT>/api/acronyms/bulkUpload" \
      -H "internal_access_token: <INTERNAL_API_TOKEN>" \
      -F "file=@acronyms.csv;type=text/csv"
    ```

---

## Rollback

### Ingestion and metadata changes

Revert the code and restart. The cleanup script's changes are **not** reverted automatically
— `payload.metadata` will stay without the `title`/`summary`/`tags`/`theme` duplicates, but
the previous code version reads `title`/`summary`/`tags` from the top level as well (this was
already true in the prior release), so rollback is safe without re-running any script. The
previous code version does not know about `theme` at all, so it simply ignores that payload
key — searches just stop supporting theme filtering until the code is rolled forward again.
No vector or collection schema changes were made in this release (the new `theme` payload
index is additive and harmless to leave in place).

#### If a full data restore is needed

Code-only rollback (above) is sufficient for this release. If anything goes wrong beyond
that — the cleanup script corrupts or drops payload data — restore the collection from the
pre-deploy snapshot taken in [step 2](#pre-deploy-checklist), then revert the code and
re-run the smoke test.

### Acronym search

**Fastest — flag off:** set `ACRONYM_SEARCH_ENABLED=false` and restart. Detection never runs and search ranks exactly as without the feature; `acronym_info` reports `{"enabled": false, "detected": false}`.

**Schema:** `alembic downgrade fcd65d39a795` drops `acronym_mapping`. ⚠️ Destructive: acronyms added or edited through the bulk endpoint are lost. Take a Postgres backup first (`pg_dump -t acronym_mapping`).

---
