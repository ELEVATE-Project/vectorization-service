# Scripts and Utilities

**Purpose.** Read this page before running anything under `scripts/` or the loose utility files in the repository root. It explains what each script does, which environment it reads, and which ones are stale. `migrate_to_sparse_vectors.py` is the only script with production impact and is documented in depth.

## 1. Inventory

| File | Type | Purpose | Status |
|---|---|---|---|
| `scripts/migrate_to_sparse_vectors.py` | Python, CLI | Add BM25 sparse vectors to existing points (in place) or copy to a new collection with the full schema (blue-green) | Current; required for the hybrid-search rollout |
| `rename_key_entities_field.py` (repo root) | Python | Rename `metadata."DOCUMENT TYPE"` to `metadata.DOCUMENT_TYPE` on every point | One-off data fix; historical |
| `scripts/insert_document.py` | Python library plus demo `__main__` | Upload documents through `POST /api/documents` | Working, but the demo block has hardcoded developer paths |
| `scripts/quick_insert_example.py` | Python, edit-and-run | Single upload with variables at the top | Working |
| `scripts/insert_document_curl.sh` | Bash | Upload through `curl` | Working |
| `scripts/data_insert_script.py` | Python | Bulk-upload a folder via `curl` subprocess | Stale (wrong endpoint) |
| `scripts/README_INSERT.md` | Docs | Usage notes for the three insert scripts | Accurate for endpoint `POST /api/documents` |
| `example_prioritized_search.py` (repo root) | Python | Example client for the search endpoint | Stale (wrong path) |
| `test_q.py` (repo root) | Empty file | none | Dead |
| `test_url_extraction.txt` (repo root) | One-line text file | Sample input for the URL/markdown upload path | Fixture-like |

## 2. `scripts/migrate_to_sparse_vectors.py`

### 2.1 Why it exists

Hybrid search needs a BM25 sparse vector named `SPARSE_VECTOR_NAME` (default `bm25`) on every point. Points ingested before the feature, or a collection created while `SPARSE_SEARCH_ENABLED=false`, lack it. The script generates the vectors with `app.core.clients.sparse_encoder.generate_sparse_vector` (fastembed `Qdrant/bm25`) from each point's `payload["text"]` and writes them with `client.update_vectors(...)`. Dense vectors are never recomputed.

### 2.2 Environment and invocation

The script reads only process environment variables via `os.getenv`. It does **not** call `load_dotenv()` and does **not** import `app.config` (it imports only `app.core.clients.sparse_encoder`). Values in `.env` are therefore ignored unless exported in the shell (`set -a; source .env; set +a`).

| Variable | Default | Meaning |
|---|---|---|
| `QDRANT_HOST` | `127.0.0.1` | |
| `QDRANT_PORT` | `6333` | |
| `COLLECTION_NAME` | `documents` | Source collection (the one to migrate or copy from) |
| `SPARSE_VECTOR_NAME` | `bm25` | |
| `QDRANT_CHECK_COMPATIBILITY` | `false` | Passed to `QdrantClient(check_compatibility=...)` |

`SPARSE_SEARCH_ENABLED`, shown in the docstring and in the release notes' command lines, is **not read** by the script. Setting it has no effect.

It must run from the repository root with `PYTHONPATH=.` so that `app` imports.

```bash
# in-place
PYTHONPATH=. COLLECTION_NAME=documents1 .venv/bin/python3 scripts/migrate_to_sparse_vectors.py [--dry-run]

# blue-green
PYTHONPATH=. COLLECTION_NAME=documents1 QDRANT_HOST=<host> QDRANT_PORT=6333 \
  .venv/bin/python3 scripts/migrate_to_sparse_vectors.py --new-collection documents1_v2
```

### 2.3 Arguments

| Flag | Default | Mode | Meaning |
|---|---|---|---|
| `--new-collection NAME` | none | selects blue-green | Target collection. Omit for in-place. Validated with `^[A-Za-z0-9_-]+$` (exit code 2 on failure). |
| `--dry-run` | off | both | Scan and report without writing. |
| `--batch-size N` | 100 | both | Points per `update_vectors` (and per flush). |
| `--scroll-limit N` | 500 | both | Points per scroll page. |
| `--skip-copy` | off | blue-green | Skip step 2 (target already populated). |
| `--skip-bm25` | off | blue-green | Skip step 3. |
| `--env-file PATH` | `.env` | blue-green | File whose `COLLECTION_NAME=` line is rewritten on success. |

### 2.4 In-place mode (`run_inplace`)

Precondition: the collection must already declare the sparse field. The script does not create it.

1. `client.count(collection)` for progress totals.
2. Scroll all points with `with_payload=["text"]` and `with_vectors=[sparse_name]`.
3. Per point (`_encode_and_queue`): 
   - existing non-empty sparse vector: counted as `skipped` (this makes the script idempotent);
   - no `text` in payload: `skipped`;
   - encoder exception: `error`;
   - empty `indices` (for example a chunk consisting only of markdown table separators): `no_tokens`, nothing written;
   - otherwise queued as `PointVectors(id, vector={sparse_name: SparseVector(...)})`.
4. When `len(pending) >= batch-size`, `_flush_update_vectors` calls `client.update_vectors`. A failing batch increments `errors` by the batch size and the loop continues.
5. Final summary line: `In-place migration complete in ...s - scanned=..., migrated=..., skipped=..., no_tokens=..., errors=...`. Exit status is 1 when `errors > 0` (`Re-run to retry (script is idempotent).`).

A scroll failure logs `Scroll failed: ...` and stops the loop, but the exit status can still be 0 if no encoding errors were counted. Verify `scanned` against the point count.

### 2.5 Blue-green mode (`run_bluegreen`)

Motivation (from the script docstring): Qdrant's vector schema is fixed at collection creation; a sparse field cannot be added to an existing collection (`Not existing vector name error: bm25` was observed on 1.18.2). So the script builds a new collection and moves data.

Note: this statement is in tension with `app/core/clients/qdrant.py::_ensure_sparse_vector_field`, which calls `qdrant_client.update_collection(sparse_vectors_config=...)` at startup for an existing collection when `SPARSE_SEARCH_ENABLED=true`. The service treats the call as best-effort: a rejection that does not contain the word "already" is logged as `Server rejected sparse-vector field update for ...` and re-raised, aborting startup. Which behaviour you get depends on the server version. Treat blue-green as the safe path for any collection that does not already have the field.

Steps:

| Step | Action |
|---|---|
| Pre | `client.count(old)`; read `client.get_collection(old).config.params.vectors` to mirror the dense schema (so a different embedding dimension is preserved). Exit 1 on failure. |
| 1 | If the target does not exist (and not dry-run), `create_collection(vectors_config=<source dense config>, sparse_vectors_config={sparse_name: SparseVectorParams(modifier=Modifier.IDF)})`. If it exists, creation is skipped. |
| 2 | Scroll the source with `with_payload=True, with_vectors=True` and `upsert` copies (`PointStruct(id, payload, vector)`) into the target. A failed upsert batch logs `Upsert batch failed: ...` and the loop continues (the count check in step 4 will catch it). Skipped by `--skip-copy`. |
| 3 | Scroll the target and write BM25 vectors exactly as in in-place mode. If `errors / (migrated + no_tokens + errors) > 10%` the script logs `BM25 encoding failure rate ... exceeds 10% threshold ... Skipping .env update.` and exits 1. Skipped by `--skip-bm25`. |
| 4 | Skipped entirely in `--dry-run`. Otherwise `_verify_migration`: (a) target count must equal source count; (b) among points with text, the share lacking a sparse vector must be at most 10%. |
| 5 | On success, `_update_env_file` rewrites the `COLLECTION_NAME=` line in `--env-file` (appends it if absent) after re-validating the name, and logs an ACTION REQUIRED block: restart the service and, after confirming search, delete the old collection manually with `curl -X DELETE http://<host>:<port>/collections/<old>`. On failed verification it exits 1 and leaves `.env` alone. |

The source collection is never modified or deleted by the script.

Operational notes:

- **Payload indexes are not created on the new collection by this script.** They are created when the service starts against the new `COLLECTION_NAME`: `ensure_collections_exist` calls `_ensure_payload_indexes` (keyword indexes on `source_id`, `metadata.company`, `metadata.type`, `tags`; text index on `metadata.DOCUMENT_TYPE`; prefix-tokenized text indexes on `title` and `summary`). Until then filters and title/summary boosts are slow or degrade.
- The restart-time `ensure_collections_exist` also runs `_ensure_sparse_vector_field` when `SPARSE_SEARCH_ENABLED=true`; on the new collection this is a benign "already" case.
- Writes that reach the source collection during the copy are not carried over. Freeze ingestion (upload, update, delete) for the duration of the migration, or re-run with `--skip-copy` only for BM25 (it does not re-copy).
- The `.env` rewrite assumes a plain `COLLECTION_NAME=value` line. Deployments that source configuration from Vault (see [Deployment](../setup/deployment.md)) must update the Vault secret instead; the rewritten local file is lost at the next deploy.
- Collection-name validation exists because `start_mac.sh` sources `.env` as a shell script; an unvalidated name written there would be code execution.

### 2.6 Recommended sequence

1. Snapshot the source collection.
2. `--dry-run` with the intended mode and read the counts.
3. Run for real (blue-green for collections without the sparse field).
4. Update the deployed configuration (`COLLECTION_NAME`, then `SPARSE_SEARCH_ENABLED=true`) and restart.
5. Smoke test `POST /api/documents/search` with `include_scoring_debug: true` and confirm `keyword_score` is non-null for keyword-matching documents.
6. Delete the old collection only after the rollback window.

## 3. `rename_key_entities_field.py`

Despite its name it renames the metadata key `DOCUMENT TYPE` (with a space) to `DOCUMENT_TYPE`, which is what the `metadata.DOCUMENT_TYPE` text index and the `resource_type` filter use.

Behaviour:

- Imports `app.config.settings` (so it needs `PYTHONPATH=.` and loads `.env` through `load_dotenv()`), creates a `QdrantClient` with `check_compatibility=settings.QDRANT_CHECK_COMPATIBILITY`.
- Scrolls the whole collection (100 per page, payload only) into memory, selects points whose `metadata` contains `DOCUMENT TYPE`, prints the count and asks for an interactive `yes`.
- For each point it rewrites the key and calls `client.set_payload(collection_name, payload=<entire payload copy>, points=[id])` one point at a time (no batching).
- Verifies by reading the first point of the collection only (not necessarily a migrated one).
- Its final message tells you to "uncomment the filter code in prioritized_search_service.py"; that comment is outdated (the filter is live in `_build_filters`).

Run once per collection, only if legacy data still carries the spaced key. The `metadata` named vector is not recomputed, so its embedding still reflects the old key text.

## 4. Document insertion scripts

All three insert scripts post multipart form data to `POST /api/documents` (there is no `/v1` segment) with fields `file`, `priority`, and optionally `source_id`, `company_id`, `title`, `summary`, `metadata` (JSON string), `tags` (JSON array string or comma-separated). The API returns HTTP 201 with `status`, `chunks_processed`, `points_uploaded`, `upload_failures`.

Note: the endpoint returns 201 even if some batches failed; always check `upload_failures`.

| Script | Notes |
|---|---|
| `insert_document.py` | `API_BASE_URL` defaults to `http://127.0.0.1:8000`. Functions `insert_document(...)` and `insert_multiple_documents(...)`. The `__main__` demo points at a path under `/Users/anujvaghani0/...` and uses metadata keys with spaces; edit before running. For non-local `ENVIRONMENT` values the URL needs the `/vector` prefix. |
| `quick_insert_example.py` | Edit `API_URL`, `FILE_PATH`, `SOURCE_ID`, etc. at the top and run. Prints the full response. |
| `insert_document_curl.sh` | Defines `insert_document file priority source_id company_id title summary tags metadata`; examples at the bottom use placeholder paths and must be edited. Passes `-H "Content-Type: multipart/form-data"` manually, which can strip the boundary; remove that header if uploads fail with 422. |
| `data_insert_script.py` | **Stale.** Posts to `https://demo-mitra.shikshalokam.org/api/documents/upload/`, a path that is not defined by this service (the route is `/api/documents`), and has a hardcoded local folder. Do not use without rewriting. |

## 5. Other root-level files

- `example_prioritized_search.py`: builds `base_url + "/api/v1/documents/search"`. The real route is `/api/documents/search`, so the example returns 404 unless a proxy rewrites the path.
- `test_q.py`: zero bytes. `pytest` collects it (matches `test_*.py`) but it contains no tests.

## Known issues / gotchas

- `migrate_to_sparse_vectors.py` ignores `.env` and `SPARSE_SEARCH_ENABLED`; the release notes' command lines imply otherwise.
- In-place mode requires an existing sparse field; blue-green does not create payload indexes (the service does on restart).
- The script's "cannot add a sparse field to an existing collection" claim conflicts with the service's `update_collection` attempt in `_ensure_sparse_vector_field`.
- `data_insert_script.py` and `example_prioritized_search.py` use endpoint paths that do not exist.
- `rename_key_entities_field.py` uses per-point `set_payload` of the full payload and an in-memory scroll of the whole collection; it is slow and memory-heavy on large collections.

## Related pages

- [Deployment](../setup/deployment.md)
- [Configuration reference](../setup/configuration.md)
- [Qdrant compatibility](qdrant_compatibility.md)
- [Troubleshooting](troubleshooting.md)
- [Qdrant data model](../architecture/qdrant_data_model.md)
- [Ingestion pipeline](../services/ingestion/pipeline.md)
