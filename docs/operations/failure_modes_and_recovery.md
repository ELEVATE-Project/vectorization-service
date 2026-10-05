# Failure Modes and Recovery

**Purpose.** Read this page to understand, stage by stage, what the service does when something fails, how to detect it, what state Qdrant is left in, and how to recover. It is a code-level companion to [Troubleshooting](troubleshooting.md) (symptom-driven) and does not repeat that page's per-symptom fixes. Every log string quoted here was taken from the source under `app/`. Qdrant recovery commands assume the default `http://localhost:6333` and collection `documents` (`COLLECTION_NAME`); substitute your own values.

The single most important fact: the service has **no retries, no transactions and no compensation logic**. Every multi-step operation is a sequence of independent Qdrant calls. When a step fails, earlier steps are not rolled back. See [What the service does NOT do](#what-the-service-does-not-do).

## 0. Verification commands used throughout

```bash
# Count all chunks of one document
curl -s -X POST localhost:6333/collections/documents/points/count \
  -H 'Content-Type: application/json' \
  -d '{"filter":{"must":[{"key":"source_id","match":{"value":"DOC_1"}}]},"exact":true}'

# Inspect chunks (payload only, no vectors)
curl -s -X POST localhost:6333/collections/documents/points/scroll \
  -H 'Content-Type: application/json' \
  -d '{"filter":{"must":[{"key":"source_id","match":{"value":"DOC_1"}}]},"limit":100,"with_payload":true,"with_vector":false}'

# Which named vectors does one point carry? (detect missing bm25 / title / ...)
curl -s -X POST localhost:6333/collections/documents/points/scroll \
  -H 'Content-Type: application/json' \
  -d '{"filter":{"must":[{"key":"source_id","match":{"value":"DOC_1"}}]},"limit":3,"with_payload":false,"with_vector":true}'

# Service logs (stdout, app.log, or journald under systemd)
grep -E "Upload failed|Failed to upload batch|Update failed|Delete failed" app.log
```

Note: the scroll and count calls above are plain Qdrant REST API. The `source_id` field has a KEYWORD payload index (created by `_ensure_payload_indexes`), so these filters are cheap.

## 1. Upload (`POST /api/v1/documents`)

Call chain: `documents.py::create_documents` -> `DocumentProcessor.process_upload` -> `UploadService.process`. Order of work inside `process`: validate `source_id` and `priority` -> merge metadata -> `ensure_collections` -> read file (or fetch URL) -> processor chunks -> `_upload_chunks` (embed, sparse encode, build points, batched upsert). The outer `try/except` in `process` re-raises `HTTPException` unchanged and converts every other exception into `HTTPException(500, "Upload failed: <msg>")`.

Qdrant is first written to in the very last step (`upload_to_qdrant`). Every failure before that step leaves the collection untouched for a plain `POST`.

| Failure | What the code does | How to detect | Data state afterwards | Recovery |
|---|---|---|---|---|
| Form parse: `metadata` not JSON or not an object; `tags` malformed JSON array | `parse_metadata_form` / `parse_tags_form` raise `HTTPException(400)` before the service runs | 400 `Invalid metadata JSON: ...`, `Metadata must be a JSON object/dict`, `Invalid tags JSON: ...`, `Tags must be a JSON array/list` | Nothing written | Fix the request and resend. Safe. |
| Empty `source_id` (default is `None`) | `validate_source_id` raises 400 | 400 `source_id is required and cannot be empty` | Nothing written | Resend with `source_id`. Safe. |
| `priority` not starting with `P` | `validate_priority` raises 400 | 400 `Invalid priority format. Must be P1, P2, P3, etc.` | Nothing written | Resend. Safe. |
| Qdrant unreachable or rejects collection setup | `ensure_collections_exist` logs `Failed to create collections: ...` and re-raises; `process` converts to 500 | 500 `Upload failed: ...`; log `Failed to create collections:` then `Upload failed:` | Nothing written | Restore Qdrant (`curl localhost:6333/healthz`), resend. Safe. |
| Unsupported extension | `_process_file_by_type` raises 400 (extension is taken from the filename after the last dot; no dot means the whole name is used) | 400 `Unsupported file type: <ext>. Supported types: [...]` | Nothing written | Resend with a supported file. Safe. |
| Empty file body | `_validate_file_content` raises `ValueError`. It is called outside the processor's own `try` for PDF, DOCX, CSV, XLSX and text, so it reaches `process` and becomes 500 | 500 `Upload failed: Empty file content for <name>` | Nothing written | Resend a non-empty file. Safe. |
| File larger than `MAX_FILE_SIZE_MB` (default 1024) | Same `ValueError` path as above. The check runs after the whole body has been read into memory | 500 `Upload failed: File <name> is too large (X MB). Maximum allowed size is N MB` | Nothing written | Reduce the file or raise `MAX_FILE_SIZE_MB`. Safe. Note: the project reference notes say size is not validated; the code does validate it in each processor. |
| Corrupt or unreadable PDF, DOCX, XLSX or CSV | Processor catches every exception, logs it, raises 500 | 500 `Error processing PDF: ...` (also DOCX / XLSX / CSV). Log `PDF processing error:`, `DOCX processing error:`, `XLSX processing error for '<file>':`, `CSV processing error:` | Nothing written | Repair or re-export the file, resend. Safe. |
| PDF with no extractable text | The processor raises `HTTPException(400)` inside its own `try`, and its `except Exception` then re-wraps it | 500 (not 400) `Error processing PDF: 400: Could not extract any text from PDF: <name>...`; log `No text extracted from <name> after processing all pages` | Nothing written | Provide a text-bearing PDF or install OCR dependencies (below). |
| OCR failure on a page (missing tesseract/poppler, OCR error) | Per page: `_extract_page_with_ocr` returns an empty string on non-import errors, or raises 500 on `ImportError`. The caller catches any exception, falls back to the (nearly empty) PyPDF2 text for that page and continues | Warning `OCR failed for page N, using extracted text: ...` or `OCR returned empty text for page N of <file>`; log `Required OCR library not installed:` | Upload succeeds with those pages empty or near-empty. **Silent content loss** (the response still says success) | Install `tesseract` and `pdftoppm`, then re-ingest with `PUT /documents/{source_id}/upsert`. Detect affected docs by `metadata.pages_with_ocr` and `metadata.ocr_pages` in the payload. |
| CSV missing `SL NO` column | `HTTPException(400)` raised inside the processor's `try`, re-wrapped by `except Exception` | 500 `Error processing CSV: 400: Missing required column: SL NO` | Nothing written | Fix the CSV header. |
| Text file undecodable as UTF-8 | Falls back to latin-1 (never fails). Whitespace-only text raises 400 | 400 `File <name> is empty or contains no readable text` | Nothing written | Resend. |
| Zero chunks produced | `process` raises 400 | 400 `No content could be extracted from the file.` | Nothing written | Resend a file with content. |
| `metadata.markdown_url` set (URL path) | The uploaded file body is not read. `URLTextExtractor.extract_text` fetches with `httpx` and a 30 s timeout (`URL_REQUEST_TIMEOUT`) | See the URL rows below | Nothing written | Fix the URL or resend. |
| URL: empty or not `http(s)://` | 400 `URL cannot be empty` / `Invalid URL format: ...` | HTTP response | Nothing written | Fix the URL. |
| URL: timeout, connect error, HTTP error status | Mapped to 408 `Request timeout while fetching URL`, 503 `Could not connect to URL`, or the upstream status code (`HTTP error N while fetching URL`). No retry | Logs `Timeout while fetching URL:`, `Connection error while fetching URL:`, `HTTP error while fetching URL:` | Nothing written | Retry later. Safe. |
| URL: no text extracted | 400 `No text content could be extracted from URL: ...` | HTTP response | Nothing written | Use a different URL or upload the file. |
| URL: any other error while chunking | `_process_url_text` converts to 500 | 500 `Failed to process URL text: ...`; log `Error processing URL text:` | Nothing written | Inspect the log. |
| Translation hook | `translation_service.process_chunk` is a no-op: it returns `(chunk_text, False)` and cannot fail | Not applicable. Importing `translation_service` pulls in `app/core/database.py` (PostgreSQL), which can fail at import time, see [Troubleshooting 1.1](troubleshooting.md) | Not applicable | Not applicable |
| Dense embedding failure (model error, OOM) | `generate_embeddings` is not wrapped. The exception propagates to `process` and becomes 500 | 500 `Upload failed: ...`; log `Upload failed:` | Nothing written (embedding happens before any upsert) | Fix resources and resend. Safe for `POST`. |
| Malformed vector (wrong dimension, NaN) | `_create_point_vectors` calls `validate_vector`, which raises `EmbeddingError` (a `ValueError`). Inside `process` this is caught by the generic handler | 500 `Upload failed: Invalid embedding dimension: expected N, got M` or `Embedding contains null or non-finite values`. Note: the 422 handler for `EmbeddingError` in `main.py` is bypassed here because `process` converts it first | Nothing written | Check `EMBEDDING_MODEL` against the collection vector size. |
| Sparse (BM25) encoding failure | Whole block is wrapped. On any exception: warning, `sparse_vectors = []`, upload continues dense-only. A single chunk with no tokens gets `None` and is stored dense-only | Warning `Sparse vector generation skipped (non-fatal): ... Only dense vectors will be stored.` Response is still 201 and contains no sparse indicator | **Points exist without a `bm25` vector.** They are invisible to BM25 retrieval but still found by dense search | Fix fastembed (`python -c "from fastembed import SparseTextEmbedding"`), then re-ingest via `PUT .../upsert`. Detect with the with-vector scroll above (no `bm25` key) or the in-place mode of `scripts/migrate_to_sparse_vectors.py` ([Scripts](scripts.md)). |
| Invalid chunk dict (missing `id`/`text`/`metadata`) | `_upload_chunks` logs and `continue`s past it | Logs `Invalid chunk type:` / `Chunk missing required fields:` | Chunk silently skipped. `points_uploaded` is lower than `chunks_processed`, but `upload_failures` stays 0 | Compare `chunks_processed` with `points_uploaded`. Re-ingest. Not expected from the built-in processors. |
| Batch upsert fails (Qdrant error, timeout, payload too large) | `upload_to_qdrant` catches per batch of 100 points, logs, adds the batch size to `error_count`, and **continues** with the next batch. Nothing is raised | Log `Failed to upload batch N: ...`, then `Upload completed: X successful, Y failed`. Response is **201** with `upload_failures > 0` and `points_uploaded < chunks_processed` | **Partial document**: successful batches are persisted, failed batches are absent. Order is by chunk order, in 100-point groups | See [Designing retries safely](#designing-retries-safely). Do not re-`POST`. Use `PUT /documents/{source_id}/upsert` (deletes then re-uploads) or `DELETE` then `POST`. Verify with the count command above. |
| Every point rejected | Same as above with `success_count = 0`. No exception | 201, `points_uploaded: 0`, `upload_failures: N` | Nothing written | Fix Qdrant, retry as above. |

Note: any non-2xx response is safe to retry with `POST` only if the failure occurred before the batch upsert (all rows above except the last two). A 201 with `upload_failures > 0` is the dangerous case, and a 500 from a network error during the final request cannot be distinguished by the client from "nothing written", so check the count first.

## 2. Update and upsert (`PUT /documents/{source_id}` and `.../upsert`)

Code: `UpdateService.update` and `UpdateService.upsert`. Sequence:

```text
validate source_id, priority
parse metadata (lenient: invalid JSON becomes {} with a warning)
check_documents_exist
count_documents
DeleteService.delete        <-- committed to Qdrant, irreversible
UploadService.process       <-- file is read and parsed HERE, after the delete
```

Because the new file is read and parsed only after the delete, **every upload-stage failure from section 1 that occurs after the delete now destroys the existing document.** Only `source_id` and `priority` validation happen before the delete.

| Failure | What the code does | How to detect | Data state afterwards | Recovery |
|---|---|---|---|---|
| Invalid `source_id` / `priority` | 400 before any Qdrant write | 400 `source_id is required...` / `Invalid priority format...` | Old document intact | Fix the request. |
| `PUT` and document not found | 404 | 404 `No documents found with source_id: X ... Use upload endpoint for new documents.` | Nothing changed | Use `POST` or `/upsert`. |
| Existence check hits a Qdrant error | `check_documents_exist` swallows the error, logs `Error checking existing documents:` and returns `False`. `count_documents` does the same (`Error counting documents:`) and returns 0 | `PUT` returns a misleading 404 (document exists). **`/upsert` skips the delete** and uploads on top, producing duplicates (`documents_deleted: 0`, `previous_document_count: 0`) | `PUT`: intact. `/upsert`: old chunks plus new chunks both present | Check the log for the lines above. For duplicates run `DELETE /documents/{source_id}` and then `/upsert` again once Qdrant is healthy. |
| Delete step fails midway | `DeleteService.delete` raises 500 | 500 `Delete failed: ...` (wrapped again by `update` as 500 only if not an `HTTPException`; the delete's own 500 passes through unchanged) | Some old chunks deleted, some remain. New file not processed | Rerun the same `PUT`/`upsert` (delete is idempotent). |
| Upload step fails after the delete (corrupt file, unsupported type, empty file, OCR/PDF error, embedding error, Qdrant down) | `update`/`upsert` re-raise the `HTTPException` from `UploadService` (or wrap others as 500 `Update failed: ...` / `Upsert failed: ...`) | Error status from section 1; log `Update failed:` / `Upsert failed:` for non-HTTP errors. Earlier log lines `Deleted batch of N documents` show the delete already happened | **Document lost: zero chunks remain.** There is no backup or rollback | Re-upload a valid file with `PUT .../upsert` (or `POST`; collection is empty for that id). Verify the count. To avoid it, validate the file with a throwaway `source_id` first. |
| Upload step partially fails (`upload_failures > 0`) | Returned in the 200 body | `upload_failures`, `points_uploaded`, `chunks_processed` in the response | New document partially present, old document already gone | Rerun `PUT .../upsert` with the same file (clean replacement). |
| `title`, `summary`, `tags` supplied originally | `update`/`upsert` call `UploadService.process` without them | Payload `title`/`summary`/`tags` are `null`; vectors `title`, `summary`, `tags` absent | Those fields are lost on every update | Re-`POST` the document with those fields after `DELETE`, or accept the loss. `PATCH` cannot set them (it writes under `metadata`). |
| `company_id` omitted on `PUT` | Filter is `source_id` only | Count before and after | Chunks of **all** companies with that `source_id` are deleted | Always pass `company_id` in multi-tenant setups. |

See [Document operations](../services/ingestion/document_operations.md) for the endpoint contracts.

## 3. Delete (`DELETE /api/v1/documents/{source_id}`)

Code: `DeleteService.delete`. Loop: scroll up to 100 point ids matching the filter, delete them, repeat until a page is empty or shorter than 100.

| Failure | What the code does | How to detect | Data state afterwards | Recovery |
|---|---|---|---|---|
| Empty `source_id` | 400 | `source_id is required and cannot be empty` | Nothing deleted | Fix the request. |
| Qdrant error on a scroll or delete call | Generic handler logs and returns 500. Earlier batches are already deleted | 500 `Delete failed: ...`; logs `Delete failed:` and `Traceback:`; progress lines `Deleted batch of N documents. Total deleted: M` | **Partial delete.** Some chunks remain | Rerun the same `DELETE`. It is idempotent and continues. Verify count equals 0. |
| Nothing matched | 404 after the loop | 404 `No documents found for source ID: X` (plus `and company ID: Y`) | Nothing changed | Treat as success if the goal is absence. A rerun after a partially failed delete returns 200 if any chunks remained, 404 if none. |
| Short page ends the loop early | The loop breaks when `len(point_ids) < 100` even if more matches exist (for example concurrent writes) | `documents_deleted` smaller than expected | Stragglers remain | Rerun `DELETE`, verify count. |
| Deleting a source that spans companies | Without `company_id` all matches are deleted | Count before and after | Cross-company deletion | Pass `company_id`. Note that `company_id` is read from a form field on `DELETE`, not the query string. |

Recovery of deleted data is not possible from the service. Restore from a Qdrant snapshot if you took one (this is a Qdrant feature, not something the service provides), or re-ingest the source file.

## 4. Metadata patch (`PATCH /api/v1/documents/{source_id}/metadata`)

Code: `MetadataService.update_metadata`. For every matching point it merges the update into the point's existing `metadata`, sets `updated_at`, and calls `set_payload` once per point with the full payload. Vectors are never touched.

| Failure | What the code does | How to detect | Data state afterwards | Recovery |
|---|---|---|---|---|
| `metadata_updates` is not valid JSON | Endpoint raises 400 | 400 `Invalid metadata JSON` | Nothing changed | Fix the request. |
| Empty dict or empty `source_id` | 400 | `metadata_updates cannot be empty` / `source_id is required...` | Nothing changed | Fix the request. |
| Update tries to change `company` while `company_id` is passed | 400 raised while merging the first point | `Cannot change company_id through metadata update` | Nothing changed (raised before the first write) | Do not change `company` through `PATCH`. If `company_id` is omitted the check is skipped and `company` can be changed. |
| Qdrant error midway | Generic handler returns 500; earlier `set_payload` calls are already applied | 500 `Metadata update failed: ...`; log `Metadata update failed:` and progress lines `Updated metadata for batch of N documents. Total updated: M` | **Partially patched document**: some chunks have the new metadata, others the old | Rerun the same `PATCH` (merge is idempotent apart from `updated_at`). Verify by scrolling and comparing `metadata.updated_at` across chunks. |
| No document matches | 404 | `No documents found with source_id: X` | Nothing changed | Check the id and `company_id`. |
| Document has 100 or more chunks | The scroll uses no offset and patched points still match, so the same first page is re-scrolled | The request never returns; `Updated metadata for batch of 100 documents. Total updated: ...` repeats with a growing total | Points already patched; request hangs and holds a worker | Restart the worker if needed, or avoid patching documents with 100 or more chunks (use `PUT .../upsert` instead). Tracked in [Document operations](../services/ingestion/document_operations.md). |
| `metadata` vector becomes stale | The `metadata` named vector is computed at upload from `"k: v ..."` and is not recomputed | No error. Search by metadata content still matches the old values | Payload and `metadata` vector disagree | Re-ingest with `PUT .../upsert` to rebuild the vector. Filters on payload fields (`metadata.company`, etc.) reflect the patch immediately because they read the payload. |
| Patch changes `metadata.title`, `summary` or `tags` | Writes under `metadata.*` only | No error | Top-level payload `title`, `summary`, `tags` (used by the title/summary boost and filters) and their vectors are unchanged | Re-ingest via `PUT`/`POST`. |

## 5. Search (`POST /api/v1/documents/search`)

Code: `PrioritizedSearchService.search`. Pipeline: preprocess query -> `embed_query` -> build filters -> candidate collection (`_parallel_batch_search` or `_hybrid_batch_search`) -> rank/filter -> title/summary boosts and injection -> late payload retrieval (sparse mode only) -> build response. Search is read-only; no failure here changes stored data.

| Failure | What the code does | How to detect | Data state afterwards | Recovery |
|---|---|---|---|---|
| `top_k <= 0` | `ValueError` raised, logged, re-raised. There is no handler for `ValueError`, so FastAPI returns a generic 500 | Log `Validation error in prioritized search: top_k must be greater than 0` | Read-only | Send a positive `top_k`. |
| Empty or whitespace query | Treated as "no query": `_get_unique_source_documents` runs (scroll, not an error) | Response `search_config.mode` is `unique_source_id` | Read-only | Not a failure. |
| Query embedding rejected (empty after preprocessing, wrong dimension, non-finite) | `EmbeddingError` is logged and re-raised; `main.py` handler returns 422 | 422 `{"detail": "..."}`; log `Query embedding invalid: service=prioritized_search query=... expected_dim=... error=...` and `Embedding validation failed for <path>` | Read-only | Fix the query or check `EMBEDDING_MODEL` against the collection vector size. |
| Embedding model runtime error (OOM) | Propagates, wrapped as `RuntimeError("Search operation failed: ...")`, which has no handler | 500 `Internal Server Error`; log `Prioritized search failed:` with traceback | Read-only | Retry; fix the host. |
| Qdrant unreachable or batch query error | `query_batch_points` is not wrapped in the dense path. In hybrid mode transport errors are deliberately not caught. Both end in the generic handler | 500; log `Prioritized search failed: ...` with traceback | Read-only | Restore Qdrant and retry. Safe. |
| Qdrant 400 `did not match any variant of untagged enum` | Not caught; surfaces as 500 | Log text above | Read-only | A feature newer than server 1.12 was introduced. See [Qdrant compatibility](qdrant_compatibility.md). |
| Sparse stage failure (`fastembed` missing, model download failure, encoder error, `ImportError` / `RuntimeError`) | `_hybrid_batch_search` catches `(ImportError, RuntimeError)` and falls back to `_parallel_batch_search` (dense only). `AttributeError` is deliberately not caught | Warning `Hybrid search unavailable (<Type>: <msg>); falling back to dense-only parallel search.` Search still returns 200; `search_config.sparse_search_enabled` is still `true`, so the response does not reveal the fallback | Read-only | Fix fastembed (see the sparse ingestion row). Until then ranking is dense-only. |
| BM25 query tokenises to nothing | Sparse request is not issued; dense carries full weight | Info `BM25 query '...' produced no tokens; sparse branch not issued` | Read-only | Not a failure. |
| BM25 returns no hits (for example points ingested without `bm25`) | Hybrid scale is kept; dense carries full weight | Info `Sparse branch was issued but matched nothing; dense will carry full weight` | Read-only | If unexpected, check that points carry `bm25` vectors. |
| Late payload retrieval fails (sparse mode) | `qdrant_client.retrieve` is wrapped; the error is logged and the search continues with the partial payload | Log `Failed late payload retrieval: ...`. Response is 200 | Read-only | Results lack the `text` field (payload was projected without it). Retry the search. |
| Title/summary match scroll fails | `_get_field_match_sources` catches, logs and returns whatever matched so far | Warning `title match scroll failed (non-fatal): ...` (or `summary`) | Read-only | Boost is silently skipped or partial. Retry. |
| Boost injection fetch fails | `_fetch_field_match_docs` catches and returns an empty list | Warning `Could not bulk fetch title-match docs: ...` (or `summary`) | Read-only | Injected documents are silently missing. Retry. |
| Result item cannot be parsed | Item is skipped | Warning `Failed to parse result item <id>: ...` | Read-only | Fewer results than `total_results` suggests; inspect the point payload. |
| No-query scroll fails | Wrapped as `RuntimeError` | 500; log `Failed to get unique source documents:` | Read-only | Restore Qdrant and retry. |

## 6. Startup and health

| Failure | What the code does | How to detect | Data state afterwards | Recovery |
|---|---|---|---|---|
| Qdrant unreachable at startup | `ensure_collections_exist` logs and re-raises; `lifespan` logs and re-raises, so the app does not start | Logs `Failed to create collections: ...` then `Startup failed: ...` | Not applicable | Start Qdrant (`curl localhost:6333/healthz`) and restart the service. |
| Existing collection lacks the sparse field and the server refuses the update | `_ensure_sparse_vector_field` raises | Log `Server rejected sparse-vector field update for '<name>': ...`; startup aborts | Collection unchanged | Use the blue-green mode of `scripts/migrate_to_sparse_vectors.py`, or set `SPARSE_SEARCH_ENABLED=false`. See [Scripts](scripts.md). |
| `qdrant-client` older than 1.9 with sparse enabled | Warns and skips sparse config | Warning `SPARSE_SEARCH_ENABLED=true but qdrant-client<1.9.0 is installed...` | Collection created without `bm25`; later sparse upserts will fail per batch | Upgrade `qdrant-client`, recreate or migrate the collection. |
| Payload index creation fails | Each index is wrapped; failure is logged and startup continues | Warning `Could not ensure payload index for '<field>': ...` or `Could not read payload schema for '<name>': ...` | Collection usable, but filters or title/summary matching may be slow or fail | Fix the cause and restart; index creation is idempotent. Verify with `curl localhost:6333/collections/documents` (`payload_schema`). |
| Embedding model fails to load | `SentenceTransformer(...)` runs at import of `embedding.py` and is not guarded | Process exits with a traceback before `lifespan` | Not applicable | Check network and model cache for `EMBEDDING_MODEL`. |
| `GET /api/health` with defaults | Calls `redis_cache.redis_client.ping()`; with `REDIS_CACHE_ENABLED=false` `redis_client` does not exist, raising `AttributeError` | 503 `Service unhealthy`; log `Health check failed:` | Not applicable | See [Troubleshooting 2.1](troubleshooting.md). Do not use it as a probe with defaults. |
| Health check: Redis connection error | 503 | `Redis connection failed` | Not applicable | Fix Redis or disable the cache. |

## Partial-failure and consistency matrix

| Scenario | Client sees | Qdrant state | Detect | Fix |
|---|---|---|---|---|
| `POST` where some 100-point batches fail | **201**, `upload_failures > 0`, `points_uploaded < chunks_processed` | Subset of chunks stored | Response fields; log `Failed to upload batch N:`; count command | `DELETE` then `POST`, or `PUT .../upsert` |
| `POST` where all batches fail | 201, `points_uploaded: 0` | Nothing | Same | Retry after fixing Qdrant |
| `POST` fails before the upsert stage (parse, embed, validation) | 4xx or 500 | Nothing | Count is 0 | Plain retry is safe |
| `POST` repeated for the same `source_id` | 201 each time | Duplicate chunks with new random ids | Count doubles | `DELETE`, then `POST` once |
| `PUT` / `upsert` where delete succeeds and upload fails | 4xx or 500 | **Document gone** | Count is 0, logs show `Deleted batch of N documents` | Re-upload with `upsert` |
| `PUT` / `upsert` where delete fails midway | 500 | Some old chunks remain | Count lower than before | Rerun the same request |
| `PUT` / `upsert` where upload partly fails | 200, `upload_failures > 0` | Old document gone, new document partial | Response fields | Rerun `upsert` with the same file |
| `upsert` when existence check errors | 200, `documents_deleted: 0` | Old chunks plus new chunks | Count higher than expected; log `Error checking existing documents:` | `DELETE`, then `upsert` |
| `PUT` / `upsert` without original `title`/`summary`/`tags` | 200 | New chunks with those fields null and vectors absent | Scroll payload | Re-`POST` with the fields after `DELETE` |
| `DELETE` fails midway | 500 | Some chunks remain | Count above 0 | Rerun `DELETE` |
| `PATCH` fails midway | 500 | Mixed old and new metadata across chunks | Compare `updated_at` across chunks | Rerun `PATCH` |
| Successful `PATCH` | 200 | Payload updated; `metadata` vector stale; top-level `title`/`summary`/`tags` unchanged | Search by changed metadata text still matches old values | Re-ingest through `upsert` |
| Sparse encoding fails at ingest | 201 (no indicator) | Points without `bm25` | Warning `Sparse vector generation skipped (non-fatal)`; with-vector scroll shows no `bm25` | Re-ingest, or the in-place mode of `migrate_to_sparse_vectors.py` |
| Sparse stage fails at search | 200 | Unchanged | Warning `Hybrid search unavailable (...)` | Fix fastembed |
| OCR fails on some PDF pages | 201 | Chunks missing those pages' text | Payload `pages_with_ocr`, `ocr_pages`; warning `OCR failed for page N` | Install OCR deps, re-ingest |

## Designing retries safely

- **`POST /documents` is not idempotent.** Each chunk id is `uuid.uuid4().hex`, generated per call by the processors (and in `_process_url_text`). Re-sending the same file creates a second full set of chunks under the same `source_id`. Qdrant `upsert` on a random id never overwrites.
- **Safe retry rule for `POST`:** retry only if you have confirmed the count for that `source_id` is 0 (use the count command at the top of this page). If the count is above 0, `DELETE` first.
- **Prefer `PUT /documents/{source_id}/upsert` for any retry.** It deletes existing chunks by `source_id` (and `company_id` if given) and then uploads, so repeating it converges to one clean copy. The cost is that the document is briefly (and, on failure, durably) absent, and that `title`, `summary` and `tags` are not forwarded.
- **A failed upsert leaves no document.** Always be ready to rerun it with the same file; keep the source file until the count matches `chunks_processed`.
- **Make the client treat `upload_failures > 0` as an error.** The service returns 201 for partial writes, so HTTP status alone is insufficient.
- **Use `company_id` consistently** on delete, update and upsert, so a retry does not remove another tenant's chunks that share the `source_id`.
- **`DELETE` and `PATCH` can be retried freely** (apart from the 100-chunk `PATCH` loop described above); a `DELETE` rerun returns 404 once nothing remains.
- **Timeouts:** a client timeout on `POST` tells you nothing about whether the server finished. Check the count before retrying.
- **Concurrency:** two simultaneous `upsert` calls for the same `source_id` can interleave delete and upload and leave duplicates. Serialise writes per `source_id` in the caller.

## What the service does NOT do

- **No retries.** Qdrant calls, URL fetches and embedding calls are attempted once. The only automatic degradation is the sparse fallback in search and the dense-only fallback at ingest.
- **No transactions or rollback.** Delete-then-upload, per-point `set_payload` and per-batch `upsert` are independent writes; none is undone on failure.
- **No dead-letter queue or failed-chunk record.** Failed batches are only logged (`Failed to upload batch N:`); the failed chunks are not persisted anywhere and the response does not list which ones failed.
- **No background job or status endpoint.** All work is synchronous inside the request, so a worker crash or client timeout mid-upload leaves a partial document with no marker.
- **No integrity check after write.** The service does not compare points stored with points built.
- **No dedup by content.** `POST /documents/check-similarity` exists as a separate endpoint but is not called automatically.
- **No Qdrant-side backups.** Snapshots, if you use them, are a Qdrant feature configured outside this service.
- **No dedicated error handlers** other than `EmbeddingError` (422); unhandled `RuntimeError` and `ValueError` surface as a generic 500.
- **No circuit breaker or rate limiting.**

## Known issues / gotchas

- Project reference notes say file size is not enforced. The code enforces `MAX_FILE_SIZE_MB` inside each processor (via `BaseFileProcessor._validate_file_content`), but only after the full body has been read into memory.
- PDF, CSV and other processors re-wrap their own 400 errors into 500 (`Error processing PDF: 400: ...`), so client-side logic keyed on 400 for "bad content" will not fire for those cases.
- `EmbeddingError` raised during ingest becomes 500, not 422, because `UploadService.process` catches it first.
- The `upsert` existence check swallows Qdrant errors and returns `False`, which turns a transient Qdrant failure into a skipped delete and duplicate chunks.
- The `/search` response does not indicate that the dense-only fallback was used; only the `Hybrid search unavailable` warning in the log does.

## Related pages

- [Troubleshooting](troubleshooting.md)
- [Scripts](scripts.md)
- [Qdrant compatibility](qdrant_compatibility.md)
- [Ingestion pipeline](../services/ingestion/pipeline.md)
- [Document operations](../services/ingestion/document_operations.md)
- [File processors](../services/ingestion/file_processors.md)
- [Search overview](../services/search/overview.md)
- [Qdrant data model](../architecture/qdrant_data_model.md)
