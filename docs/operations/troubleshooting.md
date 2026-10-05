# Troubleshooting

**Purpose.** Read this page when something is failing and you need to go from a symptom to a cause, a concrete check, and a fix. Every log string and HTTP detail quoted below was taken from the source (`app/`); none is invented. Logs go to stdout, `app.log` (in the process working directory) and, under systemd, `vectorization-service.log`. The log level is INFO (set by `logging.basicConfig` in `app/utils/language_utils.py`), so `logger.debug` messages never appear.

Each entry follows: **Symptom**, **Cause**, **Check**, **Fix**.

## 1. Startup and import failures

### 1.1 Application does not start: PostgreSQL error at import

- **Symptom**: traceback from `sqlalchemy` or `psycopg2` (`connection refused`, `password authentication failed`, `ModuleNotFoundError: psycopg2`) pointing at `app/core/database.py`; uvicorn exits.
- **Cause**: `app/core/database.py` runs `create_engine(settings.DATABASE_URL)` and `Base.metadata.create_all(bind=engine)` at import. It is imported by `translation_service.py` (used by every file processor) and `query_service.py`. PostgreSQL is not used for queries, but it is needed to import the app.
- **Check**: `echo $POSTGRES_DATABASE_URI`; `pg_isready`; `psql "$POSTGRES_DATABASE_URI" -c 'select 1'`.
- **Fix**: provide a reachable database and a valid `POSTGRES_DATABASE_URI`. The built-in default (`postgresql://anuj:1234@localhost:5432/ai_vector_service`) is a developer credential and will fail elsewhere. Use `start_mac.sh` locally, which creates the role and database.

### 1.2 Startup fails with `Failed to create collections: ...` then `Startup failed: ...`

- **Symptom**: both lines in the log; process exits during lifespan.
- **Cause**: `ensure_collections_exist()` could not talk to Qdrant (`QDRANT_HOST`/`QDRANT_PORT` wrong, container down) or Qdrant rejected the create call.
- **Check**: `curl -s http://$QDRANT_HOST:$QDRANT_PORT/healthz`; `curl http://$QDRANT_HOST:$QDRANT_PORT/collections`.
- **Fix**: start Qdrant, correct host and port. With `check_compatibility=False` (default) no version probe is done, so the first call is where connection errors surface.

### 1.3 Startup fails with `Server rejected sparse-vector field update for '<collection>': ...`

- **Symptom**: log line from `qdrant.py::_ensure_sparse_vector_field` followed by `Failed to create collections` and abort.
- **Cause**: `SPARSE_SEARCH_ENABLED=true`, the collection already exists without the `bm25` field, and the server refused `update_collection(sparse_vectors_config=...)` with a message that does not contain "already".
- **Check**: `curl http://<host>:6333/collections/<name>` and inspect `config.params.sparse_vectors`; check server version with `curl http://<host>:6333/`.
- **Fix**: run the blue-green migration (`scripts/migrate_to_sparse_vectors.py --new-collection ...`, see [Scripts](scripts.md)), set `COLLECTION_NAME` to the new collection, restart. Or temporarily set `SPARSE_SEARCH_ENABLED=false`.

### 1.4 Startup fails with a pydantic `ValidationError` mentioning `HYBRID_...`

- **Symptom**: `HYBRID_FUSION_METHOD must be one of ['rrf', 'weighted'], got ...`, `HYBRID_DENSE_WEIGHT must be a finite non-negative number, got ...`, or `HYBRID_DENSE_WEIGHT + HYBRID_SPARSE_WEIGHT must not exceed 1.0 (got a + b = c)`.
- **Cause**: `_validate_fusion_config` in `app/config.py`.
- **Check**: `grep HYBRID .env`.
- **Fix**: use `weighted` or `rrf`; keep each weight at or above 0 and the sum at or below 1.0.

### 1.5 `ValueError: invalid literal for int() with base 10` at import

- **Cause**: a numeric env var (`QDRANT_PORT`, `REDIS_PORT`, `REDIS_CACHE_TTL`, `REDIS_MAX_CACHE_SIZE`, `MAX_FILE_SIZE_MB`, `SHORT_QUERY_THRESHOLD`, `RRF_K`, `SEARCH_CANDIDATE_FANOUT`, `SEARCH_CANDIDATE_MAX`, `INJECTED_DOC_SCORING_MAX`) contains a non-integer, evaluated by `int(os.getenv(...))` in `config.py`.
- **Fix**: correct the value; remove trailing comments or whitespace on that line.

### 1.6 `AttributeError: ... get_embedding_dimension`

- **Cause**: `sentence-transformers` older than 5.4.0; `embedding.py` and `qdrant.py` call `get_embedding_dimension()` at import.
- **Fix**: `uv pip install -U "sentence-transformers>=5.4.0"` in the service venv.

### 1.7 Process starts but variables from `.env` are ignored

- **Cause**: `load_dotenv()` reads `.env` from the current working directory and never overrides existing environment variables. Under systemd the working directory is set by `WorkingDirectory`; when running ad hoc from another directory the file is not found. A real env var of the same name always wins over `.env`.
- **Check**: `env | grep QDRANT`; start from the repository root.

## 2. Health endpoint

### 2.1 `GET /api/health` returns 503 `Service unhealthy` although Qdrant and Redis are up

- **Cause**: `main.py::health_check` calls `redis_cache.redis_client.ping()`. `RedisLRUCache.__init__` sets `redis_client` only when `cache_enabled` is true, and `REDIS_CACHE_ENABLED` defaults to `False`. The resulting `AttributeError` is caught by the generic `except Exception` branch, logged as `Health check failed: 'RedisLRUCache' object has no attribute 'redis_client'`, and returned as 503.
- **Check**: the log line above; `echo $REDIS_CACHE_ENABLED`.
- **Fix**: set `REDIS_CACHE_ENABLED=true` (the env var does work) with a reachable Redis, or change the health check to tolerate a disabled cache. Do not use this endpoint as a load-balancer probe with the defaults.

### 2.2 503 `Redis connection failed`

- **Cause**: `REDIS_CACHE_ENABLED=true` and Redis unreachable (`redis.ConnectionError`; log `Redis connection failed: ...`). Note `REDIS_PASSWORD` is never used by the client, so an authenticated Redis fails here.
- **Fix**: correct `REDIS_HOST`/`REDIS_PORT`; use an unauthenticated Redis or add password support in `redis_cache.py`.

### 2.3 503 `Service unhealthy` with log `Health check failed: ...` naming Qdrant

- **Cause**: `qdrant_client.get_collections()` raised. Check connectivity as in 1.2.

### 2.4 `DELETE /api/cache/redis` returns 500

- **Cause**: `redis_cache.clear()` raised; log `Failed to clear Redis cache: ...`. With the cache disabled `clear()` returns without error, so a 500 means Redis errors with the cache enabled.

## 3. Search

### 3.1 Search returns empty results

Work through in order:

1. Request log: `[/documents/search] request body: {...}` and the `========== SEARCH REQUEST ==========` block show what was received.
2. Filters: `categories` (tags), `organizations` (`metadata.company`), `resource_type` (`metadata.DOCUMENT_TYPE`, substring text match), `file_type` (`metadata.type`) are AND-ed together. A single non-matching filter empties the result. Retry without filters.
3. Thresholds: `Filter mode: FILTER_SCORE (weighted score threshold = X)` or `Filter mode: DETAIL_FILTER_SCORE ...` tells you which applies. Lower `filter_score` or send `detail_filter_score: null`. When `detail_filter_score` is present `filter_score` is ignored (`Note: filter_score=... is ignored when detail_filter_score is provided`).
4. Index content: confirm the collection has points (`GET /collections/<name>`), and confirm the right collection (`COLLECTION_NAME`; after a blue-green migration the service must be restarted).
5. Candidate pool: with min-max normalization (hybrid mode) more candidates change how many documents clear a threshold. `SEARCH_CANDIDATE_MAX` (default 2000) bounds it.
6. Blank filter values are dropped by `_build_filters`; a filter list containing only empty strings behaves as no filter.

### 3.2 Search returns HTTP 500 for a query of three or more words and 20+ characters

- **Symptom**: log `Failed to load spaCy model 'en_core_web_sm'. Please install it using: python -m spacy download en_core_web_sm`, followed by `Prioritized search failed: ...` (`RuntimeError: Search operation failed: ...`).
- **Cause**: the spaCy model is not installed. Short queries bypass spaCy (fewer than `SHORT_QUERY_THRESHOLD` words, or fewer than 20 characters), so only longer queries fail.
- **Fix**: `.venv/bin/python -m spacy download en_core_web_sm` in the same venv the service uses. Verify with `python -c "import spacy; spacy.load('en_core_web_sm')"`.

### 3.3 Search returns 422 `Cannot embed an empty or whitespace-only query` or a vector-dimension message

- **Cause**: `EmbeddingError` from `embedding.embed_query` / `validate_vector`, mapped to 422 by `main.py` (log `Embedding validation failed for /api/documents/search: ...` and `Query embedding invalid: service=prioritized_search query=... expected_dim=384 error=...`).
- **Fix**: send a non-empty query. If the dimension is wrong, check `EMBEDDING_MODEL` against the collection's vector size (changing the model requires re-creating the collection).

### 3.4 Search returns 500 with `top_k must be greater than 0`

- **Cause**: `top_k <= 0` raises `ValueError` in `PrioritizedSearchService.search`; only `EmbeddingError` has an exception handler, so FastAPI returns 500. The endpoint docstring says 422, which is not what the code does.
- **Fix**: send `top_k >= 1`.

### 3.5 Search is slow

- Look for `TIMING: query_batch_points took Ns` and `TIMING: late retrieve of N docs took Ns` in the log to see which phase dominates.
- The default `top_k` is 1,000,000, so the candidate limit is always the cap (`SEARCH_CANDIDATE_MAX`, default 2000 per field, 6 requests in hybrid mode). Send a realistic `top_k` or lower `SEARCH_CANDIDATE_MAX` (documented trade-off: 500 is about 0.9 s, 2000 about 1.5 s, 10000 about 2.7 s on the reference query).
- Title/summary keyword scans log `title match sources found: N (exact=..., partial=...)`; large N with short queries makes scoring expensive, bounded by `INJECTED_DOC_SCORING_MAX`.
- Four uvicorn workers each hold their own model; a cold worker pays model load time on first request (BM25 model download on first sparse use).

### 3.6 Title or summary boost does not apply

1. `HYBRID_SEARCH_ENABLED` must be `true` and the request must not set `search_mode` to `semantic`.
2. Check `title match sources found: 0`: the query did not match any title in the prefix text index. Matching is token-prefix based; an infix-only fragment is only caught in memory by `_supplement_matches_from_results`.
3. `<field> match scroll failed (non-fatal): ...` means the scroll errored and the boost was skipped; the cause is in the message (index missing, server error). Check that the title/summary text indexes exist (see 3.9).

### 3.7 `Hybrid search unavailable (<ExceptionType>: ...); falling back to dense-only parallel search.`

- **Cause**: sparse encoding failed (`RuntimeError` from `generate_sparse_vector`, for example the `Qdrant/bm25` model could not download, corrupted cache, or fastembed missing). Search degrades to dense-only silently from the client's point of view.
- **Check**: preceding log lines `Failed to initialise sparse BM25 encoder: ...` or `Sparse vector generation failed: ...`; response `search_config.sparse_search_enabled` is still true but results lack `keyword_score` under `include_scoring_debug`.
- **Fix**: ensure internet access to Hugging Face from the host (or pre-cache the model), `python -c "from fastembed import SparseTextEmbedding; SparseTextEmbedding('Qdrant/bm25')"`.

### 3.8 Sparse search enabled but all hybrid scores look dense-only

- `BM25 query '<text>' produced no tokens; sparse branch not issued` means the query tokenizes to nothing (only stop words or punctuation); the sparse branch is intentionally skipped.
- If the collection lacks the `bm25` vector, Qdrant answers the sparse request with an error (Qdrant message about a non-existing vector name); run the migration (see [Scripts](scripts.md)).
- Documents ingested while `SPARSE_SEARCH_ENABLED=false` have no sparse vector and receive no BM25 contribution until back-filled.

### 3.9 Qdrant 400 `did not match any variant of untagged enum`

- **Cause**: a request shape newer than the server (`MatchPhrase`, `FormulaQuery`, newer `TextIndexParams` fields) sent to server 1.12.
- **Fix**: remove or flag the feature; see [Qdrant compatibility](qdrant_compatibility.md).

### 3.10 Results lack `text` or show partial payloads in hybrid mode

- **Cause**: late payload retrieval failed: log `Failed late payload retrieval: ...`. The code deliberately falls back to the partial payload captured during candidate collection (title, summary, tags, metadata, source_id, no `text`).
- **Check**: Qdrant connectivity at that moment, very large id lists.

### 3.11 `Failed to parse result item <id>: ...` (warning)

- A point's payload did not fit the response model; that item is dropped from results. Inspect the offending point's payload in Qdrant.

### 3.12 Title/summary filters or boosts are slow or missing after switching collections

- **Cause**: new collection without payload indexes (for example created by the migration script).
- **Check**: startup logs `Payload index created: <field> (...)` for each field, or `Could not ensure payload index for '<field>': ...` for failures; `GET /collections/<name>` `payload_schema`.
- **Fix**: restart the service so `_ensure_payload_indexes` runs; resolve any index warning.

## 4. Ingestion

### 4.1 Upload returns 400

| Detail | Cause |
|---|---|
| `source_id is required and cannot be empty` | Empty `source_id` on update, upsert, delete or metadata update (`base_operation.validate_source_id`). |
| `Invalid priority format. Must be P1, P2, P3, etc.` | `priority` does not start with `P`. |
| `Unsupported file type: <ext>. Supported types: ...` | Extension has no processor (`upload_service._process_file_by_type`). |
| `No content could be extracted from the file.` | Processor returned zero chunks. |
| `Could not extract any text from PDF: <name>. The document may be corrupted or contain no readable content.` | No text from any page after OCR (log `No text extracted from <name> after processing all pages`). |
| `File <name> is empty or contains no readable text` | Empty text file. |
| `Missing required column: SL NO` | CSV processor requires a `SL NO` column. |
| `Metadata must be a JSON object/dict` / `Invalid metadata JSON: ...` / `Tags must be a JSON array/list` / `Invalid tags JSON: ...` | Form parsing in `documents.py`. (Inside services, bad metadata or tags are ignored with warnings instead; see logs `Invalid metadata JSON provided: ... Ignoring metadata.` and `Invalid tags JSON provided, ignoring tags`.) |
| `URL cannot be empty` / `Invalid URL format: ... URL must start with http:// or https://` | `metadata.markdown_url` problems. |

URL fetch errors map to: 408 `Request timeout while fetching URL`, 503 `Could not connect to URL`, upstream status for `HTTP error <code> while fetching URL`, 400 `No text content could be extracted from URL`. The timeout is `URL_REQUEST_TIMEOUT` (30 s).

### 4.2 Upload returns 500 `Upload failed: File ... is too large` or `Error processing PDF/DOCX/XLSX/CSV: ...`

- `MAX_FILE_SIZE_MB` is enforced by `_validate_file_content` at the top of each processor's `process`, after the file is fully read. The `ValueError` is raised before the processor's `try` block, so `UploadService` wraps it as 500 `Upload failed: File <name> is too large (<x>MB). Maximum allowed size is <N>MB` (an empty file gives `Upload failed: Empty file content for <name>`). Raise the limit or split the file. Failures inside the processor body are reported as `Error processing <type>: ...`.
- OCR libraries: `OCR libraries not available. Please install pytesseract, pdf2image, and Pillow. Also ensure tesseract-ocr is installed on your system.` (log `Required OCR library not installed: ...`). Install Tesseract and Poppler (see [Developer setup](../setup/developer_setup.md)); `tesseract --version` and `pdftoppm -v` must work for the service user.
- Page-level OCR problems are non-fatal warnings: `OCR failed for page N, using extracted text`, `Could not convert page N to image for <file>`, `Returning empty string for page N due to OCR error`. OCR is triggered for pages with fewer than `PAGE_TEXT_THRESHOLD` (20) characters.

### 4.3 Upload returns 201 but documents are missing or fewer than expected

- **Cause**: the response is 201 even when some batches fail. `upload_to_qdrant` logs `Failed to upload batch N: ...` and continues; the response fields `points_uploaded` and `upload_failures` carry the counts.
- **Check**: compare `chunks_processed`, `points_uploaded`, `upload_failures`; look for `Failed to upload batch`.
- **Fix**: resolve the Qdrant error (dimension mismatch, unknown vector name, payload too large) and re-upload; use `PUT /api/documents/{source_id}/upsert` to replace.

### 4.4 Upload logs `Sparse vector generation skipped (non-fatal): ... Only dense vectors will be stored.`

- **Cause**: sparse encoder failure during ingestion (see 3.7). The document is stored without `bm25` vectors and will not benefit from keyword search until back-filled with the migration script.

### 4.5 Update or metadata patch returns 404 although the document exists

- Messages: `No documents found with source_id: <id>[ and company_id: <c>]. Use upload endpoint for new documents.` (update) or `No documents found with source_id: ...` (metadata); delete returns `No documents found for source ID: <id>`.
- **Cause 1**: wrong `source_id` or `company_id` (`company_id` filters on `metadata.company`).
- **Cause 2**: `check_documents_exist` swallows exceptions (`Error checking existing documents: ...`) and returns false, so a Qdrant outage during an update can masquerade as 404. Check the log for that line.
- **Fix**: verify with a direct scroll on `source_id` in Qdrant.

### 4.6 Metadata patch returns 400 `Cannot change company_id through metadata update` or `metadata_updates cannot be empty`

- Raised by `metadata_service`; sending `company` in the patch different from the `company_id` form field is rejected.

### 4.7 After PUT update, title, summary or tags are gone

- Known behaviour: `UpdateService` does not forward `title`, `summary` or `tags` to the upload step (the PUT endpoint signature does not accept them). Re-upload through `POST` after deleting, or patch only metadata.

### 4.8 `POST /api/documents/verify-sources` reports everything as not found

- **Cause**: `SourceVerificationService` filters on `metadata.source_id`, but ingestion stores `source_id` at the top level of the payload. This is a real bug (confirmed in `source_verification_service.py`), not a data problem.
- **Fix**: change the filter key to `source_id` (which also has a keyword index).

### 4.9 Metadata changes do not affect search

- `metadata` updates patch the payload only (`set_payload`); the `metadata` named vector is not re-embedded and stays stale. Re-upload the document to refresh it.

## 5. Redis cache

- The cache is wired only into the legacy `QueryService` (`/api/query/`). `PrioritizedSearchService` never reads or writes it, so enabling Redis does not speed up `/api/documents/search`.
- `Cache hit for query: <q>` appears only from `QueryService`.

## 6. Translation (Hindi)

- Warnings: `Rate limited. Waiting N seconds before retry.`, `Server error. Retrying in N seconds...`, `Request timeout. Retrying in N seconds...`; terminal errors `Network error: ...`, `Unexpected error: ...`, or HTTP 500 `Translation failed after N retries. Last error: ...`.
- Endpoint is `TRANSLATION_API_URL` (default is a public AI4Bharat demo host); outages there fail ingestion of Hindi chunks.

## 7. Compatibility warnings

- `UserWarning: Qdrant client version 1.18.0 is incompatible with server version ...` on QA: set `QDRANT_CHECK_COMPATIBILITY=false` (default). On a 1.18 server set it `true`.

## 8. Quick diagnostic commands

```bash
# Service
curl -s localhost:8000/api/health
journalctl -u vectorization-service-uvicorn -n 100 --no-pager
tail -f /opt/deployment/vectorization-service/vectorization-service.log

# Qdrant
curl -s http://$QDRANT_HOST:$QDRANT_PORT/                       # server version
curl -s http://$QDRANT_HOST:$QDRANT_PORT/collections/$COLLECTION_NAME | python3 -m json.tool
curl -s -X POST http://$QDRANT_HOST:$QDRANT_PORT/collections/$COLLECTION_NAME/points/count \
  -H 'Content-Type: application/json' -d '{"exact": true}'

# Dependencies
python -c "import spacy; spacy.load('en_core_web_sm'); print('spaCy OK')"
python -c "from fastembed import SparseTextEmbedding; SparseTextEmbedding('Qdrant/bm25'); print('BM25 OK')"
tesseract --version; pdftoppm -v
```

## Known issues / gotchas

- `/api/health` is red with default settings (section 2.1).
- `verify-sources` always reports not found (section 4.8).
- `top_k <= 0` yields 500, not 422 (section 3.4).
- Upload returns 201 on partial failure (section 4.3).
- `check_documents_exist` hides Qdrant errors as "not found" (section 4.5).
- Redis auth is unsupported (`REDIS_PASSWORD` unused).

## Related pages

- [Developer setup](../setup/developer_setup.md)
- [Configuration reference](../setup/configuration.md)
- [Deployment](../setup/deployment.md)
- [Scripts](scripts.md)
- [Qdrant compatibility](qdrant_compatibility.md)
- [Search overview](../services/search/overview.md)
- [Ingestion pipeline](../services/ingestion/pipeline.md)
