# System Architecture

Read this page first to understand how the Vectorization Service is assembled: which components exist, what each layer is responsible for, what must be running for the process to start, how requests are executed (sync vs async), and why the main design decisions were made. Every statement is verified against the code under `app/`; known issues are listed under [Known issues / gotchas](#known-issues-gotchas).

## Component diagram

```text
                          HTTP clients
                               |
                               v
        +----------------------------------------------------+
        |  FastAPI app  (app/main.py)                         |
        |  root_path = "" (ENVIRONMENT=local) | "/vector"     |
        |  default_response_class = CustomJSONResponse        |
        |  EmbeddingError -> HTTP 422 handler                 |
        |  GET /api/health  (inline)                          |
        +----------------------------------------------------+
                               |  include_router(prefix="/api")
                               v
        +----------------------------------------------------+
        |  API layer  app/api/v1/                             |
        |   api.py           aggregates routers               |
        |   endpoints/documents.py  (upload/update/delete/    |
        |       metadata/search/text-search/check-similarity/ |
        |       verify-sources)                               |
        |   endpoints/query.py   (/api/query/, legacy)        |
        |   endpoints/cache.py   (DELETE /api/cache/redis)    |
        +----------------------------------------------------+
                               |
                               v
        +----------------------------------------------------+
        |  Service layer  app/services/                       |
        |   document_processor.py  (facade)                   |
        |     -> document_operations/{upload,update,delete,   |
        |        metadata}_service.py  (+ base_operation.py)  |
        |     -> file_processors/{pdf,docx,xlsx,csv,text}     |
        |   url_text_extractor.py                             |
        |   prioritized_search_service.py   (main search)     |
        |   text_embedding_search_service.py                  |
        |   similarity_service.py                             |
        |   source_verification_service.py                    |
        |   query_service.py  (legacy multilingual query)     |
        |   translation_service.py  (pass-through stub)       |
        +----------------------------------------------------+
                   |                    |              |
                   v                    v              v
        +----------------+  +-------------------+  +-------------------+
        | Client layer   |  | utils/            |  | models/           |
        | core/clients/  |  |  query_preprocessor|  |  api_models.py    |
        |  qdrant.py     |  |  language_utils   |  |  db_models.py     |
        |  embedding.py  |  |  json_handler     |  +-------------------+
        |  sparse_encoder|  +-------------------+
        |  redis_cache.py|
        +----------------+
          |       |      \
          v       v       v
     Qdrant   SentenceTransformer   fastembed BM25 (lazy)    Redis (optional)
     server   (loaded at import)    Qdrant/bm25 model

     core/database.py -> PostgreSQL (SQLAlchemy engine; see "Postgres status")
```

## Layer responsibilities

| Layer | Location | Responsibility |
|---|---|---|
| App factory | `app/main.py` | Creates the `FastAPI` instance, runs `ensure_collections_exist()` in the lifespan hook, registers the `EmbeddingError` -> 422 handler, defines `/api/health`. |
| Configuration | `app/config.py` | Single `Settings(BaseSettings)` instance, `settings`. Reads `.env` via `load_dotenv()` and environment variables. Validates fusion settings in `_validate_fusion_config`. |
| API layer | `app/api/v1/endpoints/*.py` | HTTP parsing (form/JSON), dependency injection for `metadata` and `tags` form fields, delegation to services. No business logic except inline logging. |
| Service layer | `app/services/` | All business logic: file extraction and chunking, embedding and point construction, search, ranking, boosts, deduplication. |
| Client layer | `app/core/clients/` | Process-wide singletons for Qdrant, the dense embedding model, the BM25 encoder, and the (disabled) Redis cache. See [Clients](../backend/clients.md). |
| Models | `app/models/` | `api_models.py` holds the Pydantic request and response models; `db_models.py` holds one SQLAlchemy model (`TranslationRecord`). |
| Utils | `app/utils/` | spaCy query preprocessing, Hindi detection/translation helper, NaN-safe JSON response. |

Endpoints are thin: `DocumentProcessor` (`app/services/document_processor.py`) is a facade that instantiates `UploadService`, `UpdateService`, `DeleteService` and `MetadataService` and forwards calls. Search, text-search, similarity and verification endpoints call module-level service singletons directly.

## Runtime dependencies

| Dependency | Needed at startup? | Used for | Notes |
|---|---|---|---|
| Qdrant | Yes | Vector store (dense named vectors, sparse `bm25`, payload) | `QdrantClient(...)` is constructed at import of `qdrant.py`. The lifespan hook calls `ensure_collections_exist()` and re-raises on failure, so the app does not start if Qdrant is unreachable. |
| SentenceTransformer model | Yes | Dense embeddings | `embedding.py` loads `SentenceTransformer(settings.EMBEDDING_MODEL)` at import time (default `all-MiniLM-L6-v2`, 384 dimensions). First start downloads the model if not cached. |
| spaCy `en_core_web_sm` | Only when a long query is preprocessed | Stop-word removal | Loaded lazily by `app/utils/query_preprocessor.py`. |
| fastembed `Qdrant/bm25` | Only if `SPARSE_SEARCH_ENABLED=true` and a sparse vector is first needed | BM25 sparse vectors | Lazy singleton in `sparse_encoder.py`. |
| Redis | No | Query cache | The cache defaults to disabled (`REDIS_CACHE_ENABLED: bool = False` in `config.py`; pydantic-settings can still override it from the environment); see [Clients](../backend/clients.md#redis_cachepy). |
| PostgreSQL | Effectively yes, see below | Translation record table (not used at runtime) | Import-time side effect in `app/core/database.py`. |

### Postgres status

PostgreSQL is only partly used:

- No service issues an ORM query against it at runtime. `TranslationRecord` is only imported (by `translation_service.py` and `query_service.py`), never queried or written.
- However, `app/core/database.py` executes at import time:

```python
engine = create_engine(settings.DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

# Create tables
Base.metadata.create_all(bind=engine)
```

`create_all` opens a real connection. `database.py` is imported through every file processor (`pdf_processor`, `docx_processor`, `xlsx_processor`, `csv_processor`, `text_processor` all import `translation_service`, which imports `SessionLocal`), and through `endpoints/query.py` -> `query_service`. Therefore an unreachable or misconfigured `DATABASE_URL` (env var `POSTGRES_DATABASE_URI`, default `postgresql://anuj:1234@localhost:5432/ai_vector_service`) fails application import before Qdrant is even contacted. Treat Postgres as a hard startup dependency until that import is removed.

## Startup sequence

1. `uvicorn app.main:app` imports `app.main`.
2. `app.main` imports `app.api.v1.api`, which imports the endpoint modules, which in turn import services and clients. In practice this triggers, in order of module import:
    - `app.config` -> `settings = Settings()`; `TOKENIZERS_PARALLELISM=false` is set; fusion config validated (`HYBRID_FUSION_METHOD` in `{weighted, rrf}`, weights finite, non-negative, sum <= 1.0).
    - `app.core.clients.embedding` -> loads the SentenceTransformer model and resolves `EMBEDDING_DIM`.
    - `app.core.clients.qdrant` -> constructs `QdrantClient(host, port, check_compatibility=settings.QDRANT_CHECK_COMPATIBILITY)`. The constructor does not make a request.
    - `app.core.clients.redis_cache` -> constructs `redis_cache`; with the cache disabled no Redis connection object is created.
    - `app.core.database` -> engine creation and `Base.metadata.create_all` (Postgres connection).
    - Service singletons (`PrioritizedSearchService()`, `DocumentProcessor()`, etc.) are instantiated at module level.
3. `FastAPI(...)` is created with `root_path` = `""` if `ENVIRONMENT == "local"`, otherwise `"/vector"`.
4. The ASGI lifespan runs `await ensure_collections_exist()`:
    - creates `documents` (named dense vectors, plus `bm25` sparse config if `SPARSE_SEARCH_ENABLED`) if missing;
    - if it already exists and sparse is enabled, calls `update_collection` to add the sparse field (idempotent);
    - creates `qa_cache` if missing;
    - ensures payload indexes (`_ensure_payload_indexes`), logging and continuing on per-index failures.
5. The app begins serving.

Note: `ensure_collections_exist()` is also called again on every upload/update/delete/metadata request through `BaseDocumentOperation.ensure_collections()`, and by `QueryService.process_query`. It is cheap but not free: it issues `get_collections` and `get_collection` calls each time.

## Sync vs async boundaries

FastAPI endpoints are all declared `async def`. The important point is what they call.

| Endpoint | Calls | Blocking work | Effect |
|---|---|---|---|
| `POST /documents/search` | `prioritized_search_service.search(request)` (plain `def`, not awaited) | Embedding (`SentenceTransformer.encode`), `qdrant_client.query_batch_points`, `scroll`, `retrieve`, spaCy | Runs directly on the event loop thread; blocks it for the whole request. |
| `POST /documents/text-search`, `/check-similarity`, `/verify-sources` | Sync service methods | Embedding + synchronous `QdrantClient` calls | Same: blocks the event loop. |
| `POST/PUT /documents...`, `PATCH`, `DELETE` | `await document_processor.*` | Services are `async def` but use sync Qdrant calls, sync file parsing (PyPDF2, OCR, pandas), sync `encode`, sync `upload_to_qdrant` | The coroutine only yields at genuine `await` points (for example `await file.read()` and the `httpx.AsyncClient` URL fetch). Parsing, embedding and upsert block the event loop. |
| `GET /api/health` | Sync `qdrant_client.get_collections()` | Network call | Blocks briefly. |

A repository-wide search finds no `asyncio.to_thread`, `run_in_executor`, thread pools or `asyncio.gather` in `app/`. There is no offloading of blocking calls to threads. The only explicit threading construct is the `threading.Lock` guarding lazy BM25 encoder initialisation in `sparse_encoder.py`.

`QdrantClient` is the sync client (`from qdrant_client import QdrantClient`), not `AsyncQdrantClient`.

## Concurrency and performance characteristics

- **Concurrency model**: one in-flight CPU/network-heavy request per worker process at a time, because blocking calls run on the event loop. A slow search or a large PDF upload stalls every other request, including `/api/health`, on that worker.
- **Scaling**: the production unit file (`deployment/templates/vectorization-service-uvicorn.j2`) starts uvicorn with `--workers 4` (Ansible variable `uvicorn_workers: 4`, port 9000). Parallelism comes from processes. Each worker loads its own copy of the embedding model (and the BM25 model if sparse is enabled), so memory scales with worker count. `start_mac.sh` runs a single `--reload` process for development.
- **Query path cost**: dominated by one `query_batch_points` call. It sends 5 dense queries (title, text, tags, summary, metadata) plus, in hybrid mode, 1 BM25 sparse query in a single round trip. Per-field candidate depth is `min(top_k * SEARCH_CANDIDATE_FANOUT, SEARCH_CANDIDATE_MAX)` (defaults 8 and 2000). Fusion and ranking then run in Python.
- **Late payload retrieval**: in hybrid mode the candidate phase fetches payload without `text`, and a single `qdrant_client.retrieve` fetches full payloads for the final top-K only.
- **Embedding**: one query embedding is computed and reused across all dense fields. At ingestion, chunk texts are encoded in one batch; title, summary, tags and metadata embeddings are computed once per document.
- **Upload**: `upload_to_qdrant` upserts in batches of 100 sequentially; failed batches are logged and skipped (see [Clients](../backend/clients.md#batching-and-upload)).
- **Cache**: no query caching is active.

## Design decisions and rationale

| Decision | Rationale (as evidenced in code and comments) |
|---|---|
| Five named dense vectors per point (`text`, `title`, `summary`, `tags`, `metadata`) | Lets one query be scored against several representations of a document. Document-level fields are embedded once and stored identically on every chunk; only `text` is chunk-specific. |
| Client-side fusion of dense and BM25 scores | Qdrant 1.12 server cannot run `FormulaQuery`; the service runs `query_batch_points` and fuses in `_rank_results()` (min-max weighted or RRF) to obtain a calibrated 0-1 score comparable to `filter_score`. |
| Client 1.18 against server 1.12 with `check_compatibility` off | The client is pinned to 1.18 for BM25 sparse search while the server cannot be upgraded; the compatibility warning is a false alarm for the feature set used. See [Qdrant compatibility](../operations/qdrant_compatibility.md). |
| Singletons created at import | Avoids per-request model loading and connection setup. Cost: slow import, and import-time failures (model download, Postgres). |
| Lazy BM25 encoder | The service starts and works with sparse search disabled and without fastembed models downloaded. |
| Bounded candidate pool (`SEARCH_CANDIDATE_MAX`) | Caps HNSW `ef`, the dominant query cost; a large `top_k` default (1,000,000 in the request model) cannot cause a 10k-deep traversal per field. |
| Prefix-tokenized text indexes on `title`/`summary` | Enables partial-word `MatchText` queries used by the title/summary boost step. See [Qdrant data model](qdrant_data_model.md). |
| Late payload retrieval | Avoids shipping large `text` payloads for up to 2000 candidates per field. |
| Graceful sparse degradation | If the sparse encoder cannot initialise, `_hybrid_batch_search` falls back to `_parallel_batch_search` (dense only). Ingestion likewise stores dense vectors only if sparse generation fails. |
| Redis cache disabled in code | The cache is not wired into the search path; it exists only for the legacy `QueryService`. |

## Known issues / gotchas

- **Health check is broken when the cache is disabled.** `main.py::health_check` calls `redis_cache.redis_client.ping()`. With `cache_enabled=False`, `RedisLRUCache.__init__` never sets `redis_client`, so this raises `AttributeError`, which the generic `except Exception` turns into HTTP 503 "Service unhealthy". With the default `REDIS_CACHE_ENABLED = False` the endpoint cannot report healthy.
- **Postgres is an import-time dependency** (`database.py` runs `create_all`).
- **`/api/query/` is broken.** `endpoints/query.py` is `async` but does `return query_service.process_query(request)` without `await`; `process_query` is `async def`, so the endpoint returns an un-awaited coroutine. `QueryService` also depends on Redis, Postgres models and translation.
- **Ranking weights.** `SEARCH_PRIORITY_WEIGHTS` in `config.py` is `title 0.34, text 0.26, tags 0.20, summary 0.12, metadata 0.08`; 0.36/0.27/0.14/0.14/0.09 are the `DetailFilterScore` threshold defaults.
- **Blocking calls on the event loop** (see above). Latency of one request is added to every concurrent request on the same worker.
- **`SourceVerificationService` queries `metadata.source_id`**, while ingestion writes `source_id` at the top level of the payload; verification cannot find documents. Confirmed in `source_verification_service.py`.
- **`translation_service.process_chunk` is a pass-through** (`return chunk_text, False`); translation is disabled even though `language_utils` and `TRANSLATION_API_URL` exist.

## Related pages

- [Qdrant data model](qdrant_data_model.md)
- [Clients](../backend/clients.md)
- [App lifecycle](../backend/app_lifecycle.md)
- [Search overview](../services/search/overview.md)
- [Ingestion pipeline](../services/ingestion/pipeline.md)
- [Configuration](../setup/configuration.md)
- [Deployment](../setup/deployment.md)
- [Qdrant compatibility](../operations/qdrant_compatibility.md)
