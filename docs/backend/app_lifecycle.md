# Application Lifecycle

**Purpose.** Read this page to understand how the FastAPI app is constructed, what runs at startup and shutdown, how routers are aggregated, how JSON is serialized, how the health check behaves, and why the "unused" PostgreSQL layer is in fact imported at startup.

## 1. App factory (`app/main.py`)

There is no factory function; `app/main.py` builds a module-level `app`:

```python
root_path_config = "" if settings.ENVIRONMENT == "local" else "/vector"

app = FastAPI(
    title="Vector Search API",
    description="API for document ingestion, search, and management with vector database",
    version="1.0.0",
    lifespan=lifespan,
    root_path=root_path_config,
    default_response_class=CustomJSONResponse
)
app.include_router(api_router, prefix="/api")
```

Run with `uvicorn app.main:app`; `python -m app.main` runs `uvicorn.run(app, host="0.0.0.0", port=8000)`.

Key points:

- `root_path` is `""` when `ENVIRONMENT == "local"`, otherwise `/vector`. It only informs OpenAPI/Swagger about the proxy prefix; routing is unchanged (see [API endpoints](../api/endpoints.md)).
- **No middleware**: no CORS, no auth, no GZip, no request logging, no rate limiting.
- `default_response_class=CustomJSONResponse` applies to every route that does not return a `Response` directly.

## 2. Lifespan

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await ensure_collections_exist()
        logger.info("Application startup completed")
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        raise
    yield
    try:
        logger.info("Application shutdown completed")
    except Exception as e: ...
```

- **Startup:** `ensure_collections_exist()` (`app/core/clients/qdrant.py`) creates the `documents` and `qa_cache` collections and the payload indexes if missing (details in [Clients](clients.md) and [Qdrant data model](../architecture/qdrant_data_model.md)). Any exception is logged and re-raised, so the process **fails to start** if Qdrant is unreachable.
- **Shutdown:** only logs; Qdrant/Redis clients are not closed explicitly.
- Heavy singletons are not created in lifespan; they load at **import time**: the SentenceTransformer model (`app/core/clients/embedding.py`), the Qdrant client, the `redis_cache` object, and the module-level services in `documents.py`. The spaCy model and BM25 encoder are lazy (first use).

```text
import app.main
  -> app.api.v1.api -> endpoints.documents  (services, embedding model load)
                    -> endpoints.query      (QueryService -> core.database -> Postgres create_all)
                    -> endpoints.cache      (redis_cache)
  -> FastAPI(...) + include_router
startup: lifespan -> ensure_collections_exist()
```

## 3. Router aggregation (`app/api/v1/api.py`)

| Router | Prefix | Tag | Routes |
|---|---|---|---|
| `documents.router` | `""` | documents | `/documents...` |
| `query.router` | `/query` | query | `POST /query/` |
| `cache.router` | `/cache` | cache | `DELETE /cache/redis` |

All mounted under `/api` by `main.py`. The health route is registered directly on `app` as `GET /api/health`, outside the aggregated router. The `v1` is only a package name; it is not part of any URL.

## 4. Exception handling

One handler is registered: `EmbeddingError` becomes HTTP 422 with `{"detail": "<message>"}` via `CustomJSONResponse`. Everything else relies on FastAPI defaults and per-service `HTTPException` wrapping.

## 5. Custom JSON response (`app/utils/json_handler.py`)

`CustomJSONResponse.render()` first runs `_clean_floats` (recursively replacing `NaN` / `+-Infinity` floats with `None`, descending into dicts, lists and tuples), then `json.dumps(..., cls=CustomJSONEncoder, ensure_ascii=False, allow_nan=False, separators=(",", ":"))`. Effects: compact output, non-ASCII (Hindi) text emitted literally, and NaN scores become `null` rather than crashing serialization. Details in [Utils](utils.md).

Note: `_clean_floats` does not traverse non-container objects; because FastAPI serializes Pydantic models to dicts via `jsonable_encoder` before `render`, response-model output is covered. `CustomJSONEncoder.default` is only reached for non-float unknown types, so its NaN branch is effectively dead code. `setup_custom_json_handling(app)` exists but is **never called**.

## 6. Health check

```python
@app.get("/api/health")
async def health_check():
    qdrant_client.get_collections()
    redis_cache.redis_client.ping()
    return {"status": "healthy", "services": {"qdrant": "connected", "redis": "connected"}}
```

Failures: `redis.ConnectionError` gives 503 `"Redis connection failed"`; any other exception gives 503 `"Service unhealthy"`.

Gotcha: `redis_cache` is created with `cache_enabled=settings.REDIS_CACHE_ENABLED`, which is hardcoded `False`. In that mode `RedisCache.__init__` never sets `redis_client`, so `redis_cache.redis_client` raises `AttributeError`, caught by the generic handler: **the endpoint returns 503 "Service unhealthy" in the default configuration** even when Qdrant is fine. Load-balancer health probes must account for this (or the code must be fixed to skip Redis when disabled). The check is also synchronous and blocking inside an `async def`.

## 7. The "unused" database layer

Files: `app/core/database.py`, `app/models/db_models.py`.

```python
engine = create_engine(settings.DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base.metadata.create_all(bind=engine)   # runs at import time
def get_db(): ...                        # FastAPI-style generator; never used as a Depends
```

`settings.DATABASE_URL` reads env `POSTGRES_DATABASE_URI` (default `postgresql://anuj:1234@localhost:5432/ai_vector_service`, a developer credential left in `app/config.py`). The single model is `TranslationRecord` (`translations` table: `chunk_id`, `original_text`, `translated_text`, `created_at`).

PostgreSQL is only partly used. No code **writes** `TranslationRecord` (`translation_service.process_chunk` is a stub returning `(chunk_text, False)`), and `get_db` is never injected. But the module **is imported at startup**:

- `app/services/translation_service.py` imports `SessionLocal` and `TranslationRecord`, and is imported by `pdf_processor`, `docx_processor`, `xlsx_processor`, `csv_processor`, `text_processor` and (lazily) `upload_service`.
- `app/services/query_service.py` imports `SessionLocal`, `TranslationRecord`, and is instantiated by `endpoints/query.py` at import.

Because `Base.metadata.create_all(bind=engine)` executes on first import of `app.core.database`, **the process opens a connection to PostgreSQL and attempts to create the `translations` table while importing**. If Postgres is unreachable or the credentials are wrong, application import fails with an `OperationalError` before lifespan runs. Deployments must therefore provide a reachable Postgres (or the import path must be changed). Request-time use is limited to `QueryService._process_search_results` (the broken legacy route).

## 8. Known issues / gotchas

- Health check returns 503 by default (Redis client missing when cache is disabled).
- Postgres is a hard import-time dependency despite being functionally unused.
- Default Postgres URI contains a hardcoded user/password.
- `setup_custom_json_handling` is dead code; `CustomJSONEncoder` NaN branch is unreachable in practice.
- Shutdown hook does nothing; no client cleanup.
- Logging: `app/utils/language_utils.py` calls `logging.basicConfig(...)` with a `FileHandler('app.log')` at import, so importing the app writes `app.log` into the working directory and fixes the root logger config.
- Redis cache and Postgres translation features are both disabled/stubbed; only Qdrant is functionally required at runtime.

## Related pages

- [API endpoints](../api/endpoints.md)
- [API models](../api/models.md)
- [Clients](clients.md)
- [Utils](utils.md)
- [System architecture](../architecture/system_architecture.md)
- [Configuration](../setup/configuration.md)
- [Auxiliary services](../services/auxiliary_services.md)
