# Clients (`app/core/clients`)

Read this page to understand the four client modules that every service depends on: the Qdrant client and collection bootstrap (`qdrant.py`), the dense embedding model and vector validation (`embedding.py`), the BM25 sparse encoder (`sparse_encoder.py`) and the Redis cache (`redis_cache.py`). It documents when each singleton is created, every public function and its signature, batching and error behaviour, and which pieces are actually used at runtime.

## Summary

| Module | Singleton | Created | Used at runtime |
|---|---|---|---|
| `qdrant.py` | `qdrant_client` | Import time | Yes, everywhere |
| `embedding.py` | `embedding_model` (+ `EMBEDDING_DIM`) | Import time (loads model) | Yes |
| `sparse_encoder.py` | `_sparse_encoder` | Lazy, first use, thread-safe | Only if `SPARSE_SEARCH_ENABLED=true` |
| `redis_cache.py` | `redis_cache` | Import time (object only) | No search/ingestion path; used by legacy `QueryService`, `cache.py` and `/api/health` |

## `qdrant.py`

### Client construction

```python
qdrant_client = QdrantClient(
    settings.QDRANT_HOST,
    port=settings.QDRANT_PORT,
    check_compatibility=settings.QDRANT_CHECK_COMPATIBILITY,
)
```

This is the synchronous `QdrantClient` (HTTP, REST port 6333 by default). It is a module-level singleton shared by all services. Construction does not contact the server; the first network call happens in the lifespan hook. There are no custom timeouts, API key, or retry settings: client defaults apply.

### `check_compatibility` flag

- Setting: `QDRANT_CHECK_COMPATIBILITY` in `config.py`, `os.getenv("QDRANT_CHECK_COMPATIBILITY", "false").lower() == "true"`. Default `false`.
- Why: the deployment pins client 1.18 (needed for BM25 sparse search) against server 1.12. The minor-version gap exceeds Qdrant's allowed difference of 1, so with the check on the client emits a blanket `UserWarning: ... incompatible with server version`.
- Effect of `false`: only suppresses that warning and the version probe. It does not make newer features work on an old server. Do not use `MatchPhrase`, `FormulaQuery` or post-1.12 `TextIndexParams` fields while the server is 1.12. See [Qdrant compatibility](../operations/qdrant_compatibility.md).
- Set it to `true` once the server is upgraded.

Note: the code reads the flag from the environment.

### Module constants

```python
_PREFIX_TEXT_INDEX = models.TextIndexParams(
    type="text",
    tokenizer=models.TokenizerType.PREFIX,
    min_token_len=2,
    max_token_len=20,
    lowercase=True,
)

_PAYLOAD_INDEXES = [
    ("source_id",              PayloadSchemaType.KEYWORD),
    ("metadata.company",       PayloadSchemaType.KEYWORD),
    ("metadata.type",          PayloadSchemaType.KEYWORD),
    ("tags",                   PayloadSchemaType.KEYWORD),
    ("metadata.DOCUMENT_TYPE", PayloadSchemaType.TEXT),
    ("title",                  _PREFIX_TEXT_INDEX),
    ("summary",                _PREFIX_TEXT_INDEX),
]
```

See [Qdrant data model](../architecture/qdrant_data_model.md) for the meaning of each index.

### Public functions

| Function | Signature | Behaviour |
|---|---|---|
| `ensure_collections_exist` | `async def ensure_collections_exist()` -> `True` | Lists collections; creates `COLLECTION_NAME` with named dense vectors (and sparse config if enabled) when missing; otherwise, if sparse is enabled, calls `_ensure_sparse_vector_field`; creates `QA_CACHE_COLLECTION` if missing; calls `_ensure_payload_indexes`. Logs and re-raises any exception. |
| `batch_points` | `def batch_points(points: list, batch_size: int = 100)` | Generator yielding consecutive slices of `points`. |
| `upload_to_qdrant` | `def upload_to_qdrant(points: list, collection_name: str, batch_size: int = 100)` -> `dict` | Sequential batched `upsert`; see below. |

Private helpers: `_ensure_sparse_vector_field(collection_name)`, `_index_params_match(existing_schema, desired_schema)`, `_ensure_payload_indexes(collection_name)`.

Note: `ensure_collections_exist` is `async def` but performs only synchronous client calls; awaiting it blocks the event loop for its duration. It is called at startup and again by `BaseDocumentOperation.ensure_collections()` on each document operation and by `QueryService.process_query`.

### Collection creation

```python
if settings.COLLECTION_NAME not in collection_names:
    create_kwargs: dict = dict(
        collection_name=settings.COLLECTION_NAME,
        vectors_config=named_vectors_config,
    )
    if settings.SPARSE_SEARCH_ENABLED:
        try:
            from qdrant_client.models import SparseVectorParams, Modifier
            create_kwargs["sparse_vectors_config"] = {
                settings.SPARSE_VECTOR_NAME: SparseVectorParams(modifier=Modifier.IDF)
            }
        except ImportError:
            logger.warning("SPARSE_SEARCH_ENABLED=true but qdrant-client<1.9.0 is installed. ...")
    qdrant_client.create_collection(**create_kwargs)
elif settings.SPARSE_SEARCH_ENABLED:
    _ensure_sparse_vector_field(settings.COLLECTION_NAME)
```

The five dense vectors are each `VectorParams(size=embedding_model.get_embedding_dimension(), distance=COSINE)`. Sparse is attached only at creation time or through `_ensure_sparse_vector_field`.

### `_ensure_sparse_vector_field`

Calls `update_collection(sparse_vectors_config={bm25: SparseVectorParams(modifier=Modifier.IDF)})`. Error handling:

- `ImportError` (old client): warning, return.
- `UnexpectedResponse` whose body contains "already": debug log, treated as success.
- Any other `UnexpectedResponse`: logged and re-raised, which aborts startup via `ensure_collections_exist`.

### `_ensure_payload_indexes` and `_index_params_match`

```python
existing_schema = qdrant_client.get_collection(collection_name).payload_schema or {}
...
if current is not None and not _index_params_match(current, schema_type):
    qdrant_client.delete_payload_index(collection_name=collection_name, field_name=field_name)
    current = None
if current is None:
    qdrant_client.create_payload_index(
        collection_name=collection_name, field_name=field_name, field_schema=schema_type,
    )
```

- Each field is handled in its own `try/except`; failures are logged at WARNING and ignored (non-fatal). If the schema cannot be read, it is treated as empty and every index is (re)created.
- `_index_params_match` returns `True` for simple schema types if the field has any index; for `TextIndexParams` it compares only `tokenizer`. A one-time rebuild occurs when the tokenizer changes (for example TEXT -> PREFIX). Other parameter changes are not detected.

### Batching and upload

```python
def upload_to_qdrant(points: list, collection_name: str, batch_size: int = 100):
    ...
    for i, batch in enumerate(batch_points(points, batch_size)):
        try:
            qdrant_client.upsert(collection_name=collection_name, points=batch)
            success_count += len(batch)
        except Exception as e:
            error_count += len(batch)
            logger.error(f"Failed to upload batch {i + 1}: {str(e)}")
            continue

    return {
        "total_points": total_points,
        "success_count": success_count,
        "error_count": error_count
    }
```

- Batches are upserted sequentially, 100 points each (callers pass `batch_size=100` explicitly).
- There are no retries. A failed batch is counted in `error_count` and skipped; remaining batches continue. The function never raises, so callers must inspect `error_count`. A partially failed upload leaves a partially indexed document.
- `upsert` is called without `wait=` (client default), so it is synchronous at the API level.

## `embedding.py`

### Import-time loading

```python
embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
EMBEDDING_DIM: int = embedding_model.get_embedding_dimension()
```

The model loads when the module is first imported, which happens during application import. This is the slowest step of startup and downloads the model from Hugging Face on first run. `EMBEDDING_DIM` is derived from the loaded model, not hard-coded (384 for `all-MiniLM-L6-v2`). The model runs in-process on whatever device sentence-transformers selects; there is no GPU configuration in code. Every uvicorn worker holds its own copy.

### Public API

| Name | Signature | Behaviour |
|---|---|---|
| `EmbeddingError` | `class EmbeddingError(ValueError)` | Raised for empty input or malformed output vectors. `app/main.py` maps it to HTTP 422 with `{"detail": str(exc)}`. |
| `generate_embeddings` | `def generate_embeddings(texts: list)` | `embedding_model.encode(texts)`; returns a numpy array (batch). No validation. Used for ingestion batches. |
| `generate_single_embedding` | `def generate_single_embedding(text: str)` | `embedding_model.encode(text)`; no validation. |
| `validate_vector` | `def validate_vector(vec: Any) -> List[float]` | Converts to a Python `list` (via `tolist()` or `list()`); raises `EmbeddingError` if length != `EMBEDDING_DIM` or any value is `None`/non-finite. |
| `embed_query` | `def embed_query(text: str) -> List[float]` | Rejects empty/whitespace text with `EmbeddingError`; otherwise `validate_vector(generate_embeddings([text])[0])`. |

```python
raw = generate_embeddings([text])[0]
return validate_vector(raw)
```

Error semantics: only empty input and malformed output become `EmbeddingError` (422). Genuine model failures (for example out-of-memory `RuntimeError`) propagate unchanged and surface as 5xx. `UploadService._create_point_vectors` calls `validate_vector` on every vector it stores, so a bad vector fails ingestion before reaching Qdrant. Search services should use `embed_query` as the single entry point for query vectors.

## `sparse_encoder.py`

### Lazy singleton

```python
_sparse_encoder: Optional[object] = None
_encoder_lock = threading.Lock()
_SPARSE_MODEL = "Qdrant/bm25"

def _get_sparse_encoder():
    global _sparse_encoder
    if _sparse_encoder is not None:  # fast path — no lock
        return _sparse_encoder

    with _encoder_lock:
        if _sparse_encoder is not None:  # second check inside lock
            return _sparse_encoder
        try:
            from fastembed import SparseTextEmbedding
            _sparse_encoder = SparseTextEmbedding(model_name=_SPARSE_MODEL)
        except Exception as exc:
            logger.error(...)
            raise
    return _sparse_encoder
```

Double-checked locking with a `threading.Lock`. `fastembed` is imported inside the function, so the module imports cleanly even without it installed. The model name is a module constant (`Qdrant/bm25`); it is not configurable through settings. The model is fetched on first use. There is no startup warm-up: with sparse enabled, the first search or first ingestion pays the load cost. Failures are not cached; each later call retries initialisation.

### Public API

| Name | Signature | Behaviour |
|---|---|---|
| `generate_sparse_vector` | `def generate_sparse_vector(text: str) -> tuple[list[int], list[float]]` | Returns `([], [])` for empty/whitespace text or no result. Otherwise encodes with `list(encoder.embed([text]))` and returns `(indices.tolist(), values.tolist())`. Any exception is logged and re-raised as `RuntimeError("Sparse vector generation failed: ...")`. |
| `is_sparse_available` | `def is_sparse_available() -> bool` | Tries to load the encoder; `True` on success, `False` on any exception. |

Callers and their handling of failure:

- Search (`PrioritizedSearchService._hybrid_batch_search`): catches `ImportError` and `RuntimeError` and falls back to dense-only search (`_parallel_batch_search`). An empty sparse vector means the sparse query is not issued (`sparse_issued = bool(sparse_indices)`).
- Ingestion (`UploadService._upload_chunks`): wraps generation in `try/except Exception`; on failure logs a warning and stores dense vectors only for the whole batch. A chunk that yields no indices gets no sparse vector.

## `redis_cache.py`

### Class and instance

```python
redis_cache = RedisLRUCache(
    host=settings.REDIS_HOST,
    port=settings.REDIS_PORT,
    max_size=settings.REDIS_MAX_CACHE_SIZE,
    ttl=settings.REDIS_CACHE_TTL,
    cache_enabled=settings.REDIS_CACHE_ENABLED,
)
```

`settings.REDIS_CACHE_ENABLED` is declared as `REDIS_CACHE_ENABLED: bool = False` in `config.py`. Because the default is a literal and `BaseSettings` reads environment variables by field name, setting the env var `REDIS_CACHE_ENABLED=true` does override it under pydantic-settings; the documented claim that the variable "has no effect" is only true of the literal default, not of the settings mechanism. Treat the effective default as disabled and verify with the running configuration. `REDIS_PASSWORD` is defined in settings but is not passed to `redis.Redis`.

### Behaviour

| Method | Behaviour |
|---|---|
| `__init__(host, port, db, max_size, ttl, cache_enabled)` | If enabled, creates `redis.Redis(host, port, db, decode_responses=True)` and sets `max_size`, `ttl`, `access_list_key = "lru:access_list"`. If disabled, none of these attributes (including `redis_client`) are created. |
| `_generate_key(query)` | `"query:" + md5(query)` hex digest. |
| `get(query)` | Returns parsed JSON dict or `None`; refreshes LRU access time on hit. Returns `None` if disabled. |
| `set(query, response)` | `SETEX` with TTL and JSON-serialised response, then updates the LRU sorted set. No-op if disabled. |
| `_update_access_time(key)` | `ZADD` current timestamp to `lru:access_list`; if `ZCARD` exceeds `max_size`, evicts the oldest entries from the set and deletes their keys. |
| `clear()` | Scans `query:*` and deletes each key, then deletes the access list. |
| `remove(query)` | Deletes the key and its sorted-set member. |

There is no error handling inside the class: a Redis connection error propagates to the caller. LRU eviction is approximate (sorted set by last access timestamp, not atomic).

### It is unused by the main flows

Search the codebase: `redis_cache` is imported only by `app/services/query_service.py` (legacy multilingual query), `app/api/v1/endpoints/cache.py` (`DELETE /api/cache/redis`, which calls `clear()` and is a silent no-op when disabled), and `app/main.py::health_check`. `PrioritizedSearchService`, `TextEmbeddingSearchService`, `SimilarityService` and all ingestion services never touch it. See [Known issues](#known-issues-gotchas) for the health-check consequence.

## Known issues / gotchas

- **`/api/health` fails with the cache disabled.** `health_check` calls `redis_cache.redis_client.ping()`, but `redis_client` does not exist when `cache_enabled=False`; the `AttributeError` is caught by the generic handler and returns 503.
- **`upload_to_qdrant` swallows errors.** Failed batches are only logged and counted; there is no retry and no exception, so a caller that ignores `error_count` can report success.
- **`ensure_collections_exist` on every document request** adds `get_collections` and `get_collection` round trips and, with sparse on, an `update_collection` call each time.
- **`check_compatibility`** is read from `QDRANT_CHECK_COMPATIBILITY` (default false), not hard-coded.
- **`_index_params_match` compares tokenizer only**, so changing prefix length bounds or `lowercase` does not rebuild the index.
- **`REDIS_PASSWORD` is never used** when building the Redis client.
- **Sparse model is not configurable** and not preloaded; first sparse request is slow.
- **`embedding.py` API:** besides `generate_embeddings`/`generate_single_embedding`, the module also exposes `EmbeddingError`, `validate_vector`, `embed_query` and `EMBEDDING_DIM`.

## Related pages

- [System architecture](../architecture/system_architecture.md)
- [Qdrant data model](../architecture/qdrant_data_model.md)
- [App lifecycle](app_lifecycle.md)
- [Qdrant compatibility](../operations/qdrant_compatibility.md)
- [Search overview](../services/search/overview.md)
- [Ingestion pipeline](../services/ingestion/pipeline.md)
