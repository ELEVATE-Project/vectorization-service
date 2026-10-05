# API Endpoints Reference

**Purpose.** Read this page to understand every HTTP route exposed by the service: its method, full path, accepted parameters, response shape, status codes and a working example. Request/response field tables live in [API models](models.md); the behaviour behind each route is documented in the service pages linked at the end.

## 1. Base URL, `root_path` and authentication

Routes are assembled in `app/main.py` and `app/api/v1/api.py`:

```python
# app/main.py
root_path_config = "" if settings.ENVIRONMENT == "local" else "/vector"
app = FastAPI(..., root_path=root_path_config, default_response_class=CustomJSONResponse)
app.include_router(api_router, prefix="/api")
```

```python
# app/api/v1/api.py
api_router.include_router(documents.router, prefix="", tags=["documents"])
api_router.include_router(query.router, prefix="/query", tags=["query"])
api_router.include_router(cache.router, prefix="/cache", tags=["cache"])
```

| `ENVIRONMENT` | `root_path` | Public URL example |
|---|---|---|
| `local` | `""` | `http://localhost:8000/api/documents/search` |
| anything else | `/vector` | `https://host/vector/api/documents/search` |

Note: FastAPI's `root_path` does **not** change route registration. The application still serves `/api/...`; `/vector` is the prefix a reverse proxy is expected to strip before forwarding. It affects the generated OpenAPI `servers` entry and Swagger UI, so `/docs` behind the proxy resolves `openapi.json` under `/vector`.

**Authentication: none.** There is no auth dependency, API key check, CORS middleware or rate limiting anywhere in `app/`. Access control must be enforced by the network layer / gateway.

**Note on the `/v1` segment.** The Python package is `app/api/v1/`, but the URL does **not** contain `v1`. The real paths are `/api/documents...`, not `/api/v1/documents...`.

## 2. Route summary

| Method | Path (local) | Handler | Request | Success |
|---|---|---|---|---|
| POST | `/api/documents` | `create_documents` | multipart form | 201 |
| PUT | `/api/documents/{source_id}` | `update_documents` | multipart form | 200 |
| PUT | `/api/documents/{source_id}/upsert` | `upsert_documents` | multipart form | 200 |
| PATCH | `/api/documents/{source_id}/metadata` | `update_document_metadata` | form | 200 |
| DELETE | `/api/documents/{source_id}` | `delete_documents` | form (optional) | 200 |
| POST | `/api/documents/check-similarity` | `check_similarity` | JSON | 200 |
| POST | `/api/documents/search` | `prioritized_search` | JSON | 200 |
| POST | `/api/documents/text-search` | `text_embedding_search` | JSON | 200 |
| POST | `/api/documents/verify-sources` | `verify_sources` | JSON | 200 |
| POST | `/api/query/` | `query_documents` | JSON | see known issues |
| DELETE | `/api/cache/redis` | `clear_redis_cache` | none | 200 |
| GET | `/api/health` | `health_check` (in `main.py`) | none | 200 / 503 |

All document routes are in `app/api/v1/endpoints/documents.py`; every handler instantiates a module-level service (`DocumentProcessor`, `SimilarityService`, `PrioritizedSearchService`, `TextEmbeddingSearchService`) or uses the `source_verification_service` singleton.

## 3. Dependency-injection form parsers

Multipart forms cannot carry nested JSON natively, so `POST /api/documents` uses two `Depends(...)` parsers defined at the top of `documents.py`.

### `parse_metadata_form(metadata: Optional[str] = Form(default=None))`

| Input | Result |
|---|---|
| missing / empty / whitespace | `None` |
| JSON object string | parsed `dict` |
| JSON that is not an object (e.g. `[1]`) | `HTTPException(400, "Metadata must be a JSON object/dict")` |
| invalid JSON | `HTTPException(400, "Invalid metadata JSON: <decode error>")` |

### `parse_tags_form(tags: Optional[str] = Form(default=None))`

| Input | Result |
|---|---|
| missing / empty | `None` |
| starts with `[` | `json.loads`; must be a list, else 400 `"Tags must be a JSON array/list"`; bad JSON gives 400 `"Invalid tags JSON: ..."` |
| anything else | split on `,`, each item stripped, empties dropped |

Note: JSON array elements are **not** type-checked; `["a", 1]` is accepted as-is.

Only `POST /documents` uses these parsers. The PUT / upsert routes take `metadata` as a raw string `Form` and pass it to the service, which parses it (see [Document operations](../services/ingestion/document_operations.md)). PATCH parses its own `metadata_updates` with `json.loads` inline.

## 4. Document lifecycle endpoints

### 4.1 `POST /api/documents` (create)

Status code is declared `201`.

| Form field | Type | Default | Notes |
|---|---|---|---|
| `file` | file | required | PDF, DOCX, XLSX, CSV, TXT (see [file processors](../services/ingestion/file_processors.md)) |
| `priority` | str | `"P1"` | must be `P*` format, validated in `UploadService` |
| `source_id` | str | `None` | validated non-empty in the service (400 if empty) |
| `company_id` | str | `None` | stored as `metadata.company` |
| `title` | str | `None` | document-level title (embedded into `title` vector) |
| `summary` | str | `None` | document-level summary |
| `metadata` | JSON string | `None` | via `parse_metadata_form`; `metadata.markdown_url` switches to URL extraction |
| `tags` | JSON array or CSV string | `None` | via `parse_tags_form` |

```bash
curl -X POST http://localhost:8000/api/documents \
  -F "file=@policy.pdf" \
  -F "priority=P1" \
  -F "source_id=doc_123" \
  -F "company_id=acme" \
  -F "title=Leave Policy" \
  -F "summary=Annual leave rules" \
  -F 'metadata={"DOCUMENT_TYPE":"policy","type":"pdf"}' \
  -F 'tags=["hr","leave"]'
```

Response body (from `UploadService.process`; the status field is `"success"`):

```json
{
  "status": "success",
  "message": "Successfully processed 12 chunks from policy.pdf",
  "chunks_processed": 12,
  "points_uploaded": 12,
  "upload_failures": 0,
  "file_type": "pdf",
  "priority": "P1",
  "source_id": "doc_123",
  "company_id": "acme",
  "title": "Leave Policy",
  "summary": "Annual leave rules",
  "tags": ["hr", "leave"],
  "supported_file_types": ["..."],
  "sample_chunk": {"text": "...", "metadata": {"...": "..."}}
}
```

Note: `file_type` is the lowercase extension without a dot (`file.filename.split(".")[-1].lower()`), or `url_extracted` for URL ingestion. Status codes: 201 success; 400 invalid metadata/tags, empty `source_id`, invalid priority, unsupported type, or no extractable content; 422 FastAPI validation (missing `file`); 500 `"Upload failed: ..."`.

### 4.2 `PUT /api/documents/{source_id}` (replace)

Deletes all existing points for `source_id` (and `company_id` if supplied) and re-uploads. Form: `file` (required), `priority` (default `P1`), `metadata` (raw JSON string), `company_id`. Returns 404 if nothing exists to replace.

```bash
curl -X PUT http://localhost:8000/api/documents/doc_123 \
  -F "file=@policy_v2.pdf" -F "company_id=acme" -F 'metadata={"type":"pdf"}'
```

```json
{
  "status": "success",
  "operation": "update",
  "message": "Successfully updated documents for source_id: doc_123",
  "documents_deleted": 12,
  "previous_document_count": 12,
  "chunks_processed": 14,
  "points_uploaded": 14,
  "upload_failures": 0,
  "file_type": "pdf",
  "priority": "P1",
  "source_id": "doc_123"
}
```

Note: the route does not accept `title`, `summary` or `tags`; they are lost on update (the handler signature has no such fields and `UpdateService` calls `UploadService.process` with only five arguments).

### 4.3 `PUT /api/documents/{source_id}/upsert`

Same form as 4.2. Response has `"operation": "updated"` when documents existed, `"created"` otherwise; fields otherwise match 4.2 (`documents_deleted`, `previous_document_count`, `chunks_processed`, `points_uploaded`, `upload_failures`, ...). Does not 404 on missing documents.

### 4.4 `PATCH /api/documents/{source_id}/metadata`

Form fields: `metadata_updates` (required JSON string), `company_id` (optional). Invalid JSON gives 400 `"Invalid metadata JSON"`. Payload patched with `set_payload`; vectors are **not** regenerated.

```bash
curl -X PATCH http://localhost:8000/api/documents/doc_123/metadata \
  -F 'metadata_updates={"priority":"P2","department":"HR"}' -F "company_id=acme"
```

```json
{
  "status": "success",
  "message": "Successfully updated metadata for 12 documents",
  "documents_updated": 12,
  "source_id": "doc_123",
  "company_id": "acme",
  "metadata_updates": {"priority": "P2", "department": "HR"}
}
```

Status codes: 200; 400 (invalid JSON, or a `company` value in the updates that differs from the supplied `company_id`); 404 if no points match; 500.

### 4.5 `DELETE /api/documents/{source_id}`

`company_id` is declared as `Form(default=None)`, so it must be sent as a form body on a DELETE request (`-F "company_id=acme"` / `--data-urlencode`). A query-string `?company_id=` is **ignored**.

```bash
curl -X DELETE http://localhost:8000/api/documents/doc_123 -F "company_id=acme"
```

```json
{
  "status": "success",
  "message": "Successfully deleted all 12 documents with source ID: doc_123 and company ID: acme",
  "documents_deleted": 12,
  "source_id": "doc_123",
  "company_id": "acme"
}
```

Status codes: 200; 404 `"No documents found for source ID: ..."`; 500 `"Delete failed: ..."`. The handler builds a `DeleteRequest` internally.

## 5. Search and utility endpoints

### 5.1 `POST /api/documents/search`

Request model `PrioritizedSearchRequest`, response model `PrioritizedSearchResponse` (see [API models](models.md)). The handler logs the full request body (`request.model_dump_json()`) at INFO and calls `PrioritizedSearchService.search`. Algorithm: [Search overview](../services/search/overview.md).

```bash
curl -X POST http://localhost:8000/api/documents/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning algorithms",
    "top_k": 10,
    "filter_score": 0.3,
    "organizations": ["acme"],
    "file_type": ["pdf"],
    "exclude_organizations": ["legacy_co"],
    "search_mode": "hybrid",
    "include_scoring_debug": false
  }'
```

```json
{
  "query": "machine learning algorithms",
  "total_results": 1,
  "top_k": 10,
  "results": [
    {
      "id": "7b3e...-uuid",
      "text": "Machine learning algorithms are...",
      "title": "ML Guide",
      "summary": "Intro to ML",
      "tags": ["ml"],
      "metadata": {"company": "acme", "type": "pdf"},
      "source_id": "doc_123",
      "score": 0.82,
      "field_scores": {"title": 0.7, "text": 0.6, "tags": 0.3, "summary": 0.5, "metadata": 0.2},
      "match_source": null,
      "keyword_score": null,
      "title_match": null,
      "summary_match": null
    }
  ],
  "search_config": {"fusion_method": "weighted"}
}
```

Notes:

- Omitting `query` returns one document per unique `source_id` (scroll mode).
- `top_k` defaults to 1,000,000 and the model has **no** `gt=0` constraint; the endpoint docstring claim "422 if top_k <= 0" is not enforced by Pydantic. Check the service for any runtime guard.
- Debug-only fields (`rrf_score`, `dense_rank`, `sparse_rank`, `raw_dense`, `normalized_dense`, `normalized_sparse`, `title_multiplier`, `summary_multiplier`) are populated only when `include_scoring_debug=true`; `keyword_score` is surfaced whenever sparse search produced a score.
- An invalid `search_mode`, a lone-entry `any_of`, an empty `any_of` block or a misspelled key inside a block yields 422.
- `EmbeddingError` (empty or whitespace query vector) is mapped to 422 by the global handler (section 7).

### 5.2 `POST /api/documents/text-search`

Simpler single-field search; see [Auxiliary services](../services/auxiliary_services.md).

```bash
curl -X POST http://localhost:8000/api/documents/text-search \
  -H "Content-Type: application/json" \
  -d '{"query": "machine learning algorithms", "top_k": 5, "threshold": 0.4}'
```

```json
{
  "query": "machine learning algorithms",
  "total_results": 2,
  "results": [
    {"source_id": "doc_123", "text": "Machine learning algorithms are...", "score": 0.89, "metadata": {"company": "acme"}},
    {"source_id": "doc_123", "text": "Another chunk of the same doc...", "score": 0.71, "metadata": {"company": "acme"}}
  ]
}
```

Note that multiple chunks of the same `source_id` can appear (the docstring says one per document; the code does not de-duplicate).

### 5.3 `POST /api/documents/check-similarity`

```bash
curl -X POST http://localhost:8000/api/documents/check-similarity \
  -H "Content-Type: application/json" \
  -d '{"text": "Annual leave policy ...", "company_id": "acme", "threshold": 0.85, "exclude_source_id": "doc_123"}'
```

```json
{
  "has_similar": true,
  "similar_documents": [
    {
      "source_id": "doc_456",
      "similarity_score": 0.91,
      "metadata": {"company": "acme"},
      "text_preview": "First 200 characters of the chunk...",
      "chunk_id": "0f2a...-uuid"
    }
  ]
}
```

Status codes: 200; 422 for empty text (`min_length=1`, or whitespace-only via `EmbeddingError`); 500 `"Similarity check failed: ..."`.

### 5.4 `POST /api/documents/verify-sources`

```bash
curl -X POST http://localhost:8000/api/documents/verify-sources \
  -H "Content-Type: application/json" -d '{"source_ids": ["doc_123", "missing_1"]}'
```

```json
{"total_requested": 2, "found": [], "not_found": ["doc_123", "missing_1"], "found_count": 0, "not_found_count": 2}
```

The example shows the **actual current behaviour**: because of a field-path bug every ID is reported as `not_found` (see [Auxiliary services](../services/auxiliary_services.md)).

### 5.5 `POST /api/query/`

Multilingual single-field query. Registered with prefix `/query` and route `"/"`, so the canonical path has a trailing slash (`/api/query/`); `/api/query` gets a 307 redirect. Body: `MultilingualQueryRequest` (`query`, `search_limit`, `priority_filter`). Response model: `MultilingualQueryResponse`.

```bash
curl -X POST http://localhost:8000/api/query/ -H "Content-Type: application/json" \
  -d '{"query": "leave policy", "search_limit": 5, "priority_filter": "P1"}'
```

Note: this route is broken in the current code (un-awaited coroutine, see section 8 and [Auxiliary services](../services/auxiliary_services.md)).

### 5.6 `DELETE /api/cache/redis`

Calls `redis_cache.clear()` and returns `{"message": "Redis cache cleared successfully"}`; any exception becomes 500 with `detail=str(e)`. Note: with `REDIS_CACHE_ENABLED=false` (the default, hardcoded) `RedisCache.clear()` returns immediately, yet the endpoint still reports success.

### 5.7 `GET /api/health`

Defined directly on `app` (not on a router) in `main.py`. Calls `qdrant_client.get_collections()` and `redis_cache.redis_client.ping()`.

```json
{"status": "healthy", "services": {"qdrant": "connected", "redis": "connected"}}
```

Failures: 503 `"Redis connection failed"` (`redis.ConnectionError`) or 503 `"Service unhealthy"` (any other exception). See [App lifecycle](../backend/app_lifecycle.md) for a default-config caveat.

## 6. Status code conventions

| Code | Source |
|---|---|
| 201 | `POST /documents` only |
| 400 | form parser errors, service validation (`source_id`, `priority`, unsupported file type, empty extraction, protected metadata key) |
| 404 | update/patch/delete with no matching points |
| 422 | Pydantic validation; `EmbeddingError` handler |
| 500 | wrapped service exceptions (`"<Operation> failed: ..."`) |
| 503 | `/api/health` only |

## 7. Global error handling

`main.py` registers one custom handler:

```python
@app.exception_handler(EmbeddingError)
async def embedding_error_handler(request, exc: EmbeddingError):
    return CustomJSONResponse(status_code=422, content={"detail": str(exc)})
```

`EmbeddingError` (subclass of `ValueError`, `app/core/clients/embedding.py`) is raised by `embed_query` for empty/whitespace text or malformed vectors. Services that wrap errors into 500 (`SimilarityService`, `QueryService`) explicitly re-raise it first so this handler wins.

## 8. Known issues / gotchas

- `/api/query/` returns `query_service.process_query(request)` without `await`, but `process_query` is `async def`. FastAPI receives a coroutine object and the response-model validation fails (500). Tests hide this by patching `app.api.v1.endpoints.query.query_service`.
- `DELETE /documents/{source_id}` reads `company_id` from form data, not the query string.
- `PUT` and `/upsert` drop `title`, `summary`, `tags`.
- `documents.py` docstring for search says filters use OR between types; the implementation uses AND between filter types (see [Boosts, filters and results](../services/search/boosts_filters_and_results.md)).
- `text-search` docstring says default `top_k` is 5; the model default is 10.
- `/api/health` can return 503 in the default configuration (Redis client absent when cache is disabled).

## Related pages

- [API models](models.md)
- [App lifecycle](../backend/app_lifecycle.md)
- [Search overview](../services/search/overview.md)
- [Ingestion pipeline](../services/ingestion/pipeline.md)
- [Document operations](../services/ingestion/document_operations.md)
- [Auxiliary services](../services/auxiliary_services.md)
