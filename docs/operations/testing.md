# Testing

**Purpose.** Read this page to learn how the test suite is wired, what it covers, what it deliberately does not cover, and which parts are stale. The suite is a thin, fully mocked API contract test; it does not exercise ranking, fusion, ingestion, or Qdrant.

## 1. Layout

```text
tests/
  __init__.py
  conftest.py             fixtures, env setup, custom markers, result hook
  test_api.py             35 test methods in 11 classes (HTTP contract tests)
  logger/
    __init__.py
    test_logger.py        rotating file + console logger used by the fixtures
```

There is no `pytest.ini`, `pyproject.toml`, `setup.cfg` or `tox.ini`. All configuration is in `conftest.py`. `requirements.txt` lists `pytest`, `pytest-cov` and `pytest-asyncio`, but no test is `async` and no asyncio mode is configured.

## 2. Running

```bash
.venv/bin/python -m pytest tests -v
.venv/bin/python -m pytest tests/test_api.py::TestPrioritizedSearch -v
.venv/bin/python -m pytest tests --cov=app --cov-report=term-missing
```

Prerequisites even though everything is mocked:

- `conftest.py` imports `app.main` at collection time. That import loads the SentenceTransformer model, connects to PostgreSQL (`app/core/database.py` runs `create_all` at import), and requires all dependencies. If the import fails, `app` is set to `None` and the original exception is kept in `_app_import_error`; any test that uses the `client` fixture then raises `RuntimeError("The 'client' fixture requires app.main.app, which failed to import ...")` chained to the real traceback. Read the chained traceback, not the `RuntimeError`.
- Tests do not need a live Qdrant or Redis.
- `HTTP_422_UNPROCESSABLE_CONTENT` is used in `test_update_metadata_invalid_json`; it exists only in recent Starlette/FastAPI versions. An older FastAPI raises `AttributeError` for that test.

## 3. `conftest.py` in detail

### 3.1 Environment

Set before the app import:

```python
os.environ["ENVIRONMENT"] = "test"
os.environ["QDRANT_HOST"] = "localhost"
os.environ["QDRANT_PORT"] = "6333"
os.environ["REDIS_HOST"] = "localhost"
os.environ["REDIS_PORT"] = "6379"
os.environ["REDIS_CACHE_ENABLED"] = "False"
```

`ENVIRONMENT=test` makes `app/main.py` use `root_path="/vector"`. Starlette's `TestClient` still resolves `/api/...`, so tests call paths without the prefix. These assignments override real environment variables, but a `.env` file loaded by `load_dotenv()` does not override them; other settings (for example `POSTGRES_DATABASE_URI`) still come from `.env`.

### 3.2 Markers

`pytest_configure` registers `requires_qdrant` ("test needs a live Qdrant server; skipped if unreachable") and `compat` ("client/server version-compatibility guard (live Qdrant required)"). **No test uses either marker.** No live compatibility test exists in `tests/` as a regression guard for server 1.12. See [Qdrant compatibility](qdrant_compatibility.md).

### 3.3 Fixtures

| Fixture | Scope | What it provides |
|---|---|---|
| `logger` | session | The `test_logger` from `tests/logger/test_logger.py` (or `None` if it failed to import). |
| `log_test` | function | Logs a start banner, yields the logger, logs `PASSED`/`FAILED`/`COMPLETED` using the `rep_call` attribute set by the `pytest_runtest_makereport` hook. |
| `mock_qdrant_client` | function | `Mock` with `get_collections`, `search`, `upsert`, `delete`, `scroll`, `count`. Note `search` is a removed API (see section 6). |
| `mock_redis_client` | function | `Mock` with `ping`, `get`, `set`, `delete`, `flushdb`. |
| `client` | function | `TestClient(app)` created inside `patch` contexts for `app.core.clients.qdrant.qdrant_client`, `app.core.clients.redis_cache.redis_cache` (a mock with `cache_enabled=False`, `redis_client=mock_redis_client`, `clear=AsyncMock`) and `app.core.clients.qdrant.ensure_collections_exist`. |
| `sample_pdf_file`, `sample_text_file`, `sample_metadata`, `sample_tags` | function | Data helpers (the file fixtures are `BytesIO` objects with a `.name`). |
| `mock_document_processor` | function | Patches `app.api.v1.endpoints.documents.document_processor` with `AsyncMock` methods (`process_upload`, `update_documents`, `upsert_documents`, `update_metadata`, `delete_documents`). |
| `mock_similarity_service` | function | Patches `documents.similarity_service.check_similarity`. |
| `mock_prioritized_search_service` | function | Patches `documents.prioritized_search_service.search` with a side effect that echoes the request into a canned response. |
| `mock_text_embedding_search_service` | function | Patches `documents.text_embedding_search_service.search`; returns one hit when `threshold < 0.95`, none otherwise. |
| `mock_query_service` | function | Patches `app.api.v1.endpoints.query.query_service.process_query`. |
| `mock_redis_cache` | function | Patches `app.api.v1.endpoints.cache.redis_cache` with an `AsyncMock` `clear`. |

How patching works here: endpoint modules reference service objects as module attributes (`documents.prioritized_search_service`), so patching the attribute replaces the object the route uses. The `client` fixture's patch of `app.core.clients.qdrant.qdrant_client` only affects code that reads the name from that module at call time (the health endpoint does, via a function-local import). Services that did `from app.core.clients.qdrant import qdrant_client` at import hold the real client and are unaffected; the suite avoids them by mocking whole services.

The `ensure_collections_exist` patch is ineffective for the lifespan (`main.py` imported the name directly), but harmless: `TestClient(app)` used without a `with` block does not run lifespan events.

## 4. Coverage by class (`tests/test_api.py`)

| Class | Tests | Route exercised | What is asserted |
|---|---|---|---|
| `TestHealthCheck` | 2 | `GET /api/health` | 200 with `qdrant: connected`, `redis: connected`; 503 when `get_collections` raises. |
| `TestDocumentUpload` | 4 | `POST /api/documents` | 201, message and `source_id` echo; metadata and tags forms; optional `source_id`; priorities. |
| `TestDocumentUpdate` | 2 | `PUT /api/documents/{id}` | 200 with mocked processor. |
| `TestDocumentUpsert` | 2 | `PUT /api/documents/{id}/upsert` | create and update outcomes. |
| `TestDocumentMetadataUpdate` | 2 | `PATCH /api/documents/{id}/metadata` | 200 for valid JSON; 400 or 422 for invalid JSON. |
| `TestDocumentDelete` | 2 | `DELETE /api/documents/{id}` | with and without `company_id`. |
| `TestSimilarityCheck` | 4 | `POST /api/documents/check-similarity` | request shapes only; the service is a mock. |
| `TestPrioritizedSearch` | 8 | `POST /api/documents/search` | request-body validation and passthrough for query, filters (categories, organizations, resource_type, file_type), combined filters, `top_k`. Response is the mock's canned dictionary. |
| `TestTextEmbeddingSearch` | 4 | `POST /api/documents/text-search` | thresholds, `top_k`, empty results (mock logic). |
| `TestMultilingualQuery` | 3 | `POST /api/query/` | legacy query flow (mock). |
| `TestCacheManagement` | 2 | `DELETE /api/cache/redis` | success and failure (mock). |

Because every service is replaced by a mock, these tests verify routing, form/JSON parsing, pydantic request validation, and response-model serialization only.

## 5. What is not covered

- No test of `PrioritizedSearchService` internals: `_rank_results`, min-max normalization, RRF, boosts, candidate limits, late payload retrieval, dedup. These are the most failure-prone parts of the system.
- No test of file processors (PDF/OCR, DOCX, XLSX, CSV, text), chunking, translation, URL extraction.
- No test of `UploadService`, `UpdateService`, `DeleteService`, `MetadataService` logic or of `ensure_collections_exist` and payload index management.
- No test for `POST /api/documents/verify-sources`.
- No test of `config.py` validation (`HYBRID_FUSION_METHOD`, weight sums).
- No live-Qdrant or compatibility test (see section 3.2). A real server is the only way to detect the HTTP 400 `did not match any variant of untagged enum` class of regression.
- No assertions on `include_scoring_debug` output.

## 6. Stale or misleading elements

| Item | Problem |
|---|---|
| `mock_qdrant_client.search` | `QdrantClient.search` was removed in client 1.14-1.16; production code uses `query_points` and `query_batch_points`. The mock gives no signal about those. |
| `mock_document_processor` return values | Keys `chunks_created`, `deleted_count`, `updated_count` differ from the real upload response (`chunks_processed`, `points_uploaded`, `upload_failures`, ...). Tests cannot catch response-shape drift. |
| `pytest_sessionfinish` at the bottom of `test_api.py` | Pytest registers hooks only from `conftest.py` and plugins, not from test modules, so this summary logger is not invoked. The `test_results.log` summary block is therefore not written (inference from pytest's plugin model; the per-test banners from `log_test` are written). |
| `log_test_summary` in `tests/logger/test_logger.py` | Same: only reachable from that hook. |
| `Mock(name="documents")` in `mock_qdrant_client.get_collections` | `name=` configures the mock's repr, not a `.name` attribute; any code reading `collection.name` from this mock would get a `Mock`, not `"documents"`. |
| `tests/logger/test_results.log` | Written next to the logger (`LOGGER_DIR`), rotating at 10 MB with 5 backups; ignored by Git through `*.log`. |

## 7. Adding tests

Guidelines that follow from the structure:

1. Unit-test ranking and fusion directly by instantiating `PrioritizedSearchService` and calling `_rank_results(all_results, field_scores, weights, search_fields, ...)` with hand-built dictionaries. Its signature takes only in-memory structures, so no Qdrant call is made by the method itself (the constructor reads `settings` only; the module-level service instance in `documents.py` is created at import). Cover both `HYBRID_FUSION_METHOD` values by patching `settings.HYBRID_FUSION_METHOD`.
2. For anything that talks to Qdrant, add a test marked `requires_qdrant` and skip when the server is unreachable (the marker is registered but no skip logic exists yet; add it in `conftest.py`).
3. For the compatibility guard, create `tests/test_qdrant_compat.py` marked `compat` that issues `query_batch_points` with five dense requests plus one sparse request, `scroll` with `MatchText` and `MatchAny`, and `retrieve` against the configured server.
4. Prefer fixtures that patch the module attribute the code under test actually reads.
5. Keep `ENVIRONMENT=test` semantics in mind: URLs in `TestClient` calls omit `/vector`.

## Known issues / gotchas

- The suite is mock-only; a green run does not demonstrate search correctness.
- No live compatibility test exists in `tests/`; the `compat` and `requires_qdrant` markers are unused.
- The session summary hook in `test_api.py` is never executed.
- The `client` fixture needs the full app import (model load, PostgreSQL) to succeed.
- The only `test_q.py` is an empty file at the repository root; `tests/` has no such file.

## Related pages

- [Developer setup](../setup/developer_setup.md)
- [Qdrant compatibility](qdrant_compatibility.md)
- [Troubleshooting](troubleshooting.md)
- [API endpoints](../api/endpoints.md)
- [Ranking and fusion](../services/search/ranking_and_fusion.md)
