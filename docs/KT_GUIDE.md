# Vectorization Service Knowledge Transfer Guide

## Purpose

This page is the single index of every module in the Vectorization Service, intended for structuring knowledge-transfer (KT) sessions and onboarding. For each module it records where the code lives, where it is documented, and how reliable that documentation is. It is a navigation and tracking document; the module pages it links to carry the detail.

The reference pages are hand-written, not generated. Revisit this guide whenever a module is added, removed, or substantially changed.

## System Overview

A single FastAPI application (`app/main.py`) mounts one API router under `/api`. Requests flow through thin endpoint functions into service classes, which call three clients: Qdrant (storage and search), a SentenceTransformer model (dense embeddings, loaded at import), and a fastembed BM25 encoder (sparse embeddings, lazy).

```text
HTTP ──► app/api/v1/endpoints/documents.py
            │
            ├─ upload / update / delete / patch ─► DocumentProcessor ─► document_operations/*
            │                                          └─ file_processors/* , url_text_extractor
            ├─ search ───────────────────────────► PrioritizedSearchService   (core of the system)
            └─ text-search / check-similarity / verify-sources ─► auxiliary services
                                   │
                    app/core/clients: qdrant · embedding · sparse_encoder · redis_cache
```

Two flows carry almost all of the complexity: **ingestion** (file → chunks → five dense vectors + BM25 → Qdrant points) and **search** (query → embedding → batch multi-field query → fusion → boosts → de-duplication). Complete the reading order below before touching either.

## Recommended Reading Order

Go through these in sequence; each page assumes the ones before it.

1. [System Architecture](architecture/system_architecture.md) — components, startup, sync/async boundaries.
2. [Qdrant Data Model](architecture/qdrant_data_model.md) — collections, named vectors, payload, indexes. Every other page depends on this.
3. [Configuration Reference](setup/configuration.md) — every setting and which code reads it.
4. [Upload Pipeline](services/ingestion/pipeline.md) — how a document becomes points.
5. [Search Pipeline Overview](services/search/overview.md), then [Ranking and Fusion](services/search/ranking_and_fusion.md), then [Boosts, Filters and Results](services/search/boosts_filters_and_results.md).
6. [Update, Delete and Metadata](services/ingestion/document_operations.md) — mutation semantics and their sharp edges.
7. [API Endpoints](api/endpoints.md) and [Models](api/models.md) — the external contract.
8. [Qdrant Compatibility](operations/qdrant_compatibility.md) — constraints on what queries may be introduced.
9. [Migration and Restore](operations/migration_and_restore.md) — moving, backing up and re-indexing data.
10. [Failure Modes and Recovery](operations/failure_modes_and_recovery.md) — what fails, what state it leaves, how to recover.
11. [Troubleshooting](operations/troubleshooting.md).

## Topic-wise Reference Map

Use this table to find the page for a specific topic. Topics are listed in the order in which they are best studied: ingestion first, then storage, then retrieval, then lifecycle operations, then failure handling. Sections named in the Where column are on the linked page.

| # | Topic | Primary reference | Also read |
|---|---|---|---|
| 1 | End-to-end document upload and ingestion pipeline | [Upload Pipeline](services/ingestion/pipeline.md) | [API Endpoints](api/endpoints.md) (upload request contract) |
| 2 | Parsing and processing flow for PDF, text and Markdown | [File Processors](services/ingestion/file_processors.md) (PDF, DOCX, text sections) | [Upload Pipeline](services/ingestion/pipeline.md) (URL path) |
| 3 | CSV and Excel processing and ingestion into Qdrant | [File Processors](services/ingestion/file_processors.md) (XLSX, CSV sections) | [Upload Pipeline](services/ingestion/pipeline.md) (point construction and upsert) |
| 4 | Chunking strategy and document segmentation | [File Processors](services/ingestion/file_processors.md) (extension map and settings) | [Configuration Reference](setup/configuration.md) (chunk settings) |
| 5 | Chunk overlap strategy and rationale | [File Processors](services/ingestion/file_processors.md) (section 2.1, chunk overlap) | [Migration and Restore](operations/migration_and_restore.md) (re-chunking existing documents) |
| 6 | Embedding generation and vectorization flow | [Upload Pipeline](services/ingestion/pipeline.md) (embeddings) | [Clients](backend/clients.md) (embedding and sparse encoder) |
| 7 | Qdrant collection structure and named vector configuration | [Qdrant Data Model](architecture/qdrant_data_model.md) | [Clients](backend/clients.md) (collection and index setup) |
| 8 | Payload structure and metadata handling | [Qdrant Data Model](architecture/qdrant_data_model.md) (payload schema) | [Upload Pipeline](services/ingestion/pipeline.md) (metadata merging) |
| 9 | Document update flow in Qdrant without re-uploading the file | [Update, Delete and Metadata](services/ingestion/document_operations.md) (metadata patch) | [API Endpoints](api/endpoints.md) (`PATCH` contract) |
| 10 | Search and retrieval mechanism, in depth | [Search Pipeline Overview](services/search/overview.md) | [Ranking and Fusion](services/search/ranking_and_fusion.md), [Boosts, Filters and Results](services/search/boosts_filters_and_results.md) |
| 11 | Dense vector search and similarity scoring | [Ranking and Fusion](services/search/ranking_and_fusion.md) | [Search Pipeline Overview](services/search/overview.md) |
| 12 | Search filters and metadata filtering | [Boosts, Filters and Results](services/search/boosts_filters_and_results.md) | [Qdrant Data Model](architecture/qdrant_data_model.md) (filter-to-field mapping) |
| 13 | Document deletion and re-indexing flow | [Update, Delete and Metadata](services/ingestion/document_operations.md) (delete, `PUT`, upsert) | [Migration and Restore](operations/migration_and_restore.md) (re-indexing procedures) |
| 14 | Migration, restore and Qdrant collection migration | [Migration and Restore](operations/migration_and_restore.md) | [Scripts](operations/scripts.md), [Qdrant Compatibility](operations/qdrant_compatibility.md) |
| 15 | Error handling, failure scenarios and recovery | [Failure Modes and Recovery](operations/failure_modes_and_recovery.md) | [Known Issues Register](#known-issues-register) |
| 16 | Troubleshooting and debugging of ingestion and search issues | [Troubleshooting](operations/troubleshooting.md) | [Failure Modes and Recovery](operations/failure_modes_and_recovery.md) |

## Module Checklist

Coverage key: **Full** — documented against the code, including behaviour and gotchas; **Partial** — documented, but narrower than the module; **Gap** — not documented.

### 1. Foundations

| Module | Source | Documentation | Coverage |
|---|---|---|---|
| App factory, lifespan, health check | `app/main.py` | [App Lifecycle](backend/app_lifecycle.md) | Full |
| Settings | `app/config.py` | [Configuration Reference](setup/configuration.md) | Full |
| Router aggregation | `app/api/v1/api.py` | [App Lifecycle](backend/app_lifecycle.md) | Full |
| JSON response handling | `app/utils/json_handler.py` | [App Lifecycle](backend/app_lifecycle.md), [Utilities](backend/utils.md) | Full |
| Database layer (SQLAlchemy) | `app/core/database.py`, `app/models/db_models.py` | [App Lifecycle](backend/app_lifecycle.md) | Full — note it runs at import time (see register below) |
| Local environment | `start_mac.sh`, `.env.sample` | [Developer Setup](setup/developer_setup.md) | Full |

### 2. Request/Response Surface

| Module | Source | Documentation | Coverage |
|---|---|---|---|
| Endpoints | `app/api/v1/endpoints/` | [API Endpoints](api/endpoints.md) | Full |
| Pydantic models | `app/models/api_models.py` | [Models](api/models.md) | Full |

### 3. Search (highest priority)

| Module | Source | Documentation | Coverage |
|---|---|---|---|
| Search orchestration, candidate collection, late payload retrieval | `app/services/prioritized_search_service.py` | [Search Pipeline Overview](services/search/overview.md) | Full |
| Scoring: weighted vs RRF fusion, normalisation | `PrioritizedSearchService._rank_results` | [Ranking and Fusion](services/search/ranking_and_fusion.md) | Full |
| Title/summary boosts, filters, de-duplication, response items | `PrioritizedSearchService` | [Boosts, Filters and Results](services/search/boosts_filters_and_results.md) | Full |
| Query preprocessing | `app/utils/query_preprocessor.py` | [Search Pipeline Overview](services/search/overview.md), [Utilities](backend/utils.md) | Full |

### 4. Ingestion and Document Management

| Module | Source | Documentation | Coverage |
|---|---|---|---|
| Upload orchestration, embeddings, point construction | `app/services/document_operations/upload_service.py` | [Upload Pipeline](services/ingestion/pipeline.md) | Full |
| Facade | `app/services/document_processor.py` | [Upload Pipeline](services/ingestion/pipeline.md) | Full |
| File processors (PDF/OCR, DOCX, XLSX, CSV, text) | `app/services/file_processors/` | [File Processors](services/ingestion/file_processors.md) | Full |
| URL extraction | `app/services/url_text_extractor.py` | [Upload Pipeline](services/ingestion/pipeline.md) | Full |
| Update / upsert / delete / metadata patch | `app/services/document_operations/` | [Update, Delete and Metadata](services/ingestion/document_operations.md) | Full |
| Translation hook | `app/services/translation_service.py` | [Upload Pipeline](services/ingestion/pipeline.md) | Full — it is a pass-through stub |

### 5. Auxiliary Services

| Module | Source | Documentation | Coverage |
|---|---|---|---|
| Text embedding search | `app/services/text_embedding_search_service.py` | [Auxiliary Services](services/auxiliary_services.md) | Full |
| Similarity check | `app/services/similarity_service.py` | [Auxiliary Services](services/auxiliary_services.md) | Full |
| Source verification | `app/services/source_verification_service.py` | [Auxiliary Services](services/auxiliary_services.md) | Full |
| Multilingual query | `app/services/query_service.py`, `endpoints/query.py` | [Auxiliary Services](services/auxiliary_services.md) | Full — endpoint is currently broken |

### 6. Clients

| Module | Source | Documentation | Coverage |
|---|---|---|---|
| Qdrant client, collection and index setup, batch upload | `app/core/clients/qdrant.py` | [Clients](backend/clients.md), [Qdrant Data Model](architecture/qdrant_data_model.md) | Full |
| Dense embeddings | `app/core/clients/embedding.py` | [Clients](backend/clients.md) | Full |
| BM25 sparse encoder | `app/core/clients/sparse_encoder.py` | [Clients](backend/clients.md) | Full |
| Redis cache | `app/core/clients/redis_cache.py` | [Clients](backend/clients.md) | Full — unused by search |

### 7. Operations

| Module | Source | Documentation | Coverage |
|---|---|---|---|
| Deployment (Ansible) | `deployment/` | [Deployment](setup/deployment.md) | Full |
| Migration and insert scripts | `scripts/` | [Scripts](operations/scripts.md) | Full |
| Test suite | `tests/` | [Testing](operations/testing.md) | Full |
| Client/server version constraints | `app/core/clients/qdrant.py` | [Qdrant Compatibility](operations/qdrant_compatibility.md) | Full |
| Migration, backup/restore and re-indexing | `scripts/`, `app/core/clients/qdrant.py` | [Migration and Restore](operations/migration_and_restore.md) | Full — snapshot and copy procedures are Qdrant features, not repo tooling |
| Failure handling and recovery | all services | [Failure Modes and Recovery](operations/failure_modes_and_recovery.md) | Full |
| Incident procedures | — | [Troubleshooting](operations/troubleshooting.md) | Full |

## Known Issues Register

These were identified by reading the code while writing these pages; none has been fixed as part of the documentation work. Each is described in detail on the linked page. Review this list in the first KT session — several affect production behaviour.

| # | Area | Issue | Detail |
|---|---|---|---|
| 1 | Health | `/api/health` returns 503 whenever the Redis cache is disabled (the default): `redis_cache.redis_client` is never set, so the `ping()` call raises. | [App Lifecycle](backend/app_lifecycle.md) |
| 2 | Startup | `app/core/database.py` runs `Base.metadata.create_all` at import and is imported by every file processor and `query_service`, so PostgreSQL is a hard startup dependency despite being "unused". | [App Lifecycle](backend/app_lifecycle.md) |
| 3 | Query endpoint | `endpoints/query.py` returns `process_query(...)` without `await`; `/api/query/` does not work. Tests hide this by mocking the service. | [Auxiliary Services](services/auxiliary_services.md) |
| 4 | Verify sources | Filters on `metadata.source_id`, but `source_id` is stored at the payload root; every ID is reported `not_found`. | [Auxiliary Services](services/auxiliary_services.md) |
| 5 | Update | `PUT` and upsert drop `title`, `summary`, `tags`; invalid metadata silently becomes `{}`; delete runs before upload, so a failed upload loses the old document. | [Update, Delete and Metadata](services/ingestion/document_operations.md) |
| 6 | Metadata patch | The scroll has no offset and patched points still match the filter; a document with 100+ chunks can loop forever. The `metadata` vector is not re-embedded. | [Update, Delete and Metadata](services/ingestion/document_operations.md) |
| 7 | Delete | `company_id` is read from a form field; if omitted, documents with that `source_id` are deleted across all companies. | [Update, Delete and Metadata](services/ingestion/document_operations.md) |
| 8 | Swallowed errors | `check_documents_exist` / `count_documents` return `False` / `0` on Qdrant errors, which can cause duplicate creation on upsert. `upload_to_qdrant` never raises; failures return 201 with `upload_failures > 0`. | [Update, Delete and Metadata](services/ingestion/document_operations.md), [Clients](backend/clients.md) |
| 9 | Concurrency | Endpoints are `async def` but call blocking search, embedding and Qdrant code directly on the event loop; there is no thread offloading. | [System Architecture](architecture/system_architecture.md) |
| 10 | Search | A doc matching both title and summary receives both boosts; `top_k <= 0` raises a 500 rather than 422; a token-less BM25 query silently falls back to raw dense scores. | [Boosts, Filters and Results](services/search/boosts_filters_and_results.md), [Search Pipeline Overview](services/search/overview.md) |
| 11 | Filters | `metadata.type` holds processor labels (e.g. `xlsx_rag_optimized`), so a `file_type` filter of `xlsx` does not match XLSX documents. | [Qdrant Data Model](architecture/qdrant_data_model.md) |
| 12 | Processors | DOCX tables are not extracted; `.doc` cannot be opened; PDF/CSV intentional 400s surface as 500s. | [File Processors](services/ingestion/file_processors.md) |
| 13 | URL ingestion | No SSRF or response-size guard on the URL fetch. | [Upload Pipeline](services/ingestion/pipeline.md) |
| 14 | Config | Several settings are declared but unused (`MAX_SEARCH_TOP_K`, `DEFAULT_SEARCH_TOP_K`, `PDF_CHUNK_SIZE`, `PDF_CHUNK_OVERLAP`, `MIN_WEIGHTED_SCORE_THRESHOLD`, `METADATA_MATCH_BOOST`); `REDIS_PASSWORD` is never passed to the Redis client. | [Configuration Reference](setup/configuration.md) |
| 15 | Tests and release docs | no live compatibility test exists in `tests/`; the `compat` marker is unused; `mock_qdrant_client.search` targets a removed API. `release-doc/release-1.0.0.md` contradicts the QA setup (server ≥1.18 vs 1.12, candidate cap `top_k*20`/10000 vs 8/2000). | [Testing](operations/testing.md), [Qdrant Compatibility](operations/qdrant_compatibility.md) |
| 16 | Migration script | `scripts/migrate_to_sparse_vectors.py` ignores `.env` and `SPARSE_SEARCH_ENABLED`, and states a sparse field cannot be added in place, whereas `qdrant.py` does so via `update_collection`. | [Scripts](operations/scripts.md) |
| 17 | Deployment | `ansible.yml` ignores `uvicorn_port`/`uvicorn_workers` (template hardcodes 8000 and 4) and fetches secrets with `curl --insecure`. | [Deployment](setup/deployment.md) |

## Common Misconceptions

| Topic | Often assumed | Actual behaviour |
|---|---|---|
| API paths | `/api/v1/documents...` | `/api/documents...` — the router is included with prefix `/api` and no `/v1` |
| Search field weights | 0.36 / 0.27 / 0.14 / 0.14 / 0.09 | `SEARCH_PRIORITY_WEIGHTS` is 0.34 / 0.26 / 0.20 / 0.12 / 0.08 (the former values are the `DetailFilterScore` thresholds) |
| Title vs summary boost | Title takes precedence | Both boosts are applied; only the injection step excludes summary for title matches |
| PostgreSQL | Not used at runtime | Required at import (`create_all`) |
| File size limit | Not enforced | Enforced in the processors (surfaces as 500); not enforced on the URL path |
| DOCX | Tables extracted | Tables are not extracted |
| Redis flag | Env var has no effect | Hardcoded default, but pydantic-settings can still override it |
| `QDRANT_CHECK_COMPATIBILITY` | Hardcoded | Read from the environment |
| `prioritized_search_service.py` size | 1314 lines | About 1800 lines |

## Suggested KT Session Plan

| Session | Scope | Pages |
|---|---|---|
| 1 | System shape and data model | Architecture, Qdrant Data Model, Configuration |
| 2 | Ingestion | Upload Pipeline, File Processors, Update/Delete/Metadata |
| 3 | Search, part 1 | Search Pipeline Overview, Ranking and Fusion |
| 4 | Search, part 2 | Boosts/Filters/Results, Auxiliary Services |
| 5 | Operations | Qdrant Compatibility, Scripts, Testing, Deployment |
| 6 | Data lifecycle and failure handling | Migration and Restore, Failure Modes and Recovery, Troubleshooting |
| 7 | Known issues walkthrough | The register above |
