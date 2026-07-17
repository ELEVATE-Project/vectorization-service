# Vector Service — Developer Reference (`claude.md`)

> **Audience**: Developers new to this codebase who need to understand the system quickly.  
> **Last code review**: June 2026 | **Python**: 3.10+ | **Qdrant client**: ≥1.18.0, <2.0.0

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Architecture Diagram](#2-architecture-diagram)
3. [Directory Structure](#3-directory-structure)
4. [Key Dependencies](#4-key-dependencies)
5. [Environment Variables](#5-environment-variables)
6. [Qdrant Collection Structure](#6-qdrant-collection-structure)
7. [Request Flow — Ingestion Pipeline](#7-request-flow--ingestion-pipeline)
8. [Request Flow — Search Pipeline](#8-request-flow--search-pipeline)
9. [Embedding Generation](#9-embedding-generation)
10. [Hybrid Search Implementation](#10-hybrid-search-implementation)
11. [Reciprocal Rank Fusion (RRF)](#11-reciprocal-rank-fusion-rrf)
12. [Score Normalization and Final Ranking](#12-score-normalization-and-final-ranking)
13. [Field Boost (Re-ranking)](#13-field-boost-re-ranking)
14. [Metadata Filtering Strategy](#14-metadata-filtering-strategy)
15. [Query Preprocessing](#15-query-preprocessing)
16. [Configuration Management](#16-configuration-management)
17. [Module Responsibilities](#17-module-responsibilities)
18. [API Endpoints Reference](#18-api-endpoints-reference)
19. [Common Debugging Procedures](#19-common-debugging-procedures)
20. [Performance Considerations](#20-performance-considerations)
21. [Known Limitations and Improvement Opportunities](#21-known-limitations-and-improvement-opportunities)
22. [Inconsistencies and Observations](#22-inconsistencies-and-observations)

---

## 1. Project Overview

The **Vectorization Service** is a FastAPI application that provides:

- **Document ingestion**: Upload files (PDF, DOCX, XLSX, CSV, TXT) or URLs → extract text → chunk → generate multi-field embeddings → store in Qdrant.
- **Hybrid vector search**: Query Qdrant using both **dense embeddings** (cosine similarity across 5 named vector fields) and **sparse BM25 embeddings** (keyword relevance), fused with Reciprocal Rank Fusion (RRF).
- **Document management**: Update, upsert, delete, metadata-patch operations.
- **Utility endpoints**: Similarity check, source verification, simple text embedding search.

The service is stateless itself. All persistence lives in:
- **Qdrant** — vector store (dense + sparse named vectors, payload)
- **Redis** — LRU query cache (disabled by default; `REDIS_CACHE_ENABLED=false`)
- **PostgreSQL** — referenced in config (`DATABASE_URL`) but not actively used in application code today (SQLAlchemy model present in `app/models/db_models.py` but no ORM queries are made in any service)

---

## 2. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                   FastAPI Application                        │
│  root_path="" (local) | "/vector" (prod/staging)            │
│                                                             │
│  POST /api/v1/documents          → UploadService           │
│  PUT  /api/v1/documents/:id      → UpdateService           │
│  PUT  /api/v1/documents/:id/upsert → UpdateService         │
│  PATCH /api/v1/documents/:id/metadata → MetadataService    │
│  DELETE /api/v1/documents/:id    → DeleteService           │
│  POST /api/v1/documents/search   → PrioritizedSearchService│
│  POST /api/v1/documents/text-search → TextEmbeddingSearchService│
│  POST /api/v1/documents/check-similarity → SimilarityService│
│  POST /api/v1/documents/verify-sources → SourceVerificationService│
│  GET  /api/health                                          │
└────────────────────┬────────────────────────────────────────┘
                     │
         ┌───────────┴───────────────────────┐
         │                                   │
┌────────▼──────────┐             ┌──────────▼──────────┐
│   Qdrant Client   │             │  Embedding Model    │
│  (QdrantClient)   │             │  SentenceTransformer│
│  Host: QDRANT_HOST│             │  all-MiniLM-L6-v2   │
│  Port: QDRANT_PORT│             │  384-dim vectors    │
│  Collections:     │             └─────────────────────┘
│  - documents      │
│  - qa_cache       │             ┌─────────────────────┐
└───────────────────┘             │  Sparse Encoder     │
                                  │  fastembed BM25     │
                                  │  (Phase 2 only)     │
                                  └─────────────────────┘
```

---

## 3. Directory Structure

```
vectorization-service/
├── app/
│   ├── main.py                          # FastAPI app creation, lifespan, health check
│   ├── config.py                        # All settings (pydantic-settings + .env)
│   ├── api/
│   │   └── v1/
│   │       ├── api.py                   # APIRouter aggregation
│   │       └── endpoints/
│   │           ├── documents.py         # All document + search endpoints
│   │           ├── query.py             # (minimal, legacy)
│   │           └── cache.py             # Cache clear endpoint
│   ├── core/
│   │   ├── database.py                  # SQLAlchemy engine (not actively used)
│   │   └── clients/
│   │       ├── qdrant.py               # QdrantClient singleton, collection setup, batch upload
│   │       ├── embedding.py            # SentenceTransformer singleton, generate_embeddings()
│   │       ├── sparse_encoder.py       # BM25 fastembed encoder (lazy singleton)
│   │       └── redis_cache.py          # Redis LRU cache with TTL
│   ├── models/
│   │   ├── api_models.py              # All Pydantic request/response models
│   │   └── db_models.py               # SQLAlchemy model (unused in runtime)
│   ├── services/
│   │   ├── document_processor.py      # Orchestrator (thin facade over operation services)
│   │   ├── prioritized_search_service.py  # ★ Core search engine (1314 lines)
│   │   ├── similarity_service.py      # Duplicate content detection
│   │   ├── text_embedding_search_service.py  # Simple single-field text search
│   │   ├── source_verification_service.py    # Bulk source_id existence check
│   │   ├── translation_service.py     # Hindi detection + AI4Bharat translation
│   │   ├── url_text_extractor.py      # httpx + BeautifulSoup URL scraper
│   │   ├── document_operations/
│   │   │   ├── base_operation.py      # Shared: build_filter, validate, count, check
│   │   │   ├── upload_service.py      # File → chunks → embeddings → Qdrant
│   │   │   ├── update_service.py      # delete + re-upload
│   │   │   ├── delete_service.py      # Scroll + batch delete by source_id
│   │   │   └── metadata_service.py    # set_payload patch (no re-embed)
│   │   └── file_processors/
│   │       ├── base_processor.py      # Abstract base (validate_file_content)
│   │       ├── pdf_processor.py       # PyPDF2 + OCR fallback (pytesseract)
│   │       ├── docx_processor.py      # python-docx
│   │       ├── xlsx_processor.py      # openpyxl / pandas (markdown tables)
│   │       ├── csv_processor.py       # pandas
│   │       └── text_processor.py      # Plain text + markdown detection
│   └── utils/
│       ├── json_handler.py            # CustomJSONResponse (NaN/Inf safe serialization)
│       ├── language_utils.py          # translate_text(), detect_language() (Hindi)
│       └── query_preprocessor.py      # spaCy stop-word removal for long queries
├── tests/
│   ├── test_api.py                    # API integration tests
│   └── conftest.py                    # Test fixtures
├── scripts/                           # Migration and utility scripts
├── .env.sample                        # Annotated environment variable reference
├── requirements.txt                   # Python dependencies
└── start_mac.sh                       # Dev startup script (Qdrant + Redis + uvicorn)
```

---

## 4. Key Dependencies

| Package | Version Constraint | Purpose |
|---|---|---|
| `fastapi` | latest | Web framework |
| `uvicorn` | latest | ASGI server |
| `qdrant-client[fastembed]` | `>=1.18.0,<2.0.0` | Vector DB client + BM25 sparse encoder |
| `sentence-transformers` | latest | Dense embedding model |
| `spacy` | `>=3.7.0` | Query preprocessing (en_core_web_sm) |
| `langchain-text-splitters` | latest | `RecursiveCharacterTextSplitter` |
| `PyPDF2` | latest | PDF text extraction |
| `pytesseract` | latest | OCR fallback for image-based PDFs |
| `pdf2image` | latest | Convert PDF pages to images for OCR |
| `python-docx` | latest | DOCX processing |
| `openpyxl` / `pandas` | latest | XLSX/CSV processing |
| `redis` | latest | LRU query cache |
| `pydantic-settings` | latest | Config management |
| `httpx` | latest | Async URL fetching |
| `beautifulsoup4` / `lxml` | latest | HTML parsing for URL extraction |

> **Critical**: `qdrant-client>=1.18.0` is required. The `search()` and `search_batch()` APIs were removed in 1.14–1.16. All search call sites use `query_points()` / `query_batch_points()`. Do **not** downgrade below 1.18.0.

### Qdrant 1.18 client ↔ 1.12 server compatibility

This deployment runs **client 1.18 against server 1.12** — the server is pinned at
1.12 and cannot be upgraded, while the client is pinned at 1.18 for BM25 sparse
search. The minor-version gap exceeds Qdrant's "≤1" rule, so the client logs a
blanket `UserWarning: ... is incompatible with server version 1.12.0`. The warning is
**deliberately suppressed** via `check_compatibility=False` in
[app/core/clients/qdrant.py](app/core/clients/qdrant.py) (and the two utility scripts).

This is **safe**: every operation the service uses is supported by server 1.12 and has
been verified live (read path probed against the running server; write path confirmed
by 9,916 existing points carrying named-dense + `bm25` sparse vectors).

**Supported on server 1.12 (used by this service — all ✅):**

| Path | Operations |
|---|---|
| Read / search | `query_points`, `query_batch_points` (multi-field dense **+ BM25 sparse with `Modifier.IDF`**), `scroll` (`MatchText` PREFIX, `MatchAny`), `retrieve`, `get_collection` |
| Write / ingestion | `create_collection` (named dense `VectorParams` + `sparse_vectors_config` IDF), `update_collection`, `create_payload_index` (KEYWORD / TEXT / PREFIX `TextIndexParams`), `upsert` (named dense + `SparseVector`), `set_payload`, `delete_payload_index` |

**NOT supported on server 1.12 — do NOT introduce while the server stays on 1.12**
(each 400s with `"did not match any variant of untagged enum"`):

| Feature | Needs server | Breaks on |
|---|---|---|
| `MatchPhrase` / `phrase_matching` text queries | 1.15+ | search path |
| `FormulaQuery` (score boosting) | 1.14+ | search path |
| Post-1.12 `TextIndexParams` fields: `phrase_matching`, `stopwords`, `stemmer`, `ascii_folding`, `enable_hnsw` | 1.13+ | `create_payload_index` (e.g. extending `_PREFIX_TEXT_INDEX` in `qdrant.py`) |

A regression test (`tests/test_qdrant_compat.py`, marker `compat`) exercises the used
read path against the configured server so any drift that breaks it is caught early.

---

## 5. Environment Variables

All variables are read in `app/config.py` via `pydantic-settings` + `python-dotenv`.

### Qdrant
| Variable | Default | Description |
|---|---|---|
| `QDRANT_HOST` | `127.0.0.1` | Qdrant server host |
| `QDRANT_PORT` | `6333` | Qdrant server port |
| `COLLECTION_NAME` | `documents` | Main document collection |
| `QA_CACHE_COLLECTION` | `qa_cache` | Q&A cache collection (single vector) |

### Embedding & Chunking
| Variable | Default | Description |
|---|---|---|
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | SentenceTransformer model name |
| `CHUNK_SIZE` | `3000` | Characters per chunk (most processors) |
| `CHUNK_OVERLAP` | `500` | Overlap between chunks |
| `MARKDOWN_CHUNK_SIZE` | `3500` | Larger chunks for markdown/xlsx |
| `MARKDOWN_CHUNK_OVERLAP` | `800` | Markdown chunk overlap |
| `URL_EXTRACTION_CHUNK_SIZE` | `1500` | Chunks for URL-extracted content |
| `URL_EXTRACTION_CHUNK_OVERLAP` | `300` | URL chunk overlap |
| `PAGE_TEXT_THRESHOLD` | `20` | Min chars/page before OCR is triggered |
| `MAX_FILE_SIZE_MB` | `1024` | Maximum upload file size |

### Search
| Variable | Default | Description |
|---|---|---|
| `SIMILARITY_THRESHOLD` | `0.40` | Default score threshold for simple text search |
| `DEFAULT_SEARCH_TOP_K` | `10` | Default top-K (not used by search endpoint directly — request default is 1,000,000) |
| `MAX_SEARCH_TOP_K` | `100` | Maximum top-K (config only, not enforced in code) |
| `SHORT_QUERY_THRESHOLD` | `3` | Queries with fewer words skip spaCy preprocessing |

### Hybrid Search (Phase 1 — Dense + Boosts)
| Variable | Default | Description |
|---|---|---|
| `HYBRID_SEARCH_ENABLED` | `true` | Enables title/summary boosts after semantic search |
| `EXACT_TITLE_BOOST` | `2.5` | Multiplier for exact title match (capped at 1.0) |
| `PARTIAL_TITLE_BOOST` | `1.5` | Multiplier for partial title match (capped at 1.0) |
| `EXACT_SUMMARY_BOOST` | `1.4` | Multiplier for exact summary match |
| `PARTIAL_SUMMARY_BOOST` | `1.2` | Multiplier for partial summary match |
| `METADATA_MATCH_BOOST` | `1.2` | Reserved (not applied in current code) |

### Sparse BM25 Search (Phase 2)
| Variable | Default | Description |
|---|---|---|
| `SPARSE_SEARCH_ENABLED` | `false` | Enables BM25 sparse vector search |
| `SPARSE_VECTOR_NAME` | `bm25` | Field name in Qdrant for sparse vectors |
| `RRF_K` | `60` | Reciprocal Rank Fusion constant |
| `HYBRID_DENSE_WEIGHT` | `0.7` | Dense score weight in final fusion |
| `HYBRID_SPARSE_WEIGHT` | `0.3` | Sparse score weight in final fusion |
| `SEARCH_CANDIDATE_FANOUT` | `8` | Candidate pool = min(top_k × fanout, max) |
| `SEARCH_CANDIDATE_MAX` | `2000` | Hard cap on candidates per field |

### Infrastructure
| Variable | Default | Description |
|---|---|---|
| `REDIS_HOST` | `localhost` | Redis server host |
| `REDIS_PORT` | `6379` | Redis server port |
| `REDIS_PASSWORD` | `""` | Redis auth password |
| `REDIS_CACHE_TTL` | `86400` | Cache entry TTL (24 hours) |
| `REDIS_MAX_CACHE_SIZE` | `1000` | Max entries in LRU cache |
| `REDIS_CACHE_ENABLED` | `false` | Master switch for Redis cache (hardcoded false in config.py) |
| `POSTGRES_DATABASE_URI` | `postgresql://...` | Postgres URI (unused at runtime) |
| `ENVIRONMENT` | `local` | `local` → no `/vector` path prefix; others → `/vector` root path |

---

## 6. Qdrant Collection Structure

### `documents` collection (named vectors)

Created in `app/core/clients/qdrant.py :: ensure_collections_exist()`.

```
Collection: "documents"
├── Named Dense Vectors (384-dim, cosine distance):
│   ├── "text"     → embedding of the chunk text content
│   ├── "title"    → embedding of the document title (same for every chunk)
│   ├── "summary"  → embedding of the document summary (same for every chunk)
│   ├── "tags"     → embedding of comma-joined tags (same for every chunk)
│   └── "metadata" → embedding of "key: value" serialized metadata (same for every chunk)
│
├── Sparse Vector (Phase 2 only, requires SPARSE_SEARCH_ENABLED=true):
│   └── "bm25"     → BM25 sparse vector of chunk text (SparseVectorParams + Modifier.IDF)
│
└── Payload (per point):
    ├── text       (str)   → chunk content
    ├── source_id  (str)   → unique document identifier
    ├── title      (str|None) → document title
    ├── summary    (str|None) → document summary
    ├── tags       (list[str]|None) → document tags
    └── metadata   (dict)  → arbitrary metadata including:
        ├── company    (str) → maps to organizations filter
        ├── DOCUMENT_TYPE (str) → maps to resource_type filter (comma-separated)
        ├── type       (str) → maps to file_type filter (e.g. "pdf", "docx")
        ├── source     (str) → original filename
        ├── priority   (str) → P1/P2/P3 etc.
        ├── created_at (ISO datetime str)
        ├── updated_at (ISO datetime str)
        └── [other user-supplied fields]
```

### Payload Indexes (created at startup)

Defined in `_PAYLOAD_INDEXES` in `qdrant.py`:

| Field | Index Type | Purpose |
|---|---|---|
| `source_id` | KEYWORD | Fast exact match for delete, dedup, verification |
| `metadata.company` | KEYWORD | Organization filter (MatchAny) |
| `tags` | KEYWORD | Category filter (MatchAny) |
| `metadata.DOCUMENT_TYPE` | TEXT | Resource type filter (MatchText) |
| `title` | TEXT (PREFIX tokenizer) | Title boost scroll + search |
| `summary` | TEXT (PREFIX tokenizer) | Summary boost scroll + search |

The PREFIX tokenizer (min_token_len=2, max_token_len=20, lowercase=True) indexes every prefix of each token, enabling partial/typed-mid queries.

### `qa_cache` collection (single vector)

Uses `single_vector_config` (same 384-dim cosine). Used for Q&A caching — not actively written to in the current codebase (legacy design).

---

## 7. Request Flow — Ingestion Pipeline

```
POST /api/v1/documents (multipart/form-data)
  │
  ├── parse_metadata_form() [dependency injection]
  ├── parse_tags_form()     [dependency injection]
  │
  └── DocumentProcessor.process_upload()
        └── UploadService.process()
              ├── validate_source_id()      → 400 if empty
              ├── validate_priority()       → 400 if not "P*" format
              ├── ensure_collections()      → creates collections on first run
              │
              ├── IF metadata.markdown_url present:
              │     URLTextExtractor.extract_text(url)
              │       ├── httpx.AsyncClient.get(url, timeout=30s)
              │       ├── BeautifulSoup HTML parsing (lxml)
              │       └── RecursiveCharacterTextSplitter(
              │               chunk_size=1500, chunk_overlap=300)
              │
              └── ELSE (normal file processing):
                    _process_file_by_type(file_extension)
                      └── Processor.process(file_content, filename, priority)
                            │
                            ├── PDFProcessor:
                            │     ├── PyPDF2.PdfReader → page_text per page
                            │     ├── IF page_text < 20 chars → pytesseract OCR
                            │     │     (pdf2image → PIL → tesseract)
                            │     └── RecursiveCharacterTextSplitter(3000, 500)
                            │
                            ├── DOCXProcessor: python-docx → paragraphs + tables
                            ├── XLSXProcessor: openpyxl → markdown table format
                            ├── CSVProcessor:  pandas → markdown rows
                            └── TextProcessor: plain text + markdown detection
                                   → RecursiveCharacterTextSplitter(3000|3500, 500|800)
              │
              │   Each chunk has: {id, text, metadata}
              │   translation_service.process_chunk() → detect Hindi, translate if needed
              │
              └── _upload_chunks()
                    ├── generate_embeddings([chunk.text for chunks])  ← batch
                    ├── _generate_field_embeddings(title, summary, tags, metadata)
                    │     → 1 embedding each for title, summary, tags_text, metadata_kv
                    │
                    ├── IF SPARSE_SEARCH_ENABLED:
                    │     generate_sparse_vector(chunk.text) for each chunk
                    │     → fastembed BM25 → (indices, values)
                    │
                    ├── Build PointStruct per chunk:
                    │     vector={
                    │       "text": text_embedding,      ← unique per chunk
                    │       "title": title_embedding,    ← same for all chunks
                    │       "summary": summary_embedding,← same for all chunks
                    │       "tags": tags_embedding,      ← same for all chunks
                    │       "metadata": metadata_embedding,← same for all chunks
                    │       "bm25": SparseVector(...)    ← unique per chunk (Phase 2)
                    │     }
                    │     payload={text, source_id, title, summary, tags, metadata}
                    │
                    └── upload_to_qdrant(points, batch_size=100)
                          → qdrant_client.upsert() in 100-point batches
```

**Key design decision**: The title/summary/tags/metadata embeddings are **document-level** (identical across all chunks of a document), while the `text` embedding is **chunk-level** (unique per chunk). This allows searching the same query against multiple representation granularities simultaneously.

---

## 8. Request Flow — Search Pipeline

```
POST /api/v1/documents/search (JSON body: PrioritizedSearchRequest)
  │
  └── PrioritizedSearchService.search(request)
        │
        ├── IF no query → _get_unique_source_documents() [scroll all, dedup]
        │
        └── WITH query:
              ├── preprocess_query(query)  [spaCy: stop-word removal, lowercasing]
              │     → short queries (< 3 words OR < 20 chars): skip spaCy, just lowercase
              │     → long queries: remove pronouns, stop words, punctuation
              │
              ├── generate_embeddings([preprocessed_query])  → query_embedding (384-dim)
              │
              ├── _build_filters(categories, organizations, resource_types, file_types)
              │     → models.Filter(must=[...]) with AND between types, OR within type
              │
              ├── _candidate_limit(top_k)
              │     → min(top_k * 8, 2000) — bounded HNSW traversal depth
              │
              ├── IF SPARSE_SEARCH_ENABLED:
              │     _hybrid_batch_search(fields, query_text, query_embedding, filter, limit)
              │       → [see Section 10 — Hybrid Search]
              │
              └── ELSE:
                    _parallel_batch_search(fields, weights, query_embedding, filter, limit)
                      → qdrant_client.query_batch_points() across 5 fields
                      → merge by point_id, collect per-field scores
              │
              ├── _process_and_filter_results()
              │     ├── _rank_results()         → compute weighted_score per doc
              │     ├── filter by filter_score OR detail_filter_score
              │     ├── _filter_best_per_source() → deduplicate by source_id
              │     └── top_results[:top_k]
              │
              ├── IF HYBRID_SEARCH_ENABLED AND search_mode != "semantic":
              │     ├── _get_field_match_sources(query, filter, "title")
              │     │     → Qdrant scroll with MatchText(title) → classify exact/partial
              │     ├── _supplement_matches_from_results() → catch infix hits in candidates
              │     ├── _apply_field_boost(top_results, title_matches, EXACT=2.5, PARTIAL=1.5)
              │     │     → weighted_score × multiplier, capped at 1.0
              │     ├── [same for summary field]
              │     ├── _fetch_field_match_docs(missing_title_ids, ...)
              │     │     → inject FLOOR_SCORE(0.15) × boost docs absent from semantic results
              │     └── resort + re-cap top_k
              │
              ├── IF SPARSE_SEARCH_ENABLED: [Late Payload Retrieval]
              │     qdrant_client.retrieve(final top_k IDs, with_payload=True)
              │     → replace partial metadata payloads with full payloads incl. "text"
              │
              └── _build_result_items(top_results)
                    → SearchResultItem(id, text, title, summary, tags, metadata,
                                       source_id, score, field_scores,
                                       title_match, summary_match)
```

---

## 9. Embedding Generation

**Model**: `SentenceTransformer("all-MiniLM-L6-v2")` — 384-dimensional dense vectors.

Loaded as a **module-level singleton** in `app/core/clients/embedding.py`:

```python
embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)

def generate_embeddings(texts: list):    # batch encode
    return embedding_model.encode(texts)

def generate_single_embedding(text: str):  # single encode
    return embedding_model.encode(text)
```

### At Ingestion

| Vector Field | Source Text | Per-chunk or Per-doc? |
|---|---|---|
| `text` | `chunk.text` | Per-chunk (unique) |
| `title` | `title` parameter | Per-doc (same across all chunks) |
| `summary` | `summary` parameter | Per-doc (same across all chunks) |
| `tags` | `", ".join(tags)` | Per-doc (same across all chunks) |
| `metadata` | `"k: v " joined key-values` | Per-doc (same across all chunks) |

### At Search Time

The query string (after preprocessing) is embedded once: `generate_embeddings([preprocessed_query])[0]`.

This **single embedding is reused** across all 5 field searches in the batch query — the model is not called separately per field.

### BM25 Sparse Vector (Phase 2)

Module: `app/core/clients/sparse_encoder.py`

- Uses `fastembed.SparseTextEmbedding(model_name="Qdrant/bm25")`
- Lazy singleton loaded on first call
- Returns `(indices: list[int], values: list[float])` — a sparse representation of token weights
- At search time: `generate_sparse_vector(query_text)` → builds `SparseVector(indices, values)`

---

## 10. Hybrid Search Implementation

The term "hybrid" has **two distinct meanings** in this codebase:

### Phase 1 Hybrid (always on when `HYBRID_SEARCH_ENABLED=true`)
- Dense multi-field semantic search + **post-hoc keyword boosts** on title/summary fields
- Does NOT use sparse vectors
- Implemented in: the boost section of `PrioritizedSearchService.search()`

### Phase 2 Hybrid (when `SPARSE_SEARCH_ENABLED=true`)
- Dense 5-field search + **BM25 sparse vector search** → **client-side dense+sparse fusion**
  (min-max **weighted** by default, or **RRF** — selected by `HYBRID_FUSION_METHOD`, see §12)
- Candidate collection in `_hybrid_batch_search()`; the actual fusion/ranking in `_rank_results()`

### `_hybrid_batch_search()` Step by Step

```python
# 1. Build dense QueryRequests for all 5 fields
for field in ["title", "text", "tags", "summary", "metadata"]:
    QueryRequest(query=query_embedding.tolist(), using=field,
                 limit=candidate_limit, with_payload=metadata_fields_only)

# 2. Build BM25 sparse QueryRequest
sparse_indices, sparse_values = generate_sparse_vector(query_text)
QueryRequest(query=SparseVector(indices, values), using="bm25",
             limit=candidate_limit, with_payload=metadata_fields_only)

# 3. Execute all 6 requests in ONE batch call
batch_results = qdrant_client.query_batch_points(requests=all_requests)

# 4. Collect raw per-field scores (dense cosines + raw BM25 under the sparse key)

# 5. Late payload retrieval for top-K results (to get 'text' field)
```

**Payload Projection**: During candidate collection, only `["source_id", "title", "summary", "tags", "metadata"]` are fetched — NOT `"text"`. The heavy text field is deferred to a late `qdrant_client.retrieve()` call for only the final top-K results. This significantly reduces network bandwidth when `SEARCH_CANDIDATE_MAX=2000`.

---

## 11. Reciprocal Rank Fusion (RRF)

### Formula

```
RRF_Score(doc) = Σ over all result lists:  1 / (k + rank)
```

Where:
- `k` = `settings.RRF_K` (default: **60**) — constant to dampen high-rank differences
- `rank` = 1-indexed position in that list's results (Qdrant returns results sorted descending by score)

### Where RRF lives

There is exactly **one** RRF in the codebase: the ranking RRF inside `_rank_results()`, used only
when `HYBRID_FUSION_METHOD=rrf` (Section 12). It fuses **two** lists — the single combined dense list
(the 5 dense fields collapsed into one priority-weighted cosine sum, `raw_dense`) and the sparse list
(`raw_sparse`, the raw BM25 score) — as `1/(RRF_K + dense_rank) + 1/(RRF_K + sparse_rank)`, then
min-max normalizes. It does **not** RRF the 5 dense fields separately.

> **Historical note**: `_hybrid_batch_search()` previously also computed a *second*, all-field RRF
> (5 dense fields + sparse, 6 lists) and stored it under `field_scores[pid]["rrf"]`. That value was
> never consumed by ranking or filtering and has been **removed**. Hybrid mode is now detected by the
> presence of the raw BM25 score under the sparse field key (`field_scores[pid]["bm25"]`), which is
> injected only when the BM25 query actually returned hits — so an empty/skipped sparse query no
> longer spuriously activates hybrid scoring (which would have deflated dense scores by the 0.7 dense
> weight). The retired per-field `"rrf"` key is no longer produced.

### Why k=60?

k=60 is the standard empirical default from the original RRF paper (Cormack et al.). It dampens the advantage of being ranked #1 vs. #10 — ensuring that consistent mid-rank presence across multiple lists outweighs a single #1 ranking.

---

## 12. Score Normalization and Final Ranking

`_rank_results()` computes the final `weighted_score` for every candidate:

### Dense-Only Path (`SPARSE_SEARCH_ENABLED=false`)

```
weighted_score = Σ (field_weight × field_score)
               = 0.36×title_score + 0.27×text_score + 0.14×tags_score
                 + 0.14×summary_score + 0.09×metadata_score
```

Score is a raw weighted cosine similarity, range approximately [0, 1].

### Hybrid Path (`SPARSE_SEARCH_ENABLED=true`)

The dense component is **always** the weighted multi-field cosine sum (same formula as dense-only). How that dense score is fused with the sparse (BM25) score is selected by **`HYBRID_FUSION_METHOD`** (`weighted` | `rrf`, default `weighted`). Both modes produce a calibrated 0–1 `weighted_score` comparable to `filter_score`.

Shared first step (both modes):

```
raw_dense[pid]  = Σ (field_weight × field_score)        ← same as dense-only, UNCHANGED
raw_sparse[pid] = field_scores[pid].get("bm25", 0.0)    ← raw BM25 score from Qdrant
```

**Mode A — `weighted` (default).** Min-max normalize each modality separately to [0, 1], then fuse:

```
norm_dense  = (raw_dense  - min_dense)  / (max_dense  - min_dense)
norm_sparse = (raw_sparse - min_sparse) / (max_sparse - min_sparse)
weighted_score = HYBRID_DENSE_WEIGHT × norm_dense + HYBRID_SPARSE_WEIGHT × norm_sparse   # 0.7 / 0.3
```

**Mode B — `rrf`.** Rank-fuse the **combined dense list** (ranked by `raw_dense`) against the **sparse list** (ranked by `raw_sparse`, docs with a sparse hit only), then min-max normalize:

```
dense_rank[pid]  = position of pid in raw_dense  sorted desc   # 1-indexed
sparse_rank[pid] = position of pid in raw_sparse sorted desc   # 1-indexed, sparse hits only
rrf[pid] = 1/(RRF_K + dense_rank[pid]) + 1/(RRF_K + sparse_rank[pid])
weighted_score = minmax(rrf)
```

Mode B fuses by *rank position* rather than raw score (robust across retrievers that produce differently-scaled scores). It ranks the **single combined dense list**, not the 5 dense fields separately, so the dense weighting is preserved. The min-max step is what keeps RRF (raw ~0–0.03) comparable to `filter_score`.

**Edge case** (both modes): When all scores in a pool are equal (single candidate or flat pool), the range is 0 → normalized value is 1.0 if score > 0, else 0.0.

Implementation: `_rank_results()` (helpers `_min_max_normalize()`, `_rank_positions()`) in `prioritized_search_service.py`.

### Observing the fusion (response fields)

- `response.search_config.fusion_method` always reports the active method (`weighted` | `rrf`).
- Set `include_scoring_debug: true` on the request to surface the per-result breakdown on each
  `SearchResultItem`: `keyword_score` (raw BM25), `rrf_score` (raw fused value pre-normalization),
  `dense_rank`, and `sparse_rank` (`null` if the doc had no sparse hit). Off by default (responses
  stay lean given the large default `top_k`); the diagnostics are stashed internally by
  `_rank_results()` and only serialized when the flag is set (see `_build_result_items()`). The final
  `score` is always the calibrated 0–1 fused value regardless of the flag.

---

## 13. Field Boost (Re-ranking)

Applied **after** semantic ranking when `HYBRID_SEARCH_ENABLED=true` and `search_mode != "semantic"`.

### Two-Phase Approach

**Phase A — Boost existing results**:
1. `_get_field_match_sources(query, filter_conditions, "title")` — scrolls Qdrant with `MatchText(title, query)` using the PREFIX-tokenized index; classifies each match as `exact` (field == query) or `partial` (substring)
2. `_supplement_matches_from_results()` — catches infix hits in already-retrieved candidates (no extra Qdrant call)
3. `_apply_field_boost()` — multiplies `weighted_score` by `exact_boost` (2.5) or `partial_boost` (1.5), capped at 1.0

**Phase B — Inject missing results**:
- Documents that matched on title/summary but scored too low semantically (e.g., very short acronym queries like "SMC") are fetched via a single `qdrant_client.scroll(MatchAny(source_ids))` call
- Assigned a floor score: `FLOOR_SCORE (0.15) × boost_multiplier`, capped at 1.0
- Injected into results with `field_scores = {field: None for each field}` (None = not scored, distinguishable from 0.0)

### Summary Boost

Applied after title boost with lower multipliers:
- Exact match: ×1.4, Partial match: ×1.2

Title takes precedence — a doc that matched both title AND summary only gets the title boost applied.

---

## 14. Metadata Filtering Strategy

### Filter Construction (`_build_filters()`)

**Between filter types**: AND logic (`models.Filter(must=[...])`)  
**Within a filter type**: OR logic (`models.MatchAny`)

| API Parameter | Qdrant Field | Match Type |
|---|---|---|
| `categories` | `tags` | `MatchAny(any=[...])` — keyword index |
| `organizations` | `metadata.company` | `MatchAny(any=[...])` — keyword index |
| `resource_type` | `metadata.DOCUMENT_TYPE` | `MatchText(text=rt)` per value — text index |
| `file_type` | `metadata.type` | `MatchAny(any=[...])` — keyword index |

`resource_type` uses `MatchText` (not `MatchAny`) because `DOCUMENT_TYPE` is a comma-separated string field, so substring matching is needed.

### Score-Based Filtering

Two filtering modes selectable per request:

**Mode 1: `filter_score`** (simple threshold)
```
Pass if: weighted_score >= filter_score
```

**Mode 2: `detail_filter_score`** (field-level OR logic)
```
Pass if: ANY of:
  title_score  >= detail_filter_score.title  (default 0.36)
  text_score   >= detail_filter_score.text   (default 0.27)
  tags_score   >= detail_filter_score.tags   (default 0.14)
  summary_score >= detail_filter_score.summary (default 0.14)
  metadata_score >= detail_filter_score.metadata (default 0.09)
```

This is pure per-field OR logic on cosine scores, identical to the dense-only reference
implementation — there is no sparse/RRF fallback. In hybrid mode the dense fields still run, so
documents carry per-field cosine scores and are filtered the same way; the internal sparse/BM25 score
key is not a scorable field and is ignored here.

When `detail_filter_score` is provided, `filter_score` is **ignored** entirely.

---

## 15. Query Preprocessing

Module: `app/utils/query_preprocessor.py`

Uses `spacy.load("en_core_web_sm")` (singleton pattern).

### Logic Flow

```
query → strip whitespace
  │
  ├── IF word_count < SHORT_QUERY_THRESHOLD (3) OR len < 20:
  │     return query.lower()   ← bypass spaCy entirely
  │
  └── ELSE:
        doc = nlp(query)
        for token in doc:
          skip if: token.pos_ == "PRON"   (I, my, you, their...)
          skip if: token.is_stop           (the, a, is, are...)
          skip if: token.is_punct
          skip if: token.is_space
          keep: token.text.lower()
        preprocessed = " ".join(kept_tokens)
        IF preprocessed is empty: return original query lowercased
        return preprocessed
```

**Note**: Lemmatization is described in the module docstring but **not actually applied** — the code keeps `token.text` (original form), not `token.lemma_`. This is an inconsistency between the docstring and implementation.

The preprocessed query is used for **embedding generation only**. The original query is used for title/summary text matching.

---

## 16. Configuration Management

All settings live in `app/config.py` as a single `Settings(BaseSettings)` class.

```python
settings = Settings()  # singleton at module level
```

Settings are read from:
1. `.env` file (via `load_dotenv()`)
2. Environment variables (pydantic-settings auto-reads these)

**`TOKENIZERS_PARALLELISM=false`** is set programmatically at module import to suppress HuggingFace warnings when using pdf2image.

**`REDIS_CACHE_ENABLED`** is hardcoded to `False` in `config.py` regardless of the environment variable. The env var `REDIS_CACHE_ENABLED` has no effect unless you change the code:
```python
REDIS_CACHE_ENABLED: bool = False  # ← hardcoded, not from env
```

---

## 17. Module Responsibilities

| Module | Responsibility | Called By |
|---|---|---|
| `main.py` | App factory, lifespan (collection init), health check | ASGI server |
| `config.py` | All settings, read from .env | Every module |
| `qdrant.py` | Qdrant client singleton, collection setup, payload indexes, batch upload | `upload_service`, `search`, `delete`, etc. |
| `embedding.py` | SentenceTransformer singleton, `generate_embeddings()` | `upload_service`, `prioritized_search_service`, `similarity_service`, `text_embedding_search_service` |
| `sparse_encoder.py` | BM25 fastembed encoder, `generate_sparse_vector()` | `_hybrid_batch_search()`, `_upload_chunks()` |
| `redis_cache.py` | LRU cache with MD5-keyed storage (disabled by default) | Not called in current search flow |
| `document_processor.py` | Thin facade for upload/update/delete/metadata | `documents.py` endpoint |
| `upload_service.py` | File → chunks → embeddings → points → Qdrant | `document_processor.py` |
| `update_service.py` | Delete existing + re-upload (for PUT) | `document_processor.py` |
| `delete_service.py` | Batch scroll + delete by source_id | `document_processor.py`, `update_service.py` |
| `metadata_service.py` | Patch metadata via `set_payload` (no re-embed) | `document_processor.py` |
| `prioritized_search_service.py` | ★ Core search: query preprocessing → embedding → batch search → RRF → rank → boost → dedup | `documents.py` endpoint |
| `similarity_service.py` | Duplicate detection using text field cosine similarity | `documents.py` endpoint |
| `text_embedding_search_service.py` | Simple single-field text vector search | `documents.py` endpoint |
| `source_verification_service.py` | Bulk source_id existence check (N serial Qdrant queries) | `documents.py` endpoint |
| `url_text_extractor.py` | Async URL fetch + HTML extraction (httpx + BeautifulSoup) | `upload_service.py` |
| `query_preprocessor.py` | spaCy stop-word removal for long queries | `prioritized_search_service.py` |
| `language_utils.py` | Hindi character detection + AI4Bharat translation API | `translation_service.py` |
| `json_handler.py` | `CustomJSONResponse` that handles NaN/Infinity safely | `main.py` (default response class) |
| `pdf_processor.py` | PyPDF2 + per-page OCR fallback + chunking | `upload_service.py` |
| `docx_processor.py` | python-docx → text + tables + chunking | `upload_service.py` |
| `xlsx_processor.py` | openpyxl → markdown table format + chunking | `upload_service.py` |
| `csv_processor.py` | pandas → markdown rows + chunking | `upload_service.py` |
| `text_processor.py` | Plain text / markdown + chunking | `upload_service.py` |

---

## 18. API Endpoints Reference

Base URL: `http://host:port/api` (local) or `http://host:port/vector/api` (non-local environments)

| Method | Path | Service | Description |
|---|---|---|---|
| `POST` | `/v1/documents` | UploadService | Upload document file or URL |
| `PUT` | `/v1/documents/{source_id}` | UpdateService | Full replace (delete + re-upload) |
| `PUT` | `/v1/documents/{source_id}/upsert` | UpdateService | Create if not exists, replace if exists |
| `PATCH` | `/v1/documents/{source_id}/metadata` | MetadataService | Update metadata only (no re-embed) |
| `DELETE` | `/v1/documents/{source_id}` | DeleteService | Delete all chunks by source_id |
| `POST` | `/v1/documents/search` | PrioritizedSearchService | ★ Primary search endpoint |
| `POST` | `/v1/documents/text-search` | TextEmbeddingSearchService | Simple text vector search |
| `POST` | `/v1/documents/check-similarity` | SimilarityService | Duplicate content detection |
| `POST` | `/v1/documents/verify-sources` | SourceVerificationService | Check source_id existence |
| `GET` | `/health` | inline | Qdrant + Redis health check |

---

## 19. Common Debugging Procedures

### Search returns no results

1. Check logs for `SEARCH REQUEST` and `SEARCH RESULTS` sections
2. Verify `filter_score` — the default is `0.0` (from `MIN_SEARCH_FILTER_SCORE`), so results should not be filtered unless explicitly set
3. If using `detail_filter_score`, check that at least one field threshold is achievable — the defaults (0.36, 0.27, 0.14, 0.14, 0.09) can be high for sparse or ambiguous queries
4. Check `SEARCH_CANDIDATE_MAX=2000` — increasing this surfaces more results at the cost of latency
5. Confirm collection exists: `GET /health` → qdrant status
6. Confirm the document was indexed: `POST /documents/verify-sources`

### Sparse search not working

1. Check `SPARSE_SEARCH_ENABLED=true` in `.env`
2. Verify `qdrant-client>=1.18.0`: `pip show qdrant-client`
3. Verify fastembed installed: `python -c "from fastembed import SparseTextEmbedding"`
4. Collection must have the sparse vector field — if the collection pre-dates enabling sparse, run the migration script in `scripts/`
5. Check logs for `Sparse BM25 encoder initialised` on startup

### OCR not working for image PDFs

1. Verify system deps: `tesseract --version` and `pdftoppm -v`
2. Verify Python packages: `pip show pytesseract pdf2image Pillow`
3. Check `PAGE_TEXT_THRESHOLD=20` — pages with fewer than 20 chars trigger OCR

### Query preprocessing stripping important terms

Short queries (< 3 words or < 20 chars) bypass spaCy automatically. For longer queries, check if critical terms are being classified as stop words by spaCy. You can:
- Lower `SHORT_QUERY_THRESHOLD` in `.env`
- Or the search_mode can be set to `"semantic"` in the request to skip boosts entirely

### Title/summary boost not firing

1. Verify `HYBRID_SEARCH_ENABLED=true`
2. Check the request does not have `search_mode="semantic"`
3. Check logs for `title match sources found: N` — if N=0, the query didn't match any titles in the PREFIX index
4. The PREFIX tokenizer matches from start of tokens. Infix-only queries (e.g., "sur" in "insurance") may not match via the index scroll — but `_supplement_matches_from_results()` checks dense candidates in-memory

### High latency

1. Check `SEARCH_CANDIDATE_MAX` — reducing from 2000 to 500 gives ~2x speedup
2. Check number of fields: all 5 dense + 1 sparse = 6 parallel queries per search
3. Check if `_get_field_match_sources()` is slow — it runs scroll queries for title and summary; the PREFIX index should make these fast
4. Profile: look for `TIMING:` log lines which report `query_batch_points` and `late retrieve` durations

---

## 20. Performance Considerations

### Candidate Pool Sizing

The `_candidate_limit()` method controls HNSW traversal depth:
```
candidate_limit = min(top_k × SEARCH_CANDIDATE_FANOUT, SEARCH_CANDIDATE_MAX)
                = min(top_k × 8, 2000)
```

Benchmarked trade-offs for query "Classroom Observation":
- `SEARCH_CANDIDATE_MAX=500` → ~0.9s, count=21
- `SEARCH_CANDIDATE_MAX=2000` → ~1.5s, count=55
- `SEARCH_CANDIDATE_MAX=10000` → ~2.7s, count=151

Higher `SEARCH_CANDIDATE_MAX` reveals more results (min-max normalization across larger pool) but costs more latency.

### Batch Queries

`qdrant_client.query_batch_points()` sends all 5 (or 6) field queries in a **single network round trip**. This is the dominant performance optimization for multi-field search.

### Late Payload Retrieval

When `SPARSE_SEARCH_ENABLED=true`, the heavy `text` payload field is excluded from the candidate collection phase. Only the final top-K results have their full payload fetched via a single `qdrant_client.retrieve()` call. This can reduce candidate-phase network transfer by ~70% for large collections.

### Embedding Model

`all-MiniLM-L6-v2` is compact (22MB, 384 dims) and fast. The model is loaded once at import time. For embedding generation during ingestion, `generate_embeddings()` does a **batch encode** of all chunks simultaneously, which is much faster than individual calls.

### Redis Cache

The Redis LRU cache is disabled (`REDIS_CACHE_ENABLED=False` hardcoded). When enabled, it caches query responses keyed by MD5(query_text) with a 24h TTL. This would benefit repeat queries significantly.

### Source Verification N+1

`SourceVerificationService.verify_sources()` issues **one Qdrant scroll per source_id** in a serial loop. For large batches of source IDs, this is an O(N) Qdrant round trip pattern. A single `MatchAny(source_ids)` query would be far more efficient.

---

## 21. Known Limitations and Improvement Opportunities

### Limitations

1. **No query caching active**: Redis cache is disabled. Every search hits Qdrant and the embedding model.

2. **Phase 2 sparse requires collection re-index**: Existing collections must run the migration script to add BM25 vectors to historical chunks.

3. **Title/summary boosts run post-semantic search**: If semantic search returns 0 results (very short queries with high filter_score), the boost and injection steps may not recover relevant documents.

4. **Source verification is O(N) round trips**: `SourceVerificationService` issues one Qdrant query per source_id. Should use a single `MatchAny` query.

5. **No text re-embedding on metadata update**: `MetadataService.update_metadata()` only patches the payload. If `metadata` changes, the `metadata` vector becomes stale.

6. **PostgreSQL is not used**: `DATABASE_URL` is configured but never queried. `db_models.py` has a SQLAlchemy model but no runtime usage.

7. **`REDIS_CACHE_ENABLED` is hardcoded**: The environment variable has no effect; must change Python code.

8. **`MAX_SEARCH_TOP_K=100` is not enforced**: The config exists but `top_k` is never clamped in `search()`.

9. **Lemmatization claimed but not implemented**: The `query_preprocessor.py` docstring mentions lemmatization but the code uses `token.text` (original form), not `token.lemma_`.

10. **No rate limiting**: No rate limiting on any endpoint — susceptible to abuse or unintentional DoS via large file uploads or high-frequency search requests.

11. **File size not validated**: Despite `MAX_FILE_SIZE_MB=1024` config, no enforcement exists in the upload endpoint or any processor.

### Improvement Opportunities

1. **Enable Redis caching** for frequently repeated search queries
2. **Batch source verification**: Replace serial per-ID queries with a single `MatchAny` filter
3. **Re-embed metadata on metadata update**: If metadata changes, regenerate and update the `metadata` named vector
4. **Add `top_k` cap enforcement** using `MAX_SEARCH_TOP_K`
5. **Lemmatization**: Actually implement token lemmatization if the docstring intention is desired
6. **Parallel URL extraction**: Currently sequential when multiple chunks need processing
7. **Observability**: Add structured latency metrics (timing logs exist but not exported to any monitoring system)
8. **File size validation**: Enforce `MAX_FILE_SIZE_MB` at the endpoint boundary
9. **Rate limiting**: Add FastAPI middleware for rate limiting on upload and search endpoints
10. **Re-ranking with a cross-encoder**: The architecture is prepared for it (field_scores surfaced in response), but no cross-encoder re-ranking is implemented

---

## 22. Inconsistencies and Observations

| Location | Observation |
|---|---|
| `query_preprocessor.py` | Docstring claims lemmatization; code uses `token.text` not `token.lemma_` |
| `config.py` L41 | `REDIS_CACHE_ENABLED: bool = False` — hardcoded; the env var does nothing |
| `source_verification_service.py` L51 | Queries `metadata.source_id` (nested) but ingestion stores at top-level `source_id` — this is a **bug**: verifications will always return not_found |
| `config.py` L63-64 | `MAX_SEARCH_TOP_K=100` is declared but never applied in search code; requests with `top_k=1000000` (the default) are served as-is |
| `.env.sample` L88-89 | Has trailing comments `// 10` and `// 10000` after values — these are invalid `.env` syntax and will cause parse errors in strict parsers |
| `PrioritizedSearchRequest` | Default `top_k=1000000` — effectively disables pagination by default |
| `documents.py` L223 | Endpoint docstring says filter types use OR logic between categories and organizations — actually they use AND logic (`must` conditions) |
| `update_service.py` | `UpdateService` doesn't forward `title`, `summary`, or `tags` to `UploadService.process()` — these fields are lost on document update |
| `metadata_service.py` | Prevents changing `company_id` via metadata but doesn't prevent changing `source_id` |
| `delete_service.py` | Stops after first batch where `len(points) < batch_size` — but this may break if Qdrant returns partial pages during deletion due to concurrent modifications |
