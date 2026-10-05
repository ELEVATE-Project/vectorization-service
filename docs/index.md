# Vectorization Service

## What This System Does

The Vectorization Service is a FastAPI application that turns documents into searchable vectors and serves hybrid semantic search over them. It is stateless: all persistent data lives in Qdrant.

It provides:

### 1. Document Ingestion
- Upload files (PDF, DOCX, XLSX, CSV, TXT/Markdown) or extract a web page by URL
- Extract text (with OCR fallback for image-only PDF pages), chunk it, and embed it
- Store each chunk as a Qdrant point with five named dense vectors (`text`, `title`, `summary`, `tags`, `metadata`) and an optional BM25 sparse vector

### 2. Hybrid Search
- Multi-field dense search (cosine) combined with BM25 sparse search
- Client-side fusion: min-max weighted (default) or Reciprocal Rank Fusion
- Title and summary keyword boosts, metadata filters, per-source de-duplication

### 3. Document Management
- Replace (`PUT`), upsert, delete, and metadata-only patch operations, keyed by `source_id`

### 4. Utility Endpoints
- Plain text vector search, duplicate-content check, source existence verification, cache clear

## Start Here

| If you are… | Read |
|---|---|
| New to the codebase | [Knowledge Transfer Guide](KT_GUIDE.md) — module index and reading order |
| Setting up locally | [Developer Setup](setup/developer_setup.md), then [Configuration Reference](setup/configuration.md) |
| Integrating a client | [API Endpoints](api/endpoints.md) and [Models](api/models.md) |
| Changing search behaviour | [Search Pipeline Overview](services/search/overview.md) |
| Changing ingestion | [Upload Pipeline](services/ingestion/pipeline.md) |
| Debugging an incident | [Troubleshooting](operations/troubleshooting.md) |

## Technology Stack

| Layer | Technology |
|---|---|
| Web framework | FastAPI, served by uvicorn |
| Vector store | Qdrant (`qdrant-client` 1.18 against server 1.12 — see [Qdrant Compatibility](operations/qdrant_compatibility.md)) |
| Dense embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`, 384 dimensions) |
| Sparse embeddings | `fastembed` BM25 (`Qdrant/bm25`) |
| Query preprocessing | spaCy (`en_core_web_sm`) |
| Chunking | `langchain-text-splitters` |
| Cache | Redis LRU (disabled by default) |

## Building These Docs

```bash
pip install -r requirements-docs.txt
mkdocs serve        # live preview at http://127.0.0.1:8000
mkdocs build        # static site into ./site
```
