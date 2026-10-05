# Developer Setup

**Purpose.** Read this page to get the Vectorization Service running on a developer machine, to understand which external systems the process needs at import time (not only at request time), and to know what `start_mac.sh` does and does not do.

The service is a FastAPI application (`app.main:app`) that depends on Qdrant, Redis (health check and optional cache), PostgreSQL (an import-time side effect, see below), a SentenceTransformer model, a spaCy model, and system packages for OCR.

## 1. Prerequisites

| Requirement | Why it is needed | Source |
|---|---|---|
| Python 3.10+ (the `requirements.txt` comment refers to a 3.10.16 runtime) | Runtime | `requirements.txt` |
| [`uv`](https://github.com/astral-sh/uv) | `start_mac.sh` and the Ansible playbook create the venv with `uv venv` and install with `uv pip install -r requirements.txt` | `start_mac.sh`, `deployment/ansible.yml` |
| Qdrant server (Docker image `qdrant/qdrant`) | Vector store; the service refuses to start if it cannot list collections | `app/core/clients/qdrant.py` |
| Redis | Used by `/api/health` and by the optional query cache | `app/main.py`, `app/core/clients/redis_cache.py` |
| PostgreSQL | Required at **import time** (see section 3) | `app/core/database.py` |
| Tesseract OCR (`brew install tesseract` / `apt-get install tesseract-ocr`) | OCR fallback for scanned PDF pages | `SYSTEM_REQUIREMENTS.md`, `pdf_processor.py` |
| Poppler (`pdftoppm`) | `pdf2image` needs it to rasterise PDF pages for OCR | `pdf2image` dependency (not listed in `SYSTEM_REQUIREMENTS.md`) |
| spaCy model `en_core_web_sm` | Query preprocessing for queries of 3 or more words and 20 or more characters | `app/utils/query_preprocessor.py` |
| Docker | `start_mac.sh` starts Qdrant as a container | `start_mac.sh` |

Note: `SYSTEM_REQUIREMENTS.md` lists only Tesseract and the spaCy model. Poppler is an implicit requirement of `pdf2image`; without it OCR fails and the PDF processor logs `Required OCR library not installed` or `OCR extraction error for page ...`.

Note: `httpx` (used by `app/services/url_text_extractor.py`) is not listed in `requirements.txt`. It is installed transitively by `qdrant-client`. Do not rely on that if you change the Qdrant client pin.

## 2. Quick start (manual)

```bash
git clone https://github.com/ELEVATE-Project/vectorization-service.git
cd vectorization-service

uv venv
uv pip install -r requirements.txt
.venv/bin/python -m spacy download en_core_web_sm

cp .env.sample .env            # then edit values, see Configuration page

# Qdrant (same command start_mac.sh uses)
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 \
  -v "$PWD/.qdrant_storage:/qdrant/storage" qdrant/qdrant

# Run the API
.venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Swagger UI is at `http://localhost:8000/docs` when `ENVIRONMENT=local`. For any other `ENVIRONMENT` value `app/main.py` sets `root_path="/vector"`, so all routes are served under `/vector` behind a proxy (see [Deployment](deployment.md)).

Note: the first start downloads the `all-MiniLM-L6-v2` model from Hugging Face (module-level `SentenceTransformer(settings.EMBEDDING_MODEL)` in `app/core/clients/embedding.py`). If `SPARSE_SEARCH_ENABLED=true`, the `Qdrant/bm25` fastembed model is downloaded on the first sparse encode.

## 3. Import-time dependencies (read this before debugging a failed start)

Importing `app.main` triggers the following, before any request is served:

1. `app/core/clients/embedding.py` loads the SentenceTransformer model and computes `EMBEDDING_DIM = embedding_model.get_embedding_dimension()`. This method exists only in `sentence-transformers>=5.4.0`, which is why `requirements.txt` pins that floor.
2. `app/core/clients/qdrant.py` constructs the `QdrantClient` (lazy; no network call yet).
3. `app/services/translation_service.py` (imported by every file processor) imports `app/core/database.py`, which executes at import:

```python
engine = create_engine(settings.DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
# Create tables
Base.metadata.create_all(bind=engine)
```

`create_all` opens a connection. If PostgreSQL is unreachable, or `psycopg2-binary` is missing, the application fails to import. No queries are made, but it is required at startup. The `DATABASE_URL` default in `app/config.py` is `postgresql://anuj:1234@localhost:5432/ai_vector_service`, which is overridden by `POSTGRES_DATABASE_URI` in `.env`.

4. `app/utils/language_utils.py` runs `logging.basicConfig(... handlers=[FileHandler('app.log'), StreamHandler()])` at import. This configures the root logger at INFO level and creates `app.log` in the current working directory. This is the only logging configuration in the application.

5. During lifespan startup, `ensure_collections_exist()` creates the `documents` and `qa_cache` collections if absent, adds the sparse field when `SPARSE_SEARCH_ENABLED=true`, and creates payload indexes. A failure here logs `Failed to create collections: ...` followed by `Startup failed: ...` and aborts startup.

## 4. `start_mac.sh`

The script is idempotent and macOS/Homebrew specific. It runs with `set -e` and performs these steps in order:

| Step | Behaviour |
|---|---|
| Validate `.env` | Exits if `.env` is missing, then `set -a; source .env; set +a`. The file is sourced **as shell**, so values with spaces, `&`, `;`, or unquoted special characters break the script. |
| Parse Postgres URI | Extracts user, password and DB name from `POSTGRES_DATABASE_URI` with `sed`; exits if the variable is unset. |
| PostgreSQL | `pg_isready`; otherwise `brew services start postgresql@14` (falls back to `postgresql`). Creates the role and database if missing and grants privileges. |
| Redis | `redis-cli ping`; otherwise `brew services start redis`. |
| Qdrant | Starts or restarts a Docker container named `qdrant` from the **unpinned** `qdrant/qdrant` image, maps `${QDRANT_PORT}:6333` and `6334:6334`, mounts `./.qdrant_storage`, and polls `/healthz` up to 15 times. |
| Virtualenv | Recreates `.venv` if it is stale (absolute paths broken), creates it with `uv venv` if missing. |
| Dependencies | `uv pip install -r requirements.txt`, then `python -m spacy download en_core_web_sm`. |
| Start | `.venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload`. |

Note: the local Qdrant container tracks `latest`, whereas QA runs server 1.12 and production runs 1.18 (see [Qdrant compatibility](../operations/qdrant_compatibility.md)). Behaviour that works locally on a new server may fail on QA. Pin the image tag (for example `qdrant/qdrant:v1.12.x`) when you need to reproduce QA.

Note: `migrate_to_sparse_vectors.py` in blue-green mode rewrites `COLLECTION_NAME=` in `.env` (see [Scripts](../operations/scripts.md)); the same `.env` is sourced as shell by this script, which is why that script validates collection names strictly.

## 5. Verifying the setup

```bash
curl -s http://localhost:8000/api/health
```

Note: `/api/health` calls `redis_cache.redis_client.ping()`. `RedisLRUCache.__init__` only creates `redis_client` when `cache_enabled` is true, and `REDIS_CACHE_ENABLED` defaults to `False`. With the defaults the attribute does not exist, the generic `except Exception` branch fires, and the endpoint returns HTTP 503 `{"detail": "Service unhealthy"}` even when Qdrant and Redis are fine. Set `REDIS_CACHE_ENABLED=true` for a green health check. See [Troubleshooting](../operations/troubleshooting.md).

Run the unit tests (no live Qdrant needed) as described in [Testing](../operations/testing.md):

```bash
.venv/bin/python -m pytest tests -v
```

## 6. Configuration

Copy `.env.sample` to `.env`. Every variable, its default, and whether code actually reads it is documented in [Configuration reference](configuration.md). The values that must be correct for a first run are `QDRANT_HOST`, `QDRANT_PORT`, `COLLECTION_NAME`, `POSTGRES_DATABASE_URI`, `REDIS_HOST`, `REDIS_PORT`, and `ENVIRONMENT`.

## Known issues / gotchas

- PostgreSQL is required to import the app despite being unused for queries (`app/core/database.py`).
- `/api/health` returns 503 with default settings because `redis_client` is never created when the cache is disabled.
- `SYSTEM_REQUIREMENTS.md` omits Poppler; `requirements.txt` omits `httpx` and the spaCy model.
- `start_mac.sh` launches an unpinned Qdrant image and sources `.env` as shell.
- `app.log` is written to the process working directory as an import side effect of `language_utils.py`.
- Route paths are `/api/documents...`, `/api/query/`, `/api/cache/redis`. There is no `/v1` segment (`app/api/v1/api.py` mounts routers without a `v1` prefix).

## Related pages

- [Configuration reference](configuration.md)
- [Deployment](deployment.md)
- [Testing](../operations/testing.md)
- [Troubleshooting](../operations/troubleshooting.md)
- [App lifecycle](../backend/app_lifecycle.md)
- [Clients](../backend/clients.md)
