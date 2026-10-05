# Configuration Reference

**Purpose.** Read this page to find every setting in `app/config.py`, its type and default, which module consumes it, and whether it actually has any effect. Settings that are declared but never read, or that behave differently from what `.env.sample` implies, are flagged.

## 1. How configuration is loaded

`app/config.py` defines one `Settings(BaseSettings)` class and a module-level singleton `settings = Settings()`.

```python
load_dotenv()                                   # .env -> os.environ (does not override real env vars)
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # set unconditionally at import

class Settings(BaseSettings):
    QDRANT_HOST: str = os.getenv("QDRANT_HOST", "127.0.0.1")
    QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", 6333))
    # ...
```

Two mechanisms read the environment at the same time:

1. Many defaults are written as `os.getenv("NAME", "default")`. They are evaluated once, when the class body executes, after `load_dotenv()` has run.
2. Because the class extends `pydantic_settings.BaseSettings`, **every field** (including those with plain literal defaults such as `CHUNK_SIZE: int = 3000`) is also overridden by an environment variable of the same name. This was verified by running with `REDIS_CACHE_ENABLED=true CHUNK_SIZE=1234 DATABASE_URL=x`: all three took effect.

Consequences:

- Precedence: a process environment variable wins over `.env` (`load_dotenv()` does not override), and either wins over the code default.
- Every setting in the tables below can be set from `.env`, including the ones whose default is a literal. `REDIS_CACHE_ENABLED=true` works.
- `int(os.getenv(...))` defaults raise `ValueError` at import for malformed values (for example `QDRANT_PORT=abc` fails with `invalid literal for int()`), before pydantic validation.
- Boolean parsing differs between the two mechanisms. The `os.getenv(...).lower() == "true"` default only accepts the literal string `true`; pydantic then re-reads the env var with its own bool rules (`1`, `yes`, `on`, `true`). The pydantic result wins, so `HYBRID_SEARCH_ENABLED=1` evaluates to true.
- `DATABASE_URL` has two spellings: the env var `POSTGRES_DATABASE_URI` feeds the default expression, while an env var literally named `DATABASE_URL` overrides the field through pydantic. Prefer `POSTGRES_DATABASE_URI`.
- `SEARCH_PRIORITY_ORDER` (list) and `SEARCH_PRIORITY_WEIGHTS` (dict) can be overridden with JSON strings, but this is not documented anywhere else and is not recommended.

### Validation (`_validate_fusion_config`)

A `model_validator(mode="after")` raises at import (and therefore prevents startup) when:

| Condition | Error text |
|---|---|
| `HYBRID_FUSION_METHOD` not in `{"weighted", "rrf"}` | `HYBRID_FUSION_METHOD must be one of ['rrf', 'weighted'], got ...` |
| `HYBRID_DENSE_WEIGHT` or `HYBRID_SPARSE_WEIGHT` negative or not finite | `... must be a finite non-negative number, got ...` |
| `HYBRID_DENSE_WEIGHT + HYBRID_SPARSE_WEIGHT > 1.0` (tolerance 1e-9) | `HYBRID_DENSE_WEIGHT + HYBRID_SPARSE_WEIGHT must not exceed 1.0 (got ...)` |

Note: the sum may be less than 1.0 without error. `.env.sample` says the weights "should sum to 1.0"; the code only enforces "at most 1.0".

`HYBRID_FUSION_METHOD` is lower-cased before validation, so `RRF` is accepted.

## 2. Legend

- **Read by**: modules that reference the setting (verified with a repository grep). "none" means the field is declared but no code reads it.
- **In `.env.sample`**: whether the sample file documents it.

## 3. Qdrant

| Variable | Type | Default | Read by | In `.env.sample` | Notes |
|---|---|---|---|---|---|
| `QDRANT_HOST` | str | `127.0.0.1` | `core/clients/qdrant.py`, scripts | yes | |
| `QDRANT_PORT` | int | `6333` | `core/clients/qdrant.py`, scripts | yes | HTTP port; gRPC is not used. |
| `QDRANT_CHECK_COMPATIBILITY` | bool | `false` | `core/clients/qdrant.py`, `migrate_to_sparse_vectors.py`, `rename_key_entities_field.py` | yes | Passed as `check_compatibility=` to `QdrantClient`. `false` suppresses the client-1.18 vs server-1.12 `UserWarning`. See [Qdrant compatibility](../operations/qdrant_compatibility.md). |
| `COLLECTION_NAME` | str | `documents` | `qdrant.py`, all document operation services, `prioritized_search_service.py`, `query_service.py`, `similarity_service.py`, `source_verification_service.py`, `text_embedding_search_service.py`, scripts | yes | The main collection. The migration script rewrites this line in `.env`. |
| `QA_CACHE_COLLECTION` | str | `qa_cache` | `qdrant.py` only | yes | Created at startup with a single-vector config; nothing writes to or reads from it. |

## 4. Embedding, models and external services

| Variable | Type | Default | Read by | In `.env.sample` | Notes |
|---|---|---|---|---|---|
| `EMBEDDING_MODEL` | str | `all-MiniLM-L6-v2` | `core/clients/embedding.py` | yes | Changing it changes vector dimension (`EMBEDDING_DIM` is derived from the model); existing collections must be recreated. |
| `TRANSLATION_API_URL` | str | `https://demo-api.models.ai4bharat.org/inference/translation/v2` | `utils/language_utils.py` | no | Hindi translation endpoint. Not env-documented. |
| `AWS_REGION` | str | `us-east-1` | **none** | yes | Declared and documented but unused. |
| `LLAMA_MODEL_ID` | str | `meta.llama3-70b-instruct-v1:0` | **none** | yes | Unused. `.env.sample` describes it as "AWS Bedrock LLM (summarisation / tagging)"; no such code exists in `app/`. |

## 5. Chunking, ingestion and upload

| Variable | Type | Default | Read by | In `.env.sample` | Notes |
|---|---|---|---|---|---|
| `CHUNK_SIZE` | int | `3000` | `csv_processor`, `docx_processor`, `pdf_processor`, `text_processor` | yes | Characters per chunk. |
| `CHUNK_OVERLAP` | int | `500` | same as above | yes | |
| `MARKDOWN_CHUNK_SIZE` | int | `3500` | `text_processor` (markdown), `xlsx_processor` | yes | |
| `MARKDOWN_CHUNK_OVERLAP` | int | `800` | same | yes | |
| `PDF_CHUNK_SIZE` | int | `3000` | **none** | no | Unused. `pdf_processor` reads `CHUNK_SIZE`. |
| `PDF_CHUNK_OVERLAP` | int | `500` | **none** | no | Unused. |
| `URL_EXTRACTION_CHUNK_SIZE` | int | `1500` | `upload_service.py` | no | Used when `metadata.markdown_url` is supplied. |
| `URL_EXTRACTION_CHUNK_OVERLAP` | int | `300` | `upload_service.py` | no | |
| `URL_REQUEST_TIMEOUT` | int (s) | `30` | `url_text_extractor.py` | no | httpx timeout. |
| `PAGE_TEXT_THRESHOLD` | int | `20` | `pdf_processor.py` | no | A page with fewer stripped characters triggers OCR. |
| `MAX_FILE_SIZE_MB` | int | `1024` | `file_processors/base_processor.py` (`_validate_file_content`) | yes | See note below. |

Note on `MAX_FILE_SIZE_MB`: `_validate_file_content` raises `ValueError("File ... is too large (...MB). Maximum allowed size is ...MB")` and is called by the text, DOCX, CSV, PDF and XLSX processors. The limit is applied only after the whole upload has been read into memory (`await file.read()`), so it does not protect memory; and the `ValueError` is raised outside the processors' own `try` blocks, so it reaches `UploadService.process` and is returned as HTTP 500 `Upload failed: File <name> is too large (...)` rather than 413.

## 6. Search (core)

| Variable | Type | Default | Read by | In `.env.sample` | Notes |
|---|---|---|---|---|---|
| `SEARCH_PRIORITY_ORDER` | list | `["title","text","tags","summary","metadata"]` | `prioritized_search_service.py` | no | Names of the five dense vectors searched. |
| `SEARCH_PRIORITY_WEIGHTS` | dict | `title 0.34, text 0.26, tags 0.20, summary 0.12, metadata 0.08` | `prioritized_search_service.py` | no | Weights of the combined dense cosine sum. The values `0.36/0.27/0.14/0.14/0.09` are the `DetailFilterScore` threshold defaults, not these weights. |
| `MIN_SEARCH_FILTER_SCORE` | int | `0` | `prioritized_search_service.py` | no | Floor applied as `max(request.filter_score, MIN_SEARCH_FILTER_SCORE)` and used when no `filter_score` is sent. |
| `DEFAULT_SEARCH_TOP_K` | int | `10` | stored on the service (`self.default_top_k`), never used afterwards | yes | Effectively unused. The request model default `top_k` is `1000000`. |
| `MAX_SEARCH_TOP_K` | int | `100` | stored on the service (`self.max_top_k`), never used afterwards | yes | Not enforced anywhere. |
| `MIN_WEIGHTED_SCORE_THRESHOLD` | float | `0.0` | stored on the service (`self.min_score_threshold`), never used afterwards | no | Effectively unused. |
| `SIMILARITY_THRESHOLD` | float | `0.40` | `query_service.py` only | yes | Legacy `/api/query/` flow. The `/documents/text-search` endpoint takes `threshold` from the request (default `0.40` in `TextSearchRequest`); the setting does not drive it. |
| `VECTOR_SEARCH_LIMIT` | int | `1` | `models/api_models.py` (default `search_limit` of query requests), `query_service.py` | yes | Legacy query flow only. |
| `SHORT_QUERY_THRESHOLD` | int | `3` | `utils/query_preprocessor.py` | yes | Queries with fewer words skip spaCy (queries shorter than 20 characters are also skipped, hardcoded). |
| `INCLUDE_SCORING_DEBUG` | bool | `false` | `models/api_models.py` (default of request field `include_scoring_debug`) | yes | Per-request override is possible. |
| `VECTOR_FIELD_PREFIX` | (not declared) | `""` | `prioritized_search_service.py` via `getattr(settings, "VECTOR_FIELD_PREFIX", "")` | no | Not a `Settings` field, so an environment variable of that name is ignored; the prefix is always empty unless a field is added to `Settings`. |

## 7. Phase 1 hybrid: boosts

| Variable | Type | Default | Read by | In `.env.sample` | Notes |
|---|---|---|---|---|---|
| `HYBRID_SEARCH_ENABLED` | bool | `true` | `prioritized_search_service.py` | yes | Master switch for title/summary keyword boosts and injection. Also reported in `search_config`. |
| `EXACT_TITLE_BOOST` | float | `2.5` | `prioritized_search_service.py` | yes | Multiplier, final score capped at 1.0. |
| `PARTIAL_TITLE_BOOST` | float | `1.5` | same | yes | |
| `EXACT_SUMMARY_BOOST` | float | `1.4` | same | yes | |
| `PARTIAL_SUMMARY_BOOST` | float | `1.2` | same | yes | |
| `METADATA_MATCH_BOOST` | float | `1.2` | **none** | yes | Declared, documented as "reserved for future use", never read. |
| `INJECTED_DOC_SCORING_MAX` | int | `200` | `prioritized_search_service.py` | yes | Maximum keyword-matched-but-missed documents whose vectors are fetched for scoring; beyond this their `field_scores` stay `None`. |

## 8. Phase 2 hybrid: sparse BM25 and fusion

| Variable | Type | Default | Read by | In `.env.sample` | Notes |
|---|---|---|---|---|---|
| `SPARSE_SEARCH_ENABLED` | bool | `false` | `qdrant.py`, `upload_service.py`, `prioritized_search_service.py` | yes | Controls collection schema at creation (and an attempted `update_collection` on existing collections), BM25 encoding at ingest, and BM25 queries at search. |
| `SPARSE_VECTOR_NAME` | str | `bm25` | `qdrant.py`, `upload_service.py`, `prioritized_search_service.py`, migration script | yes | Must match the field name in the collection. |
| `HYBRID_FUSION_METHOD` | str | `weighted` | `prioritized_search_service.py`, `api_models.py`, `documents.py` (docstring) | yes | `weighted` or `rrf`. Validated at import. |
| `HYBRID_DENSE_WEIGHT` | float | `0.7` | `prioritized_search_service.py` | yes | Used only when fusion is `weighted`. |
| `HYBRID_SPARSE_WEIGHT` | float | `0.3` | same | yes | Used only when fusion is `weighted`. |
| `RRF_K` | int | `60` | `prioritized_search_service.py` | yes | Used only when fusion is `rrf`. The `.env.sample` comment that it also applies to "the detail_filter_score fallback" is stale: that fallback no longer exists. |
| `SEARCH_CANDIDATE_FANOUT` | int | `8` | `prioritized_search_service.py` | yes | Candidate limit per field = `min(top_k * FANOUT, MAX)`. |
| `SEARCH_CANDIDATE_MAX` | int | `2000` | `prioritized_search_service.py` | yes | Hard cap per field. The 1.0.0 release notes quote `min(top_k x 20, 10000)`; code and sample use 8 and 2000. |

## 9. Redis and PostgreSQL

| Variable | Type | Default | Read by | In `.env.sample` | Notes |
|---|---|---|---|---|---|
| `REDIS_HOST` | str | `localhost` | `core/clients/redis_cache.py` | yes | |
| `REDIS_PORT` | int | `6379` | same | yes | |
| `REDIS_PASSWORD` | str | `""` | **none** | yes | Never passed to `redis.Redis(...)`. Password-protected Redis cannot be used with the current code. |
| `REDIS_CACHE_TTL` | int (s) | `86400` | `redis_cache.py` | yes | |
| `REDIS_MAX_CACHE_SIZE` | int | `1000` | `redis_cache.py` | yes | |
| `REDIS_CACHE_ENABLED` | bool | `False` | `redis_cache.py` (constructor flag), `query_service.py` | **no** | Literal default but overridable from the environment (see section 1). When false, `RedisLRUCache` does not create `redis_client`, which breaks `/api/health` (see [Troubleshooting](../operations/troubleshooting.md)). Only the legacy `QueryService` writes to the cache; `PrioritizedSearchService` never does. |
| `MAX_CACHE_RESULTS` | int | `1` | **none** | no | Unused. |
| `DATABASE_URL` (env: `POSTGRES_DATABASE_URI`) | str | `postgresql://anuj:1234@localhost:5432/ai_vector_service` | `core/database.py` | yes (as `POSTGRES_DATABASE_URI`) | Engine and `create_all` run at import (see [Developer setup](developer_setup.md)). The default embeds a developer credential; always override. |

## 10. Environment

| Variable | Type | Default | Read by | In `.env.sample` | Notes |
|---|---|---|---|---|---|
| `ENVIRONMENT` | str | `local` | `app/main.py` | yes | `local` gives `root_path=""`; any other value (`test`, `staging`, `production`) gives `root_path="/vector"`. The test suite sets `test`. |

## 11. Hardcoded values that look configurable

| Value | Location | Note |
|---|---|---|
| Query skip rule: `len(query) < 20` | `utils/query_preprocessor.py` | Not controlled by `SHORT_QUERY_THRESHOLD`. |
| Floor score `0.15` for injected documents | `prioritized_search_service.py` (`FLOOR_SCORE`) | |
| Batch size `100` for upserts | `qdrant.upload_to_qdrant` | |
| BM25 model `Qdrant/bm25` | `sparse_encoder.py` (`_SPARSE_MODEL`) | |
| Uvicorn port and workers | `deployment/templates/vectorization-service-uvicorn.j2` (8000, 4) | Ansible vars `uvicorn_port` and `uvicorn_workers` are declared but not used by the template. |
| Log file `app.log` and INFO level | `utils/language_utils.py` | |

## 12. `.env.sample` review

The file parses cleanly with `python-dotenv` and with shell `source`. It contains no invalid trailing comments. Remaining issues:

- It documents `AWS_REGION`, `LLAMA_MODEL_ID`, `METADATA_MATCH_BOOST`, `REDIS_PASSWORD`, `DEFAULT_SEARCH_TOP_K`, `MAX_SEARCH_TOP_K` although the code does not use them.
- It omits `REDIS_CACHE_ENABLED`, `TRANSLATION_API_URL`, `PAGE_TEXT_THRESHOLD`, `URL_*`, `SEARCH_PRIORITY_*`, `MIN_*`.
- Comments say sparse search "requires qdrant-client[fastembed]>=1.9.0"; `requirements.txt` pins `>=1.18.0,<2.0.0` and the code assumes 1.18 APIs.
- `POSTGRES_DATABASE_URI` is a placeholder (`username:password`) that `start_mac.sh` will use literally to create a role if not changed.
- The file is sourced as shell by `start_mac.sh`, so any value containing spaces must be quoted.
- `deployment/json2env.sh` writes every value double-quoted (`KEY="value"`), which both `python-dotenv` and the shell accept.

## Known issues / gotchas

- Unused settings: `AWS_REGION`, `LLAMA_MODEL_ID`, `PDF_CHUNK_SIZE`, `PDF_CHUNK_OVERLAP`, `MAX_CACHE_RESULTS`, `METADATA_MATCH_BOOST`, `REDIS_PASSWORD`, `DEFAULT_SEARCH_TOP_K`, `MAX_SEARCH_TOP_K`, `MIN_WEIGHTED_SCORE_THRESHOLD`.
- `REDIS_CACHE_ENABLED` is not hardcoded in effect.
- `MAX_FILE_SIZE_MB` is enforced (post-read, in processors).
- `SIMILARITY_THRESHOLD` does not drive `/documents/text-search`.
- Default dense weights in code are 0.34/0.26/0.20/0.12/0.08; the 0.36/0.27/0.14/0.14/0.09 values are the `DetailFilterScore` threshold defaults.
- `.env.sample` has no syntax errors, but several documented keys have no effect.

## Related pages

- [Developer setup](developer_setup.md)
- [Deployment](deployment.md)
- [Search overview](../services/search/overview.md)
- [Ranking and fusion](../services/search/ranking_and_fusion.md)
- [Boosts, filters and results](../services/search/boosts_filters_and_results.md)
- [Qdrant compatibility](../operations/qdrant_compatibility.md)
