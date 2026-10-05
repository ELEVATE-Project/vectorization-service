# API Models (Pydantic)

**Purpose.** Read this page for the exact shape, defaults and validation of every Pydantic model in `app/models/api_models.py`, plus the single SQLAlchemy model in `app/models/db_models.py`. Endpoint-level usage is in [API endpoints](endpoints.md).

Defaults that reference `settings.*` are evaluated **once at import time** (class body), so changing the environment later has no effect on a running process.

## 1. Search models

### `DetailFilterScore`

Per-field thresholds for the OR-style filter. All fields `float`, `ge=0.0`, `le=1.0`.

| Field | Default | Meaning |
|---|---|---|
| `title` | 0.36 | minimum title cosine score |
| `text` | 0.27 | minimum chunk-text score |
| `tags` | 0.14 | minimum tags score |
| `summary` | 0.14 | minimum summary score |
| `metadata` | 0.09 | minimum metadata score |

A document passes if ANY field meets its threshold. Note: the defaults equal the field weights from config.

### `FilterBlock`

One alternative inside `any_of`. `model_config = ConfigDict(extra="forbid")`, so unknown keys (including a nested `any_of`) cause 422.

| Field | Type | Default | Qdrant field |
|---|---|---|---|
| `categories` | `Optional[List[str]]` | `None` | `tags` |
| `organizations` | `Optional[List[str]]` | `None` | `metadata.company` |
| `resource_type` | `Optional[List[str]]` | `None` | `metadata.DOCUMENT_TYPE` |
| `file_type` | `Optional[List[str]]` | `None` | `metadata.type` |
| `exclude_organizations` | `Optional[List[str]]` | `None` | `must_not` on `metadata.company` |
| `exclude_file_type` | `Optional[List[str]]` | `None` | `must_not` on `metadata.type` |

Validator `_reject_empty_block` (`mode="after"`): raises `ValueError` unless at least one list contains a non-blank item, because an empty block would match everything.

### `PrioritizedSearchRequest`

| Field | Type | Default | Meaning |
|---|---|---|---|
| `query` | `Optional[str]` | `None` | no query: scroll mode, one doc per `source_id` |
| `top_k` | `int` | `1000000` | result cap; no min/max constraint |
| `filter_score` | `Optional[float]` (0-1) | `None` | weighted-score threshold; ignored when `detail_filter_score` is set |
| `detail_filter_score` | `Optional[DetailFilterScore]` | `None` | per-field OR thresholds |
| `categories` | `Optional[List[str]]` | `None` | tag filter (OR within list) |
| `organizations` | `Optional[List[str]]` | `None` | company filter |
| `resource_type` | `Optional[List[str]]` | `None` | `DOCUMENT_TYPE` text-match filter |
| `file_type` | `Optional[List[str]]` | `None` | `metadata.type` filter |
| `exclude_organizations` | `Optional[List[str]]` | `None` | exclusion (`must_not`) |
| `exclude_file_type` | `Optional[List[str]]` | `None` | exclusion (`must_not`) |
| `any_of` | `Optional[List[FilterBlock]]` | `None` | OR between blocks, AND-ed with top-level filters |
| `search_mode` | `Literal["hybrid","semantic"]` | `"hybrid"` | `semantic` skips title/summary boosts |
| `include_scoring_debug` | `bool` | `settings.INCLUDE_SCORING_DEBUG` (`false`) | emit fusion breakdown |

Validator `_reject_lone_alternative` on `any_of`: if provided, requires at least 2 entries (a single alternative is an AND and must use top-level fields).

Note: the `resource_type` field description in the model says `metadata.KEY ENTITIES`, but the filter code maps it to `metadata.DOCUMENT_TYPE` (the `FilterBlock` description is correct). The `filter_score` description says "when filter_score=0" for the detail mode; actual rule is "when `detail_filter_score` is not null".

### `SearchResultItem`

| Field | Type | Default | Meaning |
|---|---|---|---|
| `id` | `str` | required | Qdrant point id |
| `text` | `str` | required | chunk text |
| `title`, `summary` | `Optional[str]` | `None` | document-level |
| `tags` | `Optional[List[str]]` | `None` | document tags |
| `metadata` | `Dict[str, Any]` | required | payload metadata |
| `source_id` | `str` | required | document id |
| `score` | `float` | required | final calibrated score (0-1 for semantic results) |
| `field_scores` | `Dict[str, Optional[float]]` | `{}` | per-field cosine; `None` for unscored fields on injected docs |
| `match_source` | `Optional[str]` | `None` | e.g. `title_keyword_match`; set only on keyword-injected docs, whose `score` is a synthetic floor |
| `keyword_score` | `Optional[float]` | `None` | raw BM25 score; not gated by debug flag |
| `rrf_score` | `Optional[float]` | `None` | raw RRF pre-normalization; debug and `HYBRID_FUSION_METHOD=rrf` |
| `dense_rank` | `Optional[int]` | `None` | rank in combined dense list; debug |
| `sparse_rank` | `Optional[int]` | `None` | rank in sparse list; debug |
| `raw_dense` | `Optional[float]` | `None` | weighted cosine sum pre-fusion; debug |
| `normalized_dense` | `Optional[float]` | `None` | min-max dense; debug, hybrid only |
| `normalized_sparse` | `Optional[float]` | `None` | min-max BM25; debug |
| `title_multiplier` | `Optional[float]` | `None` | title boost applied (1.0 = none); debug |
| `summary_multiplier` | `Optional[float]` | `None` | summary boost applied; debug |
| `title_match` | `Optional[str]` | `None` | `exact` / `partial` / `None` |
| `summary_match` | `Optional[str]` | `None` | `exact` / `partial` / `None` |

### `PrioritizedSearchResponse`

| Field | Type | Default | Meaning |
|---|---|---|---|
| `query` | `Optional[str]` | `None` | echoed query |
| `total_results` | `int` | required | `len(results)` |
| `top_k` | `int` | required | echoed top_k |
| `results` | `List[SearchResultItem]` | required | ranked results |
| `search_config` | `Dict[str, Any]` | `{}` | config used (includes `fusion_method`) |

## 2. Text search models

| Model | Field | Type | Default | Meaning |
|---|---|---|---|---|
| `TextSearchRequest` | `query` | `str` | required, `min_length=1` | query text |
| | `top_k` | `int` | `10` | max chunks (description wrongly says default 5) |
| | `threshold` | `float` | `0.40` | min cosine score (literal, not read from settings) |
| `TextSearchResultItem` | `source_id` | `str` | required | document id |
| | `text` | `str` | required | chunk text |
| | `score` | `float` | required | cosine score |
| | `metadata` | `Dict[str, Any]` | required | payload metadata |
| `TextSearchResponse` | `query` | `str` | required | echoed |
| | `total_results` | `int` | required | count |
| | `results` | `List[TextSearchResultItem]` | required | chunks |

## 3. Similarity and verification models

| Model | Field | Type | Default | Meaning |
|---|---|---|---|---|
| `SimilarityCheckRequest` | `text` | `str` | required, `min_length=1` | text to compare (first 1000 chars embedded) |
| | `company_id` | `str` | required | filter on `metadata.company` |
| | `threshold` | `float` | `0.85` | `score_threshold` for Qdrant |
| | `exclude_source_id` | `Optional[str]` | `None` | `must_not` on `source_id` |
| `SimilarityCheckResponse` | `has_similar` | `bool` | required | any hit |
| | `similar_documents` | `List[Dict[str, Any]]` | required | up to 5 hits (`source_id`, `similarity_score`, `metadata`, `text_preview`, `chunk_id`) |
| `SourceVerificationRequest` | `source_ids` | `List[str]` | required | IDs to check |
| `SourceVerificationResponse` | `total_requested` | `int` | required | count of unique IDs |
| | `found` / `not_found` | `List[str]` | required | partition of unique IDs, input order preserved |
| | `found_count` / `not_found_count` | `int` | required | list lengths |

## 4. Document management and legacy query models

| Model | Field | Type | Default | Meaning |
|---|---|---|---|---|
| `DeleteRequest` | `source_id` | `str` | required | document to delete |
| | `company_id` | `Optional[str]` | `None` | optional company scoping |
| `MultilingualQueryRequest` | `query` | `str` | required | query |
| | `search_limit` | `int` | `settings.VECTOR_SEARCH_LIMIT` (1) | per-request limit |
| | `priority_filter` | `Optional[str]` | `None` | `P1`/`P2`/... |
| `MultilingualQueryResponse` | `relevant_texts` | `List[Dict[str, Any]]` | required | hit dicts |
| | `original_query` | `str` | required | echoed |
| | `translated_query` | `Optional[str]` | `None` | always `None` (translation disabled) |
| | `language` | `str` | required | always `"en"` currently |

Models defined but not referenced by any route in `documents.py` or `query.py`: `DocumentMetadata` (`source`, `page`, `row`), `DocumentUploadRequest` (`priority`), `QueryRequest` (`query`, `search_limit`, `force_llm`), `SearchResult` (`text`, `metadata`, `score`). Treat them as dead code unless another module imports them.

## 5. Database model (`app/models/db_models.py`)

`TranslationRecord` (table `translations`, declarative `Base`):

| Column | Type | Notes |
|---|---|---|
| `chunk_id` | `String` | primary key |
| `original_text` | `Text` | not null; original (Hindi) text |
| `translated_text` | `Text` | not null |
| `created_at` | `DateTime` | default `datetime.utcnow` |

See [App lifecycle](../backend/app_lifecycle.md) for why this layer matters even though no row is currently written.

## Related pages

- [API endpoints](endpoints.md)
- [Auxiliary services](../services/auxiliary_services.md)
- [Search overview](../services/search/overview.md)
- [Boosts, filters and results](../services/search/boosts_filters_and_results.md)
