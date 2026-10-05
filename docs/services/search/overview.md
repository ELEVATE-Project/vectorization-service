# Search Pipeline Overview

Read this page first to understand how `POST /api/v1/documents/search` turns a request into a ranked response. It covers the end-to-end flow, a method-by-method map of `PrioritizedSearchService`, the no-query path, error handling, and the timing/log lines you will use when debugging. The scoring maths is on [Ranking and fusion](ranking_and_fusion.md); boosts, filters and the response shape are on [Boosts, filters and results](boosts_filters_and_results.md).

This module is responsible for all document search. The code lives in `app/services/prioritized_search_service.py` (class `PrioritizedSearchService`, about 1800 lines). The endpoint in `app/api/v1/endpoints/documents.py` (`prioritized_search`) only logs the request body and calls `prioritized_search_service.search(request)` on a module-level instance.

See [Known issues / gotchas](#known-issues-gotchas).

---

## 1. End-to-end flow

```text
POST /api/v1/documents/search   (PrioritizedSearchRequest)
  |
  v
PrioritizedSearchService.search(request)
  |
  |-- query empty/blank?  --> _get_unique_source_documents()   [no-query path, section 4]
  |-- top_k <= 0          --> ValueError
  |
  |-- 1. choose filter mode   (detail_filter_score given ? field-level : filter_score)
  |-- 2. preprocess_query(request.query)      -> query_for_embedding
  |       (original request.query is kept as query_for_keyword_match)
  |-- 3. embedding.embed_query(query_for_embedding)  -> validated list[float] (384-dim)
  |-- 4. _build_filters(...)                  -> models.Filter | None
  |-- 5. _candidate_limit(top_k)              -> per-field candidate pool size
  |
  |-- 6. SPARSE_SEARCH_ENABLED ?
  |        true : _hybrid_batch_search()   (5 dense + 1 BM25 request, one batch call)
  |        false: _parallel_batch_search() (5 dense requests, one batch call)
  |        (hybrid falls back to _parallel_batch_search on ImportError/RuntimeError)
  |       -> all_results {point_id: ScoredPoint}, field_scores {point_id: {field: score}}
  |
  |-- no candidates? -> empty PrioritizedSearchResponse (total_results=0)
  |
  |-- 7. _process_and_filter_results()
  |        _rank_results -> filter (filter_score | detail filter)
  |        -> _filter_best_per_source -> [:top_k]
  |
  |-- 8. boost_active = HYBRID_SEARCH_ENABLED and search_mode != "semantic"
  |        title: _get_field_match_sources + _supplement_matches_from_results + _apply_field_boost
  |        summary: same three calls
  |        inject missing title/summary matches via _fetch_field_match_docs
  |        re-sort, re-cap to top_k
  |
  |-- 9. late payload retrieval (only if SPARSE_SEARCH_ENABLED): qdrant_client.retrieve()
  |-- 10. _build_result_items() ; build search_config ; total_results
  v
PrioritizedSearchResponse
```

### Step details

**1. Filter mode.** If `request.detail_filter_score` is not `None`, field-level OR thresholds are used and `filter_score` is ignored (a log line notes this). Otherwise the threshold is `max(request.filter_score, MIN_SEARCH_FILTER_SCORE)` (or `MIN_SEARCH_FILTER_SCORE` when `filter_score` is `None`). `MIN_SEARCH_FILTER_SCORE` is `0` in `app/config.py`.

**2. Preprocessing.** `app/utils/query_preprocessor.py :: preprocess_query()`:

| Condition | Behaviour |
|---|---|
| Empty or whitespace | returns `""` |
| Fewer than `SHORT_QUERY_THRESHOLD` (3) words, or fewer than 20 characters | returns `stripped.lower()`; spaCy is not loaded |
| Otherwise | spaCy `en_core_web_sm` (lazy singleton `_load_spacy_model`); drops tokens where `pos_ == "PRON"`, `is_stop`, `is_punct`, `is_space`; keeps `token.text.lower()` (no lemmatisation despite the module docstring) |
| Everything stripped | falls back to `stripped.lower()` |
| Model missing | `RuntimeError` is re-raised (the search fails) |
| Any other spaCy error | returns `stripped` (original casing) |

In `search()`: `query_for_embedding = preprocessed_query if preprocessed_query.strip() else request.query`. The preprocessed text is used for the dense embedding and, in hybrid mode, for the BM25 query. The original query (`query_for_keyword_match`) is used for the title/summary keyword boost, because stop-word removal would break contiguous-substring matching (for example "ministry of education" becoming "ministry education").

**3. Embedding.** `embedding.embed_query()` rejects empty input and validates the vector length (`EMBEDDING_DIM`) and finiteness. It raises `EmbeddingError` (a `ValueError` subclass), which `app/main.py` maps to HTTP 422. The vector is encoded once and reused for every dense field request.

**4. Filters.** See [Boosts, filters and results](boosts_filters_and_results.md#3-building-filters).

**5. Candidate limit.** `_candidate_limit(top_k)`:

```python
return min(max(top_k, 1) * settings.SEARCH_CANDIDATE_FANOUT, settings.SEARCH_CANDIDATE_MAX)
```

| Setting | Default | Meaning |
|---|---|---|
| `SEARCH_CANDIDATE_FANOUT` | `8` | multiplier on `top_k` |
| `SEARCH_CANDIDATE_MAX` | `2000` | hard cap per field |

Each field query asks Qdrant for this many points. Because the request default `top_k` is `1000000`, the cap (2000) is what applies for default requests. Results beyond 2000 per field are never seen, so a query cannot return more than the union of the per-field candidate sets.

**6a. Dense-only path (`SPARSE_SEARCH_ENABLED=false`, the default).** `_parallel_batch_search()` builds one `QueryRequest` per field in `SEARCH_PRIORITY_ORDER` (`title`, `text`, `tags`, `summary`, `metadata`) with `with_payload=True` and the filter, and sends them in one `qdrant_client.query_batch_points()` call. Results are merged by point id: the first `ScoredPoint` seen for an id is kept, and `field_scores[point_id][field] = result.score`.

**6b. Hybrid path (`SPARSE_SEARCH_ENABLED=true`).** `_hybrid_batch_search()`:

1. Validates the dense vector (`embedding.validate_vector`).
2. Builds the five dense `QueryRequest`s with `with_payload=["source_id", "title", "summary", "tags", "metadata"]` (the heavy `text` field is excluded).
3. Builds the BM25 request: `query_text.replace("_", " ")` is passed to `sparse_encoder.generate_sparse_vector()`. If it returns indices, a `QueryRequest(query=SparseVector(...), using=SPARSE_VECTOR_NAME)` is appended. If the tokeniser produced nothing, the sparse request is not sent and `sparse_issued` is `False`.
4. Executes everything in one `query_batch_points` call (logged as `TIMING: query_batch_points took Xs`).
5. Merges by point id (last `ScoredPoint` seen wins; payloads are identical projections) and stores per-field raw scores. BM25 raw scores go under the key `settings.SPARSE_VECTOR_NAME` (`bm25`).
6. Remaps Qdrant vector names to semantic names via `VECTOR_FIELD_PREFIX` (read with `getattr(settings, "VECTOR_FIELD_PREFIX", "")`; the setting does not exist in `app/config.py`, so the mapping is an identity).
7. Returns `(all_results, field_scores, sparse_issued)`.

On `ImportError` or `RuntimeError` (fastembed missing, BM25 encoder failure) it logs a warning and degrades to `_parallel_batch_search()`, returning `sparse_issued=False`. `AttributeError` and Qdrant transport errors are deliberately not caught.

**7. Rank, filter, dedup.** `_process_and_filter_results()` calls `_rank_results()` (see [Ranking and fusion](ranking_and_fusion.md)), optionally captures the best chunk per source before thresholding (`prethreshold_by_source_out`, only when the boost is active), applies the filter, runs `_filter_best_per_source()`, and slices `[:top_k]`. It returns `(top_results, unique_source_results)`.

**8. Boost.** See [Boosts, filters and results](boosts_filters_and_results.md#1-title-and-summary-boost).

**9. Late payload retrieval.** When `SPARSE_SEARCH_ENABLED` is true and there are results, `qdrant_client.retrieve(ids=[final top_k ids], with_payload=True, with_vectors=False)` replaces each result's partial payload with the full payload (including `text`). On any exception it logs `Failed late payload retrieval` and keeps the partial payload (so `text` would be `''` in the response). Note this runs even if `_hybrid_batch_search` degraded to dense-only, because the condition tests the setting, not the path actually taken.

**10. Build response.** `_build_result_items()` and `search_config`. `total_results` is `len({source_ids in unique_source_results} | injected_source_ids)`.

---

## 2. Method map: `PrioritizedSearchService`

Module-level helper: `_match_any_condition(key, values)` builds a `MatchAny` `FieldCondition` from non-blank stripped values or returns `None`; used by `_build_filters` for the `exclude_*` clauses.

| Method | What it does | Called by |
|---|---|---|
| `__init__` | Copies settings: `collection_name`, `default_top_k`, `max_top_k`, `min_filter_score`, `priority_order`, `default_weights`, `min_score_threshold` | module instantiation in `documents.py` |
| `search(request)` | Orchestrates the whole flow; wraps errors (`ValueError` re-raised; everything else becomes `RuntimeError("Search operation failed: ...")`) | `documents.py :: prioritized_search` |
| `_candidate_limit(top_k)` | `min(max(top_k,1) * FANOUT, MAX)` | `search` |
| `_log_search_request` | Logs query, top_k and which filters are present | `search` |
| `_build_filters(...)` | Compiles request filters into a `models.Filter`; recursive for `any_of` | `search`, `_get_unique_source_documents`, itself |
| `_parallel_batch_search` | Dense-only multi-field batch query | `search`, `_hybrid_batch_search` (fallback) |
| `_hybrid_batch_search` | Dense + BM25 batch query, returns `sparse_issued` | `search` |
| `_process_and_filter_results` | rank, snapshot pre-threshold, filter, dedup, slice | `search` |
| `_rank_results` | Computes `weighted_score` per candidate (dense-only, weighted fusion or RRF); fills `scoring_context_out` | `_process_and_filter_results` |
| `_min_max_normalize` (static) | Min-max to [0,1]; flat pool maps positive to 1.0, zero to 0.0 | `_rank_results` |
| `_rank_positions` (static) | 1-indexed ranks by descending score | `_rank_results` (RRF branch) |
| `_cosine_similarity` (static) | numpy cosine between two vectors; `None` if missing/mismatched/zero-norm | `_fetch_field_match_docs` |
| `_apply_detail_filter` | Keeps a result if ANY of the five field scores meets its threshold | `_process_and_filter_results` |
| `_filter_best_per_source` | Keeps the first (best) result per `source_id`; drops results with no `source_id` | `_process_and_filter_results`, `search` indirectly |
| `_classify_text_match` (static) | `'exact'` if field equals query, `'partial'` if substring, else `None` | `_get_field_match_sources`, `_supplement_matches_from_results` |
| `_get_field_match_sources` | Scrolls Qdrant with `MatchText` on `title` or `summary`, classifies matches into `{source_id: type}` | `search`, `_get_title_match_sources` |
| `_supplement_matches_from_results` | In-memory substring scan of already-retrieved candidates to add infix matches | `search` |
| `_apply_field_boost` | Multiplies `weighted_score` for matched docs (capped at 1.0), records `{field}_match` and `{field}_multiplier`, re-sorts | `search`, `_apply_title_boost` |
| `_fetch_field_match_docs` | Injects docs that matched by keyword but are absent from semantic results, with floor scores | `search`, `_fetch_title_match_docs` |
| `_build_result_items` | `SearchResultItem` list from ranked entries (gates debug fields) | `search` |
| `_get_title_match_sources`, `_apply_title_boost`, `_fetch_title_match_docs` | Backward-compatible title-only wrappers over the `_field_` methods; not called by `search()` | tests / legacy callers |
| `_get_unique_source_documents` | No-query path | `search` |
| `_scroll_and_collect_unique_sources` | Paginated scroll (10000 per page) collecting one point per `source_id` | `_get_unique_source_documents` |
| `_build_source_result_items` | Result items for the no-query path (`score=1.0`, `field_scores={}`) | `_get_unique_source_documents` |

---

## 3. Field weights and configuration used by search

From `app/config.py`:

| Setting | Default | Role |
|---|---|---|
| `SEARCH_PRIORITY_ORDER` | `["title","text","tags","summary","metadata"]` | named vectors queried |
| `SEARCH_PRIORITY_WEIGHTS` | title 0.34, text 0.26, tags 0.20, summary 0.12, metadata 0.08 (sum 1.0) | dense score weights |
| `MIN_SEARCH_FILTER_SCORE` | `0` (int) | floor for `filter_score` |
| `MIN_WEIGHTED_SCORE_THRESHOLD` | `0.0` | stored on the service as `min_score_threshold`; never used |
| `DEFAULT_SEARCH_TOP_K` / `MAX_SEARCH_TOP_K` | `10` / `100` | stored on the service; never used (request default `top_k` is `1000000`) |
| `HYBRID_SEARCH_ENABLED` | env, default `true` | master switch for title/summary boost |
| `SPARSE_SEARCH_ENABLED` | env, default `false` | hybrid BM25 path and late retrieval |
| `HYBRID_FUSION_METHOD` | env, default `weighted` | `weighted` or `rrf` (validated at startup) |
| `INCLUDE_SCORING_DEBUG` | env, default `false` | default for `include_scoring_debug` |

---

## 4. The no-query path

If `request.query` is `None`, empty or whitespace, `search()` returns `_get_unique_source_documents(request)` before any filter-mode logic or `top_k` validation:

1. `_build_filters(...)` with the same request filters (including `any_of` and exclusions).
2. `_scroll_and_collect_unique_sources(filter)`: scrolls the whole filtered collection (pages of 10000, payload only, no vectors) and keeps the first point seen per `source_id`.
3. `top_k = min(request.top_k, len(unique_documents))`; takes the first `top_k` in scroll order (no ranking).
4. `_build_source_result_items`: each item has `score=1.0` and `field_scores={}`.

Response: `query=None`, `total_results=len(unique_documents)` (all sources, not just the returned ones), and `search_config` of `{"search_fields": [], "weights": {}, "priority_order": [], "filters_applied": ..., "mode": "unique_source_id"}`. Scores, boosts, `filter_score`, `detail_filter_score` and `search_mode` have no effect on this path. Scroll order is Qdrant's internal order, so which point represents a source (and thus its `text`/`id`) is arbitrary but stable.

---

## 5. Error handling

| Situation | Result |
|---|---|
| `top_k <= 0` (with a query) | `ValueError` raised; no `ValueError` handler exists in `app/main.py`, so this surfaces as a 500 (the endpoint docstring says 422) |
| Empty or malformed query vector | `EmbeddingError` -> 422 via `embedding_error_handler` |
| spaCy model not installed (long queries only) | `RuntimeError` -> wrapped as `RuntimeError("Search operation failed: ...")` -> 500 |
| Qdrant failure in batch search | wrapped `RuntimeError` -> 500 |
| BM25 encoder or fastembed failure | warning logged, dense-only fallback |
| Title/summary match scroll fails | warning logged, treated as zero matches (non-fatal) |
| Bulk fetch of missing docs fails | warning logged, nothing injected (non-fatal) |
| Late retrieve fails | error logged, partial payloads kept |
| One result fails to serialise | warning logged, that item is skipped |
| No candidates | empty response with `total_results=0` and a reduced `search_config` (no `search_mode`, `fusion_method`, etc.) |

---

## 6. Timing and logging

All logging uses the module logger `app.services.prioritized_search_service`. Useful markers, in order:

| Log line | Meaning |
|---|---|
| `Filter mode: DETAIL_FILTER_SCORE ...` / `FILTER_SCORE (weighted score threshold = X)` | which filter mode was chosen |
| `Original query` / `Preprocessed query` | preprocessing result |
| `========== SEARCH REQUEST ==========` | query, top_k, active filters |
| `Starting hybrid batch search (dense + BM25 sparse, <method> fusion)` or `Starting parallel batch search across all fields` | path taken |
| `TIMING: query_batch_points took Xs` | hybrid path only (the dense-only path has no timing line) |
| `Sparse branch was issued but matched nothing; dense will carry full weight` | zero BM25 hits |
| `BM25 query '...' produced no tokens; sparse branch not issued` | zero-token BM25 query |
| `Total documents matched` / `Ranked results` / `After filtering` / `Unique sources` | funnel counts |
| `title match sources found: N (exact=.., partial=..)` | boost scroll results (same for `summary`) |
| `Injected N title-match docs missing from semantic results` | injection counts |
| `Bulk fetch of N points ... completed in Xs` | injection fetch timing |
| `TIMING: late retrieve of N docs took Xs` | late retrieval (hybrid mode) |
| `========== SEARCH COMPLETED ==========` / `Returned N results from M unique sources` | final summary |

The endpoint also logs the full request body (`[/documents/search] request body: ...`) at INFO.

---

## Known issues / gotchas

- **Weights in the endpoint docstring.** The endpoint docstring in `documents.py` quotes weights 0.36/0.27/0.14/0.14/0.09. The code (`SEARCH_PRIORITY_WEIGHTS`) uses 0.34/0.26/0.20/0.12/0.08. The `DetailFilterScore` per-field defaults (0.36/0.27/0.14/0.14/0.09) are a separate set of thresholds and are correct as thresholds.
- **`top_k <= 0` yields 500, not 422**, contrary to the endpoint docstring.
- **`_hybrid_batch_search` replaces points last-wins; `_parallel_batch_search` first-wins.** Harmless today because payloads are identical per point.
- **Late retrieval is keyed to the setting, not the path.** After a dense-only fallback inside hybrid mode it still runs an extra `retrieve` call.
- **Dense-only path fetches full payloads** (`with_payload=True`) for up to `SEARCH_CANDIDATE_MAX` points per field, including `text`. Only the hybrid path projects payloads.
- **Config values never applied:** `MAX_SEARCH_TOP_K`, `DEFAULT_SEARCH_TOP_K`, `MIN_WEIGHTED_SCORE_THRESHOLD`, `METADATA_MATCH_BOOST`.
- **`query_preprocessor.py` docstring claims lemmatisation**; the code does not lemmatise.
- **`search_config` differs between the empty-result response and a normal one** (the empty response omits `search_mode`, `fusion_method`, `filter_score`, flags).

## Related pages

- [Ranking and fusion](ranking_and_fusion.md)
- [Boosts, filters and results](boosts_filters_and_results.md)
- [Qdrant data model](../../architecture/qdrant_data_model.md)
- [API models](../../api/models.md)
- [Configuration](../../setup/configuration.md)
- [Troubleshooting](../../operations/troubleshooting.md)
