# Boosts, Filters and Results

Read this page to understand everything that happens around the score: the title/summary keyword boost (phases A and B, floor-score injection, match classification), how request filters become a Qdrant filter, the two score-threshold modes, per-source deduplication, how the response is built, and what `search_mode` changes. It ends with a list of gotchas.

Code: `app/services/prioritized_search_service.py` (`PrioritizedSearchService`), request/response models in `app/models/api_models.py`. For the scoring formulas see [Ranking and fusion](ranking_and_fusion.md); for the overall flow see [Search overview](overview.md).

---

## 1. Title and summary boost

The boost is a keyword substring signal layered on top of the semantic score. It runs inside `search()` only when:

```python
boost_active = settings.HYBRID_SEARCH_ENABLED and search_mode != "semantic"
```

It uses the ORIGINAL request query (`query_for_keyword_match = request.query`), lower-cased and stripped, not the spaCy-preprocessed one.

### Settings

| Setting | Default | Applied as |
|---|---|---|
| `HYBRID_SEARCH_ENABLED` | `true` (env) | master switch |
| `EXACT_TITLE_BOOST` | `2.5` | multiplier for exact title match |
| `PARTIAL_TITLE_BOOST` | `1.5` | multiplier for partial title match |
| `EXACT_SUMMARY_BOOST` | `1.4` | multiplier for exact summary match |
| `PARTIAL_SUMMARY_BOOST` | `1.2` | multiplier for partial summary match |
| `INJECTED_DOC_SCORING_MAX` | `200` | max injected docs for which vectors are fetched and scored |
| `METADATA_MATCH_BOOST` | `1.2` | defined, never used |

### Match classification

`_classify_text_match(query_lower, field_lower)`:

| Result | Condition |
|---|---|
| `None` | field empty, or query not a substring of the field |
| `"exact"` | field (lower-cased) equals the query |
| `"partial"` | query is a substring of the field (prefix, middle or suffix; infix is folded into `partial`) |

The matching is plain case-insensitive substring over the whole query string, so a multi-word query matches only if those words appear contiguous and in order.

### Building the match map

For each of `title` and then `summary`, `search()` builds `{source_id: "exact"|"partial"}`:

1. `_get_field_match_sources(query, filter_conditions, field)`: scrolls Qdrant (1000 points per page, paginated until exhausted) with `MatchText(text=query_lower)` on the field, combined with the request filters (`must` from the request plus the field condition; `must_not` carried through; note `should` blocks from `any_of` live inside `must` as nested filters, so they carry through too). Each point is classified by Python substring check; the first point seen per `source_id` decides. Failures are logged and treated as no matches. The `MatchText` condition relies on the PREFIX-tokenised text index on `title` and `summary` (see [Qdrant data model](../../architecture/qdrant_data_model.md)).
2. `_supplement_matches_from_results(query, unique_source_results, field, matches)`: scans the in-memory deduplicated semantic results (the full `unique_source_results`, not just `top_k`) and adds substring matches the scroll missed (true infix hits). No Qdrant call.

Note: the match map comes from a filter-only scroll, so it includes every document whose title or summary contains the query, regardless of its semantic score.

### Phase A: boost existing results

`_apply_field_boost(top_results, matches, field, exact_boost, partial_boost)`:

```python
multiplier = exact_boost if match_type == "exact" else partial_boost
boosted = min(original * multiplier, 1.0)
result["weighted_score"] = boosted
result["field_scores"][f"{field}_match"] = match_type
result[f"{field}_multiplier"] = multiplier
```

For non-matching results it does `field_scores.setdefault("<field>_match", None)` and `result.setdefault("<field>_multiplier", 1.0)`. The list is re-sorted after each call. Title is applied first, then summary; a document matching both fields is multiplied by both multipliers (each step capped at 1.0), so the summary boost can compound with the title boost. Only the injection phase prefers title (see below).

Only `top_results` (already sliced to `top_k` before the boost) is boosted; documents ranked below `top_k` cannot be promoted by Phase A.

### Phase B: inject missing keyword matches

After both boosts, `search()` computes `present_ids` from `top_results`:

```python
missing_title   = [sid for sid in title_matches   if sid not in present_ids]
missing_summary = [sid for sid in summary_matches
                   if sid not in present_ids and sid not in title_matches]
```

Each list is resolved by `_fetch_field_match_docs(...)`, which returns one representative entry per `source_id`:

1. **Reuse pre-threshold scores.** `prethreshold_by_source` (captured in `_process_and_filter_results` before the threshold ran, best chunk per source) is checked first. Hits are copied (entry and `field_scores` are copied so the shared original is not mutated), given the floor score and tagged.
2. **Fetch the rest.** For sources never scored (absent from the candidate pool entirely), one paginated `scroll` with `MatchAny(source_id)` fetches points (page size `min(len(ids)*10, 1000)`, continuing until every source is covered or Qdrant is exhausted). If there are at most `INJECTED_DOC_SCORING_MAX` of them, vectors for the five priority fields are requested (never `True`, which would also download BM25) and per-field cosines are computed with `_cosine_similarity`; otherwise `field_scores` are `None`.
3. **Floor score.**

```python
FLOOR_SCORE = 0.15
def _score_for(match_type):
    boost = exact_boost if match_type == "exact" else partial_boost
    return min(max(FLOOR_SCORE * boost, score_floor), 1.0), boost
```

`score_floor` is `filter_score` in `filter_score` mode, or the matching field's detail threshold in detail mode (`_floor_for` in `search()`). Multiply first, then raise to the floor.

| Match | Boost | `0.15 * boost` | With `filter_score=0.5` |
|---|---|---|---|
| title exact | 2.5 | 0.375 | 0.5 |
| title partial | 1.5 | 0.225 | 0.5 |
| summary exact | 1.4 | 0.21 | 0.5 |
| summary partial | 1.2 | 0.18 | 0.5 |

With detail thresholds (title 0.36, text 0.27, ...) a title partial injected doc scores `max(0.225, 0.36) = 0.36`; a title exact scores `max(0.375, 0.36) = 0.375`.

Injected entries carry `match_source = "<field>_keyword_match"` (always surfaced in the response), `"<field>_match"` in `field_scores`, the multiplier, and either reused or recomputed per-field scores. Docs resolved via the Qdrant fetch also get a `raw_dense` value (the weighted cosine sum of scorable fields, or `None`). Docs reused from the pre-threshold pool keep their original `raw_dense`, `keyword_score` etc. but their `weighted_score` is overwritten by the floor.

Finally `top_results.sort(...)` and `top_results = top_results[:top_k]`. Injected documents bypass `filter_score` and `detail_filter_score` entirely, and can therefore appear even if semantically unrelated. `total_results` includes injected source ids.

---

## 2. `search_mode`

| Value | Effect |
|---|---|
| `"hybrid"` (default) | boost active if `HYBRID_SEARCH_ENABLED` is true |
| `"semantic"` | no title/summary scroll, supplement, boost or injection; `prethreshold_by_source` is not captured; result `title_match`/`summary_match` are `None` and multipliers are `1.0` in debug output |

Any other value is rejected with 422 (`Literal["hybrid","semantic"]`). Note that `"hybrid"` here means "with keyword boost"; it is independent of `SPARSE_SEARCH_ENABLED` (the BM25 hybrid retrieval). The chosen mode is echoed as `search_config.search_mode`.

---

## 3. Building filters

`_build_filters()` returns `models.Filter(must=..., must_not=...)` or `None`.

| Request field | Qdrant key | Match | Logic |
|---|---|---|---|
| `categories` | `tags` | `MatchAny` | OR within list |
| `organizations` | `metadata.company` | `MatchAny` | OR within list |
| `resource_type` | `metadata.DOCUMENT_TYPE` | `MatchText` per value; two or more wrapped in `Filter(should=[...])` | OR within list |
| `file_type` | `metadata.type` | `MatchAny` | OR within list |
| `exclude_organizations` | `metadata.company` | `MatchAny` in `must_not` | any listed value drops the doc |
| `exclude_file_type` | `metadata.type` | `MatchAny` in `must_not` | any listed value drops the doc |
| `any_of` (list of `FilterBlock`) | recursive | one nested `Filter(should=[branch, ...])` appended to `must` | OR between blocks, AND with everything else |

Between different fields: AND. Values are stripped and blank values dropped. A request with only exclusions is valid (`must=None`, `must_not=[...]`).

`any_of` rules (enforced by the models): at least 2 blocks (`_reject_lone_alternative`), each block needs at least one non-blank value (`FilterBlock._reject_empty_block`), blocks forbid extra fields (`extra="forbid"`, so no nesting). Each block is compiled by recursive `_build_filters` (a block cannot itself carry `any_of`).

The same `Filter` is applied to every dense request, the BM25 request, the title/summary match scrolls and the no-query scroll. The Phase B fetch by `MatchAny(source_id)` does not re-apply the request filter, but the ids it receives came from a filtered scroll or the filtered candidate pool.

---

## 4. `filter_score` versus `detail_filter_score`

Exactly one threshold mode is used per request.

**Mode 1: `filter_score`** (used when `detail_filter_score` is `None`).

```python
filter_score = max(request.filter_score, self.min_filter_score) if request.filter_score is not None else self.min_filter_score
...
filtered_results = [r for r in ranked_results if r['weighted_score'] >= threshold]
```

Compared against `weighted_score` after ranking and before boosts. `filter_score` is validated `0..1`; the default effective value is `0`, so nothing is removed by default. The threshold is echoed as `search_config.filter_score`.

**Mode 2: `detail_filter_score`** (`DetailFilterScore`, each field `0..1`, defaults title 0.36, text 0.27, tags 0.14, summary 0.14, metadata 0.09). `_apply_detail_filter` keeps a result if ANY field passes:

```python
field_score = field_score_dict.get(field, 0.0) or 0.0
if field_score >= threshold:
    passed = True
```

It compares the raw per-field cosine scores in `field_scores`; the BM25 key is ignored and `weighted_score` is not consulted (the main threshold passed in is `0`). Because a missing field counts as `0.0`, a threshold of `0.0` passes every document. `filter_score` is ignored (logged). The mode is echoed in `search_config.filter_mode` and thresholds under `search_config.detail_filter_score`.

Both thresholds are applied before the boost, so a boost cannot rescue a document below the threshold; only injection (Phase B) can re-admit one.

---

## 5. Deduplication

`_filter_best_per_source(results)` walks the score-sorted list and keeps the first result per `payload["source_id"]`, so each document appears once with its best-scoring chunk (the chunk whose `text` is returned). Results whose payload has no `source_id` are dropped. It runs after the filter and before the `top_k` slice. In Phase A the boost applies to the surviving best chunk only.

`unique_source_results` (the full deduplicated filtered list) is used for `total_results` and for the infix supplement.

---

## 6. Result building and response shape

`_build_result_items(top_results, include_scoring_debug)` produces `SearchResultItem` objects:

| Field | Source | Notes |
|---|---|---|
| `id` | `str(point id)` | chunk id |
| `text` | `payload["text"]` or `''` | in hybrid mode comes from late retrieval |
| `title`, `summary`, `tags`, `metadata`, `source_id` | payload | |
| `score` | `weighted_score` | after boosts; 0-1 in hybrid mode, raw weighted cosine (times boosts, capped) in dense-only mode |
| `field_scores` | per-field map | `title_match`/`summary_match` popped out, BM25 key removed; `None` values mean "not scored" |
| `match_source` | entry | `"title_keyword_match"`/`"summary_keyword_match"` for injected docs; not debug-gated |
| `keyword_score` | entry | raw BM25; not debug-gated |
| `title_match`, `summary_match` | `field_scores` | debug-gated |
| `rrf_score`, `dense_rank`, `sparse_rank`, `raw_dense`, `normalized_dense`, `normalized_sparse`, `title_multiplier`, `summary_multiplier` | entry | debug-gated; see [Ranking and fusion](ranking_and_fusion.md#6-scoring-debug-fields) |

An item that raises during construction is logged and skipped.

`PrioritizedSearchResponse`:

| Field | Meaning |
|---|---|
| `query` | the original request query (`None` on the no-query path) |
| `total_results` | unique sources in the filtered semantic pool plus injected sources (can exceed `len(results)` because of `top_k`) |
| `top_k` | the requested `top_k` (the no-query path returns `min(top_k, n)`) |
| `results` | ranked items |
| `search_config` | `search_fields`, `weights`, `priority_order`, `filters_applied`, `filter_mode`, `filter_score` (`None` in detail mode), `search_mode`, `hybrid_search_enabled`, `sparse_search_enabled`, `fusion_method`, optionally `detail_filter_score` and (debug) `scoring_context` |

Example (abridged, debug on, weighted fusion):

```json
{
  "query": "teacher training",
  "total_results": 42,
  "top_k": 10,
  "results": [
    {
      "id": "7f1c...", "source_id": "doc-123", "score": 1.0,
      "field_scores": {"title": 0.71, "text": 0.55, "tags": 0.40, "summary": 0.45, "metadata": 0.30},
      "keyword_score": 7.8, "raw_dense": 0.5118, "normalized_dense": 1.0, "normalized_sparse": 0.9,
      "title_match": "partial", "title_multiplier": 1.5, "summary_multiplier": 1.0
    }
  ],
  "search_config": {"filter_mode": "filter_score", "filter_score": 0.0, "search_mode": "hybrid", "fusion_method": "weighted"}
}
```

---

## Known issues / gotchas

- **Title and summary boosts can both apply** to an existing result (multiplicative, each capped). Only injection excludes summary for docs already in `title_matches`.
- **Boost scores are applied after the threshold**; injected docs bypass thresholds and can be semantically unrelated. A response where every item has `match_source` set means the threshold removed the entire semantic pool.
- **Phase A boosts only `top_k`-sliced results**; with a small `top_k`, a keyword-matching document just below the cut is not promoted but is injected only if it is missing from the final list (it is in `unique_source_results` yet not in `top_results`, so Phase B does inject it with a floor score, not its real semantic score).
- **Keyword match is contiguous substring on the whole original query.** A multi-word query with words in a different order never boosts. Short queries are the main beneficiaries.
- **`title_match`/`summary_match` are debug-gated** even though the model descriptions read as always present; `match_source` and `keyword_score` are not gated.
- **Endpoint docstring is wrong about filter logic.** `documents.py` says all filter types use OR logic ("tags OR company"); the code ANDs between filter types.
- **`resource_type` uses `MatchText`** on a TEXT-indexed field, so it is token matching (not strict substring) and the `PrioritizedSearchRequest.resource_type` description mentions a stale field name (`metadata.KEY ENTITIES`).
- **`top_k` default is 1,000,000**, effectively unbounded; `MAX_SEARCH_TOP_K` is not enforced. The candidate cap (`SEARCH_CANDIDATE_MAX`, default 2000) is the real bound.
- **Title and summary scrolls are full scrolls** over every document whose field matches the query (batched by 1000), which can be slow for very short, common queries on large collections.
- **Dedup drops points without `source_id`.**
- **The no-query path ignores all scoring options** and returns scroll-order results with `score=1.0`.

## Related pages

- [Search overview](overview.md)
- [Ranking and fusion](ranking_and_fusion.md)
- [Qdrant data model](../../architecture/qdrant_data_model.md)
- [API endpoints](../../api/endpoints.md)
- [API models](../../api/models.md)
- [Troubleshooting](../../operations/troubleshooting.md)
