# Ranking and Fusion

Read this page to understand exactly how a candidate's `weighted_score` is computed in `PrioritizedSearchService._rank_results()`: the dense-only formula, weighted min-max fusion, RRF fusion, the debug fields that expose each step, edge cases, and worked numeric examples calculated from the real formulas.

Everything here happens after candidate retrieval (see [Search overview](overview.md)) and before filtering and boosting (see [Boosts, filters and results](boosts_filters_and_results.md)).

---

## 1. Inputs to `_rank_results`

| Argument | Meaning |
|---|---|
| `all_results` | `{point_id: ScoredPoint}` candidate pool (union across fields) |
| `field_scores` | `{point_id: {field: raw_score}}`; fields are `title`, `text`, `tags`, `summary`, `metadata`, plus `bm25` (the `SPARSE_VECTOR_NAME`) in hybrid mode |
| `weights` | `settings.SEARCH_PRIORITY_WEIGHTS` |
| `search_fields` | `settings.SEARCH_PRIORITY_ORDER` |
| `scoring_context_out` | optional dict the caller passes to capture normalisation context |
| `sparse_issued` | whether the BM25 request was sent to Qdrant |

### Field weights (`app/config.py`)

| Field | Weight |
|---|---|
| `title` | 0.34 |
| `text` | 0.26 |
| `tags` | 0.20 |
| `summary` | 0.12 |
| `metadata` | 0.08 |

The weights sum to 1.0. A field a document was not retrieved by simply contributes 0 (`fs.get(field)` is `None` and is skipped); the weight is not redistributed.

Note: the endpoint docstring quotes 0.36/0.27/0.14/0.14/0.09. Those are not the weights; those numbers are the default `DetailFilterScore` thresholds.

---

## 2. Which scoring mode runs

```python
sparse_has_hits = any(sparse_name in fs for fs in field_scores.values())
is_hybrid = sparse_issued or sparse_has_hits
```

| `sparse_issued` | any doc has a `bm25` score | Mode |
|---|---|---|
| `False` | no | dense-only formula |
| `True` | yes | hybrid (`weighted` or `rrf`) |
| `True` | no (BM25 sent, zero hits) | hybrid, with the weighted branch shifting all weight to dense |
| `False` (BM25 query tokenised to nothing) | no | dense-only formula |

`sparse_issued` is `bool(sparse_indices)` in `_hybrid_batch_search`. It is `False` both when `SPARSE_SEARCH_ENABLED=false` and when the BM25 query produced no tokens (for example a query made only of stop words), so such a query silently gets dense-only scoring on the lower raw-cosine scale.

---

## 3. Dense-only path

```python
weighted_score = 0.0
for field in search_fields:
    if field in field_score_dict and field in weights:
        weighted_score += field_score_dict[field] * weights[field]
```

`weighted_score = 0.34*title + 0.26*text + 0.20*tags + 0.12*summary + 0.08*metadata` using raw cosine scores. The result is not min-max normalised and is not capped at 1.0 here (the boost step caps later). A realistic value is well below 1.0 because a single query vector rarely scores high on all five fields. In this mode `raw_dense` equals `weighted_score`.

---

## 4. Hybrid path

Step one is shared by both fusion methods:

```python
raw_dense[point_id]  = sum(score * weights[field] for each dense field present)
raw_sparse[point_id] = fs.get(sparse_name) or 0.0
```

`raw_sparse` has an entry (value 0.0) for every candidate, including those with no BM25 hit.

### 4.1 `_min_max_normalize`

```python
lo, hi = min(values), max(values)
if hi <= lo:
    return {k: (1.0 if v > 0 else 0.0) for k, v in scores.items()}
return {k: (v - lo) / (hi - lo) for k, v in scores.items()}
```

Because zeros for non-hit documents are included in `raw_sparse`, the sparse minimum is 0.0 whenever at least one candidate has no BM25 hit.

### 4.2 Mode A: `weighted` (default, `HYBRID_FUSION_METHOD=weighted`)

```text
norm_dense  = minmax(raw_dense)
norm_sparse = minmax(raw_sparse)
weighted_score = dense_w * norm_dense + sparse_w * norm_sparse
```

| Condition | `dense_w`, `sparse_w` |
|---|---|
| some candidate has a BM25 score | `HYBRID_DENSE_WEIGHT` (0.7), `HYBRID_SPARSE_WEIGHT` (0.3) |
| BM25 was issued but no candidate has a score | `1.0`, `0.0` (logged: `No sparse hits in candidate pool; dense weight raised to 1.0`) |

The weights are validated at startup (`Settings._validate_fusion_config`): finite, non-negative, and `dense + sparse <= 1.0`. They are not required to sum to exactly 1.0, so a configuration such as 0.6/0.2 caps the best possible score below 1.0.

### 4.3 Mode B: `rrf` (`HYBRID_FUSION_METHOD=rrf`)

RRF fuses exactly two lists: the combined dense list and the sparse list.

```python
dense_hits  = {pid: s for pid, s in raw_dense.items()  if s > 0.0}
dense_rank  = self._rank_positions(dense_hits)
sparse_hits = {pid: s for pid, s in raw_sparse.items() if s > 0.0}
sparse_rank = self._rank_positions(sparse_hits)
fused = 1/(RRF_K + dense_rank[pid]) (if ranked) + 1/(RRF_K + sparse_rank[pid]) (if ranked)
hybrid_scores = self._min_max_normalize(rrf_raw)
```

- `RRF_K` defaults to 60.
- Only documents with `raw_dense > 0` get a dense rank and only documents with `raw_sparse > 0` get a sparse rank.
- Ranks are 1-indexed; ties are broken by sort stability (dictionary insertion order).
- The min-max step rescales the raw RRF (about 0.016 to 0.033 for k=60) to 0-1 so `filter_score` stays meaningful. The top document always becomes exactly 1.0 and the bottom exactly 0.0 (unless the pool is flat).
- In RRF mode `dense_w`/`sparse_w` are not used and not reported; `HYBRID_DENSE_WEIGHT` and `HYBRID_SPARSE_WEIGHT` have no effect.

### 4.4 Why scores are batch-relative

Min-max normalisation is computed over the candidate pool of this single query. The same document can score differently for different queries, filters or `top_k` values (the pool size follows `_candidate_limit`). In the weighted mode the best dense candidate always gets `norm_dense = 1.0` and the worst 0.0. A pool of one document therefore yields `norm_dense = 1.0` (if its raw score is positive), regardless of how weak the match is.

---

## 5. What each result entry carries

`_rank_results` returns entries sorted by `weighted_score` descending:

| Key | Content |
|---|---|
| `id`, `payload` | point id and payload |
| `weighted_score` | final fused score (dense-only: raw weighted cosine; hybrid: calibrated 0-1) |
| `field_scores` | per-field raw scores (may include `bm25`) |
| `num_fields_matched` | count of keys in `field_scores` other than the sparse key (computed, not used elsewhere) |
| `raw_dense` | pre-fusion weighted cosine sum (both modes) |
| hybrid only: `keyword_score` | raw BM25 (`raw_sparse`; 0.0 if no hit) |
| hybrid only: `rrf_score`, `dense_rank`, `sparse_rank` | `None` unless the `rrf` branch ran (and `sparse_rank` `None` for docs without a sparse hit) |
| hybrid only: `normalized_dense`, `normalized_sparse` | `None` unless the `weighted` branch ran |

---

## 6. Scoring-debug fields

`search_config.fusion_method` is always reported. Per-result fields on `SearchResultItem` are set in `_build_result_items`:

| Field | Gated by `include_scoring_debug`? | Populated when |
|---|---|---|
| `keyword_score` | No, always surfaced when the entry has it | hybrid path |
| `rrf_score` | Yes | hybrid, `rrf` mode |
| `dense_rank` | Yes | hybrid, `rrf` mode |
| `sparse_rank` | Yes | hybrid, `rrf` mode, doc has a sparse hit |
| `raw_dense` | Yes | always (both modes) when debug is on |
| `normalized_dense`, `normalized_sparse` | Yes | hybrid, `weighted` mode |
| `title_multiplier`, `summary_multiplier` | Yes | default `1.0` when no boost applied |
| `title_match`, `summary_match` | Yes | boost active |
| `match_source` | No | only on keyword-injected docs |

`include_scoring_debug` defaults to `settings.INCLUDE_SCORING_DEBUG` (env, default `false`) and may be overridden per request. When on, `search_config.scoring_context` is added:

| Key | When present |
|---|---|
| `candidate_pool_size` | always |
| `sparse_issued`, `sparse_has_hits` | always |
| `dense_min`, `dense_max`, `sparse_min`, `sparse_max` | hybrid only (both methods) |
| `dense_weight`, `sparse_weight` | `weighted` branch only (the weights actually applied) |
| `boost_config` | added in `search()`: the four boost multipliers |

Absence of `dense_weight`/`sparse_weight` therefore means: dense-only, or RRF. Use these min/max values to reproduce `normalized_*` by hand: `(raw - min) / (max - min)`.

---

## 7. Edge cases

| Case | Behaviour |
|---|---|
| Single candidate, `raw_dense > 0` | `norm_dense = 1.0` |
| Flat pool (all scores equal and positive) | every value normalises to 1.0; equal zeros normalise to 0.0 |
| Candidate not retrieved by any BM25 hit | `raw_sparse = 0.0`; `norm_sparse` is 0.0 (when the sparse min is 0, which it is whenever some candidate lacks a hit) |
| BM25 issued, zero hits | weighted: `dense_w=1.0`, `sparse_w=0.0`; rrf: only dense ranks contribute, so the best dense doc is 1.0 and the worst is 0.0 |
| BM25 query with no tokens | `sparse_issued=False`: dense-only raw cosine scale (scores noticeably lower; a `filter_score` tuned for hybrid may then remove everything) |
| `raw_dense == 0` for a doc (RRF) | no dense rank; only its sparse term, if any |
| RRF both terms missing | `rrf_raw = 0.0`, normalises to the pool minimum |
| `HYBRID_FUSION_METHOD` invalid | app refuses to start (`ValueError` in settings validation) |
| Boost applied afterwards | multiplies the fused score and caps at 1.0; ranking-time scores are deliberately not capped |

---

## 8. Worked examples

All numbers below are computed from the formulas above using the default weights.

### 8.1 Dense-only

A chunk with raw cosines: title 0.62, text 0.55, tags 0.40, summary 0.45, metadata 0.30.

```text
0.34*0.62 = 0.2108
0.26*0.55 = 0.1430
0.20*0.40 = 0.0800
0.12*0.45 = 0.0540
0.08*0.30 = 0.0240
weighted_score = 0.5118
```

If the chunk was retrieved only by `text` and `title`, the other three terms are 0: `0.2108 + 0.1430 = 0.3538`.

### 8.2 Hybrid, `weighted`

Candidate pool of three documents. `raw_dense`: A 0.60, B 0.50, C 0.40. BM25: A 8.0, B 12.0, C no hit (`raw_sparse = 0.0`).

```text
norm_dense  : A = (0.60-0.40)/0.20 = 1.0000   B = 0.5000   C = 0.0000
norm_sparse : min=0, max=12 -> A = 8/12 = 0.6667   B = 1.0000   C = 0.0000
score = 0.7*norm_dense + 0.3*norm_sparse
A = 0.7*1.0 + 0.3*0.6667 = 0.9000
B = 0.7*0.5 + 0.3*1.0    = 0.6500
C = 0.0
```

Document C scores 0.0 even though its raw dense score is a respectable 0.40; with the default `filter_score` of 0 it is still kept (`0.0 >= 0`), but any positive `filter_score` removes it.

With zero BM25 hits in the pool (`dense_w=1.0`): A = 1.0, B = 0.5, C = 0.0.

### 8.3 Hybrid, `rrf` (k = 60)

Same pool but BM25 A 12.0, B 8.0 (C no hit). Dense ranks: A1, B2, C3. Sparse ranks: A1, B2.

```text
A = 1/61 + 1/61 = 0.0327869
B = 1/62 + 1/62 = 0.0322581
C = 1/63        = 0.0158730          (dense term only)
min-max: min=0.0158730, max=0.0327869, span=0.0169139
A = 1.0000
B = (0.0322581-0.0158730)/0.0169139 = 0.9687
C = 0.0000
```

Note how RRF compresses differences: B is almost tied with A despite a dense gap that weighted mode scores as 0.5 versus 1.0. Also note that with ranks A (dense 1, sparse 2) and B (dense 2, sparse 1) both documents get `1/61 + 1/62 = 0.0325` and tie exactly.

### 8.4 Boost on top of a fused score

After ranking, a partial title match multiplies by 1.5 and an exact one by 2.5, each capped at 1.0 (see [Boosts, filters and results](boosts_filters_and_results.md#1-title-and-summary-boost)):

```text
0.62 * 1.5 = 0.93       (partial title match)
0.75 * 2.5 = 1.875 -> 1.0  (exact title match, capped)
0.93 * 1.2 = 1.116 -> 1.0  (then partial summary match, capped)
```

---

## Known issues / gotchas

- **Weights and `keyword_score`:** the code uses 0.34/0.26/0.20/0.12/0.08 for ranking (0.36/0.27/0.14/0.14/0.09 are the `DetailFilterScore` thresholds). `keyword_score` does not need `include_scoring_debug`; it is returned by default whenever the hybrid path produced it.
- **`rrf_score`, `dense_rank`, `sparse_rank` are `None` in `weighted` mode, and `normalized_*` are `None` in `rrf` mode.** The `SearchResultItem` field descriptions for `dense_rank`/`sparse_rank` do not mention this mode dependence.
- **Zero-token BM25 query silently changes the score scale** (dense-only raw cosines instead of 0-1 fused scores), which can make a fixed `filter_score` behave very differently from query to query.
- **The `_hybrid_batch_search` docstring** implies zero-hit sparse queries stay on the hybrid scale; this is true only when the BM25 request was actually sent (tokens existed).
- **Scores are pool-relative:** `top_k`, filters and `SEARCH_CANDIDATE_MAX` change min/max and therefore the scores of the same document.
- **`HYBRID_DENSE_WEIGHT`/`HYBRID_SPARSE_WEIGHT` need not sum to 1.0**, only `<= 1.0`, so the maximum fused score can be below 1.0.
- **`METADATA_MATCH_BOOST` is defined but never applied.**

## Related pages

- [Search overview](overview.md)
- [Boosts, filters and results](boosts_filters_and_results.md)
- [Configuration](../../setup/configuration.md)
- [API models](../../api/models.md)
- [Troubleshooting](../../operations/troubleshooting.md)
