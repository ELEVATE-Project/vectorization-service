# Auxiliary Services

**Purpose.** Read this page to understand the four smaller services that sit beside the main search and ingestion pipelines: simple text search, duplicate-content check, source-ID verification and the legacy multilingual query service. For each: purpose, algorithm, thresholds and known bugs.

| Service | File | Route | Status |
|---|---|---|---|
| `TextEmbeddingSearchService` | `app/services/text_embedding_search_service.py` | `POST /api/documents/text-search` | working |
| `SimilarityService` | `app/services/similarity_service.py` | `POST /api/documents/check-similarity` | working |
| `SourceVerificationService` | `app/services/source_verification_service.py` | `POST /api/documents/verify-sources` | **broken** (always `not_found`) |
| `QueryService` | `app/services/query_service.py` | `POST /api/query/` | **broken** (un-awaited coroutine) |

All four query the `documents` collection (`settings.COLLECTION_NAME`) through the shared `qdrant_client` singleton and, for the first two and `QueryService`, embed the query with `embedding.embed_query()` (validated `list[float]`; raises `EmbeddingError` mapped to HTTP 422).

## 1. TextEmbeddingSearchService

**Purpose.** Minimal semantic search against the `text` named vector only. No filters, no boosts, no BM25, no field weighting.

Algorithm (`search`):

1. `query_vector = embedding.embed_query(request.query)` (the raw query; no spaCy preprocessing).
2. `qdrant_client.query_points(collection_name, query=query_vector, using="text", limit=request.top_k, with_payload=True)`.
3. Iterate `search_results.points`; skip points without `payload["source_id"]`; skip points with `score < request.threshold`.
4. Build `TextSearchResultItem(source_id, text, score, metadata)` per point and return `TextSearchResponse(query, total_results, results)`.

| Parameter | Default | Source |
|---|---|---|
| `top_k` | 10 | `TextSearchRequest` (a chunk limit, see below) |
| `threshold` | 0.40 | `TextSearchRequest` literal; `settings.SIMILARITY_THRESHOLD` (0.40) is **not** read here |

Any exception is logged and re-raised unchanged (so `EmbeddingError` becomes 422, anything else a plain 500).

Known issues:

- The endpoint docstring says "top chunk per unique document / grouped by source_id". The code does **no** grouping: the result is the top `top_k` chunks above threshold, so one document can return several chunks and fewer documents than `top_k`.
- The threshold is applied client-side after Qdrant has already limited to `top_k`; a high threshold therefore cannot pull in lower-ranked-but-above-threshold hits (there are none, since results are score-sorted) but also means `top_k` is spent on chunks, not documents.
- Docstrings say default `top_k` is 5; the model default is 10.

## 2. SimilarityService

**Purpose.** Pre-ingestion duplicate detection: "does this company already have content semantically close to this text?"

Algorithm (`check_similarity`):

1. Embed the first **1000 characters** of `request.text` (`embedding.embed_query(request.text[:1000])`).
2. Build the filter:

```python
filter_conditions = [models.FieldCondition(key="metadata.company",
                                           match=models.MatchValue(value=request.company_id))]
# optional
must_not_conditions.append(models.FieldCondition(key="source_id",
                           match=models.MatchValue(value=request.exclude_source_id)))
search_filter = models.Filter(must=filter_conditions, must_not=must_not_conditions or None)
```

3. `query_points(using="text", limit=5, query_filter=search_filter, score_threshold=request.threshold, with_payload=True)`.
4. Map each hit to `{source_id, similarity_score, metadata, text_preview, chunk_id}` where `text_preview = payload["text"][:200] + "..."` and `chunk_id = str(hit.id)`.
5. `has_similar = len(similar_docs) > 0`.

| Parameter | Default | Notes |
|---|---|---|
| `threshold` | 0.85 | `SimilarityCheckRequest.threshold`; passed straight to Qdrant `score_threshold` |
| result limit | 5 | hard-coded |
| text window | 1000 chars | hard-coded |

Error handling: `EmbeddingError` is re-raised (422); any other exception becomes `HTTPException(500, "Similarity check failed: ...")`.

Known issues / notes:

- Results are chunk-level: the same document can fill several of the 5 slots.
- `text_preview` always ends with `"..."`, even for chunks shorter than 200 characters.
- `self.threshold_default = 0.85` in `__init__` is never used.
- `find_duplicates_in_collection(company_id)` is a stub that returns `[]`; it is not routed.
- The `exclude_source_id` handling uses `must_not`; the earlier `invert` flag was silently ignored by Qdrant (documented in the code comment).
- Only the `text` vector of 384 dims is compared, so a long document is matched only on a chunk basis against its leading 1000 characters.

## 3. SourceVerificationService

**Purpose.** Given a list of `source_id` values, report which exist in Qdrant. Exposed as the module singleton `source_verification_service`.

Algorithm (`verify_sources`):

1. Empty list: return an all-zero response.
2. De-duplicate preserving order: `unique_source_ids = list(dict.fromkeys(request.source_ids))`. `total_requested` is the **unique** count, not the raw input length.
3. For each ID, serially: `qdrant_client.scroll(collection_name, scroll_filter=Filter(must=[FieldCondition(key="metadata.source_id", match=MatchValue(value=source_id))]), limit=1, with_payload=False, with_vectors=False)`; found if `result[0]` is non-empty.
4. Partition into `found` / `not_found`, preserving order.

Known bugs:

- **Wrong payload path (functional bug).** The filter key is `metadata.source_id`, but ingestion stores `source_id` at the **top level** of the payload (and the indexed key is `source_id`). Unless a document also carries a `source_id` inside its `metadata` object, every ID is reported `not_found`. Fix: use `key="source_id"` (indexed KEYWORD) or a single `MatchAny` filter.
- **O(N) round trips.** One scroll per ID in a serial loop. A single scroll with `MatchAny(any=ids)` over `with_payload=["source_id"]` would be sufficient.
- **Errors are swallowed per ID.** The inner `except Exception` logs and continues, so a Qdrant outage makes every ID `not_found` and still returns HTTP 200 (only errors outside the loop produce 500).

## 4. QueryService (legacy multilingual query)

**Purpose.** Older query path used before the prioritized search existed. Exposed via `POST /api/query/` in `app/api/v1/endpoints/query.py`:

```python
query_service = QueryService()

@router.post("/")
async def query_documents(request: MultilingualQueryRequest) -> MultilingualQueryResponse:
    return query_service.process_query(request)
```

`process_query` is `async def`, so the endpoint returns an un-awaited coroutine. FastAPI then fails to validate it as `MultilingualQueryResponse` and the request ends in a 500. The test suite avoids the problem by patching `app.api.v1.endpoints.query.query_service` (`tests/conftest.py :: mock_query_service`). This route has never worked end to end in this code state.

Intended flow (`process_query`):

1. Cache key `f"{request.query}_{request.priority_filter}"`; `redis_cache.get(...)`. (Cache is a no-op when `REDIS_CACHE_ENABLED` is false, which is hardcoded.)
2. `await ensure_collections_exist()` on every request.
3. `_process_query_language`: translation is **disabled**; always returns `{"original": query, "translated": None, "language": "en"}`.
4. `embedding.embed_query(search_query)`.
5. `_search_documents`:
   - with `priority_filter`: one `query_points(using="text", query_filter=metadata.priority == PRIORITY.upper(), score_threshold=settings.SIMILARITY_THRESHOLD, limit=search_limit)`;
   - without: loop over `["P1","P2","P3"]`, each limited to the remaining budget (`remaining_limit -= len(results)`), stopping at 0. Documents with other priority values are never returned.
6. `_process_search_results`: opens a `SessionLocal()` Postgres session and, for every hit, calls `_find_translation_record` (three lookups on `TranslationRecord.chunk_id`: exact, stripped, case-insensitive, then logs a sample of 5 IDs), so **a live PostgreSQL connection is required per request** even though translation is disabled.
7. `_process_content_translation` decides `display_text` / `translated_text` from `metadata["is_hindi"]` and the record's `original_text`.
8. Each result is `dict(hit.payload)` plus `qdrant_recommendation_text`, `translated_text`, `relevance_score`, `metadata`, `priority`, `chunk_id` (point id with dashes removed).
9. `_cache_response` stores a trimmed copy only if `REDIS_CACHE_ENABLED`.

| Setting | Default | Use |
|---|---|---|
| `SIMILARITY_THRESHOLD` | 0.40 | `score_threshold` on every query |
| `VECTOR_SEARCH_LIMIT` | 1 | default `search_limit` in the request model |

Known issues:

- Un-awaited coroutine (above).
- Per-chunk failures are logged and the chunk is skipped silently.
- `hit.payload["metadata"]` / `["text"]` use direct indexing; a point missing either key is skipped via the inner `except`.
- Priority fan-out relies on `metadata.priority` being exactly `P1`/`P2`/`P3` (uppercased at upload).
- The `chunk_id` used for translation lookup strips dashes, whereas `TranslationRecord` is not currently populated (see [App lifecycle](../backend/app_lifecycle.md)), so lookups always miss.
- Superseded functionally by `POST /api/documents/search`.

## Related pages

- [API endpoints](../api/endpoints.md)
- [API models](../api/models.md)
- [Search overview](search/overview.md)
- [App lifecycle](../backend/app_lifecycle.md)
- [Utils](../backend/utils.md)
