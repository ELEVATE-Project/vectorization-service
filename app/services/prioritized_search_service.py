import logging
import time
from typing import List, Dict, Optional, Any, Set
import numpy as np
from qdrant_client import models
from qdrant_client.models import QueryRequest
from app.core.clients.qdrant import qdrant_client
# Imported as a module (not `from ... import embed_query`) so a single patch point
# `app.core.clients.embedding.embed_query` works in tests, and so the validated query
# helpers are always resolved through the canonical reference.
from app.core.clients import embedding
from app.core.clients.embedding import EmbeddingError
from app.config import settings
from app.models.api_models import (
    PrioritizedSearchRequest,
    PrioritizedSearchResponse,
    SearchResultItem
)

logger = logging.getLogger(__name__)


def _match_any_condition(key: str, values: Optional[List[str]]) -> Optional[models.FieldCondition]:
    """One MatchAny FieldCondition over the non-blank values, or None.

    Generic over the payload key, so any exact-match field can be filtered the
    same way. Values are stripped and blanks dropped, matching the positive
    filter blocks in _build_filters.
    """
    valid = [value.strip() for value in (values or []) if value and value.strip()]
    if not valid:
        return None
    return models.FieldCondition(key=key, match=models.MatchAny(any=valid))


class PrioritizedSearchService:
    """
    Service for prioritized multi-field vector search with intelligent filtering.
    
    Features:
    - Multi-field vector search across title, tags, summary, metadata, and content
    - Configurable field weights and priority ordering
    - Advanced filtering with AND/OR logic combinations
    - Automatic deduplication by source_id
    - Score-based ranking with multi-field bonuses
    
    Search Priority (configurable in settings):
    1. Title - Highest priority
    2. Tags - High priority  
    3. Summary - Medium-high priority
    4. Metadata - Medium priority
    5. Text - Base priority
    """
    
    def __init__(self):
        self.collection_name = settings.COLLECTION_NAME
        self.default_top_k = settings.DEFAULT_SEARCH_TOP_K
        self.max_top_k = settings.MAX_SEARCH_TOP_K
        self.min_filter_score = settings.MIN_SEARCH_FILTER_SCORE
        self.priority_order = settings.SEARCH_PRIORITY_ORDER
        self.default_weights = settings.SEARCH_PRIORITY_WEIGHTS
        self.min_score_threshold = settings.MIN_WEIGHTED_SCORE_THRESHOLD

    def _candidate_limit(self, top_k: int) -> int:
        """Per-field candidate pool size for multi-field search.

        Each of the dense named-vector searches and the sparse BM25 search retrieves
        this many candidates; the union is fused/ranked. The CAP bounds HNSW ``ef``
        (the dominant query cost) so a large top_k can't trigger a 10k-deep traversal
        per field; the FANOUT gives small-top_k callers a re-ranking margin. The union
        across fields still fills top_k after source-level dedup. Env-tunable via
        SEARCH_CANDIDATE_FANOUT / SEARCH_CANDIDATE_MAX. For top_k=1000 → 2000 (was 10000).
        """
        return min(max(top_k, 1) * settings.SEARCH_CANDIDATE_FANOUT, settings.SEARCH_CANDIDATE_MAX)

    def _log_search_request(self, request: PrioritizedSearchRequest, top_k: int, filter_conditions):
        """Log search request details"""
        logger.info("========== SEARCH REQUEST ==========" )
        logger.info(f"Query: '{request.query}'")
        logger.info(f"Top K: {top_k}")
        if filter_conditions:
            logger.info("Filters applied:")
            if request.categories:
                logger.info(f"  - Categories: {request.categories}")
            if request.organizations:
                logger.info(f"  - Organizations: {request.organizations}")
            if request.resource_type:
                logger.info(f"  - Resource Types: {request.resource_type}")
            if request.file_type:
                logger.info(f"  - File Types: {request.file_type}")
            if request.exclude_organizations:
                logger.info(f"  - Excluded Organizations: {request.exclude_organizations}")
            if request.exclude_file_type:
                logger.info(f"  - Excluded File Types: {request.exclude_file_type}")
        else:
            logger.info("No filters applied")

    def _process_and_filter_results(self, all_results, field_scores, weights, search_fields, top_k, threshold, detail_filter_score=None, scoring_context_out=None, prethreshold_by_source_out=None):
        """Process, rank, filter and deduplicate results.

        scoring_context_out: optional mutable dict forwarded to _rank_results so the
        caller can capture the per-query normalization context (min/max/pool size).

        prethreshold_by_source_out: optional dict, filled with {source_id: best result}
        before the threshold removes anything. Lets the keyword-match step below reuse
        real scores. Pass None to skip it (semantic search, which never needs it).
        """
        logger.info(f"Total documents matched: {len(all_results)}")

        logger.info("Calculating weighted scores and ranking results")
        ranked_results = self._rank_results(all_results, field_scores, weights, search_fields, scoring_context_out)
        logger.info(f"Ranked results: {len(ranked_results)} documents")

        # Save the best chunk per source before filtering, so documents the threshold
        # removes still have their real scores available later. Results are already
        # sorted, so the first one we see per source is the best one.
        if prethreshold_by_source_out is not None:
            for entry in self._filter_best_per_source(ranked_results):
                prethreshold_by_source_out[entry['payload'].get('source_id')] = entry
            logger.info(
                f"Captured {len(prethreshold_by_source_out)} pre-threshold sources "
                f"for keyword-injection score reuse"
            )

        # Apply filtering based on conditions
        if detail_filter_score is not None:
            logger.info("Applying detail_filter_score (field-level thresholds with OR logic)")
            filtered_results = self._apply_detail_filter(ranked_results, detail_filter_score)
        else:
            logger.info(f"Applying filter_score threshold: {threshold}")
            filtered_results = [r for r in ranked_results if r['weighted_score'] >= threshold]
        
        logger.info(f"After filtering: {len(filtered_results)} documents (removed {len(ranked_results) - len(filtered_results)})")
        
        logger.info("Deduplicating by source_id (keeping best match per source)")
        unique_source_results = self._filter_best_per_source(filtered_results)
        logger.info(f"Unique sources: {len(unique_source_results)}")
        
        top_results = unique_source_results[:top_k]
        logger.info(f"Returning top {len(top_results)} results")
        
        # Return unique_source_results for total_results to show unique sources count
        return top_results, unique_source_results

    def _build_result_items(self, top_results, include_scoring_debug: bool = False) -> List[SearchResultItem]:
        """Build result items from top results.

        Strips the internal sparse (BM25) scoring key (SPARSE_VECTOR_NAME) before
        serializing to the API response — it is a ranking internal, not a per-field
        cosine similarity. Only float scores corresponding to actual dense vector
        fields (title, tags, summary, metadata, text) are surfaced to the client.

        For documents injected via _fetch_field_match_docs (keyword/text-match only),
        field scores are None — meaning no vector similarity was computed for that
        field — rather than 0.0, which would be misleading.

        When include_scoring_debug is True, the hybrid fusion breakdown stashed by
        _rank_results (keyword_score, rrf_score, dense_rank, sparse_rank) is surfaced
        on each item; otherwise those stay None to keep responses lean.
        """
        # Key used internally for ranking (raw BM25 score) but must not appear in
        # the public field_scores contract.
        INTERNAL_SCORE_KEYS = {settings.SPARSE_VECTOR_NAME}

        result_items = []
        for result_data in top_results:
            try:
                field_scores = dict(result_data['field_scores'])
                # Extract match types so they surface as top-level fields and
                # field_scores stays a pure {field: float | None} map.
                title_match = field_scores.pop("title_match", None)
                summary_match = field_scores.pop("summary_match", None)

                # Remove the internal BM25 score key — it is fusion mechanics,
                # not a per-field cosine similarity score.
                for key in INTERNAL_SCORE_KEYS:
                    field_scores.pop(key, None)

                result_items.append(SearchResultItem(
                    id=str(result_data['id']),
                    text=result_data['payload'].get('text', ''),
                    title=result_data['payload'].get('title'),
                    summary=result_data['payload'].get('summary'),
                    tags=result_data['payload'].get('tags'),
                    metadata=result_data['payload'].get('metadata', {}),
                    source_id=result_data['payload'].get('source_id', ''),
                    score=result_data['weighted_score'],
                    field_scores=field_scores,
                    # title_match/summary_match are part of the scoring breakdown, so they
                    # are debug-gated like the other breakdown fields — surfaced only when
                    # include_scoring_debug is set (None otherwise keeps the field out of
                    # the backend response, whose serializer omits None debug keys).
                    title_match=title_match if include_scoring_debug else None,
                    summary_match=summary_match if include_scoring_debug else None,
                    # keyword_score is surfaced BY DEFAULT whenever hybrid/sparse search
                    # produced a BM25 score (it is only set on the entry in the hybrid path,
                    # so it stays None in dense-only mode) — NOT debug-gated, unlike the
                    # other breakdown fields.
                    keyword_score=result_data.get('keyword_score'),
                    rrf_score=result_data.get('rrf_score') if include_scoring_debug else None,
                    dense_rank=result_data.get('dense_rank') if include_scoring_debug else None,
                    sparse_rank=result_data.get('sparse_rank') if include_scoring_debug else None,
                    raw_dense=result_data.get('raw_dense') if include_scoring_debug else None,
                    normalized_dense=result_data.get('normalized_dense') if include_scoring_debug else None,
                    normalized_sparse=result_data.get('normalized_sparse') if include_scoring_debug else None,
                    # Multipliers default to 1.0 (no boost) so debug output always carries
                    # them, even for docs the boost pass didn't touch (e.g. keyword-injected).
                    title_multiplier=result_data.get('title_multiplier', 1.0) if include_scoring_debug else None,
                    summary_multiplier=result_data.get('summary_multiplier', 1.0) if include_scoring_debug else None,
                ))
            except Exception as e:
                logger.warning(f"Failed to parse result item {result_data.get('id')}: {str(e)}")
                continue
        return result_items

    def search(self, request: PrioritizedSearchRequest) -> PrioritizedSearchResponse:
        """
        Execute prioritized multi-field search with optional filtering.
        
        Behavior:
        - With query: Performs vector search across all fields with weighted ranking
        - Without query: Returns unique source documents matching filters
        
        Filtering Conditions (Simplified):
        1. If detail_filter_score is provided (NOT null):
           Apply field-level thresholds with OR logic (any field meeting threshold passes)
           Note: filter_score is ignored when detail_filter_score is provided
        2. If detail_filter_score is null:
           Apply filter_score as weighted score threshold
        
        Args:
            request: Search request containing query, top_k, and optional filters
                    (categories, organizations, resource_types, file_types)
            
        Returns:
            PrioritizedSearchResponse with ranked, deduplicated results
        """
        try:
            if not request.query or not request.query.strip():
                return self._get_unique_source_documents(request)
            
            if request.top_k <= 0:
                raise ValueError("top_k must be greater than 0")
            
            # Use request.top_k directly without any limit
            top_k = request.top_k

            # Determine filtering strategy
            # Simple logic: If detail_filter_score is provided, use it. Otherwise use filter_score.
            use_detail_filter = False
            detail_filter_score = None
            filter_score = self.min_filter_score
            
            # Condition 1: detail_filter_score is provided (NOT null)
            if request.detail_filter_score is not None:
                use_detail_filter = True
                detail_filter_score = request.detail_filter_score
                logger.info("Filter mode: DETAIL_FILTER_SCORE (field-level thresholds with OR logic)")
                logger.info(f"Detail thresholds: title={detail_filter_score.title}, "
                           f"text={detail_filter_score.text}, tags={detail_filter_score.tags}, "
                           f"summary={detail_filter_score.summary}, metadata={detail_filter_score.metadata}")
                if request.filter_score is not None:
                    logger.info(f"Note: filter_score={request.filter_score} is ignored when detail_filter_score is provided")
            
            # Condition 2: detail_filter_score is None (use filter_score)
            else:
                use_detail_filter = False
                filter_score = (
                    max(request.filter_score, self.min_filter_score)
                    if request.filter_score is not None
                    else self.min_filter_score
                )
                logger.info(f"Filter mode: FILTER_SCORE (weighted score threshold = {filter_score})")

            search_fields = self.priority_order
            weights = self.default_weights
            
            # Preprocess query for improved search quality
            from app.utils.query_preprocessor import preprocess_query
            
            logger.info(f"Original query: '{request.query}'")
            preprocessed_query = preprocess_query(request.query)
            logger.info(f"Preprocessed query: '{preprocessed_query}'")
            
            # Use preprocessed query for embedding generation
            # Fallback to original if preprocessing returns empty
            query_for_embedding = preprocessed_query if preprocessed_query.strip() else request.query

            # Title/summary boost is a KEYWORD substring match, not a semantic match: it must
            # use the ORIGINAL query. The preprocessed query drops stop-words, which breaks the
            # contiguous-substring check in _classify_text_match when a stop-word sits between
            # content words (e.g. "ministry of education" → "ministry education"). See CLAUDE.md §15.
            query_for_keyword_match = request.query

            logger.info(f"Generating embedding for query: '{query_for_embedding}'")
            # embed_query rejects empty/whitespace input and validates the produced
            # vector (length == EMBEDDING_DIM, finite values) so a malformed vector can
            # never reach Qdrant. Returns a validated list[float].
            try:
                query_embedding = embedding.embed_query(query_for_embedding)
            except EmbeddingError as e:
                logger.error(
                    "Query embedding invalid: service=prioritized_search "
                    f"query='{query_for_embedding[:80]}' expected_dim={embedding.EMBEDDING_DIM} "
                    f"error={e}"
                )
                raise
            
            filter_conditions = self._build_filters(
                categories=request.categories,
                organizations=request.organizations,
                resource_types=request.resource_type,
                file_types=request.file_type,
                exclude_organizations=request.exclude_organizations,
                exclude_file_types=request.exclude_file_type,
                any_of=request.any_of
            )
            
            self._log_search_request(request, top_k, filter_conditions)
            
            logger.info("========== EXECUTING SEARCH ==========" )
            if settings.SPARSE_SEARCH_ENABLED:
                logger.info("Starting hybrid batch search (dense + BM25 sparse, RRF fusion)")
                all_results, field_scores = self._hybrid_batch_search(
                    search_fields=search_fields,
                    query_text=query_for_embedding,
                    query_embedding=query_embedding,
                    filter_conditions=filter_conditions,
                    # Bounded candidate pool per field — keeps HNSW ef small (dominant
                    # query cost). See _candidate_limit / SEARCH_CANDIDATE_* config.
                    limit=self._candidate_limit(top_k),
                )
            else:
                logger.info("Starting parallel batch search across all fields")
                all_results, field_scores = self._parallel_batch_search(
                    search_fields=search_fields,
                    weights=weights,
                    query_embedding=query_embedding,
                    filter_conditions=filter_conditions,
                    # Bounded candidate pool per field (see note above).
                    limit=self._candidate_limit(top_k),
                )
            
            if not all_results:
                logger.info("========== SEARCH RESULTS ==========" )
                logger.info("No results found")
                return PrioritizedSearchResponse(
                    query=request.query,
                    total_results=0,
                    top_k=top_k,
                    results=[],
                    search_config={
                        "search_fields": search_fields,
                        "weights": weights,
                        "priority_order": self.priority_order,
                        "filters_applied": filter_conditions is not None,
                        "filter_mode": "detail_filter_score" if use_detail_filter else "filter_score"
                    }
                )
            
            # "hybrid" (default) applies the title/summary boost below; "semantic" skips
            # it. Checked here, before ranking, so semantic search can also skip the
            # extra bookkeeping that only the boost step needs.
            search_mode = getattr(request, "search_mode", "hybrid")
            boost_active = settings.HYBRID_SEARCH_ENABLED and search_mode != "semantic"

            # Pass detail_filter_score if using field-level filtering.
            # scoring_context captures the per-query normalization reference (min/max/pool)
            # computed inside _rank_results; surfaced under search_config.scoring_context
            # when include_scoring_debug is set (see below).
            scoring_context: Dict[str, Any] = {}
            # Only needed by the boost step below, so semantic search skips it.
            prethreshold_by_source: Optional[Dict[str, Any]] = {} if boost_active else None
            top_results, unique_source_results = self._process_and_filter_results(
                all_results, field_scores, weights, search_fields, top_k,
                filter_score if not use_detail_filter else 0,
                detail_filter_score if use_detail_filter else None,
                scoring_context_out=scoring_context,
                prethreshold_by_source_out=prethreshold_by_source,
            )

            # Source_ids injected by the title/summary boost below (filtered out of the
            # semantic pool). Tracked so total_results counts them — otherwise the
            # response could report fewer total than it returns.
            injected_source_ids: set = set()
            if boost_active:
                logger.info("Applying hybrid title + summary boost")

                # Title boost (highest priority). Scroll retrieves prefix/partial
                # matches; the supplement adds any mid/infix matches already present
                # in the dense candidate pool that the scroll missed.
                title_matches = self._get_field_match_sources(
                    query_for_keyword_match, filter_conditions, "title"
                )
                self._supplement_matches_from_results(
                    query_for_keyword_match, unique_source_results, "title", title_matches
                )
                top_results = self._apply_field_boost(
                    top_results, title_matches, "title",
                    settings.EXACT_TITLE_BOOST, settings.PARTIAL_TITLE_BOOST,
                )

                # Summary boost (lower priority, applied after title).
                summary_matches = self._get_field_match_sources(
                    query_for_keyword_match, filter_conditions, "summary"
                )
                self._supplement_matches_from_results(
                    query_for_keyword_match, unique_source_results, "summary", summary_matches
                )
                top_results = self._apply_field_boost(
                    top_results, summary_matches, "summary",
                    settings.EXACT_SUMMARY_BOOST, settings.PARTIAL_SUMMARY_BOOST,
                )

                # Inject title/summary-matched documents that were filtered out by the
                # semantic score threshold (e.g. short abbreviation queries like "SMC").
                # Title takes precedence when a source matched on both fields.
                # These documents skip the score threshold, so we raise their score up to
                # it. Without this they could come back scoring below the threshold the
                # caller asked for. _floor_for returns whichever threshold is in use.
                def _floor_for(field_name: str) -> float:
                    if not use_detail_filter:
                        return filter_score
                    return getattr(detail_filter_score, field_name, 0.0) or 0.0

                present_ids = {r["payload"].get("source_id") for r in top_results}
                missing_title = [sid for sid in title_matches if sid not in present_ids]
                if missing_title:
                    injected = self._fetch_field_match_docs(
                        missing_title, title_matches, "title",
                        settings.EXACT_TITLE_BOOST, settings.PARTIAL_TITLE_BOOST,
                        prethreshold_by_source=prethreshold_by_source,
                        query_embedding=query_embedding,
                        score_floor=_floor_for("title"),
                    )
                    top_results = top_results + injected
                    present_ids.update(missing_title)
                    injected_source_ids.update(d["payload"].get("source_id") for d in injected)
                    logger.info(f"Injected {len(injected)} title-match docs missing from semantic results")

                missing_summary = [
                    sid for sid in summary_matches
                    if sid not in present_ids and sid not in title_matches
                ]
                if missing_summary:
                    injected = self._fetch_field_match_docs(
                        missing_summary, summary_matches, "summary",
                        settings.EXACT_SUMMARY_BOOST, settings.PARTIAL_SUMMARY_BOOST,
                        prethreshold_by_source=prethreshold_by_source,
                        query_embedding=query_embedding,
                        score_floor=_floor_for("summary"),
                    )
                    top_results = top_results + injected
                    injected_source_ids.update(d["payload"].get("source_id") for d in injected)
                    logger.info(f"Injected {len(injected)} summary-match docs missing from semantic results")

                top_results.sort(key=lambda x: x["weighted_score"], reverse=True)

                # Re-cap top_k after re-sorting
                top_results = top_results[:top_k]

            # Late Payload Retrieval: Fetch full payloads (including 'text')
            # for ONLY the final top_k results when hybrid/sparse search is active.
            if settings.SPARSE_SEARCH_ENABLED and top_results:
                points_to_fetch = [r["id"] for r in top_results]
                try:
                    t2 = time.time()
                    full_points = qdrant_client.retrieve(
                        collection_name=self.collection_name,
                        ids=points_to_fetch,
                        with_payload=True,
                        with_vectors=False
                    )
                    t3 = time.time()
                    logger.info(f"TIMING: late retrieve of {len(points_to_fetch)} docs took {t3-t2:.2f}s")
                    payload_map = {p.id: p.payload for p in full_points}
                    for r in top_results:
                        r["payload"] = payload_map.get(r["id"], r["payload"])
                    logger.info(f"Successfully retrieved full payloads for final top {len(top_results)} results")
                except Exception as exc:
                    logger.error(f"Failed late payload retrieval: {exc}")
                    # Fallback to the partial metadata payload already present rather than failing the search

            result_items = self._build_result_items(top_results, request.include_scoring_debug)

            search_config = {
                "search_fields": search_fields,
                "weights": weights,
                "priority_order": self.priority_order,
                "filters_applied": filter_conditions is not None,
                "filter_mode": "detail_filter_score" if use_detail_filter else "filter_score",
                "filter_score": None if use_detail_filter else filter_score,
                "search_mode": search_mode,
                "hybrid_search_enabled": settings.HYBRID_SEARCH_ENABLED,
                "sparse_search_enabled": settings.SPARSE_SEARCH_ENABLED,
                "fusion_method": settings.HYBRID_FUSION_METHOD,
            }
            
            if use_detail_filter:
                search_config["detail_filter_score"] = {
                    "title": detail_filter_score.title,
                    "text": detail_filter_score.text,
                    "tags": detail_filter_score.tags,
                    "summary": detail_filter_score.summary,
                    "metadata": detail_filter_score.metadata
                }

            # Additive debug block: expose the per-query normalization reference + boost
            # config so QA can reproduce normalized_dense/normalized_sparse and the boost
            # step by hand. Only when include_scoring_debug is set (keeps prod responses
            # lean). scoring_context already holds candidate_pool_size and, in hybrid mode,
            # dense_min/max + sparse_min/max from _rank_results; augment with the fusion
            # weights and the boost multiplier table (all from settings).
            if request.include_scoring_debug:
                scoring_context["dense_weight"] = settings.HYBRID_DENSE_WEIGHT
                scoring_context["sparse_weight"] = settings.HYBRID_SPARSE_WEIGHT
                scoring_context["boost_config"] = {
                    "exact_title_boost": settings.EXACT_TITLE_BOOST,
                    "partial_title_boost": settings.PARTIAL_TITLE_BOOST,
                    "exact_summary_boost": settings.EXACT_SUMMARY_BOOST,
                    "partial_summary_boost": settings.PARTIAL_SUMMARY_BOOST,
                }
                search_config["scoring_context"] = scoring_context

            # total_results = all unique matched sources (semantic pool ∪ injected boost docs),
            # so the count never reports fewer than the results actually returned.
            matched_source_ids = {r["payload"].get("source_id") for r in unique_source_results}
            matched_source_ids |= injected_source_ids
            total_results = len(matched_source_ids)

            logger.info("========== SEARCH COMPLETED ==========" )
            logger.info(f"Returned {len(result_items)} results from {total_results} unique sources")

            return PrioritizedSearchResponse(
                query=request.query,
                total_results=total_results,
                top_k=top_k,
                results=result_items,
                search_config=search_config
            )
            
        except ValueError as e:
            logger.error(f"Validation error in prioritized search: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Prioritized search failed: {str(e)}", exc_info=True)
            raise RuntimeError(f"Search operation failed: {str(e)}")
    
    def _parallel_batch_search(
        self,
        search_fields: List[str],
        weights: Dict[str, float],
        query_embedding: Any,
        filter_conditions: Optional[models.Filter],
        limit: int
    ) -> tuple[Dict[str, Any], Dict[str, Dict[str, float]]]:
        """
        Execute parallel batch search across multiple vector fields.
        
        Uses Qdrant's native search_batch for optimal performance.
        
        Args:
            search_fields: Field names to search (title, tags, summary, metadata, text)
            weights: Weight configuration for each field
            query_embedding: Query embedding vector
            filter_conditions: Optional Qdrant filter conditions
            limit: Number of results to retrieve per field
            
        Returns:
            Tuple of (all_results dict, field_scores dict)
        """
        search_requests = []
        valid_fields = []

        # Defense-in-depth: validate the dense vector immediately before it is sent to
        # Qdrant. query_embedding is already a validated list from embed_query, but this
        # guards against any future caller passing a raw/empty vector.
        dense_vector = embedding.validate_vector(query_embedding)

        for field in search_fields:
            if field not in weights:
                logger.warning(f"Field '{field}' not in weights config, skipping")
                continue

            search_requests.append(
                QueryRequest(
                    query=dense_vector,
                    using=field,
                    limit=limit,
                    with_payload=True,
                    filter=filter_conditions
                )
            )
            valid_fields.append(field)
        
        logger.info(f"Executing batch search across {len(valid_fields)} fields: {valid_fields}")
        batch_results = qdrant_client.query_batch_points(
            collection_name=self.collection_name,
            requests=search_requests
        )

        all_results = {}
        field_scores = {}

        for field, query_response in zip(valid_fields, batch_results):
            results = query_response.points
            logger.info(f"Field '{field}' returned {len(results)} results")
            
            for result in results:
                point_id = result.id
                if point_id not in all_results:
                    all_results[point_id] = result
                    field_scores[point_id] = {}
                
                field_scores[point_id][field] = result.score
        
        logger.info(f"Batch search completed: {len(all_results)} unique documents found")
        return all_results, field_scores
    
    def _hybrid_batch_search(
        self,
        search_fields: List[str],
        query_text: str,
        query_embedding: Any,
        filter_conditions: Optional[models.Filter],
        limit: int,
    ) -> tuple[Dict[str, Any], Dict[str, Dict[str, float]]]:
        """Execute hybrid search using parallel batch queries with client-side RRF fusion.

        This maintains keyword (BM25 sparse) search active while exposing raw field-level 
        similarity scores for detail_filter_score verification.

        To minimize network bandwidth and memory footprint, payloads are projected to exclude
        heavy text content; full payloads are fetched late for the final top results.
        """
        try:
            from qdrant_client.models import QueryRequest, SparseVector
            from app.core.clients.sparse_encoder import generate_sparse_vector

            search_requests = []
            valid_fields = []

            # Defense-in-depth: validate the dense vector before it reaches Qdrant
            # (prevents the "expected dim: 384, got 0" batch 400).
            dense_vector = embedding.validate_vector(query_embedding)

            # Project payload to retrieve only small metadata keys needed for filters and boosts.
            # Excludes the heavy 'text' payload field during candidate scoring.
            metadata_payload_fields = ["source_id", "title", "summary", "tags", "metadata"]

            # 1. Build Query Requests for Dense Fields
            for field in search_fields:
                if field in self.default_weights:
                    search_requests.append(
                        QueryRequest(
                            query=dense_vector,
                            using=field,
                            limit=limit,
                            with_payload=metadata_payload_fields,
                            filter=filter_conditions
                        )
                    )
                    valid_fields.append(field)

            # 2. Build Query Request for BM25 Sparse Field
            # Replace underscores with spaces so BM25 tokenises compound
            # underscore-joined terms (e.g. "agentic_engineering") as separate
            # words rather than a single unknown token that produces empty indices.
            bm25_query_text = query_text.replace("_", " ")
            sparse_indices, sparse_values = generate_sparse_vector(bm25_query_text)
            if sparse_indices:
                search_requests.append(
                    QueryRequest(
                        query=SparseVector(
                            indices=sparse_indices,
                            values=sparse_values,
                        ),
                        using=settings.SPARSE_VECTOR_NAME,
                        limit=limit,
                        with_payload=metadata_payload_fields,
                        filter=filter_conditions
                    )
                )
                valid_fields.append(settings.SPARSE_VECTOR_NAME)

            logger.info(f"Executing client-side hybrid batch search across {len(valid_fields)} fields: {valid_fields}")
            
            # 3. Execute all queries in a single network batch call
            import time
            t0 = time.time()
            batch_results = qdrant_client.query_batch_points(
                collection_name=self.collection_name,
                requests=search_requests
            )
            t1 = time.time()
            logger.info(f"TIMING: query_batch_points took {t1-t0:.2f}s")

            # 4. Collect per-field raw similarity scores from the batch results.
            #    Each doc keeps the raw cosine score for every dense field that
            #    retrieved it, plus the raw BM25 score under the sparse field key
            #    (SPARSE_VECTOR_NAME) when the sparse query participated and returned
            #    hits. The presence of that sparse key is what _rank_results uses to
            #    detect hybrid mode and to fuse dense + sparse — the actual dense/sparse
            #    fusion (weighted or two-list RRF) lives there, so there is no separate
            #    all-field RRF computed or stored here.
            all_results: Dict[str, Any] = {}
            field_scores: Dict[str, Dict[str, float]] = {}

            for field, query_response in zip(valid_fields, batch_results):
                for point in query_response.points:
                    pid = point.id
                    all_results[pid] = point
                    if pid not in field_scores:
                        field_scores[pid] = {}
                    # Store individual raw similarity score for the field (for detail_filter_score check)
                    field_scores[pid][field] = getattr(point, "score", 0.0)

            # 5. Remap raw Qdrant vector-field names to the semantic field names that
            #    _apply_detail_filter checks against ("title", "text", "tags",
            #    "summary", "metadata"). The keys of self.default_weights are the
            #    authoritative semantic names; the corresponding Qdrant vector name is
            #    VECTOR_FIELD_PREFIX + semantic_name. This collection configures its
            #    named vectors with no prefix (see app/core/clients/qdrant.py), so the
            #    mapping is an identity today — but doing it explicitly keeps the
            #    field_scores contract correct if a prefix is ever introduced.
            #    The "bm25" (SPARSE_VECTOR_NAME) key is preserved untouched — it carries
            #    the raw BM25 score consumed by _rank_results' dense+sparse fusion and
            #    signals hybrid mode.
            prefix = getattr(settings, "VECTOR_FIELD_PREFIX", "") or ""
            qdrant_to_semantic = {
                f"{prefix}{semantic}": semantic for semantic in self.default_weights
            }
            for scores in field_scores.values():
                for qdrant_name, semantic_name in qdrant_to_semantic.items():
                    if qdrant_name in scores and semantic_name not in scores:
                        scores[semantic_name] = scores[qdrant_name]

            logger.info(f"Client-side hybrid search returned {len(all_results)} unique documents (metadata-only)")
            return all_results, field_scores

        except (ImportError, RuntimeError) as exc:
            # ImportError: optional sparse deps (fastembed / qdrant SparseVector)
            # missing — the module-level `from ... import` at the top of the try fails.
            # RuntimeError: the BM25 encoder failed to initialise or encode at runtime —
            # generate_sparse_vector() wraps every encoder failure (corrupted model
            # cache, download failure, OOM, even a missing-fastembed ImportError) as
            # RuntimeError. Both are sparse-side problems, so degrade gracefully to
            # dense-only search.
            # NOTE: AttributeError is intentionally NOT caught — it is not a genuine
            # sparse-availability signal (the deps are imported, not attribute-accessed)
            # and catching it would silently swallow programming errors (e.g. a typo
            # like query_response.point) as a quiet dense-only degradation. Likewise,
            # Qdrant transport errors (timeouts, connection failures) raise other
            # exception types and surface instead of being masked.
            logger.warning(
                f"Hybrid search unavailable ({type(exc).__name__}: {exc}); "
                "falling back to dense-only parallel search."
            )
            return self._parallel_batch_search(
                search_fields=search_fields,
                weights=self.default_weights,
                query_embedding=query_embedding,
                filter_conditions=filter_conditions,
                limit=limit,
            )

    def _build_filters(
        self,
        categories: Optional[List[str]] = None,
        organizations: Optional[List[str]] = None,
        resource_types: Optional[List[str]] = None,
        file_types: Optional[List[str]] = None,
        exclude_organizations: Optional[List[str]] = None,
        exclude_file_types: Optional[List[str]] = None,
        any_of: Optional[List[Any]] = None
    ) -> Optional[models.Filter]:
        """
        Build Qdrant filter conditions with intelligent AND/OR logic.
        
        Filter Logic:
        - Within each filter type: OR condition
        - Between filter types: AND condition
        - exclude_* values become a must_not clause: matching any of them drops
          the document. A request may carry exclusions and no positive filters.
        - any_of holds FilterBlocks that are OR'ed with each other and AND'ed
          with everything above: keep if TOP-LEVEL AND (block0 OR block1 OR ...).
          The arguments above keep their meaning either way and are never
          ignored when any_of is present.

        Field Mappings:
        - categories → tags (list)
        - organizations → metadata.company (string)
        - resource_types → metadata.DOCUMENT_TYPE (comma-separated string)
        - file_types → metadata.type (string)
        
        Example:
        Input: categories=["Life Skills", "Peer Learning"], organizations=["involve"]
        Output: (tags IN ["Life Skills", "Peer Learning"]) AND (metadata.company = "involve")
        
        Args:
            categories: Tag values to filter by
            organizations: Company names to filter by
            resource_types: Key entities to filter by (uses text matching)
            file_types: Document types to filter by
            exclude_organizations: Company names to exclude (must_not)
            exclude_file_types: Document types to exclude (must_not)
            any_of: FilterBlocks to OR together and AND with the filters above

        Returns:
            Qdrant Filter object or None if no filters provided
        """
        must_conditions = []
        must_not_conditions = []
        filter_summary = []
        
        # Categories filter (tags field)
        if categories and any(cat.strip() for cat in categories):
            valid_categories = [cat.strip() for cat in categories if cat.strip()]
            condition = models.FieldCondition(key="tags", match=models.MatchAny(any=valid_categories))
            must_conditions.append(condition)
            filter_summary.append(f"tags: ({' OR '.join(valid_categories)})")
            logger.info(f"Filter - categories: {valid_categories}")

        # Organizations filter (metadata.company field)
        if organizations and any(org.strip() for org in organizations):
            valid_orgs = [org.strip() for org in organizations if org.strip()]
            condition = models.FieldCondition(key="metadata.company", match=models.MatchAny(any=valid_orgs))
            must_conditions.append(condition)
            filter_summary.append(f"organizations: ({' OR '.join(valid_orgs)})")
            logger.info(f"Filter - organizations: {valid_orgs}")

        # Resource types filter (metadata.DOCUMENT_TYPE field - comma-separated string)
        if resource_types and any(rt.strip() for rt in resource_types):
            valid_resource_types = [rt.strip() for rt in resource_types if rt.strip()]
            resource_conditions = []
            for rt in valid_resource_types:
                resource_conditions.append(
                    models.FieldCondition(
                        key="metadata.DOCUMENT_TYPE",
                        match=models.MatchText(text=rt)
                    )
                )
            if len(resource_conditions) == 1:
                must_conditions.append(resource_conditions[0])
            else:
                must_conditions.append(models.Filter(should=resource_conditions))
            filter_summary.append(f"resource_types: ({' OR '.join(valid_resource_types)})")
            logger.info(f"Filter - resource_types: {valid_resource_types}")

        # File types filter (metadata.type field)
        if file_types and any(ft.strip() for ft in file_types):
            valid_file_types = [ft.strip() for ft in file_types if ft.strip()]
            condition = models.FieldCondition(key="metadata.type", match=models.MatchAny(any=valid_file_types))
            must_conditions.append(condition)
            filter_summary.append(f"file_types: ({' OR '.join(valid_file_types)})")
            logger.info(f"Filter - file_types: {valid_file_types}")

        # Exclusions (must_not). Matching any listed value drops the document.
        for label, key, values in (
            ("organizations", "metadata.company", exclude_organizations),
            ("file_types", "metadata.type", exclude_file_types),
        ):
            condition = _match_any_condition(key, values)
            if condition is None:
                continue
            must_not_conditions.append(condition)
            filter_summary.append(f"NOT {label}: ({' OR '.join(condition.match.any)})")
            logger.info(f"Filter - exclude_{label}: {condition.match.any}")

        # Alternatives (should). Each block is itself a filter block, so it is
        # built by this same function — the branch is compiled by the identical
        # code that compiles a flat request, and each recursive call logs its own
        # per-field lines. Depth is exactly one: FilterBlock forbids extra fields,
        # so a block cannot carry its own any_of and the recursion cannot go deeper.
        if any_of:
            logger.info(f"Filter - any_of: {len(any_of)} alternatives")
            branch_filters = []
            for block in any_of:
                branch = self._build_filters(
                    categories=block.categories,
                    organizations=block.organizations,
                    resource_types=block.resource_type,
                    file_types=block.file_type,
                    exclude_organizations=block.exclude_organizations,
                    exclude_file_types=block.exclude_file_type,
                )
                if branch is not None:
                    branch_filters.append(branch)
            if branch_filters:
                must_conditions.append(models.Filter(should=branch_filters))
                filter_summary.append(f"ANY OF ({len(branch_filters)} alternatives)")

        if must_conditions or must_not_conditions:
            logger.info(
                f"Total filters: {len(must_conditions)} must, "
                f"{len(must_not_conditions)} must_not (AND logic between types)"
            )
            logger.info(f"Filter summary: {' AND '.join(filter_summary)}")
            return models.Filter(
                must=must_conditions or None,
                must_not=must_not_conditions or None,
            )

        logger.info("No filters applied")
        return None
    
    @staticmethod
    def _min_max_normalize(scores: Dict[Any, float]) -> Dict[Any, float]:
        """Min-max normalize a {key: score} map to the [0, 1] range.

        When every score is equal (single candidate or a flat pool) the range is
        zero and a min-max is undefined; a positive value maps to 1.0 and a zero
        value to 0.0 so that present candidates are never spuriously zeroed out.
        """
        if not scores:
            return {}
        values = list(scores.values())
        lo, hi = min(values), max(values)
        if hi <= lo:
            return {k: (1.0 if v > 0 else 0.0) for k, v in scores.items()}
        span = hi - lo
        return {k: (v - lo) / span for k, v in scores.items()}

    @staticmethod
    def _rank_positions(scores: Dict[Any, float]) -> Dict[Any, int]:
        """Assign 1-indexed ranks to keys ordered by descending score.

        Used by the RRF fusion path: the top-scoring key gets rank 1, the next
        rank 2, and so on. Ties are broken deterministically by the sort's
        stability so the same input always yields the same ranks.
        """
        ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return {key: idx + 1 for idx, (key, _) in enumerate(ordered)}

    @staticmethod
    def _cosine_similarity(a: Any, b: Any) -> Optional[float]:
        """How similar two vectors are, 0 to 1. Same maths Qdrant uses (Distance.COSINE),
        so the result is comparable to the scores Qdrant returns for a normal search.
        Returns None, not 0.0, for a missing or unusable vector — None means "not scored".
        """
        if a is None or b is None:
            return None
        try:
            va = np.asarray(a, dtype=np.float32)
            vb = np.asarray(b, dtype=np.float32)
        except (TypeError, ValueError):
            return None
        if va.ndim != 1 or va.shape != vb.shape:
            return None
        na, nb = float(np.linalg.norm(va)), float(np.linalg.norm(vb))
        if na == 0.0 or nb == 0.0:
            return None
        return float(np.dot(va, vb) / (na * nb))

    def _rank_results(
        self,
        all_results: Dict[str, Any],
        field_scores: Dict[str, Dict[str, float]],
        weights: Dict[str, float],
        search_fields: List[str],
        scoring_context_out: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Rank results using weighted multi-field scoring.

        Dense-only path (sparse disabled):
            Final_Score = Σ(Field_Weight × Field_Score)

        Hybrid path (dense + BM25 sparse): the dense component is always the weighted
        multi-field cosine sum; the dense+sparse fusion is selected by
        settings.HYBRID_FUSION_METHOD:
          - "weighted" (default): each modality is min-max normalized to [0, 1] across
            the candidate pool, then fused:
                Final_Score = HYBRID_DENSE_WEIGHT × dense_norm + HYBRID_SPARSE_WEIGHT × sparse_norm
          - "rrf": Reciprocal Rank Fusion over two lists — the combined dense list
            (ranked by the weighted cosine sum) and the sparse list:
                Final_Score = minmax( 1/(RRF_K+dense_rank) + 1/(RRF_K+sparse_rank) )
        Both modes keep the final score on a calibrated 0-1 scale comparable to
        filter_score, instead of the raw RRF fused value (~0-0.1) which never clears a
        cosine-scale threshold.

        Note: hybrid mode is detected by the presence of a raw sparse (BM25) score in
        field_scores (the sparse field key), not a separate all-field RRF value. The
        "rrf" fusion method below computes its own RRF over exactly TWO lists — the
        single combined dense list (the 5 dense fields collapsed into one weighted
        cosine sum, raw_dense) and the sparse list (raw_sparse) — i.e. up to two rank
        terms per doc, NOT one RRF term per dense field. _apply_detail_filter is pure
        per-field OR logic and does not consume any fusion score.

        Args:
            all_results: Dictionary of search results by point ID
            field_scores: Scores for each field per point
            weights: Weight configuration for each field
            search_fields: List of fields searched
            scoring_context_out: Optional mutable dict. When provided, it is populated
                with the per-query normalization context (candidate_pool_size and, in
                hybrid mode, dense_min/max + sparse_min/max) so the caller can surface it
                under search_config.scoring_context without recomputing raw_dense/raw_sparse
                or keeping per-request state on the (shared) service instance. Left empty
                on the dense-only path except for candidate_pool_size.

        Returns:
            List of ranked results sorted by weighted score (descending)
        """
        sparse_name = settings.SPARSE_VECTOR_NAME
        # Hybrid mode is signalled by the presence of a raw sparse (BM25) score under
        # the sparse field key — _hybrid_batch_search injects it only when the BM25
        # query actually participated and returned hits. This is the same key the rrf/
        # weighted fusion below reads as raw_sparse, so detection and fusion stay tied
        # to one signal. Detecting it this way (rather than via a global flag or a
        # separate all-field RRF marker) keeps the dense-only fallback and the
        # empty-sparse edge case on the plain weighted-sum path, avoiding score
        # deflation. In hybrid mode we fuse normalized dense + sparse scores.
        is_hybrid = any(sparse_name in fs for fs in field_scores.values())

        # Fusion method (env-selectable): "weighted" min-max score fusion, or "rrf"
        # rank fusion of the combined dense list vs the sparse list. The dense
        # component is the weighted multi-field cosine sum in BOTH modes — only the
        # dense+sparse combination step differs.
        fusion_method = settings.HYBRID_FUSION_METHOD

        norm_dense: Dict[Any, float] = {}
        norm_sparse: Dict[Any, float] = {}
        hybrid_scores: Dict[Any, float] = {}
        # Diagnostic maps surfaced via include_scoring_debug (empty outside hybrid /
        # the rrf branch, so .get() yields None for those results).
        raw_sparse: Dict[Any, float] = {}
        rrf_raw: Dict[Any, float] = {}
        dense_rank: Dict[Any, int] = {}
        sparse_rank: Dict[Any, int] = {}
        if is_hybrid:
            raw_dense: Dict[Any, float] = {}
            for point_id, fs in field_scores.items():
                dense = 0.0
                for field in search_fields:
                    if field in weights:
                        score = fs.get(field)
                        if score is not None:
                            dense += score * weights[field]
                raw_dense[point_id] = dense
                raw_sparse[point_id] = fs.get(sparse_name) or 0.0

            if fusion_method == "rrf":
                # Reciprocal Rank Fusion over two lists: the combined dense list
                # (ranked by the weighted multi-field cosine sum) and the sparse
                # list. This keeps the dense weighting intact (constraint) while
                # fusing by rank position rather than raw score.
                rrf_k = settings.RRF_K
                dense_hits = {pid: s for pid, s in raw_dense.items() if s > 0.0}
                dense_rank = self._rank_positions(dense_hits)
                # Only docs with an actual sparse hit get a sparse rank.
                sparse_hits = {pid: s for pid, s in raw_sparse.items() if s > 0.0}
                sparse_rank = self._rank_positions(sparse_hits)
                for point_id in raw_dense:
                    fused = 0.0
                    if point_id in dense_rank:
                        fused += 1.0 / (rrf_k + dense_rank[point_id])
                    if point_id in sparse_rank:
                        fused += 1.0 / (rrf_k + sparse_rank[point_id])
                    rrf_raw[point_id] = fused
                # Normalize to [0, 1] so filter_score keeps a comparable scale
                # (raw RRF values are ~0-0.03 and would never clear a threshold).
                # rrf_raw is retained for debug surfacing (pre-normalization).
                hybrid_scores = self._min_max_normalize(rrf_raw)
            else:
                # Weighted min-max score fusion (default).
                norm_dense = self._min_max_normalize(raw_dense)
                norm_sparse = self._min_max_normalize(raw_sparse)
                dense_w = settings.HYBRID_DENSE_WEIGHT
                sparse_w = settings.HYBRID_SPARSE_WEIGHT
                for point_id in raw_dense:
                    hybrid_scores[point_id] = (
                        dense_w * norm_dense.get(point_id, 0.0)
                        + sparse_w * norm_sparse.get(point_id, 0.0)
                    )

            # Expose the per-query normalization reference (min/max of each modality
            # across the candidate pool) so a debug caller can reproduce normalized_dense/
            # normalized_sparse by hand. These are the exact min/max _min_max_normalize
            # uses. Only meaningful in hybrid mode where min-max normalization runs.
            if scoring_context_out is not None:
                dense_vals = list(raw_dense.values())
                sparse_vals = list(raw_sparse.values())
                if dense_vals:
                    scoring_context_out["dense_min"] = min(dense_vals)
                    scoring_context_out["dense_max"] = max(dense_vals)
                if sparse_vals:
                    scoring_context_out["sparse_min"] = min(sparse_vals)
                    scoring_context_out["sparse_max"] = max(sparse_vals)

        # candidate_pool_size is meaningful in both modes.
        if scoring_context_out is not None:
            scoring_context_out["candidate_pool_size"] = len(all_results)

        ranked = []

        for point_id, result in all_results.items():
            field_score_dict = field_scores.get(point_id, {})

            if is_hybrid:
                # Calibrated 0-1 fused score per the selected fusion method.
                weighted_score = hybrid_scores.get(point_id, 0.0)
                # raw_dense is the pre-fusion weighted cosine sum computed above.
                entry_raw_dense = raw_dense.get(point_id, 0.0)
            else:
                # Dense multi-field path: weighted sum of per-field similarity scores.
                weighted_score = 0.0
                for field in search_fields:
                    if field in field_score_dict and field in weights:
                        field_score = field_score_dict[field]
                        weight = weights[field]
                        weighted_score += field_score * weight
                # In dense-only mode the pre-boost weighted score IS raw_dense.
                entry_raw_dense = weighted_score

            # Do NOT cap at 1.0 here — the title/summary boost (applied later) needs
            # the uncapped score to differentiate matches, and enforces its own cap.
            num_fields_matched = len([k for k in field_score_dict.keys() if k != sparse_name])

            entry = {
                'id': result.id,
                'payload': result.payload,
                'weighted_score': weighted_score,
                'field_scores': field_score_dict,
                'num_fields_matched': num_fields_matched,
                # raw_dense (pre-fusion, pre-boost weighted cosine sum) is meaningful in
                # both modes; surfaced only when include_scoring_debug (see _build_result_items).
                'raw_dense': entry_raw_dense,
            }
            if is_hybrid:
                # Internal scoring diagnostics, surfaced only when the request sets
                # include_scoring_debug (see _build_result_items). rrf_score /
                # dense_rank / sparse_rank are None outside the rrf fusion branch.
                entry['keyword_score'] = raw_sparse.get(point_id)
                entry['rrf_score'] = rrf_raw.get(point_id)
                entry['dense_rank'] = dense_rank.get(point_id)
                entry['sparse_rank'] = sparse_rank.get(point_id)
                # Per-modality normalized scores (weighted-fusion branch only; norm_dense/
                # norm_sparse are empty in rrf mode, so .get() yields None there).
                entry['normalized_dense'] = norm_dense.get(point_id)
                entry['normalized_sparse'] = norm_sparse.get(point_id)
            ranked.append(entry)

        ranked.sort(key=lambda x: x['weighted_score'], reverse=True)
        return ranked
    
    def _apply_detail_filter(
        self,
        ranked_results: List[Dict[str, Any]],
        detail_filter_score: Any
    ) -> List[Dict[str, Any]]:
        """
        Apply field-level threshold filtering with OR logic.
        
        A document passes if ANY field score meets or exceeds its threshold.
        
        Args:
            ranked_results: Ranked results with field scores
            detail_filter_score: DetailFilterScore object with field thresholds
            
        Returns:
            Filtered results where at least one field meets its threshold
        """
        filtered = []
        total_passed = 0
        
        # Get threshold values
        thresholds = {
            'title': detail_filter_score.title,
            'text': detail_filter_score.text,
            'tags': detail_filter_score.tags,
            'summary': detail_filter_score.summary,
            'metadata': detail_filter_score.metadata
        }
        
        logger.info(f"Detail filter thresholds: {thresholds}")
        
        for result in ranked_results:
            field_score_dict = result.get('field_scores', {})
            passed = False
            passing_fields = []

            # Check if ANY field meets its threshold (OR logic)
            for field, threshold in thresholds.items():
                field_score = field_score_dict.get(field, 0.0) or 0.0
                if field_score >= threshold:
                    passed = True
                    passing_fields.append(f"{field}={field_score:.3f}")

            if passed:
                total_passed += 1
                filtered.append(result)
                if total_passed <= 5:  # Log first 5 for debugging
                    logger.debug(f"Document {result.get('id')} passed with fields: {', '.join(passing_fields)}")
        
        logger.info(f"Detail filter: {total_passed}/{len(ranked_results)} documents passed")
        return filtered
    
    def _filter_best_per_source(self, ranked_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Deduplicate results by source_id, keeping only the highest scoring document.
        
        Args:
            ranked_results: Ranked results sorted by score (descending)
            
        Returns:
            Deduplicated results with best match per source_id
        """
        seen_sources = {}
        unique_results = []
        
        for result in ranked_results:
            source_id = result['payload'].get('source_id')
            
            if not source_id:
                continue
            
            if source_id not in seen_sources:
                seen_sources[source_id] = True
                unique_results.append(result)
        
        logger.info(f"Filtered {len(ranked_results)} results to {len(unique_results)} unique sources")
        return unique_results
    
    def _scroll_and_collect_unique_sources(self, filter_conditions) -> dict:
        """Scroll through documents and collect unique sources"""
        unique_sources = {}
        scroll_result = qdrant_client.scroll(
            collection_name=self.collection_name,
            scroll_filter=filter_conditions,
            limit=10000,
            with_payload=True,
            with_vectors=False
        )
        
        points, next_page_offset = scroll_result
        
        for point in points:
            source_id = point.payload.get('source_id')
            if source_id and source_id not in unique_sources:
                unique_sources[source_id] = {
                    'id': point.id,
                    'payload': point.payload,
                    'source_id': source_id
                }
        
        while next_page_offset:
            scroll_result = qdrant_client.scroll(
                collection_name=self.collection_name,
                scroll_filter=filter_conditions,
                limit=10000,
                offset=next_page_offset,
                with_payload=True,
                with_vectors=False
            )
            points, next_page_offset = scroll_result
            
            for point in points:
                source_id = point.payload.get('source_id')
                if source_id and source_id not in unique_sources:
                    unique_sources[source_id] = {
                        'id': point.id,
                        'payload': point.payload,
                        'source_id': source_id
                    }
        
        return unique_sources

    def _build_source_result_items(self, top_results) -> List[SearchResultItem]:
        """Build result items from source documents"""
        result_items = []
        for doc in top_results:
            try:
                result_items.append(SearchResultItem(
                    id=str(doc['id']),
                    text=doc['payload'].get('text', ''),
                    title=doc['payload'].get('title'),
                    summary=doc['payload'].get('summary'),
                    tags=doc['payload'].get('tags'),
                    metadata=doc['payload'].get('metadata', {}),
                    source_id=doc['payload'].get('source_id', ''),
                    score=1.0,
                    field_scores={}
                ))
            except Exception as e:
                logger.warning(f"Failed to parse result item {doc.get('id')}: {str(e)}")
                continue
        return result_items

    @staticmethod
    def _classify_text_match(query_lower: str, field_lower: str) -> Optional[str]:
        """Classify how ``query_lower`` matches ``field_lower``.

        Returns 'exact' (whole field equals query), 'partial' (substring match —
        covers both prefix and mid/infix occurrences), or None (no match). Infix
        ('mid') matches are intentionally folded into 'partial' so the response
        contract stays {exact, partial, None}.
        """
        if not field_lower or query_lower not in field_lower:
            return None
        return "exact" if field_lower == query_lower else "partial"

    def _get_field_match_sources(
        self,
        query: str,
        filter_conditions: Optional[models.Filter],
        field: str,
    ) -> Dict[str, str]:
        """Return a map of source_id → match_type ('exact'|'partial') for documents
        whose ``field`` payload (e.g. 'title' or 'summary') contains the query string.

        Uses a MatchText payload filter against the prefix-tokenized index, then refines
        with a Python substring check so prefix and mid/infix occurrences are classified.
        The scroll retrieves all matching points (batched) so no relevant document is missed.
        """
        matches: Dict[str, str] = {}
        query_lower = query.strip().lower()
        if not query_lower:
            return matches

        field_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key=field,
                    match=models.MatchText(text=query_lower),
                )
            ]
        )

        # Combine with any existing hard filters (org / category / etc.). must_not
        # has to carry through, or this scroll re-admits excluded documents.
        if filter_conditions:
            combined_filter = models.Filter(
                must=list(filter_conditions.must or []) + list(field_filter.must),
                must_not=list(filter_conditions.must_not or []) or None,
            )
        else:
            combined_filter = field_filter

        try:
            offset = None
            while True:
                scroll_kwargs: Dict[str, Any] = dict(
                    collection_name=self.collection_name,
                    scroll_filter=combined_filter,
                    limit=1000,
                    with_payload=True,
                    with_vectors=False,
                )
                if offset is not None:
                    scroll_kwargs["offset"] = offset

                points, next_offset = qdrant_client.scroll(**scroll_kwargs)

                for point in points:
                    source_id = point.payload.get("source_id")
                    raw_value = point.payload.get(field) or ""
                    if not source_id or source_id in matches:
                        continue

                    match_type = self._classify_text_match(query_lower, raw_value.lower())
                    if match_type:
                        matches[source_id] = match_type

                if not next_offset:
                    break
                offset = next_offset

        except Exception as exc:
            logger.warning(f"{field} match scroll failed (non-fatal): {exc}")

        logger.info(
            f"{field} match sources found: {len(matches)} "
            f"(exact={sum(1 for v in matches.values() if v == 'exact')}, "
            f"partial={sum(1 for v in matches.values() if v == 'partial')})"
        )
        return matches

    def _supplement_matches_from_results(
        self,
        query: str,
        ranked_results: List[Dict[str, Any]],
        field: str,
        matches: Dict[str, str],
    ) -> None:
        """Catch mid/infix matches present in the already-retrieved candidate pool.

        The prefix-tokenized index broadens MatchText recall to prefixes, but a true
        infix query (e.g. 'sur' in 'insurance') may not be retrieved by the scroll.
        This in-memory pass scans the dense candidates' ``field`` payloads with a plain
        substring check — no extra Qdrant calls — and adds any newly found matches.
        """
        query_lower = query.strip().lower()
        if not query_lower:
            return
        for result in ranked_results:
            source_id = result["payload"].get("source_id")
            if not source_id or source_id in matches:
                continue
            raw_value = (result["payload"].get(field) or "").lower()
            match_type = self._classify_text_match(query_lower, raw_value)
            if match_type:
                matches[source_id] = match_type

    def _apply_field_boost(
        self,
        ranked_results: List[Dict[str, Any]],
        matches: Dict[str, str],
        field: str,
        exact_boost: float,
        partial_boost: float,
    ) -> List[Dict[str, Any]]:
        """Multiply the weighted_score of documents whose ``field`` matched the query.

        Boost tiers (capped at 1.0): exact → ×exact_boost, partial → ×partial_boost.
        The match type is recorded in field_scores as ``f"{field}_match"`` so callers
        can surface it. The numeric multiplier actually applied is recorded on the entry
        as ``f"{field}_multiplier"`` (1.0 on the no-match path = neutral no-op, so debug
        output always carries the field). Results are re-sorted after boosting.
        """
        match_key = f"{field}_match"
        mult_key = f"{field}_multiplier"
        for result in ranked_results:
            source_id = result["payload"].get("source_id")
            match_type = matches.get(source_id)
            if not match_type:
                result["field_scores"].setdefault(match_key, None)
                # 1.0 = no boost applied (neutral no-op). setdefault so an earlier
                # boost pass on the same field is never clobbered.
                result.setdefault(mult_key, 1.0)
                continue

            multiplier = exact_boost if match_type == "exact" else partial_boost
            original = result["weighted_score"]
            boosted = min(original * multiplier, 1.0)
            result["weighted_score"] = boosted
            result["field_scores"][match_key] = match_type
            result[mult_key] = multiplier

            logger.debug(
                f"{field} boost applied to {source_id}: "
                f"{match_type} match, {original:.4f} → {boosted:.4f}"
            )

        ranked_results.sort(key=lambda x: x["weighted_score"], reverse=True)
        return ranked_results

    def _fetch_field_match_docs(
        self,
        source_ids: List[str],
        matches: Dict[str, str],
        field: str,
        exact_boost: float,
        partial_boost: float,
        prethreshold_by_source: Optional[Dict[str, Dict[str, Any]]] = None,
        query_embedding: Optional[Any] = None,
        score_floor: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Fetch one representative chunk per source_id for documents that matched
        by ``field`` but were absent from semantic search results (e.g. short
        abbreviation queries where cosine similarity is too low to pass filter_score).

        Each one gets a small score so it ranks below strong semantic results, raised to
        ``score_floor`` (the threshold in use) so it never scores below what the caller
        asked for. Per-field scores are reused from ``prethreshold_by_source`` when the
        document was already scored, otherwise computed from its stored vectors.

        Optimized to fetch all missing documents in a single MatchAny query.
        """
        # Step 1: Guard against empty lists to avoid an unnecessary network round trip
        if not source_ids:
            return []

        injected: List[Dict[str, Any]] = []
        FLOOR_SCORE = 0.15  # baseline before multiplier — keeps injected below strong semantic hits
        match_key = f"{field}_match"
        mult_key = f"{field}_multiplier"
        prethreshold_by_source = prethreshold_by_source or {}

        def _score_for(match_type: str) -> tuple:
            """Returns (score, boost). Multiply first, then raise to the floor — doing it
            the other way round (floor * boost) would push these above real search hits.
            """
            boost = exact_boost if match_type == "exact" else partial_boost
            return min(max(FLOOR_SCORE * boost, score_floor), 1.0), boost

        # Step 2: Documents we already scored earlier, before the threshold removed them.
        # Their real scores are still in memory, so reuse them instead of asking Qdrant.
        remaining: List[str] = []
        for source_id in source_ids:
            ranked = prethreshold_by_source.get(source_id)
            if ranked is None:
                remaining.append(source_id)
                continue

            match_type = matches.get(source_id, "partial")
            score, boost = _score_for(match_type)

            # Copy both dicts — the original is shared and must not be changed.
            entry = dict(ranked)
            entry["field_scores"] = dict(ranked.get("field_scores") or {})
            entry["field_scores"][match_key] = match_type
            entry["weighted_score"] = score
            entry[mult_key] = boost
            injected.append(entry)
            logger.debug(
                f"Reused pre-threshold scores for {field}-match doc {source_id} "
                f"({match_type}), score {score:.4f}"
            )

        if not remaining:
            logger.info(
                f"Resolved all {len(source_ids)} missing '{field}' documents from the "
                f"pre-threshold pool — no Qdrant request needed."
            )
            return injected

        source_ids = remaining
        num_requests_before = len(source_ids)

        # Step 3: The rest were never scored, so fetch their vectors and score them here.
        # Skipped when there are too many to be worth downloading.
        score_vectors = query_embedding is not None and len(source_ids) <= settings.INJECTED_DOC_SCORING_MAX
        if query_embedding is not None and not score_vectors:
            logger.info(
                f"Skipping vector scoring for {len(source_ids)} injected '{field}' docs "
                f"(over INJECTED_DOC_SCORING_MAX={settings.INJECTED_DOC_SCORING_MAX}); "
                f"field_scores stay None."
            )
        # List the fields we want, never True — True would also download the BM25 vector.
        with_vectors: Any = list(self.priority_order) if score_vectors else False

        logger.info(
            f"Resolving {num_requests_before} missing documents for field '{field}' boost. "
            f"Optimizing from {num_requests_before} Qdrant requests to 1 request."
        )

        try:
            start_time = time.perf_counter()
            # Step 2: Paginate scroll with MatchAny until every requested source_id has
            # at least one point or Qdrant returns no further results.  A single page
            # of len(source_ids)*10 points (capped at 1000) is almost always enough;
            # the loop only continues when sources with unusually many chunks exhaust
            # the first page before all source_ids have been seen.
            all_points: List = []
            covered_sources: Set[str] = set()
            source_ids_set = set(source_ids)
            page_size = min(len(source_ids) * 10, 1000)
            offset = None
            num_pages = 0

            while True:
                batch, offset = qdrant_client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="source_id",
                                match=models.MatchAny(any=source_ids),
                            )
                        ]
                    ),
                    limit=page_size,
                    offset=offset,
                    with_payload=True,
                    with_vectors=with_vectors,
                )
                num_pages += 1
                all_points.extend(batch)
                for pt in batch:
                    sid = pt.payload.get("source_id")
                    if sid:
                        covered_sources.add(sid)
                if offset is None or source_ids_set <= covered_sources:
                    break

            points = all_points
            elapsed_time = time.perf_counter() - start_time
            logger.info(
                f"Bulk fetch of {len(points)} points for {len(source_ids)} source IDs "
                f"completed in {elapsed_time:.3f}s ({num_pages} page(s))"
            )
        except Exception as exc:
            logger.warning(f"Could not bulk fetch {field}-match docs: {exc}")
            return []

        # Step 3: Deduplicate matching points in-memory.
        # Since we only want one representative chunk per unique source_id, we process them
        # sequentially and keep the first one we see.
        seen_sources: Set[str] = set()
        for point in points:
            source_id = point.payload.get("source_id")
            if not source_id or source_id not in source_ids:
                continue
            if source_id in seen_sources:
                continue
            seen_sources.add(source_id)

            # Step 4: Compute the boosted score, floored at the caller's threshold.
            match_type = matches.get(source_id, "partial")
            score, boost = _score_for(match_type)

            # Score each field against the query. A field is None when the document has
            # no vector for it (empty fields are not stored) or scoring was skipped.
            # None means "not scored" — never write 0.0 here, that means "scored zero".
            vector_map = (getattr(point, "vector", None) or {}) if score_vectors else {}
            field_scores: Dict[str, Any] = {
                f: (self._cosine_similarity(query_embedding, vector_map.get(f))
                    if score_vectors else None)
                for f in self.priority_order
            }
            field_scores[match_key] = match_type

            # Combine the field scores using the same weights as a normal search, so this
            # number can be compared against one. None when no field could be scored.
            scored = [
                (f, s) for f, s in field_scores.items()
                if f in self.default_weights and isinstance(s, (int, float))
            ]
            raw_dense = (
                sum(s * self.default_weights[f] for f, s in scored) if scored else None
            )

            injected.append({
                "id": point.id,
                "payload": point.payload,
                "weighted_score": score,
                "field_scores": field_scores,
                "raw_dense": raw_dense,
                mult_key: boost,
            })
            logger.debug(f"Injected {field}-match doc {source_id} ({match_type}) with floor score {score:.4f}")

        missing_after = len(source_ids) - len(seen_sources)
        logger.info(
            f"Finished resolving missing documents for field '{field}'. "
            f"Requests before: {num_requests_before}, Requests after: 1. "
            f"Successfully resolved: {len(seen_sources)}/{len(source_ids)}. "
            f"Missing/Not Found IDs: {missing_after}."
        )

        return injected

    # ── Backward-compatible title wrappers (kept for existing callers/tests) ──────
    def _get_title_match_sources(
        self,
        query: str,
        filter_conditions: Optional[models.Filter],
    ) -> Dict[str, str]:
        return self._get_field_match_sources(query, filter_conditions, "title")

    def _apply_title_boost(
        self,
        ranked_results: List[Dict[str, Any]],
        title_matches: Dict[str, str],
    ) -> List[Dict[str, Any]]:
        return self._apply_field_boost(
            ranked_results, title_matches, "title",
            settings.EXACT_TITLE_BOOST, settings.PARTIAL_TITLE_BOOST,
        )

    def _fetch_title_match_docs(
        self,
        source_ids: List[str],
        title_matches: Dict[str, str],
    ) -> List[Dict[str, Any]]:
        return self._fetch_field_match_docs(
            source_ids, title_matches, "title",
            settings.EXACT_TITLE_BOOST, settings.PARTIAL_TITLE_BOOST,
        )

    def _get_unique_source_documents(self, request: PrioritizedSearchRequest) -> PrioritizedSearchResponse:
        """
        Retrieve unique source documents without query-based ranking.
        
        Used when no search query is provided. Returns one document per unique source_id,
        optionally filtered by categories, organizations, resource_types, and file_types.
        
        Args:
            request: Search request with filters and top_k (no query)
            
        Returns:
            PrioritizedSearchResponse with unique source documents
        """
        try:
            filter_conditions = self._build_filters(
                categories=request.categories,
                organizations=request.organizations,
                resource_types=request.resource_type,
                file_types=request.file_type,
                exclude_organizations=request.exclude_organizations,
                exclude_file_types=request.exclude_file_type,
                any_of=request.any_of
            )
            
            if filter_conditions:
                logger.info("Getting unique sources with filters")
            else:
                logger.info("Getting all unique sources (no filters)")
            
            unique_sources = self._scroll_and_collect_unique_sources(filter_conditions)
            logger.info(f"Found {len(unique_sources)} unique sources")
            
            unique_documents = list(unique_sources.values())
            top_k = min(request.top_k, len(unique_documents))
            top_results = unique_documents[:top_k]
            
            result_items = self._build_source_result_items(top_results)
            
            search_config = {
                "search_fields": [],
                "weights": {},
                "priority_order": [],
                "filters_applied": filter_conditions is not None,
                "mode": "unique_source_id"
            }
            
            logger.info(f"Returning {len(result_items)} unique source documents")
            
            return PrioritizedSearchResponse(
                query=None,
                total_results=len(unique_documents),
                top_k=top_k,
                results=result_items,
                search_config=search_config
            )
            
        except Exception as e:
            logger.error(f"Failed to get unique source documents: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to retrieve unique source documents: {str(e)}")
