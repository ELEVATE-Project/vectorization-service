import logging
from typing import List, Dict, Optional, Any
from qdrant_client import models
from qdrant_client.models import SearchRequest, NamedVector
from app.core.clients.qdrant import qdrant_client
from app.core.clients.embedding import generate_embeddings
from app.config import settings
from app.models.api_models import (
    PrioritizedSearchRequest,
    PrioritizedSearchResponse,
    SearchResultItem
)

logger = logging.getLogger(__name__)


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
        else:
            logger.info("No filters applied")

    def _process_and_filter_results(self, all_results, field_scores, weights, search_fields, top_k, threshold, detail_filter_score=None):
        """Process, rank, filter and deduplicate results"""
        logger.info(f"Total documents matched: {len(all_results)}")
        
        logger.info("Calculating weighted scores and ranking results")
        ranked_results = self._rank_results(all_results, field_scores, weights, search_fields)
        logger.info(f"Ranked results: {len(ranked_results)} documents")
        
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

    def _build_result_items(self, top_results) -> List[SearchResultItem]:
        """Build result items from top results"""
        result_items = []
        for result_data in top_results:
            try:
                result_items.append(SearchResultItem(
                    id=str(result_data['id']),
                    text=result_data['payload'].get('text', ''),
                    title=result_data['payload'].get('title'),
                    summary=result_data['payload'].get('summary'),
                    tags=result_data['payload'].get('tags'),
                    metadata=result_data['payload'].get('metadata', {}),
                    source_id=result_data['payload'].get('source_id', ''),
                    score=result_data['weighted_score'],
                    field_scores=result_data['field_scores']
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
            
            logger.info(f"Generating embedding for query: '{query_for_embedding}'")
            try:
                query_embedding = generate_embeddings([query_for_embedding])[0]
            except Exception as e:
                logger.error(f"Failed to generate embeddings: {str(e)}")
                raise ValueError(f"Failed to generate embeddings for query: {str(e)}")
            
            filter_conditions = self._build_filters(
                categories=request.categories,
                organizations=request.organizations,
                resource_types=request.resource_type,
                file_types=request.file_type
            )
            
            self._log_search_request(request, top_k, filter_conditions)
            
            logger.info("========== EXECUTING SEARCH ==========" )
            logger.info("Starting parallel batch search across all fields")
            all_results, field_scores = self._parallel_batch_search(
                search_fields=search_fields,
                weights=weights,
                query_embedding=query_embedding,
                filter_conditions=filter_conditions,
                limit=top_k * 100000
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
            
            # Pass detail_filter_score if using field-level filtering
            top_results, unique_source_results = self._process_and_filter_results(
                all_results, field_scores, weights, search_fields, top_k, 
                filter_score if not use_detail_filter else 0,
                detail_filter_score if use_detail_filter else None
            )
            
            result_items = self._build_result_items(top_results)
            
            search_config = {
                "search_fields": search_fields,
                "weights": weights,
                "priority_order": self.priority_order,
                "filters_applied": filter_conditions is not None,
                "filter_mode": "detail_filter_score" if use_detail_filter else "filter_score",
                "filter_score": None if use_detail_filter else filter_score
            }
            
            if use_detail_filter:
                search_config["detail_filter_score"] = {
                    "title": detail_filter_score.title,
                    "text": detail_filter_score.text,
                    "tags": detail_filter_score.tags,
                    "summary": detail_filter_score.summary,
                    "metadata": detail_filter_score.metadata
                }
            
            logger.info("========== SEARCH COMPLETED ==========" )
            logger.info(f"Returned {len(result_items)} results from {len(unique_source_results)} unique sources")
            
            return PrioritizedSearchResponse(
                query=request.query,
                total_results=len(unique_source_results),
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
        
        for field in search_fields:
            if field not in weights:
                logger.warning(f"Field '{field}' not in weights config, skipping")
                continue
            
            search_requests.append(
                SearchRequest(
                    vector=NamedVector(name=field, vector=query_embedding.tolist()),
                    limit=limit,
                    with_payload=True,
                    filter=filter_conditions
                )
            )
            valid_fields.append(field)
        
        logger.info(f"Executing batch search across {len(valid_fields)} fields: {valid_fields}")
        batch_results = qdrant_client.search_batch(
            collection_name=self.collection_name,
            requests=search_requests
        )
        
        all_results = {}
        field_scores = {}
        
        for field, results in zip(valid_fields, batch_results):
            logger.info(f"Field '{field}' returned {len(results)} results")
            
            for result in results:
                point_id = result.id
                if point_id not in all_results:
                    all_results[point_id] = result
                    field_scores[point_id] = {}
                
                field_scores[point_id][field] = result.score
        
        logger.info(f"Batch search completed: {len(all_results)} unique documents found")
        return all_results, field_scores
    
    def _build_filters(
        self,
        categories: Optional[List[str]] = None,
        organizations: Optional[List[str]] = None,
        resource_types: Optional[List[str]] = None,
        file_types: Optional[List[str]] = None
    ) -> Optional[models.Filter]:
        """
        Build Qdrant filter conditions with intelligent AND/OR logic.
        
        Filter Logic:
        - Within each filter type: OR condition
        - Between filter types: AND condition
        
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
            
        Returns:
            Qdrant Filter object or None if no filters provided
        """
        must_conditions = []
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
        
        if must_conditions:
            logger.info(f"Total filters: {len(must_conditions)} (AND logic between types)")
            logger.info(f"Filter summary: {' AND '.join(filter_summary)}")
            return models.Filter(must=must_conditions)
        
        logger.info("No filters applied")
        return None
    
    def _rank_results(
        self,
        all_results: Dict[str, Any],
        field_scores: Dict[str, Dict[str, float]],
        weights: Dict[str, float],
        search_fields: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Rank results using weighted multi-field scoring.
        
        Scoring Formula:
        Final_Score = Σ(Field_Weight × Field_Score)
        
        Args:
            all_results: Dictionary of search results by point ID
            field_scores: Scores for each field per point
            weights: Weight configuration for each field
            search_fields: List of fields searched
            
        Returns:
            List of ranked results sorted by weighted score (descending)
        """
        ranked = []
        
        for point_id, result in all_results.items():
            weighted_score = 0.0
            field_score_dict = field_scores.get(point_id, {})
            
            # Calculate weighted score
            for field in search_fields:
                if field in field_score_dict and field in weights:
                    field_score = field_score_dict[field]
                    weight = weights[field]
                    weighted_score += field_score * weight
            
            # Ensure score doesn't exceed 1.0
            num_fields_matched = len(field_score_dict)
            weighted_score = min(weighted_score, 1.0)
            
            ranked.append({
                'id': result.id,
                'payload': result.payload,
                'weighted_score': weighted_score,
                'field_scores': field_score_dict,
                'num_fields_matched': num_fields_matched
            })
        
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
                field_score = field_score_dict.get(field, 0.0)
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
                file_types=request.file_type
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
