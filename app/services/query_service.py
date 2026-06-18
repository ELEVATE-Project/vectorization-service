import json
import logging
import traceback
from datetime import datetime
from typing import Dict, Any, List
from fastapi import HTTPException
from qdrant_client.http import models
from sqlalchemy import func
from app.core.clients.qdrant import qdrant_client, ensure_collections_exist
from app.core.clients.embedding import generate_single_embedding
from app.core.clients.redis_cache import redis_cache
from app.core.database import SessionLocal
from app.models.db_models import TranslationRecord
from app.models.api_models import MultilingualQueryRequest, MultilingualQueryResponse
from app.utils.language_utils import detect_language, translate_text
from app.config import settings

logger = logging.getLogger(__name__)


class QueryService:
    """Service for processing multilingual queries and searching documents"""

    def __init__(self):
        self.similarity_threshold = settings.SIMILARITY_THRESHOLD
        self.search_limit = settings.VECTOR_SEARCH_LIMIT

    async def process_query(self, request: MultilingualQueryRequest) -> MultilingualQueryResponse:
        """Query documents with multilingual support"""
        try:
            logger.debug(f"Received query request: {request.dict()}")

            # Check Redis cache first
            cache_key = f"{request.query}_{request.priority_filter}"
            logger.debug(f"Checking cache with key: {cache_key}")

            cached_response = redis_cache.get(cache_key)
            if cached_response:
                logger.info(f"Cache hit for query: {request.query}")
                logger.debug(f"Returning cached response: {cached_response}")
                return MultilingualQueryResponse(**cached_response)

            # Ensure collections exist
            logger.debug("Ensuring collections exist")
            await ensure_collections_exist()

            # Process the query language
            query_info = self._process_query_language(request.query)

            # Generate query embedding
            search_query = query_info["translated"] or query_info["original"]
            logger.debug(f"Generating embedding for query: {search_query}")
            query_embedding = generate_single_embedding(search_query)
            logger.debug(f"Generated embedding shape: {query_embedding.shape}")

            # Search for relevant documents
            search_results = self._search_documents(
                query_embedding, request.priority_filter, request.search_limit
            )

            if not search_results:
                logger.info("No search results found")
                response = MultilingualQueryResponse(
                    relevant_texts=[],
                    original_query=query_info["original"],
                    translated_query=query_info["translated"],
                    language=query_info["language"],
                )
                logger.debug("Caching empty response")
                self._cache_response(cache_key, response)
                return response

            # Process search results with translations
            processed_results = self._process_search_results(
                search_results, query_info["language"] == "hi"
            )

            # Create final response
            response = MultilingualQueryResponse(
                relevant_texts=processed_results,
                original_query=query_info["original"],
                translated_query=query_info["translated"],
                language=query_info["language"]
            )

            # Cache the response
            self._cache_response(cache_key, response)

            return response

        except Exception as e:
            logger.error(f"Query failed: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=str(e))

    def _process_query_language(self, query: str) -> Dict[str, Any]:
        """Process query - translation disabled"""
        # Translation disabled - return query as-is
        return {
            "original": query,
            "translated": None,
            "language": "en",
        }

    def _search_documents(self, query_embedding, priority_filter: str, search_limit: int) -> List:
        """Search documents in Qdrant with priority filtering"""
        all_search_results = []

        if priority_filter:
            # Search within specified priority
            priority_level = priority_filter.upper()
            logger.debug(f"Searching with priority filter: {priority_level}")

            search_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.priority",
                        match=models.MatchValue(value=priority_level),
                    )
                ]
            )

            logger.debug(f"Executing priority-filtered search with limit: {search_limit}")
            search_results = qdrant_client.query_points(
                collection_name=settings.COLLECTION_NAME,
                query=query_embedding.tolist(),
                using="text",
                limit=search_limit,
                query_filter=search_filter,
                score_threshold=self.similarity_threshold,
            ).points
            logger.debug(f"Found {len(search_results)} results for priority {priority_level}")
            all_search_results.extend(search_results)

        else:
            # Search across all priorities in order (P1, P2, P3)
            priorities = ["P1", "P2", "P3"]
            remaining_limit = search_limit
            logger.debug(f"Searching across all priorities with initial limit: {remaining_limit}")

            for priority in priorities:
                if remaining_limit <= 0:
                    logger.debug("Search limit reached, stopping priority iteration")
                    break

                search_filter = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="metadata.priority",
                            match=models.MatchValue(value=priority),
                        )
                    ]
                )

                logger.debug(f"Searching priority {priority} with limit {remaining_limit}")
                priority_results = qdrant_client.query_points(
                    collection_name=settings.COLLECTION_NAME,
                    query=query_embedding.tolist(),
                    using="text",
                    limit=remaining_limit,
                    query_filter=search_filter,
                    score_threshold=self.similarity_threshold,
                ).points

                logger.debug(f"Found {len(priority_results)} results for priority {priority}")
                all_search_results.extend(priority_results)
                remaining_limit -= len(priority_results)

        return all_search_results

    def _process_search_results(self, search_results: List, is_hindi_query: bool) -> List[Dict[str, Any]]:
        """Process search results and handle translations"""
        processed_results = []
        logger.debug(f"Processing {len(search_results)} search results")

        # Get database session
        db = SessionLocal()
        try:
            for hit in search_results:
                metadata = hit.payload["metadata"]
                content_text = hit.payload["text"]
                chunk_id = str(hit.id).replace("-", "")

                logger.debug(f"Processing content for chunk: {chunk_id}, Score: {hit.score}")
                logger.debug(f"Metadata: {metadata}")

                try:
                    # Try to find translation record
                    translation_record = self._find_translation_record(db, chunk_id)

                    # Process the content based on translation record
                    display_text, translated_text = self._process_content_translation(
                        content_text, metadata, translation_record, is_hindi_query
                    )
                    processed_result = dict(hit.payload)

                    processed_result.update({
                        "qdrant_recommendation_text": display_text,
                        "translated_text": translated_text,
                        "relevance_score": float(hit.score),
                        "metadata": metadata,
                        "priority": metadata.get("priority", "N/A"),
                        "chunk_id": chunk_id,
                    })

                    processed_results.append(processed_result)

                except Exception as e:
                    logger.error(f"Error processing result {chunk_id}: {str(e)}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    continue

        finally:
            logger.debug("Closing database session")
            db.close()

        return processed_results

    def _find_translation_record(self, db, chunk_id: str):
        """Find translation record with multiple fallback strategies"""
        translation_record = None

        logger.debug(f"Attempting to find translation record for chunk_id: {chunk_id}")

        # 1. Direct query with string conversion
        translation_record = (
            db.query(TranslationRecord)
            .filter(TranslationRecord.chunk_id == str(chunk_id))
            .first()
        )

        if not translation_record:
            logger.debug("Direct query failed, trying with stripped chunk_id")
            # 2. Try with stripped chunk_id
            cleaned_chunk_id = str(chunk_id).strip()
            translation_record = (
                db.query(TranslationRecord)
                .filter(TranslationRecord.chunk_id == cleaned_chunk_id)
                .first()
            )

        if not translation_record:
            logger.debug("Stripped query failed, trying case-insensitive query")
            # 3. Try case-insensitive query
            translation_record = (
                db.query(TranslationRecord)
                .filter(
                    func.lower(TranslationRecord.chunk_id) == func.lower(str(chunk_id))
                )
                .first()
            )

        if not translation_record:
            # Log all chunk_ids in DB for debugging (first 5)
            all_ids = db.query(TranslationRecord.chunk_id).limit(5).all()
            logger.debug(f"Translation record not found. Sample chunk_ids in DB: {[id[0] for id in all_ids]}")
        else:
            logger.debug("Translation record found")

        return translation_record

    def _process_content_translation(self, content_text: str, metadata: Dict, translation_record, is_hindi_query: bool):
        """Process content translation based on query language and available translations"""
        logger.debug(f"Processing content. Is Hindi: {metadata.get('is_hindi', False)}")

        if metadata.get("is_hindi", False):
            if translation_record:
                # Both Hindi and English queries get the same result
                display_text = content_text  # English (stored in Qdrant)
                translated_text = translation_record.original_text  # Hindi
                logger.debug("Using English display text with Hindi translation")
            else:
                logger.warning("Translation record not found for Hindi content")
                display_text = content_text
                translated_text = None
        else:
            display_text = content_text
            if not is_hindi_query:
                translated_text = None
            elif translation_record:
                translated_text = translation_record.original_text
            else:
                translated_text = None
            logger.debug(f"Non-Hindi content processed. Translation needed: {is_hindi_query}")

        return display_text, translated_text

    def _cache_response(self, cache_key: str, response: MultilingualQueryResponse):
        """Cache the response if caching is enabled"""
        if settings.REDIS_CACHE_ENABLED:
            # Clean the response before caching
            cache_data = {
                "relevant_texts": self._clean_response_for_cache(response.relevant_texts),
                "original_query": response.original_query,
                "translated_query": response.translated_query,
                "language": response.language
            }
            redis_cache.set(cache_key, cache_data)

    def _clean_response_for_cache(self, relevant_texts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Clean response data for caching (handle NaN values, etc.)"""
        cleaned_responses = []
        for response in relevant_texts:
            cleaned_response = {
                "qdrant_recommendation_text": response.get("qdrant_recommendation_text", ""),
                "translated_text": response.get("translated_text", ""),
                "relevance_score": float(response.get("relevance_score", 0)),
                "metadata": response.get("metadata", {}),
                "priority": response.get("priority", "N/A"),
                "chunk_id": response.get("chunk_id", "")
            }
            cleaned_responses.append(cleaned_response)
        return cleaned_responses
