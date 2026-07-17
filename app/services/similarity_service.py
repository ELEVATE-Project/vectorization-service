import logging
from typing import List, Dict, Any
from fastapi import HTTPException
from qdrant_client import models
from app.core.clients.qdrant import qdrant_client
from app.core.clients import embedding
from app.config import settings
from app.models.api_models import SimilarityCheckRequest, SimilarityCheckResponse

logger = logging.getLogger(__name__)


class SimilarityService:
    """Service for checking document similarity"""

    def __init__(self):
        self.threshold_default = 0.85

    def check_similarity(self, request: SimilarityCheckRequest) -> SimilarityCheckResponse:
        """Check if similar content already exists in the vector database"""
        try:
            logger.info(f"Checking similarity for text: {request.text[:100]}...")

            # Generate + validate the embedding (limit to 1000 chars for performance).
            # embed_query rejects empty/whitespace text and malformed vectors before Qdrant.
            text_vector = embedding.embed_query(request.text[:1000])

            # Build search filter
            filter_conditions = [
                models.FieldCondition(
                    key="metadata.company",
                    match=models.MatchValue(value=request.company_id)
                )
            ]

            # Exclude specific source_id if provided.
            # FieldCondition has no `invert` flag (it was silently ignored before) —
            # exclusion belongs in the filter's must_not clause.
            must_not_conditions = []
            if request.exclude_source_id:
                must_not_conditions.append(
                    models.FieldCondition(
                        key="source_id",
                        match=models.MatchValue(value=request.exclude_source_id),
                    )
                )

            search_filter = models.Filter(
                must=filter_conditions,
                must_not=must_not_conditions or None,
            )

            # Search for similar documents on the 'text' named vector.
            # search() was removed in qdrant-client>=1.14; the Query API requires a
            # named vector since the collection is configured with named vectors.
            logger.debug(f"Searching with threshold: {request.threshold}")
            search_results = qdrant_client.query_points(
                collection_name=settings.COLLECTION_NAME,
                query=text_vector,
                using="text",
                limit=5,  # Get top 5 similar documents
                query_filter=search_filter,
                score_threshold=request.threshold,
                with_payload=True,
            )

            # Process results
            similar_docs = []
            for hit in search_results.points:
                similar_doc = {
                    "source_id": hit.payload.get("source_id"),
                    "similarity_score": float(hit.score),
                    "metadata": hit.payload.get("metadata", {}),
                    "text_preview": hit.payload.get("text", "")[:200] + "...",
                    "chunk_id": str(hit.id)
                }
                similar_docs.append(similar_doc)

                logger.debug(
                    f"Found similar document: {similar_doc['source_id']} with score: {similar_doc['similarity_score']}")

            has_similar = len(similar_docs) > 0

            logger.info(f"Similarity check completed. Found {len(similar_docs)} similar documents")

            return SimilarityCheckResponse(
                has_similar=has_similar,
                similar_documents=similar_docs
            )

        except embedding.EmbeddingError:
            # Empty/invalid query vector — let the global handler return HTTP 422.
            raise
        except Exception as e:
            logger.error(f"Similarity check failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Similarity check failed: {str(e)}")

    def find_duplicates_in_collection(self, company_id: str) -> List[Dict[str, Any]]:
        """Find potential duplicate documents within a company's collection"""
        try:
            # This could be used for cleanup operations
            # Implementation would involve comparing all documents against each other
            # For now, return empty list as this is an advanced feature
            logger.info(f"Finding duplicates for company: {company_id}")
            return []

        except Exception as e:
            logger.error(f"Duplicate detection failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Duplicate detection failed: {str(e)}")
