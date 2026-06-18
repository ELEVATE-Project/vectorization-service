import logging
from typing import List, Set
from fastapi import HTTPException
from qdrant_client import models
from app.core.clients.qdrant import qdrant_client
from app.models.api_models import SourceVerificationRequest, SourceVerificationResponse
from app.config import settings

logger = logging.getLogger(__name__)


class SourceVerificationService:
    """Service for verifying source IDs existence in Qdrant"""

    def verify_sources(self, request: SourceVerificationRequest) -> SourceVerificationResponse:
        """
        Verify which source IDs exist in Qdrant
        
        Args:
            request: SourceVerificationRequest containing source_ids list
            
        Returns:
            SourceVerificationResponse with found and not_found lists
        """
        try:
            # Handle empty request
            if not request.source_ids:
                logger.warning("Empty source_ids list provided")
                return SourceVerificationResponse(
                    total_requested=0,
                    found=[],
                    not_found=[],
                    found_count=0,
                    not_found_count=0
                )

            # Remove duplicates while preserving order
            unique_source_ids = list(dict.fromkeys(request.source_ids))
            logger.info(f"Verifying {len(unique_source_ids)} unique source IDs (from {len(request.source_ids)} total)")

            # Track found source IDs
            found_source_ids: Set[str] = set()

            # Query Qdrant for each source_id
            for source_id in unique_source_ids:
                try:
                    # Create filter for this source_id
                    scroll_filter = models.Filter(
                        must=[
                            models.FieldCondition(
                                key="metadata.source_id",
                                match=models.MatchValue(value=source_id)
                            )
                        ]
                    )
                    
                    result = qdrant_client.scroll(
                        collection_name=settings.COLLECTION_NAME,
                        scroll_filter=scroll_filter,
                        limit=1,
                        with_payload=False,  # Don't need payload, just checking existence
                        with_vectors=False   # Don't need vectors either
                    )

                    # If we got any results, this source_id exists
                    if result[0]:  # result is a tuple (points, next_page_offset)
                        found_source_ids.add(source_id)
                        logger.debug(f"Source ID '{source_id}' found in Qdrant")
                    else:
                        logger.debug(f"Source ID '{source_id}' not found in Qdrant")

                except Exception as e:
                    logger.error(f"Error checking source_id '{source_id}': {str(e)}")
                    # Continue checking other source IDs even if one fails
                    continue

            # Categorize results
            found = [sid for sid in unique_source_ids if sid in found_source_ids]
            not_found = [sid for sid in unique_source_ids if sid not in found_source_ids]

            response = SourceVerificationResponse(
                total_requested=len(unique_source_ids),
                found=found,
                not_found=not_found,
                found_count=len(found),
                not_found_count=len(not_found)
            )

            logger.info(f"Verification complete: {len(found)} found, {len(not_found)} not found")
            return response

        except Exception as e:
            logger.error(f"Source verification failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Source verification failed: {str(e)}")


# Create singleton instance
source_verification_service = SourceVerificationService()
