import json
import logging
from typing import List, Dict, Optional
from qdrant_client import models
from app.core.clients.qdrant import qdrant_client, ensure_collections_exist
from app.config import settings

logger = logging.getLogger(__name__)


class BaseDocumentOperation:
    """Base class for document operations with common functionality"""

    async def ensure_collections(self):
        """Ensure collections exist before operations"""
        await ensure_collections_exist()

    def build_filter(self, source_id: str, company_id: Optional[str] = None) -> models.Filter:
        """Build Qdrant filter with source_id and optional company_id"""
        filter_conditions = [
            models.FieldCondition(
                key="source_id",
                match=models.MatchValue(value=source_id),
            )
        ]

        if company_id:
            filter_conditions.append(
                models.FieldCondition(
                    key="metadata.company",
                    match=models.MatchValue(value=company_id),
                )
            )

        return models.Filter(must=filter_conditions)

    def check_documents_exist(self, source_id: str, company_id: Optional[str] = None) -> bool:
        """Check if documents exist with given source_id and company_id"""
        try:
            scroll_filter = self.build_filter(source_id, company_id)

            search_response = qdrant_client.scroll(
                collection_name=settings.COLLECTION_NAME,
                scroll_filter=scroll_filter,
                limit=1,
            )

            return len(search_response[0]) > 0

        except Exception as e:
            logger.error(f"Error checking existing documents: {str(e)}")
            return False

    def count_documents(self, source_id: str, company_id: Optional[str] = None) -> int:
        """Count documents with given source_id and company_id"""
        try:
            scroll_filter = self.build_filter(source_id, company_id)

            result = qdrant_client.count(
                collection_name=settings.COLLECTION_NAME,
                count_filter=scroll_filter,
            )

            return result.count

        except Exception as e:
            logger.error(f"Error counting documents: {str(e)}")
            return 0

    def parse_metadata(self, metadata: str) -> dict:
        """Parse metadata JSON string and ensure it's a dict"""
        if not metadata:
            return {}
        try:
            parsed = json.loads(metadata)
            if not isinstance(parsed, dict):
                logger.warning(f"Metadata must be a JSON dict/object, got {type(parsed).__name__}. Ignoring metadata.")
                return {}
            return parsed
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid metadata JSON provided: {str(e)}. Ignoring metadata.")
            return {}

    def validate_source_id(self, source_id: str) -> None:
        """Validate source_id is provided"""
        if not source_id or not source_id.strip():
            from fastapi import HTTPException
            raise HTTPException(
                status_code=400,
                detail="source_id is required and cannot be empty"
            )

    def validate_priority(self, priority: str) -> None:
        """Validate priority format"""
        if not priority or not priority.upper().startswith("P"):
            from fastapi import HTTPException
            raise HTTPException(
                status_code=400,
                detail="Invalid priority format. Must be P1, P2, P3, etc."
            )
