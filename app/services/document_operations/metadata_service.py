import logging
from datetime import datetime
from typing import Dict, Optional
from fastapi import HTTPException
from qdrant_client import models
from app.services.document_operations.base_operation import BaseDocumentOperation
from app.core.clients.qdrant import qdrant_client
from app.config import settings

logger = logging.getLogger(__name__)


class MetadataService(BaseDocumentOperation):
    def _validate_metadata_update(self, source_id: str, metadata_updates: Dict, company_id: Optional[str]):
        """Validate metadata update inputs and return the normalized (source_id, company_id)"""
        # Upload stores both ids stripped, so the set_payload filter must use the same
        # values; normalize first so the company check below compares stripped ids too.
        source_id = self.validate_source_id(source_id)
        company_id = self.normalize_company_id(company_id)
        # A JSON list/string/number would otherwise fail later as a 500
        if not isinstance(metadata_updates, dict):
            raise HTTPException(
                status_code=400,
                detail="metadata_updates must be a JSON object"
            )
        if not metadata_updates:
            raise HTTPException(
                status_code=400,
                detail="metadata_updates cannot be empty"
            )

        # Prevent changing company_id through metadata update
        if company_id and 'company' in metadata_updates:
            if metadata_updates['company'] != company_id:
                raise HTTPException(
                    status_code=400,
                    detail="Cannot change company_id through metadata update"
                )

        return source_id, company_id

    async def update_metadata(self, source_id: str, metadata_updates: Dict,
                              company_id: Optional[str] = None):
        """Update only the metadata of existing documents without reprocessing"""
        try:
            # Validate inputs; the filter, 404 message and response use the normalized ids
            source_id, company_id = self._validate_metadata_update(source_id, metadata_updates, company_id)

            # Ensure collections exist
            await self.ensure_collections()

            # Build filter with company_id if provided
            scroll_filter = self.build_filter(source_id, company_id)

            # Count matching chunks up front; count errors surface as 500, not a false 404
            total_updated = qdrant_client.count(
                collection_name=settings.COLLECTION_NAME,
                count_filter=scroll_filter,
                exact=True,
            ).count

            if total_updated == 0:
                detail_msg = f"No documents found with source_id: {source_id}"
                if company_id:
                    detail_msg += f" and company_id: {company_id}"
                raise HTTPException(
                    status_code=404,
                    detail=detail_msg
                )

            # Merge only the given keys into the nested metadata object, server-side, in one call.
            # No stale payload is written back, so concurrent updates to other keys are kept.
            qdrant_client.set_payload(
                collection_name=settings.COLLECTION_NAME,
                payload={**metadata_updates, "updated_at": datetime.now().isoformat()},
                key="metadata",
                points=models.FilterSelector(filter=scroll_filter),
            )
            logger.info(f"Updated metadata for {total_updated} documents with source_id: {source_id}")

            return {
                "status": "success",
                "message": f"Successfully updated metadata for {total_updated} documents",
                "documents_updated": total_updated,
                "source_id": source_id,
                "company_id": company_id,
                "metadata_updates": metadata_updates
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Metadata update failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Metadata update failed: {str(e)}")
