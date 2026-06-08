import logging
from datetime import datetime
from typing import Dict, Optional
from fastapi import HTTPException
from qdrant_client.http import models
from app.services.document_operations.base_operation import BaseDocumentOperation
from app.core.clients.qdrant import qdrant_client
from app.config import settings

logger = logging.getLogger(__name__)


class MetadataService(BaseDocumentOperation):
    def _validate_metadata_update(self, source_id: str, metadata_updates: Dict):
        """Validate metadata update inputs"""
        self.validate_source_id(source_id)
        if not metadata_updates:
            raise HTTPException(
                status_code=400,
                detail="metadata_updates cannot be empty"
            )
    
    def _merge_and_validate_metadata(self, point_metadata: Dict, metadata_updates: Dict, company_id: Optional[str]) -> Dict:
        """Merge existing metadata with updates and validate company_id"""
        updated_metadata = point_metadata.copy()
        updated_metadata.update(metadata_updates)
        updated_metadata["updated_at"] = datetime.now().isoformat()
        
        # Prevent changing company_id through metadata update
        if company_id and 'company' in metadata_updates:
            if metadata_updates['company'] != company_id:
                raise HTTPException(
                    status_code=400,
                    detail="Cannot change company_id through metadata update"
                )
        
        return updated_metadata
    
    def _update_point_metadata(self, point, metadata_updates: Dict, company_id: Optional[str]):
        """Update metadata for a single point"""
        existing_metadata = point.payload.get("metadata", {})
        updated_metadata = self._merge_and_validate_metadata(
            existing_metadata, metadata_updates, company_id
        )
        
        updated_payload = point.payload.copy()
        updated_payload["metadata"] = updated_metadata
        
        qdrant_client.set_payload(
            collection_name=settings.COLLECTION_NAME,
            payload=updated_payload,
            points=[point.id],
        )
    
    def _process_batch(self, points, metadata_updates: Dict, company_id: Optional[str]) -> int:
        """Process a batch of points for metadata update"""
        for point in points:
            self._update_point_metadata(point, metadata_updates, company_id)
        return len(points)

    async def update_metadata(self, source_id: str, metadata_updates: Dict,
                              company_id: Optional[str] = None):
        """Update only the metadata of existing documents without reprocessing"""
        try:
            # Validate inputs
            self._validate_metadata_update(source_id, metadata_updates)

            # Ensure collections exist
            await self.ensure_collections()

            # Build filter with company_id if provided
            scroll_filter = self.build_filter(source_id, company_id)

            total_updated = 0
            batch_size = 100

            while True:
                # Get documents to update
                search_response = qdrant_client.scroll(
                    collection_name=settings.COLLECTION_NAME,
                    scroll_filter=scroll_filter,
                    limit=batch_size,
                    with_payload=True,
                    with_vectors=False,
                )

                if not search_response[0]:
                    break

                # Process batch
                batch_count = self._process_batch(
                    search_response[0], metadata_updates, company_id
                )
                total_updated += batch_count
                logger.info(
                    f"Updated metadata for batch of {batch_count} documents. Total updated: {total_updated}")

                if len(search_response[0]) < batch_size:
                    break

            if total_updated == 0:
                detail_msg = f"No documents found with source_id: {source_id}"
                if company_id:
                    detail_msg += f" and company_id: {company_id}"
                raise HTTPException(
                    status_code=404,
                    detail=detail_msg
                )

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
