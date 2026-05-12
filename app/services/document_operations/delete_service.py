import logging
import traceback
from typing import Optional
from fastapi import HTTPException
from qdrant_client.http import models
from app.services.document_operations.base_operation import BaseDocumentOperation
from app.core.clients.qdrant import qdrant_client
from app.config import settings

logger = logging.getLogger(__name__)


class DeleteService(BaseDocumentOperation):
    async def delete(self, source_id: str, company_id: Optional[str] = None):
        """Delete all documents with the given source_id and optional company_id"""
        try:
            # Validate inputs
            self.validate_source_id(source_id)

            # Ensure collections exist
            await self.ensure_collections()

            # Build filter with company_id if provided
            scroll_filter = self.build_filter(source_id, company_id)

            total_deleted = 0
            batch_size = 100

            while True:
                # Search for documents with pagination
                search_response = qdrant_client.scroll(
                    collection_name=settings.COLLECTION_NAME,
                    scroll_filter=scroll_filter,
                    limit=batch_size,
                )

                # If no more records found, break the loop
                if not search_response[0]:
                    break

                point_ids = [point.id for point in search_response[0]]

                # Delete the batch of points from Qdrant
                qdrant_client.delete(
                    collection_name=settings.COLLECTION_NAME,
                    points_selector=models.PointIdsList(points=point_ids),
                )

                total_deleted += len(point_ids)
                logger.info(f"Deleted batch of {len(point_ids)} documents. Total deleted: {total_deleted}")

                # If less than batch_size records were returned, we've reached the end
                if len(point_ids) < batch_size:
                    break

            if total_deleted == 0:
                detail_msg = f"No documents found for source ID: {source_id}"
                if company_id:
                    detail_msg += f" and company ID: {company_id}"
                raise HTTPException(
                    status_code=404,
                    detail=detail_msg,
                )

            success_msg = f"Successfully deleted all {total_deleted} documents with source ID: {source_id}"
            if company_id:
                success_msg += f" and company ID: {company_id}"

            return {
                "status": "success",
                "message": success_msg,
                "documents_deleted": total_deleted,
                "source_id": source_id,
                "company_id": company_id
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Delete failed: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")
