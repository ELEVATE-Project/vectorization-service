import logging
from datetime import datetime
from typing import Optional
from fastapi import HTTPException
from qdrant_client import models
from app.services.document_operations.base_operation import BaseDocumentOperation
from app.core.clients.qdrant import qdrant_client
from app.config import settings
from app.constants import messages as msg

logger = logging.getLogger(__name__)


class ThemeService(BaseDocumentOperation):
    def _validate_theme_update(self, source_id: str, theme: Optional[str], company_id: Optional[str]):
        """Validate theme update inputs and return the normalized (source_id, company_id, theme)"""
        # Upload stores both ids stripped, so the filter must use the same values;
        # a whitespace-only company_id is a 400, never "every company".
        source_id = self.validate_source_id(source_id)
        company_id = self.normalize_company_id(company_id)

        # theme is optional on upload but required here; an empty value never clears it.
        # validate_theme then applies the shared normalization (strip, collapse whitespace).
        if theme is None or (isinstance(theme, str) and not theme.strip()):
            raise HTTPException(status_code=400, detail=msg.THEME_REQUIRED)
        if not isinstance(theme, str):
            raise HTTPException(status_code=400, detail=msg.THEME_NOT_A_STRING)
        theme = self.validate_theme(theme, {})

        return source_id, company_id, theme

    async def update_theme(self, source_id: str, theme: Optional[str], company_id: Optional[str] = None):
        """Set the top-level theme on every chunk of a document without reprocessing it"""
        try:
            # Validate inputs; the filter, 404 message and response use the normalized values
            source_id, company_id, theme = self._validate_theme_update(source_id, theme, company_id)

            # Ensure collections exist
            await self.ensure_collections()

            # Build filter with company_id if provided
            theme_filter = self.build_filter(source_id, company_id)

            # Count matching chunks up front; count errors surface as 500, not a false 404
            total_updated = qdrant_client.count(
                collection_name=settings.COLLECTION_NAME,
                count_filter=theme_filter,
                exact=True,
            ).count

            if total_updated == 0:
                detail_msg = (
                    msg.DOCUMENTS_NOT_FOUND_FOR_COMPANY.format(source_id=source_id, company_id=company_id)
                    if company_id else msg.DOCUMENTS_NOT_FOUND.format(source_id=source_id)
                )
                raise HTTPException(status_code=404, detail=detail_msg)

            # No key=: merges theme into the top-level payload and keeps every other key.
            # Payload only, so text, dense vectors and bm25 are never touched (unlike upsert).
            points = models.FilterSelector(filter=theme_filter)
            qdrant_client.set_payload(
                collection_name=settings.COLLECTION_NAME,
                payload={"theme": theme},
                points=points,
            )

            # The theme is already stored, so a failure here is not rolled back; a retry
            # re-sets the same theme and refreshes updated_at.
            try:
                qdrant_client.set_payload(
                    collection_name=settings.COLLECTION_NAME,
                    payload={"updated_at": datetime.now().isoformat()},
                    key="metadata",
                    points=points,
                )
            except Exception as e:
                logger.error(
                    f"Theme set for source_id {source_id} (company_id {company_id}) on {total_updated} "
                    f"chunks, but metadata.updated_at was not refreshed: {str(e)}"
                )
                raise HTTPException(
                    status_code=502,
                    detail=msg.THEME_UPDATED_AT_NOT_REFRESHED.format(
                        count=total_updated, source_id=source_id, error=str(e)
                    )
                )
            logger.info(f"Updated theme for {total_updated} documents with source_id: {source_id}")

            return {
                "status": "success",
                "message": msg.THEME_UPDATE_SUCCEEDED.format(count=total_updated),
                "documents_updated": total_updated,
                "source_id": source_id,
                "company_id": company_id,
                "theme": theme
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Theme update failed: {str(e)}")
            raise HTTPException(status_code=500, detail=msg.THEME_UPDATE_FAILED.format(error=str(e)))
