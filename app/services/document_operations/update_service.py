import logging
import json
from typing import List, Optional
from fastapi import HTTPException, UploadFile
from app.services.document_operations.base_operation import BaseDocumentOperation
from app.services.document_operations.upload_service import UploadService
from app.constants import constants as const

logger = logging.getLogger(__name__)


class UpdateService(BaseDocumentOperation):
    def __init__(self):
        self.upload_service = UploadService()

    def _parse_metadata(self, metadata: str = None) -> dict:
        """Parse metadata from JSON string to dict"""
        if not metadata or not metadata.strip():
            return {}
        try:
            parsed = json.loads(metadata)
            if not isinstance(parsed, dict):
                logger.warning("Metadata should be a JSON object/dict, using empty dict")
                return {}
            return parsed
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid metadata JSON: {str(e)}, using empty dict")
            return {}

    async def _replace(self, file: UploadFile, priority: str, metadata_dict: dict,
                       source_id: str, company_id: Optional[str], require_existing: bool,
                       title: Optional[str] = None, summary: Optional[str] = None,
                       tags: Optional[List[str]] = None, theme: Optional[str] = None):
        """Store the new version, then delete the old chunks by id; returns (old_ids, upload_result)"""
        # Snapshot the old version's point ids before anything is written; a Qdrant error
        # here propagates as a 500 instead of being mistaken for "no document".
        old_ids = self.collect_point_ids(source_id, company_id)

        if require_existing and not old_ids:
            detail_msg = f"No documents found with source_id: {source_id}"
            if company_id:
                detail_msg += f" and company_id: {company_id}"
            detail_msg += ". Use upload endpoint for new documents."
            raise HTTPException(status_code=404, detail=detail_msg)

        # Upload before deleting: any validation, parsing, embedding or Qdrant failure raises
        # here while the old version is still intact (new chunks get fresh uuid point ids).
        # title/summary/tags/theme go through the same validation as POST; a bad value is
        # a 400 raised here, before the old version is touched.
        upload_result = await self.upload_service.process(
            file, priority, metadata_dict, source_id, company_id,
            title=title, summary=summary, tags=tags, theme=theme
        )

        # Delete exactly the old ids, never by filter: the new chunks share this source_id,
        # so a filter delete would also remove the version that was just stored.
        if old_ids:
            await self._delete_old_version(old_ids, source_id, company_id)

        return old_ids, upload_result

    async def _delete_old_version(self, old_ids: list, source_id: str, company_id: Optional[str]):
        """Remove the replaced chunks, retrying; 502 if they are still there afterwards"""
        # Same in-request retries as a partial-upload rollback (UPLOAD_ROLLBACK_* settings);
        # the reason label keeps these log lines from reading as a failed upload.
        if await self.upload_service._rollback_points(
            old_ids, source_id, reason=const.ROLLBACK_REASON_OLD_VERSION_CLEANUP
        ):
            return

        # The new version is stored but the old one could not be removed: log every id for
        # manual cleanup, keep retrying in the background, and tell the caller the truth.
        logger.error(
            f"Could not remove the old version of source_id {source_id} (company_id {company_id}) after "
            f"storing the new one; {len(old_ids)} old points may remain searchable. Point IDs: {old_ids}"
        )
        self.upload_service._schedule_background_rollback(
            old_ids, source_id, reason=const.ROLLBACK_REASON_OLD_VERSION_CLEANUP
        )

        # A retry is safe: it snapshots old + new ids, stores another copy, then deletes both
        raise HTTPException(
            status_code=502,
            detail=(
                f"The new version of source_id {source_id} was stored, but {len(old_ids)} chunks of the "
                f"old version could not be removed yet. Cleanup is retrying in the background; the old "
                f"chunks may be searchable until it succeeds. Retrying this request is safe."
            ),
        )

    async def update(self, file: UploadFile, priority: str, metadata: str = None,
                     source_id: str = None, company_id: str = None,
                     title: Optional[str] = None, summary: Optional[str] = None,
                     tags: Optional[List[str]] = None, theme: Optional[str] = None):
        """Update existing documents by replacing all documents with the same source_id and company_id"""
        try:
            # Use the normalized ids: upload stores them stripped, so the exists-check
            # and delete must filter on the same value or a padded id returns a false 404.
            source_id = self.validate_source_id(source_id)
            company_id = self.normalize_company_id(company_id)
            self.validate_priority(priority)

            # Parse metadata string to dict
            metadata_dict = self._parse_metadata(metadata)

            # Replace: 404 if nothing is stored yet; a failed upload leaves the old version as is
            old_ids, upload_result = await self._replace(
                file, priority, metadata_dict, source_id, company_id, require_existing=True,
                title=title, summary=summary, tags=tags, theme=theme
            )

            return {
                "status": "success",
                "operation": "update",
                "message": f"Successfully updated documents for source_id: {source_id}",
                "documents_deleted": len(old_ids),
                "previous_document_count": len(old_ids),
                "chunks_processed": upload_result["chunks_processed"],
                "points_uploaded": upload_result["points_uploaded"],
                "upload_failures": upload_result["upload_failures"],
                "file_type": upload_result["file_type"],
                "priority": upload_result["priority"],
                "source_id": source_id,
                "company_id": company_id,
                "title": upload_result.get("title"),
                "summary": upload_result.get("summary"),
                "tags": upload_result.get("tags"),
                "theme": upload_result.get("theme"),
                "sample_chunk": upload_result.get("sample_chunk")
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Update failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Update failed: {str(e)}")

    async def upsert(self, file: UploadFile, priority: str, metadata: str = None,
                     source_id: str = None, company_id: str = None,
                     title: Optional[str] = None, summary: Optional[str] = None,
                     tags: Optional[List[str]] = None, theme: Optional[str] = None):
        """Upsert documents - update if exists, create if not"""
        try:
            # Use the normalized ids: with a padded id the exists-check missed the stored
            # chunks, so upsert re-uploaded next to them and left duplicates behind.
            source_id = self.validate_source_id(source_id)
            company_id = self.normalize_company_id(company_id)
            self.validate_priority(priority)

            # Parse metadata string to dict
            metadata_dict = self._parse_metadata(metadata)

            # Replace if stored, create if not; a failed upload leaves any old version as is
            old_ids, upload_result = await self._replace(
                file, priority, metadata_dict, source_id, company_id, require_existing=False,
                title=title, summary=summary, tags=tags, theme=theme
            )

            operation = "updated" if old_ids else "created"

            return {
                "status": "success",
                "operation": operation,
                "message": f"Successfully {operation} documents for source_id: {source_id}",
                "documents_deleted": len(old_ids),
                "previous_document_count": len(old_ids),
                "chunks_processed": upload_result["chunks_processed"],
                "points_uploaded": upload_result["points_uploaded"],
                "upload_failures": upload_result["upload_failures"],
                "file_type": upload_result["file_type"],
                "priority": upload_result["priority"],
                "source_id": source_id,
                "company_id": company_id,
                "title": upload_result.get("title"),
                "summary": upload_result.get("summary"),
                "tags": upload_result.get("tags"),
                "theme": upload_result.get("theme"),
                "sample_chunk": upload_result.get("sample_chunk")
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Upsert failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Upsert failed: {str(e)}")
