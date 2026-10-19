import logging
import json
from typing import Optional
from fastapi import HTTPException, UploadFile
from app.services.document_operations.base_operation import BaseDocumentOperation
from app.services.document_operations.upload_service import UploadService
from app.services.document_operations.delete_service import DeleteService

logger = logging.getLogger(__name__)


class UpdateService(BaseDocumentOperation):
    def __init__(self):
        self.upload_service = UploadService()
        self.delete_service = DeleteService()

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

    async def update(self, file: UploadFile, priority: str, metadata: str = None,
                     source_id: str = None, company_id: str = None):
        """Update existing documents by replacing all documents with the same source_id and company_id"""
        try:
            # Validate inputs
            self.validate_source_id(source_id)
            self.validate_priority(priority)

            # Parse metadata string to dict
            metadata_dict = self._parse_metadata(metadata)

            # Check if documents exist for this source_id and company_id
            existing_docs = self.check_documents_exist(source_id, company_id)

            if not existing_docs:
                detail_msg = f"No documents found with source_id: {source_id}"
                if company_id:
                    detail_msg += f" and company_id: {company_id}"
                detail_msg += ". Use upload endpoint for new documents."

                raise HTTPException(status_code=404, detail=detail_msg)

            # Count existing documents
            existing_count = self.count_documents(source_id, company_id)

            # Delete existing documents
            delete_result = await self.delete_service.delete(source_id, company_id)

            # Upload new documents with parsed metadata dict
            upload_result = await self.upload_service.process(
                file, priority, metadata_dict, source_id, company_id
            )

            return {
                "status": "success",
                "operation": "update",
                "message": f"Successfully updated documents for source_id: {source_id}",
                "documents_deleted": delete_result["documents_deleted"],
                "previous_document_count": existing_count,
                "chunks_processed": upload_result["chunks_processed"],
                "points_uploaded": upload_result["points_uploaded"],
                "upload_failures": upload_result["upload_failures"],
                "file_type": upload_result["file_type"],
                "priority": upload_result["priority"],
                "source_id": source_id,
                "company_id": company_id,
                "sample_chunk": upload_result.get("sample_chunk")
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Update failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Update failed: {str(e)}")

    async def upsert(self, file: UploadFile, priority: str, metadata: str = None,
                     source_id: str = None, company_id: str = None):
        """Upsert documents - update if exists, create if not"""
        try:
            # Validate inputs
            self.validate_source_id(source_id)
            self.validate_priority(priority)

            # Parse metadata string to dict
            metadata_dict = self._parse_metadata(metadata)

            # Check if documents exist
            existing_docs = self.check_documents_exist(source_id, company_id)
            existing_count = 0
            documents_deleted = 0

            if existing_docs:
                # Count and delete existing documents
                existing_count = self.count_documents(source_id, company_id)
                delete_result = await self.delete_service.delete(source_id, company_id)
                documents_deleted = delete_result["documents_deleted"]

            # Upload new documents with parsed metadata dict
            upload_result = await self.upload_service.process(
                file, priority, metadata_dict, source_id, company_id
            )

            operation = "updated" if existing_docs else "created"

            return {
                "status": "success",
                "operation": operation,
                "message": f"Successfully {operation} documents for source_id: {source_id}",
                "documents_deleted": documents_deleted,
                "previous_document_count": existing_count,
                "chunks_processed": upload_result["chunks_processed"],
                "points_uploaded": upload_result["points_uploaded"],
                "upload_failures": upload_result["upload_failures"],
                "file_type": upload_result["file_type"],
                "priority": upload_result["priority"],
                "source_id": source_id,
                "company_id": company_id,
                "sample_chunk": upload_result.get("sample_chunk")
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Upsert failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Upsert failed: {str(e)}")
