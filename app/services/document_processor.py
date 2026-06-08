import logging
from typing import Dict, Optional, List, Any
from fastapi import UploadFile
from app.services.document_operations.upload_service import UploadService
from app.services.document_operations.update_service import UpdateService
from app.services.document_operations.delete_service import DeleteService
from app.services.document_operations.metadata_service import MetadataService
from app.models.api_models import DeleteRequest

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Orchestrator for all document operations"""

    def __init__(self):
        self.upload_service = UploadService()
        self.update_service = UpdateService()
        self.delete_service = DeleteService()
        self.metadata_service = MetadataService()

    def get_supported_file_types(self):
        """Get list of all supported file types"""
        return self.upload_service.get_supported_file_types()

    async def process_upload(self, file: UploadFile, priority: str, metadata: Dict[str, Any] = None,
                             source_id: str = None, company_id: str = None, 
                             title: str = None, summary: str = None, tags: List[str] = None):
        """Process file upload"""
        return await self.upload_service.process(
            file, priority, metadata, source_id, company_id, title, summary, tags
        )

    async def update_documents(self, file: UploadFile, priority: str, metadata: str = None,
                               source_id: str = None, company_id: str = None):
        """Update existing documents"""
        return await self.update_service.update(file, priority, metadata, source_id, company_id)

    async def upsert_documents(self, file: UploadFile, priority: str, metadata: str = None,
                               source_id: str = None, company_id: str = None):
        """Upsert documents - update if exists, create if not"""
        return await self.update_service.upsert(file, priority, metadata, source_id, company_id)

    async def delete_documents(self, request: DeleteRequest):
        """Delete documents by source_id and optional company_id"""
        return await self.delete_service.delete(request.source_id, request.company_id)

    async def update_metadata(self, source_id: str, metadata_updates: Dict,
                              company_id: Optional[str] = None):
        """Update only metadata without reprocessing documents"""
        return await self.metadata_service.update_metadata(source_id, metadata_updates, company_id)
