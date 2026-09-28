import json
import logging
import re
from typing import Any, List, Dict, Optional
from fastapi import HTTPException
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

    def validate_source_id(self, source_id: str, strict: bool = False) -> str:
        """Validate source_id is provided and return it stripped.

        strict=True (used on ingestion) also enforces a length cap and a safe
        character set, so the stored id always matches what callers filter on.
        """
        if not source_id or not source_id.strip():
            raise HTTPException(
                status_code=400,
                detail="source_id is required and cannot be empty"
            )

        # Strip surrounding whitespace so padded and unpadded ids never become two documents;
        # delete/update/search all filter on the exact stored string.
        source_id = source_id.strip()

        # Only ingestion is strict: delete/update must still accept legacy ids
        # that were stored before these rules existed.
        if strict:
            if len(source_id) > settings.MAX_SOURCE_ID_LENGTH:
                raise HTTPException(
                    status_code=400,
                    detail=f"source_id must be at most {settings.MAX_SOURCE_ID_LENGTH} characters"
                )
            # fullmatch: an env-overridden pattern without ^...$ must not accept a valid prefix
            if not re.fullmatch(settings.SOURCE_ID_PATTERN, source_id):
                raise HTTPException(
                    status_code=400,
                    detail=f"source_id contains invalid characters (allowed pattern: {settings.SOURCE_ID_PATTERN})"
                )
        return source_id

    def validate_priority(self, priority: str) -> str:
        """Validate priority format (P1, P2, ...) and return it upper-cased"""
        # Previously anything starting with "P" passed (e.g. "Pxyz");
        # now it must be "P" followed by digits, returned upper-cased.
        normalized = priority.strip().upper() if priority else ""
        if not re.fullmatch(settings.PRIORITY_PATTERN, normalized):
            raise HTTPException(
                status_code=400,
                detail="Invalid priority format. Must be P1, P2, P3, etc."
            )
        return normalized

    def validate_document_fields(self, source_id: str, company_id: Optional[str],
                                 title: Optional[str], summary: Optional[str],
                                 tags: Optional[List[str]], metadata: Optional[Dict[str, Any]]):
        """Validate and normalize the descriptive fields of an ingestion request.

        Returns (company_id, title, summary, tags, metadata) normalized. Raises 400 when
        a field is blank/malformed or when metadata contradicts the form fields.
        """
        # Work on a copy so the caller's dict is never mutated by the upload flow;
        # a non-dict metadata is rejected up front.
        if metadata is not None and not isinstance(metadata, dict):
            raise HTTPException(status_code=400, detail="metadata must be a JSON object")
        metadata = dict(metadata) if metadata else {}

        if company_id is not None:
            company_id = company_id.strip() or None

        title = self._validate_optional_text(title, "title")
        summary = self._validate_optional_text(summary, "summary")
        tags = self._validate_tags(tags)

        # metadata must not carry a different identity than the form fields: the
        # form source_id is what every stored point is keyed and filtered on.
        meta_source_id = metadata.get("source_id")
        if meta_source_id is not None and str(meta_source_id).strip() != source_id:
            raise HTTPException(
                status_code=400,
                detail=f"metadata.source_id ({meta_source_id}) does not match source_id ({source_id})"
            )

        # Same for the tenant: metadata.company is the organization filter key, so it
        # must agree with company_id (or fill it in when company_id was not sent).
        meta_company = metadata.get("company")
        if meta_company is not None and str(meta_company).strip():
            meta_company = str(meta_company).strip()
            if company_id and meta_company != company_id:
                raise HTTPException(
                    status_code=400,
                    detail=f"metadata.company ({meta_company}) does not match company_id ({company_id})"
                )
            company_id = company_id or meta_company

        # A non-empty markdown_url replaces the file as the content source,
        # so it must be a fetchable http(s) URL.
        markdown_url = metadata.get("markdown_url")
        if markdown_url:
            if not isinstance(markdown_url, str) or not markdown_url.strip().lower().startswith(("http://", "https://")):
                raise HTTPException(
                    status_code=400,
                    detail="metadata.markdown_url must be an http(s) URL"
                )
            metadata["markdown_url"] = markdown_url.strip()

        return company_id, title, summary, tags, metadata

    @staticmethod
    def _validate_optional_text(value: Optional[str], field: str) -> Optional[str]:
        """None stays None; a provided value must be a non-blank string (returned stripped)"""
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            raise HTTPException(status_code=400, detail=f"{field} cannot be empty when provided")
        return value.strip()

    @staticmethod
    def _validate_tags(tags: Optional[List[Any]]) -> Optional[List[str]]:
        """Tags must be non-empty strings; returned stripped and de-duplicated in order"""
        if tags is None:
            return None
        if not isinstance(tags, list):
            raise HTTPException(status_code=400, detail="tags must be a list of strings")

        # Tags feed both the "tags" payload filter (MatchAny) and the tags embedding,
        # so blanks/non-strings are rejected and duplicates dropped (order kept).
        cleaned = []
        for tag in tags:
            if not isinstance(tag, str) or not tag.strip():
                raise HTTPException(status_code=400, detail="tags must be non-empty strings")
            tag = tag.strip()
            if tag not in cleaned:
                cleaned.append(tag)
        return cleaned or None
