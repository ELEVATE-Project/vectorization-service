import json
import logging
import re
from typing import Any, List, Dict, Optional
from fastapi import HTTPException
from qdrant_client import models
from app.core.clients.qdrant import qdrant_client, ensure_collections_exist
from app.config import settings
from app.constants import messages as msg
from app.utils.generic import normalize_theme

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

    def collect_point_ids(self, source_id: str, company_id: Optional[str] = None) -> list:
        """Return the ids of every point stored for source_id (+ company_id); Qdrant errors propagate"""
        scroll_filter = self.build_filter(source_id, company_id)
        point_ids, offset = [], None

        # Errors are not swallowed (unlike check_documents_exist): a failed lookup read
        # as "no document" made upsert upload a second copy next to the existing one.
        while True:
            points, offset = qdrant_client.scroll(
                collection_name=settings.COLLECTION_NAME,
                scroll_filter=scroll_filter,
                limit=256,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            point_ids.extend(point.id for point in points)

            # Ids only, one page at a time, until Qdrant reports no next page
            if offset is None:
                return point_ids

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
                detail=msg.SOURCE_ID_REQUIRED
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
                    detail=msg.SOURCE_ID_TOO_LONG.format(max_length=settings.MAX_SOURCE_ID_LENGTH)
                )
            # fullmatch: an env-overridden pattern without ^...$ must not accept a valid prefix
            if not re.fullmatch(settings.SOURCE_ID_PATTERN, source_id):
                raise HTTPException(
                    status_code=400,
                    detail=msg.SOURCE_ID_INVALID_CHARACTERS.format(pattern=settings.SOURCE_ID_PATTERN)
                )
        return source_id

    def normalize_company_id(self, company_id: Optional[str]) -> Optional[str]:
        """Return company_id stripped; None stays None, whitespace-only is rejected (400)"""
        if company_id is None:
            return None

        # Whitespace-only must not become None: on delete/update that would widen the
        # filter from one tenant to every company that has this source_id.
        if not company_id.strip():
            raise HTTPException(status_code=400, detail="company_id cannot be blank when provided")

        # Upload stores metadata.company stripped, so the filter must use the same value.
        return company_id.strip()

    def validate_priority(self, priority: str) -> str:
        """Validate priority format (P1, P2, ...) and return it upper-cased"""
        # Previously anything starting with "P" passed (e.g. "Pxyz");
        # now it must be "P" followed by digits, returned upper-cased.
        normalized = priority.strip().upper() if priority else ""
        if not re.fullmatch(settings.PRIORITY_PATTERN, normalized):
            raise HTTPException(
                status_code=400,
                detail=msg.PRIORITY_INVALID_FORMAT
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
            raise HTTPException(status_code=400, detail=msg.METADATA_NOT_JSON_OBJECT)
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
                detail=msg.METADATA_SOURCE_ID_MISMATCH.format(
                    metadata_source_id=meta_source_id, source_id=source_id
                )
            )

        # Same for the tenant: metadata.company is the organization filter key, so it
        # must agree with company_id (or fill it in when company_id was not sent).
        meta_company = metadata.get("company")
        if meta_company is not None and str(meta_company).strip():
            meta_company = str(meta_company).strip()
            if company_id and meta_company != company_id:
                raise HTTPException(
                    status_code=400,
                    detail=msg.METADATA_COMPANY_MISMATCH.format(
                        metadata_company=meta_company, company_id=company_id
                    )
                )
            company_id = company_id or meta_company

        # A non-empty markdown_url replaces the file as the content source,
        # so it must be a fetchable http(s) URL.
        markdown_url = metadata.get("markdown_url")
        if markdown_url:
            if not isinstance(markdown_url, str) or not markdown_url.strip().lower().startswith(("http://", "https://")):
                raise HTTPException(
                    status_code=400,
                    detail=msg.MARKDOWN_URL_NOT_HTTP
                )
            metadata["markdown_url"] = markdown_url.strip()

        return company_id, title, summary, tags, metadata

    def validate_theme(self, theme: Optional[str], metadata: Dict[str, Any]) -> Optional[str]:
        """Validate the theme form field and pull any metadata copy up to the top level.

        Returns the normalized theme, or None when there is none. Removes every theme key
        (any case) from `metadata` in place, so pass the copy from validate_document_fields.
        Raises 400 for a blank form theme, a non-string metadata theme, or a mismatch.
        """
        theme = normalize_theme(self._validate_optional_text(theme, "theme"))

        # theme is top-level only and never embedded, so every metadata copy is removed (it would
        # feed the metadata vector). Any key case: caller keys arrive as THEME, TITLE, ...
        theme_keys = [key for key in metadata if str(key).lower() == "theme"]
        for meta_theme in [metadata.pop(key) for key in theme_keys]:
            if meta_theme is None:
                continue
            if not isinstance(meta_theme, str):
                raise HTTPException(status_code=400, detail=msg.THEME_NOT_A_STRING)

            # A blank metadata theme is dropped, as an empty caller tag is; a real one
            # fills in a missing form theme or must agree with it.
            meta_theme = normalize_theme(meta_theme)
            if meta_theme is None:
                continue
            if theme is None:
                theme = meta_theme
            elif meta_theme != theme:
                raise HTTPException(
                    status_code=400,
                    detail=msg.METADATA_THEME_MISMATCH.format(metadata_theme=meta_theme, theme=theme)
                )

        return theme

    @staticmethod
    def _validate_optional_text(value: Optional[str], field: str) -> Optional[str]:
        """None stays None; a provided value must be a non-blank string (returned stripped)"""
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            raise HTTPException(status_code=400, detail=msg.FIELD_BLANK_WHEN_PROVIDED.format(field=field))
        return value.strip()

    @staticmethod
    def _validate_tags(tags: Optional[List[Any]]) -> Optional[List[str]]:
        """Tags must be non-empty strings; returned stripped and de-duplicated in order"""
        if tags is None:
            return None
        if not isinstance(tags, list):
            raise HTTPException(status_code=400, detail=msg.TAGS_NOT_A_LIST)

        # Tags feed both the "tags" payload filter (MatchAny) and the tags embedding,
        # so blanks/non-strings are rejected and duplicates dropped (order kept).
        cleaned = []
        for tag in tags:
            if not isinstance(tag, str) or not tag.strip():
                raise HTTPException(status_code=400, detail=msg.TAGS_BLANK_OR_NON_STRING)
            tag = tag.strip()
            if tag not in cleaned:
                cleaned.append(tag)
        return cleaned or None
