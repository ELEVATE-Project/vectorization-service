import uuid
import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Optional, Any
from fastapi import HTTPException, UploadFile
from qdrant_client import models
from app.services.document_operations.base_operation import BaseDocumentOperation
from app.core.clients.qdrant import upload_to_qdrant, qdrant_client
from app.core.clients.embedding import generate_embeddings, validate_vector
from app.config import settings
from app.constants import messages as msg
from app.constants import constants as const
from app.services.file_processors.csv_processor import CSVProcessor
from app.services.file_processors.pdf_processor import PDFProcessor
from app.services.file_processors.docx_processor import DOCXProcessor
from app.services.file_processors.xlsx_processor import XLSXProcessor
from app.services.file_processors.text_processor import TextProcessor
from app.services.url_text_extractor import URLTextExtractor

logger = logging.getLogger(__name__)

# Background rollback tasks still running; asyncio keeps only weak references to tasks,
# so each one is held here until it finishes.
_background_rollback_tasks: set = set()


class UploadService(BaseDocumentOperation):
    def __init__(self):
        # Register all processors
        self.processors = [
            CSVProcessor(),
            PDFProcessor(),
            DOCXProcessor(),
            XLSXProcessor(),
            TextProcessor(),
        ]

        # Create a mapping for quick lookup
        self.processor_map = {}
        for processor in self.processors:
            for ext in processor.supported_extensions:
                self.processor_map[ext] = processor

    def get_supported_file_types(self) -> List[str]:
        """Get list of all supported file types"""
        return list(self.processor_map.keys())

    def validate_upload_file(self, file: UploadFile, check_type: bool = True) -> str:
        """Validate the uploaded file's name/type/declared size; returns the lowercase extension"""
        if file is None or not file.filename or not file.filename.strip():
            raise HTTPException(status_code=400, detail=msg.FILE_WITH_FILENAME_REQUIRED)

        # Reject unsupported types before reading/parsing the body, so the caller gets
        # a clear 400 instead of a late processor failure.
        file_extension = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
        if check_type and f".{file_extension}" not in self.processor_map:
            raise HTTPException(
                status_code=400,
                detail=msg.UNSUPPORTED_FILE_TYPE.format(
                    extension=file_extension or "none", supported_types=self.get_supported_file_types()
                )
            )

        # Fast-path size check when the server already knows the upload size;
        # validate_file_content() re-checks on the real bytes.
        if file.size is not None and file.size > self._max_file_size_bytes():
            raise self._file_too_large()
        return file_extension or "txt"

    def validate_file_content(self, file_content: bytes) -> None:
        """Reject empty or oversized file bodies"""
        if not file_content:
            raise HTTPException(status_code=400, detail=msg.UPLOADED_FILE_EMPTY)
        if len(file_content) > self._max_file_size_bytes():
            raise self._file_too_large()

    @staticmethod
    def _max_file_size_bytes() -> int:
        return settings.MAX_FILE_SIZE_MB * 1024 * 1024

    @staticmethod
    def _file_too_large() -> HTTPException:
        return HTTPException(
            status_code=413,
            detail=msg.FILE_TOO_LARGE.format(max_size_mb=settings.MAX_FILE_SIZE_MB)
        )

    @staticmethod
    def _delete_points(point_ids: list) -> None:
        """Delete exactly these point ids (never another request's points)"""
        qdrant_client.delete(
            collection_name=settings.COLLECTION_NAME,
            points_selector=models.PointIdsList(points=point_ids),
        )

    async def _rollback_points(self, point_ids: list, source_id: str,
                               reason: str = const.ROLLBACK_REASON_PARTIAL_UPLOAD) -> bool:
        """Delete these point ids, retrying briefly; True once they are gone.

        reason only labels the log lines (UpdateService reuses this to remove an old version).
        """
        max_attempts = max(1, settings.UPLOAD_ROLLBACK_MAX_ATTEMPTS)
        for attempt in range(1, max_attempts + 1):
            try:
                self._delete_points(point_ids)
                logger.warning(f"Deleted {len(point_ids)} points for source_id {source_id} ({reason})")
                return True
            except Exception as exc:
                logger.warning(
                    f"Delete attempt {attempt}/{max_attempts} for source_id {source_id} ({reason}) failed: {exc}"
                )
                # Qdrant just failed the upload, so give it a moment before trying again (0.5s, 1s, ...)
                if attempt < max_attempts:
                    await asyncio.sleep(settings.UPLOAD_ROLLBACK_RETRY_WAIT_SECONDS * 2 ** (attempt - 1))
        return False

    async def _background_rollback(self, point_ids: list, source_id: str,
                                   reason: str = const.ROLLBACK_REASON_PARTIAL_UPLOAD) -> None:
        """Keep retrying the delete after the request has returned"""
        for attempt in range(1, settings.UPLOAD_ROLLBACK_BACKGROUND_MAX_ATTEMPTS + 1):
            # Wait doubles each attempt, capped at the max wait (2, 4, 8, ... 60s)
            wait_seconds = min(
                settings.UPLOAD_ROLLBACK_BACKGROUND_FIRST_WAIT_SECONDS * 2 ** (attempt - 1),
                settings.UPLOAD_ROLLBACK_BACKGROUND_MAX_WAIT_SECONDS,
            )
            await asyncio.sleep(wait_seconds)
            try:
                # The Qdrant client is synchronous; run it off the event loop
                await asyncio.to_thread(self._delete_points, point_ids)
                logger.warning(
                    f"Background delete removed {len(point_ids)} points for source_id {source_id} "
                    f"({reason}, attempt {attempt})"
                )
                return
            except Exception as exc:
                logger.warning(
                    f"Background delete attempt {attempt} for source_id {source_id} ({reason}) failed: {exc}"
                )

        # Out of retries: the ids in this log line are what is needed to clean up by hand
        logger.error(
            f"Background delete gave up for source_id {source_id} ({reason}); {len(point_ids)} points may "
            f"remain searchable. Point IDs: {point_ids}"
        )

    def _schedule_background_rollback(self, point_ids: list, source_id: str,
                                      reason: str = const.ROLLBACK_REASON_PARTIAL_UPLOAD) -> None:
        """Start the background delete and keep a reference until it finishes"""
        task = asyncio.create_task(self._background_rollback(list(point_ids), source_id, reason))
        _background_rollback_tasks.add(task)
        task.add_done_callback(_background_rollback_tasks.discard)

    async def _ensure_upload_complete(self, upload_results: dict, source_id: str,
                                      company_id: Optional[str] = None) -> None:
        """Fail the request (and roll back this request's points) on an empty or partial upload"""
        if upload_results.get("success_count", 0) == 0 and upload_results.get("error_count", 0) == 0:
            raise HTTPException(
                status_code=500,
                detail=msg.NO_VALID_CHUNKS_TO_UPLOAD
            )
        if upload_results.get("error_count", 0) == 0:
            return

        # Some batches failed: delete the points this request did store so a half-indexed
        # document is never left behind. Only this request's own ids are ever deleted.
        point_ids = upload_results.get("point_ids") or []
        error_count = upload_results["error_count"]
        total_points = upload_results.get("total_points", len(point_ids))
        rolled_back = await self._rollback_points(point_ids, source_id) if point_ids else True

        if rolled_back:
            raise HTTPException(
                status_code=502,
                detail=msg.PARTIAL_UPLOAD_ROLLED_BACK.format(
                    source_id=source_id, error_count=error_count, total_points=total_points
                ),
            )

        # Rollback still failing: log every id for manual cleanup, keep retrying in the
        # background, and tell the caller the truth instead of "nothing was kept".
        logger.error(
            f"Rollback of partial upload failed for source_id {source_id} (company_id {company_id}); "
            f"{len(point_ids)} points may remain searchable. Point IDs: {point_ids}"
        )
        self._schedule_background_rollback(point_ids, source_id)
        raise HTTPException(
            status_code=502,
            detail=msg.PARTIAL_UPLOAD_ROLLBACK_FAILED.format(
                source_id=source_id, error_count=error_count, total_points=total_points,
                stored_count=upload_results.get("success_count", 0),
            ),
        )

    async def process(self, file: UploadFile, priority: str, metadata: Dict[str, Any] = None,
                      source_id: str = None, company_id: str = None,
                      title: str = None, summary: str = None, tags: List[str] = None,
                      theme: str = None):
        """Process file upload with company_id support"""
        try:
            # Validate and normalize every caller-controlled field before any processing,
            # so a bad request never reaches parsing, embedding or Qdrant.
            source_id = self.validate_source_id(source_id, strict=True)
            priority = self.validate_priority(priority)
            company_id, title, summary, tags, additional_metadata = self.validate_document_fields(
                source_id, company_id, title, summary, tags, metadata
            )
            # Runs on the validated copy: any metadata theme is moved to the top level here,
            # so it never reaches the stored metadata or the metadata vector.
            theme = self.validate_theme(theme, additional_metadata)
            logger.info(f"Received metadata: {additional_metadata}")

            # Add company_id to metadata if provided
            if company_id:
                additional_metadata['company'] = company_id

            # markdown_url (already validated) means content comes from the URL,
            # so the uploaded file itself is not parsed and its type is not checked.
            use_markdown_url = bool(additional_metadata.get('markdown_url'))
            file_extension = self.validate_upload_file(file, check_type=not use_markdown_url)

            # Ensure collections exist
            await self.ensure_collections()

            # Check if markdown_url is present in metadata
            if use_markdown_url:
                # Extract text from URL instead of processing file
                url = additional_metadata['markdown_url']
                logger.info(f"Extracting text from markdown_url: {url}")
                processed_chunks = await self._process_url_text(
                    url, file.filename, priority
                )
                file_extension = "url_extracted"  # Mark as URL-extracted content
            else:
                # Process file based on type (normal flow); size/emptiness are checked
                # on the actual bytes because UploadFile.size is not always populated.
                file_content = await file.read()
                self.validate_file_content(file_content)

                processed_chunks = await self._process_file_by_type(
                    file_content, file.filename, priority, file_extension
                )

            if not processed_chunks:
                raise HTTPException(
                    status_code=400,
                    detail=msg.NO_CONTENT_EXTRACTED
                )

            # Generate embeddings and upload
            upload_results = await self._upload_chunks(
                processed_chunks, additional_metadata, source_id, company_id, title, summary, tags,
                theme=theme
            )

            # upload_to_qdrant swallows per-batch errors and only counts them; a partial or
            # empty upload must fail the request instead of returning 201 to the caller.
            await self._ensure_upload_complete(upload_results, source_id, company_id)

            return {
                "status": "success",
                "message": msg.UPLOAD_SUCCEEDED.format(chunk_count=len(processed_chunks), filename=file.filename),
                "chunks_processed": len(processed_chunks),
                "points_uploaded": upload_results['success_count'],
                "upload_failures": upload_results['error_count'],
                "file_type": file_extension,
                "priority": priority.upper(),
                "source_id": source_id,
                "company_id": company_id,
                "title": title,
                "summary": summary,
                "tags": tags,
                "theme": theme,
                "supported_file_types": self.get_supported_file_types(),
                # Report the metadata exactly as stored in Qdrant (merged, incl. source_id),
                # not the raw processor metadata, which never contained source_id.
                "sample_chunk": {
                    "text": processed_chunks[0]["text"] if processed_chunks else None,
                    "metadata": upload_results.get("sample_metadata"),
                },
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Upload failed: {str(e)}")
            raise HTTPException(status_code=500, detail=msg.UPLOAD_FAILED.format(error=e))

    async def _process_file_by_type(self, file_content: bytes, filename: str,
                                    priority: str, file_extension: str):
        """Process file using the appropriate processor"""
        processor = self.processor_map.get(f".{file_extension}")

        if not processor:
            raise HTTPException(
                status_code=400,
                detail=msg.UNSUPPORTED_FILE_TYPE.format(
                    extension=file_extension, supported_types=self.get_supported_file_types()
                )
            )

        logger.info(f"Using {processor.__class__.__name__} for file {filename}")
        return await processor.process(file_content, filename, priority)
    
    async def _process_url_text(self, url: str, filename: str, priority: str):
        """Process text extracted from URL with custom chunk size - strict character-based chunking"""
        try:
            # Extract text from URL
            url_extractor = URLTextExtractor()
            extracted_text = await url_extractor.extract_text(url)
            
            logger.info(f"Extracted {len(extracted_text)} characters from URL: {url}")
            
            # Import required modules for direct chunking
            import uuid
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            from langchain_core.documents import Document as LangchainDocument
            from app.services.translation_service import process_chunk
            
            # Create document for chunking
            doc = LangchainDocument(
                page_content=extracted_text,
                metadata={
                    "source": filename,
                    "type": "url_extracted",
                    "priority": priority,
                    "is_markdown": False  # Force plain text processing
                }
            )
            
            # Use strict character-based splitting (no markdown detection)
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=settings.URL_EXTRACTION_CHUNK_SIZE,
                chunk_overlap=settings.URL_EXTRACTION_CHUNK_OVERLAP,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]  # Standard separators
            )
            
            # Split into chunks
            chunks = text_splitter.split_documents([doc])
            
            # Process chunks
            processed_chunks = []
            for chunk in chunks:
                chunk_id = uuid.uuid4().hex
                processed_text, is_hindi = process_chunk(
                    chunk.page_content, chunk_id
                )
                processed_chunk = {
                    "id": chunk_id,
                    "text": processed_text,
                    "metadata": {
                        **chunk.metadata,
                        "is_hindi": is_hindi,
                        "total_chunks": len(chunks),
                    },
                }
                processed_chunks.append(processed_chunk)
            
            logger.info(f"Processed URL content into {len(processed_chunks)} chunks with size {settings.URL_EXTRACTION_CHUNK_SIZE}")
            return processed_chunks
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error processing URL text: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=msg.URL_TEXT_PROCESSING_FAILED.format(error=e)
            )

    def parse_tags(self, tags: str) -> list:
        """Parse tags JSON string into a list"""
        if not tags:
            return []
        try:
            import json
            parsed = json.loads(tags)
            if isinstance(parsed, list):
                return parsed
            else:
                logger.warning("Tags should be a JSON list, ignoring tags")
                return []
        except json.JSONDecodeError:
            logger.warning("Invalid tags JSON provided, ignoring tags")
            return []

    def _generate_field_embeddings(self, title: str, summary: str, tags: List[str], additional_metadata: dict):
        """Generate embeddings for title, summary, tags, and metadata fields"""
        embeddings = {}
        
        if title and title.strip():
            logger.info(f"Generating embedding for title: {title}")
            embeddings['title'] = generate_embeddings([title])[0]
        
        if summary and summary.strip():
            logger.info(f"Generating embedding for summary: {summary}")
            embeddings['summary'] = generate_embeddings([summary])[0]
        
        if tags and len(tags) > 0:
            tags_text = ", ".join(tags)
            logger.info(f"Generating embedding for tags: {tags_text}")
            embeddings['tags'] = generate_embeddings([tags_text])[0]
        
        if additional_metadata:
            metadata_text = " ".join([f"{k}: {v}" for k, v in additional_metadata.items() if v])
            if metadata_text.strip():
                logger.info(f"Generating embedding for metadata: {metadata_text[:100]}...")
                embeddings['metadata'] = generate_embeddings([metadata_text])[0]
        
        return embeddings

    @staticmethod
    def _validate_chunks(processed_chunks: List[dict]) -> None:
        """Reject processor output with a malformed chunk before anything is embedded or stored"""
        invalid = [
            idx for idx, chunk in enumerate(processed_chunks)
            if not isinstance(chunk, dict) or not {"id", "text", "metadata"} <= chunk.keys()
        ]
        if invalid:
            logger.error(f"Processor returned {len(invalid)} malformed chunks at positions {invalid}")
            raise HTTPException(
                status_code=500,
                detail=msg.INVALID_CHUNKS.format(
                    invalid_count=len(invalid), chunk_count=len(processed_chunks)
                ),
            )

    def _prepare_chunk_metadata(self, chunk: dict, additional_metadata: dict, 
                                source_id: str, company_id: str, 
                                title: str, summary: str, tags: List[str]):
        """Prepare and merge metadata for a chunk"""
        if not isinstance(chunk["metadata"], dict):
            logger.warning("Chunk metadata is not a dict, initializing empty dict")
            processor_metadata = {}
        else:
            processor_metadata = chunk["metadata"]

        # Caller metadata first, then processor metadata on top: processor keys (source =
        # filename, type = "docx"/"pdf", is_hindi, total_chunks) describe the parsed file
        # and back the file_type filter, so a caller's "source"/"type" must not overwrite them.
        chunk_metadata = dict(additional_metadata) if additional_metadata else {}
        chunk_metadata.update(processor_metadata)

        # Keep a conflicting caller value instead of dropping it (e.g. commons-backend
        # sends source="file" and type=<MIME type>) under non-colliding keys.
        for key, preserved_key in (("source", "origin"), ("type", "mime_type")):
            caller_value = (additional_metadata or {}).get(key)
            if caller_value is not None and caller_value != processor_metadata.get(key):
                chunk_metadata.setdefault(preserved_key, caller_value)

        # company is the organization filter key; always the validated company_id.
        if company_id:
            chunk_metadata['company'] = company_id
        
        # title/summary/tags are stored once at the payload top level; drop the caller's
        # metadata copy when the top level holds it or it is empty (commons sends tags: []).
        # A value sent only in metadata is kept so it is not lost. TITLE is caller data, kept.
        top_level = {"title": title, "summary": summary, "tags": tags}
        for field in settings.OMITTED_FIELDS_FROM_METADATA:
            if field in chunk_metadata and (top_level.get(field) or not chunk_metadata[field]):
                del chunk_metadata[field]

        if source_id:
            chunk_metadata['source_id'] = source_id
        
        current_time = datetime.now().isoformat()
        chunk_metadata['created_at'] = current_time
        chunk_metadata['updated_at'] = current_time
        
        return chunk_metadata

    def _create_point_vectors(self, text_embedding, field_embeddings: dict, sparse_vector=None):
        """Build the vectors dict for a Qdrant point.

        Always includes the dense "text" vector plus any available field vectors
        (title, summary, tags, metadata). When sparse_vector is provided it is
        stored under settings.SPARSE_VECTOR_NAME (default "bm25") to enable
        BM25 keyword search alongside dense retrieval.
        """
        vectors_dict = {"text": validate_vector(text_embedding)}

        for field_name in ['title', 'summary', 'tags', 'metadata']:
            if field_name in field_embeddings:
                vectors_dict[field_name] = validate_vector(field_embeddings[field_name])

        if sparse_vector is not None:
            # sparse_vector is a qdrant_client SparseVector model instance
            vectors_dict[settings.SPARSE_VECTOR_NAME] = sparse_vector

        return vectors_dict

    async def _upload_chunks(self, processed_chunks: List[dict], additional_metadata: dict,
                             source_id: str, company_id: str = None,
                             title: str = None, summary: str = None, tags: List[str] = None,
                             theme: str = None):
        """Generate embeddings and upload chunks to Qdrant with separate embeddings for title, summary, and text"""
        # Skipping a bad chunk would store the document with a silent gap and still report
        # success, so fail the whole request before anything is embedded or written.
        self._validate_chunks(processed_chunks)

        logger.info(f"Generating embeddings for {len(processed_chunks)} chunks")
        text_embeddings = generate_embeddings([chunk["text"] for chunk in processed_chunks])

        # The metadata vector keeps its original input (metadata + title/summary/tags) so
        # ranking is unchanged, even though those fields are no longer stored in metadata.
        embedding_metadata = dict(additional_metadata or {})
        embedding_metadata.update({k: v for k, v in (("title", title), ("summary", summary), ("tags", tags)) if v})
        field_embeddings = self._generate_field_embeddings(title, summary, tags, embedding_metadata)

        # Phase 2: generate BM25 sparse vectors when enabled.
        sparse_vectors: List[object] = []
        if settings.SPARSE_SEARCH_ENABLED:
            try:
                from app.core.clients.sparse_encoder import generate_sparse_vector
                from qdrant_client.models import SparseVector  # type: ignore[import]
                for chunk in processed_chunks:
                    indices, values = generate_sparse_vector(chunk.get("text", ""))
                    sparse_vectors.append(
                        SparseVector(indices=indices, values=values) if indices else None
                    )
                logger.info(f"Sparse BM25 vectors generated for {len(sparse_vectors)} chunks")
            except Exception as exc:
                logger.warning(
                    f"Sparse vector generation skipped (non-fatal): {exc}. "
                    "Only dense vectors will be stored."
                )
                sparse_vectors = []

        # One point per chunk; strict zip raises on an embedding-count mismatch instead of
        # silently dropping the tail, so total_points always equals chunks_processed.
        points = []
        for idx, (chunk, text_embedding) in enumerate(zip(processed_chunks, text_embeddings, strict=True)):
            chunk_id = str(chunk["id"])
            chunk_metadata = self._prepare_chunk_metadata(
                chunk, additional_metadata, source_id, company_id, title, summary, tags
            )

            payload = {
                "text": chunk["text"],
                "metadata": chunk_metadata,
                "source_id": source_id,
                "title": title if title else None,
                "summary": summary if summary else None,
                "tags": tags if tags else None,
                # theme is a filter-only key: deliberately not embedded and not in metadata,
                # so it can never move a search score.
                "theme": theme,
            }

            sparse_vec = sparse_vectors[idx] if idx < len(sparse_vectors) else None
            vectors_dict = self._create_point_vectors(text_embedding, field_embeddings, sparse_vec)

            point = models.PointStruct(
                id=chunk_id,
                vector=vectors_dict,
                payload=payload
            )
            points.append(point)

        if points:
            logger.info(f"Starting batched upload of {len(points)} points to Qdrant")
            upload_results = upload_to_qdrant(
                points=points,
                collection_name=settings.COLLECTION_NAME,
                batch_size=100
            )

            logger.info(
                f"Upload completed: {upload_results['success_count']} successful, "
                f"{upload_results['error_count']} failed"
            )

            # point_ids let process() roll back a partial upload; sample_metadata is the
            # stored (merged) metadata of the first point, returned to the caller as-is.
            upload_results["point_ids"] = [point.id for point in points]
            upload_results["sample_metadata"] = points[0].payload["metadata"]
            return upload_results

        return {"total_points": 0, "success_count": 0, "error_count": 0,
                "point_ids": [], "sample_metadata": None}
