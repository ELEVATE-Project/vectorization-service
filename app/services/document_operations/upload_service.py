import uuid
import logging
from datetime import datetime
from typing import List, Dict, Optional, Any
from fastapi import HTTPException, UploadFile
from qdrant_client import models
from app.services.document_operations.base_operation import BaseDocumentOperation
from app.core.clients.qdrant import upload_to_qdrant
from app.core.clients.embedding import generate_embeddings
from app.config import settings
from app.services.file_processors.csv_processor import CSVProcessor
from app.services.file_processors.pdf_processor import PDFProcessor
from app.services.file_processors.docx_processor import DOCXProcessor
from app.services.file_processors.xlsx_processor import XLSXProcessor
from app.services.file_processors.text_processor import TextProcessor
from app.services.url_text_extractor import URLTextExtractor

logger = logging.getLogger(__name__)


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

    async def process(self, file: UploadFile, priority: str, metadata: Dict[str, Any] = None,
                      source_id: str = None, company_id: str = None,
                      title: str = None, summary: str = None, tags: List[str] = None):
        """Process file upload with company_id support"""
        try:
            # Validate inputs
            self.validate_source_id(source_id)
            self.validate_priority(priority)

            # Use metadata dict directly (already parsed by endpoint)
            additional_metadata = metadata if metadata else {}
            logger.info(f"Received metadata: {additional_metadata}")

            # Add company_id to metadata if provided
            if company_id:
                additional_metadata['company'] = company_id
            
            # Add title, summary, and tags to metadata if provided
            if title:
                additional_metadata['title'] = title
            if summary:
                additional_metadata['summary'] = summary
            if tags:
                # Tags are already a list
                additional_metadata['tags'] = tags

            # Ensure collections exist
            await self.ensure_collections()

            # Initialize file_extension
            file_extension = file.filename.split(".")[-1].lower() if file.filename else "txt"

            # Check if markdown_url is present in metadata
            if additional_metadata and 'markdown_url' in additional_metadata and additional_metadata['markdown_url']:
                # Extract text from URL instead of processing file
                url = additional_metadata['markdown_url']
                logger.info(f"Extracting text from markdown_url: {url}")
                processed_chunks = await self._process_url_text(
                    url, file.filename, priority
                )
                file_extension = "url_extracted"  # Mark as URL-extracted content
            else:
                # Process file based on type (normal flow)
                file_content = await file.read()

                processed_chunks = await self._process_file_by_type(
                    file_content, file.filename, priority, file_extension
                )

            if not processed_chunks:
                raise HTTPException(
                    status_code=400,
                    detail="No content could be extracted from the file."
                )

            # Generate embeddings and upload
            upload_results = await self._upload_chunks(
                processed_chunks, additional_metadata, source_id, company_id, title, summary, tags
            )

            return {
                "status": "success",
                "message": f"Successfully processed {len(processed_chunks)} chunks from {file.filename}",
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
                "supported_file_types": self.get_supported_file_types(),
                "sample_chunk": {
                    "text": processed_chunks[0]["text"] if processed_chunks else None,
                    "metadata": processed_chunks[0]["metadata"] if processed_chunks else None,
                },
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Upload failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

    async def _process_file_by_type(self, file_content: bytes, filename: str,
                                    priority: str, file_extension: str):
        """Process file using the appropriate processor"""
        processor = self.processor_map.get(f".{file_extension}")

        if not processor:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file_extension}. "
                       f"Supported types: {self.get_supported_file_types()}"
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
                detail=f"Failed to process URL text: {str(e)}"
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

    def _prepare_chunk_metadata(self, chunk: dict, additional_metadata: dict, 
                                source_id: str, company_id: str, 
                                title: str, summary: str, tags: List[str]):
        """Prepare and merge metadata for a chunk"""
        if not isinstance(chunk["metadata"], dict):
            logger.warning("Chunk metadata is not a dict, initializing empty dict")
            chunk_metadata = {}
        else:
            chunk_metadata = chunk["metadata"].copy()

        if additional_metadata:
            chunk_metadata.update(additional_metadata)

        if company_id and 'company' not in chunk_metadata:
            chunk_metadata['company'] = company_id
        
        if title and 'title' not in chunk_metadata:
            chunk_metadata['title'] = title
        
        if summary and 'summary' not in chunk_metadata:
            chunk_metadata['summary'] = summary
        
        if tags and 'tags' not in chunk_metadata:
            chunk_metadata['tags'] = tags
        
        if source_id:
            chunk_metadata['source_id'] = source_id
        
        current_time = datetime.now().isoformat()
        chunk_metadata['created_at'] = current_time
        chunk_metadata['updated_at'] = current_time
        
        return chunk_metadata

    def _create_point_vectors(self, text_embedding, field_embeddings: dict, sparse_vector=None):
        """Create vectors dictionary with text and field embeddings.

        When sparse_vector is provided (Phase 2 only) it is stored under the
        configured sparse vector name so Qdrant can use it for BM25 hybrid search.
        """
        vectors_dict = {"text": text_embedding.tolist()}

        for field_name in ['title', 'summary', 'tags', 'metadata']:
            if field_name in field_embeddings:
                vectors_dict[field_name] = field_embeddings[field_name].tolist()

        if sparse_vector is not None:
            # sparse_vector is a qdrant_client SparseVector model instance
            vectors_dict[settings.SPARSE_VECTOR_NAME] = sparse_vector

        return vectors_dict

    async def _upload_chunks(self, processed_chunks: List[dict], additional_metadata: dict,
                             source_id: str, company_id: str = None,
                             title: str = None, summary: str = None, tags: List[str] = None):
        """Generate embeddings and upload chunks to Qdrant with separate embeddings for title, summary, and text"""
        logger.info(f"Generating embeddings for {len(processed_chunks)} chunks")
        text_embeddings = generate_embeddings([chunk["text"] for chunk in processed_chunks])

        field_embeddings = self._generate_field_embeddings(title, summary, tags, additional_metadata)

        # Phase 2: generate BM25 sparse vectors when enabled.
        sparse_vectors: List[object] = []
        if settings.SPARSE_SEARCH_ENABLED:
            try:
                from app.core.clients.sparse_encoder import generate_sparse_vector
                from qdrant_client.http.models import SparseVector  # type: ignore[import]
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

        points = []
        for idx, (chunk, text_embedding) in enumerate(zip(processed_chunks, text_embeddings)):
            if not isinstance(chunk, dict):
                logger.error(f"Invalid chunk type: {type(chunk)}")
                continue

            if "id" not in chunk or "text" not in chunk or "metadata" not in chunk:
                logger.error(f"Chunk missing required fields: {chunk.keys()}")
                continue

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
                "tags": tags if tags else None
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

            return upload_results

        return {"total_points": 0, "success_count": 0, "error_count": 0}
