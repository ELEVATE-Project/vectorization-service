import uuid
import logging
import re
from datetime import datetime
from typing import List, Dict, Any
from fastapi import HTTPException
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_core.documents import Document as LangchainDocument

from .base_processor import BaseFileProcessor
from app.config import settings
from app.services.translation_service import process_chunk

logger = logging.getLogger(__name__)


class TextProcessor(BaseFileProcessor):
    """Processor for text and markdown files"""
    
    def get_supported_extensions(self) -> List[str]:
        return ['.txt', '.md', '.markdown', '.text']

    def _is_markdown(self, text: str) -> bool:
        """
        Detect if the text content is in markdown format.
        Checks for common markdown patterns.
        """
        markdown_patterns = [
            r'^#{1,6}\s+.+$',  # Headers (# Header)
            r'^\*{1,3}.+\*{1,3}$',  # Bold/Italic (*text* or **text** or ***text***)
            r'^\[.+\]\(.+\)$',  # Links [text](url)
            r'^```[\s\S]*?```$',  # Code blocks
            r'^\* .+$',  # Unordered lists
            r'^\d+\. .+$',  # Ordered lists
            r'^> .+$',  # Blockquotes
            r'^\|.+\|.+\|$',  # Tables
            r'^---+$',  # Horizontal rules
            r'!\[.+\]\(.+\)',  # Images ![alt](url)
        ]
        
        # Check if text contains multiple markdown patterns
        pattern_matches = 0
        for pattern in markdown_patterns:
            if re.search(pattern, text, re.MULTILINE):
                pattern_matches += 1
                if pattern_matches >= 2:  # If we find 2+ markdown patterns, consider it markdown
                    return True
        
        return False

    async def process(self, file_content: bytes, filename: str, priority: str, 
                      chunk_size: int = None, chunk_overlap: int = None) -> List[Dict[str, Any]]:
        """Process text/markdown file and return chunks
        
        Args:
            file_content: File content as bytes
            filename: Name of the file
            priority: Priority level
            chunk_size: Optional custom chunk size (defaults to settings)
            chunk_overlap: Optional custom chunk overlap (defaults to settings)
        """
        self._validate_file_content(file_content, filename)

        try:
            # Decode text content
            try:
                text_content = file_content.decode('utf-8')
            except UnicodeDecodeError:
                # Try with latin-1 encoding as fallback
                text_content = file_content.decode('latin-1')

            if not text_content.strip():
                raise HTTPException(
                    status_code=400,
                    detail=f"File {filename} is empty or contains no readable text"
                )

            # Detect if content is markdown
            is_markdown = self._is_markdown(text_content)
            file_type = "markdown" if is_markdown else "text"
            
            logger.info(f"Processing {filename} as {file_type}")

            # Create document for LangChain
            doc = LangchainDocument(
                page_content=text_content,
                metadata={
                    "source": filename,
                    "type": file_type,
                    "priority": priority,
                    "is_markdown": is_markdown
                },
            )

            # Split text based on format
            if is_markdown:
                chunks = self._split_markdown(doc, chunk_size, chunk_overlap)
            else:
                chunks = self._split_text(doc, chunk_size, chunk_overlap)

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
                        "created_at": datetime.now().isoformat(),
                        "updated_at": datetime.now().isoformat(),
                    },
                }
                processed_chunks.append(processed_chunk)

            logger.info(
                f"Successfully processed {filename} into {len(processed_chunks)} chunks "
                f"using {'markdown' if is_markdown else 'text'} splitter"
            )
            return processed_chunks

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Text/Markdown processing error: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error processing text/markdown file: {str(e)}"
            )

    def _split_markdown(self, doc: LangchainDocument, 
                              chunk_size: int = None, chunk_overlap: int = None) -> List[LangchainDocument]:
        """
        Split markdown content using MarkdownHeaderTextSplitter.
        This preserves the document structure based on headers.
        Uses smaller chunk sizes optimized for markdown structure.
        
        Args:
            doc: Document to split
            chunk_size: Optional custom chunk size (defaults to settings.MARKDOWN_CHUNK_SIZE)
            chunk_overlap: Optional custom chunk overlap (defaults to settings.MARKDOWN_CHUNK_OVERLAP)
        """
        # Use custom chunk sizes or fall back to settings
        effective_chunk_size = chunk_size if chunk_size is not None else settings.MARKDOWN_CHUNK_SIZE
        effective_chunk_overlap = chunk_overlap if chunk_overlap is not None else settings.MARKDOWN_CHUNK_OVERLAP
        
        # Define headers to split on (h1, h2, h3, h4)
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
            ("####", "Header 4"),
        ]

        # First split by headers
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False  # Keep headers in the content for context
        )
        
        # Split the document by headers
        md_header_splits = markdown_splitter.split_text(doc.page_content)
        
        # Use custom or default chunk size for better granularity
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=effective_chunk_size,
            chunk_overlap=effective_chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],  # Prioritize paragraph breaks
        )
        
        # Apply recursive splitting to each markdown section
        final_chunks = []
        for md_chunk in md_header_splits:
            # Merge original metadata with header metadata
            merged_metadata = {**doc.metadata, **md_chunk.metadata}
            
            # Create a new document with merged metadata
            temp_doc = LangchainDocument(
                page_content=md_chunk.page_content,
                metadata=merged_metadata
            )
            
            # Always split markdown sections for better granularity
            # This ensures each section is properly chunked
            if len(md_chunk.page_content) > effective_chunk_size:
                sub_chunks = text_splitter.split_documents([temp_doc])
                final_chunks.extend(sub_chunks)
            else:
                # Even small chunks go through to maintain consistency
                final_chunks.append(temp_doc)
        
        return final_chunks

    def _split_text(self, doc: LangchainDocument, 
                          chunk_size: int = None, chunk_overlap: int = None) -> List[LangchainDocument]:
        """Split plain text using RecursiveCharacterTextSplitter
        
        Args:
            doc: Document to split
            chunk_size: Optional custom chunk size (defaults to settings.CHUNK_SIZE)
            chunk_overlap: Optional custom chunk overlap (defaults to settings.CHUNK_OVERLAP)
        """
        # Use custom chunk sizes or fall back to settings
        effective_chunk_size = chunk_size if chunk_size is not None else settings.CHUNK_SIZE
        effective_chunk_overlap = chunk_overlap if chunk_overlap is not None else settings.CHUNK_OVERLAP
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=effective_chunk_size,
            chunk_overlap=effective_chunk_overlap,
            length_function=len,
        )
        
        return text_splitter.split_documents([doc])
