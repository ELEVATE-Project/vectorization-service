import io
import uuid
import PyPDF2
import logging
from datetime import datetime
from typing import List, Dict, Any
from fastapi import HTTPException
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as LangchainDocument

from .base_processor import BaseFileProcessor
from app.config import settings
from app.services.translation_service import process_chunk

logger = logging.getLogger(__name__)

class PDFProcessor(BaseFileProcessor):
    def get_supported_extensions(self) -> List[str]:
        return ['.pdf']

    async def process(self, file_content: bytes, filename: str, priority: str) -> List[Dict[str, Any]]:
        """Process PDF file with page-level OCR detection for mixed-content PDFs"""
        self._validate_file_content(file_content, filename)

        try:
            # Step 1: Initialize PDF reader
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
            total_pages = len(pdf_reader.pages)
            
            # Step 2: Process each page individually
            pages_text = []
            pages_with_text = 0
            pages_with_ocr = 0
            ocr_page_numbers = []
            
            logger.info(f"Processing {total_pages} pages from {filename}")
            
            for page_num, page in enumerate(pdf_reader.pages, start=1):
                # Try extracting text from this page
                page_text = page.extract_text()
                
                # Check if this page needs OCR
                if len(page_text.strip()) < settings.PAGE_TEXT_THRESHOLD:
                    logger.info(f"Page {page_num}/{total_pages} has minimal text, using OCR")
                    try:
                        ocr_text = await self._extract_page_with_ocr(
                            file_content, page_num - 1, filename
                        )
                        pages_text.append(ocr_text)
                        pages_with_ocr += 1
                        ocr_page_numbers.append(page_num)
                    except Exception as e:
                        logger.warning(f"OCR failed for page {page_num}, using extracted text: {str(e)}")
                        pages_text.append(page_text)
                        pages_with_text += 1
                else:
                    # Use extracted text
                    pages_text.append(page_text)
                    pages_with_text += 1
            
            # Step 3: Combine all page texts
            full_text = "\n\n".join(pages_text)
            
            # Validate that we have extracted some text
            if not full_text or len(full_text.strip()) == 0:
                logger.error(f"No text extracted from {filename} after processing all pages")
                raise HTTPException(
                    status_code=400,
                    detail=f"Could not extract any text from PDF: {filename}. "
                           "The document may be corrupted or contain no readable content."
                )
            
            # Step 4: Determine extraction method
            if pages_with_ocr == 0:
                extraction_method = "text"
            elif pages_with_text == 0:
                extraction_method = "ocr"
            else:
                extraction_method = "mixed"
            
            logger.info(
                f"Extraction complete for {filename}: "
                f"method={extraction_method}, "
                f"text_pages={pages_with_text}, "
                f"ocr_pages={pages_with_ocr}"
            )

            # Step 5: Create document for LangChain
            doc = LangchainDocument(
                page_content=full_text,
                metadata={
                    "source": filename, 
                    "type": "pdf", 
                    "priority": priority,
                    "extraction_method": extraction_method,
                    "total_pages": total_pages,
                    "pages_with_text": pages_with_text,
                    "pages_with_ocr": pages_with_ocr,
                    "ocr_pages": ocr_page_numbers if ocr_page_numbers else None
                },
            )

            # Step 6: Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP,
                length_function=len,
            )

            chunks = text_splitter.split_documents([doc])

            # Step 7: Process chunks
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

            return processed_chunks

        except Exception as e:
            logger.error(f"PDF processing error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

    async def _extract_page_with_ocr(self, file_content: bytes, page_num: int, filename: str) -> str:
        """Extract text from a single PDF page using OCR with pytesseract"""
        try:
            import pytesseract
            from pdf2image import convert_from_bytes
            from PIL import Image
            
            # Convert specific page to image
            # Note: page_num is 0-indexed, but first_page/last_page are 1-indexed
            images = convert_from_bytes(
                file_content,
                first_page=page_num + 1,
                last_page=page_num + 1,
                dpi=300  # Higher DPI for better OCR accuracy
            )
            
            if not images:
                logger.warning(f"Could not convert page {page_num + 1} to image for {filename}")
                return ""
            
            # Get the first (and only) image
            page_image = images[0]
            
            # Perform OCR on the image
            extracted_text = pytesseract.image_to_string(page_image, lang='eng')
            
            # Log extraction results
            char_count = len(extracted_text.strip()) if extracted_text else 0
            logger.info(
                f"OCR extracted {char_count} characters from page {page_num + 1} of {filename}"
            )
            
            # Warn if no text was extracted
            if not extracted_text or char_count == 0:
                logger.warning(
                    f"OCR returned empty text for page {page_num + 1} of {filename}"
                )
                return ""
            
            return extracted_text
                    
        except ImportError as e:
            logger.error(f"Required OCR library not installed: {str(e)}")
            raise HTTPException(
                status_code=500, 
                detail="OCR libraries not available. Please install pytesseract, pdf2image, and Pillow. "
                       "Also ensure tesseract-ocr is installed on your system."
            )
        except Exception as e:
            logger.error(f"OCR extraction error for page {page_num + 1} of {filename}: {str(e)}")
            # Don't raise exception, just return empty string to allow processing to continue
            logger.warning(f"Returning empty string for page {page_num + 1} due to OCR error")
            return ""

