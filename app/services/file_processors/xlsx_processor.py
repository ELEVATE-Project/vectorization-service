import io
import uuid
import pandas as pd
import numpy as np
import logging
import traceback
from datetime import datetime
from typing import List, Dict, Any
from fastapi import HTTPException
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_core.documents import Document as LangchainDocument

from .base_processor import BaseFileProcessor
from app.services.translation_service import process_chunk
from app.config import settings

logger = logging.getLogger(__name__)

class XLSXProcessor(BaseFileProcessor):
    def get_supported_extensions(self) -> List[str]:
        return ['.xlsx', '.xls']

    def _sanitize_cell_content(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans up cell content by converting internal line breaks to HTML <br> tags.
        This step runs BEFORE markdown conversion.
        """
        import re
        
        def replace_breaks(text):
            if isinstance(text, str):
                # Condense internal whitespace and clean up line breaks for <br> conversion
                text = re.sub(r'\s+', ' ', text)
                text = text.replace('\r\n', '<br>').replace('\n', '<br>').replace('\r', '<br>')
                return text.strip()
            return text
        
        return df.map(replace_breaks)

    def _post_process_markdown(self, markdown_text: str) -> str:
        """
        Performs final cleanup on the generated Markdown text by:
        1. Reducing repetitive hyphens within general text.
        2. Standardizing the machine-generated table separator line.
        3. Condensing multiple spaces.
        """
        import re
        
        lines = markdown_text.splitlines()
        cleaned_lines = []

        for line in lines:
            # 1. Condense multiple hyphens (more than 3 repeating hyphens to 1 hyphen)
            line = re.sub(r'-{4,}', '-', line)
            
            # 2. Check for the machine-generated separator line and standardize it.
            if re.search(r'^\s*\|[:\s\-]+\|', line) and not re.search(r'[a-zA-Z0-9<>]', line):
                # Count the number of delimiters (pipes) to determine column count.
                num_columns = line.count('|') - 1
                if num_columns > 0:
                    # Create the standard separator: |:---|:---|...| for left alignment
                    standard_separator = '|' + ':---|' * num_columns
                    cleaned_lines.append(standard_separator)
                    continue
            
            # 3. Condense multiple whitespace characters
            line = re.sub(r'\s{2,}', ' ', line)
            
            # Add the cleaned line
            cleaned_lines.append(line.strip())

        # 4. Remove any empty lines created by the filtering process
        final_output = '\n'.join([line for line in cleaned_lines if line])
        
        # Condense multiple blank lines for a clean look
        final_output = re.sub(r'\n{3,}', '\n\n', final_output)
        
        return final_output.strip()

    def _convert_to_markdown(self, df: pd.DataFrame, filename: str) -> str:
        """
        Convert DataFrame to markdown format with advanced sanitization and cleanup.
        Creates a markdown table with proper formatting, handling line breaks and special characters.
        """
        from tabulate import tabulate
        
        # Clean NaN and Inf values before conversion
        df_clean = df.copy()
        for col in df_clean.columns:
            df_clean[col] = df_clean[col].apply(
                lambda x: '' if pd.isna(x) or (isinstance(x, float) and np.isinf(x)) else x
            )
        
        # Convert column names to strings
        df_clean.columns = df_clean.columns.astype(str)
        
        # Sanitize cell content (handle line breaks, etc.)
        df_clean = self._sanitize_cell_content(df_clean)
        
        # Create markdown header with filename
        markdown_content = f"# {filename}\n\n"
        markdown_content += "## Data Table\n\n"
        
        # Convert DataFrame to markdown table using tabulate
        try:
            markdown_table = tabulate(df_clean, headers='keys', tablefmt='pipe', showindex=False)
        except Exception as e:
            logger.warning(f"Tabulate conversion failed: {e}, falling back to manual method")
            # Fallback to manual markdown table creation
            markdown_table = self._create_markdown_table_manual(df_clean)
        
        markdown_content += markdown_table
        markdown_content += "\n\n---\n"
        
        # Post-process the markdown for cleanup
        markdown_content = self._post_process_markdown(markdown_content)
        
        return markdown_content

    def _convert_sheet_to_markdown(self, df: pd.DataFrame, sheet_name: str) -> str:
        """
        Convert a single Excel sheet to markdown format.
        Used for multi-sheet Excel files.
        """
        from tabulate import tabulate
        
        # Clean NaN and Inf values before conversion
        df_clean = df.copy()
        for col in df_clean.columns:
            df_clean[col] = df_clean[col].apply(
                lambda x: '' if pd.isna(x) or (isinstance(x, float) and np.isinf(x)) else x
            )
        
        # Convert column names to strings
        df_clean.columns = df_clean.columns.astype(str)
        
        # Sanitize cell content (handle line breaks, etc.)
        df_clean = self._sanitize_cell_content(df_clean)
        
        # Create markdown section for this sheet
        markdown_content = f"## {sheet_name}\n\n"
        
        # Convert DataFrame to markdown table using tabulate
        try:
            markdown_table = tabulate(df_clean, headers='keys', tablefmt='pipe', showindex=False)
        except Exception as e:
            logger.warning(f"Tabulate conversion failed for sheet '{sheet_name}': {e}, falling back to manual method")
            markdown_table = self._create_markdown_table_manual(df_clean)
        
        markdown_content += markdown_table
        markdown_content += "\n\n---"
        
        # Post-process the markdown for cleanup
        markdown_content = self._post_process_markdown(markdown_content)
        
        return markdown_content

    def _create_markdown_table_manual(self, df: pd.DataFrame) -> str:
        """
        Manually create a markdown table from DataFrame.
        Fallback method if tabulate is not available.
        """
        # Create header row
        headers = "| " + " | ".join(str(col) for col in df.columns) + " |"
        separator = "|" + ":---|" * len(df.columns)
        
        # Create data rows
        rows = []
        for _, row in df.iterrows():
            row_str = "| " + " | ".join(str(val) for val in row.values) + " |"
            rows.append(row_str)
        
        # Combine all parts
        markdown_table = "\n".join([headers, separator] + rows)
        return markdown_table

    def _create_rag_optimized_chunks(self, df: pd.DataFrame, filename: str, sheet_name: str = None) -> str:
        """
        Create RAG-optimized markdown format where each row is a self-contained chunk
        with column names explicitly paired with values for better semantic retrieval.
        
        Format for each row:
        ### Row N
        **Column1**: Value1
        **Column2**: Value2
        ...
        
        This ensures each chunk is semantically complete and searchable.
        """
        # Clean NaN and Inf values
        df_clean = df.copy()
        for col in df_clean.columns:
            df_clean[col] = df_clean[col].apply(
                lambda x: '' if pd.isna(x) or (isinstance(x, float) and np.isinf(x)) else x
            )
        
        # Convert column names to strings and sanitize
        df_clean.columns = df_clean.columns.astype(str)
        df_clean = self._sanitize_cell_content(df_clean)
        
        # Build markdown content
        markdown_content = f"# {filename}\n\n"
        if sheet_name:
            markdown_content += f"## {sheet_name}\n\n"
        
        # Add each row as column-value pairs WITHOUT row numbers
        for idx, row in df_clean.iterrows():
            # Add column-value pairs directly (no row header)
            for col in df_clean.columns:
                value = row[col]
                if value and str(value).strip():  # Only include non-empty values
                    markdown_content += f"**{col}**: {value}\n\n"
            
            # Separator between rows
            markdown_content += "---\n\n"
        
        return markdown_content.strip()

    def _split_markdown(self, doc: LangchainDocument) -> List[LangchainDocument]:
        """
        Split markdown content using MarkdownHeaderTextSplitter.
        For XLSX RAG-optimized format, we only split on H1 (filename) and H2 (sheet name),
        NOT on H3 (row headers), allowing multiple rows to be grouped in one chunk.
        """
        # Only split on H1 and H2 headers (NOT H3 for rows)
        # This allows multiple ### Row N sections to be grouped together
        headers_to_split_on = [
            ("#", "Header 1"),      # Filename
            ("##", "Header 2"),     # Sheet name
        ]

        # First split by headers
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False  # Keep headers in the content for context
        )
        
        # Split the document by headers
        md_header_splits = markdown_splitter.split_text(doc.page_content)
        
        # Use markdown-specific chunk size for better granularity
        # This will now group multiple rows together based on character count
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.MARKDOWN_CHUNK_SIZE,
            chunk_overlap=settings.MARKDOWN_CHUNK_OVERLAP,
            length_function=len,
            separators=["---\n\n", "\n\n", "\n", ". ", " ", ""],  # Prioritize row separators
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
            
            # Split based on character count, allowing multiple rows per chunk
            if len(md_chunk.page_content) > settings.MARKDOWN_CHUNK_SIZE:
                sub_chunks = text_splitter.split_documents([temp_doc])
                final_chunks.extend(sub_chunks)
            else:
                # Even small chunks go through to maintain consistency
                final_chunks.append(temp_doc)
        
        return final_chunks

    async def process(self, file_content: bytes, filename: str, priority: str) -> List[Dict[str, Any]]:
        """
        Process XLSX file by converting to markdown first, then chunking.
        Flow: XLSX → Markdown → Chunking → Translation
        Supports multi-sheet Excel files.
        """
        self._validate_file_content(file_content, filename)

        try:
            # Step 1: Read XLSX with pandas
            file_extension = filename.lower().split('.')[-1]
            
            if file_extension in ['xlsx', 'xls']:
                # Handle multi-sheet Excel files
                xls = pd.ExcelFile(io.BytesIO(file_content))
                all_sheets_markdown = f"# {filename}\n\n"
                total_rows = 0
                all_columns = []
                
                logger.info(f"Processing Excel file '{filename}' with {len(xls.sheet_names)} sheet(s): {xls.sheet_names}")
                
                for sheet_name in xls.sheet_names:
                    # Read sheet without assuming header row
                    df = xls.parse(sheet_name, header=None, keep_default_na=False)
                    
                    if df.empty:
                        logger.info(f"Skipping empty sheet: {sheet_name}")
                        continue
                    
                    # Use first row as headers
                    df.columns = df.iloc[0]
                    df = df[1:]
                    df.reset_index(drop=True, inplace=True)
                    
                    # Clean column names
                    df.columns = df.columns.astype(str).str.strip()
                    
                    # Drop unnamed columns
                    unnamed_cols = [col for col in df.columns if "Unnamed:" in str(col) or col.strip() == '']
                    if unnamed_cols:
                        df = df.drop(columns=unnamed_cols)
                        logger.info(f"Dropped unnamed columns from sheet '{sheet_name}': {unnamed_cols}")
                    
                    total_rows += len(df)
                    all_columns.extend(df.columns.tolist())
                    
                    # Convert sheet to RAG-optimized markdown
                    sheet_markdown = self._create_rag_optimized_chunks(df, filename, sheet_name)
                    all_sheets_markdown += sheet_markdown + "\n\n"
                
                markdown_content = all_sheets_markdown.strip()
                
            else:
                # Single sheet processing (fallback)
                df = pd.read_excel(io.BytesIO(file_content))
                logger.info(f"Processing single-sheet file '{filename}' with {len(df)} rows")
                
                # Clean column names
                df.columns = df.columns.str.strip()
                
                # Drop unnamed columns
                unnamed_cols = [col for col in df.columns if "Unnamed:" in str(col)]
                if unnamed_cols:
                    df = df.drop(columns=unnamed_cols)
                    logger.info(f"Dropped unnamed columns: {unnamed_cols}")
                
                total_rows = len(df)
                all_columns = df.columns.tolist()
                markdown_content = self._create_rag_optimized_chunks(df, filename)
            
            logger.info(f"Converted XLSX to RAG-optimized markdown format ({len(markdown_content)} characters)")

            # Step 3: Create LangChain document from markdown
            doc = LangchainDocument(
                page_content=markdown_content,
                metadata={
                    "source": filename,
                    "type": "xlsx_rag_optimized",
                    "priority": priority,
                    "is_markdown": True,
                    "original_format": "xlsx",
                    "total_rows": total_rows,
                    "columns": list(set(all_columns)),  # Unique columns across all sheets
                    "rag_optimized": True,
                    "format_description": "Each row is a self-contained chunk with column-value pairs",
                },
            )

            # Step 4: Split markdown using the same logic as text processor
            chunks = self._split_markdown(doc)
            logger.info(f"Split markdown into {len(chunks)} RAG-optimized chunks")

            # Step 5: Process chunks (translation and metadata enrichment)
            processed_chunks = []
            for idx, chunk in enumerate(chunks):
                chunk_id = uuid.uuid4().hex
                
                # Process the chunk text (handle translations if needed)
                processed_text, is_hindi = process_chunk(
                    chunk.page_content,
                    chunk_id
                )
                
                # Create processed chunk with metadata
                processed_chunk = {
                    "id": chunk_id,
                    "text": processed_text,
                    "metadata": {
                        **chunk.metadata,
                        "is_hindi": is_hindi,
                        "chunk_index": idx,
                        "total_chunks": len(chunks),
                        "created_at": datetime.now().isoformat(),
                        "updated_at": datetime.now().isoformat(),
                    }
                }
                
                processed_chunks.append(processed_chunk)

            logger.info(f"Successfully processed XLSX '{filename}' into {len(processed_chunks)} RAG-optimized chunks")
            return processed_chunks

        except Exception as e:
            logger.error(f"XLSX processing error for '{filename}': {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"Error processing XLSX: {str(e)}")
