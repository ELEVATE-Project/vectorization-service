# File Processors

Read this page to understand how each uploaded file type is turned into chunks, which settings control chunk size, and how to add support for a new file type. The code lives in `app/services/file_processors/`. Processors are selected and invoked by `UploadService` (see [Ingestion pipeline](pipeline.md)).

## 1. Contract: `BaseFileProcessor`

`app/services/file_processors/base_processor.py` defines the abstract base (the class is named `BaseFileProcessor`):

```python
class BaseFileProcessor(ABC):
    def __init__(self):
        self.supported_extensions = self.get_supported_extensions()

    @abstractmethod
    def get_supported_extensions(self) -> List[str]: ...

    @abstractmethod
    async def process(self, file_content: bytes, filename: str, priority: str) -> List[Dict[str, Any]]: ...

    def can_process(self, file_extension: str) -> bool: ...
    def _validate_file_content(self, file_content: bytes, filename: str): ...
```

Contract:

- `get_supported_extensions()` returns extensions **with the leading dot**, lower-case (`['.pdf']`). `UploadService` looks up `f".{ext}"`.
- `process()` returns a list of chunk dicts: `{"id": <uuid4 hex>, "text": <str>, "metadata": <dict>}`. `UploadService._upload_chunks` skips any chunk missing one of these keys.
- `_validate_file_content()` raises `ValueError` for empty content and for files larger than `MAX_FILE_SIZE_MB` (default 1024). Every processor calls it first, outside its own `try`, so the `ValueError` propagates to `UploadService.process`, which maps it to HTTP 500.
- Each processor passes every chunk through `process_chunk()` (translation hook, currently a no-op returning `(text, False)`) and records `is_hindi` in metadata.
- Processors add their own metadata keys (`source`, `type`, `priority`, ...); `UploadService` later merges user metadata over them, so user keys win.

`app/services/file_processors/__init__.py` only exports `CSVProcessor`, `PDFProcessor`, `DOCXProcessor`, `XLSXProcessor`. `TextProcessor` is not exported there but is imported directly by `upload_service.py`.

## 2. Extension map and settings

| Processor | Extensions | Splitter | Size / overlap |
|---|---|---|---|
| `PDFProcessor` | `.pdf` | `RecursiveCharacterTextSplitter` | `CHUNK_SIZE` 3000 / `CHUNK_OVERLAP` 500 |
| `DOCXProcessor` | `.docx`, `.doc` | `RecursiveCharacterTextSplitter` | 3000 / 500 |
| `XLSXProcessor` | `.xlsx`, `.xls` | `MarkdownHeaderTextSplitter` (H1/H2) + recursive | `MARKDOWN_CHUNK_SIZE` 3500 / `MARKDOWN_CHUNK_OVERLAP` 800 |
| `CSVProcessor` | `.csv` | recursive, only if one record exceeds `CHUNK_SIZE` | 3000 / 500 |
| `TextProcessor` | `.txt`, `.md`, `.markdown`, `.text` | markdown: header split + recursive; plain: recursive | markdown 3500 / 800, plain 3000 / 500 |

Other relevant settings (`app/config.py`): `PAGE_TEXT_THRESHOLD` = 20 (PDF OCR trigger), `MAX_FILE_SIZE_MB` = 1024. `PDF_CHUNK_SIZE`/`PDF_CHUNK_OVERLAP` (3000/500) are defined but not used by any code.

### 2.1 Chunk overlap: behaviour and rationale

**What the code does.** Every splitter is a LangChain `RecursiveCharacterTextSplitter` with `length_function=len`, so `chunk_size` and `chunk_overlap` are measured in **characters**, not tokens. The splitter tries separators in order (`"\n\n"`, `"\n"`, `". "`, `" "`, `""`; the XLSX splitter additionally tries `"---\n\n"` first) and carries the trailing `chunk_overlap` characters of each chunk into the start of the next one.

| Content | Size | Overlap | Overlap as share of chunk |
|---|---|---|---|
| PDF, DOCX, CSV, plain text (`CHUNK_SIZE` / `CHUNK_OVERLAP`) | 3000 | 500 | ~17% |
| Markdown, XLSX (`MARKDOWN_CHUNK_SIZE` / `MARKDOWN_CHUNK_OVERLAP`) | 3500 | 800 | ~23% |
| URL-extracted content (`URL_EXTRACTION_CHUNK_SIZE` / `URL_EXTRACTION_CHUNK_OVERLAP`) | 1500 | 300 | 20% |

**Why overlap exists (general chunking rationale).** A hard cut can separate a statement from the sentence that gives it meaning. Repeating a short tail of the previous chunk at the start of the next keeps boundary-spanning content retrievable from at least one chunk, and gives each embedding some surrounding context. The larger overlap on Markdown/XLSX content is consistent with that content being structured (headers, table rows), where a split loses more context than in prose. The repository does not record why these specific values were chosen; treat the table above as the configured behaviour, not a documented decision.

**Consequences visible elsewhere in the system.**

- Overlapping text is stored twice, so adjacent chunks of one document can both match a query. Search collapses this per document with `_filter_best_per_source` (see [Boosts, Filters and Results](../search/boosts_filters_and_results.md)).
- Storage and BM25 index size grow by roughly the overlap share.
- Changing `CHUNK_SIZE` or `CHUNK_OVERLAP` affects only documents ingested afterwards. Existing points keep their old chunking until each document is re-ingested (see [Migration and Restore](../../operations/migration_and_restore.md)).
- Chunk size interacts with the embedding model's input limit. `app/core/clients/embedding.py` constructs `SentenceTransformer(settings.EMBEDDING_MODEL)` and does not override `max_seq_length`, so the model's own default applies. `all-MiniLM-L6-v2` is documented by its authors with a 256-token input limit, which is shorter than a 3000-character chunk (roughly 600-750 English tokens); text beyond the limit does not influence the dense `text` vector. Confirm the active value with `embedding_model.max_seq_length`. The BM25 sparse vector is computed from the full chunk text, so keyword search is not affected by this limit.

## 3. PDFProcessor

File: `pdf_processor.py`. Extraction is per page with an OCR fallback:

```python
for page_num, page in enumerate(pdf_reader.pages, start=1):
    page_text = page.extract_text()
    if len(page_text.strip()) < settings.PAGE_TEXT_THRESHOLD:
        try:
            ocr_text = await self._extract_page_with_ocr(file_content, page_num - 1, filename)
            pages_text.append(ocr_text); pages_with_ocr += 1
        except Exception as e:
            pages_text.append(page_text); pages_with_text += 1
    else:
        pages_text.append(page_text); pages_with_text += 1
full_text = "\n\n".join(pages_text)
```

Flow:

1. `PyPDF2.PdfReader` reads the bytes; each page is extracted individually.
2. A page with fewer than `PAGE_TEXT_THRESHOLD` characters (after `strip()`) is OCR'd: `pdf2image.convert_from_bytes(first_page=n, last_page=n, dpi=300)` then `pytesseract.image_to_string(image, lang='eng')`. Requires system `tesseract` and `poppler` (`pdftoppm`).
3. `_extract_page_with_ocr` swallows OCR runtime errors and returns `""` (so the page contributes empty text and still counts as an OCR page). Only a missing library (`ImportError`) raises an `HTTPException`, which the caller catches and falls back to the sparse PyPDF2 text.
4. Pages are joined with a blank line. If the result is empty, a 400 is raised.
5. `extraction_method` is `text`, `ocr` or `mixed`.
6. One LangChain `Document` is split with `RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, length_function=len)`.

Chunk metadata: `source`, `type="pdf"`, `priority`, `extraction_method`, `total_pages`, `pages_with_text`, `pages_with_ocr`, `ocr_pages` (list or `None`), `is_hindi`, `total_chunks`, `created_at`, `updated_at`. Page numbers per chunk are not tracked; chunks span the concatenated text.

Note: OCR is language-fixed to English (`lang='eng'`), and runs sequentially per page, so large scanned PDFs are slow and block the event loop.

## 4. DOCXProcessor

File: `docx_processor.py`. Uses `python-docx`:

```python
doc = Document(io.BytesIO(file_content))
for para in doc.paragraphs:
    text_content += para.text + "\n"
```

Only body paragraphs are read. Tables, headers/footers, text boxes and images are ignored. Splitting uses `CHUNK_SIZE`/`CHUNK_OVERLAP`. Metadata: `source`, `type="docx"`, `priority`, `is_hindi`, `created_at`, `updated_at` (no `total_chunks`).

Note: `.doc` is registered as an extension but `python-docx` cannot open legacy binary `.doc` files; such uploads fail with HTTP 500.

## 5. XLSXProcessor

File: `xlsx_processor.py`. Converts spreadsheets to a "RAG-optimized" markdown and splits it:

1. `pd.ExcelFile` reads every sheet (`xlsx` and `xls` both take this branch; the "single sheet fallback" `else` branch is unreachable because the extension is always one of the two). Each sheet is parsed with `header=None, keep_default_na=False`; the first row becomes the header; empty sheets are skipped; unnamed/blank-named columns are dropped.
2. `_create_rag_optimized_chunks` renders each row as bold column/value pairs and separates rows with `---`:

```text
# <filename>

## <sheet name>

**Column1**: value

**Column2**: value

---
```

   Empty cells are omitted. Cell text is whitespace-condensed and newlines become `<br>` (`_sanitize_cell_content`); NaN/Inf become empty strings.
3. `_split_markdown` splits on H1 and H2 headers only (`MarkdownHeaderTextSplitter`, `strip_headers=False`), then any section longer than `MARKDOWN_CHUNK_SIZE` is split with `RecursiveCharacterTextSplitter` using separators `["---\n\n", "\n\n", "\n", ". ", " ", ""]`, so multiple rows are grouped per chunk and splits prefer row boundaries.

Chunk metadata: `source`, `type="xlsx_rag_optimized"`, `priority`, `is_markdown=True`, `original_format="xlsx"`, `total_rows`, `columns` (unique column names), `rag_optimized=True`, `format_description`, header metadata (`Header 1`, `Header 2`), `is_hindi`, `chunk_index`, `total_chunks`, `created_at`, `updated_at`.

Note: `_convert_to_markdown`, `_convert_sheet_to_markdown`, `_post_process_markdown` and `_create_markdown_table_manual` (table-style output) are not called by `process()`. `.xls` additionally requires the `xlrd` engine to be installed. Each sheet's markdown starts with its own `# <filename>` line, so the combined document repeats the H1 per sheet.

## 6. CSVProcessor

File: `csv_processor.py`. This processor is schema-specific, not generic. It expects project/task style columns defined as constants (`SL NO`, `Sub-cateogry` [sic], `TITLE OF THE PROJECT`, `TARGET STAKEHOLDER`, `DURATION`, `DESCRIPTION`, `OBJECTIVE`, `PROJECT LEVEL LEARNING RESOURCE`, `TASK NAME`, `SUB TASK (If any)`, `NAME OF TASK LEVEL LEARNING RESOURCE`).

Behaviour:

- `pd.read_csv`, strip column names, drop `Unnamed:` columns.
- Only `SL NO` is mandatory (`HTTPException` 400 if missing; note this is raised inside the processor's own `try`, whose `except Exception` re-wraps it as HTTP 500). Other columns are optional.
- The "main" columns are forward-filled (`ffill`), rows are grouped by unique `SL NO`, and one text block is built per group ("Task Number: ... Tasks and Subtasks: ...").
- If a block is longer than `CHUNK_SIZE` it is split with the recursive splitter; otherwise it is one chunk.

Chunk metadata includes the main fields (`sl_no`, `sub_category`, `project_title`, ...), the full `tasks` list of dicts, `priority`, `source`, `type="project_task"`, `is_hindi`, `total_chunks`, `created_at`, `updated_at`.

Note: A generic CSV without `SL NO` is rejected. Cell values are stringified, so missing values appear as the text `nan`.

## 7. TextProcessor

File: `text_processor.py`. Handles `.txt`, `.md`, `.markdown`, `.text`.

- Decodes as UTF-8, falls back to latin-1; blank content gives 400.
- `_is_markdown` returns True when at least two of ten regex patterns (headers, bold, links, code fences, lists, blockquotes, tables, rules, images) match with `re.MULTILINE`. This applies to `.txt` files too: a `.txt` with markdown syntax is treated as markdown.
- Markdown: `MarkdownHeaderTextSplitter` on `#` to `####` (`strip_headers=False`), then sections longer than the chunk size are sub-split recursively with separators `["\n\n", "\n", ". ", " ", ""]`. Defaults `MARKDOWN_CHUNK_SIZE`/`MARKDOWN_CHUNK_OVERLAP`.
- Plain text: `RecursiveCharacterTextSplitter` with `CHUNK_SIZE`/`CHUNK_OVERLAP`.
- `process()` accepts optional `chunk_size`/`chunk_overlap` overrides, but `UploadService` never passes them.

Metadata: `source`, `type` (`"markdown"` or `"text"`), `priority`, `is_markdown`, header keys (`Header 1`..`Header 4` when present), `is_hindi`, `total_chunks`, `created_at`, `updated_at`.

## 8. How to add a new processor

1. Create `app/services/file_processors/<name>_processor.py` with a class extending `BaseFileProcessor`.
2. Implement `get_supported_extensions()` (dotted, lower-case) and `async process(file_content, filename, priority)`:
   - call `self._validate_file_content(file_content, filename)` first;
   - extract text, build a LangChain `Document` or split manually using the appropriate settings (`CHUNK_SIZE`/`CHUNK_OVERLAP`, or the markdown variants);
   - for each chunk produce `{"id": uuid.uuid4().hex, "text": processed_text, "metadata": {...}}` where `processed_text, is_hindi = process_chunk(chunk.page_content, chunk_id)`;
   - include at least `source`, `type`, `priority`; add `created_at`/`updated_at` (they are overwritten by `UploadService` anyway);
   - raise `HTTPException(400, ...)` for user errors, and avoid wrapping it in a blanket `except Exception` that converts it to 500 (see gotchas).
3. Register it in `UploadService.__init__` (`self.processors` list) in `app/services/document_operations/upload_service.py`. The extension map is built automatically.
4. Optionally export it from `file_processors/__init__.py`.
5. Add any new settings to `app/config.py` and document them in [Configuration](../../setup/configuration.md).
6. Remember that anything stored in chunk metadata ends up in the Qdrant payload; keep it small and JSON-serialisable.

## Known issues / gotchas

- PDF: `except Exception` wraps the intentional 400 "Could not extract any text" into a 500 ("Error processing PDF: 400: ..."), and OCR runtime errors yield empty page text rather than the PyPDF2 text.
- CSV: the 400 for a missing `SL NO` column becomes a 500 for the same reason.
- DOCX: tables are ignored; `.doc` not actually supported.
- XLSX: dead conversion helpers; `.xls` needs `xlrd`; repeated H1 per sheet.
- `PDF_CHUNK_SIZE`/`PDF_CHUNK_OVERLAP` unused.
- `__init__.py` does not export `TextProcessor`.
- File size violations surface as 500 (not 413) and are not checked for URL content.
- Chunk metadata is not uniform across processors (`total_chunks` missing for DOCX; `chunk_index` only for XLSX).

## Related pages

- [Ingestion pipeline](pipeline.md)
- [Document operations](document_operations.md)
- [Configuration](../../setup/configuration.md)
- [Qdrant data model](../../architecture/qdrant_data_model.md)
