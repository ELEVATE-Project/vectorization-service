# Ingestion Pipeline

Read this page to understand exactly what happens between `POST /api/v1/documents` and points landing in Qdrant: validation, extraction, chunking, metadata merging, the (disabled) translation hook, the five dense embeddings, the optional BM25 sparse vector, `PointStruct` construction and batched upsert. The module responsible is `app/services/document_operations/upload_service.py` (`UploadService`), reached through the `DocumentProcessor` facade in `app/services/document_processor.py`.

## 1. Call chain

```text
POST /api/v1/documents                      app/api/v1/endpoints/documents.py :: create_documents
  |  Depends(parse_metadata_form)           JSON string -> dict   (400 on bad JSON / non-object)
  |  Depends(parse_tags_form)               JSON array OR "a, b, c" -> list[str]
  v
DocumentProcessor.process_upload()          app/services/document_processor.py  (pass-through)
  v
UploadService.process()
  |-- validate_source_id()                  400 if None / empty / whitespace
  |-- validate_priority()                   400 unless value starts with "P" (case-insens.)
  |-- merge company_id/title/summary/tags into additional_metadata
  |-- ensure_collections()                  ensure_collections_exist() in qdrant.py
  |-- IF metadata.markdown_url:  _process_url_text()      (URL path, section 6)
  |   ELSE: file.read() -> _process_file_by_type()        (processor chosen by extension)
  |-- 400 if no chunks
  v
UploadService._upload_chunks()
  |-- generate_embeddings([chunk.text ...])             one batch encode -> "text" vectors
  |-- _generate_field_embeddings(title, summary, tags, metadata)
  |-- (optional) generate_sparse_vector(chunk.text)     only if SPARSE_SEARCH_ENABLED
  |-- per chunk: _prepare_chunk_metadata(), _create_point_vectors(), PointStruct
  v
upload_to_qdrant(points, COLLECTION_NAME, batch_size=100)   app/core/clients/qdrant.py
```

`DocumentProcessor` is a thin facade holding one instance each of `UploadService`, `UpdateService`, `DeleteService` and `MetadataService`. It contains no logic. `documents.py` creates a single module-level `DocumentProcessor()`.

## 2. Endpoint inputs

`create_documents` accepts multipart form data:

| Field | Type | Default | Notes |
|---|---|---|---|
| `file` | `UploadFile` | required | Always required, even when `metadata.markdown_url` is used (the filename is still used as `source`). |
| `priority` | str | `"P1"` | Must start with `P`. |
| `source_id` | str | `None` | Declared optional but `validate_source_id` rejects `None` with 400. |
| `company_id` | str | `None` | Stored as `metadata.company`. |
| `title`, `summary` | str | `None` | Stored top-level in the payload and in `metadata`. |
| `metadata` | JSON object string | `None` | Parsed by `parse_metadata_form`; must be an object. |
| `tags` | JSON array or CSV string | `None` | `parse_tags_form`. JSON list items are not type-checked. |

The endpoint returns HTTP 201 with the dict built at the end of `UploadService.process` (`chunks_processed`, `points_uploaded`, `upload_failures`, `file_type`, `priority` upper-cased, `sample_chunk`, `supported_file_types`, ...).

## 3. Validation

`BaseDocumentOperation` (see [Document operations](document_operations.md)) supplies:

```python
def validate_priority(self, priority: str) -> None:
    if not priority or not priority.upper().startswith("P"):
        raise HTTPException(status_code=400, detail="Invalid priority format. Must be P1, P2, P3, etc.")
```

Note: the check is only "starts with P". `"PX"` or `"please"` pass. The response upper-cases the priority, but processors write the priority into chunk metadata exactly as sent.

File size is enforced by each processor through `BaseFileProcessor._validate_file_content` (`MAX_FILE_SIZE_MB`, default 1024), not at the endpoint. The whole file is read into memory first (`await file.read()`). Unsupported extension raises 400 listing `get_supported_file_types()`. The URL path performs no size check at all.

## 4. Metadata handling

`UploadService.process` mutates the dict it receives:

```python
additional_metadata = metadata if metadata else {}
if company_id:
    additional_metadata['company'] = company_id
if title:
    additional_metadata['title'] = title
if summary:
    additional_metadata['summary'] = summary
if tags:
    additional_metadata['tags'] = tags
```

Consequences:

- `title`, `summary` and `tags` are duplicated inside `payload.metadata` as well as the top-level payload keys.
- Because `additional_metadata` feeds the metadata embedding (section 7), the `metadata` vector text includes company, title, summary and tags.

Per-chunk metadata is built in `_prepare_chunk_metadata`:

```python
chunk_metadata = chunk["metadata"].copy()
if additional_metadata:
    chunk_metadata.update(additional_metadata)     # user metadata WINS over processor keys
# ... company/title/summary/tags only if not already present ...
if source_id:
    chunk_metadata['source_id'] = source_id
current_time = datetime.now().isoformat()
chunk_metadata['created_at'] = current_time
chunk_metadata['updated_at'] = current_time
```

Note: user-supplied metadata keys such as `type`, `source`, `priority` overwrite the values the processor set. `metadata.type` backs the `file_type` search filter, so a user-supplied `type` changes how the document is filtered. `created_at`/`updated_at` always overwrite whatever the processor wrote. `source_id` is stored both top-level and in `metadata.source_id`.

## 5. Chunking choice

`UploadService.__init__` registers processors and builds an extension map:

```python
self.processors = [CSVProcessor(), PDFProcessor(), DOCXProcessor(), XLSXProcessor(), TextProcessor()]
self.processor_map = {}
for processor in self.processors:
    for ext in processor.supported_extensions:
        self.processor_map[ext] = processor
```

The extension is `file.filename.split(".")[-1].lower()` (defaults to `"txt"` if no filename) and is looked up as `f".{file_extension}"`. Supported: `.csv .pdf .docx .doc .xlsx .xls .txt .md .markdown .text`.

| Source | Splitter | Size / overlap (settings) |
|---|---|---|
| PDF, DOCX, plain text, CSV (long rows) | `RecursiveCharacterTextSplitter` | `CHUNK_SIZE`=3000 / `CHUNK_OVERLAP`=500 |
| Markdown text, XLSX | `MarkdownHeaderTextSplitter` then recursive split | `MARKDOWN_CHUNK_SIZE`=3500 / `MARKDOWN_CHUNK_OVERLAP`=800 |
| `markdown_url` content | `RecursiveCharacterTextSplitter` | `URL_EXTRACTION_CHUNK_SIZE`=1500 / `URL_EXTRACTION_CHUNK_OVERLAP`=300 |

Each processor returns `list[{"id", "text", "metadata"}]`. Details per processor are in [File processors](file_processors.md).

### Chunk id generation

Every processor generates `chunk_id = uuid.uuid4().hex` (32 hex chars, accepted by Qdrant as a UUID point id). Ids are random, not derived from content or `source_id`. Uploading the same file twice via `POST /documents` therefore creates duplicate points; deduplication only happens through the update/upsert endpoints, which delete by `source_id` first.

## 6. URL path (`metadata.markdown_url`)

If `additional_metadata['markdown_url']` is truthy the uploaded file body is ignored (never read) and `_process_url_text` runs:

```python
extracted_text = await url_extractor.extract_text(url)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=settings.URL_EXTRACTION_CHUNK_SIZE,
    chunk_overlap=settings.URL_EXTRACTION_CHUNK_OVERLAP,
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""]
)
```

`file_extension` is then set to `"url_extracted"`, which is what the response reports as `file_type`; chunk metadata gets `type: "url_extracted"`, `source: file.filename`, `priority`, `is_markdown: False`, `is_hindi`, `total_chunks`.

`URLTextExtractor` (`app/services/url_text_extractor.py`):

| Aspect | Behaviour |
|---|---|
| Validation | Empty URL: 400. Must start with `http://` or `https://`: 400. |
| Fetch | `httpx.AsyncClient(timeout=URL_REQUEST_TIMEOUT (30s), follow_redirects=True)`, desktop Chrome `User-Agent`. |
| Errors | Timeout 408, connect error 503, HTTP status error re-raised with the same status, other 500. |
| Content type | `text/html` -> BeautifulSoup (`lxml`); `text/plain` / `text/markdown` -> raw `response.text`; anything else -> attempted as HTML. |
| HTML cleanup | Removes `script, style, nav, footer, header, aside`; tries selectors `main, article, [role="main"], .main-content, #main-content, .content, #content, .post-content, .article-content, .entry-content`; falls back to `<body>` then whole document; text joined with newlines, blank lines dropped. Parse failure falls back to `html.parser`. |
| Empty result | 400 "No text content could be extracted". |

Note: there is no SSRF protection (any reachable `http(s)` URL, including internal addresses, is fetched) and no response size limit.

## 7. Embeddings

Text embeddings are one batch call:

```python
text_embeddings = generate_embeddings([chunk["text"] for chunk in processed_chunks])
```

Document-level field embeddings come from `_generate_field_embeddings`, each computed once and shared by all chunks:

```python
embeddings['title']   = generate_embeddings([title])[0]            # if title.strip()
embeddings['summary'] = generate_embeddings([summary])[0]          # if summary.strip()
embeddings['tags']    = generate_embeddings([", ".join(tags)])[0]  # if tags non-empty
metadata_text = " ".join([f"{k}: {v}" for k, v in additional_metadata.items() if v])
embeddings['metadata'] = generate_embeddings([metadata_text])[0]   # if metadata_text.strip()
```

Rules:

- A field vector is created only when its source is present. A point without a title simply has no `title` named vector (Qdrant allows partial named vectors). Documents uploaded without title/summary/tags therefore never match on those fields.
- The metadata text is built from `additional_metadata` (request metadata plus injected company/title/summary/tags), not from the chunk metadata. Values are rendered with Python `str()`, so lists appear as `['a', 'b']`. Falsy values are skipped. Processor-added keys (`source`, `type`, `priority`, `created_at`) are not part of it.
- Dimensions: `all-MiniLM-L6-v2` produces 384-dim vectors (`EMBEDDING_DIM` in `embedding.py`).

## 8. Sparse (BM25) vectors

Only when `SPARSE_SEARCH_ENABLED=true` (default `false`):

```python
for chunk in processed_chunks:
    indices, values = generate_sparse_vector(chunk.get("text", ""))
    sparse_vectors.append(SparseVector(indices=indices, values=values) if indices else None)
```

Failure is non-fatal: any exception is logged as a warning and `sparse_vectors = []`, so the batch is uploaded with dense vectors only. A chunk with empty text or no tokens gets `None` (no sparse vector). The vector is stored under `settings.SPARSE_VECTOR_NAME` (default `bm25`).

## 9. PointStruct construction

```python
payload = {
    "text": chunk["text"],
    "metadata": chunk_metadata,
    "source_id": source_id,
    "title": title if title else None,
    "summary": summary if summary else None,
    "tags": tags if tags else None
}
vectors_dict = self._create_point_vectors(text_embedding, field_embeddings, sparse_vec)
point = models.PointStruct(id=chunk_id, vector=vectors_dict, payload=payload)
```

`_create_point_vectors` always includes `text`, adds whichever of `title/summary/tags/metadata` exist, and adds the sparse vector when not `None`. Every dense vector passes through `validate_vector()` (embedding.py) which coerces to `list[float]` and raises `EmbeddingError` for wrong dimension or non-finite values. Chunks that are not dicts or lack `id/text/metadata` are logged and skipped.

## 10. Batch upsert

```python
def upload_to_qdrant(points, collection_name, batch_size=100):
    for i, batch in enumerate(batch_points(points, batch_size)):
        try:
            qdrant_client.upsert(collection_name=collection_name, points=batch)
            success_count += len(batch)
        except Exception as e:
            error_count += len(batch)
            continue
```

Batches of 100 are upserted sequentially. A failed batch is counted in `error_count` and skipped; the endpoint still returns HTTP 201 with `upload_failures > 0`. Callers must check that field. There is no rollback of batches that succeeded.

## 11. Translation hook

Every processor calls `process_chunk(chunk.page_content, chunk_id)` from `app/services/translation_service.py`. The function is currently a stub:

```python
def process_chunk(chunk_text: str, chunk_id: str) -> tuple:
    """Process a single chunk of text - translation disabled"""
    # Translation disabled - return original text as-is
    return chunk_text, False
```

So `metadata.is_hindi` is always `False` and text is never translated. `app/utils/language_utils.py` still defines `detect_language()` (Hindi character-set check) and `translate_text()` (AI4Bharat/ULCA API via `requests`, retry with exponential backoff, `TRANSLATION_API_URL`), but only `app/services/query_service.py` imports them; the ingestion path does not.

## Known issues / gotchas

- **Import side effects**: `translation_service.py` imports `SessionLocal` and `TranslationRecord` (database module) and, transitively, `language_utils.py`, which calls `logging.basicConfig(...)` with a `FileHandler('app.log')` at import time. Importing any processor therefore creates `app.log` in the working directory.
- **Duplicates on POST**: random chunk ids mean repeat uploads of the same `source_id` are not idempotent.
- **User metadata overrides processor metadata** (`type`, `source`, `priority`).
- **Partial failure is silent** (HTTP 201 with `upload_failures`).
- **Blocking calls in async code**: embedding, sparse encoding and Qdrant calls are synchronous inside `async def` methods and block the event loop during ingestion.
- **File size limit**: `MAX_FILE_SIZE_MB` is enforced inside processors (a violation raises `ValueError`, which `UploadService.process` converts into a 500, not a 413). It is not enforced for the URL path.
- **Settings `PDF_CHUNK_SIZE` / `PDF_CHUNK_OVERLAP`** exist in `config.py` but are unused; `PDFProcessor` uses `CHUNK_SIZE` / `CHUNK_OVERLAP`.
- `parse_tags` in `UploadService` is dead code (the endpoint parses tags).
- Title/summary/tags are not validated for type; a JSON tags list containing non-strings would break `", ".join(tags)`.

## Related pages

- [File processors](file_processors.md)
- [Document operations](document_operations.md)
- [Qdrant data model](../../architecture/qdrant_data_model.md)
- [API endpoints](../../api/endpoints.md)
- [Configuration](../../setup/configuration.md)
