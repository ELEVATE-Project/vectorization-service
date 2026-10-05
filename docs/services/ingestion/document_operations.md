# Document Operations

Read this page to understand the exact semantics of the document lifecycle endpoints other than the initial upload: full update (`PUT`), upsert, delete and metadata patch, together with the shared `BaseDocumentOperation` helpers. Several of these operations have non-obvious data-loss or consistency behaviour; the gotchas are listed explicitly. Code: `app/services/document_operations/` and the endpoints in `app/api/v1/endpoints/documents.py`. For upload see [Ingestion pipeline](pipeline.md).

## 1. Endpoint map

| Endpoint | Service method | Input style |
|---|---|---|
| `PUT /api/v1/documents/{source_id}` | `UpdateService.update` | multipart: `file`, `priority` (`P1`), `metadata` (JSON **string**), `company_id` |
| `PUT /api/v1/documents/{source_id}/upsert` | `UpdateService.upsert` | same as above |
| `PATCH /api/v1/documents/{source_id}/metadata` | `MetadataService.update_metadata` | form: `metadata_updates` (JSON string, required), `company_id` |
| `DELETE /api/v1/documents/{source_id}` | `DeleteService.delete` | `company_id` read from a **form field** |

All are routed through `DocumentProcessor` (`app/services/document_processor.py`), a facade that instantiates the four services and forwards arguments unchanged. `DocumentProcessor.delete_documents` wraps the arguments in a `DeleteRequest`.

## 2. `BaseDocumentOperation`

`base_operation.py`. All services inherit it.

| Helper | Behaviour |
|---|---|
| `ensure_collections()` | Awaits `ensure_collections_exist()` (creates `documents` / `qa_cache` and payload indexes if missing). |
| `build_filter(source_id, company_id=None)` | `Filter(must=[source_id == X])`, plus `metadata.company == company_id` when given. Both are `MatchValue`. |
| `check_documents_exist(...)` | `qdrant_client.scroll(limit=1)` with the filter; returns `True/False`. **Any exception is logged and returns `False`.** |
| `count_documents(...)` | `qdrant_client.count(count_filter=...)`; **any exception returns `0`**. |
| `parse_metadata(str)` | JSON string to dict; invalid JSON or non-object is logged and returns `{}` (silent). Not used by the services shown here. |
| `validate_source_id` | 400 if empty or whitespace. |
| `validate_priority` | 400 unless it starts with `P`. |

`source_id` is a plain string with no normalisation (no trimming, no case folding); it is matched exactly against the keyword-indexed top-level payload field `source_id`.

Note: when `company_id` is omitted the filter matches **all companies** for that `source_id`. Since `source_id` is the only uniqueness key, two companies using the same `source_id` will affect each other unless `company_id` is passed on every operation.

## 3. Update (`PUT /documents/{source_id}`)

`UpdateService.update` is delete-then-upload:

```text
validate_source_id, validate_priority
metadata_dict = _parse_metadata(metadata)          # invalid JSON -> {} silently
if not check_documents_exist(source_id, company_id): 404 "... Use upload endpoint for new documents."
existing_count = count_documents(...)
delete_result  = delete_service.delete(source_id, company_id)
upload_result  = upload_service.process(file, priority, metadata_dict, source_id, company_id)
```

The response contains `operation: "update"`, `documents_deleted`, `previous_document_count`, `chunks_processed`, `points_uploaded`, `upload_failures`, `file_type`, `priority`, `sample_chunk`.

Semantics and gotchas:

- **Title, summary and tags are lost.** `upload_service.process` is called without `title`, `summary` or `tags`; the PUT endpoint does not even accept them. After an update the payload has `title/summary/tags = None`, and the `title`, `summary`, `tags` named vectors do not exist on the new points (only `metadata` and `text`, plus `bm25` if enabled). Unless these are re-supplied inside the `metadata` JSON (which only affects `payload.metadata` and the metadata vector, not the top-level `title`/`summary`/`tags` that search and boosts read), the document loses title/summary/tag search and boost behaviour. Previous metadata is not carried over either; the caller must resend everything.
- **Not atomic.** Existing points are deleted before the new file is processed. If extraction, embedding or upload then fails (unsupported type, OCR failure, Qdrant error), the old document is already gone. Partially failed batches (`upload_failures`) leave the document incomplete.
- **Metadata errors are silent.** Invalid JSON in `metadata` is replaced by `{}` here, unlike `POST /documents`, which returns 400.
- **`markdown_url`** inside metadata works the same as for upload; the `file` part is still required.
- `created_at` is reset to the update time (new points).
- The update returns 404 if `check_documents_exist` returns `False`, including when Qdrant raised an error (see helper note).

## 4. Upsert (`PUT /documents/{source_id}/upsert`)

`UpdateService.upsert` is identical except that a missing document is not an error:

```python
existing_docs = self.check_documents_exist(source_id, company_id)
if existing_docs:
    existing_count = self.count_documents(source_id, company_id)
    delete_result = await self.delete_service.delete(source_id, company_id)
    documents_deleted = delete_result["documents_deleted"]
upload_result = await self.upload_service.process(file, priority, metadata_dict, source_id, company_id)
operation = "updated" if existing_docs else "created"
```

Same gotchas as update (no title/summary/tags, not atomic, silent metadata parse). Note: to create a document with title/summary/tags use `POST /documents`; re-ingestion through PUT or upsert drops them. A transient Qdrant failure in `check_documents_exist` returns `False`, so upsert would skip the delete and upload new chunks next to the old ones (duplicates).

## 5. Delete (`DELETE /documents/{source_id}`)

`DeleteService.delete`:

```python
while True:
    search_response = qdrant_client.scroll(
        collection_name=settings.COLLECTION_NAME,
        scroll_filter=scroll_filter,
        limit=batch_size,          # 100
    )
    if not search_response[0]:
        break
    point_ids = [point.id for point in search_response[0]]
    qdrant_client.delete(collection_name=settings.COLLECTION_NAME,
                         points_selector=models.PointIdsList(points=point_ids))
    total_deleted += len(point_ids)
    if len(point_ids) < batch_size:
        break
```

Semantics:

- Deletes by scroll-then-delete in batches of 100 without a page offset: because the matched points are removed, each scroll naturally returns the next set. The loop ends on an empty page or a short page.
- Returns 404 (`No documents found for source ID: ...`) when nothing was deleted. Used internally by `update`/`upsert`, where an unexpected 404 would abort the operation (only reachable if the data vanished between the existence check and the delete).
- `company_id` is read via `Form(default=None)` on a `DELETE` request, so clients must send a form-encoded body; a query parameter `?company_id=` is ignored and the delete then affects every company with that `source_id`.
- The short-page early exit could skip points if a scroll page were returned short while more matches remain (concurrent modification); a trailing empty page is the normal terminator.
- Response: `status`, `message`, `documents_deleted`, `source_id`, `company_id`.

## 6. Metadata patch (`PATCH /documents/{source_id}/metadata`)

`MetadataService.update_metadata` patches payloads in place; no re-chunking and no re-embedding.

Flow:

1. `metadata_updates` is parsed in the endpoint (`json.loads`; 400 `Invalid metadata JSON`). Non-object JSON is not rejected by the endpoint; the service only checks it is non-empty (`400 metadata_updates cannot be empty`).
2. Scroll with `build_filter(source_id, company_id)`, `limit=100`, `with_payload=True`, `with_vectors=False`.
3. For each point `_update_point_metadata` does:

```python
existing_metadata = point.payload.get("metadata", {})
updated_metadata = existing_metadata.copy(); updated_metadata.update(metadata_updates)
updated_metadata["updated_at"] = datetime.now().isoformat()
if company_id and 'company' in metadata_updates and metadata_updates['company'] != company_id:
    raise HTTPException(400, "Cannot change company_id through metadata update")
updated_payload = point.payload.copy()
updated_payload["metadata"] = updated_metadata
qdrant_client.set_payload(collection_name=..., payload=updated_payload, points=[point.id])
```

4. 404 if nothing was updated. Response: `documents_updated`, `metadata_updates`.

Semantics and gotchas:

- **Shallow merge** at the top level of `payload.metadata`: a key in the patch replaces the whole value (nested dicts are not merged). Keys cannot be deleted (no null-removal semantics; `null` is stored).
- **Stale metadata vector.** The `metadata` named vector was computed at ingestion from the request metadata. Patching changes the payload only, so metadata-field search scores reflect the old values.
- **Top-level `title`/`summary`/`tags` are not touched.** Patching `{"title": "X"}` updates `metadata.title` only; the top-level `title` payload field and the `title` vector (which search and title boost use) keep the old value.
- **`company` guard is partial.** Changing `company` is only blocked when `company_id` is supplied and differs. Without `company_id`, `company` can be reassigned freely. `source_id` is not protected: a patch containing `source_id` changes `metadata.source_id` (the searched/deleted field is the top-level `source_id`, so the two diverge).
- **Possible infinite loop for documents with 100 or more chunks.** The scroll does not use an offset and patched points still match the filter. When a page returns exactly 100 points the loop does not break and scrolls the same first 100 points again, forever (`total_updated` keeps growing). Only documents with fewer than 100 chunks terminate. Fix by passing the `next_page_offset` returned by `scroll`.
- Each point is written by its own `set_payload` call (N round trips). `set_payload` writes the full payload (including `text`) rather than only `metadata`.
- The operation is idempotent for the small-document case and not transactional.

## Known issues / gotchas (summary)

| Area | Issue |
|---|---|
| `UpdateService.update/upsert` | Do not forward `title`, `summary`, `tags`; they are lost on update. |
| `UpdateService` | Delete happens before upload; failure after delete loses data. |
| `UpdateService` | Invalid metadata JSON silently becomes `{}` (POST returns 400). |
| `BaseDocumentOperation.check_documents_exist` / `count_documents` | Swallow Qdrant errors (`False` / `0`), which can turn upsert into duplicate creation and update into 404. |
| `MetadataService.update_metadata` | Pagination without offset; loops forever for 100 or more matching points. |
| `MetadataService` | Stale `metadata` vector; top-level `title/summary/tags` and their vectors not updated; `source_id` not protected; `company` only guarded when `company_id` given. |
| `DeleteService` | `company_id` must be a form field; omitted means all companies. Early exit on short page. |
| All | `source_id` uniqueness is not enforced across companies; omit `company_id` and operations span all matches. |

The update/title loss, metadata-service `source_id` and delete-batch observations are confirmed by the code. The infinite-loop behaviour of the metadata patch is not listed there.

## Related pages

- [Ingestion pipeline](pipeline.md)
- [File processors](file_processors.md)
- [API endpoints](../../api/endpoints.md)
- [Qdrant data model](../../architecture/qdrant_data_model.md)
- [Troubleshooting](../../operations/troubleshooting.md)
