# Qdrant Data Model

Read this page to understand exactly what the service stores in Qdrant: the collections, the named dense vectors, the BM25 sparse vector, how point IDs are produced, the payload schema, the payload indexes (with tokenizer parameters), how documents and chunks map to points, how each API filter translates into a payload condition, and what happens when a collection was created before sparse search existed. Everything is derived from `app/core/clients/qdrant.py`, `app/services/document_operations/upload_service.py` and `app/services/prioritized_search_service.py`.

## Collections

| Collection | Setting | Default | Vector config | Used by |
|---|---|---|---|---|
| Documents | `COLLECTION_NAME` | `documents` | Five named dense vectors + optional sparse `bm25` | All ingestion, search, delete, metadata, similarity, verification |
| QA cache | `QA_CACHE_COLLECTION` | `qa_cache` | Single unnamed vector, size = embedding dimension, cosine | Created at startup only; no code writes to or reads from it |

Both collections are created by `ensure_collections_exist()` if absent. Vector size is `embedding_model.get_embedding_dimension()` (384 for the default `all-MiniLM-L6-v2`). Changing `EMBEDDING_MODEL` to a model with a different dimension does not alter an existing collection; you must recreate and re-ingest.

## Named dense vectors

Created in `ensure_collections_exist()` as `named_vectors_config`, each `models.VectorParams(size=<dim>, distance=models.Distance.COSINE)`:

| Vector name | Source text | Granularity |
|---|---|---|
| `text` | Chunk text | Per chunk (unique) |
| `title` | `title` form field | Per document (identical on every chunk) |
| `summary` | `summary` form field | Per document |
| `tags` | Tags joined as text | Per document |
| `metadata` | `" ".join(f"{k}: {v}")` over the request metadata (non-empty values) | Per document |

`UploadService._create_point_vectors` always writes `text` (after `validate_vector`), and writes `title`, `summary`, `tags`, `metadata` only when an embedding exists for that field in `field_embeddings`. Consequently a point may legitimately lack some named vectors if the corresponding input was empty; those fields then cannot match in the per-field dense query.

Note: dense search uses these names directly. `SEARCH_PRIORITY_ORDER = ["title", "text", "tags", "summary", "metadata"]` in `config.py` is the list of vector names queried, with weights from `SEARCH_PRIORITY_WEIGHTS` (`0.34, 0.26, 0.20, 0.12, 0.08`).

## Sparse vector (BM25)

Created only when `SPARSE_SEARCH_ENABLED=true`:

```python
create_kwargs["sparse_vectors_config"] = {
    settings.SPARSE_VECTOR_NAME: SparseVectorParams(
        modifier=Modifier.IDF
    )
}
```

- Name: `SPARSE_VECTOR_NAME` (default `bm25`).
- `Modifier.IDF`: Qdrant applies inverse document frequency server-side at query time. The stored values come from `fastembed` `Qdrant/bm25` (term-frequency component); IDF is not baked into stored vectors.
- Value per point: `SparseVector(indices, values)` generated from the chunk `text` by `generate_sparse_vector`. If the chunk produces no tokens, the point is stored without a sparse vector.
- It is a chunk-level vector only. Title, summary and tags have no sparse representation.

## Point ID scheme

Each chunk becomes one point. The ID is the chunk ID produced during extraction, `uuid.uuid4().hex` (a 32-character lowercase hex string without dashes). Qdrant accepts this form as a UUID. This is set in every processor (`pdf_processor`, `docx_processor`, `csv_processor`, ...) and in the URL branch of `UploadService`. In `_upload_chunks`:

```python
chunk_id = str(chunk["id"])
...
point = models.PointStruct(
    id=chunk_id,
    vector=vectors_dict,
    payload=payload
)
```

IDs are random per ingestion, not derived from `source_id`. Re-uploading the same document produces new point IDs; replacement is implemented by deleting all points with the `source_id` first (`UpdateService`), never by overwriting IDs. `source_id` is the only stable identity.

## Payload schema

Built in `UploadService._upload_chunks`:

```python
payload = {
    "text": chunk["text"],
    "metadata": chunk_metadata,
    "source_id": source_id,
    "title": title if title else None,
    "summary": summary if summary else None,
    "tags": tags if tags else None
}
```

| Field | Type | Meaning |
|---|---|---|
| `text` | string | Chunk content |
| `source_id` | string | Document identifier, shared by all chunks of one document |
| `title` | string or null | Document title, repeated on each chunk |
| `summary` | string or null | Document summary, repeated on each chunk |
| `tags` | list of strings or null | Tags, repeated on each chunk |
| `metadata` | object | Chunk metadata (below) |

`metadata` is assembled by `_prepare_chunk_metadata` from the processor-provided chunk metadata plus request metadata. Keys the code and filters depend on:

| Key | Origin | Used for |
|---|---|---|
| `company` | `company_id` form field (and request metadata) | Organization filter; also `company_id` scoping in delete/metadata operations |
| `type` | Set by the file processor: `pdf`, `docx`, `url_extracted`, `xlsx_rag_optimized`, `project_task` (CSV), or a text/markdown file type | `file_type` filter |
| `DOCUMENT_TYPE` | User-supplied metadata, comma-separated string | `resource_type` filter |
| `source` | Original filename | Informational |
| `priority` | `priority` form field (`P*`) | Informational |
| `created_at`, `updated_at` | Set in `_prepare_chunk_metadata` | Informational |
| `total_chunks`, `is_hindi`, others | Processor-specific | Informational |

Note: `metadata.type` values are processor labels, not raw extensions. A `file_type: ["xlsx"]` filter will not match XLSX documents whose stored type is `xlsx_rag_optimized`. Verify stored values before relying on this filter.

## Payload indexes

Defined in `_PAYLOAD_INDEXES` and applied by `_ensure_payload_indexes()` at startup:

| Field | Schema | Filter usage |
|---|---|---|
| `source_id` | KEYWORD | `MatchValue` (delete, count, scroll, dedup), `MatchAny` (boost injection) |
| `metadata.company` | KEYWORD | Organization `MatchAny` / `MatchValue` |
| `metadata.type` | KEYWORD | File type `MatchAny` |
| `tags` | KEYWORD | Category `MatchAny` |
| `metadata.DOCUMENT_TYPE` | TEXT (default tokenizer) | Resource type `MatchText` |
| `title` | TEXT, PREFIX tokenizer | Title boost scroll via `MatchText` |
| `summary` | TEXT, PREFIX tokenizer | Summary boost scroll via `MatchText` |

The prefix text index parameters:

```python
_PREFIX_TEXT_INDEX = models.TextIndexParams(
    type="text",
    tokenizer=models.TokenizerType.PREFIX,
    min_token_len=2,
    max_token_len=20,
    lowercase=True,
)
```

The PREFIX tokenizer indexes every prefix (length 2 to 20) of each token, lowercased, so partial-word queries match at index level. Infix matches ("sur" inside "insurance") are not indexed; the search service compensates in memory with `_supplement_matches_from_results`.

Index management behaviour:

- Existing schema is read with `get_collection(...).payload_schema`.
- KEYWORD/TEXT simple indexes: presence alone is a match; they are created if missing.
- For `TextIndexParams`, `_index_params_match` compares only the tokenizer. If the existing tokenizer differs from PREFIX, the index is deleted (`delete_payload_index`) and recreated once.
- Failures are logged as warnings and do not stop startup. A missing index does not break search; filters on a non-indexed field still work but are slower, and `MatchText` on a field without a text index returns no matches or an error depending on server behaviour.
- Do not add post-1.12 `TextIndexParams` fields (`phrase_matching`, `stopwords`, `stemmer`, `ascii_folding`, `enable_hnsw`) while the server is 1.12. See [Qdrant compatibility](../operations/qdrant_compatibility.md).

## Documents, chunks and points

```text
Document (source_id = "doc-1", title, summary, tags, metadata)
   |
   |  extract text -> split (3000/500, 3500/800 markdown, 1500/300 URL)
   v
Chunk 1  Chunk 2  ...  Chunk N          (each: uuid4 hex id)
   |        |            |
   v        v            v
Point 1   Point 2  ...  Point N
  vectors:  text  = embed(chunk text)         <- differs per point
            title/summary/tags/metadata       <- same on every point
            bm25  = BM25(chunk text)          <- differs per point (if enabled)
  payload:  text, source_id, title, summary, tags, metadata
```

Consequences:

- Title, summary, tags and metadata are duplicated on every chunk, in both payload and vectors. A document of N chunks stores N copies of each document-level embedding.
- Search ranks points (chunks); `_filter_best_per_source` then keeps the best point per `source_id`, so results are effectively one entry per document.
- Updating metadata with `MetadataService` patches payload with `set_payload` only. The `metadata` named vector is not recomputed and becomes stale.
- Deleting a document removes all points whose payload `source_id` matches.

## API filters to payload fields

Built in `PrioritizedSearchService._build_filters`. AND across filter types, OR within a type.

| Request field | Payload key | Qdrant condition | Index |
|---|---|---|---|
| `categories` | `tags` | `MatchAny(any=[...])` | KEYWORD |
| `organizations` | `metadata.company` | `MatchAny(any=[...])` | KEYWORD |
| `resource_type` | `metadata.DOCUMENT_TYPE` | `MatchText(text=rt)` per value; several values wrapped in `Filter(should=[...])` | TEXT |
| `file_type` | `metadata.type` | `MatchAny(any=[...])` | KEYWORD |
| `exclude_organizations` | `metadata.company` | `MatchAny` in `must_not` | KEYWORD |
| `exclude_file_type` | `metadata.type` | `MatchAny` in `must_not` | KEYWORD |
| `any_of` (list of blocks) | as above, per block | Each block built recursively, combined as `Filter(should=[...])` inside `must` | per field |

Values are stripped; blank values are dropped. If nothing remains, no filter is applied (returns `None`). Other services use their own filters: `BaseDocumentOperation.build_filter` uses `MatchValue` on `source_id` and optionally `metadata.company`.

## Collections that predate sparse vectors

- **Startup with `SPARSE_SEARCH_ENABLED=true`**: `_ensure_sparse_vector_field` calls `update_collection(sparse_vectors_config={bm25: SparseVectorParams(modifier=Modifier.IDF)})`. This adds the sparse field non-destructively; if the server reports the field "already" exists the error is swallowed, other server rejections are re-raised and abort startup.
- **Existing points have no `bm25` vector.** Adding the field does not backfill. Old points can never be returned by the sparse query; they are retrieved and scored only through the dense fields. Backfill requires `scripts/migrate_to_sparse_vectors.py`.
- **Mixed collections rank unevenly.** In hybrid mode, `_rank_results` reads the BM25 score as `fs.get(sparse_name) or 0.0`, so a point without a BM25 vector contributes a sparse score of 0 and can only earn the dense share of the fused score (0.7 weight by default). Newly ingested documents with the same dense relevance and a BM25 hit rank higher. See [Ranking and fusion](../services/search/ranking_and_fusion.md).
- **`SPARSE_SEARCH_ENABLED=false` (default)**: no sparse field is created, `_upload_chunks` skips BM25 generation, and search uses only the dense path. Switching it on later requires the steps above.
- **Collection missing the sparse field while sparse search is on**: if the field cannot be added (for example `qdrant-client<1.9.0` raises `ImportError`, logged as a warning), the sparse query would target a non-existent vector and the server would reject it. `_hybrid_batch_search` only falls back to dense-only on `ImportError` and `RuntimeError`, not on a Qdrant error, so a server 400 would surface as a failed request.
- Ingestion with sparse enabled but encoder failure: `_upload_chunks` logs a warning and stores dense vectors only (`sparse_vectors = []`).

## Known issues / gotchas

- `source_verification_service.py` filters on `metadata.source_id`, which is not a stored field (ingestion stores `source_id` at the payload root and indexes that); verification therefore reports every ID as not found.
- `_index_params_match` ignores `min_token_len`, `max_token_len` and `lowercase`; changing only those parameters will not trigger a rebuild.
- `metadata.type` filter values must be the processor labels (see table above), not file extensions.
- `UpdateService` does not forward `title`, `summary` or `tags`; verify in [Document operations](../services/ingestion/document_operations.md) before relying on updates preserving them.

## Related pages

- [System architecture](system_architecture.md)
- [Clients](../backend/clients.md)
- [Ingestion pipeline](../services/ingestion/pipeline.md)
- [Search overview](../services/search/overview.md)
- [Ranking and fusion](../services/search/ranking_and_fusion.md)
- [Qdrant compatibility](../operations/qdrant_compatibility.md)
- [Scripts](../operations/scripts.md)
