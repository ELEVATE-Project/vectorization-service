# Migration, Restore and Re-indexing

**Purpose.** Read this page before changing the shape of the `documents` collection, moving data to another Qdrant server, or re-embedding content. It separates three kinds of statement and labels them explicitly:

- **Repo behaviour**: verified in code (`app/core/clients/qdrant.py`, `scripts/migrate_to_sparse_vectors.py`, the document operation services).
- **Qdrant feature, not implemented in this repo**: general Qdrant guidance. This repository contains **no export, snapshot, backup or restore tooling**. Nothing in `app/`, `scripts/` or `deployment/` calls the snapshot API or copies a collection between servers (the blue-green mode of the migration script copies within one server only).
- **Unverified**: stated as such.

Flag-level reference for the migration script lives in [Scripts](scripts.md); client/server version rules live in [Qdrant compatibility](qdrant_compatibility.md). This page does not repeat them.

## 1. Which procedure do you need

| Situation | Procedure | Section |
|---|---|---|
| Collection exists, points have no `bm25` vector, hybrid search being enabled | Sparse backfill with `migrate_to_sparse_vectors.py` | 2 |
| Collection lacks the `bm25` field entirely | Startup attempt via `update_collection`; fall back to blue-green | 2, 3 |
| A payload index is missing or the title/summary tokenizer changed | Automatic at service startup | 3 |
| Copy data to another server or environment (QA to prod, DR) | Snapshot, or scroll-and-upsert | 4 |
| `EMBEDDING_MODEL` changed | Full re-ingest into a new collection | 5.1 |
| One document is wrong or stale | `PUT` flow | 5.2 |
| Chunk size settings changed | Re-ingest affected documents | 5.3 |
| A migration went wrong | Rollback | 6 |

## 2. Sparse-vector backfill (`scripts/migrate_to_sparse_vectors.py`)

Repo behaviour. Needed when points were ingested while `SPARSE_SEARCH_ENABLED=false`. Adding the sparse field to a collection does not backfill existing points (see [Qdrant data model](../architecture/qdrant_data_model.md)); without a `bm25` vector a point scores 0 on the sparse side and ranks below comparable new documents in hybrid mode.

### 2.1 Invocation and flags

The script reads only process environment variables (`QDRANT_HOST`, `QDRANT_PORT`, `COLLECTION_NAME`, `SPARSE_VECTOR_NAME`, `QDRANT_CHECK_COMPATIBILITY`). It does not load `.env`; export values in the shell. Run from the repository root with `PYTHONPATH=.`.

```bash
# in-place
PYTHONPATH=. COLLECTION_NAME=documents .venv/bin/python3 scripts/migrate_to_sparse_vectors.py --dry-run
PYTHONPATH=. COLLECTION_NAME=documents .venv/bin/python3 scripts/migrate_to_sparse_vectors.py

# blue-green
PYTHONPATH=. COLLECTION_NAME=documents QDRANT_HOST=<host> QDRANT_PORT=6333 \
  .venv/bin/python3 scripts/migrate_to_sparse_vectors.py --new-collection documents_v2
```

| Flag | Default | Applies to |
|---|---|---|
| `--new-collection NAME` | none (in-place) | Selects blue-green. Must match `^[A-Za-z0-9_-]+$` or exit code 2. |
| `--dry-run` | off | Both |
| `--batch-size N` | 100 | Points per `update_vectors` call |
| `--scroll-limit N` | 500 | Points per scroll page |
| `--skip-copy` | off | Blue-green: do not re-copy |
| `--skip-bm25` | off | Blue-green: skip encoding |
| `--env-file PATH` | `.env` | Blue-green: file whose `COLLECTION_NAME=` line is rewritten |

### 2.2 In-place mode, step by step (`run_inplace`)

Precondition: the collection already declares the sparse field. The script does not create it.

1. `client.count(collection)` for progress.
2. Scroll all points (`with_payload=["text"]`, `with_vectors=[sparse_name]`), `--scroll-limit` per page.
3. For each point, `_encode_and_queue` decides:
   - existing non-empty sparse vector: `skipped`;
   - no `payload["text"]`: `skipped`;
   - encoder exception: `error`;
   - encoder returns no indices: `no_tokens`, nothing written;
   - otherwise queue `PointVectors(id, vector={sparse_name: SparseVector(indices, values)})`.
4. Flush through `client.update_vectors` whenever the queue reaches `--batch-size`. Only the sparse vector is written; dense vectors and payload are untouched and never recomputed.
5. Print `In-place migration complete ... scanned, migrated, skipped, no_tokens, errors`. Exit status 1 if `errors > 0`.

### 2.3 Blue-green mode, step by step (`run_bluegreen`)

| Step | Action |
|---|---|
| Pre | Count source; read `get_collection(old).config.params.vectors` so the target mirrors the source dense schema (dimension included). |
| 1 | Create target with that dense config plus `sparse_vectors_config={sparse_name: SparseVectorParams(modifier=Modifier.IDF)}`. Skipped if the target already exists. |
| 2 | Scroll source with payload and all vectors; `upsert` into target. A failed batch is logged and skipped, not fatal. |
| 3 | Scroll target and write BM25 vectors as in in-place mode. Aborts with exit 1 if the error rate exceeds 10%. |
| 4 | `_verify_migration`: target count must equal source count; share of text-bearing points lacking a sparse vector must be at most 10%. |
| 5 | On success, rewrite `COLLECTION_NAME=` in `--env-file` and print a restart/delete-old-collection reminder. On failure, exit 1 and leave `.env` alone. |

The source collection is never modified or deleted.

### 2.4 Dry run, idempotency, resumability

- `--dry-run` scans and logs what would be written; no `update_vectors`, no collection creation, and in blue-green mode steps 4 and 5 are skipped entirely.
- Idempotent: points that already carry a non-empty sparse vector are skipped, so a re-run only processes what is missing. Re-running after errors is the documented recovery.
- Resumable in the sense of "re-run to finish". There is **no checkpoint or offset file**: every run rescans from the first point. Points classed `no_tokens` have no stored vector and are re-encoded on every run.
- In blue-green, `--skip-copy` re-runs only the encoding step against the existing target. `--skip-copy` does not pick up writes made to the source after the first copy; freeze ingestion (upload, update, delete) during the migration.
- The in-place scroll loop breaks on a scroll exception and can still exit 0 if no encoding errors were counted. Always compare `scanned` with the point count.

### 2.5 Verifying the result

After the script (repo behaviour for the script's own checks; the curl calls are plain Qdrant REST):

1. Script summary: `errors=0`, and `scanned` equals the collection point count. `no_tokens` greater than 0 is expected for table-separator chunks.
2. Count and config:

   ```bash
   curl -s http://<host>:6333/collections/<name> | python3 -m json.tool
   ```

   Check `points_count`, `config.params.sparse_vectors` contains `bm25` with `modifier: idf`.
3. Search smoke test with `include_scoring_debug: true` and confirm `keyword_score` is non-null for keyword-matching documents (see [Ranking and fusion](../services/search/ranking_and_fusion.md)).
4. Restart the service against the final `COLLECTION_NAME` so payload indexes are created (section 3).

## 3. Adding a sparse field or payload index to an existing collection

### 3.1 What `ensure_collections_exist` does at startup (repo behaviour)

Called from the app lifespan (see [App lifecycle](../backend/app_lifecycle.md)). In order:

1. Lists collections. If `COLLECTION_NAME` is absent, creates it with five named dense vectors (`text`, `title`, `summary`, `tags`, `metadata`; size `embedding_model.get_embedding_dimension()`, cosine). If `SPARSE_SEARCH_ENABLED` is true, `sparse_vectors_config` with `Modifier.IDF` is included.
2. If the collection exists and `SPARSE_SEARCH_ENABLED` is true, `_ensure_sparse_vector_field` calls `update_collection(sparse_vectors_config=...)`. An `UnexpectedResponse` whose body contains "already" is swallowed; any other rejection is logged and re-raised, which aborts startup.
3. Creates `QA_CACHE_COLLECTION` if absent (single vector).
4. `_ensure_payload_indexes(COLLECTION_NAME)` reads `payload_schema` once, then for each entry of `_PAYLOAD_INDEXES`:
   - index missing: `create_payload_index`;
   - index present and `_index_params_match` is false: `delete_payload_index`, then create;
   - each field is wrapped in its own `try/except`; failures are logged as warnings and startup continues.

Indexes managed: keyword on `source_id`, `metadata.company`, `metadata.type`, `tags`; text on `metadata.DOCUMENT_TYPE`; prefix-tokenizer text (`min_token_len=2`, `max_token_len=20`, `lowercase=True`) on `title` and `summary`.

Nothing here is a migration framework: there are no versioned migrations and nothing records what has been applied. Existing points are never rewritten by startup.

### 3.2 What `_index_params_match` checks, and its limits

```python
if isinstance(desired_schema, models.TextIndexParams):
    existing_params = getattr(existing_schema, "params", None)
    existing_tokenizer = getattr(existing_params, "tokenizer", None)
    return existing_tokenizer == desired_schema.tokenizer
return True
```

- For `TextIndexParams` (title, summary) only the **tokenizer** is compared. Changing `min_token_len`, `max_token_len` or `lowercase` in `_PREFIX_TEXT_INDEX` will not rebuild an existing index.
- For simple schemas (keyword, plain text) any existing index counts as a match. If `metadata.DOCUMENT_TYPE` was previously created with a different schema, it is not corrected.
- A field already indexed under a different type is therefore never converted; delete the index manually (`DELETE /collections/<name>/index/<field>`, Qdrant REST) and restart.
- Because per-field errors are only warnings, a missing index shows up as slow filters or missing boosts, not as a startup failure. Check logs for `Could not ensure payload index for`.
- Do not add post-1.12 `TextIndexParams` fields (`phrase_matching`, `stopwords`, `stemmer`, `ascii_folding`, `enable_hnsw`) while any server is on 1.12; see [Qdrant compatibility](qdrant_compatibility.md).

### 3.3 Adding the sparse field

Use the startup path first: set `SPARSE_SEARCH_ENABLED=true` and restart. Whether the server accepts `update_collection` with a new sparse field is not decided by this repo, and the migration script docstring states the opposite (blue-green was written because a server returned `Not existing vector name error: bm25`). If startup logs `Server rejected sparse-vector field update`, use blue-green (section 2.3). Then backfill (section 2.2).

## 4. Moving data between Qdrant servers or environments

None of the following is implemented in this repo. Use Qdrant's documented APIs. Because client 1.18 talks to server 1.12 here, prefer plain REST (`curl`) for snapshot operations rather than assuming the Python client's snapshot helpers behave identically across that gap (unverified).

### 4.1 Approach A: collection snapshot (Qdrant feature, not implemented in this repo)

Best for a full, exact copy including vectors, payload and index definitions.

```bash
# 1. create on source
curl -s -X POST http://<src>:6333/collections/<name>/snapshots
# 2. list, then download
curl -s http://<src>:6333/collections/<name>/snapshots
curl -o <name>.snapshot http://<src>:6333/collections/<name>/snapshots/<snapshot-file>
# 3. upload and recover on destination (creates or replaces the collection)
curl -X POST "http://<dst>:6333/collections/<name>/snapshots/upload?priority=snapshot" \
  -H 'Content-Type: multipart/form-data' -F 'snapshot=@<name>.snapshot'
```

Constraints:

- Snapshot restore requires the destination server to be the **same or a newer version** than the server that produced the snapshot. A 1.12 snapshot can be restored on 1.18; the reverse is not supported.
- A collection snapshot is per node; on a single-node deployment it is complete. Clustered deployments need per-node handling (not used here as far as the repo shows).
- The restored collection keeps the source name unless restored under another name through the recover-from-URL/location endpoint; confirm `COLLECTION_NAME` matches.
- Snapshot files are stored on the Qdrant host's disk; ensure free space before creating one.

### 4.2 Approach B: scroll with vectors and upsert (Qdrant feature; the same pattern exists in-repo for same-server copies)

Use when versions are incompatible, or you want to transform data in flight. The blue-green step 2 of the migration script is a same-server instance of this pattern, but it hardcodes the source and target to one `QdrantClient`; it cannot copy across servers without code changes.

Outline:

1. On the destination, create the collection with the same dense config and, if hybrid search is used, the sparse config (same as `ensure_collections_exist`), then start the service once so payload indexes are created, or create them yourself.
2. Scroll the source with `with_payload=True, with_vectors=True`, page by page using the returned offset.
3. `upsert` each page to the destination as `PointStruct(id, payload, vector)`. Point IDs are random hex strings (`uuid.uuid4().hex`) and are preserved by this copy.
4. Compare counts and spot check.

Note: upsert is idempotent per ID, so an interrupted copy can be repeated.

### 4.3 Pre-flight and post-flight checklist

Before:

- [ ] Source and destination server versions recorded (`GET /` returns `version`).
- [ ] Source `GET /collections/<name>` saved (vector names, sizes, distance, sparse config, `payload_schema`).
- [ ] Ingestion frozen (upload, update, delete) or an accepted delta window agreed.
- [ ] Destination has enough disk; embedding dimension matches the destination service's `EMBEDDING_MODEL`.

After:

- [ ] `points_count` equal on both (`GET /collections/<name>`, or `POST /collections/<name>/points/count` with `{"exact": true}`).
- [ ] Named vector config identical: `text`, `title`, `summary`, `tags`, `metadata` all size 384 cosine; `bm25` present with `modifier: idf` if sparse search is on.
- [ ] `payload_schema` lists all seven indexed fields with the expected types; if any is missing, start the service against the destination so `_ensure_payload_indexes` creates it.
- [ ] A known-good sample query through `POST /api/documents/search` returns the same top results as on the source (a few queries, with `include_scoring_debug: true`).
- [ ] `POST /api/documents/verify-sources` is not usable as a check today (see section 7).
- [ ] Point the service at the destination, restart, then check logs for `Could not ensure payload index for` and `Server rejected sparse-vector field update`.

## 5. Re-embedding and re-indexing

### 5.1 Changing `EMBEDDING_MODEL` (repo behaviour plus Qdrant constraint)

- Vector size is fixed when a collection is created. Collection creation uses `embedding_model.get_embedding_dimension()` (384 for the default `all-MiniLM-L6-v2`). A model with a different dimension cannot write into the existing collection: upserts will be rejected by Qdrant.
- Even a same-dimension model change makes old and new vectors incomparable. Old points must be re-embedded.
- Startup does not detect a mismatch: `ensure_collections_exist` only creates collections that are absent and never compares configured size with an existing collection's size.
- There is no re-embed script. The text of each chunk is in the payload, but title, summary, tags and metadata embeddings and chunk boundaries come from ingestion; the supported route is a full re-ingest of the original files into a **new collection**:
  1. Pick a new collection name, set `COLLECTION_NAME` and the new `EMBEDDING_MODEL` on a separate instance (or after a freeze), start the service so the collection and indexes are created.
  2. Re-upload every document through `POST /api/documents` with its original `source_id`, `priority`, `company_id`, `title`, `summary`, `tags` and `metadata`. The original files are not stored by the service; they must come from your source of record.
  3. Verify as in section 4.3, then switch the production `COLLECTION_NAME` and restart.
- Unverified: whether a pure payload-text re-embed (reading `payload["text"]` and recomputing only the `text` vector) gives acceptable results; it would not recompute field vectors and is not recommended.

### 5.2 Re-indexing one document

`PUT /api/documents/{source_id}` (`UpdateService.update`) checks existence (404 if none), counts, deletes all points for the `source_id` (and `company_id` if provided), then calls `UploadService.process` with the new file. `PUT .../upsert` does the same but creates when absent. Caveats, all verified in code:

- New point IDs are generated; nothing is overwritten in place.
- Delete happens before upload. If the upload fails after the delete, the document is gone; check `upload_failures` and `points_uploaded` in the response and re-run.
- `UpdateService` does not forward `title`, `summary` or `tags` to the upload, so they are dropped on update; re-supply them by deleting and posting the document again through `POST /api/documents` when they must be kept.
- The upload endpoint can return success with `upload_failures > 0` (failed batches are logged and skipped).
- Not atomic: searches during the window may miss the document.

Details: [Document operations](../services/ingestion/document_operations.md).

### 5.3 Changing chunk size

`CHUNK_SIZE`, `CHUNK_OVERLAP`, `MARKDOWN_CHUNK_SIZE`, `MARKDOWN_CHUNK_OVERLAP`, `PDF_CHUNK_SIZE`, `PDF_CHUNK_OVERLAP` and the `URL_EXTRACTION_*` values in `app/config.py` only apply at ingestion time. Existing points keep their old chunking. To apply new sizes, re-ingest each affected document (section 5.2 per document, or a full re-ingest as in 5.1). Mixed chunk sizes in one collection are valid but make scores less uniform.

## 6. Rollback guidance

| Scenario | Rollback |
|---|---|
| In-place sparse backfill misbehaves | There is no automatic undo. Dense vectors and payload are unchanged; sparse vectors added are harmless to dense-only search. Set `SPARSE_SEARCH_ENABLED=false` and restart to stop querying them. Restoring the pre-migration snapshot is the only way to remove them (Qdrant feature). |
| Blue-green not yet switched | Nothing to roll back; the source is untouched. Delete the target with `DELETE /collections/<new>` if abandoning. |
| Blue-green already switched | Set `COLLECTION_NAME` back to the old name in the deployed configuration and restart. This works only while the old collection still exists, so do not run the suggested `DELETE` on the old collection until the rollback window ends. |
| Collection deleted or corrupted | Recover from a snapshot (section 4.1) or re-ingest. |
| Payload index change | Re-run startup; indexes are rebuilt only when the tokenizer differs. Otherwise delete the index by REST and restart. |

Where the service runs from the Ansible/Vault deployment, `COLLECTION_NAME` must be changed in the Vault secret `vectorization-service`: the script's `.env` rewrite is overwritten by the next deploy because the playbook regenerates `.env` from Vault (see [Deployment](../setup/deployment.md)). Recommended: take a snapshot of the live collection before any migration.

## Known issues / gotchas

- The script ignores `.env` and does not read `SPARSE_SEARCH_ENABLED`, although its docstring example sets it.
- `_ensure_sparse_vector_field` (startup `update_collection`) and the script docstring ("a sparse field cannot be added to an existing collection") contradict each other; behaviour depends on the server.
- `ensure_collections_exist` never verifies that an existing collection's vector size equals the model dimension.
- `_index_params_match` compares only the tokenizer for text indexes and accepts any existing simple index.
- The migration script does not create payload indexes on the new collection; the service does on next start.
- No checkpointing: a re-run rescans everything; `no_tokens` points are re-encoded each time.
- Blue-green copy and verification tolerate a failed upsert batch until the count check; the 10% BM25 missing-rate threshold means up to 10% of points can lack sparse vectors and still pass.
- The ansible playbook has a step named "Run Django migrations" that only exports the environment; there are no database migrations. Collection changes are never applied by deployment, only by service startup or the script.
- `verify-sources` filters on `metadata.source_id`, which is not stored, so it cannot be used for post-migration verification.
- Snapshot, download and restore are Qdrant capabilities documented here for convenience; the repo has no code, tests or runbook that exercises them.

## Related pages

- [Scripts](scripts.md)
- [Qdrant compatibility](qdrant_compatibility.md)
- [Troubleshooting](troubleshooting.md)
- [Deployment](../setup/deployment.md)
- [Qdrant data model](../architecture/qdrant_data_model.md)
- [Document operations](../services/ingestion/document_operations.md)
- [Ingestion pipeline](../services/ingestion/pipeline.md)
- [Ranking and fusion](../services/search/ranking_and_fusion.md)
- [App lifecycle](../backend/app_lifecycle.md)
