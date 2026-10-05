# Qdrant Client / Server Compatibility

**Purpose.** Read this page before upgrading `qdrant-client`, upgrading or downgrading a Qdrant server, or using any Qdrant API feature that the service does not use today. It records the deployed version matrix, why the client-version warning is suppressed, the exact Qdrant operations the code uses (verified against the source), and the features that must not be introduced while a server is on 1.12.

Primary sources: `issueOnCompactabilityWithServerVersion1.12.md`, and `app/core/clients/qdrant.py`.

## 1. Version matrix

| Environment | Qdrant server | `qdrant-client` |
|---|---|---|
| QA | **1.12** | 1.18 |
| Production | **1.18** | 1.18 |
| Local (`start_mac.sh`) | whatever `qdrant/qdrant:latest` resolves to when the container was first created | 1.18 |

`requirements.txt` pins `qdrant-client[fastembed]>=1.18.0,<2.0.0`. The floor is required because the code uses `query_points` / `query_batch_points` (the older `search` and `search_batch` were removed from the client in 1.14-1.16), `qdrant_client.models` (not `qdrant_client.http.models`), and `fastembed.SparseTextEmbedding("Qdrant/bm25")` directly (the client's `embed_sparse` / `set_sparse_model` helpers were removed). The ceiling `<2.0.0` is there because a major version may change APIs; retest before lifting it.

Client 1.18 is used against server 1.12 because the server in QA cannot be upgraded, while the client must be at 1.18 for BM25 sparse search support.

## 2. The suppressed warning

Qdrant's client checks the server version on construction and warns when the major versions differ or the minor difference exceeds 1. With 1.18 against 1.12 it emits, on every process start:

```text
qdrant_remote.py:282: UserWarning: Qdrant client version 1.18.0 is incompatible
with server version 1.12.0. Major versions should match and minor version
difference must not exceed 1.
```

The code suppresses it through a setting rather than a warnings filter:

```python
qdrant_client = QdrantClient(
    settings.QDRANT_HOST,
    port=settings.QDRANT_PORT,
    check_compatibility=settings.QDRANT_CHECK_COMPATIBILITY,
)
```

`QDRANT_CHECK_COMPATIBILITY` defaults to `false` (`app/config.py`) and `.env.sample`. The two utility scripts (`scripts/migrate_to_sparse_vectors.py`, `rename_key_entities_field.py`) pass the same flag to their own `QdrantClient`; the migration script reads it from the process environment only.

Operational rule: leave it `false` while any environment is on 1.12. Set `true` only after the server reaches 1.18 (then the client check becomes a useful guard again). The 1.0.0 release notes' env block sets it to `true`; on QA that reintroduces the warning (see [Deployment](../setup/deployment.md)).

Note: with `check_compatibility=False` the client performs no version probe, so there is no early failure if the server is unreachable; the first real call (`get_collections` in lifespan startup) fails instead.

## 3. Operations the service uses (verified in code)

Every call below exists in the source and is reported working on server 1.12 by the compatibility notes.

| Operation | Where | Notes |
|---|---|---|
| `get_collections` | `qdrant.py::ensure_collections_exist`, `main.py` health check | |
| `get_collection` | `qdrant.py::_ensure_payload_indexes`, migration script | Reads `payload_schema` and vector config. |
| `create_collection` (named dense `VectorParams` x5, optional `sparse_vectors_config` with `Modifier.IDF`) | `qdrant.py` | Dense vectors: `text`, `title`, `summary`, `tags`, `metadata`, cosine, size from `embedding_model.get_embedding_dimension()`. `qa_cache` uses a single unnamed vector. |
| `update_collection(sparse_vectors_config=...)` | `qdrant.py::_ensure_sparse_vector_field` | Only when the collection exists and `SPARSE_SEARCH_ENABLED=true`. See section 6. |
| `create_payload_index` / `delete_payload_index` | `qdrant.py::_ensure_payload_indexes` | KEYWORD, TEXT and `TextIndexParams(type="text", tokenizer=PREFIX, min_token_len=2, max_token_len=20, lowercase=True)`. |
| `upsert` | `qdrant.upload_to_qdrant` (batches of 100), migration script | Named dense plus `SparseVector`. |
| `query_points` | `similarity_service.py`, `text_embedding_search_service.py` | Single dense query. |
| `query_batch_points` | `prioritized_search_service.py` (two call sites: dense-only batch and hybrid batch) | 5 dense requests, plus one BM25 sparse request in hybrid mode. |
| `scroll` (with `MatchText`, `MatchAny`, `MatchValue` filters) | search boosts, `base_operation`, `delete_service`, `metadata_service`, `source_verification_service`, scripts | |
| `retrieve` | `prioritized_search_service.py` late payload retrieval | |
| `count` | `base_operation.py`, migration script | |
| `delete` (points) | `delete_service.py` | |
| `set_payload` | `metadata_service.py`, `rename_key_entities_field.py` | |
| `update_vectors` | `migrate_to_sparse_vectors.py` only | Not on the request path. |

Filters used in `_build_filters`: `MatchAny` on `tags`, `metadata.company`, `metadata.type`; `MatchText` on `metadata.DOCUMENT_TYPE`. Payload indexes created at startup (`_PAYLOAD_INDEXES`): keyword on `source_id`, `metadata.company`, `metadata.type`, `tags`; text on `metadata.DOCUMENT_TYPE`; prefix-tokenized text on `title` and `summary`.

Index failures are non-fatal by design: `_ensure_payload_indexes` wraps each index in `try/except` and logs `Could not ensure payload index for '<field>': ...`.

## 4. Features that must not be introduced while a server is on 1.12

A grep of `app/` and `scripts/` finds none of these in use (only comments naming them). Each returns HTTP 400 with `did not match any variant of untagged enum` when sent to a server that predates it.

| Feature | Needs server | Breaks |
|---|---|---|
| `MatchPhrase` / `phrase_matching` text conditions | 1.15+ | search and scroll paths |
| `FormulaQuery` (score boosting) | 1.14+ | search path |
| `TextIndexParams` fields `phrase_matching`, `stopwords`, `stemmer`, `ascii_folding`, `enable_hnsw` | 1.13+ | `create_payload_index` (for example if `_PREFIX_TEXT_INDEX` in `qdrant.py` were extended) |

Also not used today, and therefore unverified on 1.12: server-side fusion (`FusionQuery`, `Prefetch`). The service deliberately fuses client-side in `_rank_results()` (see [Ranking and fusion](../services/search/ranking_and_fusion.md)).

Checklist before using any new Qdrant API:

1. Look up the server version that introduced it.
2. If it is above 1.12, it must stay behind a feature flag and be tested only against a 1.18 server.
3. Re-run the read path against QA (`query_batch_points` with five dense plus one sparse request, `scroll` with `MatchText` and `MatchAny`, `retrieve`).
4. Update this page.

## 5. Verification and the missing regression test

No live compatibility test exists in `tests/`; only the `compat` and `requires_qdrant` markers are registered in `tests/conftest.py`, and no test uses them (see [Testing](testing.md)). Until it is written, verify manually:

```bash
curl -s http://<host>:6333/ | python3 -m json.tool       # "version": "1.12.x" or "1.18.x"
curl -s http://<host>:6333/collections/<COLLECTION_NAME> | python3 -m json.tool
curl -s -X POST http://localhost:8000/api/documents/search \
  -H 'Content-Type: application/json' -d '{"query": "classroom observation"}'
```

The statement in the notes that the write path was confirmed by "9,916 existing points carrying named-dense + bm25 sparse vectors" refers to the QA data set and cannot be re-checked from the repository.

## 6. Sparse vector field on existing collections

Two parts of the repository disagree about adding the sparse field to a collection that already exists:

- `qdrant.py::_ensure_sparse_vector_field` (called at startup when `SPARSE_SEARCH_ENABLED=true` and the collection exists) calls `update_collection(sparse_vectors_config={SPARSE_VECTOR_NAME: SparseVectorParams(modifier=Modifier.IDF)})`, treating the response containing "already" as benign and re-raising any other `UnexpectedResponse`.
- `scripts/migrate_to_sparse_vectors.py` documents that "Qdrant does NOT allow adding a vector field to an existing collection", citing `Not existing vector name error: bm25` on 1.18.2, and provides the blue-green mode as the workaround.

Whether `update_collection` accepts a new sparse field depends on the server. If startup logs `Server rejected sparse-vector field update for '<collection>': ...` the server refused it; use the blue-green migration (see [Scripts](scripts.md)) and point `COLLECTION_NAME` at the new collection.

## 7. Upgrading a server from 1.12 to 1.18 (from the compatibility notes)

Before:

1. Snapshot: `POST /collections/<COLLECTION_NAME>/snapshots`, list, download.
2. Save the collection config: `GET /collections/<COLLECTION_NAME>`.

Direct-install upgrade (adjust paths and architecture): stop the `qdrant` systemd unit, download the 1.18.x binary, replace `/usr/local/bin/qdrant`, start the unit. The storage directory is unchanged. The notes state storage is forward compatible across minor versions without re-indexing; take the snapshot regardless.

After:

1. `GET /` shows `"version": "1.18.x"`.
2. `GET /collections/<COLLECTION_NAME>` shows the same `points_count`.
3. Set `QDRANT_CHECK_COMPATIBILITY=true`, restart, and confirm no `UserWarning` at startup.
4. Smoke test `GET /api/health` (see the health caveat in [Troubleshooting](troubleshooting.md)) and a `POST /api/documents/search` returning non-zero scores.

After the server is on 1.18, features in section 4 become available and can be evaluated (phrase matching, `FormulaQuery`, richer text index params), but production has been on 1.18 while QA is on 1.12, so anything adopted must still work on QA until QA is upgraded.

## Known issues / gotchas

- The 1.0.0 release notes say server `>=1.18` is required, which contradicts QA running 1.12; the QA notes and code comments are authoritative.
- `.env.sample` and `qdrant.py` comments mention `qdrant-client>=1.9.0`; the enforced floor is 1.18.0.
- The sparse-field-on-existing-collection behaviour is documented inconsistently (section 6).

## Related pages

- [Scripts](scripts.md)
- [Testing](testing.md)
- [Troubleshooting](troubleshooting.md)
- [Deployment](../setup/deployment.md)
- [Configuration reference](../setup/configuration.md)
- [Qdrant data model](../architecture/qdrant_data_model.md)
- [Clients](../backend/clients.md)
