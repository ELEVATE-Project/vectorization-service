# Release 3.0 — Qdrant 1.18 Upgrade & Search Overhaul

**Service:** vectorization-service

Upgrades `qdrant-client` to **1.18**, completes the **hybrid search** stack (dense + BM25
keyword, fused with RRF), adds **partial/mid matching + boosting** for `title` and
`summary`, and fixes three ranking bugs. This document covers the full search
implementation as it stands after the release.

> ⚠️ **Deploy code + client together.** The old code used `client.search()`,
> `NearestQuery(nearest=NamedVector(...))`, and `client.embed_sparse()` — all removed/invalid
> in qdrant-client 1.18. Old-code-on-new-client and new-code-on-old-client both break.

---

## New Features & Enhancements

- **Hybrid search** (dense + BM25 sparse, server-side RRF) — default mode.
- **Keyword/BM25 search** via `fastembed.SparseTextEmbedding` (`Qdrant/bm25`).
- **Title matching**: exact, partial (prefix), mid (infix/substring) — each boosted.
- **Summary matching** (new): summary now participates in keyword matching + boosting;
  exposed as `summary_match` (mirrors `title_match`).
- **Ranking fixes**: correct hybrid scoring, sane query limits, single score cap.
- **qdrant-client 1.18** compatibility across every call site.

---

## Search Architecture (full implementation)

### Collection & vectors (`core/clients/qdrant.py`)
- **5 dense named vectors**, all 384-dim cosine (`all-MiniLM-L6-v2`):
  `text`, `title`, `summary`, `tags`, `metadata`.
- **1 sparse vector** `bm25` (IDF modifier) — added when `SPARSE_SEARCH_ENABLED=true`.
- **Payload indexes** (created/reconciled on startup, idempotent):
  `source_id` (keyword), `metadata.company` (keyword), `tags` (keyword),
  `metadata.DOCUMENT_TYPE` (text), **`title` and `summary` (prefix-tokenizer text)**.
  Prefix tokenizer (`min_token_len=2, max_token_len=20, lowercase`) lets partial/prefix
  queries match server-side. A changed index is rebuilt once, then skipped on later starts.

### Indexing path (`document_operations/upload_service.py`)
- Generates dense embeddings for `text` + each field, and (when enabled) a **BM25 sparse
  vector per chunk** from the chunk text, stored under `bm25`.
- One Qdrant point per chunk; payload carries `text, title, summary, tags, source_id, metadata`.

### Query preprocessing (`utils/query_preprocessor.py`)
- spaCy stop-word removal/normalization, **skipped for short queries**
  (`< SHORT_QUERY_THRESHOLD` words) so abbreviations/short titles aren't mangled.
- Falls back to the raw query if preprocessing yields empty text.

### Search execution (`services/prioritized_search_service.py`)
- **`search_mode`** on the request: `hybrid` (default, with boosts) or `semantic` (vectors only).
- Two retrieval paths, selected by `SPARSE_SEARCH_ENABLED`:
  - **Hybrid** — `_hybrid_batch_search`: one `Prefetch` per dense named vector + one BM25
    sparse `Prefetch`, fused on the server with `FusionQuery(Fusion.RRF)` in a single
    `query_points` call. Falls back to dense-only if sparse encoding/import fails.
  - **Dense-only** — `_parallel_batch_search`: per-field search via `query_batch_points`,
    combined with `SEARCH_PRIORITY_WEIGHTS` (title 0.36, text 0.27, tags 0.14, summary 0.14,
    metadata 0.09).
- Named vectors use the 1.18 idiom `query=<vector>, using="<name>"`.
- **Ranking** (`_rank_results`): hybrid uses the RRF fusion score directly; dense uses the
  weighted sum. **Dedup** keeps the best chunk per `source_id`.
- **Title/summary boosting** (`hybrid` mode, `HYBRID_SEARCH_ENABLED=true`): matched docs are
  multiplied by the boost (exact > partial), capped at 1.0, then re-sorted; matched docs the
  vector search missed are injected at a floor score so they still surface. Mid/infix matches
  are caught by an in-memory substring pass over the candidate pool (no extra round-trips).
- **Filtering** (`_build_filters`): `categories→tags`, `organizations→metadata.company`,
  `resource_type→metadata.DOCUMENT_TYPE`, `file_type→metadata.type`. **AND between filter
  types, OR within a type.** Two threshold modes: `filter_score` (weighted threshold) or
  `detail_filter_score` (per-field thresholds, OR logic). Empty query → browse/scroll unique
  sources.

### Endpoints (`api/v1/endpoints/documents.py`)
`POST /documents` (create), `PUT /documents/{id}` (replace), `PUT /documents/{id}/upsert`,
`PATCH /documents/{id}/metadata`, `DELETE /documents/{id}`,
`POST /documents/search` (multi-field hybrid), `POST /documents/text-search` (single-vector),
`POST /documents/check-similarity`, `POST /documents/verify-sources`,
plus `POST /query/` (multilingual query + cache).

---

## qdrant-client 1.18 Migration

| Removed / invalid in 1.18 | Where | Fix |
|---|---|---|
| `client.search()` removed | `text_embedding_search_service.py`, `similarity_service.py`, `query_service.py` (×2) | `query_points(...).points` |
| `NearestQuery(nearest=NamedVector(...))` invalid | `prioritized_search_service.py` (dense + hybrid) and the services above | `query=<vector>, using="<name>"` |
| `client.embed_sparse()` / `set_sparse_model()` removed | `core/clients/sparse_encoder.py` | `fastembed.SparseTextEmbedding("Qdrant/bm25")` directly |
| `PointVectors` belongs to `update_vectors`, not `upsert` | `scripts/migrate_to_sparse_vectors.py` | `client.update_vectors(...)` |
| `FieldCondition(invert=True)` invalid | `similarity_service.py` | exclusion via filter `must_not` |
| `qdrant_client.http.models` (legacy) | `qdrant.py`, `source_verification_service.py` | `qdrant_client.models` |

`client.scroll()` is retained in 1.18 (no change). `query_points`/`query_batch_points`
return a response object — read `.points`.

### Ranking bugs fixed
- **Hybrid score was always 0.0** — ranking ignored the RRF fusion score; now used directly.
- **Query limit explosion** — `top_k * 100000` (up to 1M; server caps at 10k) → `min(top_k * 20, 10000)`.
- **Premature cap** — `min(score, 1.0)` ran before boosting; now applied once, after boosting.

---

## Configuration & Environment

**New env vars:**

| Variable | Default | Purpose |
|----------|---------|---------|
| `EXACT_SUMMARY_BOOST` | `1.4` | Multiplier for exact summary match. |
| `PARTIAL_SUMMARY_BOOST` | `1.2` | Multiplier for partial/mid summary match. |

**Changed value:**

| Variable | New value | Note |
|----------|-----------|------|
| `SPARSE_SEARCH_ENABLED` | `true` | Enables BM25 + hybrid. Requires BM25 back-fill for existing docs (see Migration). |

**Relevant existing vars (unchanged):** `HYBRID_SEARCH_ENABLED=true` (boost master switch),
`SPARSE_VECTOR_NAME=bm25`, `EXACT_TITLE_BOOST=2.5`, `PARTIAL_TITLE_BOOST=1.5`, `RRF_K=60`,
`SHORT_QUERY_THRESHOLD=3`, `EMBEDDING_MODEL=all-MiniLM-L6-v2`,
`COLLECTION_NAME` (service `.env` uses `documents1`). No env vars deprecated/removed.

```dotenv
HYBRID_SEARCH_ENABLED=true
SPARSE_SEARCH_ENABLED=true
SPARSE_VECTOR_NAME=bm25
RRF_K=60
EXACT_TITLE_BOOST=2.5
PARTIAL_TITLE_BOOST=1.5
EXACT_SUMMARY_BOOST=1.4
PARTIAL_SUMMARY_BOOST=1.2
SHORT_QUERY_THRESHOLD=3
```

**Automatic on startup** (`ensure_collections_exist`): adds the `bm25` sparse field
(non-destructive), rebuilds `title` as prefix-tokenized (once), creates the `summary` prefix
index.

---

## Dependencies

- `qdrant-client[fastembed]` → **`>=1.18.0,<2.0.0`** (`requirements.txt`).
- `fastembed` used directly for BM25 (pulled by the `[fastembed]` extra).
- Qdrant **server** ≥ 1.10 (validated on **1.18.2**).
- `Qdrant/bm25` model downloads on first sparse use — pre-cache in the image to avoid
  first-request latency.

---

## Deployment & Migration

1. Deploy new code + client together; `pip install -r requirements.txt`.
2. Set `SPARSE_SEARCH_ENABLED=true` (+ summary boosts if customizing).
3. Start the service — confirm logs: `Sparse vector field 'bm25' added/verified`,
   `Payload index created: title`, `Payload index created: summary`.
4. **Back-fill BM25 for existing docs** (idempotent, non-destructive — uses `update_vectors`,
   does not re-embed dense vectors):

   ```bash
   # dry run first
   COLLECTION_NAME=documents1 SPARSE_SEARCH_ENABLED=true \
     python scripts/migrate_to_sparse_vectors.py --dry-run
   # real run
   COLLECTION_NAME=documents1 SPARSE_SEARCH_ENABLED=true \
     python scripts/migrate_to_sparse_vectors.py --batch-size 100
   ```

   > Pass `COLLECTION_NAME` explicitly — the script defaults to `documents`, the service
   > uses `documents1`. New uploads get sparse vectors automatically; only pre-existing
   > docs need the back-fill.

**Rollback:** revert code **and** pin `qdrant-client[fastembed]>=1.9.0,<1.14.0`, set
`SPARSE_SEARCH_ENABLED=false`, restart. The added sparse field/indexes are inert when
disabled and can be left in place.

---

## Developer Notes

- **Named-vector queries (1.18):** always `query=<vector>, using="<name>"`. The old
  `NearestQuery(nearest=NamedVector(...))` raises a validation error.
- **BM25 encoding** lives in `core/clients/sparse_encoder.py` (`generate_sparse_vector`) —
  `fastembed.SparseTextEmbedding`, not the removed `client.embed_sparse`.
- **Adding a boosted text field** (like title/summary): add a prefix TEXT index in
  `qdrant.py`, then use the generic helpers in `prioritized_search_service.py`
  (`_get_field_match_sources`, `_supplement_matches_from_results`, `_apply_field_boost`,
  `_fetch_field_match_docs`) and surface a `<field>_match` in `_build_result_items`.
- **Backward compatibility:** request schema unchanged; response adds optional
  `summary_match` (`"exact" | "partial" | null`); infix matches report as `partial`. The
  `_get_title_match_sources` / `_apply_title_boost` / `_fetch_title_match_docs` wrappers are
  retained. Collection changes are non-destructive (dense vectors/payloads untouched).
- **Until back-fill completes**, hybrid still works but BM25 adds no signal for old docs
  (dense ranking unaffected).

---

## Validation

- Unit tests: `pytest tests/unit/ -v` — 78/78 passing.
- E2E (live Qdrant 1.18.2): sparse field + prefix indexes created; hybrid returns non-zero
  scores; title exact/partial/mid and summary matches labelled; text-search, similarity
  (with `exclude_source_id`), and verify-sources all work.

Post-deploy smoke checks:
- `GET /api/health` → healthy.
- `POST /api/v1/documents/search` (default hybrid) → results with non-zero scores; log shows
  `Starting hybrid batch search (dense + BM25 sparse, RRF fusion)`.
- Exact-term query returns the doc via BM25; title prefix query sets `title_match`;
  summary-only phrase sets `summary_match`.

> Note: `tests/integration/` currently needs PostgreSQL and still mocks the removed
> `search_batch` API — pre-existing tech debt, unrelated to this release.
