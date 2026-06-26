# Release 1.0.0 — Hybrid Search & Qdrant Client 1.18

**Service:** vectorization-service

> ⚠️ **Deploy code + client together.** Old code on new client or new code on old client will break.

---

## Table of Contents

- [What's New](#whats-new)
- [Search Architecture](#search-architecture)
- [Qdrant 1.18 Migration](#qdrant-118-migration)
- [Dependencies](#dependencies)
- [Deployment](#deployment)

---

## What's New

- **Hybrid search** — dense vectors + BM25 sparse (keyword), fused server-side with RRF. Enabled by setting `SPARSE_SEARCH_ENABLED=true` (new env key this release). When disabled, the service falls back to dense similarity search only (semantic/cosine matching across the 5 named vector fields).
- **BM25 / keyword search** — using `Qdrant/bm25` sparse model via fastembed.
- **Title matching** — exact, partial (prefix), and mid (infix/substring), each with a configurable score boost.
- **Summary matching** — summary field now participates in keyword matching and boosting (mirrors title behaviour).
- **Hybrid ranking** — two fusion methods now available, selected via `HYBRID_FUSION_METHOD`:
  - `weighted` *(default)* — dense and sparse scores are each min-max normalised independently, then combined as `0.7 × dense + 0.3 × sparse`. Scores are comparable to the plain `filter_score` threshold.
  - `rrf` — Reciprocal Rank Fusion: ranks the combined dense list against the sparse list (`1/(k + dense_rank) + 1/(k + sparse_rank)`, k=60), then min-max normalised. Rank-based fusion is more robust when the two retrievers produce very different score scales.
- **Ranking bugs fixed** — hybrid score was always 0.0 (RRF fusion score was ignored, now used directly); candidate limit could reach 1M (now capped at `min(top_k × 20, 10000)`); score cap at 1.0 ran before boosting (now applied once, after all boosts).
- **qdrant-client 1.18** compatibility across all call sites.

---

## Search Architecture

<details>
<summary>Collection structure</summary>

- **5 dense named vectors** (384-dim, cosine): `text`, `title`, `summary`, `tags`, `metadata`
- **1 sparse vector** `bm25` (IDF modifier) — active when `SPARSE_SEARCH_ENABLED=true`
- **Payload indexes** created at startup (idempotent):
  - `source_id`, `metadata.company`, `tags` — keyword
  - `metadata.DOCUMENT_TYPE` — text
  - `title`, `summary` — prefix-tokenized text (`min_token_len=2, max_token_len=20, lowercase`)

</details>

<details>
<summary>Hybrid search (dense + BM25 + RRF)</summary>

When `SPARSE_SEARCH_ENABLED=true`, one batch call is issued with:
- 5 dense prefetch queries (one per named vector field)
- 1 BM25 sparse prefetch query

Results are fused server-side using Reciprocal Rank Fusion (`FusionQuery(Fusion.RRF)`).

Falls back to dense-only if sparse encoding fails.

</details>

<details>
<summary>Dense-only search</summary>

When sparse search is disabled, 5 field queries run in a single batch and are merged with priority weights:

| Field | Weight |
|-------|--------|
| title | 0.36 |
| text | 0.27 |
| tags | 0.14 |
| summary | 0.14 |
| metadata | 0.09 |

</details>

<details>
<summary>Title & summary boosting</summary>

Applied after semantic ranking when `HYBRID_SEARCH_ENABLED=true` and `search_mode != "semantic"`:

- **Exact match** → multiply score by boost (title ×2.5, summary ×1.4), capped at 1.0
- **Partial/mid match** → multiply by lower boost (title ×1.5, summary ×1.2), capped at 1.0
- **Missing docs** (matched keyword but below semantic threshold) → injected at a floor score (0.15 × boost)

Title boost takes precedence — a doc matching both title and summary only gets the title boost.

</details>

<details>
<summary>Filtering</summary>

| API parameter | Qdrant field | Logic |
|---|---|---|
| `categories` | `tags` | OR within, AND between types |
| `organizations` | `metadata.company` | OR within, AND between types |
| `resource_type` | `metadata.DOCUMENT_TYPE` | substring match |
| `file_type` | `metadata.type` | OR within, AND between types |

Score filtering: use `filter_score` (single threshold) or `detail_filter_score` (per-field thresholds, OR logic).

</details>

---

## Qdrant 1.18 Migration

<details>
<summary>Breaking API changes fixed</summary>

| Removed / invalid in 1.18 | Fix applied |
|---|---|
| `client.search()` removed | `query_points(...).points` |
| `NearestQuery(nearest=NamedVector(...))` | `query=<vector>, using="<name>"` |
| `client.embed_sparse()` / `set_sparse_model()` | `fastembed.SparseTextEmbedding("Qdrant/bm25")` directly |
| `PointVectors` used in `upsert` | moved to `update_vectors()` |
| `FieldCondition(invert=True)` | replaced with `must_not` filter |
| `qdrant_client.http.models` (legacy path) | `qdrant_client.models` |

</details>

---

## Dependencies

- **`qdrant-client[fastembed]`** — `>=1.18.0,<2.0.0` (required; older clients are missing the APIs used here)
- **`fastembed`** — pulled in by the `[fastembed]` extra; used directly for BM25
- **Qdrant server** — `>=1.18` required
- **`Qdrant/bm25` model** — downloads on first sparse use; pre-cache in the image to avoid first-request latency

---

## Deployment

### Pre-deploy checklist

1. **Verify qdrant-client version:**
   ```bash
   pip show qdrant-client | grep Version
   # Must be >= 1.18.0
   ```

2. **Verify Qdrant server version:**
   ```bash
   curl -s http://<QDRANT_HOST>:<QDRANT_PORT>/collections | python3 -m json.tool | head
   # Or check the Qdrant dashboard — server must be >= 1.18
   ```

3. **Take a Qdrant snapshot (backup) before deploying:**
   ```bash
   # Create a snapshot of the collection
   curl -X POST "http://<QDRANT_HOST>:<QDRANT_PORT>/collections/<COLLECTION_NAME>/snapshots"
   
   # List snapshots to confirm it was created
   curl "http://<QDRANT_HOST>:<QDRANT_PORT>/collections/<COLLECTION_NAME>/snapshots"
   
   # Download the snapshot locally for safekeeping
   curl -o backup-<COLLECTION_NAME>-$(date +%Y%m%d).snapshot \
     "http://<QDRANT_HOST>:<QDRANT_PORT>/collections/<COLLECTION_NAME>/snapshots/<snapshot_name>"
   ```

### Deploy steps

4. Add the new env keys to `.env`:
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
   QDRANT_CHECK_COMPATIBILITY=false
   ```

5. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

6. Start the service 

7. **Back-fill BM25 for existing docs** (idempotent — only updates sparse vectors, does not re-embed dense):
   ```bash
   # Dry run first
   COLLECTION_NAME=documents1 SPARSE_SEARCH_ENABLED=true \
     python scripts/migrate_to_sparse_vectors.py --dry-run

   # Real run
   COLLECTION_NAME=documents1 SPARSE_SEARCH_ENABLED=true \
     python scripts/migrate_to_sparse_vectors.py
   ```
   > Pass `COLLECTION_NAME` explicitly — the migration script defaults to `documents`, the service uses `documents1`. New uploads get BM25 vectors automatically; only pre-existing docs need back-fill.

### Rollback

Revert the code, pin `qdrant-client[fastembed]>=1.9.0,<1.14.0`, set `SPARSE_SEARCH_ENABLED=false`, and restart. The sparse field and prefix indexes added by this release are inert when sparse is disabled and can be left in place.

---
