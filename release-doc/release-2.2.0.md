# Release 2.2.0 — Ingestion Validation, Metadata Cleanup & Hybrid Score Fixes

**Service:** vectorization-service

> ⚠️ **Run the metadata cleanup script during this deploy.** Existing points still carry
> `title`/`summary`/`tags` duplicated inside `payload.metadata`; this release's code
> assumes they live only at the top level. See [Deployment](#deployment).

---

## Table of Contents

- [What's New](#whats-new)
- [New Configuration](#new-configuration)
- [Dependencies](#dependencies)
- [Deployment](#deployment)

---

## What's New

- **Ingestion is now validated up front** — `source_id`, `priority`, `title`/`summary`/`tags`,
  and metadata (`company`/`source_id`/`markdown_url`) are checked and normalized before any
  file parsing or embedding. Bad input now fails fast with a 400 instead of being silently
  dropped or overwritten.
- **No more silent partial uploads** — an empty file, or a batch that partially fails to
  write to Qdrant, now fails the request (500/502) instead of returning `201`. Any points
  already written for a failed request are rolled back automatically.
- **`title`/`summary`/`tags` live only at the payload top level now** — no longer duplicated
  inside `metadata`. Existing points need a one-time cleanup script run (see Deployment).
- **Search results flag keyword-injected matches** — `SearchResultItem.match_source` marks
  results injected by the title/summary keyword-match step, so their floor score isn't
  mistaken for a real semantic score.
- **Fixed a hybrid-search scoring bug** — a BM25 query that matched nothing was still
  capping dense scores at 70% of scale (the configured `0.7×dense` weight). Dense now gets
  full weight for that query when sparse contributes nothing.

---

## New Configuration

Add these to `.env` (defaults shown match `app/config.py`, so omitting them is safe too):

| Variable | Default | Description |
|---|---|---|
| `MAX_SOURCE_ID_LENGTH` | `255` | Max length of `source_id` on new uploads |
| `SOURCE_ID_PATTERN` | `^[A-Za-z0-9_\-.:]+$` | Allowed character set for `source_id` on new uploads (must fully match) |
| `PRIORITY_PATTERN` | `^P\d+$` | Required format for `priority` (upper-cased before matching) |

---

## Dependencies

No changes. Still requires `qdrant-client[fastembed]>=1.18.0,<2.0.0` and Qdrant server
1.18.2 (the version every environment runs).

---

## Deployment

### Pre-deploy checklist

1. **Confirm current callers' `priority` and `source_id` values are compliant** — check
   recent ingestion requests (or with upstream teams, e.g. commons-backend) for priorities
   that aren't `P<digits>` or source_ids with characters outside `[A-Za-z0-9_\-.:]`.
   Non-compliant callers will start getting 400s on new uploads.

2. **Back up the collection — take a Qdrant snapshot before doing anything else:**
   ```bash
   # Create a snapshot of the collection
   curl -X POST "http://<QDRANT_HOST>:<QDRANT_PORT>/collections/<COLLECTION_NAME>/snapshots"

   # List snapshots to get the snapshot name and confirm it was created
   curl "http://<QDRANT_HOST>:<QDRANT_PORT>/collections/<COLLECTION_NAME>/snapshots"

   # Download the snapshot locally for safekeeping
   curl -o backup-<COLLECTION_NAME>-$(date +%Y%m%d).snapshot \
     "http://<QDRANT_HOST>:<QDRANT_PORT>/collections/<COLLECTION_NAME>/snapshots/<snapshot_name>"
   ```

### Deploy steps

3. Pull the new code and add the new env keys to `.env` (see [New Configuration](#new-configuration)):
   ```dotenv
   MAX_SOURCE_ID_LENGTH=255
   SOURCE_ID_PATTERN='^[A-Za-z0-9_\-.:]+$'
   PRIORITY_PATTERN='^P\d+$'
   ```

4. Install dependencies (no version bumps this release, but keep this step for parity):
   ```bash
   pip install -r requirements.txt
   ```

5. **Run the metadata cleanup script** — removes the now-redundant `title`/`summary`/`tags`
   copies from `payload.metadata` on existing points (backfilling the top level first for
   any legacy point that only had the value there). Idempotent — safe to re-run, and safe
   to run before, during, or after the code deploy/restart; it only patches payloads via
   `set_payload` and does not touch vectors, so it can run against a live collection while
   the previous code version is still serving traffic.
   ```bash
   # Dry run first — reports scanned/cleaned/backfilled counts, writes nothing
   PYTHONPATH=. .vector-env/bin/python3 scripts/release_2.2.0/cleanup_metadata_duplicates.py --dry-run

   # Real run
   PYTHONPATH=. .vector-env/bin/python3 scripts/release_2.2.0/cleanup_metadata_duplicates.py
   ```

6. Restart the service.

7. **Smoke test:**
   - Upload a document with a valid `source_id`/`priority` → expect `201`.
   - Upload with a non-numeric priority (e.g. `priority=Pxyz`) → expect `400`.
   - Run a search with `include_scoring_debug: true` for a query where BM25 matches
     nothing and confirm `search_config.scoring_context.sparse_has_hits` is `false` and
     scores are no longer capped at 0.7.
   - Spot check a few pre-existing documents via `search` and confirm `title`/`summary`/`tags`
     still return correctly at the top level of each result.

### Rollback

Revert the code and restart. The cleanup script's changes are **not** reverted automatically
— `payload.metadata` will stay without the `title`/`summary`/`tags` duplicates, but the
previous code version reads those fields from the top level as well (this was already true
in the prior release), so rollback is safe without re-running any script. No vector or
collection schema changes were made in this release.

#### If a full data restore is needed

Code-only rollback (above) is sufficient for this release. If anything goes wrong beyond
that — the cleanup script corrupts or drops payload data — restore the collection from the
pre-deploy snapshot taken in [step 2](#pre-deploy-checklist), then revert the code and
re-run the smoke test.

---
