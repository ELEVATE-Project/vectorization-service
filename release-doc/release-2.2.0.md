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

**New implementation**

- **New: theme support, end to end** — upload/replace/upsert accept an optional `theme`
  form field (stored top-level, never embedded, so it can't move a search score). A new
  `PATCH /documents/{source_id}/theme` endpoint sets or changes the theme alone, without
  reprocessing the file. Search gets matching `themes` / `exclude_themes` filters, and
  `theme` is now returned on every search result.
- **Ingestion is now validated up front** — `source_id`, `priority`, `title`/`summary`/`tags`,
  and metadata (`company`/`source_id`/`markdown_url`) are checked and normalized before any
  file parsing or embedding. Bad input now fails fast with a 400 instead of being silently
  dropped or overwritten.
- **No more silent partial uploads** — an empty file, or a batch that partially fails to
  write to Qdrant, now fails the request (500/502) instead of returning `201`. Any points
  already written for a failed request are rolled back automatically.
- **Search results flag keyword-injected matches** — `SearchResultItem.match_source` marks
  results injected by the title/summary keyword-match step, so their floor score isn't
  mistaken for a real semantic score.
- **Stricter tag validation** — a blank or non-string entry anywhere in the `tags` array is
  now a 400 instead of being silently stored; duplicate tags are de-duplicated.
- **Postman collection added** — `postman/vectorization-service.postman_collection.json`
  covers every endpoint with a request per success/validation case and a saved example
  response. `README.md` documents how to import it and generate the OpenAPI file.

**New scripts**

- **Metadata cleanup script** — `scripts/release_2.2.0/cleanup_metadata_duplicates.py`
  removes the now-redundant `title`/`summary`/`tags`/`theme` copies from `payload.metadata`
  on existing points (backfilling the top level first for any legacy point that only had
  the value there). Idempotent — safe to re-run.
- **New: vector-coverage audit script** — `scripts/release_2.2.0/analyze_missing_vectors.py`
  is a read-only report of how many points are missing their `title`/`summary`/`tags` named
  vectors (documents uploaded without those fields never get them). With `--theme`, it also
  reports theme coverage per document/point. Talks to Qdrant directly; does not start the
  service or load the embedding model.

**Fixes on current**

- **`title`/`summary`/`tags` live only at the payload top level now** — no longer duplicated
  inside `metadata`. Existing points need a one-time cleanup script run (see Deployment).
- **Fixed a hybrid-search scoring bug** — a BM25 query that matched nothing was still
  capping dense scores at 70% of scale (the configured `0.7×dense` weight). Dense now gets
  full weight for that query when sparse contributes nothing.
- **Replace (`PUT`/upsert) is now safe against failed uploads** — previously the old
  version was deleted *before* the new one was uploaded, so a failed upload could lose the
  document entirely. It now uploads the new version first and only deletes the old chunks
  (by id) once that succeeds; a failed upload leaves the existing document untouched.
- **Replace no longer drops `title`/`summary`/`tags`/`theme`** — `PUT`/upsert previously
  forwarded only the file and metadata to the upload step, silently losing these fields on
  every update. They're now accepted as form fields and carried through like `POST`.
- **`PATCH /documents/{source_id}/metadata` no longer scrolls and read-modify-writes** —
  the old implementation fetched every chunk in batches of 100, merged client-side, and
  wrote each one back (not safe under concurrent updates). It's now a single server-side
  `set_payload` merge over a filter, independent of chunk count.

---

## New Configuration

Add these to `.env` (defaults shown match `app/config.py`, so omitting them is safe too):

| Variable | Default | Description |
|---|---|---|
| `MAX_SOURCE_ID_LENGTH` | `255` | Max length of `source_id` on new uploads |
| `SOURCE_ID_PATTERN` | `^[A-Za-z0-9_\-.:]+$` | Allowed character set for `source_id` on new uploads (must fully match) |
| `PRIORITY_PATTERN` | `^P\d+$` | Required format for `priority` (upper-cased before matching) |
| `UPLOAD_ROLLBACK_MAX_ATTEMPTS` | `3` | In-request rollback retries for a partial upload |
| `UPLOAD_ROLLBACK_RETRY_WAIT_SECONDS` | `0.5` | First in-request retry wait (doubles each attempt) |
| `UPLOAD_ROLLBACK_BACKGROUND_MAX_ATTEMPTS` | `9` | Background rollback retries after the request has already returned a 502 |
| `UPLOAD_ROLLBACK_BACKGROUND_FIRST_WAIT_SECONDS` | `2` | First background retry wait (doubles each attempt) |
| `UPLOAD_ROLLBACK_BACKGROUND_MAX_WAIT_SECONDS` | `60` | Cap on each background wait (~5 min of total retrying by default) |

> `QDRANT_CHECK_COMPATIBILITY` default also flipped from `false` to `true` — every
> environment now runs Qdrant server 1.18.2, which is within the client's allowed
> client/server version gap, so the compatibility check is safe to leave on.

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
   The `UPLOAD_ROLLBACK_*` retry/backoff variables are optional tuning knobs — their
   defaults already match the code, so omitting them is safe.

4. Install dependencies (no version bumps this release, but keep this step for parity):
   ```bash
   pip install -r requirements.txt
   ```

5. **Optional — audit current vector/theme coverage** before cleaning up, as a baseline to
   compare against afterwards. Read-only; talks to Qdrant directly and does not start the
   service.
   ```bash
   PYTHONPATH=. .vector-env/bin/python3 scripts/release_2.2.0/analyze_missing_vectors.py --theme
   ```

6. **Run the metadata cleanup script** — removes the now-redundant `title`/`summary`/`tags`/
   `theme` copies from `payload.metadata` on existing points (backfilling the top level first
   for any legacy point that only had the value there). Idempotent — safe to re-run, and safe
   to run before, during, or after the code deploy/restart; it only patches payloads via
   `set_payload` and does not touch vectors, so it can run against a live collection while
   the previous code version is still serving traffic.
   ```bash
   # Dry run first — reports scanned/cleaned/backfilled counts, writes nothing
   PYTHONPATH=. .vector-env/bin/python3 scripts/release_2.2.0/cleanup_metadata_duplicates.py --dry-run

   # Real run
   PYTHONPATH=. .vector-env/bin/python3 scripts/release_2.2.0/cleanup_metadata_duplicates.py
   ```

7. Restart the service. This also creates the new `theme` payload index automatically
   (`create_payload_index` is idempotent and runs on every startup) — no manual indexing
   step needed. It also runs on the first request if the process isn't restarted.

8. **Smoke test:**
   - Upload a document with a valid `source_id`/`priority` → expect `201`.
   - Upload with a non-numeric priority (e.g. `priority=Pxyz`) → expect `400`.
   - Run a search with `include_scoring_debug: true` for a query where BM25 matches
     nothing and confirm `search_config.scoring_context.sparse_has_hits` is `false` and
     scores are no longer capped at 0.7.
   - Spot check a few pre-existing documents via `search` and confirm `title`/`summary`/`tags`
     still return correctly at the top level of each result.
   - Upload with a `theme`, then `PATCH /documents/{source_id}/theme` to change it, and
     confirm `search` with a matching `themes` filter returns it and `exclude_themes` drops it.
   - Replace a document (`PUT`) with a `title`/`summary`/`tags`/`theme` and confirm they
     come back in the response and in a subsequent search (previously dropped on replace).

### Rollback

Revert the code and restart. The cleanup script's changes are **not** reverted automatically
— `payload.metadata` will stay without the `title`/`summary`/`tags`/`theme` duplicates, but
the previous code version reads `title`/`summary`/`tags` from the top level as well (this was
already true in the prior release), so rollback is safe without re-running any script. The
previous code version does not know about `theme` at all, so it simply ignores that payload
key — searches just stop supporting theme filtering until the code is rolled forward again.
No vector or collection schema changes were made in this release (the new `theme` payload
index is additive and harmless to leave in place).

#### If a full data restore is needed

Code-only rollback (above) is sufficient for this release. If anything goes wrong beyond
that — the cleanup script corrupts or drops payload data — restore the collection from the
pre-deploy snapshot taken in [step 2](#pre-deploy-checklist), then revert the code and
re-run the smoke test.

---
