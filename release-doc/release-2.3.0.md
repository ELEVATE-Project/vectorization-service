# Release 2.3.0 — API Access Token for Selected Routes

## Table of Contents

- [Release 2.3.0 — API Access Token for Selected Routes](#release-230--api-access-token-for-selected-routes)
  - [Table of Contents](#table-of-contents)
  - [What's New](#whats-new)
  - [New Configuration](#new-configuration)
  - [Dependencies](#dependencies)
  - [Deployment](#deployment)
  - [Rollback](#rollback)

---

## What's New

**API access**

- **Token check for selected routes.** With `INTERNAL_TOKEN_ENFORCED=true`, these routes need the `internal_access_token` header (same value as `INTERNAL_API_TOKEN`); a missing or wrong token gets `401`:
  - `POST /api/documents` (upload), `PUT /api/documents/{source_id}` (replace), `PUT /api/documents/{source_id}/upsert`
  - `PATCH /api/documents/{source_id}/metadata`, `PATCH /api/documents/{source_id}/theme`
  - `DELETE /api/documents/{source_id}`, `DELETE /api/cache/redis`
  - `GET /api/acronyms`, `POST /api/acronyms/bulkUpload`
- **Open routes.** Search and read routes stay open: `/api/documents/search`, `/api/documents/text-search`, `/api/documents/check-similarity`, `/api/documents/verify-sources`, `/api/query/` and `/api/health`.
- The check is set on each protected route (`dependencies=[Depends(verify_internal_token_if_enforced)]`); a new route that needs protection must add it too.
- **Off by default.** With `INTERNAL_TOKEN_ENFORCED=false` (the default), every route is open, so current callers keep working until they send the header.
- **Change from 2.2.0:** the acronym endpoints always needed the token in 2.2.0. They now need it only when `INTERNAL_TOKEN_ENFORCED=true`, so with the default they are open, including `POST /api/acronyms/bulkUpload`.

---

## New Configuration

| Variable | Default | Description |
|---|---|---|
| `INTERNAL_TOKEN_ENFORCED` | `false` | `true`: the document write routes, cache clear and the acronym endpoints need the `internal_access_token` header; search routes stay open. Keep `false` until every caller sends the header on write calls. |

`INTERNAL_API_TOKEN` (from 2.2.0) is the token value for this check. It is still required: the service does not start without it.

---

## Dependencies

No new dependencies.

---

## Deployment

1. Deploy the code with the flag off (the default):
   ```dotenv
   INTERNAL_TOKEN_ENFORCED=false
   ```
   Restart the service. Behaviour is the same as 2.2.0.

2. Change every caller to send the header on each write call (upload, update, upsert, metadata, theme, delete, cache clear):
   ```
   internal_access_token: <INTERNAL_API_TOKEN>
   ```

3. Turn the check on and restart the service:
   ```dotenv
   INTERNAL_TOKEN_ENFORCED=true
   ```

4. **Smoke test** (the delete uses an id that does not exist, so no data changes):
   ```bash
   # Write route, no token -> expect 401
   curl -s -o /dev/null -w "%{http_code}\n" -X DELETE "http://<HOST>:<PORT>/api/documents/smoke-test-no-such-id"

   # Write route, with token -> expect 404 (passes the token check; the id does not exist)
   curl -s -o /dev/null -w "%{http_code}\n" -X DELETE "http://<HOST>:<PORT>/api/documents/smoke-test-no-such-id" \
     -H "internal_access_token: <INTERNAL_API_TOKEN>"

   # Search, no token -> expect 200
   curl -s -o /dev/null -w "%{http_code}\n" -X POST "http://<HOST>:<PORT>/api/documents/search" \
     -H 'Content-Type: application/json' -d '{"query":"test","top_k":1}'
   ```
   Also run one upload from each caller and confirm it is indexed (not `401`).

---

## Rollback

**Fastest — flag off:** set `INTERNAL_TOKEN_ENFORCED=false` and restart. Routes behave as in 2.2.0. No data or schema changes were made in this release.
