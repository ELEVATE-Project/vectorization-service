# Release 2.3.0 — API Access Token for Selected Routes and Safe URL Fetching

## Table of Contents

- [Release 2.3.0 — API Access Token for Selected Routes and Safe URL Fetching](#release-230--api-access-token-for-selected-routes-and-safe-url-fetching)
  - [Table of Contents](#table-of-contents)
  - [What's New](#whats-new)
  - [New Configuration](#new-configuration)
  - [Dependencies](#dependencies)
  - [Deployment](#deployment)
  - [Known limits](#known-limits)
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

**URL fetching (`metadata.markdown_url`)**

When an upload, replace or upsert sends `metadata.markdown_url`, the service fetches that URL and embeds its text. These checks stop the URL from being used to read internal systems:

- **Allowed hosts.** The URL's host must be in `ALLOWED_URL_HOSTS`, or the request gets `400` before anything is fetched. The host is read with `urlparse`, so `http://allowed.org@169.254.169.254/` is checked as `169.254.169.254`. The default `*` allows every host, so behaviour is the same as 2.2.0 until the list is set.
- **No redirects.** The service no longer follows redirects. A redirect answer gets `400` with the target address, so the caller can send the final URL. Before, an allowed host could redirect the service to an internal address.
- **Size limit.** The page is read in parts and stops with `413` once it passes `MAX_FILE_SIZE_MB` (the same limit as file uploads).
- `http://` and `https://` URLs are both still accepted; other schemes get `400` as before.

---

## New Configuration

| Variable | Default | Description |
|---|---|---|
| `INTERNAL_TOKEN_ENFORCED` | `false` | `true`: the document write routes, cache clear and the acronym endpoints need the `internal_access_token` header; search routes stay open. Keep `false` until every caller sends the header on write calls. |
| `ALLOWED_URL_HOSTS` | `*` | Hosts `metadata.markdown_url` may point to, comma-separated, exact match (any case). Host only: no scheme, port or path. `*` allows every host. Empty allows no host, so every URL upload gets `400`. Example: `static.example.org`. |

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

5. **URL hosts.** With the default `ALLOWED_URL_HOSTS=*`, no `.env` change is needed. To restrict URLs, set the host the callers' markdown files are served from and restart the service:
   ```dotenv
   ALLOWED_URL_HOSTS=static.example.org
   ```
   Smoke test (with the list set):
   ```bash
   # Host not in the list -> expect 400 "metadata.markdown_url host 'localhost' is not allowed ..."
   curl -s -w " HTTP %{http_code}\n" -X POST "http://<HOST>:<PORT>/api/documents" \
     -H "internal_access_token: <INTERNAL_API_TOKEN>" \
     -F "file=@any.txt" -F "source_id=url-check-1" -F "priority=P1" \
     -F 'metadata={"markdown_url":"http://localhost:6333/collections"}'
   ```
   Then upload one document from each caller with a `markdown_url` on the allowed host and confirm it is indexed. A caller that uploads in a background job can show "uploaded" even when this service refused the document, so check the document itself (for example with `/api/documents/verify-sources`), not only the caller's message.

---

## Known limits

- **URL host names are not resolved to IP addresses.** With `*`, or with a listed host whose DNS points to an internal IP, the service still fetches that address. A check that refuses private, loopback and link-local IPs (including `169.254.169.254`) after the DNS lookup is not part of this release.
- `http://` URLs are allowed, so their content can be read or changed on the way.
- **A wrong host list fails quietly for background callers.** If `ALLOWED_URL_HOSTS` leaves out the host a caller's markdown files are served from, every upload with a `markdown_url` from that caller gets `400` and is not indexed. A caller that uploads in a background job may still show the upload as successful.

---

## Rollback

**Fastest — flag off:** set `INTERNAL_TOKEN_ENFORCED=false` and restart. Routes behave as in 2.2.0. No data or schema changes were made in this release.

**URL hosts:** set `ALLOWED_URL_HOSTS=*` and restart to allow every host again. Redirects and the size limit are in code; to remove them, revert the code.
