# Release 2.3.0 — API Access Token and Safe URL Fetching

## Table of Contents

- [Release 2.3.0 — API Access Token and Safe URL Fetching](#release-230--api-access-token-and-safe-url-fetching)
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

- **Token check for every `/api` route.** With `INTERNAL_TOKEN_ENFORCED=true`, every route under `/api` except `/api/health` needs the `internal_access_token` header (same value as `INTERNAL_API_TOKEN`). A missing or wrong token gets `401`. The check is set once on the parent `/api` router (`app/api/v1/api.py`), so new routes get it too.
- **Off by default.** With `INTERNAL_TOKEN_ENFORCED=false` (the default), routes behave as in 2.2.0, so current callers keep working until they send the header. The acronym endpoints need the token in both cases, as in 2.2.0.

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
| `INTERNAL_TOKEN_ENFORCED` | `false` | `true`: every `/api` route except `/api/health` needs the `internal_access_token` header. Keep `false` until every caller (commons-backend) sends the header. |
| `ALLOWED_URL_HOSTS` | `*` | Hosts `metadata.markdown_url` may point to, comma-separated, exact match (any case). `*` allows every host. Empty allows no host, so every URL upload gets `400`. Example: `static.example.org`. |

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

2. Change every caller to send the header on each call to the vector service:
   ```
   internal_access_token: <INTERNAL_API_TOKEN>
   ```

3. Turn the check on and restart the service:
   ```dotenv
   INTERNAL_TOKEN_ENFORCED=true
   ```

4. **Smoke test:**
   ```bash
   # No token -> expect 401
   curl -s -o /dev/null -w "%{http_code}\n" -X POST "http://<HOST>:<PORT>/api/documents/search" \
     -H 'Content-Type: application/json' -d '{"query":"test","top_k":1}'

   # With token -> expect 200
   curl -s -o /dev/null -w "%{http_code}\n" -X POST "http://<HOST>:<PORT>/api/documents/search" \
     -H 'Content-Type: application/json' -H "internal_access_token: <INTERNAL_API_TOKEN>" \
     -d '{"query":"test","top_k":1}'

   # Health needs no token -> expect 200
   curl -s -o /dev/null -w "%{http_code}\n" "http://<HOST>:<PORT>/api/health"
   ```
   Also run one search from commons-backend and confirm it returns results (not `401`).

5. **URL hosts.** With the default `ALLOWED_URL_HOSTS=*`, no `.env` change is needed. To restrict URLs, set the hosts the markdown files are served from and restart the service:
   ```dotenv
   ALLOWED_URL_HOSTS=static.example.org
   ```
   Smoke test (with the list set):
   ```bash
   # Host not in the list -> expect 400 "host 'localhost' is not in ALLOWED_URL_HOSTS"
   curl -s -w " HTTP %{http_code}\n" -X POST "http://<HOST>:<PORT>/api/documents" \
     -H "internal_access_token: <INTERNAL_API_TOKEN>" \
     -F "file=@any.txt" -F "source_id=url-check-1" -F "priority=P1" \
     -F 'metadata={"markdown_url":"http://localhost:6333/collections"}'
   ```
   Then upload one document with a `markdown_url` on an allowed host and confirm it is indexed.

---

## Known limits

- **URL host names are not resolved to IP addresses.** With `*`, or with a listed host whose DNS points to an internal IP, the service still fetches that address. A check that refuses private, loopback and link-local IPs (including `169.254.169.254`) after the DNS lookup is not part of this release.
- `http://` URLs are allowed, so their content can be read or changed on the way.

---

## Rollback

**Fastest — flag off:** set `INTERNAL_TOKEN_ENFORCED=false` and restart. Routes behave as in 2.2.0. No data or schema changes were made in this release.

**URL hosts:** set `ALLOWED_URL_HOSTS=*` and restart to allow every host again. Redirects and the size limit are in code; to remove them, revert the code.
