# Release 2.3.0 — API Access Token for All Routes

## Table of Contents

- [Release 2.3.0 — API Access Token for All Routes](#release-230--api-access-token-for-all-routes)
  - [Table of Contents](#table-of-contents)
  - [What's New](#whats-new)
  - [New Configuration](#new-configuration)
  - [Dependencies](#dependencies)
  - [Deployment](#deployment)
  - [Rollback](#rollback)

---

## What's New

**API access**

- **Token check for every `/api` route.** With `INTERNAL_TOKEN_ENFORCED=true`, every route under `/api` except `/api/health` needs the `internal_access_token` header (same value as `INTERNAL_API_TOKEN`). A missing or wrong token gets `401`. The check is set once on the parent `/api` router (`app/api/v1/api.py`), so new routes get it too.
- **Off by default.** With `INTERNAL_TOKEN_ENFORCED=false` (the default), routes behave as in 2.2.0, so current callers keep working until they send the header. The acronym endpoints need the token in both cases, as in 2.2.0.

---

## New Configuration

| Variable | Default | Description |
|---|---|---|
| `INTERNAL_TOKEN_ENFORCED` | `false` | `true`: every `/api` route except `/api/health` needs the `internal_access_token` header. Keep `false` until every caller (commons-backend) sends the header. |

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

---

## Rollback

**Fastest — flag off:** set `INTERNAL_TOKEN_ENFORCED=false` and restart. Routes behave as in 2.2.0. No data or schema changes were made in this release.
