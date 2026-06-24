# Qdrant client 1.18 ↔ server 1.12 compatibility warning

## Context

On every startup the service logs:

> `qdrant_remote.py:282: UserWarning: Qdrant client version 1.18.0 is incompatible
> with server version 1.12.0. Major versions should match and minor version
> difference must not exceed 1.`

This investigation determined whether the warning is valid, what actually breaks,
and what to do given a hard constraint surfaced by the user:

- **The server cannot be upgraded from 1.12.**
- **The client must stay on 1.18** (project requirement for BM25 sparse search).

So the usual remedy (align the server to the client's minor) is off the table. The
goal becomes: confirm the combination is safe for everything the service uses,
silence the false-alarm warning deliberately, and prevent anyone from later adding
a feature that the 1.12 server can't handle.

## Findings (evidence-based)

**Versions confirmed:** client `qdrant_client-1.18.0` (`.venv/.../dist-info`); server
`1.12.0` live (`curl http://127.0.0.1:6333/` → `"version":"1.12.0"`).

**The warning is legitimate** per Qdrant's own rule (major must match — it does;
minor diff must be ≤1 — here it is 6). It is a *blanket* gap check, not proof that
any used feature is broken.

**Empirical read-only probe (1.18 client → live 1.12 server)** — every operation the
search path uses PASSES:

| Operation (used by the service) | Result |
|---|---|
| `get_collection`, `scroll`, `retrieve` | ✅ |
| `query_points` (dense named vector) | ✅ |
| `query_batch_points` (multi-field dense) | ✅ |
| `query_batch_points` **+ sparse BM25 (IDF)** | ✅ |
| `scroll` `MatchText` (PREFIX index), `MatchAny` | ✅ |

Features **newer than server 1.12 FAIL** with `400 … "did not match any variant of
untagged enum"` (server can't parse the newer request shape) — none are used today:

| Feature | Needs server | Result |
|---|---|---|
| `MatchPhrase` / `phrase_matching` | 1.15+ | ❌ |
| `FormulaQuery` (score boosting) | 1.14+ | ❌ |
| Post-1.12 text-index params: `stopwords`, `stemmer`, `ascii_folding`, `phrase_matching` | 1.13+ | ❌ if set (today passed as `None` ⇒ OK) |

**Separate, unrelated issue (NOT version-caused):** `app.log` shows
`400 … "Vector dimension error: expected dim: 384, got 0"` on `query/batch`. This is
an app bug — an *empty* query vector being sent — independent of the version gap.
Flagged for separate follow-up; out of scope here.

## Conclusion

- The warning is **valid but, for this service, a false alarm.** Server 1.12 already
  supports the entire dense + sparse-BM25 Query-API path the code relies on.
- It **can be safely ignored functionally**, but should be *consciously suppressed
  and documented* (not left noisy, not blindly trusted forever).
- **Do not downgrade the client** (project needs 1.18; would also drop below where
  `query_points` ergonomics/fastembed BM25 are expected).
- **Do not upgrade the server** (user constraint).

## Fix to implement

1. **Suppress the warning at the source — the one legitimate use of the flag.**
   In [app/core/clients/qdrant.py:12](vectorization-service/app/core/clients/qdrant.py#L12):
   ```python
   qdrant_client = QdrantClient(
       settings.QDRANT_HOST,
       port=settings.QDRANT_PORT,
       check_compatibility=False,  # server pinned at 1.12, client 1.18 for BM25;
                                   # combination verified — see CLAUDE.md compat matrix
   )
   ```
   Apply the same flag + comment to the other two instantiations for consistency:
   [scripts/migrate_to_sparse_vectors.py:624](vectorization-service/scripts/migrate_to_sparse_vectors.py#L624)
   and [rename_key_entities_field.py:17](vectorization-service/rename_key_entities_field.py#L17).

2. **Document the supported/unsupported matrix** so the gap stays safe over time.
   Add a short "Qdrant 1.18 client ↔ 1.12 server compatibility" subsection to
   [vectorization-service/CLAUDE.md](vectorization-service/claude.md) (near §4 Key
   Dependencies) capturing the PASS/FAIL table above and the rule: **do not introduce
   `MatchPhrase`/`phrase_matching`, `FormulaQuery`, or post-1.12 text-index params
   while the server is on 1.12** — they 400 against this server.

3. **Regression guard (recommended).** Add a lightweight test in `tests/` that runs
   the core read path (`query_batch_points` with a 384-dim dense vector + a `bm25`
   sparse vector) against the configured server and asserts a non-error response, so
   any future client/server drift that breaks the used path is caught in CI rather
   than at runtime. Reuse the probe in
   `scratchpad/probe.py` as the starting point.

## Verification

- Re-run the read-only probe: all "used path" rows PASS, newer-feature rows FAIL
  (expected) — confirms the boundary is unchanged.
- Restart the service (`./start_mac.sh`) and confirm: (a) the `UserWarning` no longer
  appears on startup, (b) `POST /api/v1/documents/search` with `SPARSE_SEARCH_ENABLED=true`
  returns results (BM25 path exercised).
- `GET /api/health` → qdrant status ok.
- Run the new guard test: `pytest tests/ -k compat`.

## Out of scope (note to user)

The `expected dim: 384, got 0` 400s in `app.log` are a separate empty-vector bug in
the search path, not caused by the version gap. Recommend a follow-up to find where
an empty query/sparse vector is constructed in
[prioritized_search_service.py](vectorization-service/app/services/prioritized_search_service.py)
`_hybrid_batch_search`.
