# Qdrant Client 1.18 ↔ Server Version Compatibility

## Environment Summary

| Environment | Qdrant Server | Client |
|---|---|---|
| QA | **1.12** | 1.18 |
| Production | **1.18** | 1.18 |

Client 1.18 is required across both environments for BM25 sparse search support.

---

## Issue: QA on Server 1.12

### Warning on every startup

```
qdrant_remote.py:282: UserWarning: Qdrant client version 1.18.0 is incompatible
with server version 1.12.0. Major versions should match and minor version
difference must not exceed 1.
```

This is Qdrant's blanket version gap check — client 1.18 vs server 1.12 is a 6 minor-version gap, exceeding the allowed ≤1. The warning is suppressed via the `QDRANT_CHECK_COMPATIBILITY` env var (defaults to `false`). No code change needed — once the server is upgraded to 1.18, set `QDRANT_CHECK_COMPATIBILITY=true` in `.env` to re-enable the check.

### What actually breaks on server 1.12

Features introduced after 1.12 return `400 … "did not match any variant of untagged enum"` — the server cannot parse the newer request shape:

| Feature | Needs server | Status on QA |
|---|---|---|
| `MatchPhrase` / `phrase_matching` | 1.15+ | ❌ 400 error |
| `FormulaQuery` (score boosting) | 1.14+ | ❌ 400 error |
| Text-index params: `stopwords`, `stemmer`, `ascii_folding`, `phrase_matching` | 1.13+ | ❌ 400 error if set |

> **Do not introduce any of the above while QA server is on 1.12.** Test these only after the server upgrade.

### What works fine on server 1.12 (verified)

All operations the service currently uses pass on 1.12:

| Operation | QA (1.12) |
|---|---|
| `get_collection`, `scroll`, `retrieve` | ✅ |
| `query_points` (dense named vector) | ✅ |
| `query_batch_points` (multi-field dense) | ✅ |
| `query_batch_points` + sparse BM25 (IDF) | ✅ |
| `scroll` with `MatchText` (PREFIX index), `MatchAny` | ✅ |
| `upsert`, `set_payload`, `create_payload_index` | ✅ |

So on QA, the service runs correctly — the warning is a false alarm for the current feature set.

---

## Upgrading QA Server from 1.12 → 1.18

### Before upgrading

1. **Take a snapshot backup:**
   ```bash
   # Create snapshot
   curl -X POST "http://<QA_QDRANT_HOST>:6333/collections/<COLLECTION_NAME>/snapshots"

   # List to confirm
   curl "http://<QA_QDRANT_HOST>:6333/collections/<COLLECTION_NAME>/snapshots"

   # Download locally
   curl -o backup-qa-$(date +%Y%m%d).snapshot \
     "http://<QA_QDRANT_HOST>:6333/collections/<COLLECTION_NAME>/snapshots/<snapshot_name>"
   ```

2. **Note current collection config** (vector names, dimensions, sparse config):
   ```bash
   curl "http://<QA_QDRANT_HOST>:6333/collections/<COLLECTION_NAME>" | python3 -m json.tool
   ```

### Upgrade steps (direct install)

```bash
# Stop the running Qdrant service
sudo systemctl stop qdrant

# Confirm it stopped
sudo systemctl status qdrant

# Download the 1.18 binary (Linux x86_64 — adjust arch if needed)
wget https://github.com/qdrant/qdrant/releases/download/v1.18.2/qdrant-x86_64-unknown-linux-musl.tar.gz

# Extract and replace the binary (check your install path first)
tar -xzf qdrant-x86_64-unknown-linux-musl.tar.gz
sudo mv qdrant /usr/local/bin/qdrant          # adjust path to match existing install

# Start the service — storage directory is unchanged, data is preserved
sudo systemctl start qdrant
```

> Qdrant's storage format is forward-compatible across minor versions — collections and vectors survive an in-place binary swap without re-indexing. The storage path in your config file (`config.yaml`) stays the same.

### After upgrading

1. **Verify server version:**
   ```bash
   curl http://<QA_QDRANT_HOST>:6333/ | python3 -m json.tool
   # "version" should be "1.18.x"
   ```

2. **Verify collection is intact:**
   ```bash
   curl "http://<QA_QDRANT_HOST>:6333/collections/<COLLECTION_NAME>" | python3 -m json.tool
   # points_count should match pre-upgrade count
   ```

3. **Re-enable the compatibility check** — set `QDRANT_CHECK_COMPATIBILITY=true` in `.env`. No code change needed.

4. **Restart the service** and confirm no `UserWarning` on startup.

5. **Run smoke checks:**
   ```bash
   GET /api/health          → healthy
   POST /api/v1/documents/search   → results with non-zero scores
   ```

---
