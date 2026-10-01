# Release 2.2.0 — Acronym-Aware Search

**Service:** vectorization-service

> ⚠️ **Run `alembic upgrade head` before starting the service.** The acronym cache warms at startup by reading the `acronym_mapping` table. If the table does not exist yet, warm-up fails (logged, non-fatal) and acronym detection finds nothing until the migration has run.

> ℹ️ **About [release 2.1.0](release-2.1.0.md).** That note describes an earlier, unreleased design of acronym search (a ranking bonus, a UUID `code` key). This release is the first to ship acronym search; where the two notes differ, this one applies.

---

## Table of Contents

- [What's New](#whats-new)
- [How Acronym Search Works](#how-acronym-search-works)
- [API](#api)
- [Configuration](#configuration)
- [Dependencies](#dependencies)
- [Deployment](#deployment)
- [Rollback](#rollback)

---

## What's New

- **Acronym detection in search queries.** A query containing a registered acronym (e.g. `DIET`, `PTM`, `SSC`) is detected and expanded before retrieval, so a search for the acronym also finds documents that only spell out its full meaning. Gated by `ACRONYM_SEARCH_ENABLED` (default `true`).
- **Acronym-aware ranking.** Acronym queries blend the score for the query with the score for its expansion, then apply the title and summary boosts on the acronym and a text boost when the document body uses the acronym. A title that merely contains the everyday word ("Healthy Diet Guide") does not earn the body boost.
- **Acronym dictionary.** New Postgres table `acronym_mapping`, seeded with 486 acronyms. One acronym can have several meanings (`SSC` → "Staff Selection Commission" / "Sainik School Society"). Lookups go through a Redis cache.
- **Bulk upload endpoint.** `POST /api/acronyms/bulk` creates, updates, enables or disables acronyms from a CSV, and reports which acronyms were created, updated and deactivated.
- **`acronym_info` in the search response**, showing whether acronym search is enabled, whether an acronym was detected, and its expansions.
- **Scoring debug output** for acronym queries (with `include_scoring_debug=true`), so the final score can be rebuilt from its parts.

---

## How Acronym Search Works

<details>
<summary>Detect → look up → expand and retrieve</summary>

1. **Detect** (`acronym_query_service.detect_acronyms`): every token of 2+ letters, and every contiguous phrase of 2 to `ACRONYM_MAX_PHRASE_WORDS` words, is checked against the dictionary, whatever its case. Multi-word entries (`RTE ACT`, `PM SHRI`) are found too. The dictionary is the filter: a lowercase word that is not registered resolves to nothing. Stopwords are kept out of the ordinary candidate list but still looked up, so an acronym that is also a stopword (`BE`, `ME`) is found. Two recovery passes handle text commons has already changed: a plural retry (`DIETS` → `DIET`) and gluing single-letter runs (`D.I.E.T.` arrives as `d i e t` → `DIET`).
2. **Look up** (`acronym_service.get_expansions_batch`): one batched Redis round trip for every candidate, with Postgres for the misses (written back to Redis).
3. **Expand and retrieve**: up to `ACRONYM_MAX_DENSE_VARIANTS` dense query texts (the original plus one per expansion meaning, each with the acronym replaced by the expansion, never concatenated), and one BM25 query with every expansion's words appended. A query that already spells out the expansion gets no duplicate variant.

</details>

<details>
<summary>Ranking</summary>

```
acronym_pre_boost_score = (1 - W) × score vs query + W × score vs expansion     (W = ACRONYM_EXPANSION_SCORE_WEIGHT, 0.5)
score = acronym_pre_boost_score × title multiplier × summary multiplier × text multiplier,  capped at 1.0
```

- **Blend.** Retrieval merges the query and expansion results; scoring then averages them, so a document must match both. Only the top `ACRONYM_RESCORE_POOL_LIMIT` candidates are blended; the rest keep their retrieval score.
- **Title / summary multipliers.** The standard `EXACT_TITLE_BOOST` / `PARTIAL_TITLE_BOOST` / `EXACT_SUMMARY_BOOST` / `PARTIAL_SUMMARY_BOOST`, matched on the query and the acronym as **whole words**: "DIET" never matches inside "dietary"; "_DIET_" in file-name titles does.
- **Text multiplier.** `EXACT_TEXT_BOOST` when the body uses the acronym **as written** (capitals for single-word keys, so "a healthy diet" does not count), else 1.0.
- `filter_score` judges the pre-boost score. The boosts are applied before the `top_k` cut, so they can lift a document onto the page.
- Many strong documents reach the 1.0 cap; their order then follows the score before the cap.
- Ordinary (non-acronym) queries and `search_mode="semantic"` keep the existing ranking. In semantic mode, acronym expansion still widens retrieval.

</details>

<details>
<summary>Body check (text multiplier)</summary>

1. One grouped BM25 query per acronym over the checked documents; their top `ACRONYM_BODY_CHECK_TOP_CHUNKS` chunks must use the acronym in capitals. A BM25 hit that fails there has all its chunks read before it is rejected.
2. Chunks without a BM25 vector (indexed before BM25 was enabled) are listed with a `has_vector` filter and checked directly with the same rule.
3. Only candidates that can still reach the page are checked (score × `EXACT_TEXT_BOOST`, capped at 1.0, reaches the k-th score), at most `ACRONYM_BODY_CHECK_MAX_SOURCES`.

Reported as `search_config.scoring_context.acronym_ranking.body_check`:

| Value | Meaning |
|---|---|
| `bm25` | Every checked document has BM25 vectors |
| `bm25+scan` | Some documents were read directly |
| `scan` | All checked documents were read directly |
| `bm25 (scan unsupported)` | Qdrant before 1.13 rejected the `has_vector` filter; chunks without BM25 vectors were not checked. Remembered until restart |
| `unavailable` | The check could not run (sparse search off, encoder or query failure); no text boost |
| `none` | Nothing to check |

⚠️ The direct read needs **Qdrant server 1.13 or newer**. On an older server it is skipped with a warning, and documents without BM25 vectors get no text boost; run the BM25 backfill ([Deployment](#deployment), step 2).

</details>

<details>
<summary>Cache</summary>

- **Cache-aside reads** with negative caching ("not an acronym", `REDIS_NEGATIVE_CACHE_TTL`) and a short cache for database-outage misses (`REDIS_DB_ERROR_CACHE_TTL`). Write-back uses `SET NX`, so a lookup cannot overwrite what a concurrent upload wrote.
- **After an upload**, active acronyms are re-cached and deactivated ones get a short-lived "not an acronym" marker, so disabling takes effect immediately.
- **Warm-up** at startup loads every active acronym in one pipelined write. If it fails, the service boots with a cold cache (slower lookups, same results).
- **`CACHE_ENABLED=false`** skips Redis entirely and reads Postgres directly.
- A Redis or Postgres outage never fails a search: acronym lookups fall back, and detection finds nothing if both are down.

</details>

---

## API

<details>
<summary><code>POST /api/documents/search</code> — <code>acronym_info</code></summary>

Always an object for a request with query text (`null` only without query text):

| Situation | Value |
|---|---|
| `ACRONYM_SEARCH_ENABLED=false` | `{"enabled": false, "detected": false}` |
| Enabled, no acronym in the query | `{"enabled": true, "detected": false}` |
| Acronym detected | `{"enabled": true, "detected": true, "mapping": {"DIET": ["District Institute of Education and Training"]}}`, plus `"ambiguous": ["SSC"]` for acronyms with several meanings |

</details>

<details>
<summary><code>POST /api/documents/search</code> — debug fields (<code>include_scoring_debug=true</code>)</summary>

Per result:

| Field | When set | Meaning |
|---|---|---|
| `acronym_pre_boost_score` | Acronym queries, rescored documents | The score before the title, summary and text boosts |
| `title_multiplier`, `summary_multiplier` | All queries | Title / summary boost applied (1.0 = none) |
| `acronym_in_body_match`, `acronym_in_body_multiplier` | Acronym queries | `"exact"` and `EXACT_TEXT_BOOST` when the body uses the acronym |
| `pre_floor_score` | Keyword-injected rows (`match_source` set) | The relevance measured before the threshold dropped the document; its `score` is the floor value |

`search_config.scoring_context` also reports `boost_config` (mode `acronym_field_boost` with the multipliers in effect) and `acronym_ranking` (blend weight, rescore pool, `body_check`).

</details>

<details>
<summary><code>POST /api/acronyms/bulk</code></summary>

Requires **both** headers, `internal_access_token` and `admin_auth_token`.

| Column | Required | Notes |
|---|---|---|
| `acronym` | Yes | Case-insensitive, stored uppercase, max 32 characters. Letters only, 1 to `ACRONYM_MAX_PHRASE_WORDS` space-separated words (detection drops digits and punctuation, so `G20` or `COVID-19` cannot be registered). |
| `expansions` | Yes | Several meanings separated by `\|`, e.g. `Staff Selection Commission\|Sainik School Society` |
| `description` | No | Free text |
| `is_active` | No | `true` / `false`; missing means active. Any other value is a row error. |

Response: `received`, `created`, `updated`, `deactivated`, `cache_refreshed`, `errors` (one entry per rejected row; the other rows still save), and `created_acronyms`, `updated_acronyms`, `deactivated_acronyms` in CSV order. Uploads are upserts on `acronym`, so re-uploading a file is safe. A database error returns 503 (unreachable) or 500 and says the outcome could not be confirmed.

`GET /api/acronyms` (header `internal_access_token`) lists the dictionary, with `prefix`, `is_active`, `limit` and `offset` filters.

</details>

---

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `ACRONYM_SEARCH_ENABLED` | `true` | Master switch for acronym search |
| `INTERNAL_API_TOKEN` | — (required) | Header `internal_access_token` for the acronym endpoints |
| `ADMIN_API_TOKEN` | — (required) | Header `admin_auth_token`, additionally required for bulk upload |
| `ACRONYM_BULK_UPLOAD_MAX_SIZE_MB` | `5` | Largest CSV accepted |
| `EXACT_TEXT_BOOST` | `2.0` | Text multiplier when the body uses the acronym |
| `ACRONYM_EXPANSION_SCORE_WEIGHT` | `0.5` | Expansion share of the pre-boost blend |
| `ACRONYM_RESCORE_POOL_LIMIT` | `200` | Candidates the blend rescores |
| `ACRONYM_BODY_CHECK_TOP_CHUNKS` | `3` | Top BM25 chunks per document read by the body check |
| `ACRONYM_BODY_CHECK_MAX_SOURCES` | `200` | Most documents the body check reads per query |
| `ACRONYM_MAX_DENSE_VARIANTS` | `3` | Dense query variants per acronym query, including the original |
| `ACRONYM_MAX_PHRASE_WORDS` | `4` | Longest multi-word acronym detected and accepted on upload |
| `ACRONYM_MIN_PREFIX_MATCH_LEN` | `4` | Shortest word stem allowed to prefix-match an expansion word |
| `ACRONYM_PREFIX_SUFFIX_CAP` | `3` | Most extra characters in such a prefix match ("tests" ~ "test") |
| `CACHE_ENABLED` | `true` | Use Redis for acronym lookups |
| `REDIS_DB` | `2` | Redis database for the acronym cache (commons uses db 0 on a shared Redis) |
| `REDIS_CACHE_TTL` / `REDIS_NEGATIVE_CACHE_TTL` / `REDIS_DB_ERROR_CACHE_TTL` | `86400` / `3600` / `30` | Cache lifetimes (seconds) for hits, "not an acronym" and database-outage misses |
| `REDIS_SOCKET_CONNECT_TIMEOUT` / `REDIS_SOCKET_TIMEOUT` | `1` / `1` | Redis timeouts (seconds) |
| `POSTGRES_CONNECT_TIMEOUT` / `POSTGRES_STATEMENT_TIMEOUT_MS` | `3` / `5000` | Postgres connect timeout (seconds) and per-statement timeout |

`SPARSE_SEARCH_ENABLED=true` (BM25) is needed for the text boost; without it, `body_check` reports `unavailable`.

---

## Dependencies

- **`alembic`** manages the `acronym_mapping` schema.
- **`psycopg[binary]`** (psycopg 3) alongside `psycopg2-binary`: a `postgresql+psycopg://` URL, as used by the deployed environments, loads psycopg 3; a plain `postgresql://` URL loads psycopg2.
- **`click`** is an explicit dependency: `spacy` imports it, and recent `typer` releases no longer pull it in.
- The stopword list is `spacy`'s built-in list; no extra model download. The `en_core_web_sm` model is installed by the Ansible deploy step.
- **`data/acronyms.csv`** (486 acronyms) is committed; the migration seeds the table from it.
- **Qdrant server 1.13+** is needed for the body check's direct read of chunks without BM25 vectors.

---

## Deployment

### Pre-deploy checklist

1. **Postgres is reachable.** `acronym_mapping` is new; the existing `translations` table is untouched.
2. **BM25 coverage** (recommended; required on a Qdrant server older than 1.13): add BM25 vectors to any chunks that lack them. Already-indexed points are skipped, so it is safe to re-run:
   ```bash
   PYTHONPATH=. COLLECTION_NAME=<collection> SPARSE_SEARCH_ENABLED=true \
     .venv/bin/python3 scripts/migrate_to_sparse_vectors.py --dry-run   # report only
   PYTHONPATH=. COLLECTION_NAME=<collection> SPARSE_SEARCH_ENABLED=true \
     .venv/bin/python3 scripts/migrate_to_sparse_vectors.py
   ```

### Deploy steps

3. **Set the env keys** (see `.env.sample`). Required:
   ```dotenv
   ACRONYM_SEARCH_ENABLED=true
   SPARSE_SEARCH_ENABLED=true
   INTERNAL_API_TOKEN=<generate-a-real-secret>
   ADMIN_API_TOKEN=<generate-a-different-real-secret>
   ```
   The two tokens must be different secrets: `INTERNAL_API_TOKEN` may be shared with other internal callers that should not be able to write the dictionary. All other settings have working defaults ([Configuration](#configuration)).
4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
5. **Run the migrations** (`deployment/ansible.yml` already does this):
   ```bash
   alembic upgrade head
   ```
   This creates `acronym_mapping` (primary key `id`, an identity column that is `GENERATED ALWAYS`; `acronym` unique) and seeds it from `data/acronyms.csv`. On a database that already has the table from a pre-release build, the second migration (`5b7dea819fa4`) converts it to the same shape and keeps every row ([design notes, Schema history](acronym-design-notes.md#schema-history-acronym_mapping-key)).
6. **Start the service** and check the log line:
   ```
   Acronym cache warmed: 486 active acronym(s)
   ```
   A warning instead ("Acronym cache warm-up failed …") means the service booted with a cold cache; lookups still work through Postgres.
7. **Verify:**
   ```bash
   curl -s -X POST "http://<HOST>:<PORT>/api/documents/search" -H 'Content-Type: application/json' \
     -d '{"query":"DIET","top_k":5,"filter_score":0.35,"include_scoring_debug":true}' \
     | jq '{acronym_info, mode: .search_config.scoring_context.boost_config.mode,
            body_check: .search_config.scoring_context.acronym_ranking.body_check,
            top: [.results[] | {title, score, acronym_pre_boost_score, title_multiplier, acronym_in_body_multiplier}]}'
   ```
   Expect `acronym_info` with `"enabled": true, "detected": true` and the DIET expansion, mode `acronym_field_boost`, `body_check` `bm25` or `bm25+scan`, and `acronym_in_body_multiplier: 2.0` on documents that use "DIET" in their body.
8. **Optional — add or update acronyms:**
   ```bash
   curl -X POST "http://<HOST>:<PORT>/api/acronyms/bulk" \
     -H "internal_access_token: <INTERNAL_API_TOKEN>" \
     -H "admin_auth_token: <ADMIN_API_TOKEN>" \
     -F "file=@acronyms.csv;type=text/csv"
   ```

---

## Rollback

**Fastest — flag off:** set `ACRONYM_SEARCH_ENABLED=false` and restart. Detection never runs and search ranks exactly as without the feature; `acronym_info` reports `{"enabled": false, "detected": false}`.

**Schema:** `alembic downgrade fcd65d39a795` drops `acronym_mapping`. ⚠️ Destructive: acronyms added or edited through the bulk endpoint are lost. Take a Postgres backup first (`pg_dump -t acronym_mapping`).
