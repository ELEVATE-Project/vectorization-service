# Acronym search — design notes

How acronym search works today, where the code is, and why it works that way.
Code comments stay short and point here. In the code, every acronym-specific spot
has a `# Acronym:` comment, and every acronym log line starts with `[ACRONYM]`.

## Where the code is

| File | What it does |
|---|---|
| `app/services/acronym_search.py` | Detects acronyms in a query and builds the texts the query is searched with |
| `app/services/acronym_service.py` | The `acronym_mapping` table and its Redis cache: lookup, warm-up, upload |
| `app/services/prioritized_search_service.py` | The search; acronym steps are `if acronym_query:` blocks |
| `app/validations/acronym.py` | Checks for the bulk upload CSV (header and rows) |
| `app/validations/generic.py` | Token comparison and the startup check of required environment variables |
| `app/api/v1/endpoints/acronyms.py` | `GET /api/acronyms` and `POST /api/acronyms/bulkUpload` |
| `app/constants/constants.py`, `messages.py` | Regex patterns, method names, CSV columns; API messages |

## Search flow

For `POST /api/documents/search`, in order:

1. **Clean the query**, as for any query (`preprocess_query`).
2. **Build the acronym query** (`build_acronym_query`), only when
   `ACRONYM_SEARCH_ENABLED` is true. It detects acronyms; with none it returns
   `None` and the rest of the search is the normal search, unchanged.
3. **Search** with the texts it built: the semantic (dense) texts and the keyword
   (BM25) text or texts, all in one Qdrant batch call. When several texts score the
   same chunk for the same field, the higher score is kept (`max()`). Every search
   logs `Search method`, `Dense query` and `Sparse query` (the texts sent to Qdrant).
4. **Rank** as for any query.
5. **Boost** (hybrid mode only): the normal title/summary boosts for the query as
   typed, plus the expansion boosts for the query with the acronym written out.
6. **Add back** title/summary matches that the score threshold dropped.
7. **Respond** with `acronym_info`, and the debug fields when
   `include_scoring_debug` is true.

## Detection (`detect_acronyms`)

Every word of 2+ letters, and every run of 2 to `ACRONYM_MAX_PHRASE_WORDS` words,
is looked up in the dictionary in one batch. The dictionary decides what is an
acronym, so a normal word only matches if it is also a registered acronym.

- **Any case.** The commons media API lowercases queries, so "diet" must find DIET.
- **Stopwords are still looked up.** BE, ME and SO are acronyms and stopwords at
  the same time; other stopwords are not in the dictionary and resolve to nothing.
- **Multi-word keys** ("RTE ACT", "PM SHRI") are found through the word runs. A run
  that contains a word with no letters ("&", "2024") is skipped, so two words that
  are not next to each other are never joined.
- **Plurals.** A word ending in "S" that is not found ("DIETS") is tried once
  without the S ("DIET") and reported under the singular key.
- **Dotted initialisms.** Commons turns "D.I.E.T." into "d i e t", so a run of
  single letters is also tried joined ("DIET"). Side effect: "a i" can join into "AI".
- **Digits and punctuation are removed per word**: "PTM2024" and "PTM's" find PTM.
- **Only the first `ACRONYM_MAX_QUERY_WORDS` (30) words** are checked. Each word adds
  up to `ACRONYM_MAX_PHRASE_WORDS` lookups and cache entries, so a very long query
  would otherwise mean unbounded Redis and Postgres work.
- **Entries with no expansions are dropped**, so the search never sees one.

## Building the search texts (`build_acronym_query`)

It returns an `AcronymQuery`:

| Field | Used for |
|---|---|
| `dense_texts` | Semantic search |
| `sparse_text` | One keyword search: the query as typed plus the words of every meaning |
| `sparse_texts` | Keyword OR: one keyword search per text (or `None`) |
| `written_out` | The query with each acronym written out, one per meaning, for the expansion boosts |

**The method** (`ACRONYM_SEARCH_METHOD`):

- **`inline`** (default): one semantic text with every meaning written next to its
  acronym, then cleaned like any query: "PTM documents" → "PTM (Parent Teacher
  Meeting) documents" → "ptm parent teacher meeting documents". An acronym with
  two meanings gets both: "MIP (Micro Improvement Project, Micro Improvement Plan)".
- **`dual_query`**: the cleaned query plus one text per meaning, each searched
  separately: "ptm documents", "Parent Teacher Meeting documents". At most
  `ACRONYM_MAX_DENSE_VARIANTS` texts, the query included. Keyword OR is always on.

**Keyword OR** (`ACRONYM_KEYWORD_OR`, or always with `dual_query`): one keyword
search per text ("diet", then "District Institute of Education and Training"),
not capped; each chunk keeps its higher raw BM25 score. Without it, one keyword
search uses `sparse_text`. BM25 just counts words, so the extra words of the
meanings cannot dilute it.

**Writing an acronym out** (`_acronym_substitution_pattern`, `_expand_match`):

- One regex finds every detected acronym in one pass, so a meaning that contains
  another acronym is never replaced again. Longest key first ("RTE ACT" before "RTE").
- It accepts the same shapes as detection: letters split by punctuation or single
  spaces ("D.I.E.T.", "d i e t"), multi-word keys ("R.T.E. Act"), a plural "S".
- Letters next to an acronym block a match ("ptmabc", "dietary").
- Digits next to it are kept as their own word: "ptm2024" → "Parent Teacher Meeting
  2024", "123ptm" → "123 Parent Teacher Meeting".
- An acronym is **not** written out when the query already spells its meaning out
  (`_phrase_in_text`), so the meaning is not added twice. Every content word of the
  meaning must be in the query, each matched to its own word, so a repeated word
  must appear as often ("mbbs bachelor medicine surgery" does not spell out
  "Bachelor of Medicine and Bachelor of Surgery"). Word forms match by prefix
  ("institute" ~ "institutes"), limited by `ACRONYM_MIN_PREFIX_MATCH_LEN` and
  `ACRONYM_PREFIX_SUFFIX_CAP` ("tests" ~ "test", but not "testimony").
- An empty mapping is rejected (`ValueError`): its pattern would match bare digits.

The regex pieces are constants: `ACRONYM_MATCH_TEMPLATE`, `ACRONYM_LETTER_GAP_PATTERN`,
`ACRONYM_WORD_GAP_PATTERN`.

## Boosts and add-back (acronym queries)

final score = score before boosts × every multiplier that applies, capped at 1.0.

| Multiplier | When |
|---|---|
| `EXACT_TITLE_BOOST` / `PARTIAL_TITLE_BOOST` (any query) | The title equals / contains the query as typed |
| `ACRONYM_EXPANSION_EXACT_TITLE_BOOST` (1.7) / `..._PARTIAL_TITLE_BOOST` (1.4) | The title equals / contains a written-out query |
| `EXACT_SUMMARY_BOOST` / `PARTIAL_SUMMARY_BOOST` (any query) | The same for the summary |
| `ACRONYM_EXPANSION_EXACT_SUMMARY_BOOST` (1.4) / `..._PARTIAL_SUMMARY_BOOST` (1.2) | The same for the summary |

- For "PTM": "Parent Teacher Meeting" as a title gets 1.7, "Parent Teacher
  Meeting Guidelines" 1.4, "PTM Guidelines" the normal 1.5. A title with both forms
  gets both (1.5 × 1.4). Word order and plurals are not matched.
- **One scroll per field, no extra Qdrant call.** The title (and summary) scroll
  matches the query as typed OR any written-out query; each document is then
  classified against both in Python.
- **Add-back.** Title/summary matches that the threshold dropped are added with the
  floor score `min(max(0.15 × boost, threshold), 1.0)`: title matches first, then
  expansion-title, summary, expansion-summary; a document is added once.
- **Semantic mode** (`search_mode="semantic"`, or `HYBRID_SEARCH_ENABLED=false`):
  no boosts and no add-back, but the search still uses the acronym texts.
- 1.0 turns a boost off; a value below 1.0 stops the service at startup.

**Response.** `acronym_info`: `enabled`, `detected`, `method`, `method_description`
(`semantic`, `keyword`, `boosting`),
and when found the `mapping` and the `ambiguous` acronyms (two or more meanings).
Debug only: `title_expansion_multiplier`, `summary_expansion_multiplier`,
`title_expansion_match`, `summary_expansion_match`, and in
`scoring_context` the expansion boosts.

## Cache (`acronym_service`)

`get_expansions_batch`: one Redis `MGET` for all candidates, then one Postgres
query for the misses, then a write back.

| Cached value | Meaning | TTL |
|---|---|---|
| A JSON list | The acronym's meanings | `REDIS_CACHE_TTL` (24 h) |
| `null` | Not an acronym | `REDIS_NEGATIVE_CACHE_TTL` (1 h) |
| `null` after a Postgres error | Treated as not found for now | `REDIS_DB_ERROR_CACHE_TTL` (30 s) |

- **Write back uses `SET NX`**, so a value an upload wrote after our read is kept.
- **A broken value** (not JSON, or the wrong shape) is a miss: Postgres answers and
  the key is deleted; the next lookup refills it.
- **Full load** (`load_acronym_cache`, at startup and after each upload) writes every
  active acronym and overwrites existing keys, so it also repairs keys changed outside
  the upload API (a migration, a SQL fix). A failed load is logged; the service still
  starts.
- **Redis down**: lookups go to Postgres. **Postgres down**: acronyms are treated as
  not found, and search keeps working.

## Bulk upload (`POST /api/acronyms/bulkUpload`)

Needs the `internal_access_token` header. Bad rows go to `errors`; the other rows
are saved in one transaction.

1. **File:** at most `ACRONYM_BULK_UPLOAD_MAX_SIZE_MB` (413 otherwise), UTF-8; an
   Excel BOM is removed.
2. **Header** (`validate_csv_header`): names are trimmed and lowercased. Duplicate or
   unknown columns reject the file (400), so a typo such as `is_actve` cannot
   silently save a row as active. `acronym` and `expansions` are required. A column
   with no name (a trailing comma) also rejects the file.
3. **Each row** (`validate_acronym_row`): as many fields as the header (an unquoted
   comma would shift values); an acronym of letters only, 1 to
   `ACRONYM_MAX_PHRASE_WORDS` words, at most 32 characters, one word of at least 2
   letters; `is_active` empty, `true` or `false`; expansions required for active rows.
4. **Batch rules** (`bulk_upsert`): a repeated acronym: the last row wins.
   `is_active=false` only turns an existing acronym off and keeps its stored
   expansions and description, so the row needs only the key (given expansions are
   ignored). An unknown acronym is a row error; an acronym cannot be added as inactive.
5. **Save:** in batches of `ACRONYM_BULK_UPLOAD_BATCH_SIZE` rows (200), in one
   transaction; Postgres accepts at most 65,535 values per statement. Active rows are
   upserted, and created vs updated comes from `RETURNING (xmax = 0)`. Inactive rows
   are one `UPDATE ... SET is_active = false`; an acronym it does not find is a row error.
6. **Cache:** the keys of all uploaded acronyms are deleted, then every active
   acronym is loaded again. Deactivated keys stay deleted, so the next lookup reads
   Postgres. If the delete or the load fails, `cache_refreshed` is `false`; the rows
   are saved either way.

A database error returns 503 or 500 and says the outcome is unconfirmed; retrying
the same file is safe, because it is an upsert.

## Settings

| Setting | Default | Meaning |
|---|---|---|
| `ACRONYM_SEARCH_ENABLED` | `true` | Master switch; off gives the normal search |
| `ACRONYM_SEARCH_METHOD` | `inline` | `inline` or `dual_query`; empty means `inline` |
| `ACRONYM_KEYWORD_OR` | `false` | One keyword search per meaning (always on with `dual_query`) |
| `ACRONYM_MAX_DENSE_VARIANTS` | `3` | `dual_query`: semantic texts per query, the query included |
| `ACRONYM_MAX_PHRASE_WORDS` | `4` | Longest acronym in words (detection and upload) |
| `ACRONYM_MAX_QUERY_WORDS` | `30` | Words of the query checked for acronyms |
| `ACRONYM_MIN_PREFIX_MATCH_LEN` | `4` | Shortest word that may prefix-match a meaning word |
| `ACRONYM_PREFIX_SUFFIX_CAP` | `3` | Most extra letters in a prefix match |
| `ACRONYM_EXPANSION_EXACT_TITLE_BOOST` | `1.7` | Title equals a written-out query |
| `ACRONYM_EXPANSION_PARTIAL_TITLE_BOOST` | `1.4` | Title contains a written-out query |
| `ACRONYM_EXPANSION_EXACT_SUMMARY_BOOST` | `1.4` | Summary equals a written-out query |
| `ACRONYM_EXPANSION_PARTIAL_SUMMARY_BOOST` | `1.2` | Summary contains a written-out query |
| `ACRONYM_BULK_UPLOAD_MAX_SIZE_MB` | `5` | Largest CSV accepted |
| `ACRONYM_BULK_UPLOAD_BATCH_SIZE` | `200` | Rows saved per database statement (1 to 10000) |
| `INTERNAL_API_TOKEN` | — (required) | Secret for the acronym endpoints |
| `CACHE_ENABLED` | `true` | Use Redis for acronym lookups |
| `REDIS_DB` | `2` | Redis database (commons uses 0) |
| `REDIS_CACHE_TTL` | `86400` | Seconds a found acronym stays cached |
| `REDIS_NEGATIVE_CACHE_TTL` | `3600` | Seconds a "not an acronym" stays cached |
| `REDIS_DB_ERROR_CACHE_TTL` | `30` | Seconds a miss after a Postgres error stays cached |
| `REDIS_SOCKET_CONNECT_TIMEOUT`, `REDIS_SOCKET_TIMEOUT` | `1` | Seconds before a Redis call gives up |
| `POSTGRES_CONNECT_TIMEOUT` | `3` | Seconds to open a Postgres connection |
| `POSTGRES_STATEMENT_TIMEOUT_MS` | `5000` | Longest query on Postgres |

**At startup the service stops** if `INTERNAL_API_TOKEN` is not set (the list is
`REQUIRED_ENV_VARIABLES` in `app/constants/constants.py`), `ACRONYM_SEARCH_METHOD` is
unknown, an expansion boost is below 1.0, a cache TTL is 0 or less,
`ACRONYM_BULK_UPLOAD_BATCH_SIZE` is outside 1 to 10000, or the `translations` table
cannot be created.

## Database

- `acronym_mapping`: `id` (identity, primary key), `acronym` (unique, uppercase, up
  to 32 characters), `expansions` (JSON list), `description`, `is_active`,
  `created_by` / `updated_by` (`SYSTEM` for now), `created_at`, `updated_at`.
- Migrations: `fcd65d39a795` is the starting point; `f13a664a31b6` creates
  `acronym_mapping` and loads `data/acronyms.csv`; `5b7dea819fa4` converts tables made
  by earlier versions of `f13a664a31b6` to the shape above. The deploy runs
  `alembic upgrade head` before restarting the service.
- The `translations` table predates Alembic and is created by `init_db()` at startup.

## Known limits

- **`dual_query` and stopword acronyms.** In a long query, the clean-up removes BE,
  ME and SO before they are written out, so no semantic text with their meaning is
  made; the keyword text still has the meaning's words. `inline` is not affected.
- **Acronyms after word 30** are not detected; the words are still searched.
- **Two uploads of the same acronym at the same moment**, or an upload while a pod
  loads the cache, can leave the older value in the cache until a re-upload or restart.
- **A search running during an upload** can read the old meaning just before the
  commit and write it back just after the cache delete. A deactivated acronym can
  then stay on until its cache entry expires (`REDIS_CACHE_TTL`, 24 h) or the next
  upload. Uploads are rare, so this is accepted.
- **A frozen Postgres server** (connection open, no answer) can make a search wait:
  `statement_timeout` is enforced by the server, so it cannot help there.
- **Meanings with a repeated word** (MBBS: "Bachelor of ... Bachelor of ..."): a query
  that types the word once is not seen as spelling the meaning out, so the meaning
  is added, and can appear twice in the search text.
