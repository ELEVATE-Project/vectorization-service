# Acronym search — design notes

Why the acronym code works the way it does. Code comments stay short and point
here; each section keeps the reasoning (and the bug or measurement behind it)
that used to live inline.

## Detection (`acronym_query_service.detect_acronyms`)

**Case-insensitive on purpose.** Every token of 2+ letters is a candidate,
whatever its case, and the dictionary decides. An uppercase-only rule was
stricter ("diet" in "the diet chart" should not match DIET), but the commons
media API lowercases queries before forwarding them, which silently disabled
acronym search for every multi-word query. The dictionary is the filter: a
lowercase word that is not a registered acronym resolves to nothing, so only
words that are both everyday English and on file can match by accident.

**Stopwords are checked, not discarded.** In a multi-word query, stopwords
(THE, AND, ON) are kept out of the ordinary candidate list but still looked up
in the same batch. A stopword that is also a registered acronym (BE, ME, SO) is
kept; one that is not stays out. Before this, "BE admission" found nothing,
because "be" was thrown away before anyone checked. Single-word queries are
never stripped.

**Multi-word entries.** The dictionary stores "RTE ACT" and "PM SHRI" as one key
with a space. Detection also tries every contiguous window of 2 to
`ACRONYM_MAX_PHRASE_WORDS` words, each word normalized like a single token.
Phrases get no stopword filter: a whole phrase is specific enough that the
lookup itself is the filter. A window containing a word that normalizes to
nothing (a pure number, or "&") is skipped rather than bridged, so two
non-adjacent words are never glued into a phrase.

**Recovery for text commons already mangled.** Commons replaces every
punctuation character with a space, so "D.I.E.T." and "DIET's" never arrive
intact.
- *Plural retry:* a single-token candidate ending in "S" that misses as typed
  ("DIETS") is retried once without the S ("DIET"), and reported under the
  singular key. The dictionary still decides: "ADMISSIONS" retries
  "ADMISSION" and finds nothing.
- *Single-letter gluing:* a run of 2+ one-letter tokens ("d i e t") is what a
  dotted initialism becomes, and normal English almost never produces it, so
  the run is also tried glued ("DIET"). Accepted trade-off: "a i" would glue
  into the registered "AI"; the dot-vs-space difference is already gone by the
  time the query reaches us.

**Digits.** Digits are stripped per word ("PTM2024" -> "PTM"). So "RTE2024 ACT"
is detected as "RTE ACT", but not substituted in the query, because trailing
digits only attach to the last word of a match. "RTE ACT 2024" is unaffected.

**Empty expansions are dropped.** The `expansions` column defaults to `[]`, and
only `bulk_upsert` checks for a non-empty list, so a row written another way
could be empty. Filtering here protects both users in `search()` (the
substitution regex and the sparse OR-join) at once.

## Query substitution (`_acronym_substitution_pattern`)

The dense expansion variant is built by replacing the acronym in the query with
its expansion. A plain `\bDIET\b` only found a bare "DIET", while detection also
accepts "D.I.E.T.", "d i e t", "DIETS" and "PTM2024". Those were detected and
reported, but the dense expansion was silently never built. So the pattern
accepts the same shapes:
- Letters separated by optional punctuation *or* whitespace ("D.I.E.T.",
  "d i e t"). Only single characters can take the whitespace path, the same
  shape detection needs before it glues a run, so ordinary prose is not
  matched.
- Multi-word keys ("RTE ACT"): each word gets its own letter pattern, joined by
  required whitespace with optional punctuation around it ("R.T.E. Act").
- `(?<![A-Za-z])` / `(?![A-Za-z])`: "dietary" is still rejected.
- An optional trailing "S", matching detection's plural retry. "diets" matches;
  "dietary" still does not, because "ary" is letters.
- Trailing digits are captured, not consumed, so "ptm2024" expands to
  "Parent Teacher Meeting 2024".
- Longest key first ("RTE ACT" before "RTE", PMS before PM), because regex
  alternation is leftmost-first, not longest-match.

## Title and summary matching

Acronym queries get the same title/summary match as any other query: the query
as typed, anywhere in the title or summary, in any case ("exact" when the whole
field is the query). The acronym alone is not matched separately. This is the
pre-acronym behaviour. The acronym-specific boosts (whole-word acronym match, the
old acronym body check, and the soft, full-form, form, exact-only and bonus
options) were removed on 2026-10-09. Known effect: for "DIET", a "Healthy Diet
Guide" title gets the partial title boost (it contains "diet"). A title or summary
that spells out the full name gets the expansion boost instead (see Field boosts).

`_phrase_in_text` is also used for query building: it decides whether the query
already spells out an expansion (see Search pipeline). It requires **all**
content words, allowing inflections by prefix ("institute" ~ "institutes");
`_words_match` limits prefix matches by `ACRONYM_MIN_PREFIX_MATCH_LEN` and
`ACRONYM_PREFIX_SUFFIX_CAP`, so "tests" matches "test" but "testimony" does not.

## Field boosts (acronym queries)

final = score before boosts x title multiplier x summary multiplier, capped at
1.0. The multipliers are the ordinary `EXACT_*`/`PARTIAL_*` boosts (see Title
and summary matching), plus acronym-only title and summary boosts
(matched by the title/summary scroll, whole collection; see Title match scroll).
They use the ordinary rule
(`_classify_text_match`) on the query as typed with the acronym written out,
one text per meaning: "PTM document" -> "Parent Teacher Meeting document".
- `ACRONYM_EXPANSION_EXACT_TITLE_BOOST` (1.7): the title equals the written-out
  query, case ignored ("Parent Teacher Meeting Document"). Debug
  `title_expansion_match` "exact".
- `ACRONYM_EXPANSION_PARTIAL_TITLE_BOOST` (1.4): the title contains it as one phrase
  ("Parent Teacher Meeting document in school"). Debug `title_expansion_match`
  "partial". The summary boosts below report `summary_expansion_match` the same
  way; `title_match` / `summary_match` cover only the query as typed.
As for the ordinary boost, plurals and word order are not matched: "Parent
Teachers Meeting Document" and "Parent Teacher Meeting Guidelines" get no
acronym boost for "PTM document". This boost stacks with the ordinary title
boost: a title with both forms gets both (e.g. 1.5 x 1.4). For a bare "PTM" the written-out query is
"Parent Teacher Meeting": that title gets 1.7, "Parent Teacher Meeting
Guidelines" 1.4, and "PTM Guidelines" the ordinary 1.5.
The summary gets the same two matches with `ACRONYM_EXPANSION_EXACT_SUMMARY_BOOST`
(1.4) and `ACRONYM_EXPANSION_PARTIAL_SUMMARY_BOOST` (1.2); it stacks with the ordinary
summary boost the same way. Title and summary
multipliers stack, as for any query. (Renamed on 2026-10-20 to say exact/partial like the
ordinary boosts; the earlier names were `ACRONYM_EXPANDED_QUERY_TITLE_BOOST`,
`ACRONYM_EXPANSION_TITLE_BOOST`, `ACRONYM_EXPANDED_QUERY_SUMMARY_BOOST` and
`ACRONYM_EXPANSION_SUMMARY_BOOST`, with matching `boost_config` debug keys.)
Matches outside `top_k` are injected like other title/summary matches (a title
match wins when a document matched both). 1.0 turns a boost off; values below
1.0 fail at startup.

History: an acronym bonus (final = relevance x (1 + bonus)) was replaced by
acronym-specific field boosts with a body text boost; on 2026-10-09 those were
removed, and the expansion title boosts above were added. On 2026-10-12 a text
boost came back; it was removed again on 2026-10-19 (see below).

## Text boost (removed 2026-10-19)

`ACRONYM_TEXT_BOOST` / `ACRONYM_EXPANSION_TEXT_BOOST` multiplied the score of top_k
documents without a title boost whose body (any chunk) contained the query as
typed or the query with the acronym written out (`_sources_with_text_containing`,
one extra Qdrant call; debug `acronym_in_body_match` / `acronym_in_body_multiplier`).
Removed because testing showed no gain and some harm:
- Labelled nDCG@10 was unchanged for inline, inline + keyword OR and max on 54
  acronym queries (QA copy and the fresh set), and on a purpose-built PTM set
  (24 graded documents): the top 10 never changed, as it is held by title-boosted
  documents at the 1.0 cap.
- Below the top 10 it lifted off-topic documents that write the acronym or the full
  form once (exam roster, budget, meal record) and pushed down relevant documents
  that do not use the word (e.g. "SMC PTM Module 1", "FLN Diary").
- The first search already retrieves 90-98% of the documents that contain the
  phrase (they score low and the threshold drops them), and BM25 on the chunk text
  already rewards the words, so the boost counted the body a second time.
- Scanning the whole collection instead of top_k added ~1100 documents at the floor
  score, none in the top 10, for +135-405 ms.
Code before the removal: branch `backup/acronym-inline-expansion-2026-10-19-before-text-boost-removal`.

## Relevance blend rescore (removed 2026-10-16)

The old default method (empty `ACRONYM_SEARCH_METHOD`, "pushed") searched one
dense text per variant, then rescored the top `ACRONYM_RESCORE_POOL_LIMIT` (200)
candidates against the query and the expansion separately (two more Qdrant round
trips) and averaged them: (1 - W) x score vs query + W x score vs expansion, with
W = `ACRONYM_EXPANSION_SCORE_WEIGHT` (0.5). In the 2026-10-15 comparison it was
slower than inline (p50 971 ms vs 475 ms at top_k 1000) and worse on real data
(nDCG@10 0.741 vs 0.803), so it was removed together with both settings, the
`normalization_reference` support in `_rank_results` and the external range in
`_min_max_normalize`. The code is kept on branch
`backup/acronym-inline-expansion-2026-10-16-before-pushed-removal`.

## Settings (`app/config.py`)

**`ACRONYM_MAX_DENSE_VARIANTS` = 3.** AC-11: an ambiguous acronym (SSC = Staff
Selection Commission or Sainik School Society) used to get a dense variant for
its first expansion only. Now one variant per expansion, capped by this setting
(including the original query). Only SSC, DM and MIP are ambiguous today.
Used by max (one search per variant) and by keyword OR; inline writes every
meaning into one text instead.

**`ACRONYM_SEARCH_METHOD` = inline.** Picks how an acronym query searches the
query and its expansions: `inline` (default; empty also means inline) or `max`,
both described below. Any other value fails at startup. (Until 2026-10-10 these
were three switches, `ACRONYM_INLINE_EXPANSION`, `ACRONYM_MAX_MERGE` and
`ACRONYM_BLENDED_RETRIEVAL`; the `blend` method and the pushed method with its
rescore were removed on 2026-10-16.)

**`ACRONYM_SEARCH_METHOD` = inline.** The query is searched as one
text with each expansion written next to its acronym, as if the user had typed
it: "PTM documents" -> "PTM (Parent Teacher Meeting) documents"; an ambiguous
acronym gets every meaning ("MIP (Micro Improvement Project, Micro Improvement
Plan)"); an expansion the query already spells out is not added again. One text
means one embedding and no max() merge (`ACRONYM_MAX_DENSE_VARIANTS` then only
affects keyword OR). The sparse text is unchanged.

**Inline text form.** The inline text gets the usual query clean-up, as if the
user had typed it: "PTM (Parent Teacher Meeting) documents" becomes "ptm parent
teacher meeting documents" (lower case, no brackets or stopwords). The
`ACRONYM_INLINE_TEXT_FORM` switch (`cleaned` | `raw` | `raw_no_brackets`) was
removed on 2026-10-19: on the 2026-10-10 tests the difference was small and
topic-dependent (cleaned best for DIET, raw better for SMC), and `cleaned` was the
default.

**`ACRONYM_KEYWORD_OR` = false.** When on, the keyword side runs one BM25 search
per variant (the query as typed, then the query with each meaning substituted:
"diet" OR "District Institute of Education and Training") instead of one search
with every expansion's words. Each chunk keeps the higher of its raw BM25
scores; the usual min-max normalization across the candidate pool follows, as
for one BM25 search. Raw BM25 grows with the number of query words (~9 for
"diet", ~42 for the full name), so the full-name search usually gives the
higher score. max_merge merges its two BM25 lists the same way. Other merges
were tested on 2026-10-19 (62 queries, labelled nDCG@10, inline + keyword OR):
raw max 0.879, raw sum 0.880, each list scaled to its top then max 0.876 (the
former `ACRONYM_KEYWORD_OR_SCALE_LISTS=true`, removed 2026-10-19) or sum 0.882,
RRF of the two lists 0.842. The score-based merges reorder the top 10 but keep
83-97% of its documents; none is a clear gain, so raw max stays. Works with or without
`ACRONYM_SEARCH_METHOD=inline`; the dense side is unchanged. Cost: one extra BM25
search per meaning, same call.

**`ACRONYM_SEARCH_METHOD` = max.** One dense text per variant, each field
keeps the higher of its scores (an OR, so a document needs to match only one
form), and keyword OR is on. The retrieval score is the score before boosts. One
Qdrant call (12 searches for one meaning). On 2026-10-07 tests this mixed
acronym-titled and full-name-titled documents best on the main data; it lets
"diet" (food) documents in through the acronym side, so it relies on the soft
and full-name boosts.

**max_merge merge.** `ACRONYM_SEARCH_METHOD=max` merges the query side and the
expansion side per field: each field keeps its higher score, so a chunk can take
its title score from one side and its text score from the other; keyword stays OR.
The `ACRONYM_MERGE_LEVEL` options (`raw_dense`: the side with the higher weighted
dense sum keeps all its fields; `final`: each side scored in full, higher kept;
`rank`: reciprocal rank fusion of the two rankings) were removed on 2026-10-12;
the per-field merge is the only behaviour.

**`blend` method (removed 2026-10-16).** It mixed the query and expansion vectors
into one, m = (1 - W) x query + W x expansion, and the two BM25 vectors the same
way, so retrieval returned the 50/50 blend directly with no rescore. Weak point:
a chunk outside a field's top `top_k x 8` got 0 for that field, which lifted
generic SMC documents unless the candidate pool was larger (fanout 32). In the
2026-10-15 comparison it was weaker than inline on real SMC/PTM data, so it was
removed. The code is kept on branch
`backup/acronym-inline-expansion-2026-10-16-before-blend-removal`.

**Candidate pool.** Acronym queries use the same pool as every query: `top_k x
SEARCH_CANDIDATE_FANOUT` per field, capped by `SEARCH_CANDIDATE_MAX` (2000 for the
top_k of 1000 that commons sends). An acronym-only fanout (`ACRONYM_CANDIDATE_FANOUT`)
was removed on 2026-10-19: it was added for the blend method (removed 2026-10-16),
fanout 32 was the weakest option in the 2026-10-08 evaluation, and at top_k 1000 the
cap makes any fanout of 2 or more give the same pool.

**Field weights.** Acronym queries use the same field weights as every query
(`SEARCH_PRIORITY_WEIGHTS`: title 0.34, text 0.26, ...). A short acronym query
reads like a title, so an acronym-titled document wins twice: in the search
score and again in the title boost. Same-content test documents that differ
only in title (2026-10-09) showed this: with every boost off, a neutral-titled
copy still ranked #11-#12 for "DIET". The `ACRONYM_FIELD_WEIGHTS` switch (separate
weights for acronym queries, e.g. title 0.20) was removed on 2026-10-19: the
field weights are not going to change for acronym queries.

**`ACRONYM_MIN_PREFIX_MATCH_LEN` = 4.** Below four characters a "prefix" is an
initial, not a stem: "s" would stand in for "school", so "S M C Handbook" read
as School Management Committee. It is a stem length for expansion words, not a
limit on acronym length (that is the column width, 32).

**`ACRONYM_PREFIX_SUFFIX_CAP` = 3.** Real inflections add a few characters
("test" ~ "tests"); unrelated words sharing a prefix add more ("testimony").
Checked against the dictionary to keep plural/tense forms while blocking
collisions on short expansion words ("post", "work", "home", "master").

**`ACRONYM_SEARCH_ENABLED` defaults to true**, matching `.env.sample` and the
release note. With it off, ranking is exactly the pre-feature behaviour
(pinned by `TestFusionFormulaHasNoAcronymBranch`).

**`REDIS_NEGATIVE_CACHE_TTL` (1 h)** caches "not an acronym" so ordinary query
words do not hit Postgres on every request. Uploads refresh the cache for their
acronyms, so this TTL only bounds staleness in rare races.

**Postgres timeouts.** psycopg has no connect timeout by default: a blackholed
host (packets dropped, not refused) hung a connect past 15 s in a live test,
hence `POSTGRES_CONNECT_TIMEOUT`. An open connection to an unresponsive server
(stuck lock, frozen backend) hung every query, found in a pause test, hence
`POSTGRES_STATEMENT_TIMEOUT_MS` = 5 s. Every query here is a small indexed
lookup, so 5 s never fires against a healthy database.

## Cache (`acronym_service`)

`get_expansions_batch` is cache-aside: one Redis `MGET` for every candidate of a
query, Postgres for the misses, then a write-back. Batching also turns N
connection timeouts into one during a Redis outage.
- **Bad cache values are misses.** An unparseable value once raised
  `JSONDecodeError` out of `search()` as a 500. A cached bare string would make
  `expansions[0]` its first *character* ("D"). Both are now treated as a miss,
  and Postgres answers.
- **Negative entries.** A cached `null` means "not an acronym" and is not sent
  to Postgres, or every ordinary word would be queried on every request.
- **Postgres outage** degrades to "not found" (search did not depend on Postgres
  before this feature), cached for the short `REDIS_DB_ERROR_CACHE_TTL`.
- **Write-back uses `SET NX`.** What a lookup writes is only as fresh as its
  read, and an upload can commit in between. An unconditional write buried a
  new acronym under `null`, or restored an old or deactivated expansion for
  24 h. With `SET NX`, whatever the upload wrote wins. Keys holding a broken
  value are overwritten instead, or NX would keep them forever.
- **Deactivation writes a `null` marker, not a delete.** A deleted key is empty,
  so a late lookup could write the old value back. The marker is written only
  for acronyms Postgres still shows as inactive (a second upload may have
  reactivated one), and lives only 30 s: long enough for in-flight lookups, short
  enough that a lost race costs seconds. If the re-read fails, the keys are
  deleted, which heals on the next lookup.
- **`invalidate_cache` and `mark_deactivated_in_cache` swallow Redis errors**
  (the upload has already committed; a 500 would be wrong) but return False, so
  the upload reports `cache_refreshed=false` instead of silently leaving
  deactivated acronyms live for 24 h.
- **Warm-up** (`load_acronym_cache`) writes every active acronym in one
  pipelined call at startup. It is all or nothing; a failure is logged and the
  service boots with a cold cache, which costs latency, not correctness.

## Bulk upload (`bulk_upsert`, `POST /api/acronyms/bulk`)

- The CSV is decoded as `utf-8-sig`: an Excel/Sheets BOM would otherwise join
  the first header name and fail every row.
- Keys must be letters in up to `ACRONYM_MAX_PHRASE_WORDS` words, the only shape
  detection can find. A single-letter whole key ("A") is rejected; a
  single-letter word inside a phrase ("RBI GRADE B") is fine.
- Over-length values are rejected per row, because one reaching Postgres fails
  the single multi-row INSERT for the whole batch.
- `is_active` is optional (missing means active, for older CSVs); anything other
  than true/false is a per-row error.
- Create vs update comes from `RETURNING (xmax = 0)` in the same upsert (an
  inserted row has xmax 0), not from timestamps: two uploads can share the same
  `now`. A database error says the outcome is unconfirmed (a lost connection
  during the commit can leave it saved) and that retrying the whole file is
  safe, since it is an upsert.
- After the commit, active rows are refreshed in the cache and deactivated rows
  get the marker (see Cache). If the refresh fails, active keys are deleted and
  deactivated keys still get the marker. The request succeeds either way (the
  database change is done); `cache_refreshed` says whether the cache caught up.
- `_split_expansions` is duplicated in migration `f13a664a31b6` on purpose:
  migrations must stay frozen snapshots.

## Schema history (`acronym_mapping` key)

Revision `f13a664a31b6` was edited before the table reached production, so a
database recorded at that revision can hold one of these shapes:

| Shape | Where it comes from |
|---|---|
| `id` primary key, identity GENERATED ALWAYS | `f13a664a31b6` as it is now (fresh databases) |
| `id` primary key, identity GENERATED BY DEFAULT | `f13a664a31b6` on 2026-10-01, before the switch to ALWAYS |
| UUID `code` primary key only | `f13a664a31b6` as it was from 2026-09-30 |
| UUID `code` primary key plus a unique identity `id` | earlier versions of `f13a664a31b6` |
| `id` bigserial primary key | the first version of `f13a664a31b6` |

Revision `5b7dea819fa4` converts all of them to the first: it drops `code` and
adds `id` as the key, or promotes the existing `id` (dropping its own unique
constraint), and then makes `id` GENERATED ALWAYS (a serial column gets an
identity that continues after the highest id). GENERATED ALWAYS means nothing
can insert its own `id`, so the identity can never collide with a hand-written
value. Deployment runs `alembic upgrade head` (`deployment/ansible.yml`), so every
environment ends up with the same table without manual SQL.

Its downgrade restores a UUID `code` primary key (new UUIDs; the old values are
gone) and keeps `id` as a unique column. That shape is read by every model
version, so an app rolled back together with `alembic downgrade -1` keeps
working. All five histories were tested with upgrade, downgrade and upgrade on a
throwaway database: rows kept, new rows continue after the highest id, an
explicit id is rejected, `uq_acronym` and the expansions check stay.

## Search pipeline (`prioritized_search_service.search`)

**Query variants.** Each expansion meaning gets its own dense query string,
never concatenated with the original: embedding "PTM meeting" + "Parent Teacher
Meeting" as one text averages two concepts into a point close to neither. All
acronyms are substituted in one regex pass over the original text; sequential
substitution could re-scan an expansion that contains another acronym (NFST ->
"...for ST" when ST is also detected). The replacement is a callable, because a
string replacement treats backslashes specially. The matched span is normalized
per word exactly like detection (one tokenizer: `_normalize_token`), and both the
spaced and glued key are tried, so "d i e t" maps back to DIET. A variant is
skipped when the query already spells the expansion out (a doubled phrase embeds
to a distorted point), and when it differs only in case (embeddings ignore case).
One variant per expansion index (AC-11), capped by `ACRONYM_MAX_DENSE_VARIANTS`;
max searches every variant, inline writes all meanings into one text. The sparse query appends every expansion's words: BM25 is a bag of
words, so extra words cannot dilute it, and its tokenizer has no OR or phrases.
All dense texts are embedded in one `encode()` call and validated before Qdrant.
Retrieval merges same-field hits across embeddings with `max()`.

**Semantic mode.** Acronym ranking (the expansion title/summary boosts) is
lexical, so it follows the same switch as the title/summary boost: off in
`search_mode="semantic"` and when `HYBRID_SEARCH_ENABLED` is false. Retrieval is
not gated: the expanded queries still run, so an acronym still widens *which*
documents are found.

**Boost placement.** `filter_score` judges the pre-boost score. Acronym queries
are boosted like any other query: the title/summary boosts apply to the `top_k`
results, and title/summary matches outside them are injected (see Injected
documents). An earlier hard tiering sorted by (tier, score) and let any
acronym-titled document outrank any untitled one however much weaker:
measured, 16-43 cases per top 20 of a document ranked above one more than 20%
more relevant.

**Injected documents** (title/summary matches the threshold dropped). If the
pipeline scored the document before the threshold, its real field scores are
reused and its real relevance is shown as `pre_floor_score`; the score itself
still uses the floor formula, so it is never mistaken for a semantic hit. A
document never retrieved gets the floor score.

**Debug fields.** `title_multiplier` / `summary_multiplier` carry the boost from the
query as typed; on acronym queries `title_expansion_multiplier` /
`summary_expansion_multiplier` carry the boost from the written-out query (1.0 = no
boost). All are debug-only. `pre_floor_score` and the `title_expansion_boost` /
`summary_expansion_boost` flags were removed on 2026-10-16. The score before boosts is
not a separate field (`acronym_pre_boost_score` was removed on 2026-10-12): for inline
and max it is 0.7 x `normalized_dense` + 0.3 x `normalized_sparse`.

**Keyword breakdown (removed 2026-10-20).** `ACRONYM_KEYWORD_BREAKDOWN=true` added
`keyword_score_acronym` / `keyword_score_expansion` to each result: the chunk's raw
BM25 score against the query as typed and against the written-out query, from one
extra Qdrant call. It was a report-only debugging aid (never used in ranking, off by
default) and was removed once the keyword merge was settled on raw max.

**Title match scroll.** One scroll per field (title, summary), as for any query.
On acronym queries its filter is the query as typed OR each written-out query
(a `should` of `MatchText`), so the expansion boosts add no Qdrant call; each
document is then classified in Python against the typed query and against the
written-out queries. `must_not` filters carry into the scroll, or it would
re-admit excluded documents. A scan of the candidate pool
(`ACRONYM_EXPANSION_MATCH_SOURCE=memory`, `_expansion_field_matches`) was
removed on 2026-10-19: it gave the same rankings in testing but could not see
documents outside the candidate pool (backup branch
`backup/acronym-inline-expansion-2026-10-19-before-scan-removal`).
