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

## Title and summary matching (`_build_field_match_queries`)

The boost/injection path and the bonus path used to disagree about what
"matches" means. Substring matching boosted "Dietary Guidelines" for DIET and
"Basic Maths" for BA, while a title that really was the expansion (plural, or
different connectives) earned a grade but was never boosted or injected. Now
each check carries its own rule, the same rules `_acronym_bonus` uses:
- `substring`: the pre-acronym behaviour, used only when no acronym is
  detected, so ordinary queries are unchanged.
- `word`: whole-word, for the query and the acronym.
- `phrase`: all content words present, for expansions.

For expansions the Qdrant `MatchText` prefilter gets only the content words.
Sending the full phrase made it demand "of" and "and", so "District Institute
for Education & Training" was never retrieved to be classified.

`_term_in_text` uses letter/digit lookarounds instead of `\b`, because `\b`
treats `_` as a word character and missed "DIET" in file-name titles like
"..._DIET Empowerment..." (seen on real titles).

`_phrase_in_text` requires **all** content words, allowing inflections by prefix
("institute" ~ "institutes"). A partial threshold would promote near-misses:
"District Primary Education Programme" shares district and education with the
DIET expansion but is a different programme. `_words_match` limits prefix
matches by `ACRONYM_MIN_PREFIX_MATCH_LEN` and `ACRONYM_PREFIX_SUFFIX_CAP`, so
"tests" matches "test" but "testimony" does not.

## Body check (`_sources_with_acronym_in_body`)

A title or summary is a claim about the topic, not evidence (the corpus has a
"DIET Reference Handbook" about coastal navigation). So the acronym grades of
the bonus need the document's body to back the acronym.
- One BM25 query per acronym, restricted to the claiming sources and grouped by
  source. Restricting to the candidates makes the answer independent of
  `top_k`.
- BM25 folds case, so "diet" and "DIET" look the same. The top chunks
  (`ACRONYM_BODY_CHECK_TOP_CHUNKS`) must use the acronym in capitals
  (`_acronym_use_pattern`), so a food article titled "DIET Handbook" fails.
  Multi-word keys ("RTE Act", "NIPUN Bharat") match in any case.
- Top chunks rank by how often the word appears, not by case, so a BM25 hit
  that fails in its top chunks has all its chunks read before it is rejected.
- A body that spells out the expansion ("Parent Teacher Meeting") also backs
  the acronym. Without this, such a PTM document ranked about 15th instead of
  about 5th.
- Kept per acronym: in "DIET SMC", body evidence for SMC must not back a DIET
  title claim.
- Returns None when BM25 cannot answer (sparse search off, encoder missing,
  query failed), so the caller falls back instead of treating "could not check"
  as "absent". A BM25 index with no vectors would reject everything, so if no
  pooled row has a sparse score, BM25 is treated as unavailable.
- The fallback (`_body_matched`) only checks for a dense `text` score. That is
  pool membership, not relevance, and applies to all acronyms alike; it exists
  only so a missing BM25 index does not demote every acronym document.

## Bonus (`_acronym_bonus`)

final = relevance x (1 + bonus). Grades: title acronym 0.40, title expansion
0.30, summary acronym 0.20, summary expansion 0.10. They replace the old tiers
4/2/3/1, but scale relevance instead of placing documents in bands, so a title
can close a gap but never lift a much weaker document over a stronger one.
- Acronym grades need body backing for *that* acronym; expansion grades do not,
  because spelling the expansion out is itself evidence. All grades are
  checked, so an unbacked title acronym can still earn the expansion grade.
- With several acronyms, each gets its own best grade and they are summed, then
  capped at `ACRONYM_BONUS_MULTI_MATCH_CAP` (0.60): a title with both SMC and
  DIET outranks one with only DIET, without the multiplier running away.
- Computed once per source (all chunks share title, summary and body); a live
  acronym query ranks about 2,500 chunks over about 590 sources.

## Relevance blend (`_blended_acronym_relevance`)

relevance = (1 - W) x score vs query + W x score vs expansion,
W = `ACRONYM_EXPANSION_SCORE_WEIGHT` (0.5). Retrieval merges both texts with
max(), which is right for finding documents but wrong for scoring them: a
generic expansion scores any education document highly ("District Primary
Education Programme" went from 0.28 against "DIET" to 0.89 with the expansion,
above every real DIET document), while a bare "ptm" means almost nothing.
Averaging needs both to agree.

Only the top `ACRONYM_RESCORE_POOL_LIMIT` candidates are rescored; the rest keep
their retrieval score. The full pool's min/max (`normalization_reference`) is
passed so the rescored subset stays on the same scale; otherwise ordering and
`filter_score` break at the cutoff. RRF mode rescores the whole pool, because
RRF scores are rank-based and cannot use that reference.

## Settings (`app/config.py`)

**`ACRONYM_EXPANSION_SCORE_WEIGHT` = 0.5, with bonuses 0.40/0.30/0.20/0.10.**
Chosen on 2026-09-29 by comparing 0.2, 0.35, 0.65, 0.8 and 0.7-with-doubled-
bonuses on 19 acronyms of the local corpus
(`scripts/simulate_soft_acronym_boost.py`). At 0.2 the acronym side dominated:
a nutrition document titled "Healthy Diet Guide" outranked real DIET documents
(MiniLM reads "diet" as food), and documents that only spell out the expansion
ranked far below their content. Above 0.5 the expansion's generic words took
over ("District ... Education" pulled "District Primary Education Programme"
into DIET's top 5; "School ... Committee" pushed SMC modules out of SMC's top
10). At 0.5 with doubled bonuses, DIET's top 10 is all DIET documents and every
Parent Teacher Meeting document fills PTM's top 7. Known cost: thin documents
with the acronym only in the title (for example scanned PDFs) slip below richer
on-topic documents.

**`ACRONYM_BONUS_MULTI_MATCH_CAP` = 0.60.** Several acronyms sum their grades
(AC-15: taking only the best made a document about SMC and DIET rank the same
as one about DIET). Two full title matches would sum to 0.80; the cap gives a
visibly higher ceiling than one match (1.6x vs 1.4x) without approaching 2x.

**`ACRONYM_RESCORE_POOL_LIMIT` = 200.** The blend costs two more Qdrant round
trips. Uncapped it raised latency by 115% at top_k=10 and 297% at top_k=1000
(measured). Blending only reweights two signals the pool is already ranked by,
so a document far down is not going to reach the top.

**`ACRONYM_MAX_DENSE_VARIANTS` = 3.** AC-11: an ambiguous acronym (SSC = Staff
Selection Commission or Sainik School Society) used to get a dense variant for
its first expansion only. Now one variant per expansion, capped by this setting
(including the original query). Only SSC, DM and MIP are ambiguous today.

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
- Create vs update is decided before the upsert, because `ON CONFLICT` does not
  report which branch ran.
- After the commit, active rows are refreshed in the cache and deactivated rows
  get the marker (see Cache). If the refresh fails, active keys are deleted and
  deactivated keys still get the marker. The request succeeds either way (the
  database change is done); `cache_refreshed` says whether the cache caught up.
- `_split_expansions` is duplicated in migration `f13a664a31b6` on purpose:
  migrations must stay frozen snapshots.

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
only the first substituted variant feeds the relevance blend, the others widen
retrieval. The sparse query appends every expansion's words: BM25 is a bag of
words, so extra words cannot dilute it, and its tokenizer has no OR or phrases.
All dense texts are embedded in one `encode()` call and validated before Qdrant.
Retrieval merges same-field hits across embeddings with `max()`.

**Semantic mode.** The bonus is a lexical signal, so it follows the same switch
as the title/summary boost: off in `search_mode="semantic"` and when
`HYBRID_SEARCH_ENABLED` is false. Retrieval is not gated: the expanded queries
still run, so an acronym still widens *which* documents are found.

**Bonus placement.** Applied after the threshold (so `filter_score` judges real
relevance) and before dedup and the `top_k` cap (so the bonus can lift a
document onto the page). It replaced hard tiering, which sorted by (tier, score)
and let any acronym-titled document outrank any untitled one however much
weaker: measured, 16-43 cases per top 20 of a document ranked above one more
than 20% more relevant.

**Title/summary boost on acronym queries.** The bonus already *is* the
title/summary signal, so the boost multipliers go neutral (1.0) instead of
counting it twice. The match passes still run, for injection candidates and
the `title_match`/`summary_match` debug fields. A neutral multiplier leaves the
score untouched, so scores above 1.0 are not clamped.

**Injected documents** (title/summary matches the threshold dropped). If the
pipeline scored the document before the threshold, its real field scores and
bonus are reused, and its real relevance is shown as `measured_relevance`; the
score itself still uses the floor formula, so it is never mistaken for a
semantic hit. A document never retrieved gets the floor score, and its bonus is
computed from its own scrolled title/summary with the same body check, done in
one batch. Without that check the bonus would depend on `top_k`, because which
documents miss the pool depends on the pool size. If BM25 cannot answer, these
documents get no acronym grades, since nothing was measured about them.

**Debug fields.** `acronym_bonus` is None (not 0.0) for non-acronym queries, so
"never computed" is distinguishable from "computed as zero". `measured_relevance`
is set only for reused documents.

**Title match scroll.** The literal query and every expansion are checked in
one scroll with an OR filter, not one scroll per text, classified as if each
had its own scroll ("exact" is never downgraded). `must_not` filters carry
into the scroll, or it would re-admit excluded documents.
