"""Acronym-aware scoring and matching, split out of PrioritizedSearchService.

Everything here is new relative to release-2.1.0 (verified by diffing method
lists against that tag) — the soft-boost bonus system, the query/expansion
relevance blend, and the shared title/summary matching rules that both of
those depend on. None of it existed before this feature; it's grouped here
purely to keep prioritized_search_service.py from growing without bound, not
because it's meant to be reused independently — every method still relies on
self._hybrid_batch_search / self._parallel_batch_search / self._rank_results /
self.collection_name from the core service, so this is a mixin, not a
standalone module.
"""
import logging
import re
from functools import lru_cache
from typing import Any, Dict, List, NamedTuple, Optional, Set, Tuple

from qdrant_client import models
# Same stopword list acronym detection uses, so the two agree on what counts as
# a content word (see _expansion_content_words / acronym_query_service.detect_acronyms).
from spacy.lang.en.stop_words import STOP_WORDS

from app.config import settings
from app.core.clients.qdrant import qdrant_client

logger = logging.getLogger(__name__)

# Precompiled once: _expansion_content_words and _phrase_in_text run this
# against every candidate's title and summary, so re-parsing the pattern per
# call showed up.
_WORD_RE = re.compile(r"[A-Za-z0-9]+")


class FieldMatchQuery(NamedTuple):
    """One title/summary check to run, and how strictly to run it.

    Replaces a bare (str, str, str) tuple — three same-typed positions were
    otherwise indistinguishable at every call site, forcing the reader to
    remember or re-check which position meant what.

    scroll_text: what's sent to Qdrant's MatchText prefilter to find candidate
        documents. Differs from match_text only for expansions — see
        _build_field_match_queries.
    match_text: what's actually compared against a document's field value,
        via _classify_field_match.
    rule: which comparison to use ("substring" | "word" | "phrase"), see
        _classify_field_match.
    """
    scroll_text: str
    match_text: str
    rule: str


class AcronymRankingMixin:
    @staticmethod
    def _acronym_substitution_pattern(acronyms: Dict[str, List[str]]) -> re.Pattern:
        """One pattern matching every detected acronym in the query text, in the
        same shapes detect_acronyms accepts.

        `\\bDIET\\b` only ever matched a bare, unpunctuated DIET. Detection, by
        contrast, strips non-letters before looking a token up — so "D.I.E.T."
        and "PTM2024" were detected, reported back to the caller in
        acronym_info, and used for tiering and the sparse query, while the DENSE
        expansion was silently never built, because the substitution step could
        not find the acronym again in the user's text.

        Each acronym is therefore matched letter by letter with a separator
        between them that is EITHER optional non-alphanumeric punctuation
        (dotted initials) OR whitespace:

            D (?:[^A-Za-z0-9\\s]*|\\s+) I (?:[^A-Za-z0-9\\s]*|\\s+) E ... T

        The whitespace option exists because detect_acronyms() now recovers
        exactly this shape: a caller that replaces every "." with a space
        (the commons media API does) turns "D.I.E.T." into "d i e t" — four
        one-letter words detection glues back together. Substitution has to
        be able to find that same run in the text, or detection succeeds
        while the dense expansion silently never gets built (the original
        form of this bug, for "D.I.E.T." before the punctuation-replacement
        behaviour was even known about). This does NOT reopen "D I E T"
        matching arbitrary prose: the two-letters-of-a-real-word case
        (letters/digits between the acronym's letters) is still excluded by
        construction — only actual whitespace-separated single characters
        can take this path, exactly the shape detection itself requires
        before it will glue a run together.

        A multi-word acronym (detect_acronyms now also detects phrases, e.g.
        "RTE ACT") is a space-separated sequence of such words, not one long
        run of letters — the space in the key marks a required word boundary,
        not a letter to be matched with the same dotted-initials tolerance as
        the letters around it. So each word of the acronym gets its own
        letter-by-letter pattern as above, and consecutive words are joined by
        `[^A-Za-z0-9\\s]*\\s+[^A-Za-z0-9\\s]*` — whitespace is still mandatory
        (detection only ever bridges words across whitespace, splitting on
        it), but optional non-alphanumeric punctuation is now tolerated on
        either side of it, so a dotted abbreviation followed by a space still
        matches its phrase ("R.T.E. Act" -> "RTE ACT", not just "RTE").

        The guards keep the old whole-word safety, with one deliberate
        loosening:
          - `(?<![A-Za-z])` / `(?![A-Za-z])` mean "dietary" is still rejected —
            a real word with extra LETTERS glued on never matches.
          - an optional trailing "S" (`S?`, case-insensitive with the rest of
            the pattern) is now allowed right before those digits — detection
            recovers a plural the same way ("DIETS" retries "DIET"), so
            substitution has to accept it too, or detection succeeds while
            the expansion silently never gets built. "diets" now matches;
            "dietary" still doesn't, because the boundary check runs on
            whatever follows the optional S, and "ary" is still letters.
          - trailing digits are captured, not consumed, so "ptm2024" can expand
            without losing the 2024 (see the replacement callable).

        Longest acronym first, so a longer key is preferred over a shorter one
        that prefixes it (PMS before PM, or "RTE ACT" before "RTE") — Python
        alternation is leftmost-first, not longest-match.
        """
        def word_pattern(word: str) -> str:
            return r"(?:[^A-Za-z0-9\s]*|\s+)".join(re.escape(char) for char in word)

        alternatives = [
            r"[^A-Za-z0-9\s]*\s+[^A-Za-z0-9\s]*".join(word_pattern(word) for word in acronym.split())
            for acronym in sorted(acronyms, key=len, reverse=True)
        ]
        return re.compile(
            r"(?<![A-Za-z])(?:" + "|".join(alternatives) + r")S?(?P<digits>\d*)(?![A-Za-z])",
            re.IGNORECASE,
        )

    def _build_field_match_queries(
        self,
        literal_query: str,
        acronyms_detected: Optional[Dict[str, List[str]]],
    ) -> List[FieldMatchQuery]:
        """Build the FieldMatchQuery entries that the title and summary match
        passes work from.

        Each entry carries its OWN rule, because the boost/injection path and
        the tiering path used to disagree about what "matches" means and that
        disagreement ran both ways: substring matching boosted "Dietary
        Guidelines" for DIET and "Basic Maths" for BA (which tiering correctly
        refused), while a title that genuinely IS the expansion — pluralised, or
        with different connectives — earned tier 2 and yet got no boost and was
        never injected. The rules here are the ones _acronym_bonus applies, so
        the two agree by construction:

          - "substring": the pre-acronym behaviour, used for the literal query
            whenever NO acronym was detected. Untouched on purpose — it is what
            keeps ordinary queries byte-identical with the feature on or off.
          - "word": whole-word, the rule the acronym bonus grades use.
          - "phrase": all content words present, the rule the expansion bonus
            grades use.

        scroll_text differs from match_text only for expansions: the Qdrant
        MatchText prefilter is handed the expansion's CONTENT WORDS rather than
        the full phrase. Sending the whole phrase made the filter demand "of"
        and "and" as tokens, so "District Institute for Education & Training"
        was never even retrieved to be classified — which is the real reason
        valid expansion variants could take tier 2 but never be injected.
        """
        queries: List[FieldMatchQuery] = []
        literal = (literal_query or "").strip()

        if not acronyms_detected:
            if literal:
                queries.append(FieldMatchQuery(literal, literal, "substring"))
            return queries

        if literal:
            queries.append(FieldMatchQuery(literal, literal, "word"))

        for acronym, expansions in acronyms_detected.items():
            # The acronym on its own, matching how tiers 4/3 judge it. Without
            # this, the query "DIET handbook guidelines" would never boost a
            # document titled "DIET Handbook", because only the full literal
            # string was ever checked — while tiering gave it tier 4.
            queries.append(FieldMatchQuery(acronym, acronym, "word"))
            for expansion in expansions:
                content = " ".join(self._expansion_content_words(expansion))
                if content:
                    queries.append(FieldMatchQuery(content, expansion, "phrase"))

        return queries

    @staticmethod
    def _normalize_field_match_queries(
        queries: List[FieldMatchQuery],
    ) -> List[FieldMatchQuery]:
        """Lowercase, trim and dedupe the match queries, preserving order.

        Deduped on (match_text, rule) rather than text alone: the same string
        can legitimately appear under two rules — a single-word query like
        "DIET" is both the literal query and the detected acronym — and one
        entry is enough, but a text under a different rule is a different test.
        """
        normalized: List[FieldMatchQuery] = []
        seen: Set[Tuple[str, str]] = set()
        for scroll_text, match_text, rule in queries:
            scroll_lower = (scroll_text or "").strip().lower()
            match_lower = (match_text or "").strip().lower()
            if not scroll_lower or not match_lower:
                continue
            key = (match_lower, rule)
            if key in seen:
                continue
            seen.add(key)
            normalized.append(FieldMatchQuery(scroll_lower, match_lower, rule))
        return normalized

    def _classify_field_match(
        self, match_text: str, rule: str, field_lower: str
    ) -> Optional[str]:
        """Classify one (text, rule) pair against a lowercased field value.

        'exact' still means the field IS the text, whatever the rule, so the
        exact-vs-partial boost split is unchanged. Everything else is 'partial'
        or None, keeping the response contract at {exact, partial, None}.
        """
        # The untouched pre-acronym path, delegated so there is exactly one
        # definition of the old behaviour.
        if rule == "substring":
            return self._classify_text_match(match_text, field_lower)

        if not field_lower:
            return None
        if field_lower == match_text:
            return "exact"
        if rule == "word":
            return "partial" if self._term_in_text(match_text, field_lower) else None
        return "partial" if self._phrase_in_text(match_text, field_lower) else None

    @staticmethod
    def _term_in_text(term: str, text: Optional[str]) -> bool:
        """Whole-word (not substring) match — 'DIET' must not match inside 'dietary'.

        Uses letter/digit lookarounds rather than \\b: \\b treats underscore as a
        word character, so it would miss 'DIET' in filename-style titles like
        '..._DIET Empowerment...' where underscores are used as word separators
        (verified live — several real titles were silently excluded this way).
        """
        if not term or not text:
            return False
        pattern = rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])"
        return re.search(pattern, text, re.IGNORECASE) is not None

    @staticmethod
    @lru_cache(maxsize=2048)
    def _expansion_content_words(phrase: str) -> Tuple[str, ...]:
        """Alphanumeric words of `phrase` (an acronym EXPANSION), lowercased,
        minus English stopwords. The phrase side of _phrase_in_text's match —
        see _document_words for the document side.

        Stopwords come from the same spacy list acronym detection already uses,
        so "of"/"and"/"for"/"the" never decide whether an expansion matches.

        Cached: the expansions are fixed for the whole request (and in practice
        across requests — the dictionary changes rarely), but this used to be
        recomputed inside every candidate's tier call, for every expansion.
        Returns a TUPLE rather than a list precisely because the result is now
        shared between callers — handing out a mutable cached object invites a
        caller to corrupt every later hit.
        """
        words = _WORD_RE.findall(phrase.lower())
        return tuple(w for w in words if w not in STOP_WORDS)

    @staticmethod
    @lru_cache(maxsize=2048)
    def _document_words(text: str) -> Tuple[str, ...]:
        """Alphanumeric words of `text` (a document's title/summary),
        lowercased — the document side of _phrase_in_text's match, cached the
        same way _expansion_content_words already caches the phrase side.

        A title/summary is checked once per expansion of every detected
        acronym (_acronym_bonus loops both), so the same document text was
        being re-split from scratch on every one of those checks — for a
        query naming two acronyms, that's every title tokenized twice for no
        reason, not just for the rare double-meaning entries (SSC/DM/MIP).
        Stopwords are deliberately NOT filtered here (unlike
        _expansion_content_words): _words_match only ever matches against
        already-stopword-filtered terms, so a stopword surviving on this side
        is inert, not wrong.
        """
        return tuple(_WORD_RE.findall(text.lower()))

    def _phrase_in_text(self, phrase: str, text: Optional[str]) -> bool:
        """True when every content word of `phrase` appears in `text`.

        Used for the multi-word acronym EXPANSION tiers, where _term_in_text's
        exact-substring rule is far too rigid: a real title never reproduces an
        expansion verbatim. "District Institute of Education and Training" has
        to match "Strengthening of District Institutes of Education and
        Training" (plural) and "District Institute for Education & Training"
        (different connectives) — both of which the literal matcher rejects,
        which is why the expansion tiers previously never fired at all.

        Word order and connectives are ignored; inflections are tolerated by
        accepting a prefix relationship in either direction ("institute" ~
        "institutes", "education" ~ "educational").

        ALL content words are required, deliberately. A partial threshold (say
        70%) would promote near-misses that merely share common words —
        "District Primary Education Programme" shares district+education with
        the DIET expansion but is a different programme entirely. That is the
        same false-positive failure mode an OR-bag of expansion words has as a
        BM25 signal, so this errs strictly toward under-promoting.

        The single-token acronym itself still goes through _term_in_text: 'DIET'
        must not match 'dietary', and prefix tolerance would allow exactly that.
        """
        if not phrase or not text:
            return False
        terms = self._expansion_content_words(phrase)
        if not terms:
            return False
        words = self._document_words(text)
        if not words:
            return False
        return all(
            any(self._words_match(word, term) for word in words)
            for term in terms
        )

    @staticmethod
    def _words_match(word: str, term: str) -> bool:
        """One document word against one expansion content word.

        Exact match always counts. A prefix relationship in either direction
        counts only when the shorter string clears
        settings.ACRONYM_MIN_PREFIX_MATCH_LEN (stops initialisms and stray
        characters matching the words they appear to abbreviate) AND the
        longer string doesn't exceed it by more than
        settings.ACRONYM_PREFIX_SUFFIX_CAP extra characters. That second check
        is what stops "test" from counting "testimony" as a match while still
        allowing "tests"/"testing"/"tested" — real inflections add a few
        characters, unrelated words that share a prefix tend to add many more.
        """
        if word == term:
            return True
        shorter, longer = (word, term) if len(word) < len(term) else (term, word)
        if len(shorter) < settings.ACRONYM_MIN_PREFIX_MATCH_LEN:
            return False
        if len(longer) - len(shorter) > settings.ACRONYM_PREFIX_SUFFIX_CAP:
            return False
        return longer.startswith(shorter)

    @staticmethod
    def _body_matched(field_scores: Optional[Dict[str, Any]]) -> bool:
        """FALLBACK body check, used only when BM25 can't answer (see
        _sources_with_acronym_in_body, which is the real gate).

        Reads whether the chunk carries a dense `text` similarity at all. That
        is pool membership, not relevance: the pool scales with top_k, and
        MiniLM cosines are effectively never <= 0, so `> 0` never rejects
        anything. Kept only so a missing BM25 index degrades to the previous
        behaviour instead of demoting every acronym document at once.
        """
        if not field_scores:
            return False
        score = field_scores.get("text")
        return score is not None and score > 0

    def _sources_claiming_acronym(
        self,
        rows: List[Dict[str, Any]],
        acronyms_detected: Dict[str, List[str]],
    ) -> Dict[str, Set[str]]:
        """Group the sources whose title or summary carries an acronym, by acronym.

        Those are the only sources that could earn an acronym bonus, and the
        acronym present is the claim the bonus would rest on — so that is the
        acronym their content is then checked for. Evaluated once per source:
        every chunk of a source carries the same title and summary.
        """
        sources_by_acronym: Dict[str, Set[str]] = {}
        seen: Set[str] = set()
        for r in rows:
            source_id = r['payload'].get('source_id')
            if source_id is None or str(source_id) in seen:
                continue
            seen.add(str(source_id))
            title, summary = r['payload'].get('title'), r['payload'].get('summary')
            for acronym in acronyms_detected:
                if self._term_in_text(acronym, title) or self._term_in_text(acronym, summary):
                    sources_by_acronym.setdefault(acronym, set()).add(str(source_id))
        return sources_by_acronym

    def _sources_with_acronym_in_body(
        self, sources_by_acronym: Dict[str, Set[str]]
    ) -> Optional[Dict[str, Set[str]]]:
        """Which of the candidate sources mention THEIR OWN acronym in the body,
        kept separate per acronym.

        One BM25 query per acronym for the bare acronym alone — not the
        expansion bag the retrieval query uses, which matches any document
        containing "education" — restricted to exactly the candidate sources and
        grouped by source_id, so each source answers once however many of its
        chunks match. Restricting to the candidates is what makes this
        independent of top_k: the answer depends on the document, never on how
        large a retrieval pool happened to be drawn. Checked per SOURCE, across
        all of its chunks, including chunks that never made the pool.

        Returned per-acronym rather than as one flat set: a query naming several
        acronyms (e.g. "DIET SMC") must not let a document's body evidence for
        SMC count as evidence for a DIET title claim on the same document —
        each acronym's set here only ever contains sources verified for THAT
        acronym specifically.

        Returns None when BM25 can't answer (sparse search disabled, the encoder
        unavailable, or the query failed) so the caller falls back rather than
        treating "couldn't check" as "checked and absent". About 12 ms per
        acronym on the local corpus.
        """
        if not sources_by_acronym:
            return {}
        if not settings.SPARSE_SEARCH_ENABLED:
            return None
        try:
            from app.core.clients.sparse_encoder import generate_sparse_vector

            backed: Dict[str, Set[str]] = {}
            for acronym, sources in sources_by_acronym.items():
                indices, values = generate_sparse_vector(acronym)
                if not indices:
                    continue
                response = qdrant_client.query_points_groups(
                    collection_name=self.collection_name,
                    query=models.SparseVector(indices=indices, values=values),
                    using=settings.SPARSE_VECTOR_NAME,
                    query_filter=models.Filter(must=[models.FieldCondition(
                        key="source_id", match=models.MatchAny(any=sorted(sources)),
                    )]),
                    group_by="source_id",
                    group_size=1,
                    limit=len(sources),
                    with_payload=False,
                )
                backed[acronym] = {str(group.id) for group in response.groups}
            return backed
        except Exception as exc:
            logger.warning(
                f"Acronym body check failed, falling back to pool membership: {exc}"
            )
            return None

    def _acronym_bonus(
        self,
        title: Optional[str],
        summary: Optional[str],
        acronyms_detected: Dict[str, List[str]],
        backed_acronyms: Set[str],
    ) -> float:
        """The proportional bonus a document earns: the best that applies of

            acronym in title       ACRONYM_BONUS_TITLE_ACRONYM      (0.20)
            expansion in title     ACRONYM_BONUS_TITLE_EXPANSION    (0.15)
            acronym in summary     ACRONYM_BONUS_SUMMARY_ACRONYM    (0.10)
            expansion in summary   ACRONYM_BONUS_SUMMARY_EXPANSION  (0.05)

        applied as relevance x (1 + bonus). The four grades are the old tiers
        4/2/3/1, but a grade now scales a document's relevance instead of
        placing it in a band above everything lower — so a title can close at
        most a (bonus) relative gap and never lifts a weak document over a much
        stronger one.

        The two acronym grades also require backed_acronyms: a title or summary
        is a claim about the topic, not evidence of it (the corpus holds
        documents titled "DIET Reference Handbook" whose content is coastal
        navigation). Expansion grades don't — spelling the expansion out is
        itself the evidence. Every grade is evaluated rather than stopping at
        the first match, so an unbacked acronym in the title still leaves the
        document eligible for its expansion grade.

        backed_acronyms is per-DOCUMENT, not a single flag: it's the subset of
        acronyms_detected whose body content was actually verified for THIS
        document. A query naming several acronyms (e.g. "DIET SMC") must not
        let body evidence for one acronym count as evidence for another — a
        document titled "DIET Handbook" whose body only ever says "SMC" earns
        the DIET title grade only if "DIET" is itself in backed_acronyms, not
        because SOME acronym's evidence was found somewhere in the document.

        Several detected acronyms: each gets its OWN best grade (computed
        independently, exactly as a single acronym would be), and those are
        SUMMED rather than maxed, capped at ACRONYM_BONUS_MULTI_MATCH_CAP — a
        document matching both SMC and DIET in its title now visibly outranks
        one matching only DIET, rather than being identical to it, while the
        cap keeps a query naming several acronyms from letting the multiplier
        run away unbounded.

        The acronym is matched literally (_term_in_text: "DIET" never matches
        "dietary"); expansions on content words (_phrase_in_text), because real
        titles pluralise and swap connectives ("District Institutes of
        Education & Training") and literal expansion matching never fired.
        """
        title_acronym = settings.ACRONYM_BONUS_TITLE_ACRONYM
        title_expansion = settings.ACRONYM_BONUS_TITLE_EXPANSION
        summary_acronym = settings.ACRONYM_BONUS_SUMMARY_ACRONYM
        summary_expansion = settings.ACRONYM_BONUS_SUMMARY_EXPANSION
        total = 0.0
        for acronym, expansions in acronyms_detected.items():
            content_backs_acronym = acronym in backed_acronyms
            best = 0.0
            if content_backs_acronym and self._term_in_text(acronym, title):
                best = max(best, title_acronym)
            # The expansion checks tokenize the whole title/summary, so they
            # only run when they could still raise THIS acronym's best grade.
            if best < title_expansion and any(
                self._phrase_in_text(exp, title) for exp in expansions
            ):
                best = max(best, title_expansion)
            if content_backs_acronym and best < summary_acronym and self._term_in_text(acronym, summary):
                best = max(best, summary_acronym)
            if best < summary_expansion and any(
                self._phrase_in_text(exp, summary) for exp in expansions
            ):
                best = max(best, summary_expansion)
            total += best
        return min(total, settings.ACRONYM_BONUS_MULTI_MATCH_CAP)

    def _assign_acronym_bonuses(
        self,
        rows: List[Dict[str, Any]],
        acronyms_detected: Dict[str, List[str]],
    ) -> None:
        """Set r['acronym_bonus'] on every row.

        Content backing comes from one BM25 check per acronym over exactly the
        candidate sources (_sources_with_acronym_in_body), kept separate per
        acronym so a query naming several acronyms can't let one acronym's body
        evidence count as backing for another. When BM25 can't answer, it falls
        back to the old pool-membership check per chunk rather than withholding
        every acronym bonus at once — that fallback is a single document-wide
        signal (no per-acronym distinction is possible from it), so it applies
        uniformly to every detected acronym, same as before.

        Memoized per source when BM25 answered — the bonus depends only on
        title, summary and the source's content, which every chunk shares, and
        the ranked pool is chunks (~2500 over ~590 sources on a live acronym
        query), so recomputing per chunk was most of the cost.
        """
        body_sources = None
        # A BM25 index that exists in config but holds no vectors would answer
        # "no" for every document. Retrieval has just run a BM25 query including
        # the acronym, so if nothing in the pool scored on it, treat BM25 as
        # unavailable for this request.
        if any((r.get('field_scores') or {}).get(settings.SPARSE_VECTOR_NAME) for r in rows):
            body_sources = self._sources_with_acronym_in_body(
                self._sources_claiming_acronym(rows, acronyms_detected)
            )
        by_source: Dict[str, float] = {}
        for r in rows:
            payload = r['payload']
            source_id = payload.get('source_id')
            key = str(source_id) if source_id is not None else None
            if body_sources is not None and key is not None:
                if key not in by_source:
                    backed_acronyms = {
                        acronym for acronym, sources in body_sources.items()
                        if key in sources
                    }
                    by_source[key] = self._acronym_bonus(
                        payload.get('title'), payload.get('summary'),
                        acronyms_detected, backed_acronyms,
                    )
                r['acronym_bonus'] = by_source[key]
            else:
                # Uniform fallback, applied to every acronym alike — see docstring.
                backed_acronyms = (
                    set(acronyms_detected) if self._body_matched(r.get('field_scores')) else set()
                )
                r['acronym_bonus'] = self._acronym_bonus(
                    payload.get('title'), payload.get('summary'),
                    acronyms_detected, backed_acronyms,
                )

    def _blended_acronym_relevance(
        self,
        pool_ids: List[Any],
        search_fields: List[str],
        weights: Dict[str, float],
        query_text: str,
        query_embedding: Any,
        expansion_text: str,
        expansion_embedding: Any,
        normalization_reference: Optional[Dict[str, float]] = None,
    ) -> Optional[Dict[Any, float]]:
        """Relevance for an acronym query:
            (1 - W) x score_against_query + W x score_against_expansion
        with W = ACRONYM_EXPANSION_SCORE_WEIGHT, each side the usual 70/30 fusion.

        Retrieval uses both texts and merges per field with max(), which is right
        for FINDING documents but wrong for SCORING them: an expansion made of
        generic words ("District Institute of Education and Training") scores
        any district/education document highly — measured, "District Primary
        Education Programme" goes from 0.28 against "DIET" to 0.89 with the
        expansion, above every real DIET document — while a bare acronym
        ("ptm") carries almost no meaning on its own. Averaging needs both
        signals to agree, which damps whichever one is noisy for this acronym.

        Rescores pool_ids (HasIdCondition), once per text, so every candidate
        passed in is measured against both. The caller may pass a subset of the
        full retrieval pool (see ACRONYM_RESCORE_POOL_LIMIT) rather than every
        candidate -- ids left out simply keep their retrieval score. Returns
        None — callers keep the retrieval scores — when the pool is empty or
        rescoring fails.

        normalization_reference: forwarded to _rank_results (see its docstring).
        When pool_ids is a capped SUBSET of a larger retrieval pool, this should
        be the full pool's own {dense_min, dense_max, sparse_min, sparse_max} —
        without it, min-max normalizing the subset on its own gives it a
        different scale than the untouched remainder that kept its original
        retrieval score, which corrupts both ordering and the filter_score
        threshold across the cutoff boundary. None when pool_ids already IS the
        full pool (nothing was excluded, so self-normalizing is already correct).
        """
        if not pool_ids:
            return None
        only_pool = models.Filter(must=[models.HasIdCondition(has_id=list(pool_ids))])

        def score(text: str, vector: Any) -> Dict[Any, float]:
            if settings.SPARSE_SEARCH_ENABLED:
                results, scores, sparse_issued = self._hybrid_batch_search(
                    search_fields=search_fields, query_text=text,
                    query_embeddings=[vector], filter_conditions=only_pool,
                    limit=len(pool_ids),
                )
            else:
                results, scores = self._parallel_batch_search(
                    search_fields=search_fields, weights=weights,
                    query_embeddings=[vector], filter_conditions=only_pool,
                    limit=len(pool_ids),
                )
                sparse_issued = False
            return {r['id']: r['weighted_score']
                    for r in self._rank_results(
                        results, scores, weights, search_fields,
                        sparse_issued=sparse_issued,
                        normalization_reference=normalization_reference,
                    )}

        try:
            by_query = score(query_text, query_embedding)
            by_expansion = score(expansion_text, expansion_embedding)
        except Exception as exc:
            logger.warning(f"Acronym relevance rescoring failed, keeping retrieval scores: {exc}")
            return None

        w = settings.ACRONYM_EXPANSION_SCORE_WEIGHT
        return {
            pid: (1.0 - w) * by_query[pid] + w * by_expansion[pid]
            for pid in pool_ids
            if pid in by_query and pid in by_expansion
        }
