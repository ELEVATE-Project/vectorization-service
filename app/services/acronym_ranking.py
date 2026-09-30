"""Acronym-aware matching and scoring, mixed into PrioritizedSearchService.

Relies on the core service's search and rank methods. Design: release-doc/acronym-design-notes.md.
"""
import logging
import re
from functools import lru_cache
from typing import Any, Dict, List, NamedTuple, Optional, Set, Tuple

from qdrant_client import models
# Same stopword list detection uses, so both agree on what a content word is.
from spacy.lang.en.stop_words import STOP_WORDS

from app.config import settings
from app.constants import ACRONYM_USE_PREFIX, ACRONYM_USE_SUFFIX, WORD_TOKEN_PATTERN
from app.core.clients.qdrant import qdrant_client

logger = logging.getLogger(__name__)

# Compiled once: runs against every candidate's title and summary.
_WORD_RE = re.compile(WORD_TOKEN_PATTERN)


@lru_cache(maxsize=1024)
def _acronym_use_pattern(acronym: str) -> "re.Pattern[str]":
    """Pattern for an acronym written as the acronym: "DIET", "DIETs", "PTM2024".

    Single-word keys are case-sensitive ("diet" is the everyday word); multi-word keys match any case.
    """
    words = r"\s+".join(re.escape(w) for w in acronym.split())
    flags = re.IGNORECASE if " " in acronym.strip() else 0
    return re.compile(f"{ACRONYM_USE_PREFIX}{words}{ACRONYM_USE_SUFFIX}", flags)


class FieldMatchQuery(NamedTuple):
    """One title/summary check: scroll_text (Qdrant prefilter), match_text (compared
    with the field) and rule ("substring" | "word" | "phrase").
    """
    scroll_text: str
    match_text: str
    rule: str


class AcronymRankingMixin:
    @staticmethod
    def _acronym_substitution_pattern(acronyms: Dict[str, List[str]]) -> re.Pattern:
        """One pattern finding every detected acronym in the query text, in the shapes
        detection accepts ("D.I.E.T.", "d i e t", "R.T.E. Act", "DIETS", "ptm2024").
        Longest key first; trailing digits are captured. See design notes, Query substitution.
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
        """Title/summary checks for the match passes, each with its own rule.

        "substring" when no acronym is detected; else "word" for query and acronym, "phrase" for expansions.
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
            # The acronym alone, so "DIET handbook guidelines" still matches a
            # "DIET Handbook" title.
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
        """Lowercase, trim and dedupe on (match_text, rule), keeping order."""
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
        """'exact' if the field is the text, 'partial' if it matches under `rule`, else None."""
        # Pre-acronym behaviour, defined in one place.
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
        """Whole-word, case-insensitive match: "DIET" never matches inside "dietary".

        Letter/digit lookarounds, not \\b, so "_DIET_" in file-name titles still matches.
        """
        if not term or not text:
            return False
        pattern = rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])"
        return re.search(pattern, text, re.IGNORECASE) is not None

    @staticmethod
    @lru_cache(maxsize=2048)
    def _expansion_content_words(phrase: str) -> Tuple[str, ...]:
        """Lowercased words of an expansion minus stopwords (the list detection uses).

        Cached, and a tuple so callers cannot change the shared result.
        """
        words = _WORD_RE.findall(phrase.lower())
        return tuple(w for w in words if w not in STOP_WORDS)

    @staticmethod
    @lru_cache(maxsize=2048)
    def _document_words(text: str) -> Tuple[str, ...]:
        """Lowercased words of a title/summary, cached.

        Stopwords are kept: they never match a stopword-free expansion word.
        """
        return tuple(_WORD_RE.findall(text.lower()))

    def _phrase_in_text(self, phrase: str, text: Optional[str]) -> bool:
        """True when every content word of `phrase` is in `text`, in any order.

        Inflections allowed ("institutes"); all words required, so near-misses fail.
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
        """Exact match, or a prefix match within ACRONYM_MIN_PREFIX_MATCH_LEN and
        ACRONYM_PREFIX_SUFFIX_CAP ("tests" ~ "test", but not "testimony").
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
        """Fallback body check when BM25 cannot answer: any dense `text` score counts.

        Weak on purpose; it only stops every acronym document being demoted without BM25.
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
        """{acronym: sources whose title or summary contains it}, one check per source."""
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
        self,
        sources_by_acronym: Dict[str, Set[str]],
        acronyms_detected: Optional[Dict[str, List[str]]] = None,
        backing_out: Optional[Dict[str, Dict[str, str]]] = None,
    ) -> Optional[Dict[str, Set[str]]]:
        """{acronym: sources whose body backs it}: the acronym in capitals, or its expansion.

        backing_out, if given, gets {acronym: {source: "acronym" | "expansion"}}.
        Returns None when BM25 cannot answer, so the caller falls back. See design notes, Body check.
        """
        if not sources_by_acronym:
            return {}
        if not settings.SPARSE_SEARCH_ENABLED:
            return None
        try:
            from app.core.clients.sparse_encoder import generate_sparse_vector

            backed: Dict[str, Set[str]] = {}
            for acronym, sources in sources_by_acronym.items():
                found: Set[str] = set()
                indices, values = generate_sparse_vector(acronym)
                if indices:
                    # BM25 folds case ("diet" = "DIET"), so it only narrows to the
                    # top chunks; one must use the acronym as written (capitals).
                    response = qdrant_client.query_points_groups(
                        collection_name=self.collection_name,
                        query=models.SparseVector(indices=indices, values=values),
                        using=settings.SPARSE_VECTOR_NAME,
                        query_filter=models.Filter(must=[models.FieldCondition(
                            key="source_id", match=models.MatchAny(any=sorted(sources)),
                        )]),
                        group_by="source_id",
                        group_size=settings.ACRONYM_BODY_CHECK_TOP_CHUNKS,
                        limit=len(sources),
                        with_payload=["text"],
                    )
                    pattern = _acronym_use_pattern(acronym)
                    found = {
                        str(group.id) for group in response.groups
                        if any(
                            pattern.search((hit.payload or {}).get("text") or "")
                            for hit in group.hits
                        )
                    }
                    # Top chunks rank by word count, not case, so read every chunk
                    # of a rejected BM25 hit before deciding.
                    rejected = {str(group.id) for group in response.groups} - found
                    if rejected:
                        found |= self._sources_using_acronym_in_any_chunk(rejected, pattern)
                if backing_out is not None:
                    backing_out[acronym] = {source: "acronym" for source in found}
                # Expansion check only for sources the acronym did not back.
                unbacked = sources - found
                if unbacked and acronyms_detected and acronyms_detected.get(acronym):
                    by_expansion = self._sources_with_expansion_in_body(
                        unbacked, acronyms_detected[acronym]
                    )
                    found |= by_expansion
                    if backing_out is not None:
                        backing_out[acronym].update({source: "expansion" for source in by_expansion})
                backed[acronym] = found
            return backed
        except Exception as exc:
            logger.warning(
                f"Acronym body check failed, falling back to pool membership: {exc}"
            )
            return None

    def _sources_using_acronym_in_any_chunk(
        self, sources: Set[str], pattern: "re.Pattern[str]"
    ) -> Set[str]:
        """Sources with any chunk matching `pattern`, reading all their chunks.

        Raises on a Qdrant failure; the caller treats that as a failed body check.
        """
        found: Set[str] = set()
        offset = None
        while True:
            points, offset = qdrant_client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(must=[models.FieldCondition(
                    key="source_id", match=models.MatchAny(any=sorted(sources - found)),
                )]),
                limit=256,
                offset=offset,
                with_payload=["source_id", "text"],
                with_vectors=False,
            )
            for point in points:
                payload = point.payload or {}
                if pattern.search(payload.get("text") or ""):
                    found.add(str(payload.get("source_id")))
            if offset is None or found == sources:
                return found

    def _sources_with_expansion_in_body(
        self, sources: Set[str], expansions: List[str]
    ) -> Set[str]:
        """Sources whose top BM25 chunks spell out one of `expansions` (_phrase_in_text).

        Raises on failure; the caller treats that as a failed body check.
        """
        from app.core.clients.sparse_encoder import generate_sparse_vector

        found: Set[str] = set()
        for expansion in expansions:
            remaining = sources - found
            if not remaining:
                break
            indices, values = generate_sparse_vector(
                " ".join(self._expansion_content_words(expansion))
            )
            if not indices:
                continue
            response = qdrant_client.query_points_groups(
                collection_name=self.collection_name,
                query=models.SparseVector(indices=indices, values=values),
                using=settings.SPARSE_VECTOR_NAME,
                query_filter=models.Filter(must=[models.FieldCondition(
                    key="source_id", match=models.MatchAny(any=sorted(remaining)),
                )]),
                group_by="source_id",
                group_size=settings.ACRONYM_BODY_CHECK_TOP_CHUNKS,
                limit=len(remaining),
                with_payload=["text"],
            )
            for group in response.groups:
                if any(
                    self._phrase_in_text(expansion, (hit.payload or {}).get("text"))
                    for hit in group.hits
                ):
                    found.add(str(group.id))
        return found

    def _acronym_bonus(
        self,
        title: Optional[str],
        summary: Optional[str],
        acronyms_detected: Dict[str, List[str]],
        backed_acronyms: Set[str],
    ) -> float:
        """Bonus for relevance x (1 + bonus): best title/summary grade per acronym, summed, capped.

        Acronym grades need body backing for that acronym. See design notes, Bonus.
        """
        return self._acronym_bonus_detail(title, summary, acronyms_detected, backed_acronyms)[0]

    def _acronym_bonus_detail(
        self,
        title: Optional[str],
        summary: Optional[str],
        acronyms_detected: Dict[str, List[str]],
        backed_acronyms: Set[str],
    ) -> Tuple[float, Dict[str, Optional[str]]]:
        """(bonus, {acronym: grade earned}); grade is "title_acronym", "title_expansion",
        "summary_acronym", "summary_expansion" or None."""
        grades = (
            ("title_acronym", settings.ACRONYM_BONUS_TITLE_ACRONYM),
            ("title_expansion", settings.ACRONYM_BONUS_TITLE_EXPANSION),
            ("summary_acronym", settings.ACRONYM_BONUS_SUMMARY_ACRONYM),
            ("summary_expansion", settings.ACRONYM_BONUS_SUMMARY_EXPANSION),
        )
        values = dict(grades)
        total = 0.0
        earned: Dict[str, Optional[str]] = {}
        for acronym, expansions in acronyms_detected.items():
            content_backs_acronym = acronym in backed_acronyms
            best, best_name = 0.0, None

            def consider(name: str) -> None:
                nonlocal best, best_name
                if values[name] > best:
                    best, best_name = values[name], name

            if content_backs_acronym and self._term_in_text(acronym, title):
                consider("title_acronym")
            # Expansion checks tokenize the text; run them only if they can raise the grade.
            if best < values["title_expansion"] and any(
                self._phrase_in_text(exp, title) for exp in expansions
            ):
                consider("title_expansion")
            if (content_backs_acronym and best < values["summary_acronym"]
                    and self._term_in_text(acronym, summary)):
                consider("summary_acronym")
            if best < values["summary_expansion"] and any(
                self._phrase_in_text(exp, summary) for exp in expansions
            ):
                consider("summary_expansion")
            total += best
            earned[acronym] = best_name
        return min(total, settings.ACRONYM_BONUS_MULTI_MATCH_CAP), earned

    @staticmethod
    def _body_backing(
        source_id: str,
        acronyms_detected: Dict[str, List[str]],
        body_sources: Dict[str, Set[str]],
        backing: Dict[str, Dict[str, str]],
    ) -> Dict[str, Optional[str]]:
        """{acronym: "acronym" | "expansion" | None} for one source; body_sources decides."""
        return {
            acronym: (backing.get(acronym, {}).get(source_id, "acronym")
                      if source_id in body_sources.get(acronym, ()) else None)
            for acronym in acronyms_detected
        }

    def _assign_acronym_bonuses(
        self,
        rows: List[Dict[str, Any]],
        acronyms_detected: Dict[str, List[str]],
    ) -> str:
        """Set acronym_bonus, acronym_grades and body_backed on every row, once per source.

        Returns the body check used: "bm25", "dense_fallback" (BM25 unavailable) or
        "none" (no document claims the acronym).
        """
        body_sources = None
        claiming: Dict[str, Set[str]] = {}
        backing: Dict[str, Dict[str, str]] = {}
        # An empty BM25 index would reject every document, so if no pooled row has
        # a sparse score, treat BM25 as unavailable.
        if any((r.get('field_scores') or {}).get(settings.SPARSE_VECTOR_NAME) for r in rows):
            claiming = self._sources_claiming_acronym(rows, acronyms_detected)
            body_sources = self._sources_with_acronym_in_body(
                claiming, acronyms_detected=acronyms_detected, backing_out=backing,
            )
        by_source: Dict[str, Tuple[float, Dict[str, Optional[str]], Dict[str, Optional[str]]]] = {}
        for r in rows:
            payload = r['payload']
            source_id = payload.get('source_id')
            key = str(source_id) if source_id is not None else None
            if body_sources is not None and key is not None:
                if key not in by_source:
                    body_backed = self._body_backing(key, acronyms_detected, body_sources, backing)
                    bonus, grades = self._acronym_bonus_detail(
                        payload.get('title'), payload.get('summary'), acronyms_detected,
                        {acronym for acronym, how in body_backed.items() if how},
                    )
                    by_source[key] = (bonus, grades, body_backed)
                r['acronym_bonus'], r['acronym_grades'], r['body_backed'] = by_source[key]
            else:
                # Uniform fallback, applied to every acronym alike — see docstring.
                matched = self._body_matched(r.get('field_scores'))
                r['body_backed'] = {
                    acronym: ("dense_fallback" if matched else None) for acronym in acronyms_detected
                }
                r['acronym_bonus'], r['acronym_grades'] = self._acronym_bonus_detail(
                    payload.get('title'), payload.get('summary'), acronyms_detected,
                    set(acronyms_detected) if matched else set(),
                )
        if body_sources is None:
            return "dense_fallback"
        return "bm25" if claiming else "none"

    @staticmethod
    def _acronym_ranking_context(
        applied: bool,
        search_mode: str,
        dense_variants: int,
        rescore: Optional[Dict[str, Any]],
        body_check: str,
    ) -> Dict[str, Any]:
        """scoring_context.acronym_ranking: how an acronym query was scored (debug only).

        When ranking is off (semantic mode, hybrid disabled), only retrieval was widened.
        """
        if not applied:
            reason = ("search_mode is semantic" if search_mode == "semantic"
                      else "HYBRID_SEARCH_ENABLED is false")
            return {"applied": False, "reason": reason, "dense_variants": dense_variants}
        return {
            "applied": True,
            "relevance": "(1 - W) x query_score + W x expansion_score",
            "expansion_score_weight": settings.ACRONYM_EXPANSION_SCORE_WEIGHT,
            "final_score": "relevance x (1 + acronym_bonus)",
            "bonus_grades": {
                "title_acronym": settings.ACRONYM_BONUS_TITLE_ACRONYM,
                "title_expansion": settings.ACRONYM_BONUS_TITLE_EXPANSION,
                "summary_acronym": settings.ACRONYM_BONUS_SUMMARY_ACRONYM,
                "summary_expansion": settings.ACRONYM_BONUS_SUMMARY_EXPANSION,
            },
            "bonus_cap": settings.ACRONYM_BONUS_MULTI_MATCH_CAP,
            "dense_variants": dense_variants,
            # None when the query already spelled out the expansion (nothing to blend).
            "rescore": rescore,
            "body_check": body_check,
        }

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
        """Relevance = (1 - W) x score vs query + W x score vs expansion (W: ACRONYM_EXPANSION_SCORE_WEIGHT).

        Pass the full pool's normalization_reference when pool_ids is a capped subset.
        Returns None on failure. See design notes, Relevance blend.
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
