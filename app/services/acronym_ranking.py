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
    with the field) and rule ("substring" | "word").
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

        "substring" when no acronym is detected; else "word" for the query and each acronym.
        Expansions are not matched here: retrieval and the relevance blend cover them.
        """
        queries: List[FieldMatchQuery] = []
        literal = (literal_query or "").strip()

        if not acronyms_detected:
            if literal:
                queries.append(FieldMatchQuery(literal, literal, "substring"))
            return queries

        if literal:
            queries.append(FieldMatchQuery(literal, literal, "word"))

        for acronym in acronyms_detected:
            # The acronym alone, so "DIET handbook guidelines" still matches a
            # "DIET Handbook" title.
            queries.append(FieldMatchQuery(acronym, acronym, "word"))

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
        return "partial" if self._term_in_text(match_text, field_lower) else None

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

    def _sources_with_acronym_in_body(
        self, sources_by_acronym: Dict[str, Set[str]],
        scanned_out: Optional[Set[str]] = None,
    ) -> Optional[Dict[str, Set[str]]]:
        """{acronym: sources whose body uses the acronym as written (capitals)}.

        Chunks without a BM25 vector (indexed before BM25 was enabled) cannot answer the
        BM25 query, so their text is read directly; scanned_out, if given, gets the
        sources that needed it. Returns None when BM25 cannot answer. See design notes,
        Body check.
        """
        if not sources_by_acronym:
            return {}
        if not settings.SPARSE_SEARCH_ENABLED:
            return None
        try:
            from app.core.clients.sparse_encoder import generate_sparse_vector

            unindexed = self._chunks_without_bm25(set().union(*sources_by_acronym.values()))
            if scanned_out is not None:
                scanned_out.update(unindexed)
            backed: Dict[str, Set[str]] = {}
            for acronym, sources in sources_by_acronym.items():
                found: Set[str] = set()
                pattern = _acronym_use_pattern(acronym)
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
                found |= {
                    source for source in sources - found
                    if any(pattern.search(text) for text in unindexed.get(source, ()))
                }
                backed[acronym] = found
            return backed
        except Exception as exc:
            logger.warning(f"Acronym body check failed, skipping the text boost: {exc}")
            return None

    def _chunks_without_bm25(self, sources: Set[str]) -> Dict[str, List[str]]:
        """{source: texts of its chunks that have no BM25 vector}, for the given sources.

        Usually empty. On a Qdrant without has_vector filtering (before 1.13) it is
        empty too, and those chunks stay unchecked rather than failing the check.
        """
        texts: Dict[str, List[str]] = {}
        offset = None
        try:
            while True:
                points, offset = qdrant_client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=models.Filter(
                        must=[models.FieldCondition(
                            key="source_id", match=models.MatchAny(any=sorted(sources)),
                        )],
                        must_not=[models.HasVectorCondition(has_vector=settings.SPARSE_VECTOR_NAME)],
                    ),
                    limit=256,
                    offset=offset,
                    with_payload=["source_id", "text"],
                    with_vectors=False,
                )
                for point in points:
                    payload = point.payload or {}
                    texts.setdefault(str(payload.get("source_id")), []).append(payload.get("text") or "")
                if offset is None:
                    return texts
        except Exception as exc:
            logger.warning(f"Could not list chunks without BM25 vectors, leaving them unchecked: {exc}")
            return {}

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

    def _text_matches_for_acronyms(
        self, source_ids: Set[str], acronyms_detected: Dict[str, List[str]],
        scanned_out: Optional[Set[str]] = None,
    ) -> Optional[Dict[str, str]]:
        """{source: "exact"} for sources whose body uses a detected acronym, for the
        text boost; None when BM25 cannot answer. scanned_out: see
        _sources_with_acronym_in_body."""
        body_sources = self._sources_with_acronym_in_body(
            {acronym: set(source_ids) for acronym in acronyms_detected},
            scanned_out=scanned_out,
        )
        if body_sources is None:
            return None
        return {source: "exact" for sources in body_sources.values() for source in sources}

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
            "pre_boost_score": "(1 - W) x query_score + W x expansion_score",
            "expansion_score_weight": settings.ACRONYM_EXPANSION_SCORE_WEIGHT,
            "final_score": "pre_boost_score x title, summary and text multipliers "
                           "(see boost_config), capped at 1.0",
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
