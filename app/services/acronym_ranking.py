"""Acronym-aware query rewriting and scoring, mixed into PrioritizedSearchService.

Relies on the core service's search and rank methods. Design: release-doc/acronym-design-notes.md.
"""
import logging
import re
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

# Same stopword list detection uses, so both agree on what a content word is.
from spacy.lang.en.stop_words import STOP_WORDS

from app.config import settings
from app.constants import constants as const
from app.constants import messages as msg

logger = logging.getLogger(__name__)

# Compiled once: runs against every expansion and query text.
_WORD_RE = re.compile(const.WORD_TOKEN_PATTERN)


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
    def _acronym_method_description() -> str:
        """acronym_info.method_description: ACRONYM_SEARCH_METHOD (and keyword OR) in words."""
        if settings.ACRONYM_SEARCH_METHOD == "max":
            return msg.ACRONYM_METHOD_MAX
        if settings.ACRONYM_KEYWORD_OR:
            return msg.ACRONYM_METHOD_INLINE + msg.ACRONYM_METHOD_INLINE_KEYWORD_OR
        return msg.ACRONYM_METHOD_INLINE + msg.ACRONYM_METHOD_INLINE_KEYWORD

    @staticmethod
    def _acronym_ranking_context(
        applied: bool,
        search_mode: str,
        dense_variants: int,
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
            "method": settings.ACRONYM_SEARCH_METHOD,
            "final_score": "score before boosts x title and summary multipliers "
                           "(see boost_config), capped at 1.0",
            "dense_variants": dense_variants,
        }
