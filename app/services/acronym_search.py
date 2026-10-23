"""Acronym search: detect acronyms in a query and build the texts it is searched with.

Storage and cache live in acronym_service.py. Design: release-doc/acronym-design-notes.md.
"""
import logging
import re
from dataclasses import dataclass
from functools import lru_cache, partial
from typing import Any, Dict, List, Optional, Tuple

# Same stopword list for detection and phrase matching, so both agree on content words.
from spacy.lang.en.stop_words import STOP_WORDS

from app.config import settings
from app.constants import constants as const
from app.constants import messages as msg
from app.services.acronym_service import get_expansions_batch
from app.utils.query_preprocessor import preprocess_query

logger = logging.getLogger(__name__)

# Letters only: "D.I.E.T." -> "DIET", "PTM2024" -> "PTM" (see acronym-design-notes.md).
_NON_LETTER_RE = re.compile(const.ACRONYM_NON_LETTER_PATTERN)
# Compiled once: runs against every expansion and query text.
_WORD_RE = re.compile(const.WORD_TOKEN_PATTERN)


# =============================================================================
# Detection
# =============================================================================

def _normalize_token(raw_token: str) -> str:
    """Keep only the letters of a word, the shape of an acronym key: "D.I.E.T." -> "DIET",
    "PTM2024" -> "PTM". Detection and substitution both use it, so they agree on the key.
    """
    return _NON_LETTER_RE.sub("", raw_token)


def detect_acronyms(query: str) -> Dict[str, List[str]]:
    """Return {acronym: expansions} for every registered acronym in `query`.

    Checks single words, 2-N word phrases, plurals and glued single letters in one
    batched lookup. Rules: release-doc/acronym-design-notes.md, Detection.
    """
    if not query or not query.strip():
        return {}

    # Only the first ACRONYM_MAX_QUERY_WORDS words, to bound Redis/Postgres lookups.
    raw_tokens = query.strip().split()[:settings.ACRONYM_MAX_QUERY_WORDS]
    is_single_word_query = len(raw_tokens) == 1

    # Deduped before lookup, so a repeated word is looked up once.
    candidates: List[str] = []
    seen = set()
    # Stopwords are still looked up; only registered ones (BE, ME, SO) come back.
    stopword_candidates: List[str] = []
    # Plural -> singular to try when the plural is not found ("DIETS" -> "DIET").
    plural_retry: Dict[str, str] = {}
    for raw_token in raw_tokens:
        normalized = _normalize_token(raw_token)
        if len(normalized) < 2:
            continue

        candidate = normalized.upper()
        if candidate in seen:
            continue
        seen.add(candidate)

        if not is_single_word_query and normalized.lower() in STOP_WORDS:
            stopword_candidates.append(candidate)
        else:
            candidates.append(candidate)

        if candidate.endswith("S") and len(candidate) > 2:
            plural_retry[candidate] = candidate[:-1]

    # Also try groups of 2 to N words next to each other ("RTE ACT"); a group with a
    # word that has no letters ("&", "2024") is skipped, so no gap is ever bridged.
    if len(raw_tokens) > 1:
        normalized_words = [_normalize_token(t) for t in raw_tokens]
        max_window = min(settings.ACRONYM_MAX_PHRASE_WORDS, len(raw_tokens))
        for window in range(2, max_window + 1):
            for start in range(0, len(raw_tokens) - window + 1):
                words = normalized_words[start:start + window]
                if any(len(w) < 1 for w in words):
                    continue
                candidate = " ".join(words).upper()
                if candidate in seen:
                    continue
                seen.add(candidate)
                candidates.append(candidate)

        # Commons turns "D.I.E.T." into "d i e t", so a run of single letters is
        # also tried joined: "DIET".
        i = 0
        while i < len(normalized_words):
            if len(normalized_words[i]) == 1:
                j = i + 1
                while j < len(normalized_words) and len(normalized_words[j]) == 1:
                    j += 1
                if j - i >= 2:
                    glued = "".join(normalized_words[i:j]).upper()
                    if glued not in seen:
                        seen.add(glued)
                        candidates.append(glued)
                i = j
            else:
                i += 1

    # Same batched lookup; the dictionary decides which stopwords are acronyms.
    candidates.extend(stopword_candidates)

    if not candidates:
        return {}

    expansions_by_acronym = get_expansions_batch(candidates)

    # Second lookup only for plurals that missed ("DIETS"); reported under the
    # singular key ("DIET"), the registered acronym.
    retry_targets = {
        singular
        for typed, singular in plural_retry.items()
        if not expansions_by_acronym.get(typed)
    }
    if retry_targets:
        for singular, expansions in get_expansions_batch(list(retry_targets)).items():
            if expansions:
                expansions_by_acronym[singular] = expansions

    # Drop empty expansion lists (the column defaults to []), so search() never
    # sees one.
    return {
        acronym: expansions
        for acronym, expansions in expansions_by_acronym.items()
        if expansions
    }


# =============================================================================
# Query building
# =============================================================================

@dataclass
class AcronymQuery:
    """The texts an acronym query is searched with (built by build_acronym_query)."""
    detected: Dict[str, List[str]]      # acronym -> its meanings
    dense_texts: List[str]              # semantic search texts
    sparse_text: str                    # one BM25 text: the query plus every meaning's words
    sparse_texts: Optional[List[str]]   # keyword OR / dual_query: one BM25 text per meaning
    written_out: List[str]              # the query with each acronym written out (expansion boosts)


def build_acronym_query(original_query: str, cleaned_query: str) -> Optional[AcronymQuery]:
    """Detect acronyms and build the search texts for ACRONYM_SEARCH_METHOD; None if none."""
    detected = detect_acronyms(original_query)
    if not detected:
        logger.info("[ACRONYM] no acronym in the query; normal search")
        return None
    logger.info(f"[ACRONYM] detected: {detected}")

    # Replace all acronyms in one pass, so a meaning that contains another acronym
    # is not replaced a second time.
    pattern = _acronym_substitution_pattern(detected)
    max_expansions = max(len(exps) for exps in detected.values())

    def write_out(text: str, index: int) -> str:
        """`text` with each acronym replaced by its meaning at `index`."""
        return pattern.sub(partial(_expand_match, detected=detected, index=index,
                                   cleaned_query=cleaned_query), text)

    # Semantic texts: the cleaned query, then one per meaning; a text that differs
    # only in case is skipped (the embedding ignores case).
    variants = [cleaned_query]
    for index in range(max_expansions):
        substituted = write_out(cleaned_query, index)
        if substituted.lower() not in {v.lower() for v in variants}:
            variants.append(substituted)
    dense_texts = variants[:max(1, settings.ACRONYM_MAX_DENSE_VARIANTS)]
    logger.info(f"***1***Dense embedding variants (for semantic search): {dense_texts}")

    written_out = [
        text for text in dict.fromkeys(write_out(original_query, i) for i in range(max_expansions))
        if text.strip().lower() != original_query.strip().lower()
    ]

    # Keyword OR: one BM25 search per meaning (not capped); each chunk keeps the higher score.
    sparse_texts = None
    if (settings.ACRONYM_KEYWORD_OR or settings.ACRONYM_SEARCH_METHOD == const.ACRONYM_SEARCH_METHOD_DUAL_QUERY) and len(variants) > 1:
        sparse_texts = list(variants)

    if settings.ACRONYM_SEARCH_METHOD == const.ACRONYM_SEARCH_METHOD_INLINE:
        # Every meaning in one text ("MIP (Micro Improvement Project, Micro Improvement Plan)").
        inline_query = pattern.sub(partial(_inline_meanings, detected=detected,
                                           max_expansions=max_expansions,
                                           cleaned_query=cleaned_query), original_query)
        # Same clean-up as any other query, as if the user had typed it this way.
        dense_texts = [preprocess_query(inline_query).strip() or inline_query]

    # Keyword search counts matching words, so adding the meanings' words only adds matches.
    expansion_words = " ".join(e for exps in detected.values() for e in exps)
    sparse_text = f"{original_query} {expansion_words}"

    logger.info(f"[ACRONYM] written-out queries (expansion boosts): {written_out}")
    return AcronymQuery(detected, dense_texts, sparse_text, sparse_texts, written_out)


def _acronym_substitution_pattern(acronyms: Dict[str, List[str]]) -> re.Pattern:
    """One pattern finding every detected acronym in the query text, in the shapes
    detection accepts ("D.I.E.T.", "d i e t", "R.T.E. Act", "DIETS", "ptm2024").
    Longest key first; digits on either side are captured. See design notes, Building the search texts.
    """
    # With no acronym the pattern would match empty text and bare digits.
    if not acronyms:
        raise ValueError("_acronym_substitution_pattern needs at least one acronym")

    def word_pattern(word: str) -> str:
        return const.ACRONYM_LETTER_GAP_PATTERN.join(re.escape(char) for char in word)

    # One regex per acronym, longest first ("RTE ACT" before "RTE"); joined with "|" below.
    acronym_patterns = [
        const.ACRONYM_WORD_GAP_PATTERN.join(word_pattern(word) for word in acronym.split())
        for acronym in sorted(acronyms, key=len, reverse=True)
    ]
    return re.compile(
        const.ACRONYM_MATCH_TEMPLATE.format(alternatives="|".join(acronym_patterns)),
        re.IGNORECASE,
    )


def _expand_match(match: re.Match, *, detected: Dict[str, List[str]], index: int,
                  cleaned_query: str) -> str:
    """Replace one matched acronym with its meaning number `index` (its last one if it has fewer)."""
    matched = match.group(0)
    # Normalize per word like detection ("d.i.e.t." -> DIET, "rte  act" -> "RTE ACT").
    words = [_normalize_token(w) for w in matched.split()]
    # "d i e t" is registered glued ("DIET"), so try both joins.
    resolved = None
    for key in dict.fromkeys([" ".join(words).upper(), "".join(words).upper()]):
        resolved = detected.get(key)
        if resolved:
            break
        # Plural: try the singular, like detection's retry.
        if key.endswith("S") and len(key) > 2:
            resolved = detected.get(key[:-1])
            if resolved:
                break
    if not resolved:
        return matched
    expansion = resolved[min(index, len(resolved) - 1)]
    # Keep the acronym if the query already spells the meaning out.
    if _phrase_in_text(expansion, cleaned_query):
        return matched
    # Keep digits as their own word: "ptm2024" -> "Parent Teacher Meeting 2024",
    # "123ptm" -> "123 Parent Teacher Meeting".
    return " ".join(part for part in (match.group("lead"), expansion, match.group("digits")) if part)


def _inline_meanings(match: re.Match, *, detected: Dict[str, List[str]],
                     max_expansions: int, cleaned_query: str) -> str:
    """The matched acronym followed by its meanings: "MIP (Micro Improvement Project, ...)"."""
    matched = match.group(0)
    meanings = list(dict.fromkeys(
        _expand_match(match, detected=detected, index=i, cleaned_query=cleaned_query)
        for i in range(max_expansions)
    ))
    meanings = [m for m in meanings if m != matched]
    return f"{matched} ({', '.join(meanings)})" if meanings else matched


# =============================================================================
# Phrase matching (does a text already contain a meaning?)
# =============================================================================

@lru_cache(maxsize=2048)
def _expansion_content_words(phrase: str) -> Tuple[str, ...]:
    """The important words of a meaning, lowercase, without stopwords.

    "District Institute of Education" -> ("district", "institute", "education").
    Cached; a tuple, so no caller can change the shared cached value.
    """
    words = _WORD_RE.findall(phrase.lower())
    return tuple(w for w in words if w not in STOP_WORDS)


@lru_cache(maxsize=2048)
def _document_words(text: str) -> Tuple[str, ...]:
    """Split a text into lowercase words: "PTM Notes 2024" -> ("ptm", "notes", "2024").

    Used to check whether the query already contains a meaning's words. Cached for speed.
    """
    return tuple(_WORD_RE.findall(text.lower()))


def _phrase_in_text(phrase: str, text: Optional[str]) -> bool:
    """True when every content word of `phrase` is in `text`, in any order.

    Inflections allowed ("institutes"); each content word needs its own word in the
    text, so one the phrase repeats ("Bachelor of ... Bachelor of ...") must appear as often.
    """
    if not phrase or not text:
        return False
    terms = _expansion_content_words(phrase)
    if not terms:
        return False
    words = _document_words(text)
    if not words:
        return False
    # Give each meaning word its own text word. If the word it needs is taken, try to
    # move the word that took it to another match, so a prefix match cannot block one.
    candidates = [[i for i, word in enumerate(words) if _words_match(word, term)] for term in terms]
    owner: Dict[int, int] = {}  # position of a text word -> the meaning word it is given to

    def assign(term_index: int, tried: set) -> bool:
        for i in candidates[term_index]:
            if i not in tried:
                tried.add(i)
                if i not in owner or assign(owner[i], tried):
                    owner[i] = term_index
                    return True
        return False

    return all(assign(t, set()) for t in range(len(terms)))


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


# =============================================================================
# Response and debug output
# =============================================================================

def acronym_method_description() -> Dict[str, str]:
    """acronym_info.method_description: how acronym queries are searched (semantic, keyword) and boosted."""
    dual_query = settings.ACRONYM_SEARCH_METHOD == const.ACRONYM_SEARCH_METHOD_DUAL_QUERY
    semantic = msg.ACRONYM_SEMANTIC_DUAL_QUERY if dual_query else msg.ACRONYM_SEMANTIC_INLINE
    # dual_query always runs one keyword search per meaning; inline only with ACRONYM_KEYWORD_OR.
    keyword = (msg.ACRONYM_KEYWORD_PER_MEANING if dual_query or settings.ACRONYM_KEYWORD_OR
               else msg.ACRONYM_KEYWORD_SINGLE)
    boosting = msg.ACRONYM_BOOSTING.format(
        exact_title=settings.ACRONYM_EXPANSION_EXACT_TITLE_BOOST,
        partial_title=settings.ACRONYM_EXPANSION_PARTIAL_TITLE_BOOST,
        exact_summary=settings.ACRONYM_EXPANSION_EXACT_SUMMARY_BOOST,
        partial_summary=settings.ACRONYM_EXPANSION_PARTIAL_SUMMARY_BOOST,
    )
    return {"semantic": semantic, "keyword": keyword, "boosting": boosting}


def acronym_info(acronym_query: Optional[AcronymQuery]) -> Dict[str, Any]:
    """The acronym_info response field."""
    info: Dict[str, Any] = {
        "enabled": settings.ACRONYM_SEARCH_ENABLED,
        "detected": acronym_query is not None,
    }
    if settings.ACRONYM_SEARCH_ENABLED:
        info["method"] = settings.ACRONYM_SEARCH_METHOD
        info["method_description"] = acronym_method_description()
    if acronym_query:
        info["mapping"] = acronym_query.detected
        ambiguous = [a for a, exps in acronym_query.detected.items() if len(exps) > 1]
        if ambiguous:
            info["ambiguous"] = ambiguous
    return info
