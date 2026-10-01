import re
from typing import Dict, List

from spacy.lang.en.stop_words import STOP_WORDS

from app.config import settings
from app.constants import ACRONYM_NON_LETTER_PATTERN
from app.services.acronym_service import get_expansions_batch

# Letters only: "D.I.E.T." -> "DIET", "PTM2024" -> "PTM" (see acronym-design-notes.md).
_NON_LETTER_RE = re.compile(ACRONYM_NON_LETTER_PATTERN)


def _normalize_token(raw_token: str) -> str:
    return _NON_LETTER_RE.sub("", raw_token)


def detect_acronyms(query: str) -> Dict[str, List[str]]:
    """Return {acronym: expansions} for every registered acronym in `query`.

    Checks single words, 2-N word phrases, plurals and glued single letters in one
    batched lookup. Rules and trade-offs: release-doc/acronym-design-notes.md#detection.
    """
    if not query or not query.strip():
        return {}

    raw_tokens = query.strip().split()
    is_single_word_query = len(raw_tokens) == 1

    # Deduped before lookup, so a repeated word is looked up once.
    candidates: List[str] = []
    seen = set()
    # Stopwords are still looked up; only registered ones (BE, ME, SO) come back.
    stopword_candidates: List[str] = []
    # {typed: singular guess}, tried only for candidates that miss as typed.
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

    # Phrase windows of 2-N words; a window with an empty word ("&", "2024") is
    # skipped, so non-adjacent words are never glued into a phrase.
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

        # A run of 2+ one-letter words is a dotted initialism commons split up
        # ("D.I.E.T." -> "d i e t"), so also try it glued ("DIET").
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
