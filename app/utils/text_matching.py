"""Title/summary text matching, shared by ordinary and acronym queries.

Classifies how a query text matches a title or summary field: "exact" (the whole
field), "partial" (a match under the rule) or None. The caller picks the rule:
- SUBSTRING: the text anywhere in the field (ordinary queries).
- WORD: the text as a whole word, case-insensitive (acronym queries), so "DIET"
  never matches inside "dietary" but does match "_DIET_" in file-name titles.

Only the comparison lives here; the Qdrant scroll and the boosts stay in the
search service.
"""
import re
from typing import List, NamedTuple, Optional, Set, Tuple

SUBSTRING = "substring"
WORD = "word"


class FieldMatchQuery(NamedTuple):
    """One title/summary check: scroll_text (Qdrant prefilter), match_text (compared
    with the field) and rule (SUBSTRING | WORD).
    """
    scroll_text: str
    match_text: str
    rule: str


def normalize_queries(queries: List[FieldMatchQuery]) -> List[FieldMatchQuery]:
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


def classify_text_match(query_lower: str, field_lower: str) -> Optional[str]:
    """Classify how ``query_lower`` matches ``field_lower`` (the SUBSTRING rule).

    Returns 'exact' (whole field equals query), 'partial' (substring match —
    covers both prefix and mid/infix occurrences), or None (no match). Infix
    ('mid') matches are intentionally folded into 'partial' so the response
    contract stays {exact, partial, None}.
    """
    if not field_lower or query_lower not in field_lower:
        return None
    return "exact" if field_lower == query_lower else "partial"


def term_in_text(term: str, text: Optional[str]) -> bool:
    """Whole-word, case-insensitive match (the WORD rule): "DIET" never matches inside
    "dietary".

    Letter/digit lookarounds, not \\b, so "_DIET_" in file-name titles still matches.
    """
    if not term or not text:
        return False
    pattern = rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])"
    return re.search(pattern, text, re.IGNORECASE) is not None


def classify_field_match(match_text: str, rule: str, field_lower: str) -> Optional[str]:
    """'exact' if the field is the text, 'partial' if it matches under `rule`, else None."""
    if rule == SUBSTRING:
        return classify_text_match(match_text, field_lower)
    if not field_lower:
        return None
    if field_lower == match_text:
        return "exact"
    return "partial" if term_in_text(match_text, field_lower) else None
