import re
from typing import Dict, List

from app.services.acronym_service import get_expansion

# Letters only — strips internal dots ("D.I.E.T." -> "DIET"), digits, and any
# surrounding punctuation in one pass. Known simplification: a token like
# "PTM2024" also normalizes to "PTM", which isn't addressed by the spec.
_NON_LETTER_RE = re.compile(r"[^A-Za-z]")


def _normalize_token(raw_token: str) -> str:
    return _NON_LETTER_RE.sub("", raw_token)


def detect_acronyms(query: str) -> Dict[str, List[str]]:
    """Detect acronyms in `query`, returning {acronym: expansions} for every match
    found in the acronym dictionary (via acronym_service.get_expansion's cache-aside
    lookup). Must be called with the case-preserving query (query_for_keyword_match),
    not a lowercased/preprocessed one — these rules depend on case.

    Case rules:
      - A fully-uppercase token (letters-only length > 1) is always checked.
      - If the WHOLE query is a single word, it's checked regardless of case — a
        user typing just "diet" with no shift key is still plausibly searching
        for the acronym.
      - A lowercase/mixed-case word inside a longer, multi-word query is skipped.
        This is the precision guard: "the diet chart for kids" must not match
        the DIET acronym just because one word happens to collide with it.
    """
    if not query or not query.strip():
        return {}

    raw_tokens = query.strip().split()
    is_single_word_query = len(raw_tokens) == 1

    detected: Dict[str, str] = {}
    for raw_token in raw_tokens:
        normalized = _normalize_token(raw_token)
        if len(normalized) < 2:
            continue

        if not (normalized.isupper() or is_single_word_query):
            continue

        candidate = normalized.upper()
        if candidate in detected:
            continue

        expansion = get_expansion(candidate)
        # `is not None` alone would accept an empty list — the expansions
        # column defaults to '[]'::jsonb and only bulk_upsert() validates
        # non-empty before insert, so a row written via any other path (raw
        # SQL, a future writer) could still have expansions=[]. Guarding
        # here (falsy, not just None) rejects that case the same way as
        # "acronym not found," closing both downstream call sites
        # (build_dense_queries' expansions[0], build_sparse_query's OR-join)
        # at once — neither ever sees an empty-list acronym in its mapping.
        if expansion:
            detected[candidate] = expansion

    return detected


def build_dense_queries(query_for_embedding: str, mapping: Dict[str, List[str]]) -> List[str]:
    """Build the dense query variant(s) for an acronym-detected query: the original
    (embedding-ready) query, plus — only if substitution actually changes anything —
    a fully-substituted version with every detected acronym replaced by its primary
    (first) expansion, per spec §9. Only the first expansion is used here even when
    an acronym has several — the point of this variant is one coherent embeddable
    sentence, not every possible reading of the query.

    Never concatenated into one blended string: embedding "PTM meeting" + "Parent
    Teacher Meeting" together would average two concepts into one point in vector
    space, close to neither. Two separate query vectors instead (see
    ACRONYM_SEARCH_PLAN.md's Technical Design Notes).
    """
    if not mapping:
        return [query_for_embedding]

    substituted = query_for_embedding
    for acronym, expansions in mapping.items():
        pattern = re.compile(rf"\b{re.escape(acronym)}\b", re.IGNORECASE)
        # Function replacement, not a string one — re.sub interprets backslashes
        # in a string replacement specially (\1, \g<name>, \t, ...), so an
        # expansion containing a literal backslash (e.g. a pasted Windows path)
        # would crash with re.error or silently corrupt the text. A callable's
        # return value is substituted literally, with no escape processing.
        substituted = pattern.sub(lambda _m: expansions[0], substituted)

    # Case-insensitive: query_for_embedding is always lowercased upstream
    # (preprocess_query), but expansions keep their stored casing (e.g.
    # BLUETOOTH -> "Bluetooth"). A case-only difference is not a real second
    # reading of the query — confirmed empirically, the embedding model is
    # case-insensitive in practice (cosine similarity 1.0 between
    # "bluetooth" and "Bluetooth") — so returning it as a second variant
    # doubles the per-field Qdrant fan-out (search()'s use_acronym_multi_query
    # gate) for zero benefit. ~49 real seeded acronyms hit this exact case
    # (BLUETOOTH, DIGILOCKER, VEDANTU, UDAAN, ...).
    if substituted.lower() == query_for_embedding.lower():
        return [query_for_embedding]
    return [query_for_embedding, substituted]


def build_sparse_query(query_for_keyword_match: str, mapping: Dict[str, List[str]]) -> str:
    """Build the combined sparse (BM25) query string: original query text plus
    the words of every detected acronym's expansion(s) — all of them, not just
    the first, unlike build_dense_queries. Appending every known phrasing's
    words only adds more ways to match, with no dilution risk (the opposite
    rule from build_dense_queries, which must pick one expansion to stay a
    coherent embedding).

    Plain word-appending, NOT structured boolean/phrase syntax: the sparse
    encoder (fastembed's Qdrant/bm25) is a bag-of-words tokenizer with no
    notion of `OR` or quoted phrases — confirmed empirically, a wrapped
    query like `PTM OR "Parent Teacher Meeting"` and the plain
    `PTM Parent Teacher Meeting` produce byte-identical token sets ("or" is
    stripped as a stopword either way). An earlier version of this docstring
    claimed boolean syntax worked here; it doesn't, and the OR/quote
    wrapping was purely decorative — this appends the words directly instead
    of writing text that looks structured but isn't.
    """
    if not mapping:
        return query_for_keyword_match

    expansion_words = " ".join(
        expansion for expansions in mapping.values() for expansion in expansions
    )
    return f"{query_for_keyword_match} {expansion_words}"
