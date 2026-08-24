import re
from typing import Dict, List

from app.services.acronym_service import get_expansions_batch

# Letters only — strips internal dots ("D.I.E.T." -> "DIET"), digits, and any
# surrounding punctuation in one pass. Known simplification: a token like
# "PTM2024" also normalizes to "PTM", which isn't addressed by the spec.
_NON_LETTER_RE = re.compile(r"[^A-Za-z]")


def _normalize_token(raw_token: str) -> str:
    return _NON_LETTER_RE.sub("", raw_token)


def detect_acronyms(query: str) -> Dict[str, List[str]]:
    """Detect acronyms in `query`, returning {acronym: expansions} for every match
    found in the acronym dictionary (via acronym_service.get_expansions_batch's
    batched cache-aside lookup — one Redis round-trip and, for whatever's still
    missing, one Postgres round-trip for every candidate token in the query,
    instead of one of each per token). Must be called with the case-preserving
    query (query_for_keyword_match), not a lowercased/preprocessed one — these
    rules depend on case.

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

    # Collect every case/length-filtered candidate first, deduped, before any
    # lookup — as opposed to deduping only successfully-resolved acronyms
    # (which would mean two occurrences of the same non-acronym uppercase
    # word each triggered their own lookup).
    candidates: List[str] = []
    seen = set()
    for raw_token in raw_tokens:
        normalized = _normalize_token(raw_token)
        if len(normalized) < 2:
            continue

        if not (normalized.isupper() or is_single_word_query):
            continue

        candidate = normalized.upper()
        if candidate in seen:
            continue
        seen.add(candidate)
        candidates.append(candidate)

    if not candidates:
        return {}

    expansions_by_acronym = get_expansions_batch(candidates)
    # `if expansions` (falsy, not just "in the dict") rejects an empty-list
    # expansions value — the expansions column defaults to '[]'::jsonb and
    # only bulk_upsert() validates non-empty before insert, so a row written
    # via any other path (raw SQL, a future writer) could still have
    # expansions=[]. Guarding here closes both downstream call sites
    # (build_dense_queries' expansions[0], build_sparse_query's OR-join) at
    # once — neither ever sees an empty-list acronym in its mapping.
    return {
        acronym: expansions
        for acronym, expansions in expansions_by_acronym.items()
        if expansions
    }


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

    # Single pass over the ORIGINAL text, not a loop of sequential
    # substitutions — looping would let one acronym's expansion text (which
    # can itself contain another detected acronym as a plain word, e.g.
    # NFST -> "National Fellowship for ST" when ST is *also* detected in the
    # same query) get re-scanned and re-substituted by a later iteration,
    # producing garbled/duplicated text ("...Scheduled Tribes Scheduled
    # Tribes..." for "NFST ST fellowship" — confirmed empirically). One
    # alternation regex resolves every match against the ORIGINAL string in
    # a single left-to-right pass, so text inserted by resolving one match
    # is never re-scanned within this call.
    combined_pattern = re.compile(
        r"\b(?:" + "|".join(re.escape(acronym) for acronym in mapping) + r")\b",
        re.IGNORECASE,
    )

    def _resolve(match: re.Match) -> str:
        # Function replacement, not a string one — re.sub interprets
        # backslashes in a string replacement specially (\1, \g<name>, \t,
        # ...), so an expansion containing a literal backslash (e.g. a
        # pasted Windows path) would crash with re.error or silently
        # corrupt the text. A callable's return value is substituted
        # literally, with no escape processing. .upper() to resolve back to
        # the mapping's key regardless of the matched text's original case
        # (matching is case-insensitive; mapping keys are always uppercase).
        return mapping[match.group(0).upper()][0]

    substituted = combined_pattern.sub(_resolve, query_for_embedding)

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
