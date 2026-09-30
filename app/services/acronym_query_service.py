import re
from typing import Dict, List

from spacy.lang.en.stop_words import STOP_WORDS

from app.config import settings
from app.constants import ACRONYM_NON_LETTER_PATTERN
from app.services.acronym_service import get_expansions_batch

# Digits are stripped per word, so "RTE2024 ACT" is detected as "RTE ACT" but
# not substituted (trailing digits only attach to the last word of a match).
_NON_LETTER_RE = re.compile(ACRONYM_NON_LETTER_PATTERN)


def _normalize_token(raw_token: str) -> str:
    return _NON_LETTER_RE.sub("", raw_token)


def detect_acronyms(query: str) -> Dict[str, List[str]]:
    """Detect acronyms in `query` via one batched Redis/Postgres lookup covering
    every candidate token AND every candidate phrase.

    Case-insensitive: every token of length > 1 is a candidate whatever its
    case, and the acronym table is what decides. Requiring uppercase was the
    stricter rule ("diet" in "the diet chart" must not match DIET), but it
    assumed callers preserve the case the user typed, and the commons media API
    lowercases the query before forwarding it — which silently disabled acronym
    search for every multi-word query reaching this service. The table itself is
    the narrow filter: a lowercase token that is not a registered acronym
    resolves to nothing, so the only words that can now match spuriously are the
    ones that are both everyday English and an acronym on file.

    Remaining rules:
      - A token shorter than 2 letters after normalization is skipped.
      - A stopword (THE, AND, ON) in a multi-word query is still stripped out
        of the ordinary candidate list, same as before — but it isn't just
        discarded. It's checked against the dictionary too, separately: if
        it turns out to be a registered acronym (BE, ME, SO), it's put back;
        if not (THE, AND, TO), it stays out, exactly like the old
        strip-and-discard behaviour. "BE admission" used to find nothing
        because "be" was thrown away before anyone checked whether it was
        registered, even though BE (Bachelor of Engineering) is on file.
        Single-word queries are exempt from stripping in the first place
        (unchanged), so a lone "BE" was already fine.

    Multi-word entries (e.g. "RTE ACT", "PM SHRI"): the dictionary already
    stores these as a single opaque key with a space in it — the cache and DB
    lookups never assumed a single word, only detection did. So alongside
    each single-token candidate, every contiguous run of 2..
    settings.ACRONYM_MAX_PHRASE_WORDS raw tokens is also normalized (per word) and joined
    into one space-separated phrase candidate. No stopword filtering is
    applied to phrases: unlike a bare word, a whole adjacent phrase is
    specific enough that the dictionary lookup itself is the filter — a
    phrase that isn't a registered acronym simply resolves to nothing, same
    as any other miss.

    Two more recovery passes, both aimed at text the caller already mangled
    before we ever see it (the commons media API replaces every punctuation
    character with a space, so "D.I.E.T." and "DIET's" never reach us intact):

      - Plural retry: a single-token candidate ending in "S" that doesn't
        match as typed ("DIETS") gets one more try with the trailing S
        stripped ("DIET"), only on a miss. The dictionary is still what
        decides — "ADMISSIONS" retries "ADMISSION", finds nothing, stays
        undetected either way.
      - Single-letter run gluing: a run of 2+ consecutive tokens that each
        normalize to exactly one letter is what "D.I.E.T." looks like by the
        time it reaches us (the caller turns each "." into a space, leaving
        four one-letter words) — normal English essentially never produces
        that pattern otherwise, so the run is also tried glued together with
        no separator ("d i e t" -> "DIET"). Accepted trade-off: "AI" is
        registered, so two genuinely separate one-letter words "a" and "i"
        sitting next to each other would also glue into it — rare, and
        there's no way to tell the two situations apart any more, since the
        caller already destroyed the dot-vs-space distinction before this
        function ever sees the query.
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
    # Stripped out of the ordinary candidate list below (same stopword rule
    # as before), but not discarded — checked against the dictionary
    # separately, and put back only if it turns out to be registered.
    stopword_candidates: List[str] = []
    # {as-typed candidate: trailing-S-stripped guess} — only consulted after
    # the first lookup, and only for a candidate that missed (see below).
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

    # Phrase candidates: every contiguous 2..N word window, each word
    # normalized the same way a single-token candidate would be. A window
    # containing a token that normalizes to nothing (pure punctuation like
    # "&", or a pure number) is skipped entirely rather than silently
    # closing the gap — that would glue two non-adjacent words together
    # under a phrase neither side of the gap actually forms.
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

        # Single-letter run gluing: a maximal run of 2+ consecutive tokens
        # that each normalize to exactly one letter is what a dotted
        # initialism looks like once a caller has replaced every "." with a
        # space ("D.I.E.T." -> "d i e t") — each letter is too short to be a
        # candidate on its own (the len(normalized) < 2 guard above already
        # dropped all four), so the only way to recover the original word is
        # to also try the run glued together with no separator.
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

    # Stripped stopwords go into the SAME batched lookup as everything else —
    # the dictionary is the only thing that decides whether one comes back:
    # a genuine acronym (BE, ME, SO) is found and kept, an ordinary stopword
    # (THE, AND, TO) is not found and stays out, same as the old
    # strip-and-discard result for it.
    candidates.extend(stopword_candidates)

    if not candidates:
        return {}

    expansions_by_acronym = get_expansions_batch(candidates)

    # Plural retry: only for a candidate that missed on the first pass — a
    # second batched lookup, fired only when at least one candidate ending in
    # "S" didn't match as typed. A result is reported under the STRIPPED key
    # ("DIET"), which is the real registered acronym, not the plural someone
    # typed.
    retry_targets = {
        singular
        for typed, singular in plural_retry.items()
        if not expansions_by_acronym.get(typed)
    }
    if retry_targets:
        for singular, expansions in get_expansions_batch(list(retry_targets)).items():
            if expansions:
                expansions_by_acronym[singular] = expansions

    # `if expansions` (falsy, not just "in the dict") rejects an empty-list
    # expansions value — the expansions column defaults to '[]'::jsonb and
    # only bulk_upsert() validates non-empty before insert, so a row written
    # via any other path (raw SQL, a future writer) could still have
    # expansions=[]. Guarding here closes both downstream call sites in
    # search() (the acronym-substitution regex, the sparse OR-join) at once —
    # neither ever sees an empty-list acronym in its mapping.
    return {
        acronym: expansions
        for acronym, expansions in expansions_by_acronym.items()
        if expansions
    }
