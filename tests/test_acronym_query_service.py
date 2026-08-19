"""Unit tests for detection + query expansion (app/services/acronym_query_service.py).

Pure logic tests — get_expansion is mocked so these don't touch Redis/Postgres.
"""
from unittest.mock import patch

from app.services.acronym_query_service import (
    build_dense_queries,
    build_sparse_query,
    detect_acronyms,
)

# Mirrors the real dataset's shape: acronym -> list of expansions (JSONB array,
# per PR1's schema). SSC is a real multi-expansion case from the seeded data.
_ACRONYMS = {
    "DIET": ["District Institute of Education and Training"],
    "PTM": ["Parent Teacher Meeting"],
    "SMC": ["School Management Committee"],
    "SSC": ["Staff Selection Commission", "Sainik School Society"],
}


def _fake_get_expansion(acronym):
    return _ACRONYMS.get(acronym)


class TestDetectAcronyms:
    def setup_method(self):
        self._patcher = patch(
            "app.services.acronym_query_service.get_expansion",
            side_effect=_fake_get_expansion,
        )
        self._patcher.start()

    def teardown_method(self):
        self._patcher.stop()

    def test_empty_query(self):
        assert detect_acronyms("") == {}
        assert detect_acronyms("   ") == {}

    def test_uppercase_token_in_multi_word_query_is_checked(self):
        assert detect_acronyms("next PTM schedule") == {"PTM": ["Parent Teacher Meeting"]}

    def test_lowercase_word_inside_sentence_is_skipped(self):
        # "diet" here is the common English word, not the acronym — must not match.
        assert detect_acronyms("the diet chart for kids") == {}

    def test_lone_lowercase_word_is_checked(self):
        # Single-word query, no shift key used — still plausibly the acronym.
        assert detect_acronyms("diet") == {"DIET": ["District Institute of Education and Training"]}

    def test_lone_uppercase_word_is_checked(self):
        assert detect_acronyms("PTM") == {"PTM": ["Parent Teacher Meeting"]}

    def test_single_word_not_in_dictionary_returns_empty(self):
        assert detect_acronyms("something") == {}

    def test_punctuation_and_dots_are_normalized(self):
        assert detect_acronyms("D.I.E.T.") == {"DIET": ["District Institute of Education and Training"]}
        assert detect_acronyms("what about PTM?") == {"PTM": ["Parent Teacher Meeting"]}

    def test_multiple_acronyms_in_one_query(self):
        result = detect_acronyms("PTM and SMC meeting schedule")
        assert result == {
            "PTM": ["Parent Teacher Meeting"],
            "SMC": ["School Management Committee"],
        }

    def test_single_letter_uppercase_is_ignored(self):
        # len > 1 guard — avoids "A"/"I" false-triggering.
        assert detect_acronyms("A I") == {}

    def test_mixed_case_word_inside_sentence_is_skipped(self):
        assert detect_acronyms("Diet plans for children") == {}

    def test_unknown_uppercase_token_returns_empty(self):
        assert detect_acronyms("ask about XYZ today") == {}

    def test_multi_expansion_acronym_detected_with_full_list(self):
        assert detect_acronyms("SSC recruitment") == {
            "SSC": ["Staff Selection Commission", "Sainik School Society"]
        }

    def test_empty_expansions_list_is_treated_as_not_found(self):
        """Regression test: the expansions JSONB column defaults to '[]' and
        only bulk_upsert() validates non-empty before insert — a row written
        via any other path (raw SQL, a future writer) could have
        expansions=[]. `is not None` alone would accept that; must be
        rejected the same way as "acronym not found", or it flows into
        build_dense_queries' expansions[0] and crashes with IndexError."""
        with patch(
            "app.services.acronym_query_service.get_expansion",
            side_effect=lambda a: [] if a == "XYZQ" else _fake_get_expansion(a),
        ):
            assert detect_acronyms("XYZQ report") == {}


class TestBuildDenseQueries:
    def test_no_mapping_returns_original_only(self):
        assert build_dense_queries("next ptm schedule", {}) == ["next ptm schedule"]

    def test_substitution_produces_two_variants_never_concatenated(self):
        result = build_dense_queries("next ptm schedule", {"PTM": ["Parent Teacher Meeting"]})
        assert result == ["next ptm schedule", "next Parent Teacher Meeting schedule"]

    def test_substitution_is_case_insensitive_and_word_bounded(self):
        # query_for_embedding is always lowercased upstream; mapping keys are uppercase.
        result = build_dense_queries("ptm", {"PTM": ["Parent Teacher Meeting"]})
        assert result == ["ptm", "Parent Teacher Meeting"]

    def test_no_op_substitution_does_not_produce_duplicate_variant(self):
        # Acronym detected in the original-case query but absent from this particular
        # (already-transformed) embedding string — substitution changes nothing.
        result = build_dense_queries("parent teacher meeting", {"PTM": ["Parent Teacher Meeting"]})
        assert result == ["parent teacher meeting"]

    def test_multiple_acronyms_all_substituted(self):
        result = build_dense_queries(
            "ptm and smc schedule",
            {"PTM": ["Parent Teacher Meeting"], "SMC": ["School Management Committee"]},
        )
        assert result == [
            "ptm and smc schedule",
            "Parent Teacher Meeting and School Management Committee schedule",
        ]

    def test_multi_expansion_acronym_uses_only_the_primary_first_expansion(self):
        """Spec §9: substitution uses the primary (first) expansion — SSC has
        two, only "Staff Selection Commission" (index 0) should appear."""
        result = build_dense_queries(
            "ssc recruitment",
            {"SSC": ["Staff Selection Commission", "Sainik School Society"]},
        )
        assert result == ["ssc recruitment", "Staff Selection Commission recruitment"]
        assert "Sainik School Society" not in result[1]

    def test_expansion_with_backslash_does_not_raise_and_is_substituted_literally(self):
        """Regression test: expansions come from an unvalidated admin CSV
        upload. re.sub interprets backslashes in a *string* replacement
        specially (\\1, \\g<name>, \\t, ...) — a pasted Windows-style path
        like "C:\\temp\\Some Office" used to raise re.error (bad escape) or
        silently corrupt the text. Must substitute literally instead."""
        result = build_dense_queries(
            "wfh policy",
            {"WFH": [r"C:\temp\Work From Home"]},
        )
        assert result == ["wfh policy", "C:\\temp\\Work From Home policy"]

    def test_full_pipeline_never_passes_an_empty_expansions_list_through(self):
        """End-to-end confirmation: detect_acronyms' empty-list guard means
        build_dense_queries never actually receives {"XYZQ": []} in
        practice — the mapping it's given only contains acronyms that
        already have real expansions, so its own expansions[0] is safe by
        construction, not because build_dense_queries itself guards it."""
        with patch(
            "app.services.acronym_query_service.get_expansion",
            side_effect=lambda a: [] if a == "XYZQ" else None,
        ):
            mapping = detect_acronyms("XYZQ report")
        assert mapping == {}
        # No IndexError even though the underlying data is empty-expansions —
        # because it never made it into the mapping at all.
        assert build_dense_queries("xyzq report", mapping) == ["xyzq report"]

    def test_expansion_with_backreference_like_text_is_substituted_literally(self):
        # A string replacement would either raise ("invalid group reference")
        # or silently splice in a capture group if the pattern had one.
        result = build_dense_queries(
            "smc rules",
            {"SMC": [r"Section \1 Management Committee"]},
        )
        assert result == ["smc rules", r"Section \1 Management Committee rules"]


class TestBuildSparseQuery:
    """build_sparse_query() appends plain expansion words — see the function's
    docstring for why the OR/quote-wrapped format this used to produce was
    dropped (confirmed empirically inert: the sparse encoder tokenizes it
    identically to plain word-appending, since it has no boolean/phrase
    syntax support at all)."""

    def test_no_mapping_returns_original(self):
        assert build_sparse_query("PTM", {}) == "PTM"

    def test_single_acronym_appends_expansion_words(self):
        assert build_sparse_query("PTM", {"PTM": ["Parent Teacher Meeting"]}) == "PTM Parent Teacher Meeting"

    def test_multi_word_query_appends_expansion_words(self):
        result = build_sparse_query("next PTM schedule", {"PTM": ["Parent Teacher Meeting"]})
        assert result == "next PTM schedule Parent Teacher Meeting"

    def test_multiple_acronyms_all_appended(self):
        result = build_sparse_query(
            "PTM and SMC",
            {"PTM": ["Parent Teacher Meeting"], "SMC": ["School Management Committee"]},
        )
        assert result == "PTM and SMC Parent Teacher Meeting School Management Committee"

    def test_multi_expansion_acronym_appends_every_expansion(self):
        """Unlike build_dense_queries, ALL expansions get appended here — BM25
        has no dilution risk from adding more matchable words."""
        result = build_sparse_query(
            "SSC recruitment",
            {"SSC": ["Staff Selection Commission", "Sainik School Society"]},
        )
        assert result == "SSC recruitment Staff Selection Commission Sainik School Society"

    def test_wrapped_and_plain_forms_tokenize_identically(self):
        """Regression test proving the OR/quote removal is behavior-preserving:
        the old wrapped format and the new plain-appended format must produce
        the exact same BM25 token set, confirming the wrapping was always
        inert decoration, never functional."""
        from app.core.clients.sparse_encoder import generate_sparse_vector

        old_wrapped = 'PTM OR "Parent Teacher Meeting"'
        new_plain = build_sparse_query("PTM", {"PTM": ["Parent Teacher Meeting"]})

        idx_old, _ = generate_sparse_vector(old_wrapped)
        idx_new, _ = generate_sparse_vector(new_plain)
        assert set(idx_old) == set(idx_new)
