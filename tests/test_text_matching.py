"""Title/summary text matching shared by ordinary and acronym queries."""
import pytest

from app.utils.text_matching import (
    SUBSTRING, WORD, FieldMatchQuery, classify_field_match, classify_text_match,
    normalize_queries, term_in_text,
)


class TestSubstringRule:
    """Ordinary queries: the release-2.0.0 substring classification, unchanged."""

    @pytest.mark.parametrize("query, field, expected", [
        ("basic maths", "basic maths", "exact"),
        ("maths", "basic maths", "partial"),
        ("sur", "insurance", "partial"),
        ("science", "basic maths", None),
        ("maths", "", None),
    ])
    def test_classify(self, query, field, expected):
        assert classify_text_match(query, field) == expected
        assert classify_field_match(query, SUBSTRING, field) == expected


class TestWordRule:
    """Acronym queries: whole words, any case, file-name separators allowed."""

    @pytest.mark.parametrize("field, expected", [
        ("diet", "exact"),
        ("diet stakeholder identity map", "partial"),
        ("source_doc_coe_am4c2_diet_empowerment", "partial"),
        ("dietary guidelines", None),
        ("", None),
    ])
    def test_classify(self, field, expected):
        assert classify_field_match("diet", WORD, field) == expected

    def test_term_in_text_ignores_case(self):
        assert term_in_text("DIET", "The diet faculty")
        assert not term_in_text("DIET", "Dietary fibre")
        assert not term_in_text("", "anything")
        assert not term_in_text("DIET", None)


def test_normalize_queries_lowercases_trims_and_dedupes():
    queries = [FieldMatchQuery(" DIET ", "DIET", WORD), FieldMatchQuery("diet", "diet", WORD),
               FieldMatchQuery("diet", "diet", SUBSTRING), FieldMatchQuery("", "x", WORD)]
    assert normalize_queries(queries) == [
        FieldMatchQuery("diet", "diet", WORD), FieldMatchQuery("diet", "diet", SUBSTRING)]
