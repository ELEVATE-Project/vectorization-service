"""Regression tests for the Release 2.0 scoring/ranking contract on the acronym branch.

Acronym support must widen candidate RETRIEVAL and re-order within the acronym
path only — it must never disturb an ordinary query. These tests pin:

1. The Release 2.0 fusion formula: no acronym-scoped dense/sparse weight flip.
2. filter_score / ordering / top_k semantics.
3. The injection contract: a document the pipeline actually scored keeps its
   real score and field_scores when re-injected; one never retrieved still
   takes the untouched Release 2.0 floor.
4. Title/summary multipliers unchanged.
5. Acronym queries: the same title/summary boosts as any other query (the query
   as typed, anywhere in the field), plus the expansion title/summary boosts on
   the query written out, stacked on top. No acronym-only or body boost.
   Expansion phrase matching (_phrase_in_text, on content words) is still used
   to decide whether the query already spells out an expansion.
6. The prefix-length floor (_words_match / ACRONYM_MIN_PREFIX_MATCH_LEN): content-word
   matching tolerates inflections via a prefix relationship, which unbounded
   let a single document letter satisfy a whole expansion word ("S M C
   Handbook" matching "School Management Committee"). Inflections must keep
   matching; initialisms and stray one-character tokens must not.
"""
import types

import pytest

from app.config import settings
from app.services.prioritized_search_service import PrioritizedSearchService

SPARSE = settings.SPARSE_VECTOR_NAME
FIELDS = list(settings.SEARCH_PRIORITY_ORDER)
WEIGHTS = dict(settings.SEARCH_PRIORITY_WEIGHTS)


@pytest.fixture
def service():
    return PrioritizedSearchService()


def _point(pid, source_id, **payload):
    """Minimal stand-in for a Qdrant point as _rank_results consumes it."""
    payload.setdefault("source_id", source_id)
    return types.SimpleNamespace(id=pid, payload=payload)


def _weighted_dense(field_scores):
    """The weighted multi-field cosine sum _rank_results computes internally."""
    return sum(
        score * WEIGHTS[field]
        for field, score in field_scores.items()
        if field in WEIGHTS and score is not None
    )


PTM = {"PTM": ["Parent Teacher Meeting"]}

# The expansion boost defaults, set by the tests themselves so a local .env does not
# change the results.
EXPANSION_BOOSTS = {
    "ACRONYM_EXPANSION_EXACT_TITLE_BOOST": 1.7,
    "ACRONYM_EXPANSION_PARTIAL_TITLE_BOOST": 1.4,
    "ACRONYM_EXPANSION_EXACT_SUMMARY_BOOST": 1.4,
    "ACRONYM_EXPANSION_PARTIAL_SUMMARY_BOOST": 1.2,
}


def _set_expansion_boosts(monkeypatch, **overrides):
    for name, value in {**EXPANSION_BOOSTS, **overrides}.items():
        monkeypatch.setattr(settings, name, value)


def _fake_scroll(monkeypatch, points, calls=None):
    """Stub qdrant_client.scroll to return `points` whatever the filter.

    The title/summary match scroll then classifies them itself, as it does the real
    MatchText hits, and the add-back fetch keeps only the source_ids it asked for.
    """
    def scroll(**kwargs):
        if calls is not None:
            calls.append(kwargs)
        return list(points), None

    monkeypatch.setattr("app.services.prioritized_search_service.qdrant_client.scroll", scroll)


# ── 1. Expansion phrase matching (used to build the expansion query) ──────

ACR = {"DIET": ["District Institute of Education and Training"]}


class TestPhraseInText:
    """_phrase_in_text decides whether a query already spells out an expansion."""

    @pytest.mark.parametrize("title", [
        # the exact expansion
        "District Institute of Education and Training",
        # plural inflection — the real title that used to score tier 0
        "Strengthening of District Institutes of Education and Training",
        # different connectives ("for" / "&" instead of "of" / "and")
        "District Institute for Education & Training",
        # word order is irrelevant once connectives are dropped
        "Education and Training — District Institute",
        # extra surrounding words don't matter
        "Annexure 3: District Institute of Education and Training (DIET) norms",
    ])
    def test_matches_real_expansion_variants(self, service, title):
        assert service._phrase_in_text(ACR["DIET"][0], title) is True

    @pytest.mark.parametrize("title", [
        # shares district+education but is a different programme
        "District Primary Education Programme Guidelines",
        "Tumkur District Education Transformation Program",
        "District Education Guidelines",
        "District Collectives - Enabling Partnerships for Systemic Transformation",
        # shares training/education only
        "Teacher training",
        "Establishing Discipline Through Clear School Rules",
    ])
    def test_rejects_near_misses(self, service, title):
        """ALL content words are required, so partial word overlap never promotes."""
        assert service._phrase_in_text(ACR["DIET"][0], title) is False

    @pytest.mark.parametrize("expansion,title", [
        # An initialism must not match the phrase it abbreviates: with no length
        # floor "school".startswith("s") and "management".startswith("man"), so
        # every content word was satisfied by a single letter of "S M C".
        ("School Management Committee", "S M C Handbook"),
        ("School Management Committee", "SMC Handbook"),
        # Stray one-character tokens from ordinary punctuation did the same:
        # "Brain's" tokenizes to [brain, s] and that lone "s" satisfied both
        # "software" and "service".
        ("Software as a Service", "Brain's ability to grow"),
        # "R.E.A.D" -> [r, e, a, d]; "r" satisfied "right", "e" satisfied
        # "education".
        ("Right to Education", "R.E.A.D programme overview"),
    ])
    def test_rejects_initialisms_and_stray_tokens(self, service, expansion, title):
        """Short document tokens must not satisfy long expansion words.

        Each of these returned True before ACRONYM_MIN_PREFIX_MATCH_LEN existed
        — 97 of 110 title x expansion matches across the corpus were spurious
        this way.
        """
        assert service._phrase_in_text(expansion, title) is False

    @pytest.mark.parametrize("expansion,title", [
        ("School Management Committee", "School Management Committees: a handbook"),
        ("Parent Teacher Meeting", "Parent Teachers Meetings guide"),
        ("National Curriculum Framework", "National Curriculum Frameworks"),
    ])
    def test_inflections_still_match(self, service, expansion, title):
        """The length floor must not cost the inflection tolerance it guards."""
        assert service._phrase_in_text(expansion, title) is True


class TestWordsMatch:
    """_words_match is the per-word rule _phrase_in_text applies."""

    def test_exact_match(self, service):
        assert service._words_match("district", "district") is True

    @pytest.mark.parametrize("word,term", [
        ("institutes", "institute"),   # plural document word, singular expansion
        ("institute", "institutes"),   # and the reverse
        ("programme", "program"),
        ("teachers", "teacher"),
    ])
    def test_prefix_counts_once_the_shorter_side_is_long_enough(self, service, word, term):
        assert service._words_match(word, term) is True

    @pytest.mark.parametrize("word,term", [
        ("s", "school"),          # initialism letter vs the word it abbreviates
        ("man", "management"),
        ("edu", "education"),
        ("committee", "com"),     # short side is the expansion word, same rule
    ])
    def test_prefix_below_the_floor_is_rejected(self, service, word, term):
        assert service._words_match(word, term) is False

    def test_floor_applies_to_the_shorter_side_regardless_of_order(self, service):
        """Length is measured on the shorter string, not on `word` or on `term`."""
        assert service._words_match("edu", "education") is service._words_match(
            "education", "edu")

    def test_unrelated_words_never_match(self, service):
        assert service._words_match("district", "training") is False

    def test_empty_inputs(self, service):
        assert service._phrase_in_text("", "anything") is False
        assert service._phrase_in_text(ACR["DIET"][0], None) is False
        assert service._phrase_in_text(ACR["DIET"][0], "") is False

    def test_stopword_only_phrase_never_matches_everything(self, service):
        """A phrase with no content words must not match every document."""
        assert service._phrase_in_text("of and the", "totally unrelated title") is False

    def test_content_words_drops_stopwords(self, service):
        # A tuple, not a list: _expansion_content_words is cached (the
        # expansions are fixed for the request but it used to be recomputed
        # inside every candidate's ranking call), so every caller now receives
        # the SAME object. Handing out a mutable one would let any caller
        # corrupt every later hit, so the immutability is part of the
        # contract, not incidental.
        assert service._expansion_content_words(ACR["DIET"][0]) == (
            "district", "institute", "education", "training")


class TestAcronymFieldMatch:
    """Acronym queries use the title/summary match of any other query: the query as
    typed, anywhere in the field, any case. The query written out gets its own
    expansion match (TestExpansionTitleBoost); the acronym alone and the full name
    alone are not matched."""

    @pytest.mark.parametrize("title, expected", [
        ("DIET Stakeholder Identity Map", "partial"),
        ("DIET", "exact"),
        # The full name does not match the query as typed.
        ("Strengthening of District Institutes of Education and Training", None),
        ("Establishing Discipline Through Clear School Rules", None),
        # A plain substring, as for any query.
        ("Dietary guidelines for schools", "partial"),
        ("source_doc_COE_AM4C2_DIET_Empowerment_Design.xlsx", "partial"),
    ])
    def test_title_match(self, service, title, expected):
        assert service._classify_text_match("diet", title.lower()) == expected

    def test_only_the_expansion_boost_differs_from_acronym_search_off(
            self, service, monkeypatch):
        _set_expansion_boosts(monkeypatch)
        titles = {"A": "DIET Handbook", "B": "District Institute of Education and Training"}
        run = TestAcronymRankingFollowsLexicalRanking()._run

        def multipliers(response):
            return {r.source_id: (r.title_multiplier, r.title_expansion_multiplier,
                                  r.summary_multiplier, r.summary_expansion_multiplier)
                    for r in response.results}

        on = run(service, monkeypatch, "hybrid", titles=titles, scroll=True,
                 include_scoring_debug=True)
        off = run(service, monkeypatch, "hybrid", titles=titles, scroll=True,
                  include_scoring_debug=True, acronym_search_enabled=False)
        # "DIET" written out is the full name itself, so B's title equals the written-out query.
        assert multipliers(on) == {
            "A": (settings.PARTIAL_TITLE_BOOST, 1.0, 1.0, 1.0),
            "B": (1.0, settings.ACRONYM_EXPANSION_EXACT_TITLE_BOOST, 1.0, 1.0)}
        assert multipliers(off) == {
            "A": (settings.PARTIAL_TITLE_BOOST, 1.0, 1.0, 1.0),
            "B": (1.0, 1.0, 1.0, 1.0)}


class TestExpansionTitleBoost:
    """Acronym queries: the ordinary title/summary rule on the query written out ("PTM Doc"
    -> "Parent Teacher Meeting Doc"): "exact" when the field equals it, "partial" when it
    contains it, any case. ACRONYM_EXPANSION_{EXACT,PARTIAL}_TITLE_BOOST (and the _SUMMARY_
    pair) apply on top of the ordinary boost, so a document matching both gets both. They
    are recorded as title/summary_expansion_match and _multiplier, next to the ordinary
    title/summary_match and _multiplier."""

    @staticmethod
    def _points(field, *texts):
        return [_point(f"pt-{i}", str(i), **{field: t}) for i, t in enumerate(texts)]

    def test_match_levels(self, service, monkeypatch):
        _fake_scroll(monkeypatch, self._points(
            "title",
            "Parent Teacher Meeting Doc",            # equals, any case
            "parent teacher meeting doc in school",  # contains
            "Parent teachers meeting Doc",           # plural: no match
            "Doc on the Parent Teacher Meeting",     # other order: no match
            "Parent Teacher Meeting Guidelines",     # full name only: no match
            "PTM Doc"))                              # the query as typed: ordinary match
        expansion_matches = {}
        matches = service._get_field_match_sources(
            "PTM Doc", None, "title", expansions=["Parent Teacher Meeting Doc"],
            expansion_matches_out=expansion_matches)
        assert expansion_matches == {"0": "exact", "1": "partial"}
        assert matches == {"5": "exact"}

    def test_best_match_over_meanings(self, service, monkeypatch):
        _fake_scroll(monkeypatch, self._points(
            "title", "Micro Improvement Plan", "Micro Improvement Project for schools"))
        expansion_matches = {}
        service._get_field_match_sources(
            "MIP", None, "title",
            expansions=["Micro Improvement Project", "Micro Improvement Plan"],
            expansion_matches_out=expansion_matches)
        assert expansion_matches == {"0": "exact", "1": "partial"}

    def test_one_scroll_matches_the_query_and_every_written_out_query(
            self, service, monkeypatch):
        """No extra Qdrant call: the written-out queries are OR'ed into the same filter."""
        calls = []
        _fake_scroll(monkeypatch, [], calls)
        service._get_field_match_sources(
            "MIP Doc", None, "summary",
            expansions=["Micro Improvement Project Doc", "Micro Improvement Plan Doc"],
            expansion_matches_out={})
        assert len(calls) == 1
        (any_of,) = calls[0]["scroll_filter"].must
        assert {(c.key, c.match.text) for c in any_of.should} == {
            ("summary", "mip doc"),
            ("summary", "micro improvement project doc"),
            ("summary", "micro improvement plan doc")}

    def test_hard_filters_still_apply(self, service, monkeypatch):
        """The written-out queries must not re-admit documents the filters exclude."""
        from qdrant_client import models

        calls = []
        _fake_scroll(monkeypatch, [], calls)
        org = models.FieldCondition(key="organization", match=models.MatchValue(value="X"))
        hidden = models.FieldCondition(key="file_type", match=models.MatchValue(value="zip"))
        service._get_field_match_sources(
            "PTM", models.Filter(must=[org], must_not=[hidden]), "title",
            expansions=["Parent Teacher Meeting"], expansion_matches_out={})
        scroll_filter = calls[0]["scroll_filter"]
        assert scroll_filter.must[0] == org
        assert scroll_filter.must_not == [hidden]

    def _search(self, service, monkeypatch, docs, query="PTM Doc", field="title",
                acronyms=None, scroll_only=None, search_mode="hybrid"):
        """docs: {source_id: text}, the candidates; the text goes in `field`, the other of
        title/summary matches nothing. scroll_only: {source_id: (title, summary)} found
        only by the title/summary scroll, not among the candidates."""
        from app.models.api_models import PrioritizedSearchRequest

        monkeypatch.setattr("app.services.prioritized_search_service.detect_acronyms",
                            lambda q: dict(PTM if acronyms is None else acronyms))
        monkeypatch.setattr(settings, "ACRONYM_SEARCH_ENABLED", True)
        monkeypatch.setattr(settings, "ACRONYM_SEARCH_METHOD", "inline")
        monkeypatch.setattr(settings, "ACRONYM_KEYWORD_OR", False)
        monkeypatch.setattr(settings, "HYBRID_SEARCH_ENABLED", True)
        monkeypatch.setattr(settings, "SPARSE_SEARCH_ENABLED", False)
        monkeypatch.setattr("app.services.prioritized_search_service.embedding.embed_query",
                            lambda texts: [[0.0] * 8 for _ in texts])
        neutral = {"title": "Annual report", "summary": ""}
        all_results = {f"pt-{s}": _point(f"pt-{s}", s, text="x", **{**neutral, field: t})
                       for s, t in docs.items()}
        field_scores = {pid: {"title": 0.5} for pid in all_results}
        monkeypatch.setattr(PrioritizedSearchService, "_parallel_batch_search",
                            lambda self, **kw: (all_results, field_scores))
        scroll_only_points = [_point(f"pt-{s}", s, text="x", title=title, summary=summary)
                              for s, (title, summary) in (scroll_only or {}).items()]
        _fake_scroll(monkeypatch, list(all_results.values()) + scroll_only_points)
        return service.search(PrioritizedSearchRequest(
            query=query, top_k=10, search_mode=search_mode, include_scoring_debug=True))

    @staticmethod
    def _multipliers(response, field="title"):
        """{source_id: (ordinary multiplier, expansion multiplier)} for `field`."""
        return {r.source_id: (getattr(r, f"{field}_multiplier"),
                              getattr(r, f"{field}_expansion_multiplier"))
                for r in response.results}

    def test_search_stacks_the_two_boosts(self, service, monkeypatch):
        _set_expansion_boosts(monkeypatch)
        response = self._search(service, monkeypatch, {
            "written_out": "Parent Teacher Meeting Doc",
            "contains_written_out": "Parent Teacher Meeting Doc in school",
            "plural": "Parent teachers meeting Doc",
            "expansion_only": "Parent Teacher Meeting Guidelines",
            "acronym": "PTM Doc",
            "contains_query": "PTM Doc in school",
            "both": "PTM Doc - Parent Teacher Meeting Doc",
            "none": "Annual report",
        })
        assert self._multipliers(response) == {
            "written_out": (1.0, settings.ACRONYM_EXPANSION_EXACT_TITLE_BOOST),
            "contains_written_out": (1.0, settings.ACRONYM_EXPANSION_PARTIAL_TITLE_BOOST),
            "plural": (1.0, 1.0),
            "expansion_only": (1.0, 1.0),
            "acronym": (settings.EXACT_TITLE_BOOST, 1.0),   # the whole title is the query
            "contains_query": (settings.PARTIAL_TITLE_BOOST, 1.0),
            # Contains the query and the written-out query: both boosts.
            "both": (settings.PARTIAL_TITLE_BOOST, settings.ACRONYM_EXPANSION_PARTIAL_TITLE_BOOST),
            "none": (1.0, 1.0),
        }
        both = next(r for r in response.results if r.source_id == "both")
        assert (both.title_match, both.title_expansion_match) == ("partial", "partial")
        assert both.score == pytest.approx(min(
            0.5 * WEIGHTS["title"] * settings.PARTIAL_TITLE_BOOST
            * settings.ACRONYM_EXPANSION_PARTIAL_TITLE_BOOST, 1.0))

    def test_boost_values_come_from_settings(self, service, monkeypatch):
        _set_expansion_boosts(monkeypatch, ACRONYM_EXPANSION_EXACT_TITLE_BOOST=1.9,
                              ACRONYM_EXPANSION_PARTIAL_TITLE_BOOST=1.0)
        response = self._search(service, monkeypatch, {
            "written_out": "Parent Teacher Meeting Doc",
            "contains_written_out": "Parent Teacher Meeting Doc in school",
        })
        assert self._multipliers(response) == {
            "written_out": (1.0, 1.9), "contains_written_out": (1.0, 1.0)}

    def test_summary_gets_the_same_rule_with_summary_boosts(self, service, monkeypatch):
        _set_expansion_boosts(monkeypatch, ACRONYM_EXPANSION_EXACT_SUMMARY_BOOST=1.3,
                              ACRONYM_EXPANSION_PARTIAL_SUMMARY_BOOST=1.25)
        response = self._search(service, monkeypatch, {
            "written_out": "parent teacher meeting doc",
            "contains_written_out": "Notes on the Parent Teacher Meeting Doc",
            "expansion_only": "Parent Teacher Meeting Guidelines",
            "acronym": "About the PTM Doc",
            "both": "PTM Doc - Parent Teacher Meeting Doc",
            "none": "Annual report",
        }, field="summary")
        assert self._multipliers(response, "summary") == {
            "written_out": (1.0, 1.3),
            "contains_written_out": (1.0, 1.25),
            "expansion_only": (1.0, 1.0),
            "acronym": (settings.PARTIAL_SUMMARY_BOOST, 1.0),
            "both": (settings.PARTIAL_SUMMARY_BOOST, 1.25),
            "none": (1.0, 1.0),
        }
        # Summary matches leave the title multipliers alone.
        assert set(self._multipliers(response, "title").values()) == {(1.0, 1.0)}

    def test_ordinary_queries_get_no_expansion_boost(self, service, monkeypatch):
        _set_expansion_boosts(monkeypatch)
        response = self._search(service, monkeypatch, {
            "written_out": "Parent Teacher Meeting Doc", "acronym": "PTM Doc"}, acronyms={})
        assert self._multipliers(response) == {
            "written_out": (1.0, 1.0), "acronym": (settings.EXACT_TITLE_BOOST, 1.0)}
        assert all(r.title_expansion_match is None for r in response.results)

    def test_semantic_mode_gets_no_expansion_boost(self, service, monkeypatch):
        _set_expansion_boosts(monkeypatch)
        response = self._search(service, monkeypatch, {
            "written_out": "Parent Teacher Meeting Doc", "acronym": "PTM Doc"},
            search_mode="semantic")
        assert set(self._multipliers(response).values()) == {(1.0, 1.0)}

    def test_missing_written_out_matches_are_added_back(self, service, monkeypatch):
        """A document only the title/summary scroll finds is added back like an ordinary
        match: 0.15 x its boost, raised to the threshold. One that matches the query too
        gets both boosts. Title wins when both fields match."""
        _set_expansion_boosts(monkeypatch)
        response = self._search(service, monkeypatch, {"candidate": "Annual report"}, scroll_only={
            "title_written_out": ("Parent Teacher Meeting Doc", ""),
            "title_both": ("PTM Doc - Parent Teacher Meeting Doc", ""),
            "summary_written_out": ("Annual report", "Parent Teacher Meeting Doc"),
            "title_and_summary": ("Parent Teacher Meeting Doc", "Parent Teacher Meeting Doc"),
        })
        floor = TestFieldMatchInjection.FLOOR
        ids = [r.source_id for r in response.results]
        assert len(ids) == len(set(ids))
        by_source = {r.source_id: r for r in response.results}
        assert set(by_source) == {"candidate", "title_written_out", "title_both",
                                  "summary_written_out", "title_and_summary"}
        assert response.total_results == 5

        added = by_source["title_written_out"]
        assert added.match_source == "title_keyword_match"
        assert (added.title_match, added.title_expansion_match) == (None, "exact")
        assert (added.title_multiplier, added.title_expansion_multiplier) == (
            1.0, settings.ACRONYM_EXPANSION_EXACT_TITLE_BOOST)
        assert added.score == pytest.approx(floor * settings.ACRONYM_EXPANSION_EXACT_TITLE_BOOST)

        both = by_source["title_both"]
        assert (both.title_match, both.title_expansion_match) == ("partial", "partial")
        assert both.score == pytest.approx(
            floor * settings.PARTIAL_TITLE_BOOST * settings.ACRONYM_EXPANSION_PARTIAL_TITLE_BOOST)

        summary = by_source["summary_written_out"]
        assert summary.match_source == "summary_keyword_match"
        assert (summary.summary_match, summary.summary_expansion_match) == (None, "exact")
        assert summary.summary_expansion_multiplier == settings.ACRONYM_EXPANSION_EXACT_SUMMARY_BOOST
        assert summary.score == pytest.approx(
            floor * settings.ACRONYM_EXPANSION_EXACT_SUMMARY_BOOST)

        assert by_source["title_and_summary"].match_source == "title_keyword_match"

    @pytest.mark.parametrize("name", list(EXPANSION_BOOSTS))
    def test_values_below_one_fail_at_startup(self, monkeypatch, name):
        from app.config import Settings

        monkeypatch.setenv(name, "0.9")
        with pytest.raises(ValueError, match=name):
            Settings()


class TestFusionFormulaHasNoAcronymBranch:
    def test_weighted_blend_is_the_plain_release_two_zero_formula(self, service):
        """No dense/sparse weight flip may survive for any point, sparse hit or not."""
        all_results = {"p1": _point("p1", "1"), "p2": _point("p2", "2")}
        field_scores = {
            "p1": {"title": 0.9, SPARSE: 10.0},   # strong dense, strong sparse
            "p2": {"title": 0.1, SPARSE: 0.0},    # weak dense, no sparse hit
        }
        ranked = {r["id"]: r for r in service._rank_results(
            all_results, field_scores, WEIGHTS, FIELDS)}

        for entry in ranked.values():
            expected = (
                settings.HYBRID_DENSE_WEIGHT * entry["normalized_dense"]
                + settings.HYBRID_SPARSE_WEIGHT * entry["normalized_sparse"]
            )
            assert entry["weighted_score"] == pytest.approx(expected)

    def test_rank_results_takes_no_acronym_flag(self, service):
        import inspect
        params = inspect.signature(service._rank_results).parameters
        assert "is_acronym_query" not in params


# ── 3. Filtering / ordering contract ──────────────────────────────────────────

class TestFilterAndOrderingUnchanged:
    def _fixture(self):
        all_results = {f"p{i}": _point(f"p{i}", str(i)) for i in range(1, 5)}
        field_scores = {
            "p1": {"title": 0.90, SPARSE: 0.0},
            "p2": {"title": 0.60, SPARSE: 0.0},
            "p3": {"title": 0.30, SPARSE: 0.0},
            "p4": {"title": 0.05, SPARSE: 0.0},
        }
        return all_results, field_scores

    def test_threshold_drops_below_score_and_order_is_descending(self, service):
        all_results, field_scores = self._fixture()
        top, unique = service._process_and_filter_results(
            all_results, field_scores, WEIGHTS, FIELDS, top_k=10, threshold=0.5,
        )
        scores = [r["weighted_score"] for r in top]
        assert scores == sorted(scores, reverse=True)
        assert all(s >= 0.5 for s in scores)

    def test_top_k_caps_the_returned_list(self, service):
        all_results, field_scores = self._fixture()
        top, _ = service._process_and_filter_results(
            all_results, field_scores, WEIGHTS, FIELDS, top_k=2, threshold=0.0,
        )
        assert len(top) == 2

    def test_prethreshold_by_source_out_captures_pre_filter_entries(self, service):
        """The snapshot must include sources the threshold then removes — that is
        the whole point of taking it before filtering."""
        all_results, field_scores = self._fixture()
        snapshot = {}
        top, _ = service._process_and_filter_results(
            all_results, field_scores, WEIGHTS, FIELDS, top_k=10, threshold=0.9,
            prethreshold_by_source_out=snapshot,
        )
        kept = {r["payload"]["source_id"] for r in top}
        assert set(snapshot) == {"1", "2", "3", "4"}
        assert kept < set(snapshot)          # strictly fewer survived the threshold
        assert snapshot["4"]["field_scores"]["title"] == 0.05

    def test_snapshot_is_optional(self, service):
        all_results, field_scores = self._fixture()
        top, _ = service._process_and_filter_results(
            all_results, field_scores, WEIGHTS, FIELDS, top_k=10, threshold=0.0,
        )
        assert top  # no out-param supplied, nothing raised


# ── 4. Injection: real scores vs the Release 2.0 floor ────────────────────────

class TestFieldMatchInjection:
    FLOOR = 0.15

    def _patch_scroll(self, monkeypatch, source_ids):
        """Stub the Qdrant scroll _fetch_field_match_docs uses."""
        points = [_point(f"pt-{sid}", sid, title=f"Doc {sid}") for sid in source_ids]

        def fake_scroll(**kwargs):
            return points, None

        monkeypatch.setattr(
            "app.services.prioritized_search_service.qdrant_client.scroll",
            fake_scroll,
        )

    def test_unretrieved_source_keeps_the_floor_path(self, service, monkeypatch):
        """Release 2.0 behavior for a document that was never scored."""
        self._patch_scroll(monkeypatch, ["77"])
        injected = service._fetch_field_match_docs(
            ["77"], {"77": "partial"}, "title",
            settings.EXACT_TITLE_BOOST, settings.PARTIAL_TITLE_BOOST,
        )
        assert len(injected) == 1
        entry = injected[0]
        assert entry["weighted_score"] == pytest.approx(
            self.FLOOR * settings.PARTIAL_TITLE_BOOST)
        # field scores stay None so callers can tell "not scored" from "scored 0.0"
        assert all(entry["field_scores"][f] is None for f in service.priority_order)
        assert entry["raw_dense"] is None
        assert entry["title_multiplier"] == settings.PARTIAL_TITLE_BOOST

    def test_reused_source_keeps_real_field_scores_not_weighted_score(self, service, monkeypatch):
        """A reused (pre-threshold) source still gets the floor/score_floor formula
        for weighted_score, same as a never-scored document — the guarantee that an
        injected document never reads as scoring below the caller's threshold applies
        uniformly. What DOES survive from the real pipeline scoring is the per-field
        breakdown (field_scores/raw_dense/keyword_score), so debug output stays
        accurate even though the ranking score itself is the floor-based one."""
        self._patch_scroll(monkeypatch, ["219"])
        real = {
            "id": "pt-219",
            "payload": {"source_id": "219"},
            "weighted_score": 0.31,
            "field_scores": {"text": 0.29, "tags": 0.61, "summary": 0.40},
            "raw_dense": 0.2228,
            "keyword_score": 0.0,
        }
        injected = service._fetch_field_match_docs(
            ["219"], {"219": "partial"}, "title",
            settings.EXACT_TITLE_BOOST, settings.PARTIAL_TITLE_BOOST,
            prethreshold_by_source={"219": real},
        )
        entry = injected[0]
        assert entry["weighted_score"] == pytest.approx(
            self.FLOOR * settings.PARTIAL_TITLE_BOOST)
        # real per-field similarities survive instead of being blanked
        assert entry["field_scores"]["tags"] == 0.61
        assert entry["field_scores"]["title_match"] == "partial"
        assert entry["raw_dense"] == 0.2228
        assert entry["keyword_score"] == 0.0

    def test_reused_and_never_scored_sources_land_on_the_same_floor(self, service, monkeypatch):
        """A reused source and a never-scored source get the identical weighted_score
        (both go through the same floor/score_floor formula) — only their field_scores
        metadata differs (real values vs None)."""
        self._patch_scroll(monkeypatch, ["219"])
        real = {
            "payload": {"source_id": "219"}, "weighted_score": 0.31,
            "field_scores": {"tags": 0.61}, "raw_dense": 0.2228, "keyword_score": 0.0,
        }
        floor_only = service._fetch_field_match_docs(
            ["219"], {"219": "partial"}, "title",
            settings.EXACT_TITLE_BOOST, settings.PARTIAL_TITLE_BOOST,
        )[0]
        with_real = service._fetch_field_match_docs(
            ["219"], {"219": "partial"}, "title",
            settings.EXACT_TITLE_BOOST, settings.PARTIAL_TITLE_BOOST,
            prethreshold_by_source={"219": real},
        )[0]
        assert floor_only["weighted_score"] == pytest.approx(0.225)
        assert with_real["weighted_score"] == pytest.approx(floor_only["weighted_score"])
        assert floor_only["field_scores"]["tags"] is None
        assert with_real["field_scores"]["tags"] == 0.61

    def test_boost_cap_still_applies(self, service, monkeypatch):
        """The floor/boost/score_floor formula never produces a score above 1.0,
        no matter how high score_floor (the caller's threshold) is."""
        self._patch_scroll(monkeypatch, ["5"])
        entry = service._fetch_field_match_docs(
            ["5"], {"5": "exact"}, "title",
            settings.EXACT_TITLE_BOOST, settings.PARTIAL_TITLE_BOOST,
            score_floor=2.0,
        )[0]
        assert entry["weighted_score"] == 1.0

    def test_empty_source_list_short_circuits(self, service):
        assert service._fetch_field_match_docs(
            [], {}, "title",
            settings.EXACT_TITLE_BOOST, settings.PARTIAL_TITLE_BOOST) == []

    def test_rescued_entry_identifies_the_chunk_that_earned_the_score(
            self, service, monkeypatch):
        """A multi-chunk source must not pair one chunk's text with another's score.

        prethreshold_by_source holds the BEST chunk per source; the scroll returns chunks
        in arbitrary order. Taking id/payload from the scroll while taking the score
        from the fallback returns a passage that never earned it — and the id also
        drives search()'s late payload retrieval, which then locks the wrong text in.
        """
        # Scroll returns chunk 1 first; chunk 7 is the one that was actually scored.
        chunks = [
            _point("pt-900-c1", "900", title="SMC Handbook", text="table of contents"),
            _point("pt-900-c7", "900", title="SMC Handbook", text="the passage that scored"),
        ]
        monkeypatch.setattr(
            "app.services.prioritized_search_service.qdrant_client.scroll",
            lambda **kwargs: (chunks, None),
        )
        best_chunk = {
            "id": "pt-900-c7",
            "payload": chunks[1].payload,
            "weighted_score": 0.31,
            "field_scores": {"text": 0.29},
            "raw_dense": 0.2228,
            "keyword_score": 0.0,
        }
        entry = service._fetch_field_match_docs(
            ["900"], {"900": "partial"}, "title",
            settings.EXACT_TITLE_BOOST, settings.PARTIAL_TITLE_BOOST,
            prethreshold_by_source={"900": best_chunk},
        )[0]
        assert entry["id"] == "pt-900-c7"
        assert entry["payload"]["text"] == "the passage that scored"
        assert entry["weighted_score"] == pytest.approx(
            self.FLOOR * settings.PARTIAL_TITLE_BOOST)

    def test_floor_path_still_takes_the_scrolled_chunk(self, service, monkeypatch):
        """No fallback means no scores to be consistent with — any chunk will do."""
        self._patch_scroll(monkeypatch, ["901"])
        entry = service._fetch_field_match_docs(
            ["901"], {"901": "partial"}, "title",
            settings.EXACT_TITLE_BOOST, settings.PARTIAL_TITLE_BOOST,
        )[0]
        assert entry["id"] == "pt-901"
        assert entry["payload"]["title"] == "Doc 901"

    def test_injected_acronym_title_takes_the_plain_floor(self, service, monkeypatch):
        """Floor x title boost, like any title match."""
        monkeypatch.setattr(
            "app.services.prioritized_search_service.qdrant_client.scroll",
            lambda **kwargs: ([_point("pt-e", "E", title="DIET Handbook", summary="About the DIET")], None),
        )
        entry = service._fetch_field_match_docs(
            ["E"], {"E": "partial"}, "title",
            settings.EXACT_TITLE_BOOST, settings.PARTIAL_TITLE_BOOST,
        )[0]
        assert entry["weighted_score"] == pytest.approx(self.FLOOR * settings.PARTIAL_TITLE_BOOST)
        assert "acronym_bonus" not in entry

    def test_expansion_add_back_records_its_own_keys(self, service, monkeypatch):
        """mult_key keeps the expansion match apart from the ordinary one, on both the
        reused and the never-scored path."""
        self._patch_scroll(monkeypatch, ["77"])
        real = {"id": "pt-219", "payload": {"source_id": "219"}, "weighted_score": 0.31,
                "field_scores": {"tags": 0.61}}
        entries = service._fetch_field_match_docs(
            ["77", "219"], {"77": "exact", "219": "partial"}, "title", 1.7, 1.4,
            prethreshold_by_source={"219": real}, mult_key="title_expansion_multiplier",
        )
        by_source = {e["payload"]["source_id"]: e for e in entries}
        for source_id, match_type, boost in (("77", "exact", 1.7), ("219", "partial", 1.4)):
            entry = by_source[source_id]
            assert entry["field_scores"]["title_expansion_match"] == match_type
            assert entry["title_expansion_multiplier"] == boost
            assert "title_match" not in entry["field_scores"]
            assert "title_multiplier" not in entry
            assert entry["weighted_score"] == pytest.approx(self.FLOOR * boost)
            assert entry["match_source"] == "title_keyword_match"
        # The shared pre-threshold entry is copied, not changed.
        assert real["field_scores"] == {"tags": 0.61}


# ── 5. Multipliers themselves are untouched ───────────────────────────────────

class TestFieldBoostUnchanged:
    def test_exact_and_partial_multipliers_and_resort(self, service):
        ranked = [
            {"payload": {"source_id": "a"}, "weighted_score": 0.20,
             "field_scores": {}},
            {"payload": {"source_id": "b"}, "weighted_score": 0.30,
             "field_scores": {}},
            {"payload": {"source_id": "c"}, "weighted_score": 0.25,
             "field_scores": {}},
        ]
        out = service._apply_field_boost(
            ranked, {"a": "exact", "b": "partial"}, "title",
            settings.EXACT_TITLE_BOOST, settings.PARTIAL_TITLE_BOOST,
        )
        by_source = {r["payload"]["source_id"]: r for r in out}
        assert by_source["a"]["weighted_score"] == pytest.approx(
            0.20 * settings.EXACT_TITLE_BOOST)
        assert by_source["b"]["weighted_score"] == pytest.approx(
            0.30 * settings.PARTIAL_TITLE_BOOST)
        # unmatched keeps its score and records a neutral 1.0 multiplier
        assert by_source["c"]["weighted_score"] == 0.25
        assert by_source["c"]["title_multiplier"] == 1.0
        assert by_source["c"]["field_scores"]["title_match"] is None
        # re-sorted descending after boosting
        scores = [r["weighted_score"] for r in out]
        assert scores == sorted(scores, reverse=True)


# ── 6. Acronym retrieval still works ──────────────────────────────────────────

class TestAcronymExpansionStillDrivesRetrieval:
    """Patch target note: this suite used to stub `get_acronym_mapping` with
    `raising=False`. That function was renamed to `get_expansions_batch`, and
    because `raising=False` suppresses monkeypatch's "target does not exist"
    error, the stub silently became a no-op — the real lookup ran instead,
    hitting Redis and Postgres and failing anywhere the acronym_mapping table
    wasn't populated. Never pass `raising=False` here: the whole point of the
    check is to fail loudly the next time the function moves.
    """

    @staticmethod
    def _fake_lookup(expansions_by_acronym):
        """Stand-in for get_expansions_batch, matching its real contract:
        takes the candidate tokens, returns only the ones it recognizes. A
        stub that ignored its argument would pass even if detect_acronyms
        looked up entirely the wrong tokens."""
        def _lookup(acronyms):
            return {a: expansions_by_acronym[a] for a in acronyms if a in expansions_by_acronym}
        return _lookup

    def test_detected_acronym_yields_expansions(self, monkeypatch):
        """DIET must expand so 'District Institute...' documents are reachable."""
        import app.services.acronym_query_service as aqs

        monkeypatch.setattr(
            aqs, "get_expansions_batch",
            self._fake_lookup({"DIET": ["District Institute of Education and Training"]}),
        )
        detected = aqs.detect_acronyms("DIET")
        assert "DIET" in detected
        assert any("District Institute" in e for e in detected["DIET"])

    def test_lookup_receives_normalized_candidates(self, monkeypatch):
        """The tokens sent to the dictionary are uppercased and stripped of
        punctuation. Pinning this is what the old stub couldn't do: it took no
        argument, so detect_acronyms could have sent anything at all."""
        import app.services.acronym_query_service as aqs

        seen = {}

        def _capture(acronyms):
            seen["candidates"] = list(acronyms)
            return {"DIET": ["District Institute of Education and Training"]}

        monkeypatch.setattr(aqs, "get_expansions_batch", _capture)
        aqs.detect_acronyms("D.I.E.T. handbook")

        # "D.I.E.T." normalizes to DIET; "handbook" is a candidate too (the
        # dictionary, not the tokenizer, is what rejects ordinary words).
        assert "DIET" in seen["candidates"]
        assert all(c == c.upper() for c in seen["candidates"])

    def test_empty_expansions_are_dropped(self, monkeypatch):
        """A row with an empty expansions list must not reach the caller —
        downstream code indexes expansions[0] unguarded."""
        import app.services.acronym_query_service as aqs

        monkeypatch.setattr(
            aqs, "get_expansions_batch",
            lambda acronyms: {"DIET": [], "SMC": ["School Management Committee"]},
        )
        detected = aqs.detect_acronyms("DIET SMC")
        assert "DIET" not in detected
        assert detected["SMC"] == ["School Management Committee"]


# ── 7. Acronym ranking is lexical and rides the lexical switch ────────────

class TestAcronymRankingFollowsLexicalRanking:
    """Acronym ranking (the title/summary boosts, expansion boosts included) rides the
    same switch as the title/summary boost: on in hybrid mode, off in semantic mode and
    when HYBRID_SEARCH_ENABLED is false.

    Doc A carries the acronym in its title; doc B carries none. In hybrid mode A's
    title earns the partial title boost, so A wins only when it is close enough for
    the boost to make up the difference. In the default fixture A (0.135) is well
    under B (0.306), so B wins.
    """

    def _run(self, service, monkeypatch, search_mode, hybrid_enabled=True,
             field_scores=None, acronyms=None, top_k=10, titles=None,
             acronym_search_enabled=True, method="inline", keyword_or=False,
             scroll=False, **request_fields):
        from app.models.api_models import PrioritizedSearchRequest

        detected = dict(ACR) if acronyms is None else acronyms
        monkeypatch.setattr(
            "app.services.prioritized_search_service.detect_acronyms",
            lambda q: dict(detected))
        monkeypatch.setattr(settings, "ACRONYM_SEARCH_ENABLED", acronym_search_enabled)
        # Pinned, so the result does not depend on the local .env.
        monkeypatch.setattr(settings, "ACRONYM_SEARCH_METHOD", method)
        monkeypatch.setattr(settings, "ACRONYM_KEYWORD_OR", keyword_or)
        monkeypatch.setattr(settings, "HYBRID_SEARCH_ENABLED", hybrid_enabled)
        monkeypatch.setattr(settings, "SPARSE_SEARCH_ENABLED", False)
        monkeypatch.setattr(
            "app.services.prioritized_search_service.embedding.embed_query",
            lambda texts: [[0.0] * 8 for _ in texts])

        titles = titles or {"A": "DIET Handbook", "B": "Nutrition guide"}
        all_results = {
            "pt-A": _point("pt-A", "A", title=titles["A"], summary="", text="a"),
            "pt-B": _point("pt-B", "B", title=titles["B"], summary="", text="b"),
        }
        field_scores = field_scores or {
            "pt-A": {"title": 0.30, "text": 0.10},
            "pt-B": {"title": 0.90},
        }
        monkeypatch.setattr(
            PrioritizedSearchService, "_parallel_batch_search",
            lambda self, **kw: (all_results, field_scores))
        if scroll:
            # The title/summary scroll sees both documents and classifies them itself,
            # for the query as typed and the written-out query.
            _fake_scroll(monkeypatch, list(all_results.values()))
        else:
            # No scroll: title matches come only from the candidate pool (the supplement).
            monkeypatch.setattr(
                PrioritizedSearchService, "_get_field_match_sources",
                lambda self, *a, **kw: {})

        return service.search(
            PrioritizedSearchRequest(query="DIET", top_k=top_k, search_mode=search_mode,
                                     **request_fields))

    @staticmethod
    def _method_info(method):
        """The acronym_info keys reported whenever acronym search is on."""
        return {"method": method,
                "method_description": PrioritizedSearchService._acronym_method_description()}

    def test_semantic_mode_ranks_by_similarity_alone(self, service, monkeypatch):
        response = self._run(service, monkeypatch, "semantic")
        assert [r.source_id for r in response.results] == ["B", "A"]
        assert response.results[0].score == pytest.approx(0.90 * WEIGHTS["title"])
        assert response.results[1].score == pytest.approx(
            0.30 * WEIGHTS["title"] + 0.10 * WEIGHTS["text"])

    def test_semantic_mode_order_is_reproducible_from_the_scores(
            self, service, monkeypatch):
        """The invariant: a score-sorting caller must not change the order."""
        response = self._run(service, monkeypatch, "semantic")
        as_returned = [r.source_id for r in response.results]
        rescored = sorted(response.results, key=lambda r: r.score, reverse=True)
        assert [r.source_id for r in rescored] == as_returned

    def test_hybrid_mode_title_boost_cannot_lift_a_much_weaker_document(
            self, service, monkeypatch):
        response = self._run(service, monkeypatch, "hybrid")
        assert [r.source_id for r in response.results] == ["B", "A"]
        relevance_a = 0.30 * WEIGHTS["title"] + 0.10 * WEIGHTS["text"]
        assert response.results[0].score == pytest.approx(0.90 * WEIGHTS["title"])
        assert response.results[1].score == pytest.approx(
            relevance_a * settings.PARTIAL_TITLE_BOOST)

    def test_hybrid_mode_title_boost_closes_a_small_gap(self, service, monkeypatch):
        response = self._run(
            service, monkeypatch, "hybrid",
            field_scores={"pt-A": {"title": 0.80, "text": 0.10}, "pt-B": {"title": 0.90}})
        assert [r.source_id for r in response.results] == ["A", "B"]

    def test_hybrid_mode_order_is_reproducible_from_the_scores(
            self, service, monkeypatch):
        """Same invariant on the other side of the switch."""
        response = self._run(service, monkeypatch, "hybrid")
        as_returned = [r.source_id for r in response.results]
        rescored = sorted(response.results, key=lambda r: r.score, reverse=True)
        assert [r.source_id for r in rescored] == as_returned

    def test_a_weak_document_is_still_returned(self, service, monkeypatch):
        """filter_score reads the pre-boost score, never the boosts."""
        response = self._run(
            service, monkeypatch, "hybrid",
            field_scores={"pt-A": {"title": 0.30}, "pt-B": {"title": 0.90}})
        assert "A" in [r.source_id for r in response.results]

    def test_hybrid_kill_switch_off_also_drops_the_boosts(self, service, monkeypatch):
        """HYBRID_SEARCH_ENABLED=false trips the same switch as semantic mode."""
        response = self._run(service, monkeypatch, "hybrid", hybrid_enabled=False)
        assert [r.source_id for r in response.results] == ["B", "A"]
        assert response.results[0].score == pytest.approx(0.90 * WEIGHTS["title"])

    def test_semantic_mode_still_reports_the_detected_acronym(
            self, service, monkeypatch):
        """Retrieval widening is NOT gated — only ranking is."""
        response = self._run(service, monkeypatch, "semantic")
        assert response.acronym_info == {
            "enabled": True, "detected": True, "mapping": dict(ACR),
            **self._method_info("inline")}

    @pytest.mark.parametrize("enabled, acronyms, expected", [
        (False, None, {"enabled": False, "detected": False}),
        (True, {}, {"enabled": True, "detected": False}),
        (True, None, {"enabled": True, "detected": True, "mapping": dict(ACR)}),
        (True, {"SSC": ["Staff Selection Commission", "Sainik School Society"]},
         {"enabled": True, "detected": True, "ambiguous": ["SSC"],
          "mapping": {"SSC": ["Staff Selection Commission", "Sainik School Society"]}}),
    ])
    def test_acronym_info_separates_disabled_from_not_detected(
            self, service, monkeypatch, enabled, acronyms, expected):
        response = self._run(service, monkeypatch, "hybrid", acronyms=acronyms,
                             acronym_search_enabled=enabled)
        if enabled:
            # The method is reported whenever acronym search is on, acronym or not.
            expected = {**expected, **self._method_info("inline")}
        assert response.acronym_info == expected

    @pytest.mark.parametrize("method, keyword_or, phrase", [
        ("max", False, "searched separately"),
        ("max", True, "searched separately"),
        ("inline", True, "once more for each meaning"),
        ("inline", False, "one keyword search"),
    ])
    def test_acronym_info_describes_the_method(
            self, service, monkeypatch, method, keyword_or, phrase):
        info = self._run(service, monkeypatch, "hybrid", method=method,
                         keyword_or=keyword_or).acronym_info
        assert info["method"] == method
        assert phrase in info["method_description"]


# ── scoring_context reports how an acronym query was actually scored ──────

class TestAcronymScoringContext:
    """search_config.scoring_context (debug only) must describe the scoring that
    ran: the field boosts and the acronym ranking block."""

    def _context(self, service, monkeypatch, search_mode="hybrid", acronyms=None, **run):
        response = TestAcronymRankingFollowsLexicalRanking()._run(
            service, monkeypatch, search_mode, acronyms=acronyms, include_scoring_debug=True,
            **run)
        return response.search_config["scoring_context"]

    @pytest.mark.parametrize("method, dense_variants", [("inline", 1), ("max", 2)])
    def test_acronym_query_reports_field_boosts_and_ranking(
            self, service, monkeypatch, method, dense_variants):
        ctx = self._context(service, monkeypatch, method=method)
        # The same boosts as any other query, plus the expansion boosts.
        assert ctx["boost_config"] == {
            "exact_title_boost": settings.EXACT_TITLE_BOOST,
            "partial_title_boost": settings.PARTIAL_TITLE_BOOST,
            "exact_summary_boost": settings.EXACT_SUMMARY_BOOST,
            "partial_summary_boost": settings.PARTIAL_SUMMARY_BOOST,
            "expansion_exact_title_boost": settings.ACRONYM_EXPANSION_EXACT_TITLE_BOOST,
            "expansion_partial_title_boost": settings.ACRONYM_EXPANSION_PARTIAL_TITLE_BOOST,
            "expansion_exact_summary_boost": settings.ACRONYM_EXPANSION_EXACT_SUMMARY_BOOST,
            "expansion_partial_summary_boost": settings.ACRONYM_EXPANSION_PARTIAL_SUMMARY_BOOST,
        }
        assert ctx["acronym_ranking"] == {
            "applied": True,
            "method": method,
            "final_score": "score before boosts x title and summary multipliers "
                           "(see boost_config), capped at 1.0",
            "dense_variants": dense_variants,
        }

    def test_keyword_or_texts_are_reported(self, service, monkeypatch):
        """One keyword search per text: the query as typed, then the full name."""
        texts = self._context(service, monkeypatch, keyword_or=True)["keyword_or_texts"]
        assert len(texts) == 2
        assert texts[1] == ACR["DIET"][0]
        assert "keyword_or_texts" not in self._context(service, monkeypatch)

    @pytest.mark.parametrize("method, dense_variants", [("inline", 1), ("max", 3)])
    def test_keyword_or_searches_every_meaning(
            self, service, monkeypatch, method, dense_variants):
        """The dense searches stop at ACRONYM_MAX_DENSE_VARIANTS; the keyword searches do not."""
        monkeypatch.setattr(settings, "ACRONYM_MAX_DENSE_VARIANTS", 3)
        meanings = ["District Institute of Education and Training",
                    "Daily Intake Energy Table", "Dietary Information Exchange Tool"]
        ctx = self._context(service, monkeypatch, acronyms={"DIET": meanings},
                            method=method, keyword_or=True)
        assert ctx["keyword_or_texts"][1:] == meanings
        assert ctx["acronym_ranking"]["dense_variants"] == dense_variants

    @pytest.mark.parametrize("search_mode, hybrid_enabled, reason", [
        ("semantic", True, "search_mode is semantic"),
        ("hybrid", False, "HYBRID_SEARCH_ENABLED is false"),
    ])
    def test_ranking_off_is_reported_with_its_reason(
            self, service, monkeypatch, search_mode, hybrid_enabled, reason):
        ctx = self._context(service, monkeypatch, search_mode=search_mode,
                            hybrid_enabled=hybrid_enabled)
        # Boosts off: every multiplier reads 1.0, and no expansion boosts.
        assert ctx["boost_config"] == {
            "exact_title_boost": 1.0,
            "partial_title_boost": 1.0,
            "exact_summary_boost": 1.0,
            "partial_summary_boost": 1.0,
        }
        assert ctx["acronym_ranking"] == {
            "applied": False, "reason": reason, "dense_variants": 1,
        }

    def test_ordinary_query_keeps_configured_boosts(self, service, monkeypatch):
        ctx = self._context(service, monkeypatch, acronyms={})
        assert ctx["boost_config"] == {
            "exact_title_boost": settings.EXACT_TITLE_BOOST,
            "partial_title_boost": settings.PARTIAL_TITLE_BOOST,
            "exact_summary_boost": settings.EXACT_SUMMARY_BOOST,
            "partial_summary_boost": settings.PARTIAL_SUMMARY_BOOST,
        }
        assert "acronym_ranking" not in ctx


# ── Acronym debug fields: the expansion match and multiplier ──────────────

class TestAcronymDebugFields:
    """title/summary_expansion_match and _multiplier sit next to title/summary_match and
    _multiplier: top-level, debug-only, and kept out of field_scores."""

    @staticmethod
    def _row(**extra):
        return {"id": "p1", "payload": {"source_id": "S", "title": "T", "text": "x"},
                "field_scores": {"title": 0.4, "title_match": "partial",
                                 "title_expansion_match": "exact",
                                 "summary_expansion_match": None},
                "weighted_score": 0.8, "title_multiplier": 1.5,
                "title_expansion_multiplier": 1.7, **extra}

    def test_expansion_fields_surface_with_debug(self, service):
        item = service._build_result_items([self._row()], True)[0]
        assert (item.title_match, item.title_expansion_match) == ("partial", "exact")
        assert (item.title_multiplier, item.title_expansion_multiplier) == (1.5, 1.7)
        # Not set on the row: no boost.
        assert (item.summary_expansion_match, item.summary_expansion_multiplier) == (None, 1.0)
        assert item.field_scores == {"title": 0.4}

    def test_fields_hidden_without_debug(self, service):
        item = service._build_result_items([self._row()], False)[0]
        assert item.title_expansion_match is None
        assert item.title_expansion_multiplier is None
        assert item.summary_expansion_multiplier is None
        assert item.field_scores == {"title": 0.4}

    def test_injected_row_keeps_its_match_source(self, service):
        item = service._build_result_items(
            [self._row(match_source="title_keyword_match")], True)[0]
        assert item.match_source == "title_keyword_match"

    @pytest.mark.parametrize("name", [
        "pre_floor_score", "acronym_pre_boost_score", "keyword_score_acronym",
        "keyword_score_expansion", "acronym_in_body_match", "acronym_in_body_multiplier",
    ])
    def test_removed_fields_are_gone_from_the_response(self, name):
        from app.models.api_models import SearchResultItem

        assert name not in SearchResultItem.model_fields


# ── Acronym queries: the final score from the four multipliers ────────────

class TestAcronymFieldBoosts:
    """The score of an acronym query rebuilds from the score before boosts and the four
    title/summary multipliers, capped at 1.0 (acronym_ranking.final_score)."""

    def test_score_rebuilds_from_the_multipliers(self, service, monkeypatch):
        _set_expansion_boosts(monkeypatch)
        scores = {"pt-A": {"title": 0.30, "text": 0.10}, "pt-B": {"title": 0.60}}
        response = TestAcronymRankingFollowsLexicalRanking()._run(
            service, monkeypatch, "hybrid", scroll=True, include_scoring_debug=True,
            titles={"A": "DIET Handbook", "B": "District Institute of Education and Training"},
            field_scores={pid: dict(s) for pid, s in scores.items()})
        by_source = {r.source_id: r for r in response.results}
        assert by_source["A"].title_multiplier == settings.PARTIAL_TITLE_BOOST
        assert by_source["B"].title_expansion_multiplier == (
            settings.ACRONYM_EXPANSION_EXACT_TITLE_BOOST)
        for source_id, r in by_source.items():
            assert r.score == pytest.approx(min(
                _weighted_dense(scores[f"pt-{source_id}"])
                * r.title_multiplier * r.summary_multiplier
                * r.title_expansion_multiplier * r.summary_expansion_multiplier, 1.0))
