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
   as typed, anywhere in the field); no acronym-only, full-name or body boost.
   Retrieval and the relevance blend cover expansions. Expansion phrase matching
   (_phrase_in_text, on content words) is still used to decide whether the query
   already spells out an expansion.
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
    typed, anywhere in the field, any case. The acronym alone and the full name are
    not matched separately."""

    @pytest.mark.parametrize("title, expected", [
        ("DIET Stakeholder Identity Map", "partial"),
        ("DIET", "exact"),
        # The full name is not matched in titles.
        ("Strengthening of District Institutes of Education and Training", None),
        ("Establishing Discipline Through Clear School Rules", None),
        # A plain substring, as for any query.
        ("Dietary guidelines for schools", "partial"),
        ("source_doc_COE_AM4C2_DIET_Empowerment_Design.xlsx", "partial"),
    ])
    def test_title_match(self, service, title, expected):
        assert service._classify_text_match("diet", title.lower()) == expected

    def test_only_the_expansion_title_boost_differs_from_acronym_search_off(
            self, service, monkeypatch):
        titles = {"A": "DIET Handbook", "B": "District Institute of Education and Training"}
        run = TestAcronymRankingFollowsLexicalRanking()._run

        def multipliers(response):
            return {r.source_id: (r.title_multiplier, r.summary_multiplier)
                    for r in response.results}

        on = run(service, monkeypatch, "hybrid", titles=titles, include_scoring_debug=True)
        off = run(service, monkeypatch, "hybrid", titles=titles, include_scoring_debug=True,
                  acronym_search_enabled=False)
        # "DIET" written out is the expansion itself, so B's title is the expanded query.
        assert multipliers(on) == {
            "A": (settings.PARTIAL_TITLE_BOOST, 1.0),
            "B": (settings.ACRONYM_EXPANDED_QUERY_TITLE_BOOST, 1.0)}
        assert multipliers(off) == {"A": (settings.PARTIAL_TITLE_BOOST, 1.0), "B": (1.0, 1.0)}


class TestExpansionTitleBoost:
    """Acronym queries: the ordinary title rule on the query written out ("PTM Doc" ->
    "Parent Teacher Meeting Doc"): ACRONYM_EXPANDED_QUERY_TITLE_BOOST when the title
    equals it, ACRONYM_EXPANSION_TITLE_BOOST when it contains it; the higher of that
    and the ordinary title boost applies, never both."""

    PTM = {"PTM": ["Parent Teacher Meeting"]}

    @staticmethod
    def _rows(*titles):
        return [{"payload": {"source_id": str(i), "title": t}} for i, t in enumerate(titles)]

    def test_match_levels(self, service):
        matches = service._expansion_field_matches(
            self._rows("Parent Teacher Meeting Doc",            # equals, any case
                       "parent teacher meeting doc in school",  # contains
                       "Parent teachers meeting Doc",           # plural: no match
                       "Doc on the Parent Teacher Meeting",     # other order: no match
                       "Parent Teacher Meeting Guidelines",     # expansion only: no match
                       "PTM Doc"),                              # the acronym: ordinary boost
            ["Parent Teacher Meeting Doc"])
        assert matches == {"0": "exact", "1": "partial"}

    def test_best_match_over_meanings(self, service):
        matches = service._expansion_field_matches(
            self._rows("Micro Improvement Plan", "Micro Improvement Project for schools"),
            ["Micro Improvement Project", "Micro Improvement Plan"])
        assert matches == {"0": "exact", "1": "partial"}

    def _search(self, service, monkeypatch, titles, query, field="title"):
        from app.models.api_models import PrioritizedSearchRequest

        monkeypatch.setattr("app.services.prioritized_search_service.detect_acronyms",
                            lambda q: dict(self.PTM))
        monkeypatch.setattr(settings, "ACRONYM_SEARCH_ENABLED", True)
        monkeypatch.setattr(settings, "HYBRID_SEARCH_ENABLED", True)
        monkeypatch.setattr(settings, "SPARSE_SEARCH_ENABLED", False)
        monkeypatch.setattr("app.services.prioritized_search_service.embedding.embed_query",
                            lambda texts: [[0.0] * 8 for _ in texts])
        # The texts go in `field`; the other of title/summary matches nothing.
        neutral = {"title": "Annual report", "summary": ""}
        all_results = {f"pt-{s}": _point(f"pt-{s}", s, text="x", **{**neutral, field: t})
                       for s, t in titles.items()}
        field_scores = {f"pt-{s}": {"title": 0.5} for s in titles}
        monkeypatch.setattr(PrioritizedSearchService, "_parallel_batch_search",
                            lambda self, **kw: (all_results, field_scores))
        monkeypatch.setattr(PrioritizedSearchService, "_get_field_match_sources",
                            lambda self, *a, **kw: {})
        response = service.search(PrioritizedSearchRequest(
            query=query, top_k=10, search_mode="hybrid", include_scoring_debug=True))
        return {r.source_id: getattr(r, f"{field}_multiplier") for r in response.results}

    def test_search_applies_the_higher_boost_once(self, service, monkeypatch):
        multipliers = self._search(service, monkeypatch, {
            "written_out": "Parent Teacher Meeting Doc",
            "contains_written_out": "Parent Teacher Meeting Doc in school",
            "plural": "Parent teachers meeting Doc",
            "expansion_only": "Parent Teacher Meeting Guidelines",
            "acronym": "PTM Doc",
            "contains_query": "PTM Doc in school",
            "both": "PTM Doc - Parent Teacher Meeting Doc",
            "none": "Annual report",
        }, "PTM Doc")
        assert multipliers == {
            "written_out": settings.ACRONYM_EXPANDED_QUERY_TITLE_BOOST,
            "contains_written_out": settings.ACRONYM_EXPANSION_TITLE_BOOST,
            "plural": 1.0,
            "expansion_only": 1.0,
            "acronym": settings.EXACT_TITLE_BOOST,          # the whole title is the query
            "contains_query": settings.PARTIAL_TITLE_BOOST,
            # Contains the query (x1.5) and the written-out query (x1.4): the higher only.
            "both": settings.PARTIAL_TITLE_BOOST,
            "none": 1.0,
        }

    def test_boost_values_come_from_settings(self, service, monkeypatch):
        monkeypatch.setattr(settings, "ACRONYM_EXPANDED_QUERY_TITLE_BOOST", 1.9)
        monkeypatch.setattr(settings, "ACRONYM_EXPANSION_TITLE_BOOST", 1.0)
        multipliers = self._search(service, monkeypatch, {
            "written_out": "Parent Teacher Meeting Doc",
            "contains_written_out": "Parent Teacher Meeting Doc in school",
        }, "PTM Doc")
        assert multipliers == {"written_out": 1.9, "contains_written_out": 1.0}

    def test_summary_gets_the_same_rule_with_summary_boosts(self, service, monkeypatch):
        monkeypatch.setattr(settings, "ACRONYM_EXPANDED_QUERY_SUMMARY_BOOST", 1.3)
        monkeypatch.setattr(settings, "ACRONYM_EXPANSION_SUMMARY_BOOST", 1.25)
        multipliers = self._search(service, monkeypatch, {
            "written_out": "parent teacher meeting doc",
            "contains_written_out": "Notes on the Parent Teacher Meeting Doc",
            "expansion_only": "Parent Teacher Meeting Guidelines",
            "acronym": "About the PTM Doc",
            "both": "PTM Doc - Parent Teacher Meeting Doc",
            "none": "Annual report",
        }, "PTM Doc", field="summary")
        assert multipliers == {
            "written_out": 1.3,
            "contains_written_out": 1.25,
            "expansion_only": 1.0,
            "acronym": settings.PARTIAL_SUMMARY_BOOST,
            # Contains the query (x1.2) and the written-out query (x1.25): the higher only.
            "both": 1.25,
            "none": 1.0,
        }

    @pytest.mark.parametrize("name", ["ACRONYM_EXPANDED_QUERY_TITLE_BOOST",
                                      "ACRONYM_EXPANSION_TITLE_BOOST",
                                      "ACRONYM_EXPANDED_QUERY_SUMMARY_BOOST",
                                      "ACRONYM_EXPANSION_SUMMARY_BOOST",
                                      "ACRONYM_TEXT_BOOST",
                                      "ACRONYM_EXPANSION_TEXT_BOOST"])
    def test_values_below_one_fail_at_startup(self, monkeypatch, name):
        from app.config import Settings

        monkeypatch.setenv(name, "0.9")
        with pytest.raises(ValueError, match=name):
            Settings()


class TestAcronymTextBoost:
    """Acronym queries: ACRONYM_TEXT_BOOST when the body contains the query as typed
    ("PTM Doc"), ACRONYM_EXPANSION_TEXT_BOOST when it contains the query written out
    ("Parent Teacher Meeting Doc"), any case; the higher of the two, never both. Only
    top_k documents without a title boost are checked."""

    PTM = {"PTM": ["Parent Teacher Meeting"]}

    def test_case_variants(self, service):
        assert service._case_variants(" PTM Doc ") == [
            "PTM DOC", "PTM Doc", "Ptm Doc", "Ptm doc", "ptm doc"]
        assert service._case_variants("  ") == []

    def test_one_qdrant_call_returns_the_sources_per_phrase_set(self, service, monkeypatch):
        calls = []

        def fake_batch(collection_name, requests):
            calls.append(requests)
            return [types.SimpleNamespace(points=[
                types.SimpleNamespace(payload={"source_id": "1"}),
                types.SimpleNamespace(payload={"source_id": "1"}),   # second chunk, same doc
            ]), types.SimpleNamespace(points=[])]

        monkeypatch.setattr("app.services.acronym_ranking.qdrant_client.query_batch_points",
                            fake_batch)
        found = service._sources_with_text_containing(
            ["1", "2"], [["PTM Doc"], ["Parent Teacher Meeting Doc"]])
        assert found == [{"1"}, set()]
        assert len(calls) == 1 and len(calls[0]) == 2
        request = calls[0][0]
        assert request.filter.must[0].key == "source_id"
        assert request.filter.must[0].match.any == ["1", "2"]
        # Any case: every spelling goes in as a "should" substring match on the body.
        assert {c.key for c in request.filter.should} == {"text"}
        assert {c.match.text for c in request.filter.should} == set(
            service._case_variants("PTM Doc"))

    def test_failed_check_returns_none(self, service, monkeypatch):
        def broken(**kwargs):
            raise RuntimeError("qdrant down")

        monkeypatch.setattr("app.services.acronym_ranking.qdrant_client.query_batch_points",
                            broken)
        assert service._sources_with_text_containing(["1"], [["PTM Doc"]]) is None

    def _search(self, service, monkeypatch, docs, query="PTM Doc", acronyms=None):
        """docs: {source_id: (title, body)}. Returns ({source_id: (match, multiplier)},
        the source_ids the text check was asked about)."""
        from app.models.api_models import PrioritizedSearchRequest

        monkeypatch.setattr("app.services.prioritized_search_service.detect_acronyms",
                            lambda q: dict(self.PTM if acronyms is None else acronyms))
        monkeypatch.setattr(settings, "ACRONYM_SEARCH_ENABLED", True)
        monkeypatch.setattr(settings, "HYBRID_SEARCH_ENABLED", True)
        monkeypatch.setattr(settings, "SPARSE_SEARCH_ENABLED", False)
        monkeypatch.setattr("app.services.prioritized_search_service.embedding.embed_query",
                            lambda texts: [[0.0] * 8 for _ in texts])
        all_results = {f"pt-{s}": _point(f"pt-{s}", s, text=body, title=title, summary="")
                       for s, (title, body) in docs.items()}
        field_scores = {f"pt-{s}": {"title": 0.5} for s in docs}
        monkeypatch.setattr(PrioritizedSearchService, "_parallel_batch_search",
                            lambda self, **kw: (all_results, field_scores))
        monkeypatch.setattr(PrioritizedSearchService, "_get_field_match_sources",
                            lambda self, *a, **kw: {})
        asked = []

        def fake_text_check(self, source_ids, phrase_sets):
            asked.extend(source_ids)
            return [{s for s in source_ids
                     if any(p.lower() in docs[s][1].lower() for p in phrases)}
                    for phrases in phrase_sets]

        monkeypatch.setattr(PrioritizedSearchService, "_sources_with_text_containing",
                            fake_text_check)
        response = service.search(PrioritizedSearchRequest(
            query=query, top_k=10, search_mode="hybrid", include_scoring_debug=True))
        return ({r.source_id: (r.acronym_in_body_match, r.acronym_in_body_multiplier)
                 for r in response.results}, asked)

    DOCS = {
        "query": ("Annual report", "Notes from the ptm doc review."),
        "written_out": ("Annual report", "The PARENT TEACHER MEETING DOC is attached."),
        "both": ("Annual report", "PTM Doc - Parent Teacher Meeting Doc"),
        "acronym_only": ("Annual report", "The PTM was held in July."),
        "expansion_only": ("Annual report", "Parent Teacher Meeting guidelines."),
        "titled": ("PTM Doc for schools", "Notes from the PTM Doc review."),
        "none": ("Annual report", "Water and roads."),
    }

    def test_body_contains_the_query_or_the_query_written_out(self, service, monkeypatch):
        monkeypatch.setattr(settings, "ACRONYM_TEXT_BOOST", 1.2)
        monkeypatch.setattr(settings, "ACRONYM_EXPANSION_TEXT_BOOST", 1.2)
        boosts, asked = self._search(service, monkeypatch, self.DOCS)
        assert boosts == {
            "query": ("query", 1.2),
            "written_out": ("expansion", 1.2),
            "both": ("query", 1.2),                # equal boosts: applied once
            "acronym_only": (None, 1.0),           # the bare acronym is not the query
            "expansion_only": (None, 1.0),         # the full name alone is not the query
            "titled": (None, 1.0),                 # has a title boost: not checked
            "none": (None, 1.0),
        }
        assert "titled" not in asked

    def test_higher_boost_applies_never_both(self, service, monkeypatch):
        monkeypatch.setattr(settings, "ACRONYM_TEXT_BOOST", 1.2)
        monkeypatch.setattr(settings, "ACRONYM_EXPANSION_TEXT_BOOST", 1.5)
        boosts, _ = self._search(service, monkeypatch, self.DOCS)
        assert boosts["both"] == ("expansion", 1.5)
        assert boosts["query"] == ("query", 1.2)

    def test_one_side_off(self, service, monkeypatch):
        monkeypatch.setattr(settings, "ACRONYM_TEXT_BOOST", 1.0)
        monkeypatch.setattr(settings, "ACRONYM_EXPANSION_TEXT_BOOST", 1.3)
        boosts, _ = self._search(service, monkeypatch, self.DOCS)
        assert boosts["query"] == (None, 1.0)
        assert boosts["both"] == ("expansion", 1.3)

    def test_off_by_default_makes_no_check(self, service, monkeypatch):
        monkeypatch.setattr(settings, "ACRONYM_TEXT_BOOST", 1.0)
        monkeypatch.setattr(settings, "ACRONYM_EXPANSION_TEXT_BOOST", 1.0)
        boosts, asked = self._search(service, monkeypatch, self.DOCS)
        assert asked == []
        assert set(boosts.values()) == {(None, None)}

    def test_ordinary_queries_get_no_text_boost(self, service, monkeypatch):
        monkeypatch.setattr(settings, "ACRONYM_TEXT_BOOST", 1.2)
        monkeypatch.setattr(settings, "ACRONYM_EXPANSION_TEXT_BOOST", 1.2)
        boosts, asked = self._search(service, monkeypatch, self.DOCS, acronyms={})
        assert asked == []
        assert set(boosts.values()) == {(None, None)}


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
        # the real relevance the pipeline measured is kept, not just dropped
        assert entry["pre_floor_score"] == pytest.approx(0.31)

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
        # never-scored path has nothing measured to preserve; reused path does
        assert "pre_floor_score" not in floor_only
        assert with_real["pre_floor_score"] == pytest.approx(0.31)

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

    def test_reused_doc_reports_its_pre_floor_score(self, service, monkeypatch):
        """A document scored before the threshold takes the floor, its real score is
        kept as pre_floor_score, and it shows no pre-boost score."""
        self._patch_scroll(monkeypatch, [])
        real = {"id": "pt-f", "payload": {"source_id": "F"}, "weighted_score": 0.31,
                "acronym_pre_boost_score": 0.31, "field_scores": {}}
        entry = service._fetch_field_match_docs(
            ["F"], {"F": "partial"}, "title",
            settings.EXACT_TITLE_BOOST, settings.PARTIAL_TITLE_BOOST,
            prethreshold_by_source={"F": real},
        )[0]
        assert entry["weighted_score"] == pytest.approx(self.FLOOR * settings.PARTIAL_TITLE_BOOST)
        assert entry["pre_floor_score"] == 0.31
        assert "acronym_pre_boost_score" not in entry


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
    """Acronym ranking (relevance blend, title/summary boosts) rides the same
    switch as the title/summary boost: on in hybrid mode, off in semantic mode and
    when HYBRID_SEARCH_ENABLED is false.

    Doc A carries the acronym in its title; doc B carries none. In hybrid mode A's
    title earns the partial title boost, so A wins only when it is close enough for
    the boost to make up the difference. In the default fixture A (0.135) is well
    under B (0.306), so B wins.
    """

    def _run(self, service, monkeypatch, search_mode, hybrid_enabled=True,
             field_scores=None, acronyms=None, top_k=10, titles=None,
             acronym_search_enabled=True, **request_fields):
        from app.models.api_models import PrioritizedSearchRequest

        detected = dict(ACR) if acronyms is None else acronyms
        monkeypatch.setattr(
            "app.services.prioritized_search_service.detect_acronyms",
            lambda q: dict(detected))
        monkeypatch.setattr(settings, "ACRONYM_SEARCH_ENABLED", acronym_search_enabled)
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
        # No scroll: title matches come only from the candidate pool (the supplement).
        monkeypatch.setattr(
            PrioritizedSearchService, "_get_field_match_sources",
            lambda self, *a, **kw: {})

        return service.search(
            PrioritizedSearchRequest(query="DIET", top_k=top_k, search_mode=search_mode,
                                     **request_fields))

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
        assert response.acronym_info == {"enabled": True, "detected": True, "mapping": dict(ACR)}

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
        assert response.acronym_info == expected


# ── scoring_context reports how an acronym query was actually scored ──────

class TestAcronymScoringContext:
    """search_config.scoring_context (debug only) must describe the scoring that
    ran: the field boosts and the acronym ranking block."""

    def _context(self, service, monkeypatch, search_mode="hybrid", acronyms=None):
        response = TestAcronymRankingFollowsLexicalRanking()._run(
            service, monkeypatch, search_mode, acronyms=acronyms, include_scoring_debug=True)
        return response.search_config["scoring_context"]

    def test_acronym_query_reports_field_boosts_and_ranking(self, service, monkeypatch):
        ctx = self._context(service, monkeypatch)
        # The same boosts as any other query, plus the expansion and text boosts.
        assert ctx["boost_config"] == {
            "mode": "title_summary_boost",
            "exact_title_boost": settings.EXACT_TITLE_BOOST,
            "partial_title_boost": settings.PARTIAL_TITLE_BOOST,
            "exact_summary_boost": settings.EXACT_SUMMARY_BOOST,
            "partial_summary_boost": settings.PARTIAL_SUMMARY_BOOST,
            "expanded_query_title_boost": settings.ACRONYM_EXPANDED_QUERY_TITLE_BOOST,
            "expansion_title_boost": settings.ACRONYM_EXPANSION_TITLE_BOOST,
            "expanded_query_summary_boost": settings.ACRONYM_EXPANDED_QUERY_SUMMARY_BOOST,
            "expansion_summary_boost": settings.ACRONYM_EXPANSION_SUMMARY_BOOST,
            "text_boost": settings.ACRONYM_TEXT_BOOST,
            "expansion_text_boost": settings.ACRONYM_EXPANSION_TEXT_BOOST,
        }
        ranking = ctx["acronym_ranking"]
        assert ranking["applied"] is True
        assert ranking["expansion_score_weight"] == settings.ACRONYM_EXPANSION_SCORE_WEIGHT
        assert ranking["dense_variants"] == 2
        assert ranking["rescore"] == {
            "applied": True, "candidate_pool": 2, "rescored": 2,
            "limit": settings.ACRONYM_RESCORE_POOL_LIMIT, "capped": False,
            "normalization": "self",
        }
        assert "body_check" not in ranking

    def test_capped_rescore_is_reported(self, service, monkeypatch):
        monkeypatch.setattr(settings, "ACRONYM_RESCORE_POOL_LIMIT", 1)
        monkeypatch.setattr(settings, "HYBRID_FUSION_METHOD", "weighted")
        rescore = self._context(service, monkeypatch)["acronym_ranking"]["rescore"]
        assert rescore["capped"] is True
        assert rescore["rescored"] == 1
        assert rescore["normalization"] == "full_pool"

    def test_semantic_mode_reports_ranking_off(self, service, monkeypatch):
        ctx = self._context(service, monkeypatch, search_mode="semantic")
        assert ctx["boost_config"]["mode"] == "off"
        assert ctx["acronym_ranking"] == {
            "applied": False, "reason": "search_mode is semantic", "dense_variants": 2,
        }

    def test_ordinary_query_keeps_configured_boosts(self, service, monkeypatch):
        ctx = self._context(service, monkeypatch, acronyms={})
        assert ctx["boost_config"] == {
            "mode": "title_summary_boost",
            "exact_title_boost": settings.EXACT_TITLE_BOOST,
            "partial_title_boost": settings.PARTIAL_TITLE_BOOST,
            "exact_summary_boost": settings.EXACT_SUMMARY_BOOST,
            "partial_summary_boost": settings.PARTIAL_SUMMARY_BOOST,
        }
        assert "acronym_ranking" not in ctx


# ── Acronym debug field: pre-boost score ──────────────────────────────────

class TestAcronymDebugFields:
    """acronym_pre_boost_score is set only on rows the relevance blend rescored, and
    is debug-only."""

    @staticmethod
    def _row(**extra):
        return {"id": "p1", "payload": {"source_id": "S", "title": "T", "text": "x"},
                "field_scores": {}, "weighted_score": 0.8,
                "match_source": "title_keyword_match", **extra}

    def test_ordinary_injected_row_has_no_acronym_fields(self, service):
        item = service._build_result_items([self._row()], True)[0]
        assert item.acronym_pre_boost_score is None
        assert item.match_source == "title_keyword_match"

    def test_rescored_row_shows_its_acronym_pre_boost_score(self, service):
        item = service._build_result_items([self._row(acronym_pre_boost_score=0.5)], True)[0]
        assert item.acronym_pre_boost_score == 0.5

    def test_fields_hidden_without_debug(self, service):
        item = service._build_result_items([self._row(acronym_pre_boost_score=0.5)], False)[0]
        assert item.acronym_pre_boost_score is None


# ── Acronym queries: the title/summary boosts of any query, no body boost ──

class TestAcronymFieldBoosts:
    """Acronym queries get only the title/summary boosts every query gets, so the
    score rebuilds from acronym_pre_boost_score and those two multipliers."""

    def _run(self, service, monkeypatch, include_scoring_debug=True):
        response = TestAcronymRankingFollowsLexicalRanking()._run(
            service, monkeypatch, "hybrid", include_scoring_debug=include_scoring_debug,
            field_scores={"pt-A": {"title": 0.30, "text": 0.10, SPARSE: 5.0},
                          "pt-B": {"title": 0.90, SPARSE: 2.0}})
        return {r.source_id: r for r in response.results}

    def test_score_rebuilds_from_the_acronym_pre_boost_score(self, service, monkeypatch):
        by_source = self._run(service, monkeypatch)
        # A's title "DIET Handbook" contains the query; B's does not.
        assert by_source["A"].title_multiplier == settings.PARTIAL_TITLE_BOOST
        assert by_source["B"].title_multiplier == 1.0
        for r in by_source.values():
            assert r.score == pytest.approx(min(
                r.acronym_pre_boost_score * r.title_multiplier * r.summary_multiplier, 1.0))

    def test_acronym_pre_boost_score_hidden_without_debug(self, service, monkeypatch):
        by_source = self._run(service, monkeypatch, include_scoring_debug=False)
        assert all(r.acronym_pre_boost_score is None for r in by_source.values())


# ── The acronym rescore cap is skipped in RRF mode ────────────────────────

class TestRescoreCapFollowsFusionMethod:
    """RRF scores a point by its rank within the list it is given and ignores
    normalization_reference, so a capped subset would be ranked against itself
    while the remainder kept full-pool scores. RRF mode therefore rescores the
    whole pool; weighted mode keeps the cap."""

    def _rescored_pool(self, service, monkeypatch, fusion_method):
        seen = {}

        def spy(self, pool_ids, *args, **kwargs):
            seen["pool_ids"] = list(pool_ids)
            seen["normalization_reference"] = kwargs.get("normalization_reference")
            return {}

        monkeypatch.setattr(settings, "HYBRID_FUSION_METHOD", fusion_method)
        monkeypatch.setattr(settings, "ACRONYM_RESCORE_POOL_LIMIT", 1)
        monkeypatch.setattr(PrioritizedSearchService, "_blended_acronym_relevance", spy)
        TestAcronymRankingFollowsLexicalRanking()._run(service, monkeypatch, "hybrid")
        return seen

    def test_weighted_mode_caps_the_pool(self, service, monkeypatch):
        seen = self._rescored_pool(service, monkeypatch, "weighted")
        assert len(seen["pool_ids"]) == 1
        assert seen["normalization_reference"] is not None

    def test_rrf_mode_rescores_the_full_pool(self, service, monkeypatch):
        seen = self._rescored_pool(service, monkeypatch, "rrf")
        assert sorted(seen["pool_ids"]) == ["pt-A", "pt-B"]
        assert seen["normalization_reference"] is None

    def test_sent_but_empty_bm25_still_records_the_full_pool_range(
            self, service, monkeypatch):
        """BM25 was issued but matched nothing: the preliminary ranking must
        take the same hybrid path as the rescore, or it records no min/max and
        the capped subset self-normalizes."""
        from app.models.api_models import PrioritizedSearchRequest

        seen = {}

        def spy(self, pool_ids, *args, **kwargs):
            seen["normalization_reference"] = kwargs.get("normalization_reference")
            return {}

        all_results = {
            "pt-A": _point("pt-A", "A", title="DIET Handbook", summary="", text="a"),
            "pt-B": _point("pt-B", "B", title="Nutrition guide", summary="", text="b"),
        }
        field_scores = {"pt-A": {"title": 0.30, "text": 0.10}, "pt-B": {"title": 0.90}}

        monkeypatch.setattr(
            "app.services.prioritized_search_service.detect_acronyms", lambda q: dict(ACR))
        monkeypatch.setattr(settings, "ACRONYM_SEARCH_ENABLED", True)
        monkeypatch.setattr(settings, "HYBRID_SEARCH_ENABLED", True)
        monkeypatch.setattr(settings, "SPARSE_SEARCH_ENABLED", True)
        monkeypatch.setattr(settings, "HYBRID_FUSION_METHOD", "weighted")
        monkeypatch.setattr(settings, "ACRONYM_RESCORE_POOL_LIMIT", 1)
        monkeypatch.setattr(
            "app.services.prioritized_search_service.embedding.embed_query",
            lambda texts: [[0.0] * 8 for _ in texts])
        monkeypatch.setattr(
            "app.core.clients.sparse_encoder.generate_sparse_vector",
            lambda text: ([1], [1.0]))
        # Sent (third value True), but no field_scores entry carries a sparse score.
        monkeypatch.setattr(
            PrioritizedSearchService, "_hybrid_batch_search",
            lambda self, **kw: (all_results, field_scores, True))
        monkeypatch.setattr(
            PrioritizedSearchService, "_get_field_match_sources", lambda self, *a, **kw: {})
        monkeypatch.setattr(PrioritizedSearchService, "_blended_acronym_relevance", spy)

        service.search(PrioritizedSearchRequest(query="DIET", top_k=10, search_mode="hybrid"))

        ref = seen["normalization_reference"]
        assert ref and "dense_min" in ref and "dense_max" in ref


# ── keyword_score_acronym / keyword_score_expansion are report-only ───────

class TestKeywordBreakdown:
    def test_scores_exactly_the_given_points(self, service, monkeypatch):
        import app.services.prioritized_search_service as pss

        def point(pid, score):
            return types.SimpleNamespace(id=pid, score=score)

        seen = []

        def batch(collection_name, requests):
            seen.extend(requests)
            return [types.SimpleNamespace(points=[point("a", 3.0)]),                   # acronym form
                    types.SimpleNamespace(points=[point("a", 7.0), point("b", 2.0)]),  # meaning 1
                    types.SimpleNamespace(points=[point("a", 8.0)])]                   # meaning 2

        monkeypatch.setattr(pss.qdrant_client, "query_batch_points", batch)
        monkeypatch.setattr("app.core.clients.sparse_encoder.generate_sparse_vector",
                            lambda text: ([1], [1.0]))
        scores = service._keyword_scores_for_points(
            ["a", "b", "c"], {"acronym": ["ptm doc"],
                              "expansion": ["meaning one doc", "meaning two doc"]})
        # Raw scores; the higher one over several meanings; no shared word -> 0.0.
        assert scores == {"acronym": {"a": 3.0, "b": 0.0, "c": 0.0},
                          "expansion": {"a": 8.0, "b": 2.0, "c": 0.0}}
        # Every request is limited to exactly these points.
        assert all(r.filter.must[0].has_id == ["a", "b", "c"] and r.limit == 3 for r in seen)

    def test_failure_leaves_the_fields_empty(self, service, monkeypatch):
        import app.services.prioritized_search_service as pss

        def boom(**kwargs):
            raise RuntimeError("qdrant down")

        monkeypatch.setattr(pss.qdrant_client, "query_batch_points", boom)
        monkeypatch.setattr("app.core.clients.sparse_encoder.generate_sparse_vector",
                            lambda text: ([1], [1.0]))
        assert service._keyword_scores_for_points(["a"], {"acronym": ["ptm"]}) == {}

    def test_fields_are_left_out_of_the_response_unless_set(self):
        from app.models.api_models import SearchResultItem

        base = dict(id="1", text="t", metadata={}, source_id="S", score=0.5, field_scores={})
        assert "keyword_score_acronym" not in SearchResultItem(**base).model_dump()
        shown = SearchResultItem(**base, keyword_score_acronym=3.0,
                                 keyword_score_expansion=0.0).model_dump()
        assert (shown["keyword_score_acronym"], shown["keyword_score_expansion"]) == (3.0, 0.0)
        # Other None fields are unchanged.
        assert "rrf_score" in SearchResultItem(**base).model_dump()
