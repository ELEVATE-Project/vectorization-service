"""Regression tests for the Release 2.0 scoring/ranking contract on the acronym branch.

Acronym support must widen candidate RETRIEVAL and re-order within the acronym
path only — it must never disturb an ordinary query. These tests pin:

1. The Release 2.0 fusion formula: no acronym-scoped dense/sparse weight flip.
2. filter_score / ordering / top_k semantics.
3. The injection contract: a document the pipeline actually scored keeps its
   real score and field_scores when re-injected; one never retrieved still
   takes the untouched Release 2.0 floor.
4. Title/summary multipliers unchanged.
5. Acronym tiering — in particular that the EXPANSION tiers (2/1) actually fire.
   Matching expansions as a literal phrase made them unreachable in practice
   (verified live on q=DIET: every result landed in tier 4 or tier 0), so
   _phrase_in_text matches on content words instead. These tests pin both that
   it now fires for real title variants and that it does NOT fire for
   near-misses that merely share common words.
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


# ── 1. Acronym tiering: expansion tiers must actually fire ────────────────

ACR = {"DIET": ["District Institute of Education and Training"]}


class TestPhraseInText:
    """_phrase_in_text backs tiers 2/1 (expansion matched in title/summary)."""

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
    """_words_match is the per-word rule _phrase_in_text applies (tiers 2/1)."""

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
        assert service._content_words(ACR["DIET"][0]) == [
            "district", "institute", "education", "training"]


class TestAssignAcronymTier:
    def test_acronym_in_title_is_tier_4(self, service):
        assert service._assign_acronym_tier("DIET Stakeholder Identity Map", None, ACR) == 4

    def test_acronym_in_summary_is_tier_3(self, service):
        assert service._assign_acronym_tier("Some Title", "About the DIET programme", ACR) == 3

    def test_expansion_in_title_is_tier_2(self, service):
        """The regression this fix targets: previously 0 because of the plural."""
        assert service._assign_acronym_tier(
            "Strengthening of District Institutes of Education and Training", None, ACR) == 2

    def test_expansion_in_summary_is_tier_1(self, service):
        assert service._assign_acronym_tier(
            "Some Title", "Run by the District Institute of Education and Training", ACR) == 1

    def test_unrelated_is_tier_0(self, service):
        assert service._assign_acronym_tier(
            "Establishing Discipline Through Clear School Rules", "school rules", ACR) == 0

    def test_acronym_beats_expansion_when_both_present(self, service):
        assert service._assign_acronym_tier(
            "DIET — District Institute of Education and Training", None, ACR) == 4

    def test_literal_acronym_is_not_substring_matched(self, service):
        """'DIET' must never match inside 'dietary' — tier 4/3 stays exact."""
        assert service._assign_acronym_tier("Dietary guidelines for schools", None, ACR) == 0

    def test_underscore_separated_title_still_matches(self, service):
        """Filename-style titles use '_' as a separator; \\b would miss these."""
        assert service._assign_acronym_tier(
            "source_doc_COE_AM4C2_DIET_Empowerment_Design.xlsx", None, ACR) == 4

    def test_multiple_acronyms_take_the_max_tier(self, service):
        acr = {
            "DIET": ["District Institute of Education and Training"],
            "SMC": ["School Management Committee"],
        }
        # expansion-only match for one, literal match for the other
        assert service._assign_acronym_tier(
            "SMC handbook", "District Institute of Education and Training", acr) == 4


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

    def test_prefilter_scores_out_captures_pre_filter_entries(self, service):
        """The snapshot must include sources the threshold then removes — that is
        the whole point of taking it before filtering."""
        all_results, field_scores = self._fixture()
        snapshot = {}
        top, _ = service._process_and_filter_results(
            all_results, field_scores, WEIGHTS, FIELDS, top_k=10, threshold=0.9,
            prefilter_scores_out=snapshot,
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

    def test_scored_but_filtered_source_keeps_its_real_score(self, service, monkeypatch):
        """The fix: the multiplier must apply to a real score, not a constant."""
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
            prefilter_scores={"219": real},
        )
        entry = injected[0]
        assert entry["weighted_score"] == pytest.approx(
            0.31 * settings.PARTIAL_TITLE_BOOST)
        # real per-field similarities survive instead of being blanked
        assert entry["field_scores"]["tags"] == 0.61
        assert entry["field_scores"]["title_match"] == "partial"
        assert entry["raw_dense"] == 0.2228
        assert entry["keyword_score"] == 0.0

    def test_real_score_beats_the_floor_it_replaced(self, service, monkeypatch):
        """Concretely: source 219 for q=DIET stops being pinned at 0.225."""
        self._patch_scroll(monkeypatch, ["219"])
        real = {
            "payload": {"source_id": "219"}, "weighted_score": 0.31,
            "field_scores": {"tags": 0.61}, "raw_dense": 0.2228, "keyword_score": 0.0,
        }
        floor_only = service._fetch_field_match_docs(
            ["219"], {"219": "partial"}, "title",
            settings.EXACT_TITLE_BOOST, settings.PARTIAL_TITLE_BOOST,
        )[0]["weighted_score"]
        with_real = service._fetch_field_match_docs(
            ["219"], {"219": "partial"}, "title",
            settings.EXACT_TITLE_BOOST, settings.PARTIAL_TITLE_BOOST,
            prefilter_scores={"219": real},
        )[0]["weighted_score"]
        assert floor_only == pytest.approx(0.225)
        assert with_real > floor_only

    def test_boost_cap_still_applies(self, service, monkeypatch):
        """Release 2.0 caps a boosted score at 1.0; that must not regress."""
        self._patch_scroll(monkeypatch, ["5"])
        real = {"payload": {"source_id": "5"}, "weighted_score": 0.95,
                "field_scores": {"title": 0.9}}
        entry = service._fetch_field_match_docs(
            ["5"], {"5": "exact"}, "title",
            settings.EXACT_TITLE_BOOST, settings.PARTIAL_TITLE_BOOST,
            prefilter_scores={"5": real},
        )[0]
        assert entry["weighted_score"] == 1.0

    def test_empty_source_list_short_circuits(self, service):
        assert service._fetch_field_match_docs(
            [], {}, "title",
            settings.EXACT_TITLE_BOOST, settings.PARTIAL_TITLE_BOOST) == []

    def test_rescued_entry_identifies_the_chunk_that_earned_the_score(
            self, service, monkeypatch):
        """A multi-chunk source must not pair one chunk's text with another's score.

        prefilter_scores holds the BEST chunk per source; the scroll returns chunks
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
            "tier": 4,
        }
        entry = service._fetch_field_match_docs(
            ["900"], {"900": "partial"}, "title",
            settings.EXACT_TITLE_BOOST, settings.PARTIAL_TITLE_BOOST,
            prefilter_scores={"900": best_chunk},
        )[0]
        assert entry["id"] == "pt-900-c7"
        assert entry["payload"]["text"] == "the passage that scored"
        assert entry["weighted_score"] == pytest.approx(
            0.31 * settings.PARTIAL_TITLE_BOOST)

    def test_floor_path_still_takes_the_scrolled_chunk(self, service, monkeypatch):
        """No fallback means no scores to be consistent with — any chunk will do."""
        self._patch_scroll(monkeypatch, ["901"])
        entry = service._fetch_field_match_docs(
            ["901"], {"901": "partial"}, "title",
            settings.EXACT_TITLE_BOOST, settings.PARTIAL_TITLE_BOOST,
        )[0]
        assert entry["id"] == "pt-901"
        assert entry["payload"]["title"] == "Doc 901"

    def _patch_scroll_with(self, monkeypatch, points):
        monkeypatch.setattr(
            "app.services.prioritized_search_service.qdrant_client.scroll",
            lambda **kwargs: (points, None),
        )

    def _inject_one(self, service, source_id, field="title"):
        boosts = (
            (settings.EXACT_TITLE_BOOST, settings.PARTIAL_TITLE_BOOST)
            if field == "title"
            else (settings.EXACT_SUMMARY_BOOST, settings.PARTIAL_SUMMARY_BOOST)
        )
        return service._fetch_field_match_docs(
            [source_id], {source_id: "partial"}, field, *boosts,
            acronyms_detected=ACR,
        )[0]

    def test_injected_tier_comes_from_the_shared_rule_not_the_lookup(
            self, service, monkeypatch):
        """A never-retrieved doc must be tiered like every scored doc.

        Deriving the tier from which scroll found the document gave a title
        carrying the literal acronym tier 2, ranking it below a doc that merely
        mentions the acronym in its summary (tier 3). _assign_acronym_tier says 4.
        """
        self._patch_scroll_with(monkeypatch, [
            _point("pt-a", "A", title="DIET Handbook 2024", summary="Annual guidance"),
        ])
        assert self._inject_one(service, "A")["tier"] == 4

    def test_injected_tier_zero_when_the_doc_has_no_acronym_signal(
            self, service, monkeypatch):
        """The title lookup also matches ordinary query words.

        'Teacher training calendar' is found by the scroll for a query like
        'DIET training' but carries no acronym or expansion — the shared rule
        says tier 0, where the old lookup-derived guess handed out tier 2.
        """
        self._patch_scroll_with(monkeypatch, [
            _point("pt-b", "B", title="Teacher training calendar", summary="Dates"),
        ])
        assert self._inject_one(service, "B")["tier"] == 0

    def test_injected_summary_match_gets_the_summary_tier(self, service, monkeypatch):
        """Acronym in the summary but not the title is tier 3, not the guessed 1."""
        self._patch_scroll_with(monkeypatch, [
            _point("pt-c", "C", title="Annual Report", summary="The DIET met quarterly"),
        ])
        assert self._inject_one(service, "C", field="summary")["tier"] == 3

    def test_injected_expansion_in_title_is_tier_two(self, service, monkeypatch):
        """The expansion tiers still fire through the injection path."""
        self._patch_scroll_with(monkeypatch, [
            _point("pt-d", "D",
                   title="District Institutes of Education and Training", summary=""),
        ])
        assert self._inject_one(service, "D")["tier"] == 2

    def test_no_acronyms_detected_keeps_every_injected_doc_at_tier_zero(
            self, service, monkeypatch):
        """Tiering stays a no-op for ordinary queries — unchanged contract."""
        self._patch_scroll_with(monkeypatch, [
            _point("pt-e", "E", title="DIET Handbook", summary="About the DIET"),
        ])
        entry = service._fetch_field_match_docs(
            ["E"], {"E": "partial"}, "title",
            settings.EXACT_TITLE_BOOST, settings.PARTIAL_TITLE_BOOST,
        )[0]
        assert entry["tier"] == 0

    def test_scored_doc_still_keeps_its_pipeline_tier(self, service, monkeypatch):
        """The fallback path is untouched — it already carried the real tier."""
        self._patch_scroll_with(monkeypatch, [
            _point("pt-f", "F", title="Unrelated chunk title", summary=""),
        ])
        real = {"id": "pt-f", "payload": {"source_id": "F"}, "weighted_score": 0.31,
                "field_scores": {}, "tier": 3}
        entry = service._fetch_field_match_docs(
            ["F"], {"F": "partial"}, "title",
            settings.EXACT_TITLE_BOOST, settings.PARTIAL_TITLE_BOOST,
            prefilter_scores={"F": real}, acronyms_detected=ACR,
        )[0]
        assert entry["tier"] == 3


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
    def test_detected_acronym_yields_expansions(self, monkeypatch):
        """DIET must expand so 'District Institute...' documents are reachable."""
        import app.services.acronym_query_service as aqs

        monkeypatch.setattr(
            aqs, "get_acronym_mapping",
            lambda: {"DIET": ["District Institute of Education and Training"]},
            raising=False,
        )
        detected = aqs.detect_acronyms("DIET")
        assert "DIET" in detected
        assert any("District Institute" in e for e in detected["DIET"])


# ── 7. Tiering is a lexical signal and rides the lexical switch ───────────────

class TestTieringFollowsLexicalRanking:
    """Acronym tiering must not outlive the boost block it is scored alongside.

    Tier ordering used to run unconditionally while the rewrite that folds tier
    into the exposed score sat inside the `HYBRID_SEARCH_ENABLED and search_mode
    != "semantic"` boost block. A semantic-mode acronym query therefore came back
    tier-ORDERED with raw scores that contradicted the order, and any caller
    re-sorting by score undid the ranking. Tiering is lexical — the same family
    as the title boost semantic mode already opts out of — so it rides the same
    switch.

    Doc A scores worse semantically but carries the acronym in its title; doc B
    scores better with no acronym signal. Hybrid ranks A first (tier 4 beats tier
    0); semantic ranks B first (pure similarity, no tiers).
    """

    def _run(self, service, monkeypatch, search_mode, hybrid_enabled=True,
             field_scores=None):
        from app.models.api_models import PrioritizedSearchRequest

        monkeypatch.setattr(
            "app.services.prioritized_search_service.detect_acronyms",
            lambda q: dict(ACR))
        monkeypatch.setattr(settings, "ACRONYM_SEARCH_ENABLED", True)
        monkeypatch.setattr(settings, "HYBRID_SEARCH_ENABLED", hybrid_enabled)
        monkeypatch.setattr(settings, "SPARSE_SEARCH_ENABLED", False)
        monkeypatch.setattr(
            "app.services.prioritized_search_service.embedding.embed_query",
            lambda texts: [[0.0] * 8 for _ in texts])

        all_results = {
            "pt-A": _point("pt-A", "A", title="DIET Handbook", summary="", text="a"),
            "pt-B": _point("pt-B", "B", title="Nutrition guide", summary="", text="b"),
        }
        # pt-A carries a `text` score as well as the title one: an acronym tier
        # now requires the document's own body to have matched too, and a
        # title-only fixture models the document that rule demotes, not the
        # genuine acronym hit this class is about. pt-B stays title-only — it
        # has no acronym signal either way.
        field_scores = field_scores or {
            "pt-A": {"title": 0.30, "text": 0.10},
            "pt-B": {"title": 0.90},
        }
        monkeypatch.setattr(
            PrioritizedSearchService, "_parallel_batch_search",
            lambda self, **kw: (all_results, field_scores))
        # Keep the boost itself out of it — this is about tiering, not multipliers.
        monkeypatch.setattr(
            PrioritizedSearchService, "_get_field_match_sources",
            lambda self, *a, **kw: {})

        return service.search(
            PrioritizedSearchRequest(query="DIET", top_k=10, search_mode=search_mode))

    def test_semantic_mode_ranks_by_similarity_alone(self, service, monkeypatch):
        """THE fix: no tier ordering, so nothing contradicts the raw scores."""
        response = self._run(service, monkeypatch, "semantic")
        assert [r.source_id for r in response.results] == ["B", "A"]
        # raw weighted scores, untouched by any tier rewrite
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

    def test_hybrid_mode_still_tiers_and_bands_the_scores(self, service, monkeypatch):
        """Acronym ranking is untouched where it was already coherent."""
        response = self._run(service, monkeypatch, "hybrid")
        assert [r.source_id for r in response.results] == ["A", "B"]
        # A sits in the tier-4 band, B in the tier-0 band
        assert 0.8 <= response.results[0].score < 1.0
        assert 0.0 <= response.results[1].score < 0.2

    def test_hybrid_mode_order_is_reproducible_from_the_scores(
            self, service, monkeypatch):
        """Same invariant on the other side of the switch."""
        response = self._run(service, monkeypatch, "hybrid")
        as_returned = [r.source_id for r in response.results]
        rescored = sorted(response.results, key=lambda r: r.score, reverse=True)
        assert [r.source_id for r in rescored] == as_returned

    def test_title_acronym_without_a_body_match_does_not_take_tier_4(
            self, service, monkeypatch):
        """The gate: a title is a claim of topic, not evidence of it.

        Documents carrying DIET in the title over an unrelated body outranked
        genuine DIET material on the live corpus — tier 4 owns [0.8, 1.0] after
        the score rewrite, so no relevance score can cross the boundary. Same
        fixture as the class default, minus pt-A's `text` score: pt-A keeps its
        DIET title but its body never matched.
        """
        response = self._run(
            service, monkeypatch, "hybrid",
            field_scores={"pt-A": {"title": 0.30}, "pt-B": {"title": 0.90}})
        # A loses the tier-4 band it would otherwise hold, so B's higher
        # similarity wins.
        assert [r.source_id for r in response.results] == ["B", "A"]
        # Both land in the tier-0 band — demoted, NOT removed.
        assert len(response.results) == 2
        assert all(r.score < 0.2 for r in response.results)

    def test_a_demoted_document_is_still_returned(self, service, monkeypatch):
        """filter_score reads weighted_score and never the tier.

        A title match IS a legitimate match; it just isn't proof of topic, so
        it ranks low rather than disappearing.
        """
        response = self._run(
            service, monkeypatch, "hybrid",
            field_scores={"pt-A": {"title": 0.30}, "pt-B": {"title": 0.90}})
        assert "A" in [r.source_id for r in response.results]

    def test_body_matched_reads_the_dense_text_score(self, service):
        """The gate's signal is the dense `text` similarity, and only that."""
        assert service._body_matched({"text": 0.4}) is True
        assert service._body_matched({"text": None}) is False
        assert service._body_matched({"text": 0.0}) is False
        assert service._body_matched({}) is False
        assert service._body_matched(None) is False
        # A sparse score alone must not satisfy it.
        assert service._body_matched({settings.SPARSE_VECTOR_NAME: 37.2}) is False

    def test_hybrid_kill_switch_off_also_drops_tiering(self, service, monkeypatch):
        """HYBRID_SEARCH_ENABLED=false trips the same switch as semantic mode."""
        response = self._run(service, monkeypatch, "hybrid", hybrid_enabled=False)
        assert [r.source_id for r in response.results] == ["B", "A"]
        assert response.results[0].score == pytest.approx(0.90 * WEIGHTS["title"])

    def test_semantic_mode_still_reports_the_detected_acronym(
            self, service, monkeypatch):
        """Retrieval widening is NOT gated — only ranking is."""
        response = self._run(service, monkeypatch, "semantic")
        assert response.acronym_info == {"detected": True, "mapping": dict(ACR)}

    def test_tiering_is_skipped_not_just_unused_in_semantic_mode(
            self, service, monkeypatch):
        """_process_and_filter_results must receive None, not an empty gesture."""
        seen = {}
        original = PrioritizedSearchService._process_and_filter_results

        def spy(self, *args, **kwargs):
            seen["acronyms_detected"] = kwargs.get("acronyms_detected")
            return original(self, *args, **kwargs)

        monkeypatch.setattr(
            PrioritizedSearchService, "_process_and_filter_results", spy)
        self._run(service, monkeypatch, "semantic")
        assert seen["acronyms_detected"] is None

    def test_tiering_is_passed_through_in_hybrid_mode(self, service, monkeypatch):
        seen = {}
        original = PrioritizedSearchService._process_and_filter_results

        def spy(self, *args, **kwargs):
            seen["acronyms_detected"] = kwargs.get("acronyms_detected")
            return original(self, *args, **kwargs)

        monkeypatch.setattr(
            PrioritizedSearchService, "_process_and_filter_results", spy)
        self._run(service, monkeypatch, "hybrid")
        assert seen["acronyms_detected"] == dict(ACR)
