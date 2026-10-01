"""Regression tests for the Release 2.0 scoring/ranking contract on the acronym branch.

Acronym support must widen candidate RETRIEVAL and re-order within the acronym
path only — it must never disturb an ordinary query. These tests pin:

1. The Release 2.0 fusion formula: no acronym-scoped dense/sparse weight flip.
2. filter_score / ordering / top_k semantics.
3. The injection contract: a document the pipeline actually scored keeps its
   real score and field_scores when re-injected; one never retrieved still
   takes the untouched Release 2.0 floor.
4. Title/summary multipliers unchanged.
5. The acronym bonus that replaced hard tiers — in particular that the EXPANSION
   grades actually fire. Matching expansions as a literal phrase made them
   unreachable in practice (verified live on q=DIET), so _phrase_in_text matches
   on content words instead. These tests pin both that it fires for real title
   variants and that it does NOT fire for near-misses that merely share common
   words — and that the bonus scales relevance rather than overriding it.
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


# ── 1. Acronym bonus: expansion grades must actually fire ─────────────────

ACR = {"DIET": ["District Institute of Education and Training"]}


class TestPhraseInText:
    """_phrase_in_text backs the expansion grades (expansion in title/summary)."""

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
    """_words_match is the per-word rule _phrase_in_text applies (expansion grades)."""

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


class TestAcronymBonus:
    """The four grades that replaced tiers 4/2/3/1. `backed_acronyms` is the set
    of acronyms whose content is verified to mention them — required for the
    acronym grades only, and per-acronym (see test_body_evidence_does_not_leak_
    between_acronyms below for why it can't just be a single flag)."""

    def test_acronym_in_title(self, service):
        assert service._acronym_bonus(
            "DIET Stakeholder Identity Map", None, ACR, {"DIET"}) == settings.ACRONYM_BONUS_TITLE_ACRONYM

    def test_acronym_in_summary(self, service):
        assert service._acronym_bonus(
            "Some Title", "About the DIET programme", ACR, {"DIET"}) == settings.ACRONYM_BONUS_SUMMARY_ACRONYM

    def test_expansion_in_title(self, service):
        """The plural regression: literal expansion matching never fired."""
        assert service._acronym_bonus(
            "Strengthening of District Institutes of Education and Training", None, ACR, set()
        ) == settings.ACRONYM_BONUS_TITLE_EXPANSION

    def test_expansion_in_summary(self, service):
        assert service._acronym_bonus(
            "Some Title", "Run by the District Institute of Education and Training", ACR, set()
        ) == settings.ACRONYM_BONUS_SUMMARY_EXPANSION

    def test_unrelated_earns_nothing(self, service):
        assert service._acronym_bonus(
            "Establishing Discipline Through Clear School Rules", "school rules", ACR, {"DIET"}) == 0.0

    def test_acronym_grade_beats_expansion_grade_when_both_present(self, service):
        assert service._acronym_bonus(
            "DIET — District Institute of Education and Training", None, ACR, {"DIET"}
        ) == settings.ACRONYM_BONUS_TITLE_ACRONYM

    def test_literal_acronym_is_not_substring_matched(self, service):
        """'DIET' must never match inside 'dietary', backed or not."""
        assert service._acronym_bonus("Dietary guidelines for schools", None, ACR, {"DIET"}) == 0.0

    def test_underscore_separated_title_still_matches(self, service):
        """Filename-style titles use '_' as a separator; \\b would miss these."""
        assert service._acronym_bonus(
            "source_doc_COE_AM4C2_DIET_Empowerment_Design.xlsx", None, ACR, {"DIET"}
        ) == settings.ACRONYM_BONUS_TITLE_ACRONYM

    def test_multiple_acronyms_sum_their_own_best_grades(self, service):
        """AC-15: a document matching both SMC and DIET must rank above one
        matching only DIET — so multiple detected acronyms sum their own best
        grades (each computed independently) rather than the whole bonus
        collapsing to a single max across all of them."""
        acr = {
            "DIET": ["District Institute of Education and Training"],
            "SMC": ["School Management Committee"],
        }
        # literal match for SMC (title), expansion-only match for DIET (summary)
        assert service._acronym_bonus(
            "SMC handbook", "District Institute of Education and Training", acr, {"DIET", "SMC"}
        ) == settings.ACRONYM_BONUS_TITLE_ACRONYM + settings.ACRONYM_BONUS_SUMMARY_EXPANSION

    def test_multiple_acronyms_summed_bonus_is_capped(self, service):
        """The sum is bounded — two acronyms both landing a full title match
        must not double the bonus outright."""
        acr = {
            "DIET": ["District Institute of Education and Training"],
            "SMC": ["School Management Committee"],
        }
        uncapped = settings.ACRONYM_BONUS_TITLE_ACRONYM * 2
        assert uncapped > settings.ACRONYM_BONUS_MULTI_MATCH_CAP
        assert service._acronym_bonus(
            "SMC DIET handbook", None, acr, {"DIET", "SMC"}) == settings.ACRONYM_BONUS_MULTI_MATCH_CAP

    def test_body_evidence_does_not_leak_between_acronyms(self, service):
        """A document's body backing SMC must not also count as backing for a
        DIET title claim on the same document — each acronym's grade is only
        gated on ITS OWN entry in backed_acronyms."""
        acr = {
            "DIET": ["District Institute of Education and Training"],
            "SMC": ["School Management Committee"],
        }
        # Title claims both acronyms; only SMC's content is actually backed.
        assert service._acronym_bonus(
            "DIET SMC Handbook", None, acr, {"SMC"}
        ) == settings.ACRONYM_BONUS_TITLE_ACRONYM  # SMC's grade only, not DIET's too

    def test_title_claim_without_content_evidence_earns_nothing(self, service):
        """A title is a claim about the topic, not evidence — the corpus holds
        documents titled 'DIET Reference Handbook' about coastal navigation."""
        assert service._acronym_bonus("DIET Reference Handbook", None, ACR, set()) == 0.0

    def test_unbacked_acronym_keeps_its_expansion_grade(self, service):
        """Every grade is evaluated, so an unbacked acronym never erases the
        expansion grade the same title earns on its own — adding 'DIET' to a
        title that spells the expansion out must not LOWER its rank."""
        assert service._acronym_bonus(
            f"DIET — {ACR['DIET'][0]}", None, ACR, set()
        ) == settings.ACRONYM_BONUS_TITLE_EXPANSION


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
        assert entry["measured_relevance"] == pytest.approx(0.31)

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
        assert "measured_relevance" not in floor_only
        assert with_real["measured_relevance"] == pytest.approx(0.31)

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

    def _patch_scroll_with(self, monkeypatch, points):
        monkeypatch.setattr(
            "app.services.prioritized_search_service.qdrant_client.scroll",
            lambda **kwargs: (points, None),
        )

    def _inject_one(self, service, monkeypatch, source_id, field="title", backed=()):
        """`backed`: source ids whose content mentions the acronym (the BM25
        check is stubbed, so these tests never reach Qdrant). ACR is always the
        single-acronym {"DIET": [...]} fixture in this class, so the mocked
        per-acronym mapping only ever needs the one "DIET" key."""
        monkeypatch.setattr(
            PrioritizedSearchService, "_sources_with_acronym_in_body",
            lambda self, candidates, acronyms_detected=None, **kw: {"DIET": set(backed)})
        boosts = (
            (settings.EXACT_TITLE_BOOST, settings.PARTIAL_TITLE_BOOST)
            if field == "title"
            else (settings.EXACT_SUMMARY_BOOST, settings.PARTIAL_SUMMARY_BOOST)
        )
        return service._fetch_field_match_docs(
            [source_id], {source_id: "partial"}, field, *boosts,
            acronyms_detected=ACR,
        )[0]

    def test_injected_title_acronym_gets_no_bonus_without_content_evidence(
            self, service, monkeypatch):
        """A never-retrieved doc can't claim the acronym bonus on its title alone.

        The bonus comes from the shared rule (_acronym_bonus), not from which
        scroll found the document — and that rule needs the content to mention
        the acronym. Ungated, the old tiers put floor 0.15 x the 1.5 title
        boost at tier 4: an exposed 0.845 against a scored document's 0.14.
        """
        self._patch_scroll_with(monkeypatch, [
            _point("pt-a", "A", title="DIET Handbook 2024", summary="Annual guidance"),
        ])
        entry = self._inject_one(service, monkeypatch, "A")
        assert entry["acronym_bonus"] == 0.0
        assert entry["weighted_score"] == pytest.approx(
            self.FLOOR * settings.PARTIAL_TITLE_BOOST)

    def test_injected_title_acronym_with_content_evidence_earns_the_bonus(
            self, service, monkeypatch):
        """The content check is per document, so a document missing from the
        retrieval pool — which pool depends on top_k — still earns the bonus
        its content supports. Without this the bonus would be page-size
        dependent on the injection path."""
        self._patch_scroll_with(monkeypatch, [
            _point("pt-a", "A", title="DIET Handbook 2024", summary="Annual guidance"),
        ])
        entry = self._inject_one(service, monkeypatch, "A", backed={"A"})
        assert entry["acronym_bonus"] == settings.ACRONYM_BONUS_TITLE_ACRONYM
        assert entry["weighted_score"] == pytest.approx(
            self.FLOOR * settings.PARTIAL_TITLE_BOOST
            * (1 + settings.ACRONYM_BONUS_TITLE_ACRONYM))

    def test_injected_doc_with_no_acronym_signal_earns_nothing(
            self, service, monkeypatch):
        """The title lookup also matches ordinary query words.

        'Teacher training calendar' is found by the scroll for a query like
        'DIET training' but carries no acronym or expansion.
        """
        self._patch_scroll_with(monkeypatch, [
            _point("pt-b", "B", title="Teacher training calendar", summary="Dates"),
        ])
        assert self._inject_one(service, monkeypatch, "B", backed={"B"})["acronym_bonus"] == 0.0

    def test_injected_summary_acronym_also_needs_content_evidence(self, service, monkeypatch):
        """The summary grade is gated on content for the same reason the title one is."""
        self._patch_scroll_with(monkeypatch, [
            _point("pt-c", "C", title="Annual Report", summary="The DIET met quarterly"),
        ])
        assert self._inject_one(service, monkeypatch, "C", field="summary")["acronym_bonus"] == 0.0

    def test_injected_expansion_in_title_earns_the_expansion_grade(self, service, monkeypatch):
        """Expansion grades fire through the injection path, no content check needed."""
        self._patch_scroll_with(monkeypatch, [
            _point("pt-d", "D",
                   title="District Institutes of Education and Training", summary=""),
        ])
        assert self._inject_one(service, monkeypatch, "D")["acronym_bonus"] == \
            settings.ACRONYM_BONUS_TITLE_EXPANSION

    def test_unbacked_acronym_does_not_erase_the_expansion_grade(
            self, service, monkeypatch):
        """Adding the acronym to a title must never LOWER its rank: an unbacked
        "DIET - District Institute of Education and Training" keeps the
        expansion grade the same title earns without "DIET" in it."""
        expansion = "District Institute of Education and Training"
        self._patch_scroll_with(monkeypatch, [
            _point("pt-f", "F", title=f"DIET — {expansion}", summary=""),
        ])
        assert self._inject_one(service, monkeypatch, "F")["acronym_bonus"] == \
            settings.ACRONYM_BONUS_TITLE_EXPANSION

        # Summary-only expansion keeps the summary expansion grade.
        self._patch_scroll_with(monkeypatch, [
            _point("pt-g", "G", title="Annual Report",
                   summary=f"Issued by DIET, the {expansion}"),
        ])
        assert self._inject_one(service, monkeypatch, "G", field="summary")["acronym_bonus"] == \
            settings.ACRONYM_BONUS_SUMMARY_EXPANSION

    def test_no_acronyms_detected_gives_no_injected_doc_a_bonus(
            self, service, monkeypatch):
        """The bonus is a no-op for ordinary queries — Release 2.0 floor x boost."""
        self._patch_scroll_with(monkeypatch, [
            _point("pt-e", "E", title="DIET Handbook", summary="About the DIET"),
        ])
        entry = service._fetch_field_match_docs(
            ["E"], {"E": "partial"}, "title",
            settings.EXACT_TITLE_BOOST, settings.PARTIAL_TITLE_BOOST,
        )[0]
        assert entry["acronym_bonus"] == 0.0
        assert entry["weighted_score"] == pytest.approx(
            self.FLOOR * settings.PARTIAL_TITLE_BOOST)

    def test_scored_doc_keeps_its_pipeline_bonus_applied_once(self, service, monkeypatch):
        """The fallback carries the bonus the pipeline already decided upstream, and
        that bonus is applied to the floor/score_floor relevance exactly once here —
        its OWN weighted_score may already include the bonus (a source cut by the
        top_k cap), so re-reading acronym_bonus and multiplying it in again, rather
        than reusing the fallback's weighted_score directly, is what stops the title
        from being counted twice."""
        self._patch_scroll_with(monkeypatch, [
            _point("pt-f", "F", title="Unrelated chunk title", summary=""),
        ])
        bonus = settings.ACRONYM_BONUS_SUMMARY_ACRONYM
        real = {"id": "pt-f", "payload": {"source_id": "F"},
                "relevance": 0.31, "weighted_score": 0.31 * (1 + bonus),
                "acronym_bonus": bonus, "field_scores": {}}
        entry = service._fetch_field_match_docs(
            ["F"], {"F": "partial"}, "title",
            settings.EXACT_TITLE_BOOST, settings.PARTIAL_TITLE_BOOST,
            prethreshold_by_source={"F": real}, acronyms_detected=ACR,
        )[0]
        assert entry["acronym_bonus"] == bonus
        assert entry["weighted_score"] == pytest.approx(
            self.FLOOR * settings.PARTIAL_TITLE_BOOST * (1 + bonus))


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


# ── 7. The acronym bonus is lexical, scales relevance, rides the lexical switch ─

class TestAcronymBonusFollowsLexicalRanking:
    """The acronym bonus replaced hard tiers, and rides the same switch as the
    title/summary boost: on in hybrid mode, off in semantic mode and when
    HYBRID_SEARCH_ENABLED is false.

    Doc A carries the acronym in its title; doc B carries none. Under tiers A
    always won — tier 4 beat tier 0 however much weaker A was. Now the bonus
    multiplies A's relevance by (1 + ACRONYM_BONUS_TITLE_ACRONYM), so A wins only
    when it is close enough for the title to make up the difference. In the
    default fixture A (0.135) is well under half of B (0.324), so B wins.
    """

    def _run(self, service, monkeypatch, search_mode, hybrid_enabled=True,
             field_scores=None, acronyms=None, **request_fields):
        from app.models.api_models import PrioritizedSearchRequest

        detected = dict(ACR) if acronyms is None else acronyms
        monkeypatch.setattr(
            "app.services.prioritized_search_service.detect_acronyms",
            lambda q: dict(detected))
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
        # pt-A carries a `text` score as well as the title one: the acronym
        # bonus requires the document's content to back the title, and with
        # BM25 off here that falls back to "the body matched". A title-only
        # fixture models the document that rule denies. pt-B stays title-only —
        # it has no acronym signal either way.
        field_scores = field_scores or {
            "pt-A": {"title": 0.30, "text": 0.10},
            "pt-B": {"title": 0.90},
        }
        monkeypatch.setattr(
            PrioritizedSearchService, "_parallel_batch_search",
            lambda self, **kw: (all_results, field_scores))
        # Keep the title/summary multipliers out of it — this is about the bonus.
        monkeypatch.setattr(
            PrioritizedSearchService, "_get_field_match_sources",
            lambda self, *a, **kw: {})

        return service.search(
            PrioritizedSearchRequest(query="DIET", top_k=10, search_mode=search_mode,
                                     **request_fields))

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

    def test_hybrid_mode_bonus_cannot_lift_a_much_weaker_document(self, service, monkeypatch):
        """THE requirement: a title nudges, it never buries a stronger document.

        Under tiers A came first here. Its relevance is well under half of B's,
        so a title worth +20% can't close that gap — and scores are no longer
        banded, so each reads as relevance, bonus included.
        """
        response = self._run(service, monkeypatch, "hybrid")
        assert [r.source_id for r in response.results] == ["B", "A"]
        relevance_a = 0.30 * WEIGHTS["title"] + 0.10 * WEIGHTS["text"]
        relevance_b = 0.90 * WEIGHTS["title"]
        assert response.results[0].score == pytest.approx(relevance_b)
        assert response.results[1].score == pytest.approx(
            relevance_a * (1 + settings.ACRONYM_BONUS_TITLE_ACRONYM))

    def test_hybrid_mode_bonus_closes_a_small_gap(self, service, monkeypatch):
        """...but a title does lift a document that's nearly as relevant.

        A at 0.80 title similarity is ~3% short of B; the +20% title bonus
        carries it past. This is the case tiers got right, kept."""
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

    def test_title_acronym_without_content_evidence_earns_no_bonus(
            self, service, monkeypatch):
        """A title is a claim of topic, not evidence of it.

        The close-gap fixture again, minus pt-A's body score: with BM25 off here
        the check falls back to "did the body match at all", and it didn't. A
        loses the bonus that carried it past B above, so B's higher relevance
        wins — and A keeps its plain relevance, demoted but not removed.
        """
        response = self._run(
            service, monkeypatch, "hybrid",
            field_scores={"pt-A": {"title": 0.80}, "pt-B": {"title": 0.90}})
        assert [r.source_id for r in response.results] == ["B", "A"]
        assert response.results[1].score == pytest.approx(0.80 * WEIGHTS["title"])

    def test_scored_path_keeps_the_expansion_grade_when_the_acronym_is_unbacked(
            self, service):
        """Same rule on the scored path as the injection one."""
        both = f"DIET — {ACR['DIET'][0]}"
        assert service._acronym_bonus(both, None, dict(ACR), {"DIET"}) == \
            settings.ACRONYM_BONUS_TITLE_ACRONYM
        assert service._acronym_bonus(both, None, dict(ACR), set()) == \
            settings.ACRONYM_BONUS_TITLE_EXPANSION
        # A title with no expansion in it has nothing to fall back to.
        assert service._acronym_bonus("DIET Handbook", None, dict(ACR), set()) == 0.0

    def test_a_demoted_document_is_still_returned(self, service, monkeypatch):
        """filter_score reads relevance and never the bonus.

        A title match IS a legitimate match; it just isn't proof of topic, so
        it ranks low rather than disappearing.
        """
        response = self._run(
            service, monkeypatch, "hybrid",
            field_scores={"pt-A": {"title": 0.30}, "pt-B": {"title": 0.90}})
        assert "A" in [r.source_id for r in response.results]

    def test_injected_never_scored_doc_gets_no_acronym_bonus(self, service, monkeypatch):
        """A document no vector query reached must not take an acronym bonus
        unless its content backs it.

        _fetch_field_match_docs injects title/summary matches the threshold
        dropped. On the floor branch nothing was ever measured about the
        document, so "never measured" would otherwise outrank "measured and
        found unrelated": floor 0.15 x the 1.5 title boost, tier 4, an exposed
        0.845 against the scored document's 0.14.
        """
        chunk = _point("pt-Z", "Z", title="DIET Handbook", summary="", text="unrelated")
        monkeypatch.setattr(
            "app.services.prioritized_search_service.qdrant_client.scroll",
            lambda **kwargs: ([chunk], None),
        )
        # Content check: this document's content doesn't mention DIET.
        monkeypatch.setattr(
            PrioritizedSearchService, "_sources_with_acronym_in_body",
            lambda self, candidates, acronyms_detected=None, **kw: {})

        injected = service._fetch_field_match_docs(
            ["Z"], {"Z": "partial"}, "title",
            settings.EXACT_TITLE_BOOST, settings.PARTIAL_TITLE_BOOST,
            prethreshold_by_source=None, acronyms_detected=dict(ACR))

        assert len(injected) == 1
        entry = injected[0]
        # Still injected — the gate demotes, it never removes.
        assert entry["payload"]["source_id"] == "Z"
        # ...but with no acronym bonus, since nothing backs the title's claim.
        assert entry["acronym_bonus"] == 0.0
        assert all(v is None for v in entry["field_scores"].values()
                   if not isinstance(v, str))

    def test_body_matched_reads_the_dense_text_score(self, service):
        """The FALLBACK content check (BM25 unavailable) reads the dense `text`
        similarity, and only that."""
        assert service._body_matched({"text": 0.4}) is True
        assert service._body_matched({"text": None}) is False
        assert service._body_matched({"text": 0.0}) is False
        assert service._body_matched({}) is False
        assert service._body_matched(None) is False
        # A sparse score alone must not satisfy it.
        assert service._body_matched({settings.SPARSE_VECTOR_NAME: 37.2}) is False

    def test_hybrid_kill_switch_off_also_drops_the_bonus(self, service, monkeypatch):
        """HYBRID_SEARCH_ENABLED=false trips the same switch as semantic mode."""
        response = self._run(service, monkeypatch, "hybrid", hybrid_enabled=False)
        assert [r.source_id for r in response.results] == ["B", "A"]
        assert response.results[0].score == pytest.approx(0.90 * WEIGHTS["title"])

    def test_semantic_mode_still_reports_the_detected_acronym(
            self, service, monkeypatch):
        """Retrieval widening is NOT gated — only ranking is."""
        response = self._run(service, monkeypatch, "semantic")
        assert response.acronym_info == {"detected": True, "mapping": dict(ACR)}

    def test_bonus_is_skipped_not_just_unused_in_semantic_mode(
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

    def test_bonus_is_passed_through_in_hybrid_mode(self, service, monkeypatch):
        seen = {}
        original = PrioritizedSearchService._process_and_filter_results

        def spy(self, *args, **kwargs):
            seen["acronyms_detected"] = kwargs.get("acronyms_detected")
            return original(self, *args, **kwargs)

        monkeypatch.setattr(
            PrioritizedSearchService, "_process_and_filter_results", spy)
        self._run(service, monkeypatch, "hybrid")
        assert seen["acronyms_detected"] == dict(ACR)


# ── scoring_context reports how an acronym query was actually scored ──────

class TestAcronymScoringContext:
    """search_config.scoring_context (debug only) must describe the scoring that
    ran: neutral title/summary multipliers and the acronym ranking block."""

    def _context(self, service, monkeypatch, search_mode="hybrid", acronyms=None):
        response = TestAcronymBonusFollowsLexicalRanking()._run(
            service, monkeypatch, search_mode, acronyms=acronyms, include_scoring_debug=True)
        return response.search_config["scoring_context"]

    def test_acronym_query_reports_neutral_boosts_and_ranking(self, service, monkeypatch):
        ctx = self._context(service, monkeypatch)
        assert ctx["boost_config"] == {
            "mode": "acronym_bonus", "exact_title_boost": 1.0, "partial_title_boost": 1.0,
            "exact_summary_boost": 1.0, "partial_summary_boost": 1.0,
        }
        ranking = ctx["acronym_ranking"]
        assert ranking["applied"] is True
        assert ranking["expansion_score_weight"] == settings.ACRONYM_EXPANSION_SCORE_WEIGHT
        assert ranking["bonus_cap"] == settings.ACRONYM_BONUS_MULTI_MATCH_CAP
        assert ranking["bonus_values"]["title_acronym_bonus"] == settings.ACRONYM_BONUS_TITLE_ACRONYM
        assert ranking["dense_variants"] == 2
        assert ranking["rescore"] == {
            "applied": True, "candidate_pool": 2, "rescored": 2,
            "limit": settings.ACRONYM_RESCORE_POOL_LIMIT, "capped": False,
            "normalization": "self",
        }
        # Sparse is off in this fixture, so the body check falls back to dense.
        assert ranking["body_check"] == "dense_fallback"

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


# ── Per-document acronym breakdown: named bonus, matched text, body evidence ──

class TestPerDocumentAcronymBreakdown:
    """Each result explains its bonus in named fields: which bonus each acronym
    earned, the text that matched, how the body backs it, and the relevance."""

    EXP = ACR["DIET"][0]

    def test_title_acronym_bonus_names_the_match(self, service):
        bonus, breakdown = service._acronym_bonus_detail(
            "DIET Handbook", None, dict(ACR), {"DIET"},
            {"DIET": ("expansion_in_body", self.EXP)})
        assert bonus == settings.ACRONYM_BONUS_TITLE_ACRONYM
        assert breakdown == {"DIET": {
            "bonus_type": "title_acronym_bonus",
            "bonus_value": settings.ACRONYM_BONUS_TITLE_ACRONYM,
            "matched_in": "title", "matched_text": "DIET",
            "body_evidence": "expansion_in_body", "body_evidence_text": self.EXP,
        }}

    def test_expansion_bonus_names_the_expansion_that_matched(self, service):
        acr = {"SSC": ["Staff Selection Commission", "Sainik School Society"]}
        _, breakdown = service._acronym_bonus_detail(
            "Sainik School Society admissions", None, acr, set())
        assert breakdown["SSC"]["bonus_type"] == "title_expansion_bonus"
        assert breakdown["SSC"]["matched_text"] == "Sainik School Society"
        assert breakdown["SSC"]["body_evidence"] is None
        assert breakdown["SSC"]["body_evidence_text"] is None

    def test_unbacked_title_acronym_earns_none(self, service):
        _, breakdown = service._acronym_bonus_detail("DIET Handbook", None, dict(ACR), set())
        assert breakdown["DIET"]["bonus_type"] == "none"
        assert breakdown["DIET"]["bonus_value"] == 0.0

    def test_backing_records_evidence_and_text(self, service, monkeypatch):
        TestExpansionInBodyBacksAcronymClaim._stub_qdrant(
            self, monkeypatch, acronym_hits=["A"],
            chunk_text={"B": "District Institute of Education and Training staff."})
        backing = {}
        backed = service._sources_with_acronym_in_body(
            {"DIET": {"A", "B"}}, acronyms_detected=ACR, backing_out=backing)
        assert backed == {"DIET": {"A", "B"}}
        assert backing == {"DIET": {"A": ("acronym_in_body", "DIET"),
                                    "B": ("expansion_in_body", self.EXP)}}

    def test_results_carry_named_fields_in_debug(self, service, monkeypatch):
        response = TestAcronymBonusFollowsLexicalRanking()._run(
            service, monkeypatch, "hybrid", include_scoring_debug=True)
        by_source = {r.source_id: r for r in response.results}
        a = by_source["A"]
        # Sparse is off in this fixture, so backing comes from the dense fallback.
        assert a.acronym_bonus_breakdown["DIET"]["bonus_type"] == "title_acronym_bonus"
        assert a.acronym_bonus_breakdown["DIET"]["body_evidence"] == "dense_fallback"
        assert a.title_acronym_bonus == settings.ACRONYM_BONUS_TITLE_ACRONYM
        assert a.title_expansion_bonus == a.summary_acronym_bonus == a.summary_expansion_bonus == 0.0
        assert a.score == pytest.approx(a.blended_score * (1 + a.acronym_bonus))
        assert by_source["B"].acronym_bonus_breakdown["DIET"]["bonus_type"] == "none"

    def test_named_fields_hidden_without_debug(self, service, monkeypatch):
        response = TestAcronymBonusFollowsLexicalRanking()._run(
            service, monkeypatch, "hybrid", include_scoring_debug=False)
        assert all(r.acronym_bonus_breakdown is None and r.title_acronym_bonus is None
                   and r.blended_score is None for r in response.results)


# ── Acronym debug fields stay null on ordinary queries ────────────────────

class TestAcronymDebugFieldsOnlyOnAcronymPath:
    """Injected keyword matches carry a floor relevance and a 0.0 bonus on every
    query; only rows that went through acronym scoring may expose them."""

    @staticmethod
    def _row(**extra):
        return {"id": "p1", "payload": {"source_id": "S", "title": "T", "text": "x"},
                "field_scores": {}, "weighted_score": 0.8,
                "match_source": "title_keyword_match", **extra}

    def test_ordinary_injected_row_hides_acronym_fields(self, service):
        item = service._build_result_items(
            [self._row(relevance=0.8, acronym_bonus=0.0)], True)[0]
        assert item.blended_score is None and item.acronym_bonus is None
        assert item.acronym_bonus_breakdown is None and item.title_acronym_bonus is None
        assert item.match_source == "title_keyword_match"

    def test_acronym_injected_row_keeps_acronym_fields(self, service):
        item = service._build_result_items([self._row(
            weighted_score=0.8 * 1.4, relevance=0.8, acronym_bonus=0.4,
            acronym_bonus_breakdown={"DIET": {
                "bonus_type": "title_acronym_bonus", "bonus_value": 0.4,
                "matched_in": "title", "matched_text": "DIET",
                "body_evidence": "acronym_in_body", "body_evidence_text": "DIET"}})], True)[0]
        assert item.blended_score == 0.8 and item.acronym_bonus == 0.4
        assert item.title_acronym_bonus == 0.4 and item.title_expansion_bonus == 0.0
        assert item.score == pytest.approx(item.blended_score * (1 + item.acronym_bonus))


# ── ACRONYM_USE_FIELD_BOOSTS swaps the acronym bonus for field boosts ──────

class TestAcronymFieldBoostSwitch:
    """Switch on: acronym queries get title/summary boosts plus a text boost (body
    uses the acronym = exact, only the expansion = partial) instead of the bonus."""

    def _run(self, service, monkeypatch, switch_on, text_matches=None):
        monkeypatch.setattr(settings, "ACRONYM_USE_FIELD_BOOSTS", switch_on)
        monkeypatch.setattr(
            PrioritizedSearchService, "_text_matches_for_acronyms",
            lambda self, sources, acronyms: dict(text_matches or {}))
        response = TestAcronymBonusFollowsLexicalRanking()._run(
            service, monkeypatch, "hybrid", include_scoring_debug=True)
        return response, {r.source_id: r for r in response.results}

    def test_switch_on_applies_text_boost_instead_of_bonus(self, service, monkeypatch):
        _, off = self._run(service, monkeypatch, False)
        base_a, base_b = off["A"].blended_score, off["B"].blended_score
        response, on = self._run(service, monkeypatch, True, {"A": "exact", "B": "partial"})
        assert on["A"].acronym_bonus is None and on["A"].acronym_bonus_breakdown is None
        assert (on["A"].text_match, on["A"].text_multiplier) == ("exact", settings.EXACT_TEXT_BOOST)
        assert (on["B"].text_match, on["B"].text_multiplier) == ("partial", settings.PARTIAL_TEXT_BOOST)
        # Title boosts apply too: A's title "DIET Handbook" is a partial title match.
        assert on["A"].title_multiplier == settings.PARTIAL_TITLE_BOOST
        assert on["A"].score == pytest.approx(
            min(base_a * settings.PARTIAL_TITLE_BOOST * settings.EXACT_TEXT_BOOST, 1.0))
        assert on["B"].title_multiplier == 1.0
        assert on["B"].score == pytest.approx(min(base_b * settings.PARTIAL_TEXT_BOOST, 1.0))

    def test_switch_on_reports_field_boost_mode(self, service, monkeypatch):
        response, _ = self._run(service, monkeypatch, True)
        ctx = response.search_config["scoring_context"]
        assert ctx["boost_config"]["mode"] == "acronym_field_boost"
        assert ctx["boost_config"]["exact_text_boost"] == settings.EXACT_TEXT_BOOST
        assert ctx["boost_config"]["exact_title_boost"] == settings.EXACT_TITLE_BOOST
        assert ctx["acronym_ranking"]["bonus_mode"] == "field_boosts"
        assert "bonus_values" not in ctx["acronym_ranking"]

    def test_switch_off_keeps_the_acronym_bonus(self, service, monkeypatch):
        response, off = self._run(service, monkeypatch, False, {"A": "exact"})
        assert off["A"].acronym_bonus_breakdown is not None
        assert off["A"].text_match is None and off["A"].text_multiplier is None
        assert response.search_config["scoring_context"]["acronym_ranking"]["bonus_mode"] == "acronym_bonus"


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
        TestAcronymBonusFollowsLexicalRanking()._run(service, monkeypatch, "hybrid")
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
        monkeypatch.setattr(
            PrioritizedSearchService, "_sources_with_acronym_in_body",
            lambda self, *a, **kw: {})
        monkeypatch.setattr(PrioritizedSearchService, "_blended_acronym_relevance", spy)

        service.search(PrioritizedSearchRequest(query="DIET", top_k=10, search_mode="hybrid"))

        ref = seen["normalization_reference"]
        assert ref and "dense_min" in ref and "dense_max" in ref


# ── Expansion in the body backs a title/summary acronym claim ─────────────

class TestExpansionInBodyBacksAcronymClaim:
    """A document titled with the acronym whose body spells out the expansion
    but never repeats the bare acronym ("PTM Handbook" / "Parent Teacher
    Meeting" throughout) is backed, so it keeps the title grade. A body that
    only shares some expansion words is not."""

    def _stub_qdrant(self, monkeypatch, acronym_hits, chunk_text, all_chunks=None):
        """acronym_hits: source -> body text the bare-acronym BM25 query returns
        (a list is shorthand for chunks that use "DIET" in capitals).
        chunk_text: source -> body text returned for the expansion query.
        all_chunks: source -> every chunk's text, for the full-scan scroll
        (defaults to just the BM25 chunk)."""
        if not isinstance(acronym_hits, dict):
            acronym_hits = {s: "The DIET faculty met." for s in acronym_hits}
        all_chunks = all_chunks or {s: [t] for s, t in acronym_hits.items()}
        calls = {"acronym": 0, "expansion": 0, "scroll": 0}

        def fake_scroll(**kwargs):
            calls["scroll"] += 1
            wanted = set(kwargs["scroll_filter"].must[0].match.any)
            points = [
                types.SimpleNamespace(payload={"source_id": s, "text": t})
                for s, texts in all_chunks.items() if s in wanted for t in texts]
            return points, None
        acronym_index, expansion_index = 1, 2

        def fake_groups(**kwargs):
            wanted = set(kwargs["query_filter"].must[0].match.any)
            if kwargs["query"].indices == [acronym_index]:
                calls["acronym"] += 1
                texts = acronym_hits
            else:
                calls["expansion"] += 1
                texts = chunk_text
            return types.SimpleNamespace(groups=[
                types.SimpleNamespace(id=s, hits=[
                    types.SimpleNamespace(payload={"text": texts[s]})])
                for s in texts if s in wanted])

        monkeypatch.setattr(settings, "SPARSE_SEARCH_ENABLED", True)
        monkeypatch.setattr(
            "app.services.acronym_ranking.qdrant_client.query_points_groups", fake_groups)
        monkeypatch.setattr(
            "app.services.acronym_ranking.qdrant_client.scroll", fake_scroll)
        monkeypatch.setattr(
            "app.core.clients.sparse_encoder.generate_sparse_vector",
            lambda text: ([acronym_index if text in ACR else expansion_index], [1.0]))
        return calls

    def test_spelled_out_expansion_backs_the_title_claim(self, service, monkeypatch):
        self._stub_qdrant(monkeypatch, acronym_hits=[], chunk_text={
            "A": "Each District Institute of Education and Training runs in-service courses."})
        backed = service._sources_with_acronym_in_body({"DIET": {"A"}}, acronyms_detected=ACR)
        assert backed == {"DIET": {"A"}}
        assert service._acronym_bonus("DIET Handbook", None, ACR, {"DIET"}) == \
            settings.ACRONYM_BONUS_TITLE_ACRONYM

    def test_expansion_variant_in_body_still_counts(self, service, monkeypatch):
        self._stub_qdrant(monkeypatch, acronym_hits=[], chunk_text={
            "A": "Visits to District Institutes for Education & Training across the state."})
        assert service._sources_with_acronym_in_body(
            {"DIET": {"A"}}, acronyms_detected=ACR) == {"DIET": {"A"}}

    def test_near_miss_body_does_not_back_the_claim(self, service, monkeypatch):
        # Shares district + education + training, but not "institute": a
        # different programme, the DPEP case the phrase rule exists to reject.
        self._stub_qdrant(monkeypatch, acronym_hits=[], chunk_text={
            "A": "District Primary Education Programme teacher training schedule."})
        assert service._sources_with_acronym_in_body(
            {"DIET": {"A"}}, acronyms_detected=ACR) == {"DIET": set()}

    def test_unrelated_body_does_not_back_the_claim(self, service, monkeypatch):
        self._stub_qdrant(monkeypatch, acronym_hits=[], chunk_text={
            "A": "Check tyre pressure weekly and lubricate the bicycle chain."})
        assert service._sources_with_acronym_in_body(
            {"DIET": {"A"}}, acronyms_detected=ACR) == {"DIET": set()}

    def test_bare_acronym_hit_skips_the_expansion_query(self, service, monkeypatch):
        calls = self._stub_qdrant(monkeypatch, acronym_hits=["A"], chunk_text={"A": ""})
        assert service._sources_with_acronym_in_body(
            {"DIET": {"A"}}, acronyms_detected=ACR) == {"DIET": {"A"}}
        assert calls == {"acronym": 1, "expansion": 0, "scroll": 0}

    def test_only_unbacked_sources_are_checked_for_the_expansion(self, service, monkeypatch):
        calls = self._stub_qdrant(monkeypatch, acronym_hits=["A"], chunk_text={
            "A": "", "B": "District Institute of Education and Training faculty list."})
        assert service._sources_with_acronym_in_body(
            {"DIET": {"A", "B"}}, acronyms_detected=ACR) == {"DIET": {"A", "B"}}
        assert calls["expansion"] == 1

    def test_without_expansions_behaviour_is_unchanged(self, service, monkeypatch):
        calls = self._stub_qdrant(monkeypatch, acronym_hits=[], chunk_text={
            "A": "District Institute of Education and Training"})
        assert service._sources_with_acronym_in_body({"DIET": {"A"}}) == {"DIET": set()}
        assert calls["expansion"] == 0


# ── A bare-acronym BM25 hit must be the acronym, not the everyday word ────

class TestBareAcronymHitMustUseCapitals:
    """BM25 folds case, so "diet" in a food article matched a DIET query and
    backed a "DIET Handbook" title. The hit now has to use the acronym in
    capitals in one of the returned chunks."""

    _stub_qdrant = TestExpansionInBodyBacksAcronymClaim._stub_qdrant

    @pytest.mark.parametrize("body", [
        "The DIET faculty met on Monday.",
        "Two DIETs in the district ran the course.",
        "All DIETS reported enrolment.",
        "(DIET) coordinators, 2024 cohort.",
    ])
    def test_acronym_in_capitals_backs_the_claim(self, service, monkeypatch, body):
        self._stub_qdrant(monkeypatch, acronym_hits={"A": body}, chunk_text={})
        assert service._sources_with_acronym_in_body({"DIET": {"A"}}) == {"DIET": {"A"}}

    @pytest.mark.parametrize("body", [
        "Eat a balanced diet with plenty of vegetables.",
        "Diet and exercise both matter.",
        "Dietary fibre helps digestion.",
    ])
    def test_everyday_word_does_not_back_the_claim(self, service, monkeypatch, body):
        self._stub_qdrant(monkeypatch, acronym_hits={"A": body}, chunk_text={})
        assert service._sources_with_acronym_in_body({"DIET": {"A"}}) == {"DIET": set()}

    def test_everyday_word_still_falls_through_to_the_expansion_check(
            self, service, monkeypatch):
        calls = self._stub_qdrant(
            monkeypatch,
            acronym_hits={"A": "A healthy diet for trainees."},
            chunk_text={"A": "District Institute of Education and Training timetable."})
        assert service._sources_with_acronym_in_body(
            {"DIET": {"A"}}, acronyms_detected=ACR) == {"DIET": {"A"}}
        assert calls == {"acronym": 1, "expansion": 1, "scroll": 1}

    def test_capitals_outside_the_top_chunks_still_back_the_claim(
            self, service, monkeypatch):
        """The top BM25 chunks are the ones with the most "diet"s; a real DIET
        document can use the acronym in capitals only further down."""
        calls = self._stub_qdrant(
            monkeypatch,
            acronym_hits={"A": "Diet charts for the mid-day meal; a balanced diet."},
            chunk_text={},
            all_chunks={"A": ["Diet charts for the mid-day meal; a balanced diet.",
                              "Prepared by the DIET faculty, Pune."]})
        assert service._sources_with_acronym_in_body({"DIET": {"A"}}) == {"DIET": {"A"}}
        assert calls["scroll"] == 1

    def test_lowercase_in_every_chunk_is_still_rejected(self, service, monkeypatch):
        self._stub_qdrant(
            monkeypatch,
            acronym_hits={"A": "Eat a balanced diet."},
            chunk_text={},
            all_chunks={"A": ["Eat a balanced diet.", "A diet rich in fibre.", "Diet tips."]})
        assert service._sources_with_acronym_in_body({"DIET": {"A"}}) == {"DIET": set()}

    @pytest.mark.parametrize("key, body", [
        ("RTE ACT", "Under the RTE Act, every child is entitled to schooling."),
        ("RTE ACT", "the rte act mandates free education"),
        ("NIPUN BHARAT", "Targets set by NIPUN Bharat for grade 3."),
    ])
    def test_multi_word_key_matches_in_any_case(self, key, body):
        from app.services.acronym_ranking import _acronym_use_pattern
        assert _acronym_use_pattern(key).search(body)

    def test_single_word_key_still_needs_capitals(self):
        from app.services.acronym_ranking import _acronym_use_pattern
        assert not _acronym_use_pattern("DIET").search("a balanced diet")
        assert not _acronym_use_pattern("DIET").search("Diet tips")
        assert _acronym_use_pattern("DIET").search("the DIET faculty")

    def test_accepted_in_top_chunks_needs_no_full_scan(self, service, monkeypatch):
        calls = self._stub_qdrant(
            monkeypatch, acronym_hits={"A": "The DIET faculty met."}, chunk_text={})
        assert service._sources_with_acronym_in_body({"DIET": {"A"}}) == {"DIET": {"A"}}
        assert calls["scroll"] == 0
