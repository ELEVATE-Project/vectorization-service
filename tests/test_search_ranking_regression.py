"""Regression tests for the Release 2.0 scoring/ranking contract on the acronym branch.

Acronym support must widen candidate RETRIEVAL and re-order within the acronym
path only — it must never disturb an ordinary query. These tests pin:

1. The Release 2.0 fusion formula: no acronym-scoped dense/sparse weight flip.
2. filter_score / ordering / top_k semantics.
3. The injection contract: a document the pipeline actually scored keeps its
   real score and field_scores when re-injected; one never retrieved still
   takes the untouched Release 2.0 floor.
4. Title/summary multipliers unchanged.
5. Acronym queries: title/summary boosts on the acronym, plus a text boost when
   the body uses the acronym as written. Expansions are not matched in titles,
   summaries or bodies; retrieval and the relevance blend cover them. Expansion
   phrase matching (_phrase_in_text, on content words) is still used to decide
   whether the query already spells out an expansion.
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
    """Title/summary match rules for acronym queries (_classify_field_match): the
    query and the acronym as whole words. The expansion is not matched in titles."""

    @staticmethod
    def _match(service, title):
        queries = service._normalize_field_match_queries(
            service._build_field_match_queries("DIET", ACR))
        found = {service._classify_field_match(text, rule, title.lower())
                 for _, text, rule in queries}
        return "exact" if "exact" in found else "partial" if "partial" in found else None

    @pytest.mark.parametrize("title, expected", [
        ("DIET Stakeholder Identity Map", "partial"),
        ("DIET", "exact"),
        # The expansion alone no longer matches a title.
        ("Strengthening of District Institutes of Education and Training", None),
        ("Establishing Discipline Through Clear School Rules", None),
        # Never inside a longer word...
        ("Dietary guidelines for schools", None),
        # ...but file-name titles use '_' as a separator.
        ("source_doc_COE_AM4C2_DIET_Empowerment_Design.xlsx", "partial"),
    ])
    def test_title_match(self, service, title, expected):
        assert self._match(service, title) == expected


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

    def test_reused_doc_reports_its_measured_score(self, service, monkeypatch):
        """A document scored before the threshold takes the floor, its real score is
        kept as measured_relevance, and it shows no pre-boost score."""
        self._patch_scroll(monkeypatch, [])
        real = {"id": "pt-f", "payload": {"source_id": "F"}, "weighted_score": 0.31,
                "pre_boost_score": 0.31, "field_scores": {}}
        entry = service._fetch_field_match_docs(
            ["F"], {"F": "partial"}, "title",
            settings.EXACT_TITLE_BOOST, settings.PARTIAL_TITLE_BOOST,
            prethreshold_by_source={"F": real},
        )[0]
        assert entry["weighted_score"] == pytest.approx(self.FLOOR * settings.PARTIAL_TITLE_BOOST)
        assert entry["measured_relevance"] == 0.31
        assert "pre_boost_score" not in entry


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
    """Acronym ranking (relevance blend, title/summary/text boosts) rides the same
    switch as the title/summary boost: on in hybrid mode, off in semantic mode and
    when HYBRID_SEARCH_ENABLED is false.

    Doc A carries the acronym in its title; doc B carries none. In hybrid mode A's
    title earns the partial title boost, so A wins only when it is close enough for
    the boost to make up the difference. In the default fixture A (0.135) is well
    under B (0.306), so B wins.
    """

    def _run(self, service, monkeypatch, search_mode, hybrid_enabled=True,
             field_scores=None, acronyms=None, top_k=10, titles=None, **request_fields):
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
        assert response.acronym_info == {"detected": True, "mapping": dict(ACR)}

    @pytest.mark.parametrize("search_mode, expected", [("semantic", False), ("hybrid", True)])
    def test_text_boost_runs_only_in_hybrid_mode(
            self, service, monkeypatch, search_mode, expected):
        seen = []
        monkeypatch.setattr(
            PrioritizedSearchService, "_text_matches_for_acronyms",
            lambda self, sources, acronyms, **kw: seen.append(sources) or {})
        self._run(service, monkeypatch, search_mode, field_scores={
            "pt-A": {"title": 0.30, "text": 0.10, SPARSE: 5.0}, "pt-B": {"title": 0.90}})
        assert bool(seen) is expected


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
        assert ctx["boost_config"] == {
            "mode": "acronym_field_boost",
            "exact_title_boost": settings.EXACT_TITLE_BOOST,
            "partial_title_boost": settings.PARTIAL_TITLE_BOOST,
            "exact_summary_boost": settings.EXACT_SUMMARY_BOOST,
            "partial_summary_boost": settings.PARTIAL_SUMMARY_BOOST,
            "exact_text_boost": settings.EXACT_TEXT_BOOST,
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
        # No candidate has a BM25 score in this fixture, so the body check is skipped.
        assert ranking["body_check"] == "unavailable"

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


# ── Acronym debug fields: pre-boost score and text boost ──────────────────

class TestAcronymDebugFields:
    """pre_boost_score is set only on rows the relevance blend rescored; text_match
    and text_multiplier only where the text boost ran. All are debug-only."""

    @staticmethod
    def _row(**extra):
        return {"id": "p1", "payload": {"source_id": "S", "title": "T", "text": "x"},
                "field_scores": {}, "weighted_score": 0.8,
                "match_source": "title_keyword_match", **extra}

    def test_ordinary_injected_row_has_no_acronym_fields(self, service):
        item = service._build_result_items([self._row()], True)[0]
        assert item.pre_boost_score is None and item.text_multiplier is None
        assert item.match_source == "title_keyword_match"

    def test_rescored_row_shows_its_pre_boost_score(self, service):
        item = service._build_result_items([self._row(pre_boost_score=0.5)], True)[0]
        assert item.pre_boost_score == 0.5

    def test_fields_hidden_without_debug(self, service):
        item = service._build_result_items(
            [self._row(pre_boost_score=0.5, text_multiplier=2.0)], False)[0]
        assert item.pre_boost_score is None and item.text_multiplier is None


# ── Acronym queries: title/summary boosts plus the text boost ─────────────

class TestAcronymFieldBoosts:
    """Acronym queries get the title/summary boosts plus a text boost when the body
    uses the acronym (exact). The text boost needs BM25-indexed candidates."""

    # Pool rows with a BM25 score: the text boost needs BM25-indexed candidates.
    INDEXED = {"pt-A": {"title": 0.30, "text": 0.10, SPARSE: 5.0},
               "pt-B": {"title": 0.90, SPARSE: 2.0}}

    def _run(self, service, monkeypatch, text_matches=None, field_scores=None,
             include_scoring_debug=True):
        monkeypatch.setattr(
            PrioritizedSearchService, "_text_matches_for_acronyms",
            lambda self, sources, acronyms, **kw: dict(text_matches or {}))
        # Copied per run: the search writes match keys into these dicts.
        field_scores = {pid: dict(s) for pid, s in (field_scores or self.INDEXED).items()}
        response = TestAcronymRankingFollowsLexicalRanking()._run(
            service, monkeypatch, "hybrid", include_scoring_debug=include_scoring_debug,
            field_scores=field_scores)
        return response, {r.source_id: r for r in response.results}

    def test_text_boost_stacks_on_the_title_boost(self, service, monkeypatch):
        _, by_source = self._run(service, monkeypatch, {"A": "exact"})
        a, b = by_source["A"], by_source["B"]
        assert (a.text_match, a.text_multiplier) == ("exact", settings.EXACT_TEXT_BOOST)
        assert (b.text_match, b.text_multiplier) == (None, 1.0)
        # A's title "DIET Handbook" is a partial title match; B's title has no match.
        assert a.title_multiplier == settings.PARTIAL_TITLE_BOOST
        assert a.score == pytest.approx(
            min(a.pre_boost_score * settings.PARTIAL_TITLE_BOOST * settings.EXACT_TEXT_BOOST, 1.0))
        assert b.title_multiplier == 1.0
        assert b.score == pytest.approx(b.pre_boost_score)

    def test_score_rebuilds_from_the_pre_boost_score(self, service, monkeypatch):
        _, by_source = self._run(service, monkeypatch, {"A": "exact"})
        for r in by_source.values():
            assert r.score == pytest.approx(min(
                r.pre_boost_score * r.title_multiplier * r.summary_multiplier
                * r.text_multiplier, 1.0))

    def test_pre_boost_score_hidden_without_debug(self, service, monkeypatch):
        _, by_source = self._run(service, monkeypatch, {"A": "exact"},
                                 include_scoring_debug=False)
        assert all(r.pre_boost_score is None for r in by_source.values())

    def _status(self, service, monkeypatch, scanned_sources, result=True):
        """body_check for a run whose body check read `scanned_sources` directly;
        result=False makes the check unable to answer."""
        def text_matches(self, sources, acronyms, scanned_out=None):
            scanned_out.update(scanned_sources)
            return {"A": "exact"} if result else None

        monkeypatch.setattr(PrioritizedSearchService, "_text_matches_for_acronyms", text_matches)
        unindexed = {"pt-A": {"title": 0.30, "text": 0.10}, "pt-B": {"title": 0.90}}
        response = TestAcronymRankingFollowsLexicalRanking()._run(
            service, monkeypatch, "hybrid", include_scoring_debug=True, field_scores=unindexed)
        return response, {r.source_id: r for r in response.results}

    def test_unindexed_pool_still_gets_the_body_check(self, service, monkeypatch):
        """No candidate has a BM25 score (documents indexed before BM25): their
        chunks are read directly, so A still earns the text boost."""
        response, by_source = self._status(service, monkeypatch, {"A", "B"})
        assert by_source["A"].text_match == "exact"
        assert by_source["A"].text_multiplier == settings.EXACT_TEXT_BOOST
        assert response.search_config["scoring_context"]["acronym_ranking"]["body_check"] == "scan"

    @pytest.mark.parametrize("scanned, result, expected", [
        (set(), True, "bm25"),
        ({"A"}, True, "bm25+scan"),
        ({"A", "B"}, True, "scan"),
        (set(), False, "unavailable"),
    ])
    def test_body_check_status(self, service, monkeypatch, scanned, result, expected):
        response, _ = self._status(service, monkeypatch, scanned, result)
        assert response.search_config["scoring_context"]["acronym_ranking"]["body_check"] == expected

    def test_indexed_pool_runs_the_text_boost(self, service, monkeypatch):
        response, by_source = self._run(service, monkeypatch, {"A": "exact"})
        assert by_source["A"].text_match == "exact"
        assert response.search_config["scoring_context"]["acronym_ranking"]["body_check"] == "bm25"

    def test_text_boost_can_lift_a_document_onto_the_page(self, service, monkeypatch):
        """The cut to top_k comes after the boosts. With top_k=1 and dense/BM25 weights
        0.6/0.4, B (best dense) starts at 0.6 and A (only BM25 hit) at 0.4, and A has
        no title match to re-inject it. A's body uses the acronym: x2.0 gives 0.8, so
        A must win the single slot."""
        monkeypatch.setattr(settings, "HYBRID_DENSE_WEIGHT", 0.6)
        monkeypatch.setattr(settings, "HYBRID_SPARSE_WEIGHT", 0.4)
        monkeypatch.setattr(
            PrioritizedSearchService, "_text_matches_for_acronyms",
            lambda self, sources, acronyms, **kw: {"A": "exact"})
        response = TestAcronymRankingFollowsLexicalRanking()._run(
            service, monkeypatch, "hybrid", top_k=1, include_scoring_debug=True,
            titles={"A": "Annual handbook", "B": "Nutrition guide"},
            field_scores={"pt-A": {"title": 0.30, "text": 0.10, SPARSE: 5.0},
                          "pt-B": {"title": 0.90}})
        assert [r.source_id for r in response.results] == ["A"]
        assert response.results[0].text_multiplier == settings.EXACT_TEXT_BOOST
        assert response.results[0].score == pytest.approx(0.4 * settings.EXACT_TEXT_BOOST)

    def test_body_check_skips_candidates_that_cannot_reach_the_page(
            self, service, monkeypatch):
        """With top_k=1 and dense/BM25 weights 0.8/0.2, B starts at 0.8 and A at 0.2;
        even x2.0 leaves A at 0.4, below B, so A's body is never checked."""
        monkeypatch.setattr(settings, "HYBRID_DENSE_WEIGHT", 0.8)
        monkeypatch.setattr(settings, "HYBRID_SPARSE_WEIGHT", 0.2)
        seen = []
        monkeypatch.setattr(
            PrioritizedSearchService, "_text_matches_for_acronyms",
            lambda self, sources, acronyms, **kw: seen.append(set(sources)) or {})
        response = TestAcronymRankingFollowsLexicalRanking()._run(
            service, monkeypatch, "hybrid", top_k=1,
            titles={"A": "Annual handbook", "B": "Nutrition guide"},
            field_scores={"pt-A": {"title": 0.30, "text": 0.10, SPARSE: 5.0},
                          "pt-B": {"title": 0.90, SPARSE: 1.0}})
        assert seen == [{"B"}]
        assert [r.source_id for r in response.results] == ["B"]

    def test_bm25_evidence_outside_the_checked_pool_still_counts(
            self, service, monkeypatch):
        """Only B can reach the page and it is a dense-only hit (no BM25 score); A
        has one but cannot reach the page. The index is still BM25-backed, so B's
        body is checked and boosted rather than the whole check being skipped."""
        monkeypatch.setattr(settings, "HYBRID_DENSE_WEIGHT", 0.8)
        monkeypatch.setattr(settings, "HYBRID_SPARSE_WEIGHT", 0.2)
        seen = []
        monkeypatch.setattr(
            PrioritizedSearchService, "_text_matches_for_acronyms",
            lambda self, sources, acronyms, **kw: seen.append(set(sources)) or {"B": "exact"})
        response = TestAcronymRankingFollowsLexicalRanking()._run(
            service, monkeypatch, "hybrid", top_k=1, include_scoring_debug=True,
            titles={"A": "Annual handbook", "B": "Nutrition guide"},
            field_scores={"pt-A": {"title": 0.30, "text": 0.10, SPARSE: 5.0},
                          "pt-B": {"title": 0.90}})
        assert seen == [{"B"}]
        assert response.results[0].source_id == "B"
        assert response.results[0].text_match == "exact"
        assert response.search_config["scoring_context"]["acronym_ranking"]["body_check"] == "bm25"

    def test_body_check_reports_a_direct_read(self, service, monkeypatch):
        """Sources whose chunks lack a BM25 vector are read directly; the debug
        status says so instead of claiming a plain BM25 check."""
        def text_matches(self, sources, acronyms, scanned_out=None):
            scanned_out.add("A")
            return {"A": "exact"}

        monkeypatch.setattr(PrioritizedSearchService, "_text_matches_for_acronyms", text_matches)
        response = TestAcronymRankingFollowsLexicalRanking()._run(
            service, monkeypatch, "hybrid", include_scoring_debug=True,
            field_scores={pid: dict(s) for pid, s in self.INDEXED.items()})
        assert {r.source_id: r for r in response.results}["A"].text_match == "exact"
        assert response.search_config["scoring_context"]["acronym_ranking"]["body_check"] == "bm25+scan"


class TestBodyCheckPool:
    """_body_check_pool: candidates that can still reach the page, best first, capped."""

    @staticmethod
    def _rows(*scores):
        return [{"id": f"p{i}", "weighted_score": s} for i, s in enumerate(scores)]

    def test_keeps_only_candidates_that_can_reach_the_page(self):
        # top_k=2: the k-th score is 0.6; 0.31 x 2 = 0.62 can pass it, 0.29 x 2 cannot.
        pool = PrioritizedSearchService._body_check_pool(self._rows(0.9, 0.6, 0.31, 0.29), 2)
        assert [r["weighted_score"] for r in pool] == [0.9, 0.6, 0.31]

    def test_a_boosted_tie_with_the_kth_score_is_kept(self):
        pool = PrioritizedSearchService._body_check_pool(self._rows(0.9, 0.6, 0.3), 2)
        assert [r["weighted_score"] for r in pool] == [0.9, 0.6, 0.3]

    def test_everything_is_checked_when_the_page_is_not_full(self):
        pool = PrioritizedSearchService._body_check_pool(self._rows(0.2, 0.9, 0.1), 10)
        assert [r["weighted_score"] for r in pool] == [0.9, 0.2, 0.1]

    def test_the_pool_is_capped_best_first(self, monkeypatch):
        monkeypatch.setattr(settings, "ACRONYM_BODY_CHECK_MAX_SOURCES", 2)
        pool = PrioritizedSearchService._body_check_pool(self._rows(0.5, 0.9, 0.7, 0.8), 10)
        assert [r["weighted_score"] for r in pool] == [0.9, 0.8]


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
        monkeypatch.setattr(
            PrioritizedSearchService, "_sources_with_acronym_in_body",
            lambda self, *a, **kw: {})
        monkeypatch.setattr(PrioritizedSearchService, "_blended_acronym_relevance", spy)

        service.search(PrioritizedSearchRequest(query="DIET", top_k=10, search_mode="hybrid"))

        ref = seen["normalization_reference"]
        assert ref and "dense_min" in ref and "dense_max" in ref


# ── The body check reads the acronym only, not its expansion ──────────────

class TestBodyCheckUsesTheAcronym:
    """The text boost needs the body to use the acronym as written. A body that
    only spells out the expansion does not count: retrieval and the relevance
    blend already cover expansion matches."""

    def _stub_qdrant(self, monkeypatch, acronym_hits, all_chunks=None, unindexed=None):
        """acronym_hits: source -> body text the bare-acronym BM25 query returns
        (a list is shorthand for chunks that use "DIET" in capitals).
        all_chunks: source -> every chunk's text, for the full-scan scroll
        (defaults to just the BM25 chunk).
        unindexed: source -> texts of its chunks without a BM25 vector (none by default)."""
        if not isinstance(acronym_hits, dict):
            acronym_hits = {s: "The DIET faculty met." for s in acronym_hits}
        all_chunks = all_chunks or {s: [t] for s, t in acronym_hits.items()}
        unindexed = unindexed or {}
        calls = {"acronym": 0, "scroll": 0, "unindexed": 0}

        def fake_scroll(**kwargs):
            wanted = set(kwargs["scroll_filter"].must[0].match.any)
            # The "chunks without a BM25 vector" lookup filters with must_not has_vector.
            source = unindexed if kwargs["scroll_filter"].must_not else all_chunks
            calls["unindexed" if kwargs["scroll_filter"].must_not else "scroll"] += 1
            points = [
                types.SimpleNamespace(payload={"source_id": s, "text": t})
                for s, texts in source.items() if s in wanted for t in texts]
            return points, None

        def fake_groups(**kwargs):
            calls["acronym"] += 1
            wanted = set(kwargs["query_filter"].must[0].match.any)
            return types.SimpleNamespace(groups=[
                types.SimpleNamespace(id=s, hits=[
                    types.SimpleNamespace(payload={"text": acronym_hits[s]})])
                for s in acronym_hits if s in wanted])

        monkeypatch.setattr(settings, "SPARSE_SEARCH_ENABLED", True)
        monkeypatch.setattr(
            "app.services.acronym_ranking.qdrant_client.query_points_groups", fake_groups)
        monkeypatch.setattr(
            "app.services.acronym_ranking.qdrant_client.scroll", fake_scroll)
        monkeypatch.setattr(
            "app.core.clients.sparse_encoder.generate_sparse_vector",
            lambda text: ([1], [1.0]))
        return calls

    def test_spelled_out_expansion_alone_does_not_count(self, service, monkeypatch):
        self._stub_qdrant(monkeypatch, acronym_hits={
            "A": "Each District Institute of Education and Training runs in-service courses."})
        assert service._sources_with_acronym_in_body({"DIET": {"A"}}) == {"DIET": set()}

    def test_unrelated_body_does_not_count(self, service, monkeypatch):
        self._stub_qdrant(monkeypatch, acronym_hits={
            "A": "Check tyre pressure weekly and lubricate the bicycle chain."})
        assert service._sources_with_acronym_in_body({"DIET": {"A"}}) == {"DIET": set()}

    def test_text_matches_are_exact_for_bodies_using_the_acronym(self, service, monkeypatch):
        self._stub_qdrant(monkeypatch, acronym_hits={
            "A": "The DIET faculty met.",
            "B": "District Institute of Education and Training staff."})
        assert service._text_matches_for_acronyms({"A", "B"}, ACR) == {"A": "exact"}

    def test_text_matches_none_without_bm25(self, service, monkeypatch):
        monkeypatch.setattr(settings, "SPARSE_SEARCH_ENABLED", False)
        assert service._text_matches_for_acronyms({"A"}, ACR) is None

    def test_chunks_without_a_bm25_vector_are_read_directly(self, service, monkeypatch):
        """B has no BM25 vector, so the BM25 query never returns it; its text is read
        directly instead. C is unindexed too but only uses the everyday word."""
        calls = self._stub_qdrant(
            monkeypatch, acronym_hits={"A": "The DIET faculty met."},
            unindexed={"B": ["Prepared by the DIET faculty, Pune."], "C": ["A healthy diet."]})
        scanned = set()
        assert service._sources_with_acronym_in_body(
            {"DIET": {"A", "B", "C"}}, scanned_out=scanned) == {"DIET": {"A", "B"}}
        assert scanned == {"B", "C"}
        assert calls["unindexed"] == 1

    def test_fully_indexed_sources_report_no_scan(self, service, monkeypatch):
        self._stub_qdrant(monkeypatch, acronym_hits={"A": "The DIET faculty met."})
        scanned = set()
        assert service._text_matches_for_acronyms({"A"}, ACR, scanned_out=scanned) == {"A": "exact"}
        assert scanned == set()

    def test_a_failed_unindexed_lookup_keeps_the_bm25_answer(self, service, monkeypatch):
        """An older Qdrant without has_vector filtering: the lookup fails, those chunks
        stay unchecked, and the BM25 result still stands."""
        from app.services import acronym_ranking

        self._stub_qdrant(monkeypatch, acronym_hits={"A": "The DIET faculty met."})
        stub_scroll = acronym_ranking.qdrant_client.scroll

        def scroll_without_has_vector(**kwargs):
            if kwargs["scroll_filter"].must_not:
                raise RuntimeError("has_vector is not supported")
            return stub_scroll(**kwargs)

        monkeypatch.setattr(
            "app.services.acronym_ranking.qdrant_client.scroll", scroll_without_has_vector)
        scanned = set()
        assert service._sources_with_acronym_in_body(
            {"DIET": {"A"}}, scanned_out=scanned) == {"DIET": {"A"}}
        assert scanned == set()


# ── A bare-acronym BM25 hit must be the acronym, not the everyday word ────

class TestBareAcronymHitMustUseCapitals:
    """BM25 folds case, so "diet" in a food article matched a DIET query and
    backed a "DIET Handbook" title. The hit now has to use the acronym in
    capitals in one of the returned chunks."""

    _stub_qdrant = TestBodyCheckUsesTheAcronym._stub_qdrant

    @pytest.mark.parametrize("body", [
        "The DIET faculty met on Monday.",
        "Two DIETs in the district ran the course.",
        "All DIETS reported enrolment.",
        "(DIET) coordinators, 2024 cohort.",
    ])
    def test_acronym_in_capitals_backs_the_claim(self, service, monkeypatch, body):
        self._stub_qdrant(monkeypatch, acronym_hits={"A": body})
        assert service._sources_with_acronym_in_body({"DIET": {"A"}}) == {"DIET": {"A"}}

    @pytest.mark.parametrize("body", [
        "Eat a balanced diet with plenty of vegetables.",
        "Diet and exercise both matter.",
        "Dietary fibre helps digestion.",
    ])
    def test_everyday_word_does_not_back_the_claim(self, service, monkeypatch, body):
        self._stub_qdrant(monkeypatch, acronym_hits={"A": body})
        assert service._sources_with_acronym_in_body({"DIET": {"A"}}) == {"DIET": set()}

    def test_capitals_outside_the_top_chunks_still_back_the_claim(
            self, service, monkeypatch):
        """The top BM25 chunks are the ones with the most "diet"s; a real DIET
        document can use the acronym in capitals only further down."""
        calls = self._stub_qdrant(
            monkeypatch,
            acronym_hits={"A": "Diet charts for the mid-day meal; a balanced diet."},
            all_chunks={"A": ["Diet charts for the mid-day meal; a balanced diet.",
                              "Prepared by the DIET faculty, Pune."]})
        assert service._sources_with_acronym_in_body({"DIET": {"A"}}) == {"DIET": {"A"}}
        assert calls["scroll"] == 1

    def test_lowercase_in_every_chunk_is_still_rejected(self, service, monkeypatch):
        self._stub_qdrant(
            monkeypatch,
            acronym_hits={"A": "Eat a balanced diet."},
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
            monkeypatch, acronym_hits={"A": "The DIET faculty met."})
        assert service._sources_with_acronym_in_body({"DIET": {"A"}}) == {"DIET": {"A"}}
        assert calls["scroll"] == 0
