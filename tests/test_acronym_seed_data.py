"""Data-integrity regression tests for data/acronyms.csv — the seed file
Alembic uses to populate acronym_mapping. Pure file-content assertions, no
DB/Redis involved.

Guards against two Devin-flagged findings being silently reintroduced by a
future CSV edit:
- app/services/acronym_query_service.py:43 — common English words seeded as
  acronyms (THE, SET, ACT, PROJECT, INTERNAL, ORAL, LIBRARY, HOSTEL,
  CANTEEN) caused all-caps queries to be silently rewritten into unrelated
  topics (e.g. "THE NEW EDUCATION POLICY" -> "Times Higher Education ...").
- app/services/acronym_query_service.py:12 — multi-word keys can never be
  detected (detection is strictly per-token); 4 of them (BA LLB, BBA LLB,
  BCOM LLB, BSC LLB) were pure dead weight since both halves already exist
  as safe standalone entries (BA, LLB, etc.) and would be detected
  independently anyway.
"""
import csv
from pathlib import Path

_CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "acronyms.csv"


def _load_rows():
    with open(_CSV_PATH, newline="") as f:
        return list(csv.DictReader(f))


def _acronyms():
    return {row["acronym"].strip().upper() for row in _load_rows()}


class TestDangerousCommonWordAcronymsRemoved:
    """These 9 rows are common English words — detected as acronyms whenever
    a query happens to be typed in all-caps, silently rewriting unrelated
    queries into a different topic. Deactivated in the DB and removed from
    the seed CSV; must never be reintroduced."""

    DANGEROUS_WORDS = {
        "THE", "SET", "ACT", "PROJECT", "INTERNAL",
        "ORAL", "LIBRARY", "HOSTEL", "CANTEEN",
    }

    def test_dangerous_words_not_in_seed_csv(self):
        acronyms = _acronyms()
        present = self.DANGEROUS_WORDS & acronyms
        assert not present, f"Common-word acronyms reintroduced into seed CSV: {present}"


class TestRedundantMultiWordEntriesRemoved:
    """These 4 rows were pure dead weight: multi-word keys can never be
    detected (per-token detection only), and both halves of each already
    exist as safe, independently-detectable standalone entries — so the
    combined row added zero coverage a user could ever benefit from."""

    REDUNDANT_ENTRIES = {"BA LLB", "BBA LLB", "BCOM LLB", "BSC LLB"}

    def test_redundant_entries_not_in_seed_csv(self):
        acronyms = _acronyms()
        present = self.REDUNDANT_ENTRIES & acronyms
        assert not present, f"Redundant multi-word entries reintroduced into seed CSV: {present}"

    def test_constituent_parts_still_present_and_safe(self):
        # The whole point of removing the combined rows is that BA, LLB,
        # BBA, BCOM are still independently detectable — confirm they exist.
        acronyms = _acronyms()
        for part in ("BA", "LLB", "BBA", "BCOM"):
            assert part in acronyms, f"{part} should still exist as a standalone entry"


class TestNewSafeStandaloneEntriesAdded:
    """JRF, UG, BSE were split out from multi-word combinations (CSIR JRF,
    UG DIPLOMA, BSE ODISHA, etc.) as genuinely safe, specific, reusable
    standalone acronyms — unlike the generic English words filtered out
    above, these aren't ordinary words at risk of false-triggering."""

    EXPECTED = {
        "JRF": ["Junior Research Fellowship"],
        "UG": ["Under Graduate"],
        "BSE": ["Board of Secondary Education"],
    }

    def test_new_entries_present_with_correct_expansion(self):
        rows_by_acronym = {row["acronym"].strip().upper(): row for row in _load_rows()}
        for acronym, expected_expansions in self.EXPECTED.items():
            assert acronym in rows_by_acronym, f"{acronym} missing from seed CSV"
            actual = [e.strip() for e in rows_by_acronym[acronym]["expansions"].split("|")]
            assert actual == expected_expansions


class TestUnreachablePunctuatedEntriesRemoved:
    """10 rows contained a hyphen, apostrophe, or digit that
    _normalize_token strips before lookup, so they could never be detected
    no matter what the user typed — see acronym_query_service.py:12
    finding. Judged not worth fixing (renaming to the normalized form):
    not valuable enough to bother. DDU-GKY was a special case — a
    duplicate of an already-existing DDUGKY row, just removed rather than
    renamed to avoid a collision. 4G/5G are a fundamentally different,
    harder problem than the rest: _normalize_token strips digits entirely,
    so both collapse to the single letter "G" — below the length-2
    detection guard and colliding with each other — no renaming could ever
    fix them without changing the normalization/guard logic itself."""

    REMOVED = {
        "WI-FI", "E-PATHSHALA", "DDU-GKY", "ANTI-BULLYING", "CO-SCHOLASTIC",
        "PRE-BOARD", "RE-EVALUATION", "BYJU'S", "4G", "5G",
    }

    def test_unreachable_entries_not_in_seed_csv(self):
        acronyms = _acronyms()
        present = self.REMOVED & acronyms
        assert not present, f"Unreachable punctuated entries reintroduced into seed CSV: {present}"

    def test_ddugky_survivor_still_present(self):
        # DDU-GKY was removed as a duplicate, not fixed by renaming — DDUGKY
        # (the form the hyphenated version would normalize to anyway) must
        # still be there so the concept isn't lost entirely.
        assert "DDUGKY" in _acronyms()
