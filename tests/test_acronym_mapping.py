"""Tests for PR1: Alembic bootstrap + acronym_mapping schema.

DB-backed tests require a live Postgres (docker: vecsvc-postgres) and are
skipped automatically if it's unreachable, mirroring the `requires_qdrant`
marker pattern already used in tests/conftest.py.
"""
import importlib
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
import sqlalchemy as sa
from sqlalchemy import create_engine, inspect, text

REPO_ROOT = Path(__file__).resolve().parent.parent
MIGRATION_PATH = (
    REPO_ROOT
    / "migrations"
    / "versions"
    / "f13a664a31b6_create_acronym_mapping_table.py"
)


def _load_migration_module():
    spec = importlib.util.spec_from_file_location("acronym_migration", MIGRATION_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestSplitExpansions:
    """Pure-function tests for the CSV pipe-splitting helper used by the seed
    migration (and, per spec §7, the same rule the bulk-upload endpoint must
    follow) — no DB required."""

    @classmethod
    def setup_class(cls):
        cls.migration = _load_migration_module()

    def test_single_expansion(self):
        result = self.migration._split_expansions(
            "District Institute of Education and Training"
        )
        assert result == ["District Institute of Education and Training"]

    def test_multiple_expansions_pipe_separated(self):
        result = self.migration._split_expansions(
            "Staff Selection Commission|Sainik School Society"
        )
        assert result == ["Staff Selection Commission", "Sainik School Society"]

    def test_trims_whitespace_around_pipes(self):
        result = self.migration._split_expansions(
            "  Parent Teacher Association  |  Parent Teachers Association  "
        )
        assert result == ["Parent Teacher Association", "Parent Teachers Association"]

    def test_dedupes_while_preserving_order(self):
        assert self.migration._split_expansions("A|B|A|C|B") == ["A", "B", "C"]

    def test_empty_segments_ignored(self):
        assert self.migration._split_expansions("A||B|") == ["A", "B"]

    def test_blank_input_returns_empty_list(self):
        assert self.migration._split_expansions("") == []


def _postgres_available() -> bool:
    try:
        from app.config import settings

        engine = create_engine(settings.DATABASE_URL)
        with engine.connect():
            pass
        return True
    except Exception:
        return False


requires_postgres = pytest.mark.skipif(
    not _postgres_available(),
    reason="live Postgres required (docker: vecsvc-postgres)",
)


def _run_alembic(*args):
    result = subprocess.run(
        [sys.executable, "-m", "alembic", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result


@pytest.fixture
def postgres_engine():
    from app.config import settings

    return create_engine(settings.DATABASE_URL)


@pytest.fixture
def migrated_db(postgres_engine):
    """Drop everything Alembic manages and re-run migrations from scratch,
    so each test in this fixture's scope starts from a known, real state."""
    with postgres_engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS acronym_mapping"))
        conn.execute(text("DROP TABLE IF EXISTS translations"))
        conn.execute(text("DROP TABLE IF EXISTS alembic_version"))
    _run_alembic("upgrade", "head")
    yield postgres_engine


@requires_postgres
class TestCreateAllDoesNotRaceAlembic:
    """Regression test for the exact bug Devin flagged on PR1: app startup's
    Base.metadata.create_all() must not create acronym_mapping — that table
    is Alembic-managed exclusively. If this regresses, a fresh deploy that
    boots the app before running migrations would end up with an empty,
    Alembic-incompatible acronym_mapping table (DuplicateTable on next
    `alembic upgrade head`)."""

    def test_app_boot_before_migrations_only_creates_translations(self, postgres_engine):
        with postgres_engine.begin() as conn:
            conn.execute(text("DROP TABLE IF EXISTS acronym_mapping"))
            conn.execute(text("DROP TABLE IF EXISTS translations"))

        # Simulate "app boots before any migration has run" by (re-)importing
        # app.core.database, whose module-level create_all() call is exactly
        # what a real app boot triggers.
        import app.core.database as db_module

        importlib.reload(db_module)

        tables = inspect(postgres_engine).get_table_names()
        assert "translations" in tables
        assert "acronym_mapping" not in tables

    def test_migration_then_succeeds_without_duplicate_table_error(self, postgres_engine):
        # Continues from the state left by the previous test: translations
        # exists (via create_all), acronym_mapping does not. This is the
        # actual failure mode Devin flagged — assert it no longer happens.
        with postgres_engine.begin() as conn:
            conn.execute(text("DROP TABLE IF EXISTS alembic_version"))
        result = _run_alembic("upgrade", "head")
        assert "DuplicateTable" not in result.stdout
        assert "DuplicateTable" not in result.stderr

        tables = inspect(postgres_engine).get_table_names()
        assert "acronym_mapping" in tables


@requires_postgres
class TestAcronymMappingSchema:
    """Verifies the migrated schema matches spec §3 exactly."""

    def test_columns_and_types(self, migrated_db):
        inspector = inspect(migrated_db)
        columns = {c["name"]: c for c in inspector.get_columns("acronym_mapping")}

        assert set(columns) == {
            "id",
            "acronym",
            "expansions",
            "description",
            "is_active",
            "created_at",
            "updated_at",
        }
        assert not columns["acronym"]["nullable"]
        assert not columns["expansions"]["nullable"]
        assert columns["description"]["nullable"]
        assert not columns["is_active"]["nullable"]
        assert isinstance(columns["expansions"]["type"], sa.dialects.postgresql.JSONB)

    def test_primary_key_is_id_not_acronym(self, migrated_db):
        inspector = inspect(migrated_db)
        pk = inspector.get_pk_constraint("acronym_mapping")
        assert pk["constrained_columns"] == ["id"]

    def test_unique_constraint_on_acronym(self, migrated_db):
        inspector = inspect(migrated_db)
        unique_constraints = inspector.get_unique_constraints("acronym_mapping")
        assert any(
            uc["column_names"] == ["acronym"] for uc in unique_constraints
        )

    def test_composite_index_on_acronym_and_is_active(self, migrated_db):
        inspector = inspect(migrated_db)
        indexes = inspector.get_indexes("acronym_mapping")
        assert any(
            idx["name"] == "idx_acronym_active"
            and idx["column_names"] == ["acronym", "is_active"]
            for idx in indexes
        )


@requires_postgres
class TestSeedData:
    """Verifies the real acronym dataset seeds correctly. Row count is 589,
    not the original 599 — 13 rows were removed (9 common-English-word
    false positives like THE/SET/ACT, 4 redundant multi-word entries whose
    parts already exist standalone) and 3 added (JRF, UG, BSE) during a
    curation pass, see acronym_query_service.py:43 and :12 findings."""

    def test_row_count(self, migrated_db):
        with migrated_db.connect() as conn:
            count = conn.execute(
                text("SELECT count(*) FROM acronym_mapping")
            ).scalar()
        assert count == 589

    def test_multi_expansion_acronym(self, migrated_db):
        with migrated_db.connect() as conn:
            row = conn.execute(
                text("SELECT expansions FROM acronym_mapping WHERE acronym = 'SSC'")
            ).first()
        assert row is not None
        assert row[0] == ["Staff Selection Commission", "Sainik School Society"]

    def test_acronyms_stored_uppercase(self, migrated_db):
        with migrated_db.connect() as conn:
            non_upper = conn.execute(
                text("SELECT acronym FROM acronym_mapping WHERE acronym != upper(acronym)")
            ).fetchall()
        assert non_upper == []

    def test_single_expansion_acronym(self, migrated_db):
        with migrated_db.connect() as conn:
            row = conn.execute(
                text("SELECT expansions FROM acronym_mapping WHERE acronym = 'DIET'")
            ).first()
        assert row[0] == ["District Institute of Education and Training"]


@requires_postgres
class TestMigrationIdempotency:
    """Spec §5: the seed migration must be safe to re-run against a DB that
    already has seed data (ON CONFLICT upsert, not a plain INSERT)."""

    def test_downgrade_then_upgrade_cycle_is_clean(self, migrated_db):
        _run_alembic("downgrade", "-1")
        result = _run_alembic("upgrade", "head")
        assert result.returncode == 0

        with migrated_db.connect() as conn:
            count = conn.execute(
                text("SELECT count(*) FROM acronym_mapping")
            ).scalar()
        assert count == 589
