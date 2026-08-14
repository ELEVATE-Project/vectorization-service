"""Tests for PR3's /api/acronyms/bulk endpoint: BOM handling and the
cache-outage-must-not-500 regression.

Uses the real app (via TestClient) against live Postgres + Redis — the
endpoint itself hits both directly, so mocking them out would defeat the
point of these specific regression tests. Skipped automatically if either
is unreachable, mirroring the other acronym test files.
"""
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine


def _infra_available() -> bool:
    try:
        from app.config import settings
        from app.core.clients.redis_client import redis_client

        engine = create_engine(settings.DATABASE_URL)
        with engine.connect():
            pass
        redis_client.ping()
        return True
    except Exception:
        return False


requires_infra = pytest.mark.skipif(
    not _infra_available(),
    reason="live Postgres + Redis required (docker: vecsvc-postgres, vecsvc-redis)",
)


@pytest.fixture
def cleanup_test_acronyms():
    """Delete any acronym rows a test creates, by acronym string, after it runs."""
    created = []
    yield created
    if created:
        from app.core.database import SessionLocal
        from app.core.clients.redis_client import redis_client
        from app.models.db_models import AcronymMapping

        db = SessionLocal()
        try:
            db.query(AcronymMapping).filter(AcronymMapping.acronym.in_(created)).delete(
                synchronize_session=False
            )
            db.commit()
        finally:
            db.close()
        for acronym in created:
            redis_client.delete(f"acronym:{acronym}")


@requires_infra
class TestBomHandling:
    """Regression test: a CSV exported from Excel/Sheets typically has a
    leading UTF-8 BOM. Decoding with plain 'utf-8' leaves it attached to the
    first header name ('﻿acronym'), so every row's acronym lookup
    returns None and the whole file silently imports zero rows."""

    def test_bom_prefixed_csv_still_parses_correctly(self, client, cleanup_test_acronyms):
        from app.config import settings

        csv_bytes = (
            b"\xef\xbb\xbfacronym,expansions,description\n"
            b"BOMTEST,BOM Test Expansion,\n"
        )
        cleanup_test_acronyms.append("BOMTEST")

        response = client.post(
            "/api/acronyms/bulk",
            headers={"X-Internal-Token": settings.INTERNAL_API_TOKEN},
            files={"file": ("acronyms.csv", csv_bytes, "text/csv")},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["created"] == 1
        assert body["updated"] == 0
        assert body["errors"] == []

    def test_plain_utf8_csv_without_bom_still_works(self, client, cleanup_test_acronyms):
        """Confirms utf-8-sig doesn't regress the non-BOM case."""
        from app.config import settings

        csv_bytes = b"acronym,expansions,description\nNOBOMTEST,No Bom Expansion,\n"
        cleanup_test_acronyms.append("NOBOMTEST")

        response = client.post(
            "/api/acronyms/bulk",
            headers={"X-Internal-Token": settings.INTERNAL_API_TOKEN},
            files={"file": ("acronyms.csv", csv_bytes, "text/csv")},
        )

        assert response.status_code == 200
        assert response.json()["created"] == 1


@requires_infra
class TestCacheOutageDuringBulkUpload:
    """Regression test: if the post-commit cache refresh fails, the endpoint
    falls back to invalidate_cache() per row — that fallback must not itself
    raise, or a fully successful, already-committed upload surfaces to the
    caller as an opaque 500 with the response body (and the create/update
    counts) lost."""

    def test_cache_refresh_and_invalidate_both_failing_still_returns_200(
        self, client, cleanup_test_acronyms
    ):
        from app.config import settings

        csv_bytes = b"acronym,expansions,description\nCACHEDOWNTEST,Cache Down Test,\n"
        cleanup_test_acronyms.append("CACHEDOWNTEST")

        with patch(
            "app.services.acronym_service.cache_client.set", side_effect=Exception("Redis down")
        ), patch(
            "app.services.acronym_service.cache_client.delete", side_effect=Exception("Redis down")
        ):
            response = client.post(
                "/api/acronyms/bulk",
                headers={"X-Internal-Token": settings.INTERNAL_API_TOKEN},
                files={"file": ("acronyms.csv", csv_bytes, "text/csv")},
            )

        assert response.status_code == 200
        body = response.json()
        assert body["created"] == 1
        assert body["errors"] == []

        # The row must actually be committed despite the cache failures.
        from app.core.database import SessionLocal
        from app.models.db_models import AcronymMapping

        db = SessionLocal()
        try:
            row = (
                db.query(AcronymMapping)
                .filter(AcronymMapping.acronym == "CACHEDOWNTEST")
                .first()
            )
            assert row is not None
            assert row.expansions == ["Cache Down Test"]
        finally:
            db.close()
