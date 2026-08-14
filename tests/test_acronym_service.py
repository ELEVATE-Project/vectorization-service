"""Tests for PR2: acronym cache-aside lookup + startup warm-up.

DB/Redis-backed tests require live Postgres + Redis (docker: vecsvc-postgres,
vecsvc-redis) and are skipped automatically if unreachable, mirroring
tests/test_acronym_mapping.py.
"""
import asyncio
from unittest.mock import AsyncMock, patch

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


@requires_infra
class TestCaseNormalization:
    """Regression test: get_expansion()/invalidate_cache() must resolve
    regardless of the caller's input case, since rows are stored uppercase
    (spec §3/§8) but nothing enforces callers to uppercase first."""

    @pytest.fixture(autouse=True)
    def flush_redis(self):
        from app.core.clients.redis_client import redis_client

        redis_client.flushdb()
        yield
        redis_client.flushdb()

    @pytest.mark.parametrize("variant", ["mba", "Mba", "MBA", "  mba  "])
    def test_get_expansion_case_insensitive(self, variant):
        from app.services import acronym_service

        result = acronym_service.get_expansion(variant)
        assert result == ["Master of Business Administration"]

    def test_lowercase_and_uppercase_share_the_same_cache_entry(self):
        from app.services import acronym_service
        from app.core.clients.redis_client import redis_client

        acronym_service.get_expansion("mba")
        assert redis_client.exists("acronym:MBA")
        assert not redis_client.exists("acronym:mba")

    def test_invalidate_cache_with_lowercase_clears_the_real_key(self):
        from app.services import acronym_service
        from app.core.clients.redis_client import redis_client

        acronym_service.get_expansion("MBA")
        assert redis_client.exists("acronym:MBA")

        acronym_service.invalidate_cache("mba")
        assert not redis_client.exists("acronym:MBA")

    def test_unknown_acronym_still_returns_none(self):
        from app.services import acronym_service

        assert acronym_service.get_expansion("notreal") is None


@requires_infra
class TestCacheOutageFallsBackToPostgres:
    """Regression test: a Redis outage during a lookup (not just at startup
    warm-up) must degrade to Postgres, not raise. This is what main.py's own
    comment claims ("get_expansion() already falls back to Postgres per
    lookup") — that claim was false for cache *errors* (as opposed to cache
    *misses*) until this fix."""

    @pytest.fixture(autouse=True)
    def flush_redis(self):
        from app.core.clients.redis_client import redis_client

        redis_client.flushdb()
        yield
        redis_client.flushdb()

    def test_cache_read_failure_falls_back_to_postgres(self):
        from app.services import acronym_service

        with patch("app.core.clients.cache_client.get", side_effect=Exception("Redis down")):
            result = acronym_service.get_expansion("MBA")
        assert result == ["Master of Business Administration"]

    def test_cache_write_through_failure_does_not_lose_the_result(self):
        from app.services import acronym_service

        with patch("app.core.clients.cache_client.set", side_effect=Exception("Redis down")):
            result = acronym_service.get_expansion("DIET")
        assert result == ["District Institute of Education and Training"]

    def test_invalidate_cache_failure_does_not_raise(self):
        """Regression test: invalidate_cache() is itself the degraded-mode
        fallback the bulk-upload endpoint calls when warm_cache() has
        already failed — if invalidate_cache() also raises on the same
        outage, a successful, already-committed upload would surface to the
        caller as a 500."""
        from app.services import acronym_service

        with patch("app.core.clients.cache_client.delete", side_effect=Exception("Redis down")):
            acronym_service.invalidate_cache("MBA")  # must not raise


class TestRedisPasswordConfiguration:
    """Regression test: the shared Redis connection must forward
    REDIS_PASSWORD, or every acronym cache operation silently fails with
    AuthenticationError in any environment whose Redis requires one."""

    def test_password_is_forwarded_when_configured(self):
        import importlib

        import app.core.clients.redis_client as redis_client_module

        try:
            with patch("app.config.settings.REDIS_PASSWORD", "s3cret"):
                importlib.reload(redis_client_module)
                kwargs = redis_client_module.redis_client.connection_pool.connection_kwargs
                assert kwargs["password"] == "s3cret"
        finally:
            # Restore the real client *after* the patch has been undone,
            # not while REDIS_PASSWORD is still overridden — otherwise this
            # "restore" reload would rebuild the client with the patched
            # value still in effect, leaking "s3cret" into later tests.
            importlib.reload(redis_client_module)

    def test_blank_password_becomes_none_not_empty_string(self):
        """redis-py treats an empty-string password as a real credential and
        attempts AUTH with it; None means 'skip AUTH entirely'. REDIS_PASSWORD
        defaults to "" (app/config.py), so this must be normalized."""
        from app.core.clients.redis_client import redis_client

        kwargs = redis_client.connection_pool.connection_kwargs
        assert kwargs["password"] is None


class TestStartupFaultTolerance:
    """Regression test: a failed acronym cache warm-up (e.g. Postgres not yet
    migrated, or a transient Redis outage) must not take down the whole
    service — get_expansion() already has a per-lookup DB fallback, so
    warm-up is an optimization, not a hard dependency."""

    def test_warm_cache_failure_does_not_prevent_startup(self):
        with patch(
            "app.main.warm_acronym_cache",
            AsyncMock(side_effect=RuntimeError("Postgres unreachable")),
        ), patch("app.main.ensure_collections_exist", AsyncMock(return_value=None)):
            from app.main import app, lifespan

            async def run():
                async with lifespan(app):
                    pass

            # Must not raise.
            asyncio.run(run())

    def test_collections_failure_still_prevents_startup(self):
        """Unlike the acronym warm-up, ensure_collections_exist() failing
        should still abort startup — Qdrant is a hard dependency for search,
        with no fallback path."""
        with patch(
            "app.main.ensure_collections_exist",
            AsyncMock(side_effect=RuntimeError("Qdrant unreachable")),
        ), patch("app.main.warm_acronym_cache", AsyncMock(return_value=0)):
            from app.main import app, lifespan

            async def run():
                async with lifespan(app):
                    pass

            with pytest.raises(RuntimeError, match="Qdrant unreachable"):
                asyncio.run(run())
