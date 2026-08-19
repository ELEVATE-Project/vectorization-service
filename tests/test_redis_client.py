"""Regression test for the shared Redis client's connection timeouts.

Devin-flagged (app/core/clients/redis_client.py:9): redis-py defaults both
socket_timeout and socket_connect_timeout to None (no timeout at all) — a
blackholed connection (packets silently dropped, unlike a clean "connection
refused") would hang every cache read/write until the OS-level TCP timeout,
which can be minutes. This runs synchronously on the search request path, so
an unresponsive cache would stall search entirely instead of degrading to
Postgres per acronym_service.get_expansion()'s existing except/fallback.

Also covers a gap found while verifying that fix live: redis-py's own
DEFAULT retry policy is 10 attempts with exponential backoff on
ConnectionError/TimeoutError — timeouts alone aren't enough, a blackholed
lookup would still retry internally for the better part of a minute before
finally raising. Confirmed with a real local "black hole" TCP server (accepts
the connection, never responds): with only socket_timeout set, the client
hung past 20s; with retries disabled too, it failed in exactly the
configured timeout (2.00s).

Pure inspection of the module-level client's configured connection kwargs —
no live Redis connection required for these; the black-hole timing proof
above was done manually, not as an automated test (a real TCP server +
thread is heavier than warranted for a unit test — the retry/timeout kwargs
inspected here are what actually controls that behavior).
"""
from redis.backoff import NoBackoff

from app.config import settings
from app.core.clients.redis_client import redis_client


def test_socket_timeout_is_set_not_none():
    kwargs = redis_client.connection_pool.connection_kwargs
    assert kwargs.get("socket_timeout") is not None
    assert kwargs["socket_timeout"] == settings.REDIS_SOCKET_TIMEOUT


def test_socket_connect_timeout_is_set_not_none():
    kwargs = redis_client.connection_pool.connection_kwargs
    assert kwargs.get("socket_connect_timeout") is not None
    assert kwargs["socket_connect_timeout"] == settings.REDIS_SOCKET_CONNECT_TIMEOUT


def test_retries_disabled_so_a_single_timeout_fails_fast():
    kwargs = redis_client.connection_pool.connection_kwargs
    retry = kwargs.get("retry")
    assert retry is not None
    assert retry._retries == 0, "a nonzero retry count defeats fail-fast — confirmed live it compounds past a minute"
    assert isinstance(retry._backoff, NoBackoff)
