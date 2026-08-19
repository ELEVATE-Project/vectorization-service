import redis
from redis.backoff import NoBackoff
from redis.retry import Retry

from app.config import settings

# Shared, always-on Redis connection for cache-aside style lookups (see cache_client.py).
# Independent of app.core.clients.redis_cache.redis_cache, which is the query-result LRU
# cache gated behind REDIS_CACHE_ENABLED (hardcoded False in config.py today) — that flag
# only concerns that specific feature, not Redis availability in general.
redis_client = redis.Redis(
    host=settings.REDIS_HOST,
    port=settings.REDIS_PORT,
    password=settings.REDIS_PASSWORD or None,
    db=0,
    decode_responses=True,
    # Without these, redis-py defaults to no timeout at all — a blackholed
    # connection would hang every read/write (and everything downstream on the
    # synchronous search path) until the OS-level TCP timeout, not the
    # exception acronym_service.get_expansion()'s try/except is built to catch.
    socket_connect_timeout=settings.REDIS_SOCKET_CONNECT_TIMEOUT,
    socket_timeout=settings.REDIS_SOCKET_TIMEOUT,
    # redis-py's own default retry policy is 10 attempts with exponential
    # backoff on ConnectionError/TimeoutError — confirmed empirically that
    # combined with the timeouts above, a single blackholed lookup could
    # still take the better part of a minute before finally raising. That
    # defeats the point: acronym_service.get_expansion() already has its
    # own except/fallback-to-Postgres logic, so the client itself should
    # fail on the FIRST timeout and let that fallback run, not retry
    # internally for many seconds first. (retry_on_timeout/retry_on_error
    # are deprecated in this redis-py version — zero retries here already
    # covers both ConnectionError and TimeoutError.)
    retry=Retry(NoBackoff(), 0),
)
