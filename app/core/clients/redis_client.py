import redis
from redis.backoff import NoBackoff
from redis.retry import Retry

from app.config import settings

# Redis connection for the acronym cache (used by cache_client.py). Not related to
# redis_cache.py, the query-result cache behind REDIS_CACHE_ENABLED.
redis_client = redis.Redis(
    host=settings.REDIS_HOST,
    port=settings.REDIS_PORT,
    password=settings.REDIS_PASSWORD or None,
    db=settings.REDIS_DB,
    decode_responses=True,
    # redis-py has no timeout by default, so a Redis that does not respond would
    # hang the search request.
    socket_connect_timeout=settings.REDIS_SOCKET_CONNECT_TIMEOUT,
    socket_timeout=settings.REDIS_SOCKET_TIMEOUT,
    # No retries: on an error the acronym lookup falls back to Postgres at once;
    # retries would only delay that.
    retry=Retry(NoBackoff(), 0),
)
