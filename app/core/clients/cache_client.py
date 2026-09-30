from typing import Dict, List, Optional, Tuple

from app.core.clients.redis_client import redis_client


def get(key: str) -> Optional[str]:
    return redis_client.get(key)


def set(key: str, value: str, ttl: int) -> None:
    redis_client.setex(key, ttl, value)


def delete_many(keys: List[str]) -> None:
    """Batched delete: DEL takes multiple keys natively, so this is one
    round-trip for N keys instead of N. Empty list is a no-op."""
    if not keys:
        return
    redis_client.delete(*keys)


def get_many(keys: List[str]) -> List[Optional[str]]:
    """Batched read: one round-trip for N keys instead of N. Empty list is a
    no-op — redis-py's MGET requires at least one key argument."""
    if not keys:
        return []
    return redis_client.mget(keys)


def set_many(items: Dict[str, Tuple[str, int]]) -> None:
    """Batched write: items is {key: (value, ttl)}. Still N SETEX commands
    server-side (each key needs its own TTL, so there's no single Redis
    command for this), but pipelining sends them as one round-trip instead
    of N sequential ones."""
    if not items:
        return
    pipe = redis_client.pipeline()
    for key, (value, ttl) in items.items():
        pipe.setex(key, ttl, value)
    pipe.execute()


def set_many_if_absent(items: Dict[str, Tuple[str, int]]) -> None:
    """Like set_many, but each key is written ONLY if it does not already
    exist (Redis SET ... NX). Existing keys are left untouched.

    This is the safe way to write a value you computed from possibly-stale
    information: the "does it exist" check and the write happen together
    inside Redis, so a value someone else wrote between your read and your
    write survives instead of being overwritten. Used for negative
    ("not an acronym") entries, where a blind overwrite could bury a real
    acronym an upload had just cached and hide it for the whole negative TTL.
    """
    if not items:
        return
    pipe = redis_client.pipeline()
    for key, (value, ttl) in items.items():
        pipe.set(key, value, ex=ttl, nx=True)
    pipe.execute()
