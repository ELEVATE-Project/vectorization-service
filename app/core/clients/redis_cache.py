import redis
from typing import Optional, Dict, Any
import json
from datetime import datetime
import hashlib
from app.config import settings
import logging

logger = logging.getLogger(__name__)

class RedisLRUCache:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        max_size: int = 1000,
        ttl: int = 3600,  # 1 hour default TTL
        cache_enabled: bool = True
    ):
        """
        Initialize Redis LRU Cache

        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            max_size: Maximum number of items in cache
            ttl: Time to live for cache items in seconds
        """
        self.cache_enabled = cache_enabled
        if self.cache_enabled:
            self.redis_client = redis.Redis(
                host=host, port=port, db=db, decode_responses=True
            )
            self.max_size = max_size
            self.ttl = ttl
            self.access_list_key = "lru:access_list"

    def _generate_key(self, query: str) -> str:
        """Generate a unique key for the query"""
        # NOSONAR - MD5 used for cache key generation (non-cryptographic), not security
        return f"query:{hashlib.md5(query.encode()).hexdigest()}"  # NOSONAR

    def _update_access_time(self, key: str):
        """Update access time for LRU implementation"""
        if not self.cache_enabled:
            return
        current_time = datetime.now().timestamp()
        self.redis_client.zadd(self.access_list_key, {key: current_time})

        # Check if we need to remove old entries
        cache_size = self.redis_client.zcard(self.access_list_key)
        if cache_size > self.max_size:
            # Get oldest entries to remove
            oldest_entries = self.redis_client.zrange(
                self.access_list_key, 0, cache_size - self.max_size - 1
            )
            if oldest_entries:
                # Remove from sorted set and delete the actual cache entries
                self.redis_client.zrem(self.access_list_key, *oldest_entries)
                self.redis_client.delete(*oldest_entries)

    def get(self, query: str) -> Optional[Dict[str, Any]]:
        """Get cached response for a query"""
        if not self.cache_enabled:
            return None

        key = self._generate_key(query)
        cached_data = self.redis_client.get(key)

        if cached_data:
            self._update_access_time(key)
            return json.loads(cached_data)
        return None

    def set(self, query: str, response: Dict[str, Any]):
        """Cache a query response"""
        if not self.cache_enabled:
            return

        key = self._generate_key(query)
        self.redis_client.setex(key, self.ttl, json.dumps(response))
        self._update_access_time(key)

    def clear(self):
        """Clear all cached data"""
        if not self.cache_enabled:
            return

        for key in self.redis_client.scan_iter("query:*"):
            self.redis_client.delete(key)
        self.redis_client.delete(self.access_list_key)

    def remove(self, query: str):
        """Remove specific query from cache"""
        if not self.cache_enabled:
            return

        key = self._generate_key(query)
        self.redis_client.delete(key)
        self.redis_client.zrem(self.access_list_key, key)

# Initialize the cache instance
redis_cache = RedisLRUCache(
    host=settings.REDIS_HOST,
    port=settings.REDIS_PORT,
    max_size=settings.REDIS_MAX_CACHE_SIZE,
    ttl=settings.REDIS_CACHE_TTL,
    cache_enabled=settings.REDIS_CACHE_ENABLED,
)