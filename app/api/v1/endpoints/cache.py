from fastapi import APIRouter, HTTPException
from app.core.clients.redis_cache import redis_cache
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.delete("/redis")
async def clear_redis_cache():
    """Clear the Redis cache"""
    try:
        redis_cache.clear()
        return {"message": "Redis cache cleared successfully"}
    except Exception as e:
        logger.error(f"Failed to clear Redis cache: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
