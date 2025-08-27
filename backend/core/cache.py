"""
Redis cache management
"""

import json
import logging
from typing import Any, Optional, Union, List, Dict
from datetime import datetime, timedelta

import redis.asyncio as redis
from redis.exceptions import RedisError

from .config import settings

logger = logging.getLogger(__name__)

# Global Redis client
redis_client: Optional[redis.Redis] = None


async def init_redis():
    """Initialize Redis connection"""
    global redis_client
    
    try:
        redis_client = redis.from_url(
            settings.REDIS_URL,
            max_connections=settings.REDIS_MAX_CONNECTIONS,
            decode_responses=True,
            encoding="utf-8"
        )
        
        # Test connection
        await redis_client.ping()
        logger.info("Redis connection initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize Redis: {e}")
        raise


async def close_redis():
    """Close Redis connection"""
    global redis_client
    
    if redis_client:
        await redis_client.close()
        logger.info("Redis connection closed")


class CacheManager:
    """Redis cache operations manager"""
    
    def __init__(self):
        if not redis_client:
            raise RuntimeError("Redis not initialized")
        self.client = redis_client
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            value = await self.client.get(key)
            if value:
                return json.loads(value)
            return None
        except (RedisError, json.JSONDecodeError) as e:
            logger.error(f"Cache get error for key {key}: {e}")
            return None
    
    async def set(
        self, 
        key: str, 
        value: Any, 
        expire: Optional[int] = None
    ) -> bool:
        """Set value in cache"""
        try:
            serialized_value = json.dumps(
                value, 
                default=self._json_serializer,
                ensure_ascii=False
            )
            
            if expire:
                return await self.client.setex(key, expire, serialized_value)
            else:
                return await self.client.set(key, serialized_value)
                
        except (RedisError, json.JSONEncodeError) as e:
            logger.error(f"Cache set error for key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        try:
            result = await self.client.delete(key)
            return result > 0
        except RedisError as e:
            logger.error(f"Cache delete error for key {key}: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists"""
        try:
            return bool(await self.client.exists(key))
        except RedisError as e:
            logger.error(f"Cache exists error for key {key}: {e}")
            return False
    
    async def expire(self, key: str, seconds: int) -> bool:
        """Set expiration for key"""
        try:
            return await self.client.expire(key, seconds)
        except RedisError as e:
            logger.error(f"Cache expire error for key {key}: {e}")
            return False
    
    async def ttl(self, key: str) -> int:
        """Get time to live for key"""
        try:
            return await self.client.ttl(key)
        except RedisError as e:
            logger.error(f"Cache TTL error for key {key}: {e}")
            return -1
    
    async def increment(self, key: str, amount: int = 1) -> Optional[int]:
        """Increment counter"""
        try:
            return await self.client.incrby(key, amount)
        except RedisError as e:
            logger.error(f"Cache increment error for key {key}: {e}")
            return None
    
    async def decrement(self, key: str, amount: int = 1) -> Optional[int]:
        """Decrement counter"""
        try:
            return await self.client.decrby(key, amount)
        except RedisError as e:
            logger.error(f"Cache decrement error for key {key}: {e}")
            return None
    
    # Hash operations
    async def hget(self, name: str, key: str) -> Optional[Any]:
        """Get field from hash"""
        try:
            value = await self.client.hget(name, key)
            if value:
                return json.loads(value)
            return None
        except (RedisError, json.JSONDecodeError) as e:
            logger.error(f"Hash get error for {name}:{key}: {e}")
            return None
    
    async def hset(self, name: str, key: str, value: Any) -> bool:
        """Set field in hash"""
        try:
            serialized_value = json.dumps(
                value, 
                default=self._json_serializer,
                ensure_ascii=False
            )
            result = await self.client.hset(name, key, serialized_value)
            return result > 0
        except (RedisError, json.JSONEncodeError) as e:
            logger.error(f"Hash set error for {name}:{key}: {e}")
            return False
    
    async def hgetall(self, name: str) -> Dict[str, Any]:
        """Get all fields from hash"""
        try:
            data = await self.client.hgetall(name)
            return {
                k: json.loads(v) for k, v in data.items()
            }
        except (RedisError, json.JSONDecodeError) as e:
            logger.error(f"Hash getall error for {name}: {e}")
            return {}
    
    async def hdel(self, name: str, key: str) -> bool:
        """Delete field from hash"""
        try:
            result = await self.client.hdel(name, key)
            return result > 0
        except RedisError as e:
            logger.error(f"Hash delete error for {name}:{key}: {e}")
            return False
    
    # List operations
    async def lpush(self, key: str, *values) -> Optional[int]:
        """Left push to list"""
        try:
            serialized_values = [
                json.dumps(v, default=self._json_serializer, ensure_ascii=False)
                for v in values
            ]
            return await self.client.lpush(key, *serialized_values)
        except (RedisError, json.JSONEncodeError) as e:
            logger.error(f"List lpush error for key {key}: {e}")
            return None
    
    async def rpush(self, key: str, *values) -> Optional[int]:
        """Right push to list"""
        try:
            serialized_values = [
                json.dumps(v, default=self._json_serializer, ensure_ascii=False)
                for v in values
            ]
            return await self.client.rpush(key, *serialized_values)
        except (RedisError, json.JSONEncodeError) as e:
            logger.error(f"List rpush error for key {key}: {e}")
            return None
    
    async def lpop(self, key: str) -> Optional[Any]:
        """Left pop from list"""
        try:
            value = await self.client.lpop(key)
            if value:
                return json.loads(value)
            return None
        except (RedisError, json.JSONDecodeError) as e:
            logger.error(f"List lpop error for key {key}: {e}")
            return None
    
    async def rpop(self, key: str) -> Optional[Any]:
        """Right pop from list"""
        try:
            value = await self.client.rpop(key)
            if value:
                return json.loads(value)
            return None
        except (RedisError, json.JSONDecodeError) as e:
            logger.error(f"List rpop error for key {key}: {e}")
            return None
    
    async def lrange(self, key: str, start: int = 0, end: int = -1) -> List[Any]:
        """Get range from list"""
        try:
            values = await self.client.lrange(key, start, end)
            return [json.loads(v) for v in values]
        except (RedisError, json.JSONDecodeError) as e:
            logger.error(f"List range error for key {key}: {e}")
            return []
    
    async def llen(self, key: str) -> int:
        """Get list length"""
        try:
            return await self.client.llen(key)
        except RedisError as e:
            logger.error(f"List length error for key {key}: {e}")
            return 0
    
    # Set operations
    async def sadd(self, key: str, *values) -> Optional[int]:
        """Add to set"""
        try:
            serialized_values = [
                json.dumps(v, default=self._json_serializer, ensure_ascii=False)
                for v in values
            ]
            return await self.client.sadd(key, *serialized_values)
        except (RedisError, json.JSONEncodeError) as e:
            logger.error(f"Set add error for key {key}: {e}")
            return None
    
    async def smembers(self, key: str) -> List[Any]:
        """Get all set members"""
        try:
            values = await self.client.smembers(key)
            return [json.loads(v) for v in values]
        except (RedisError, json.JSONDecodeError) as e:
            logger.error(f"Set members error for key {key}: {e}")
            return []
    
    async def sismember(self, key: str, value: Any) -> bool:
        """Check if value is in set"""
        try:
            serialized_value = json.dumps(
                value, 
                default=self._json_serializer,
                ensure_ascii=False
            )
            return await self.client.sismember(key, serialized_value)
        except (RedisError, json.JSONEncodeError) as e:
            logger.error(f"Set ismember error for key {key}: {e}")
            return False
    
    async def srem(self, key: str, *values) -> Optional[int]:
        """Remove from set"""
        try:
            serialized_values = [
                json.dumps(v, default=self._json_serializer, ensure_ascii=False)
                for v in values
            ]
            return await self.client.srem(key, *serialized_values)
        except (RedisError, json.JSONEncodeError) as e:
            logger.error(f"Set remove error for key {key}: {e}")
            return None
    
    # Pub/Sub operations
    async def publish(self, channel: str, message: Any) -> Optional[int]:
        """Publish message to channel"""
        try:
            serialized_message = json.dumps(
                message,
                default=self._json_serializer,
                ensure_ascii=False
            )
            return await self.client.publish(channel, serialized_message)
        except (RedisError, json.JSONEncodeError) as e:
            logger.error(f"Publish error for channel {channel}: {e}")
            return None
    
    # Utility methods
    async def flush_all(self) -> bool:
        """Flush all keys (use with caution)"""
        try:
            await self.client.flushall()
            return True
        except RedisError as e:
            logger.error(f"Flush all error: {e}")
            return False
    
    async def get_info(self) -> Dict[str, Any]:
        """Get Redis server info"""
        try:
            return await self.client.info()
        except RedisError as e:
            logger.error(f"Info error: {e}")
            return {}
    
    async def keys_pattern(self, pattern: str) -> List[str]:
        """Get keys matching pattern"""
        try:
            return await self.client.keys(pattern)
        except RedisError as e:
            logger.error(f"Keys pattern error: {e}")
            return []
    
    @staticmethod
    def _json_serializer(obj):
        """JSON serializer for datetime and other objects"""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        else:
            return str(obj)


# Cache key generators
class CacheKeys:
    """Cache key generators"""
    
    @staticmethod
    def site_health(site_id: str) -> str:
        return f"site:health:{site_id}"
    
    @staticmethod
    def site_metrics(site_id: str) -> str:
        return f"site:metrics:{site_id}"
    
    @staticmethod
    def kpi_values(site_id: str = None) -> str:
        if site_id:
            return f"kpi:values:{site_id}"
        return "kpi:values:all"
    
    @staticmethod
    def active_events(site_id: str = None) -> str:
        if site_id:
            return f"events:active:{site_id}"
        return "events:active:all"
    
    @staticmethod
    def user_session(user_id: str) -> str:
        return f"user:session:{user_id}"
    
    @staticmethod
    def rate_limit(identifier: str) -> str:
        return f"rate_limit:{identifier}"
    
    @staticmethod
    def correlation_results(correlation_id: str) -> str:
        return f"correlation:{correlation_id}"
    
    @staticmethod
    def aiops_prediction(site_id: str, model_name: str) -> str:
        return f"aiops:prediction:{site_id}:{model_name}"
    
    @staticmethod
    def report_cache(report_type: str, params_hash: str) -> str:
        return f"report:{report_type}:{params_hash}"


# Dependency for FastAPI
async def get_cache_manager() -> CacheManager:
    """Get cache manager instance"""
    return CacheManager()