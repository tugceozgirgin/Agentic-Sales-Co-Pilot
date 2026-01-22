"""
Caching module for tool results to reduce latency.
Supports TTL-based expiration and thread-safe operations.
"""

import hashlib
import time
import threading
from typing import Dict, Any, Tuple, Optional
from functools import wraps


class ToolCache:
    """Thread-safe cache with TTL for tool results"""
    
    def __init__(self, ttl_seconds: int = 300, max_size: int = 1000):
        self._cache: Dict[str, Tuple[Any, float]] = {}
        self.ttl = ttl_seconds
        self.max_size = max_size
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
    
    def _hash_key(self, tool_name: str, args: str) -> str:
        return hashlib.md5(f"{tool_name}:{args}".encode()).hexdigest()
    
    def get(self, tool_name: str, args: str) -> Optional[Any]:
        key = self._hash_key(tool_name, args)
        with self._lock:
            if key in self._cache:
                value, timestamp = self._cache[key]
                if time.time() - timestamp < self.ttl:
                    self._hits += 1
                    return value
                del self._cache[key]
            self._misses += 1
            return None
    
    def set(self, tool_name: str, args: str, value: Any):
        key = self._hash_key(tool_name, args)
        with self._lock:
            if len(self._cache) >= self.max_size:
                oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][1])
                del self._cache[oldest_key]
            self._cache[key] = (value, time.time())
    
    def clear(self):
        with self._lock:
            self._cache.clear()
    
    def stats(self) -> Dict[str, Any]:
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0
            return {
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": f"{hit_rate:.2%}",
                "size": len(self._cache)
            }


structured_db_cache = ToolCache(ttl_seconds=300)
semantic_db_cache = ToolCache(ttl_seconds=600)


def cached_tool(cache: ToolCache, tool_name: str):
    """Decorator to add caching to tool functions"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = str(args) + str(sorted(kwargs.items()))
            
            cached_result = cache.get(tool_name, cache_key)
            if cached_result is not None:
                print(f"[CACHE HIT] {tool_name}")
                return cached_result

            print(f"[CACHE MISS] {tool_name}")
            result = func(*args, **kwargs)
            cache.set(tool_name, cache_key, result)
            return result
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            cache_key = str(args) + str(sorted(kwargs.items()))
            cached_result = cache.get(tool_name, cache_key)
            if cached_result is not None:
                print(f"[CACHE HIT] {tool_name}")
                return cached_result
            
            print(f"[CACHE MISS] {tool_name}")
            import asyncio
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            cache.set(tool_name, cache_key, result)
            return result
        
        wrapper.async_version = async_wrapper
        return wrapper
    return decorator
