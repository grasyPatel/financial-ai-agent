# Redis-based Memory Store for LangGraph Checkpointing
# Implements short-term memory and streaming fan-out for the Deep Finance Research Chatbot

import json
import pickle
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import redis
import asyncio
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver, Checkpoint
import logging

logger = logging.getLogger(__name__)

class RedisMemoryStore:
    """Redis-based memory store for LangGraph checkpoints and streaming state"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379", db: int = 0):
        """Initialize Redis connection for memory management"""
        self.redis_client = redis.from_url(redis_url, db=db, decode_responses=False)
        self.async_redis = redis.asyncio.from_url(redis_url, db=db, decode_responses=False)
        logger.info(f"✅ Redis Memory Store initialized: {redis_url}")
    
    # ==========================================
    # SHORT-TERM MEMORY (Thread-scoped)
    # ==========================================
    
    async def store_thread_memory(self, thread_id: str, session_id: str, data: Dict[str, Any], ttl: int = 3600):
        """Store thread-scoped working memory with TTL"""
        try:
            key = f"thread_memory:{thread_id}:{session_id}"
            serialized_data = json.dumps(data, default=str)
            await self.async_redis.setex(key, ttl, serialized_data)
            logger.debug(f"Stored thread memory: {key}")
        except Exception as e:
            logger.error(f"Failed to store thread memory: {e}")
    
    async def get_thread_memory(self, thread_id: str, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve thread-scoped working memory"""
        try:
            key = f"thread_memory:{thread_id}:{session_id}"
            data = await self.async_redis.get(key)
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            logger.error(f"Failed to get thread memory: {e}")
            return None
    
    # ==========================================
    # STREAMING STATE MANAGEMENT
    # ==========================================
    
    async def set_streaming_state(self, session_id: str, state: Dict[str, Any], ttl: int = 300):
        """Store streaming state for real-time updates"""
        try:
            key = f"streaming:{session_id}"
            serialized_state = json.dumps(state, default=str)
            await self.async_redis.setex(key, ttl, serialized_state)
            
            # Publish state update for real-time subscriptions
            await self.async_redis.publish(f"stream_updates:{session_id}", serialized_state)
            logger.debug(f"Updated streaming state: {session_id}")
        except Exception as e:
            logger.error(f"Failed to set streaming state: {e}")
    
    async def get_streaming_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get current streaming state"""
        try:
            key = f"streaming:{session_id}"
            data = await self.async_redis.get(key)
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            logger.error(f"Failed to get streaming state: {e}")
            return None
    
    # ==========================================
    # RESEARCH PIPELINE CHECKPOINTS
    # ==========================================
    
    async def save_research_checkpoint(self, checkpoint_id: str, step: str, data: Dict[str, Any], ttl: int = 1800):
        """Save research pipeline checkpoint for resumability"""
        try:
            key = f"research_checkpoint:{checkpoint_id}:{step}"
            checkpoint_data = {
                "step": step,
                "data": data,
                "timestamp": datetime.utcnow().isoformat(),
                "status": "completed"
            }
            serialized = json.dumps(checkpoint_data, default=str)
            await self.async_redis.setex(key, ttl, serialized)
            
            # Update checkpoint progress
            progress_key = f"research_progress:{checkpoint_id}"
            await self.async_redis.hset(progress_key, step, "completed")
            await self.async_redis.expire(progress_key, ttl)
            
            logger.debug(f"Saved research checkpoint: {checkpoint_id}:{step}")
        except Exception as e:
            logger.error(f"Failed to save research checkpoint: {e}")
    
    async def get_research_checkpoint(self, checkpoint_id: str, step: str) -> Optional[Dict[str, Any]]:
        """Retrieve research pipeline checkpoint"""
        try:
            key = f"research_checkpoint:{checkpoint_id}:{step}"
            data = await self.async_redis.get(key)
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            logger.error(f"Failed to get research checkpoint: {e}")
            return None
    
    async def get_research_progress(self, checkpoint_id: str) -> Dict[str, str]:
        """Get overall research pipeline progress"""
        try:
            progress_key = f"research_progress:{checkpoint_id}"
            progress = await self.async_redis.hgetall(progress_key)
            return {k.decode(): v.decode() for k, v in progress.items()} if progress else {}
        except Exception as e:
            logger.error(f"Failed to get research progress: {e}")
            return {}
    
    # ==========================================
    # SOURCE DEDUPLICATION CACHE
    # ==========================================
    
    async def cache_source(self, url: str, source_data: Dict[str, Any], ttl: int = 86400):
        """Cache web source data for deduplication"""
        try:
            key = f"source_cache:{hash(url)}"
            source_data['cached_at'] = datetime.utcnow().isoformat()
            serialized = json.dumps(source_data, default=str)
            await self.async_redis.setex(key, ttl, serialized)
            logger.debug(f"Cached source: {url}")
        except Exception as e:
            logger.error(f"Failed to cache source: {e}")
    
    async def get_cached_source(self, url: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached source data"""
        try:
            key = f"source_cache:{hash(url)}"
            data = await self.async_redis.get(key)
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            logger.error(f"Failed to get cached source: {e}")
            return None
    
    # ==========================================
    # SESSION MANAGEMENT
    # ==========================================
    
    async def create_research_session(self, session_id: str, config: Dict[str, Any], ttl: int = 7200):
        """Create a new research session with configuration"""
        try:
            key = f"research_session:{session_id}"
            session_data = {
                "session_id": session_id,
                "config": config,
                "created_at": datetime.utcnow().isoformat(),
                "status": "active",
                "steps_completed": [],
                "sources_found": [],
                "analysis_results": {}
            }
            serialized = json.dumps(session_data, default=str)
            await self.async_redis.setex(key, ttl, serialized)
            logger.info(f"Created research session: {session_id}")
        except Exception as e:
            logger.error(f"Failed to create research session: {e}")
    
    async def update_research_session(self, session_id: str, updates: Dict[str, Any]):
        """Update research session with new data"""
        try:
            key = f"research_session:{session_id}"
            current_data = await self.async_redis.get(key)
            
            if current_data:
                session_data = json.loads(current_data)
                session_data.update(updates)
                session_data['updated_at'] = datetime.utcnow().isoformat()
                
                serialized = json.dumps(session_data, default=str)
                ttl = await self.async_redis.ttl(key)
                await self.async_redis.setex(key, max(ttl, 3600), serialized)
                logger.debug(f"Updated research session: {session_id}")
        except Exception as e:
            logger.error(f"Failed to update research session: {e}")
    
    async def get_research_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get research session data"""
        try:
            key = f"research_session:{session_id}"
            data = await self.async_redis.get(key)
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            logger.error(f"Failed to get research session: {e}")
            return None
    
    # ==========================================
    # METRICS & MONITORING
    # ==========================================
    
    async def increment_counter(self, counter_name: str, increment: int = 1) -> int:
        """Increment performance counter"""
        try:
            key = f"metrics:counter:{counter_name}"
            result = await self.async_redis.incrby(key, increment)
            await self.async_redis.expire(key, 86400)  # 24 hour TTL
            return result
        except Exception as e:
            logger.error(f"Failed to increment counter: {e}")
            return 0
    
    async def record_latency(self, operation: str, latency_ms: float):
        """Record operation latency for monitoring"""
        try:
            key = f"metrics:latency:{operation}"
            timestamp = datetime.utcnow().timestamp()
            await self.async_redis.zadd(key, {str(latency_ms): timestamp})
            
            # Keep only last 1000 measurements
            await self.async_redis.zremrangebyrank(key, 0, -1001)
            logger.debug(f"Recorded latency for {operation}: {latency_ms}ms")
        except Exception as e:
            logger.error(f"Failed to record latency: {e}")
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics"""
        try:
            metrics = {}
            
            # Get counters
            counter_keys = await self.async_redis.keys("metrics:counter:*")
            for key in counter_keys:
                counter_name = key.decode().split(":")[-1]
                value = await self.async_redis.get(key)
                metrics[f"counter_{counter_name}"] = int(value) if value else 0
            
            # Get average latencies
            latency_keys = await self.async_redis.keys("metrics:latency:*")
            for key in latency_keys:
                operation = key.decode().split(":")[-1]
                latencies = await self.async_redis.zrange(key, 0, -1)
                if latencies:
                    avg_latency = sum(float(l) for l in latencies) / len(latencies)
                    metrics[f"avg_latency_{operation}"] = round(avg_latency, 2)
            
            return metrics
        except Exception as e:
            logger.error(f"Failed to get performance metrics: {e}")
            return {}
    
    # ==========================================
    # CLEANUP & MAINTENANCE
    # ==========================================
    
    async def cleanup_expired_sessions(self):
        """Clean up expired research sessions and checkpoints"""
        try:
            # Find expired session keys
            session_keys = await self.async_redis.keys("research_session:*")
            expired_count = 0
            
            for key in session_keys:
                ttl = await self.async_redis.ttl(key)
                if ttl == -1 or ttl < 300:  # No TTL or expiring soon
                    await self.async_redis.delete(key)
                    expired_count += 1
            
            logger.info(f"Cleaned up {expired_count} expired sessions")
            return expired_count
        except Exception as e:
            logger.error(f"Failed to cleanup expired sessions: {e}")
            return 0
    
    async def get_memory_usage(self) -> Dict[str, Any]:
        """Get Redis memory usage statistics"""
        try:
            info = await self.async_redis.info('memory')
            return {
                "used_memory": info.get('used_memory', 0),
                "used_memory_human": info.get('used_memory_human', '0B'),
                "used_memory_peak": info.get('used_memory_peak', 0),
                "memory_usage_percentage": info.get('used_memory_rss', 0) / info.get('total_system_memory', 1) * 100
            }
        except Exception as e:
            logger.error(f"Failed to get memory usage: {e}")
            return {}
    
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check for Redis memory store"""
        try:
            # Test basic connectivity
            await self.async_redis.ping()
            
            # Get system info
            memory_info = await self.get_memory_usage()
            metrics = await self.get_performance_metrics()
            
            # Count active sessions
            session_count = len(await self.async_redis.keys("research_session:*"))
            checkpoint_count = len(await self.async_redis.keys("research_checkpoint:*"))
            
            return {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "redis_connected": True,
                "memory_info": memory_info,
                "performance_metrics": metrics,
                "active_sessions": session_count,
                "active_checkpoints": checkpoint_count
            }
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
                "redis_connected": False
            }


class LangGraphRedisCheckpointer(BaseCheckpointSaver):
    """LangGraph-compatible Redis checkpointer for workflow state persistence"""
    
    def __init__(self, redis_store: RedisMemoryStore):
        self.redis_store = redis_store
        super().__init__()
    
    def put(self, config: RunnableConfig, checkpoint: Checkpoint) -> RunnableConfig:
        """Save checkpoint to Redis"""
        try:
            thread_id = config.get("configurable", {}).get("thread_id", "default")
            checkpoint_id = f"{thread_id}:{checkpoint.get('ts', datetime.utcnow().timestamp())}"
            
            # Serialize checkpoint data
            checkpoint_data = {
                "checkpoint": checkpoint,
                "config": config,
                "saved_at": datetime.utcnow().isoformat()
            }
            
            # Save to Redis with 1 hour TTL
            key = f"langgraph_checkpoint:{checkpoint_id}"
            serialized = pickle.dumps(checkpoint_data)
            self.redis_store.redis_client.setex(key, 3600, serialized)
            
            logger.debug(f"Saved LangGraph checkpoint: {checkpoint_id}")
            return config
        except Exception as e:
            logger.error(f"Failed to save LangGraph checkpoint: {e}")
            return config
    
    def get(self, config: RunnableConfig) -> Optional[Checkpoint]:
        """Retrieve checkpoint from Redis"""
        try:
            thread_id = config.get("configurable", {}).get("thread_id", "default")
            
            # Find latest checkpoint for this thread
            pattern = f"langgraph_checkpoint:{thread_id}:*"
            keys = self.redis_store.redis_client.keys(pattern)
            
            if not keys:
                return None
            
            # Get the most recent checkpoint
            latest_key = max(keys, key=lambda k: k.decode().split(":")[-1])
            data = self.redis_store.redis_client.get(latest_key)
            
            if data:
                checkpoint_data = pickle.loads(data)
                return checkpoint_data["checkpoint"]
            
            return None
        except Exception as e:
            logger.error(f"Failed to get LangGraph checkpoint: {e}")
            return None
    
    def list(self, config: RunnableConfig) -> List[Checkpoint]:
        """List all checkpoints for a thread"""
        try:
            thread_id = config.get("configurable", {}).get("thread_id", "default")
            pattern = f"langgraph_checkpoint:{thread_id}:*"
            keys = self.redis_store.redis_client.keys(pattern)
            
            checkpoints = []
            for key in keys:
                data = self.redis_store.redis_client.get(key)
                if data:
                    checkpoint_data = pickle.loads(data)
                    checkpoints.append(checkpoint_data["checkpoint"])
            
            # Sort by timestamp
            checkpoints.sort(key=lambda c: c.get('ts', 0), reverse=True)
            return checkpoints
        except Exception as e:
            logger.error(f"Failed to list LangGraph checkpoints: {e}")
            return []


# Export main classes
__all__ = ['RedisMemoryStore', 'LangGraphRedisCheckpointer']