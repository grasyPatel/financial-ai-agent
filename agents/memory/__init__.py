# Memory System Package Initialization
# This package provides comprehensive memory management for the Deep Finance Research Chatbot
# Includes Redis-based short-term memory and vector-based long-term memory

from .memory_pipeline import MemoryPipeline, ConversationContext
from .redis_store import RedisMemoryStore, LangGraphRedisCheckpointer
from .vector_store import VectorMemoryStore, MemoryRecord

__all__ = [
    'MemoryPipeline',
    'ConversationContext', 
    'RedisMemoryStore',
    'LangGraphRedisCheckpointer',
    'VectorMemoryStore',
    'MemoryRecord'
]

__version__ = '1.0.0'