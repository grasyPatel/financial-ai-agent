# Memory Pipeline - Unified memory management for the Deep Finance Research Chatbot
# Combines short-term Redis memory with long-term vector storage

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import json

from .redis_store import RedisMemoryStore
from .vector_store import VectorMemoryStore, MemoryRecord

logger = logging.getLogger(__name__)

@dataclass
class ConversationContext:
    """Structured conversation context for memory operations"""
    session_id: str
    thread_id: str
    user_id: str
    query: str
    intent: str
    entities: Dict[str, Any]
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


class MemoryPipeline:
    """
    Unified memory management system that coordinates:
    1. Short-term memory (Redis) - Recent conversation, working memory, checkpoints
    2. Long-term memory (Vector DB) - Historical insights, learned patterns, domain knowledge
    3. Context synthesis - Intelligent retrieval and context building
    """
    
    def __init__(self, redis_config: Dict[str, Any], vector_config: Dict[str, Any]):
        """Initialize the memory pipeline with both stores"""
        self.redis_store = RedisMemoryStore(
            redis_url=redis_config.get('url', 'redis://localhost:6379'),
            db=redis_config.get('db', 0)
        )
        self.vector_store = VectorMemoryStore(
            config=vector_config,
            embedding_model=vector_config.get('embedding_model', 'sentence-transformers/all-MiniLM-L6-v2')
        )
        
        # Memory thresholds and policies
        self.short_term_ttl = 3600  # 1 hour for working memory
        self.context_window = 100   # Max messages in context
        self.similarity_threshold = 0.75
        self.max_long_term_results = 10
        
        logger.info("✅ Memory Pipeline initialized with Redis + Vector stores")
    
    # ==========================================
    # CONVERSATION MEMORY MANAGEMENT
    # ==========================================
    
    async def process_user_input(self, context: ConversationContext) -> Dict[str, Any]:
        """
        Process new user input and update both short-term and long-term memory
        Returns enriched context with relevant memories
        """
        try:
            # 1. Store in short-term memory (Redis)
            await self._store_short_term_context(context)
            
            # 2. Search long-term memory for relevant insights
            relevant_memories = await self._retrieve_relevant_memories(context)
            
            # 3. Get recent conversation history
            recent_history = await self._get_recent_conversation(context.session_id, context.thread_id)
            
            # 4. Build enriched context
            enriched_context = {
                'current_context': asdict(context),
                'recent_history': recent_history,
                'relevant_memories': [record.to_dict() for record in relevant_memories],
                'memory_insights': await self._generate_memory_insights(context, relevant_memories),
                'context_summary': await self._generate_context_summary(context, recent_history, relevant_memories)
            }
            
            logger.debug(f"Processed user input for session {context.session_id}")
            return enriched_context
            
        except Exception as e:
            logger.error(f"Failed to process user input: {e}")
            return {'current_context': asdict(context), 'error': str(e)}
    
    async def _store_short_term_context(self, context: ConversationContext):
        """Store conversation context in Redis"""
        conversation_data = {
            'user_id': context.user_id,
            'query': context.query,
            'intent': context.intent,
            'entities': context.entities,
            'timestamp': context.timestamp.isoformat()
        }
        
        # Store in thread-scoped memory
        await self.redis_store.store_thread_memory(
            context.thread_id,
            context.session_id,
            conversation_data,
            ttl=self.short_term_ttl
        )
        
        # Update conversation history
        history_key = f"conversation_history:{context.session_id}:{context.thread_id}"
        await self.redis_store.async_redis.lpush(history_key, json.dumps(conversation_data, default=str))
        await self.redis_store.async_redis.ltrim(history_key, 0, self.context_window - 1)
        await self.redis_store.async_redis.expire(history_key, self.short_term_ttl * 2)
    
    async def _retrieve_relevant_memories(self, context: ConversationContext) -> List[MemoryRecord]:
        """Retrieve relevant memories from vector store"""
        try:
            # Build search query from context
            search_query = f"{context.query} {context.intent}"
            if context.entities:
                entity_text = " ".join(str(v) for v in context.entities.values())
                search_query += f" {entity_text}"
            
            # Search with metadata filters for user context
            metadata_filter = {
                'user_id': context.user_id,
                'domain': 'finance'  # Domain-specific filtering
            }
            
            memories = await self.vector_store.search_similar_memories(
                query=search_query,
                limit=self.max_long_term_results,
                min_similarity=self.similarity_threshold,
                metadata_filter=metadata_filter
            )
            
            return memories
            
        except Exception as e:
            logger.error(f"Failed to retrieve relevant memories: {e}")
            return []
    
    async def _get_recent_conversation(self, session_id: str, thread_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent conversation history from Redis"""
        try:
            history_key = f"conversation_history:{session_id}:{thread_id}"
            history_raw = await self.redis_store.async_redis.lrange(history_key, 0, limit - 1)
            
            history = []
            for item in history_raw:
                try:
                    conversation_item = json.loads(item)
                    history.append(conversation_item)
                except json.JSONDecodeError:
                    continue
            
            return history
            
        except Exception as e:
            logger.error(f"Failed to get recent conversation: {e}")
            return []
    
    # ==========================================
    # RESEARCH SESSION MEMORY
    # ==========================================
    
    async def create_research_session(self, session_id: str, user_id: str, config: Dict[str, Any]) -> bool:
        """Create new research session with memory context"""
        try:
            # Create session in Redis
            await self.redis_store.create_research_session(session_id, {
                'user_id': user_id,
                'config': config,
                'memory_context': True,
                'created_at': datetime.utcnow().isoformat()
            })
            
            # Initialize session context in vector memory
            session_memory = MemoryRecord(
                id=f"session_{session_id}",
                content=f"Research session started: {config.get('research_topic', 'general')}",
                metadata={
                    'type': 'session_start',
                    'user_id': user_id,
                    'session_id': session_id,
                    'domain': 'finance',
                    'research_topic': config.get('research_topic', ''),
                    'research_scope': config.get('research_scope', '')
                }
            )
            
            await self.vector_store.store_memory(
                session_memory.content,
                session_memory.metadata
            )
            
            logger.info(f"Created research session: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create research session: {e}")
            return False
    
    async def store_research_insight(self, 
                                   session_id: str, 
                                   user_id: str,
                                   insight: str, 
                                   sources: List[Dict[str, Any]], 
                                   analysis_type: str) -> str:
        """Store research insight in long-term memory"""
        try:
            # Create rich metadata for the insight
            metadata = {
                'type': 'research_insight',
                'user_id': user_id,
                'session_id': session_id,
                'domain': 'finance',
                'analysis_type': analysis_type,
                'source_count': len(sources),
                'sources': sources[:5],  # Store top 5 sources
                'insight_category': self._categorize_insight(insight),
                'confidence_score': self._calculate_confidence(insight, sources)
            }
            
            # Store in vector memory for long-term recall
            memory_id = await self.vector_store.store_memory(insight, metadata)
            
            # Also cache in Redis for immediate access
            await self.redis_store.save_research_checkpoint(
                session_id,
                f"insight_{memory_id}",
                {
                    'insight': insight,
                    'sources': sources,
                    'analysis_type': analysis_type,
                    'memory_id': memory_id
                }
            )
            
            logger.debug(f"Stored research insight: {memory_id}")
            return memory_id
            
        except Exception as e:
            logger.error(f"Failed to store research insight: {e}")
            return ""
    
    def _categorize_insight(self, insight: str) -> str:
        """Categorize insight based on content analysis"""
        insight_lower = insight.lower()
        
        if any(word in insight_lower for word in ['financial', 'revenue', 'profit', 'earnings']):
            return 'financial_performance'
        elif any(word in insight_lower for word in ['risk', 'volatility', 'uncertainty']):
            return 'risk_analysis'
        elif any(word in insight_lower for word in ['trend', 'forecast', 'prediction']):
            return 'market_trends'
        elif any(word in insight_lower for word in ['comparison', 'peer', 'competitor']):
            return 'comparative_analysis'
        else:
            return 'general_insight'
    
    def _calculate_confidence(self, insight: str, sources: List[Dict[str, Any]]) -> float:
        """Calculate confidence score based on insight quality and sources"""
        base_confidence = 0.5
        
        # Boost for multiple sources
        source_boost = min(0.3, len(sources) * 0.05)
        
        # Boost for quantitative content
        quantitative_boost = 0.1 if any(char.isdigit() for char in insight) else 0
        
        # Boost for longer, detailed insights
        length_boost = min(0.1, len(insight) / 1000)
        
        return min(1.0, base_confidence + source_boost + quantitative_boost + length_boost)
    
    # ==========================================
    # CONTEXT SYNTHESIS & INSIGHTS
    # ==========================================
    
    async def _generate_memory_insights(self, 
                                      context: ConversationContext, 
                                      memories: List[MemoryRecord]) -> Dict[str, Any]:
        """Generate insights from retrieved memories"""
        if not memories:
            return {'status': 'no_memories', 'insights': []}
        
        insights = {
            'total_memories_found': len(memories),
            'memory_types': {},
            'confidence_distribution': {'high': 0, 'medium': 0, 'low': 0},
            'temporal_distribution': {'recent': 0, 'older': 0},
            'key_patterns': []
        }
        
        now = datetime.utcnow()
        
        for memory in memories:
            # Categorize by type
            memory_type = memory.metadata.get('type', 'unknown')
            insights['memory_types'][memory_type] = insights['memory_types'].get(memory_type, 0) + 1
            
            # Confidence distribution
            if memory.relevance_score >= 0.9:
                insights['confidence_distribution']['high'] += 1
            elif memory.relevance_score >= 0.8:
                insights['confidence_distribution']['medium'] += 1
            else:
                insights['confidence_distribution']['low'] += 1
            
            # Temporal distribution
            age_hours = (now - memory.created_at).total_seconds() / 3600
            if age_hours <= 24:
                insights['temporal_distribution']['recent'] += 1
            else:
                insights['temporal_distribution']['older'] += 1
        
        # Identify key patterns
        if insights['memory_types'].get('research_insight', 0) >= 2:
            insights['key_patterns'].append('Multiple research insights available')
        
        if insights['confidence_distribution']['high'] >= 3:
            insights['key_patterns'].append('High-confidence memories dominate')
        
        return insights
    
    async def _generate_context_summary(self, 
                                      context: ConversationContext, 
                                      history: List[Dict[str, Any]], 
                                      memories: List[MemoryRecord]) -> str:
        """Generate a context summary for the AI agent"""
        summary_parts = []
        
        # Current context
        summary_parts.append(f"Current query: {context.query}")
        summary_parts.append(f"Detected intent: {context.intent}")
        
        if context.entities:
            entities_str = ", ".join(f"{k}: {v}" for k, v in context.entities.items())
            summary_parts.append(f"Key entities: {entities_str}")
        
        # Recent conversation
        if history:
            recent_queries = [item.get('query', '') for item in history[-3:] if item.get('query')]
            if recent_queries:
                summary_parts.append(f"Recent topics: {' | '.join(recent_queries)}")
        
        # Memory insights
        if memories:
            high_relevance = [m for m in memories if m.relevance_score >= 0.9]
            if high_relevance:
                summary_parts.append(f"Highly relevant memories found: {len(high_relevance)}")
            
            categories = set(m.metadata.get('analysis_type', 'general') for m in memories)
            if categories:
                summary_parts.append(f"Available analysis types: {', '.join(categories)}")
        
        return " | ".join(summary_parts)
    
    # ==========================================
    # MEMORY MAINTENANCE & OPTIMIZATION
    # ==========================================
    
    async def optimize_memory_usage(self) -> Dict[str, Any]:
        """Optimize memory usage across both stores"""
        results = {
            'redis_cleanup': 0,
            'vector_cleanup': 0,
            'optimizations_applied': []
        }
        
        try:
            # Clean up expired Redis sessions
            redis_cleaned = await self.redis_store.cleanup_expired_sessions()
            results['redis_cleanup'] = redis_cleaned
            
            # Clean up old vector memories (30+ days)
            vector_cleaned = await self.vector_store.cleanup_old_memories(days_old=30)
            results['vector_cleanup'] = vector_cleaned
            
            # Additional optimizations
            if redis_cleaned > 0:
                results['optimizations_applied'].append('Expired Redis sessions cleaned')
            
            if vector_cleaned > 0:
                results['optimizations_applied'].append('Old vector memories archived')
            
            logger.info(f"Memory optimization completed: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Memory optimization failed: {e}")
            results['error'] = str(e)
            return results
    
    async def get_memory_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status of memory systems"""
        try:
            redis_health = await self.redis_store.health_check()
            vector_health = await self.vector_store.health_check()
            
            # Aggregate health status
            overall_status = 'healthy'
            if redis_health.get('status') != 'healthy' or vector_health.get('status') != 'healthy':
                overall_status = 'degraded'
            
            return {
                'overall_status': overall_status,
                'redis_store': redis_health,
                'vector_store': vector_health,
                'memory_pipeline_status': 'operational',
                'last_check': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {
                'overall_status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    async def export_user_memories(self, user_id: str, format: str = 'json') -> Dict[str, Any]:
        """Export all memories for a specific user"""
        try:
            # Get memories from vector store
            user_memories = await self.vector_store.search_similar_memories(
                query="",  # Empty query to get all
                limit=1000,
                min_similarity=0.0,  # Include all similarities
                metadata_filter={'user_id': user_id}
            )
            
            export_data = {
                'user_id': user_id,
                'export_timestamp': datetime.utcnow().isoformat(),
                'total_memories': len(user_memories),
                'memories': [memory.to_dict() for memory in user_memories]
            }
            
            if format == 'json':
                return export_data
            else:
                # Could add other formats (CSV, XML, etc.)
                return {'error': f'Unsupported export format: {format}'}
                
        except Exception as e:
            logger.error(f"Failed to export memories for user {user_id}: {e}")
            return {'error': str(e)}
    
    # ==========================================
    # INTEGRATION HELPERS
    # ==========================================
    
    async def prepare_agent_context(self, session_id: str, thread_id: str, user_query: str) -> Dict[str, Any]:
        """Prepare enriched context for AI agent processing"""
        try:
            # Create conversation context
            context = ConversationContext(
                session_id=session_id,
                thread_id=thread_id,
                user_id=session_id,  # Using session_id as user_id for now
                query=user_query,
                intent=self._extract_intent(user_query),
                entities=self._extract_entities(user_query)
            )
            
            # Process through memory pipeline
            enriched_context = await self.process_user_input(context)
            
            return enriched_context
            
        except Exception as e:
            logger.error(f"Failed to prepare agent context: {e}")
            return {
                'current_context': {'query': user_query, 'error': str(e)},
                'recent_history': [],
                'relevant_memories': [],
                'memory_insights': {},
                'context_summary': f"Error preparing context: {str(e)}"
            }
    
    def _extract_intent(self, query: str) -> str:
        """Extract intent from user query (simplified)"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['analyze', 'analysis', 'examine']):
            return 'analysis_request'
        elif any(word in query_lower for word in ['compare', 'comparison', 'versus', 'vs']):
            return 'comparison_request'
        elif any(word in query_lower for word in ['forecast', 'predict', 'future', 'outlook']):
            return 'prediction_request'
        elif any(word in query_lower for word in ['risk', 'risks', 'volatility', 'danger']):
            return 'risk_assessment'
        elif any(word in query_lower for word in ['report', 'summary', 'overview']):
            return 'report_request'
        else:
            return 'general_inquiry'
    
    def _extract_entities(self, query: str) -> Dict[str, Any]:
        """Extract entities from user query (simplified)"""
        entities = {}
        
        # Look for stock symbols (simple pattern)
        import re
        stock_pattern = r'\b[A-Z]{2,5}\b'
        stocks = re.findall(stock_pattern, query)
        if stocks:
            entities['stocks'] = stocks
        
        # Look for financial terms
        financial_terms = ['revenue', 'profit', 'earnings', 'debt', 'assets', 'liabilities']
        found_terms = [term for term in financial_terms if term in query.lower()]
        if found_terms:
            entities['financial_metrics'] = found_terms
        
        # Look for time periods
        time_periods = ['quarterly', 'annual', 'monthly', 'yearly', 'q1', 'q2', 'q3', 'q4']
        found_periods = [period for period in time_periods if period in query.lower()]
        if found_periods:
            entities['time_periods'] = found_periods
        
        return entities


# Export main class
__all__ = ['MemoryPipeline', 'ConversationContext']