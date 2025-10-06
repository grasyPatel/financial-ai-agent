# Vector Memory System with Pinecone Integration
# Long-term memory storage for the Deep Finance Research Chatbot

import os
import json
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import asyncio
import numpy as np
from dataclasses import dataclass, asdict
import logging

# Optional imports - system should work without them
try:
    import pinecone
    from pinecone import Pinecone
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class MemoryRecord:
    """Structured memory record for financial research data"""
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    created_at: datetime = None
    relevance_score: float = 0.0
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryRecord':
        """Create from dictionary"""
        if isinstance(data.get('created_at'), str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        return cls(**data)


class VectorMemoryStore:
    """Advanced vector-based memory store with multiple backend support"""
    
    def __init__(self, 
                 config: Dict[str, Any],
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize vector memory store with configurable backend
        
        Args:
            config: Configuration dictionary with backend settings
            embedding_model: Model for generating embeddings
        """
        self.config = config
        self.embedding_model_name = embedding_model
        self.embedding_model = None
        self.vector_store = None
        self.dimension = 384  # Default for all-MiniLM-L6-v2
        
        # Initialize based on available dependencies and config
        self._initialize_embedding_model()
        self._initialize_vector_store()
        
        logger.info(f"✅ Vector Memory Store initialized with {self.config.get('backend', 'fallback')} backend")
    
    def _initialize_embedding_model(self):
        """Initialize embedding model based on available dependencies"""
        try:
            if OPENAI_AVAILABLE and self.config.get('use_openai_embeddings'):
                self.embedding_type = 'openai'
                openai.api_key = self.config.get('openai_api_key', os.getenv('OPENAI_API_KEY'))
                self.dimension = 1536  # OpenAI ada-002 dimension
                logger.info("Using OpenAI embeddings")
            
            elif SENTENCE_TRANSFORMERS_AVAILABLE:
                self.embedding_type = 'sentence_transformers'
                self.embedding_model = SentenceTransformer(self.embedding_model_name)
                self.dimension = self.embedding_model.get_sentence_embedding_dimension()
                logger.info(f"Using Sentence Transformers: {self.embedding_model_name}")
            
            else:
                self.embedding_type = 'fallback'
                logger.warning("No embedding models available, using fallback hash-based similarity")
                
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            self.embedding_type = 'fallback'
    
    def _initialize_vector_store(self):
        """Initialize vector store backend"""
        backend = self.config.get('backend', 'memory')
        
        try:
            if backend == 'pinecone' and PINECONE_AVAILABLE:
                self._init_pinecone()
            elif backend == 'memory':
                self._init_memory_backend()
            else:
                logger.warning(f"Backend '{backend}' not available, falling back to memory")
                self._init_memory_backend()
                
        except Exception as e:
            logger.error(f"Failed to initialize vector store backend: {e}")
            self._init_memory_backend()
    
    def _init_pinecone(self):
        """Initialize Pinecone vector database"""
        try:
            api_key = self.config.get('pinecone_api_key', os.getenv('PINECONE_API_KEY'))
            if not api_key:
                raise ValueError("Pinecone API key not found")
            
            pc = Pinecone(api_key=api_key)
            
            index_name = self.config.get('pinecone_index', 'deqode-finance-memory')
            
            # Create index if it doesn't exist
            if index_name not in [idx.name for idx in pc.list_indexes()]:
                pc.create_index(
                    name=index_name,
                    dimension=self.dimension,
                    metric='cosine',
                    spec=pinecone.ServerlessSpec(
                        cloud=self.config.get('pinecone_cloud', 'aws'),
                        region=self.config.get('pinecone_region', 'us-east-1')
                    )
                )
                logger.info(f"Created Pinecone index: {index_name}")
            
            self.vector_store = pc.Index(index_name)
            self.backend_type = 'pinecone'
            logger.info(f"Connected to Pinecone index: {index_name}")
            
        except Exception as e:
            logger.error(f"Pinecone initialization failed: {e}")
            self._init_memory_backend()
    
    def _init_memory_backend(self):
        """Initialize in-memory vector store as fallback"""
        self.vector_store = {
            'records': {},  # id -> MemoryRecord
            'embeddings': {},  # id -> embedding vector
            'index': []  # List of ids for iteration
        }
        self.backend_type = 'memory'
        logger.info("Using in-memory vector store")
    
    # ==========================================
    # EMBEDDING GENERATION
    # ==========================================
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text"""
        try:
            if self.embedding_type == 'openai':
                return await self._generate_openai_embedding(text)
            elif self.embedding_type == 'sentence_transformers':
                return self._generate_sentence_transformer_embedding(text)
            else:
                return self._generate_fallback_embedding(text)
                
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return self._generate_fallback_embedding(text)
    
    async def _generate_openai_embedding(self, text: str) -> List[float]:
        """Generate OpenAI embedding"""
        response = await openai.Embedding.acreate(
            input=text,
            model="text-embedding-ada-002"
        )
        return response['data'][0]['embedding']
    
    def _generate_sentence_transformer_embedding(self, text: str) -> List[float]:
        """Generate Sentence Transformer embedding"""
        return self.embedding_model.encode(text).tolist()
    
    def _generate_fallback_embedding(self, text: str) -> List[float]:
        """Generate fallback hash-based embedding"""
        # Simple hash-based approach for when no ML models available
        text_hash = hashlib.md5(text.encode()).hexdigest()
        # Convert hash to fixed-size vector
        hash_ints = [int(text_hash[i:i+2], 16) for i in range(0, min(len(text_hash), 32), 2)]
        # Pad to dimension and normalize
        vector = hash_ints + [0] * (16 - len(hash_ints))
        norm = np.linalg.norm(vector) or 1
        return [float(x) / norm for x in vector]
    
    # ==========================================
    # MEMORY OPERATIONS
    # ==========================================
    
    async def store_memory(self, content: str, metadata: Dict[str, Any]) -> str:
        """Store new memory record with embedding"""
        try:
            # Generate unique ID
            memory_id = hashlib.sha256(f"{content}_{datetime.utcnow().isoformat()}".encode()).hexdigest()[:16]
            
            # Generate embedding
            embedding = await self.generate_embedding(content)
            
            # Create memory record
            record = MemoryRecord(
                id=memory_id,
                content=content,
                metadata=metadata,
                embedding=embedding
            )
            
            # Store based on backend
            if self.backend_type == 'pinecone':
                await self._store_pinecone(record)
            else:
                await self._store_memory_backend(record)
            
            logger.debug(f"Stored memory: {memory_id}")
            return memory_id
            
        except Exception as e:
            logger.error(f"Failed to store memory: {e}")
            return ""
    
    async def _store_pinecone(self, record: MemoryRecord):
        """Store record in Pinecone"""
        self.vector_store.upsert([{
            'id': record.id,
            'values': record.embedding,
            'metadata': {
                'content': record.content,
                'created_at': record.created_at.isoformat(),
                **record.metadata
            }
        }])
    
    async def _store_memory_backend(self, record: MemoryRecord):
        """Store record in memory backend"""
        self.vector_store['records'][record.id] = record
        self.vector_store['embeddings'][record.id] = record.embedding
        if record.id not in self.vector_store['index']:
            self.vector_store['index'].append(record.id)
    
    async def search_similar_memories(self, 
                                    query: str, 
                                    limit: int = 10,
                                    min_similarity: float = 0.7,
                                    metadata_filter: Optional[Dict[str, Any]] = None) -> List[MemoryRecord]:
        """Search for similar memories using vector similarity"""
        try:
            # Generate query embedding
            query_embedding = await self.generate_embedding(query)
            
            # Search based on backend
            if self.backend_type == 'pinecone':
                return await self._search_pinecone(query_embedding, limit, min_similarity, metadata_filter)
            else:
                return await self._search_memory_backend(query_embedding, limit, min_similarity, metadata_filter)
                
        except Exception as e:
            logger.error(f"Failed to search memories: {e}")
            return []
    
    async def _search_pinecone(self, 
                             query_embedding: List[float], 
                             limit: int,
                             min_similarity: float,
                             metadata_filter: Optional[Dict[str, Any]]) -> List[MemoryRecord]:
        """Search Pinecone for similar vectors"""
        # Build filter for Pinecone
        filter_dict = metadata_filter or {}
        
        # Query Pinecone
        results = self.vector_store.query(
            vector=query_embedding,
            top_k=limit,
            include_metadata=True,
            filter=filter_dict if filter_dict else None
        )
        
        # Convert to MemoryRecord objects
        memories = []
        for match in results.matches:
            if match.score >= min_similarity:
                metadata = match.metadata.copy()
                content = metadata.pop('content', '')
                created_at = datetime.fromisoformat(metadata.pop('created_at', datetime.utcnow().isoformat()))
                
                record = MemoryRecord(
                    id=match.id,
                    content=content,
                    metadata=metadata,
                    embedding=None,  # Don't store embedding in results
                    created_at=created_at,
                    relevance_score=match.score
                )
                memories.append(record)
        
        return memories
    
    async def _search_memory_backend(self, 
                                   query_embedding: List[float],
                                   limit: int,
                                   min_similarity: float,
                                   metadata_filter: Optional[Dict[str, Any]]) -> List[MemoryRecord]:
        """Search memory backend for similar vectors"""
        similarities = []
        
        for record_id in self.vector_store['index']:
            record = self.vector_store['records'][record_id]
            embedding = self.vector_store['embeddings'][record_id]
            
            # Apply metadata filter
            if metadata_filter:
                if not all(record.metadata.get(k) == v for k, v in metadata_filter.items()):
                    continue
            
            # Calculate cosine similarity
            similarity = self._cosine_similarity(query_embedding, embedding)
            
            if similarity >= min_similarity:
                record.relevance_score = similarity
                similarities.append((similarity, record))
        
        # Sort by similarity and return top results
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [record for _, record in similarities[:limit]]
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            norm_a = sum(a * a for a in vec1) ** 0.5
            norm_b = sum(b * b for b in vec2) ** 0.5
            
            if norm_a == 0 or norm_b == 0:
                return 0.0
            
            return dot_product / (norm_a * norm_b)
        except:
            return 0.0
    
    # ==========================================
    # MEMORY MANAGEMENT
    # ==========================================
    
    async def get_memory(self, memory_id: str) -> Optional[MemoryRecord]:
        """Retrieve specific memory by ID"""
        try:
            if self.backend_type == 'pinecone':
                results = self.vector_store.fetch([memory_id])
                if memory_id in results.vectors:
                    vector_data = results.vectors[memory_id]
                    metadata = vector_data.metadata
                    content = metadata.pop('content', '')
                    created_at = datetime.fromisoformat(metadata.pop('created_at', datetime.utcnow().isoformat()))
                    
                    return MemoryRecord(
                        id=memory_id,
                        content=content,
                        metadata=metadata,
                        created_at=created_at
                    )
            else:
                return self.vector_store['records'].get(memory_id)
            
            return None
        except Exception as e:
            logger.error(f"Failed to get memory {memory_id}: {e}")
            return None
    
    async def delete_memory(self, memory_id: str) -> bool:
        """Delete memory by ID"""
        try:
            if self.backend_type == 'pinecone':
                self.vector_store.delete([memory_id])
            else:
                if memory_id in self.vector_store['records']:
                    del self.vector_store['records'][memory_id]
                    del self.vector_store['embeddings'][memory_id]
                    self.vector_store['index'].remove(memory_id)
            
            logger.debug(f"Deleted memory: {memory_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete memory {memory_id}: {e}")
            return False
    
    async def update_memory(self, memory_id: str, content: str = None, metadata: Dict[str, Any] = None) -> bool:
        """Update existing memory"""
        try:
            current_record = await self.get_memory(memory_id)
            if not current_record:
                return False
            
            # Update content and/or metadata
            if content is not None:
                current_record.content = content
                current_record.embedding = await self.generate_embedding(content)
            
            if metadata is not None:
                current_record.metadata.update(metadata)
            
            # Re-store the updated record
            if self.backend_type == 'pinecone':
                await self._store_pinecone(current_record)
            else:
                await self._store_memory_backend(current_record)
            
            logger.debug(f"Updated memory: {memory_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to update memory {memory_id}: {e}")
            return False
    
    # ==========================================
    # ANALYTICS & INSIGHTS
    # ==========================================
    
    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics"""
        try:
            if self.backend_type == 'pinecone':
                stats = self.vector_store.describe_index_stats()
                return {
                    'total_vectors': stats.total_vector_count,
                    'dimension': stats.dimension,
                    'index_fullness': stats.index_fullness,
                    'backend': 'pinecone'
                }
            else:
                return {
                    'total_vectors': len(self.vector_store['records']),
                    'dimension': self.dimension,
                    'backend': 'memory',
                    'memory_usage_mb': len(str(self.vector_store)) / (1024 * 1024)
                }
        except Exception as e:
            logger.error(f"Failed to get memory stats: {e}")
            return {'error': str(e)}
    
    async def get_recent_memories(self, limit: int = 50) -> List[MemoryRecord]:
        """Get most recently stored memories"""
        try:
            if self.backend_type == 'pinecone':
                # Pinecone doesn't support sorting by metadata, so we'll do a broad query
                # This is a limitation - in production, you'd want to track recency separately
                results = self.vector_store.query(
                    vector=[0.0] * self.dimension,  # Dummy vector
                    top_k=limit,
                    include_metadata=True
                )
                
                memories = []
                for match in results.matches:
                    metadata = match.metadata.copy()
                    content = metadata.pop('content', '')
                    created_at = datetime.fromisoformat(metadata.pop('created_at', datetime.utcnow().isoformat()))
                    
                    record = MemoryRecord(
                        id=match.id,
                        content=content,
                        metadata=metadata,
                        created_at=created_at
                    )
                    memories.append(record)
                
                # Sort by created_at
                memories.sort(key=lambda x: x.created_at, reverse=True)
                return memories[:limit]
            else:
                records = list(self.vector_store['records'].values())
                records.sort(key=lambda x: x.created_at, reverse=True)
                return records[:limit]
                
        except Exception as e:
            logger.error(f"Failed to get recent memories: {e}")
            return []
    
    async def cleanup_old_memories(self, days_old: int = 30) -> int:
        """Clean up memories older than specified days"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_old)
            deleted_count = 0
            
            if self.backend_type == 'pinecone':
                # Pinecone cleanup is more complex - would need to scan all vectors
                logger.warning("Automated cleanup not implemented for Pinecone backend")
                return 0
            else:
                to_delete = []
                for record_id, record in self.vector_store['records'].items():
                    if record.created_at < cutoff_date:
                        to_delete.append(record_id)
                
                for record_id in to_delete:
                    await self.delete_memory(record_id)
                    deleted_count += 1
            
            logger.info(f"Cleaned up {deleted_count} old memories")
            return deleted_count
        except Exception as e:
            logger.error(f"Failed to cleanup old memories: {e}")
            return 0
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for vector memory system"""
        try:
            stats = await self.get_memory_stats()
            
            # Test embedding generation
            test_embedding = await self.generate_embedding("test query")
            embedding_works = len(test_embedding) > 0
            
            # Test search
            search_results = await self.search_similar_memories("test", limit=1)
            search_works = isinstance(search_results, list)
            
            return {
                'status': 'healthy' if embedding_works and search_works else 'degraded',
                'backend_type': self.backend_type,
                'embedding_type': self.embedding_type,
                'stats': stats,
                'embedding_generation': 'working' if embedding_works else 'failed',
                'search_functionality': 'working' if search_works else 'failed',
                'timestamp': datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }


# Export main classes
__all__ = ['VectorMemoryStore', 'MemoryRecord']