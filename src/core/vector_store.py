# src/core/vector_store.py
# HackRx 6.0 - FAISS Vector Store for Semantic Search

import asyncio
import pickle
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import numpy as np

import faiss

from ..utils.config import get_settings
from ..utils.logger import get_logger, log_performance_metric
from ..utils.helpers import generate_hash
from .embeddings import EmbeddingResult

logger = get_logger(__name__)


@dataclass  
class SearchResult:
    """
    Result from vector similarity search
    """
    text: str
    score: float
    chunk_index: int
    page_number: Optional[int] = None
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class VectorStoreStats:
    """
    Statistics about the vector store
    """
    total_vectors: int
    dimension: int
    index_type: str
    memory_usage_mb: float
    last_updated: str


class FAISSVectorStore:
    """
    FAISS-based vector store for semantic search
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.index = None
        self.metadata_store = []  # Store metadata for each vector
        self.dimension = None
        self.document_hash = None
        
    def _create_index(self, dimension: int) -> faiss.Index:
        """
        Create FAISS index based on the number of vectors
        
        Args:
            dimension: Vector dimension
        
        Returns:
            FAISS index instance
        """
        # For small datasets (< 1000 vectors), use flat index for exact search
        # For larger datasets, use IVF index for approximate search
        
        if hasattr(self, 'metadata_store') and len(self.metadata_store) < 1000:
            # Exact search using L2 distance
            index = faiss.IndexFlatL2(dimension)
            logger.info(f"Created flat L2 index for exact search (dim: {dimension})")
        else:
            # Approximate search using IVF
            nlist = 100  # Number of clusters
            quantizer = faiss.IndexFlatL2(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            logger.info(f"Created IVF index for approximate search (dim: {dimension}, nlist: {nlist})")
        
        return index
    
    async def initialize(self, dimension: int):
        """
        Initialize the vector store with given dimension
        
        Args:
            dimension: Vector dimension for the index
        """
        try:
            self.dimension = dimension
            self.index = self._create_index(dimension)
            self.metadata_store = []
            
            logger.info(f"✅ Vector store initialized with dimension {dimension}")
            
        except Exception as e:
            logger.error(f"❌ Vector store initialization failed: {e}")
            raise RuntimeError(f"Failed to initialize vector store: {e}")
    
    async def add_embeddings(self, embeddings: List[EmbeddingResult], document_hash: str):
        """
        Add embeddings to the vector store
        
        Args:
            embeddings: List of embedding results to add
            document_hash: Hash of the source document
        
        Raises:
            RuntimeError: If adding embeddings fails
        """
        if not embeddings:
            logger.warning("No embeddings to add")
            return
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Initialize if needed
            if self.index is None:
                await self.initialize(len(embeddings[0].embedding))
            
            logger.info(f"Adding {len(embeddings)} embeddings to vector store")
            
            # Prepare vectors
            vectors = np.array([emb.embedding for emb in embeddings]).astype('float32')
            
            # Prepare metadata
            metadata_batch = []
            for i, emb in enumerate(embeddings):
                metadata = {
                    "text": emb.text,
                    "chunk_index": emb.chunk_index,
                    "page_number": emb.page_number,
                    "document_hash": document_hash,
                    "vector_index": len(self.metadata_store) + i,
                    **emb.metadata
                }
                metadata_batch.append(metadata)
            
            # Add to FAISS index
            if isinstance(self.index, faiss.IndexIVFFlat):
                # Train IVF index if not already trained
                if not self.index.is_trained:
                    logger.info("Training IVF index...")
                    self.index.train(vectors)
            
            # Add vectors to index
            self.index.add(vectors)
            
            # Add metadata
            self.metadata_store.extend(metadata_batch)
            
            # Update document hash
            self.document_hash = document_hash
            
            processing_time = asyncio.get_event_loop().time() - start_time
            log_performance_metric("vector_store_add", processing_time, len(embeddings))
            
            logger.info(
                f"✅ Added {len(embeddings)} vectors in {processing_time:.2f}s. "
                f"Total vectors: {self.index.ntotal}"
            )
            
        except Exception as e:
            processing_time = asyncio.get_event_loop().time() - start_time
            logger.error(f"❌ Failed to add embeddings after {processing_time:.2f}s: {e}")
            raise RuntimeError(f"Failed to add embeddings to vector store: {e}")
    
    async def search(self, query_embedding: np.ndarray, top_k: int = None) -> List[SearchResult]:
        """
        Search for similar vectors
        
        Args:
            query_embedding: Query vector to search for
            top_k: Number of top results to return
        
        Returns:
            List of search results ordered by similarity
        
        Raises:
            RuntimeError: If search fails
        """
        if self.index is None or self.index.ntotal == 0:
            logger.warning("Vector store is empty")
            return []
        
        if top_k is None:
            top_k = self.settings.top_k_results
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Ensure query is the right shape and type
            query_vector = np.array([query_embedding]).astype('float32')
            
            # Perform search
            distances, indices = self.index.search(query_vector, top_k)
            
            # Convert to search results
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx == -1:  # FAISS returns -1 for empty results
                    continue
                
                # Convert distance to similarity score (lower distance = higher similarity)
                # For L2 distance, convert to similarity between 0 and 1
                similarity_score = 1.0 / (1.0 + distance)
                
                # Skip results below similarity threshold
                if similarity_score < self.settings.similarity_threshold:
                    continue
                
                # Get metadata
                metadata = self.metadata_store[idx]
                
                result = SearchResult(
                    text=metadata["text"],
                    score=similarity_score,
                    chunk_index=metadata["chunk_index"],
                    page_number=metadata.get("page_number"),
                    metadata={
                        "distance": float(distance),
                        "vector_index": idx,
                        "document_hash": metadata["document_hash"],
                        **{k: v for k, v in metadata.items() 
                           if k not in ["text", "chunk_index", "page_number"]}
                    }
                )
                
                results.append(result)
            
            processing_time = asyncio.get_event_loop().time() - start_time
            log_performance_metric("vector_search", processing_time, len(results))
            
            logger.info(
                f"✅ Found {len(results)} similar vectors in {processing_time:.3f}s "
                f"(searched {self.index.ntotal} total vectors)"
            )
            
            return results
            
        except Exception as e:
            processing_time = asyncio.get_event_loop().time() - start_time
            logger.error(f"❌ Vector search failed after {processing_time:.3f}s: {e}")
            raise RuntimeError(f"Vector search failed: {e}")
    
    async def hybrid_search(self, query_embedding: np.ndarray, query_text: str, 
                           top_k: int = None) -> List[SearchResult]:
        """
        Perform hybrid search combining vector similarity and keyword matching
        
        Args:
            query_embedding: Query vector for semantic search
            query_text: Query text for keyword matching
            top_k: Number of results to return
        
        Returns:
            List of search results with combined scoring
        """
        if top_k is None:
            top_k = self.settings.top_k_results
        
        # Get more results from vector search to allow for re-ranking
        vector_results = await self.search(query_embedding, top_k * 2)
        
        if not vector_results:
            return []
        
        # Keyword matching boost
        query_terms = set(query_text.lower().split())
        
        # Re-score results combining vector similarity and keyword overlap
        for result in vector_results:
            text_terms = set(result.text.lower().split())
            keyword_overlap = len(query_terms.intersection(text_terms)) / max(len(query_terms), 1)
            
            # Combine scores (70% vector similarity, 30% keyword overlap)
            combined_score = 0.7 * result.score + 0.3 * keyword_overlap
            result.score = combined_score
            result.metadata["keyword_overlap"] = keyword_overlap
            result.metadata["original_vector_score"] = result.score
        
        # Sort by combined score and return top k
        vector_results.sort(key=lambda x: x.score, reverse=True)
        
        logger.info(f"Hybrid search returning {min(len(vector_results), top_k)} results")
        return vector_results[:top_k]
    
    def get_stats(self) -> VectorStoreStats:
        """
        Get statistics about the vector store
        
        Returns:
            VectorStoreStats with current state information
        """
        import time
        
        if self.index is None:
            return VectorStoreStats(
                total_vectors=0,
                dimension=0,
                index_type="not_initialized",
                memory_usage_mb=0.0,
                last_updated="never"
            )
        
        # Estimate memory usage (rough approximation)
        vector_size = self.dimension * 4  # 4 bytes per float32
        metadata_size = sum(len(str(meta)) for meta in self.metadata_store)
        memory_usage_mb = (self.index.ntotal * vector_size + metadata_size) / (1024 * 1024)
        
        return VectorStoreStats(
            total_vectors=self.index.ntotal,
            dimension=self.dimension or 0,
            index_type=type(self.index).__name__,
            memory_usage_mb=memory_usage_mb,
            last_updated=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        )
    
    async def clear(self):
        """
        Clear all vectors and metadata from the store
        """
        logger.info("Clearing vector store")
        
        if self.index:
            self.index.reset()
        
        self.metadata_store.clear()
        self.document_hash = None
        
        logger.info("✅ Vector store cleared")
    
    async def save_to_disk(self, filepath: str):
        """
        Save vector store to disk
        
        Args:
            filepath: Path to save the index and metadata
        """
        try:
            if self.index is None or self.index.ntotal == 0:
                logger.warning("No vectors to save")
                return
            
            base_path = Path(filepath)
            base_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save FAISS index
            index_path = str(base_path.with_suffix('.index'))
            faiss.write_index(self.index, index_path)
            
            # Save metadata
            metadata_path = str(base_path.with_suffix('.metadata'))
            with open(metadata_path, 'wb') as f:
                pickle.dump({
                    'metadata_store': self.metadata_store,
                    'dimension': self.dimension,
                    'document_hash': self.document_hash
                }, f)
            
            logger.info(f"✅ Vector store saved to {filepath}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save vector store: {e}")
            raise RuntimeError(f"Failed to save vector store: {e}")
    
    async def load_from_disk(self, filepath: str):
        """
        Load vector store from disk
        
        Args:
            filepath: Path to load the index and metadata from
        """
        try:
            base_path = Path(filepath)
            index_path = str(base_path.with_suffix('.index'))
            metadata_path = str(base_path.with_suffix('.metadata'))
            
            if not Path(index_path).exists() or not Path(metadata_path).exists():
                raise FileNotFoundError(f"Vector store files not found at {filepath}")
            
            # Load FAISS index
            self.index = faiss.read_index(index_path)
            
            # Load metadata
            with open(metadata_path, 'rb') as f:
                data = pickle.load(f)
                self.metadata_store = data['metadata_store']
                self.dimension = data['dimension']
                self.document_hash = data['document_hash']
            
            logger.info(f"✅ Vector store loaded from {filepath}")
            logger.info(f"Loaded {self.index.ntotal} vectors, dimension {self.dimension}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load vector store: {e}")
            raise RuntimeError(f"Failed to load vector store: {e}")


# Global vector store instance
_vector_store = None


async def get_vector_store() -> FAISSVectorStore:
    """
    Get or create the global vector store instance
    
    Returns:
        FAISSVectorStore instance
    """
    global _vector_store
    
    if _vector_store is None:
        _vector_store = FAISSVectorStore()
    
    return _vector_store