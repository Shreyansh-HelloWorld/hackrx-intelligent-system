# src/core/embeddings.py
# HackRx 6.0 - Text Embeddings Generation Module

import asyncio
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import os
from pathlib import Path

from sentence_transformers import SentenceTransformer
import torch

from ..utils.config import get_settings
from ..utils.logger import get_logger, log_performance_metric
from ..utils.helpers import generate_hash
from .document_processor import DocumentChunk, ProcessedDocument

logger = get_logger(__name__)


@dataclass
class EmbeddingResult:
    """
    Result of embedding generation
    """
    text: str
    embedding: np.ndarray
    chunk_index: int
    page_number: Optional[int] = None
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class EmbeddingGenerator:
    """
    Handles text embedding generation using sentence-transformers
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.model = None
        self.model_name = self.settings.embedding_model
        self.device = self._get_device()
        
    def _get_device(self) -> str:
        """
        Determine the best device for model inference
        
        Returns:
            Device string ('cuda', 'mps', or 'cpu')
        """
        if torch.cuda.is_available():
            device = "cuda"
            logger.info("🚀 Using CUDA GPU for embeddings")
        elif torch.backends.mps.is_available():
            device = "mps"
            logger.info("🚀 Using Apple MPS for embeddings")
        else:
            device = "cpu"
            logger.info("💻 Using CPU for embeddings")
        
        return device
    
    async def initialize(self) -> None:
        """
        Initialize the embedding model
        
        Raises:
            RuntimeError: If model loading fails
        """
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            
            # Create models directory
            models_dir = Path(self.settings.models_dir) / "embeddings"
            models_dir.mkdir(parents=True, exist_ok=True)
            
            # Load model (will download if not cached)
            self.model = SentenceTransformer(
                self.model_name,
                cache_folder=str(models_dir),
                device=self.device
            )
            
            # Test model with dummy input
            test_embedding = await self._generate_single_embedding("test")
            logger.info(f"✅ Model loaded successfully, embedding dim: {len(test_embedding)}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load embedding model: {e}")
            raise RuntimeError(f"Embedding model initialization failed: {e}")
    
    async def generate_document_embeddings(self, processed_doc: ProcessedDocument) -> List[EmbeddingResult]:
        """
        Generate embeddings for all chunks in a document
        
        Args:
            processed_doc: Processed document with chunks
        
        Returns:
            List of embedding results
        
        Raises:
            RuntimeError: If embedding generation fails
        """
        if not self.model:
            await self.initialize()
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            logger.info(f"Generating embeddings for {len(processed_doc.chunks)} chunks")
            
            # Extract text from chunks
            chunk_texts = [chunk.text for chunk in processed_doc.chunks]
            
            # Generate embeddings in batch for efficiency
            embeddings = await self._generate_batch_embeddings(chunk_texts)
            
            # Create embedding results
            results = []
            for i, (chunk, embedding) in enumerate(zip(processed_doc.chunks, embeddings)):
                result = EmbeddingResult(
                    text=chunk.text,
                    embedding=embedding,
                    chunk_index=i,
                    page_number=chunk.page_number,
                    metadata={
                        "document_hash": processed_doc.document_hash,
                        "document_title": processed_doc.title,
                        "chunk_metadata": chunk.metadata,
                        "embedding_model": self.model_name,
                        "embedding_dim": len(embedding)
                    }
                )
                results.append(result)
            
            processing_time = asyncio.get_event_loop().time() - start_time
            log_performance_metric("embedding_generation", processing_time, len(results))
            
            logger.info(f"✅ Generated {len(results)} embeddings in {processing_time:.2f}s")
            return results
            
        except Exception as e:
            processing_time = asyncio.get_event_loop().time() - start_time
            logger.error(f"❌ Embedding generation failed after {processing_time:.2f}s: {e}")
            raise RuntimeError(f"Failed to generate embeddings: {e}")
    
    async def generate_query_embedding(self, query: str) -> np.ndarray:
        """
        Generate embedding for a single query
        
        Args:
            query: Query text to embed
        
        Returns:
            Query embedding vector
        
        Raises:
            RuntimeError: If embedding generation fails
        """
        if not self.model:
            await self.initialize()
        
        try:
            logger.debug(f"Generating query embedding for: '{query[:50]}...'")
            
            embedding = await self._generate_single_embedding(query)
            
            logger.debug(f"Query embedding generated, dim: {len(embedding)}")
            return embedding
            
        except Exception as e:
            logger.error(f"Query embedding generation failed: {e}")
            raise RuntimeError(f"Failed to generate query embedding: {e}")
    
    async def _generate_single_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text
        
        Args:
            text: Text to embed
        
        Returns:
            Embedding vector
        """
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None, 
            lambda: self.model.encode(text, convert_to_numpy=True)
        )
        
        return embedding
    
    async def _generate_batch_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts in batch
        
        Args:
            texts: List of texts to embed
        
        Returns:
            List of embedding vectors
        """
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        
        # Process in batches to manage memory
        batch_size = 32  # Adjust based on available memory
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            logger.debug(f"Processing embedding batch {i//batch_size + 1}: {len(batch)} texts")
            
            batch_embeddings = await loop.run_in_executor(
                None,
                lambda b=batch: self.model.encode(b, convert_to_numpy=True, show_progress_bar=False)
            )
            
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
        
        Returns:
            Cosine similarity score (-1 to 1)
        """
        try:
            # Normalize vectors
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # Calculate cosine similarity
            similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Similarity calculation failed: {e}")
            return 0.0
    
    def get_model_info(self) -> Dict[str, any]:
        """
        Get information about the loaded model
        
        Returns:
            Dictionary with model information
        """
        if not self.model:
            return {"status": "not_loaded"}
        
        try:
            # Get model dimensions
            test_embedding = self.model.encode("test", convert_to_numpy=True)
            
            return {
                "status": "loaded",
                "model_name": self.model_name,
                "embedding_dimension": len(test_embedding),
                "device": self.device,
                "max_sequence_length": getattr(self.model, 'max_seq_length', 'unknown'),
                "model_path": str(self.model._cache_folder) if hasattr(self.model, '_cache_folder') else 'unknown'
            }
            
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return {"status": "error", "error": str(e)}
    
    async def cleanup(self):
        """
        Cleanup model resources
        """
        if self.model:
            logger.info("Cleaning up embedding model resources")
            
            # Clear CUDA cache if using GPU
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.model = None
            logger.info("✅ Embedding model resources cleaned up")


# Global embedding generator instance
_embedding_generator = None


async def get_embedding_generator() -> EmbeddingGenerator:
    """
    Get or create the global embedding generator instance
    
    Returns:
        Initialized EmbeddingGenerator instance
    """
    global _embedding_generator
    
    if _embedding_generator is None:
        _embedding_generator = EmbeddingGenerator()
        await _embedding_generator.initialize()
    
    return _embedding_generator