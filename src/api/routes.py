# src/api/routes.py
# HackRx 6.0 - API Routes Integration

import asyncio
from typing import Dict, Any

from ..utils.config import get_settings
from ..utils.logger import get_logger  
from ..reasoning.query_processor import QueryProcessor
from ..core.embeddings import get_embedding_generator
from ..core.vector_store import get_vector_store
from ..core.llm_handler import get_llm_handler

logger = get_logger(__name__)


class IntegratedQueryProcessor:
    """
    Integrated query processor that coordinates all system components
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.query_processor = QueryProcessor()
        self.initialized = False
    
    async def initialize(self):
        """
        Initialize all system components
        
        Raises:
            RuntimeError: If initialization fails
        """
        if self.initialized:
            return
        
        try:
            logger.info("🚀 Initializing integrated query processor...")
            
            # Initialize all components in parallel for faster startup
            await asyncio.gather(
                get_embedding_generator(),  # This initializes the embedding model
                get_vector_store(),         # This initializes the FAISS store
                get_llm_handler(),          # This initializes LLM clients
            )
            
            self.initialized = True
            logger.info("✅ Integrated query processor initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Query processor initialization failed: {e}")
            raise RuntimeError(f"Failed to initialize query processor: {e}")
    
    async def process_document_queries(self, document_url: str, questions: list) -> list:
        """
        Process queries against a document
        
        Args:
            document_url: URL of document to analyze
            questions: List of questions to answer
        
        Returns:
            List of answers
        
        Raises:
            RuntimeError: If processing fails
        """
        if not self.initialized:
            await self.initialize()
        
        try:
            # Use the query processor to handle the request
            answers = await self.query_processor.process_document_queries(
                document_url, questions
            )
            
            return answers
            
        except Exception as e:
            logger.error(f"Document query processing failed: {e}")
            raise RuntimeError(f"Failed to process document queries: {e}")
    
    async def get_system_status(self) -> Dict[str, Any]:
        """
        Get status of all system components
        
        Returns:
            Dictionary with system status information
        """
        try:
            status = {
                "initialized": self.initialized,
                "timestamp": asyncio.get_event_loop().time()
            }
            
            if self.initialized:
                # Get component statuses
                try:
                    embedding_generator = await get_embedding_generator()
                    status["embedding_model"] = embedding_generator.get_model_info()
                except Exception as e:
                    status["embedding_model"] = {"status": "error", "error": str(e)}
                
                try:
                    vector_store = await get_vector_store()
                    stats = vector_store.get_stats()
                    status["vector_store"] = {
                        "total_vectors": stats.total_vectors,
                        "dimension": stats.dimension,
                        "index_type": stats.index_type,
                        "memory_usage_mb": stats.memory_usage_mb
                    }
                except Exception as e:
                    status["vector_store"] = {"status": "error", "error": str(e)}
                    
                try:
                    llm_handler = await get_llm_handler()
                    status["llm_providers"] = llm_handler.get_provider_status()
                except Exception as e:
                    status["llm_providers"] = {"status": "error", "error": str(e)}
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            return {
                "initialized": False,
                "error": str(e),
                "timestamp": asyncio.get_event_loop().time()
            }
    
    async def cleanup(self):
        """
        Cleanup all system resources
        """
        logger.info("🧹 Cleaning up integrated query processor...")
        
        try:
            # Cleanup all components
            cleanup_tasks = []
            
            if self.query_processor:
                cleanup_tasks.append(self.query_processor.cleanup())
            
            try:
                embedding_generator = await get_embedding_generator()
                cleanup_tasks.append(embedding_generator.cleanup())
            except:
                pass
            
            try:
                llm_handler = await get_llm_handler()
                cleanup_tasks.append(llm_handler.cleanup())
            except:
                pass
            
            # Wait for all cleanup tasks
            if cleanup_tasks:
                await asyncio.gather(*cleanup_tasks, return_exceptions=True)
            
            self.initialized = False
            logger.info("✅ Integrated query processor cleanup completed")
            
        except Exception as e:
            logger.error(f"❌ Cleanup failed: {e}")


# Global integrated query processor instance
_query_processor = None


async def create_query_processor() -> IntegratedQueryProcessor:
    """
    Create and initialize the global query processor instance
    
    Returns:
        Initialized IntegratedQueryProcessor instance
    """
    global _query_processor
    
    if _query_processor is None:
        _query_processor = IntegratedQueryProcessor()
        await _query_processor.initialize()
    
    return _query_processor


async def get_query_processor() -> IntegratedQueryProcessor:
    """
    Get the global query processor instance
    
    Returns:
        IntegratedQueryProcessor instance
    
    Raises:
        RuntimeError: If processor not initialized
    """
    global _query_processor
    
    if _query_processor is None:
        raise RuntimeError("Query processor not initialized")
    
    return _query_processor