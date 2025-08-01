# src/main.py
# HackRx 6.0 - FastAPI Main Application Entry Point

import time
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .utils.config import get_settings, validate_environment, create_directories
from .utils.logger import get_logger, log_api_request, log_performance_metric
from .api.middleware import AuthMiddleware, RequestLoggingMiddleware
from .api.routes import create_query_processor

logger = get_logger(__name__)


# Pydantic Models for API
class HackRXRequest(BaseModel):
    """Request model for /hackrx/run endpoint"""
    documents: str = Field(..., description="URL of the document to process")
    questions: List[str] = Field(..., min_items=1, description="List of questions to answer")


class HackRXResponse(BaseModel):
    """Response model for /hackrx/run endpoint"""
    answers: List[str] = Field(..., description="List of answers corresponding to questions")


class HealthResponse(BaseModel):
    """Health check response model"""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="Application version")
    timestamp: str = Field(..., description="Current timestamp")
    environment: str = Field(..., description="Environment name")


# Global query processor instance
query_processor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan management
    Handles startup and shutdown events
    """
    # Startup
    logger.info("🚀 Starting HackRx Intelligent Query-Retrieval System")
    
    try:
        # Validate environment
        if not validate_environment():
            logger.error("❌ Environment validation failed")
            raise RuntimeError("Invalid environment configuration")
        
        # Create necessary directories
        create_directories()
        
        # Initialize query processor
        global query_processor
        query_processor = await create_query_processor()
        
        logger.info("✅ System initialization complete")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🛬 Shutting down HackRx system")
    
    # Cleanup resources if needed
    if query_processor:
        await query_processor.cleanup()


# Create FastAPI application
app = FastAPI(
    title="HackRx Intelligent Query-Retrieval System",
    description="LLM-powered document analysis and query answering system for HackRx 6.0",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure as needed for production
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Add custom middleware
app.add_middleware(AuthMiddleware)
app.add_middleware(RequestLoggingMiddleware)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler"""
    logger.error(f"HTTP {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """General exception handler for unhandled errors"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "status_code": 500}
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint
    Returns system status and basic information
    """
    settings = get_settings()
    
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        environment=settings.environment
    )


@app.get("/")
async def root():
    """
    Root endpoint with system information
    """
    return {
        "message": "HackRx 6.0 Intelligent Query-Retrieval System",
        "status": "operational",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "hackrx": "/hackrx/run"
        }
    }


@app.post("/hackrx/run", response_model=HackRXResponse)
async def hackrx_run(request: HackRXRequest):
    """
    Main HackRx endpoint for document processing and question answering
    
    This endpoint:
    1. Downloads and processes the document from the provided URL
    2. Analyzes each question using semantic search and LLM reasoning
    3. Returns structured answers with explainable reasoning
    
    Args:
        request: HackRXRequest containing document URL and questions
    
    Returns:
        HackRXResponse with answers to all questions
    
    Raises:
        HTTPException: If processing fails or invalid input provided
    """
    start_time = time.time()
    
    try:
        logger.info(f"Processing request with {len(request.questions)} questions")
        logger.info(f"Document URL: {request.documents}")
        
        # Validate inputs
        if not request.documents or not request.documents.strip():
            raise HTTPException(status_code=400, detail="Document URL is required")
        
        if not request.questions:
            raise HTTPException(status_code=400, detail="At least one question is required")
        
        # Process the request using query processor
        if not query_processor:
            raise HTTPException(status_code=503, detail="Query processor not initialized")
        
        answers = await query_processor.process_document_queries(
            document_url=request.documents,
            questions=request.questions
        )
        
        # Validate response
        if len(answers) != len(request.questions):
            logger.error(f"Answer count mismatch: {len(answers)} vs {len(request.questions)}")
            raise HTTPException(status_code=500, detail="Processing error: answer count mismatch")
        
        processing_time = time.time() - start_time
        log_performance_metric("hackrx_run", processing_time, len(request.questions))
        
        logger.info(f"Successfully processed {len(request.questions)} questions in {processing_time:.2f}s")
        
        return HackRXResponse(answers=answers)
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Request processing failed after {processing_time:.2f}s: {e}", exc_info=True)
        
        raise HTTPException(
            status_code=500,
            detail=f"Processing failed: {str(e)}"
        )


# Development server configuration
if __name__ == "__main__":
    import uvicorn
    
    settings = get_settings()
    
    logger.info("🚀 Starting development server")
    
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
        access_log=True
    )