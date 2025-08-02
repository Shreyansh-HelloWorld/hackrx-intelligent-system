# src/main.py
# HackRx 6.0 - FastAPI Main Application Entry Point (RENDER DEPLOYMENT OPTIMIZED)

import os
import time
import asyncio
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .utils.config import get_settings, validate_environment, create_directories
from .utils.logger import get_logger, log_api_request, log_performance_metric
from .api.middleware import AuthMiddleware, RequestLoggingMiddleware

logger = get_logger(__name__)

# Global state management
class SystemState:
    def __init__(self):
        self.query_processor: Optional[object] = None
        self.is_initializing = False
        self.initialization_complete = False
        self.initialization_error: Optional[str] = None
        self.initialization_start_time: Optional[float] = None
        self._lock = asyncio.Lock()
    
    async def get_processor(self):
        """Get processor, initializing if needed"""
        if self.initialization_complete and self.query_processor:
            return self.query_processor
        
        if self.initialization_error:
            raise RuntimeError(f"System initialization failed: {self.initialization_error}")
        
        if not self.is_initializing:
            # Start initialization in background
            asyncio.create_task(self._initialize_system())
        
        # Wait for initialization with timeout
        max_wait = 90  # 90 seconds max wait
        wait_interval = 1  # Check every second
        waited = 0
        
        while waited < max_wait:
            if self.initialization_complete:
                return self.query_processor
            if self.initialization_error:
                raise RuntimeError(f"System initialization failed: {self.initialization_error}")
            
            await asyncio.sleep(wait_interval)
            waited += wait_interval
        
        raise RuntimeError("System initialization timeout - please try again")
    
    async def _initialize_system(self):
        """Initialize system components with proper locking"""
        async with self._lock:
            if self.is_initializing or self.initialization_complete:
                return
            
            self.is_initializing = True
            self.initialization_start_time = time.time()
            
            try:
                logger.info("🚀 Starting system initialization...")
                
                # Validate environment
                if not validate_environment():
                    raise RuntimeError("Invalid environment configuration")
                
                # Create necessary directories
                create_directories()
                
                # Initialize query processor
                from .api.routes import create_query_processor
                self.query_processor = await create_query_processor()
                
                self.initialization_complete = True
                self.is_initializing = False
                
                init_time = time.time() - self.initialization_start_time
                logger.info(f"✅ System initialization complete in {init_time:.2f}s")
                
            except Exception as e:
                self.initialization_error = str(e)
                self.is_initializing = False
                logger.error(f"❌ System initialization failed: {e}")
                raise
    
    def get_status(self):
        """Get current initialization status"""
        elapsed = 0
        if self.initialization_start_time:
            elapsed = time.time() - self.initialization_start_time
        
        if self.initialization_complete:
            return {"status": "ready", "elapsed_time": elapsed}
        elif self.is_initializing:
            return {"status": "initializing", "elapsed_time": elapsed}
        elif self.initialization_error:
            return {"status": "error", "error": self.initialization_error, "elapsed_time": elapsed}
        else:
            return {"status": "pending", "elapsed_time": 0}

# Global system state
system_state = SystemState()

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
    system_initialization: dict = Field(..., description="System initialization status")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan management - OPTIMIZED for fast Render startup
    """
    logger.info("🚀 Starting HackRx Intelligent Query-Retrieval System (Fast Boot Mode)")
    
    try:
        # Minimal startup - just basic validation
        create_directories()
        logger.info("✅ Fast startup complete - system ready for health checks")
        
        # Start heavy initialization in background (don't wait)
        asyncio.create_task(system_state._initialize_system())
        
    except Exception as e:
        logger.error(f"❌ Fast startup failed: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🛬 Shutting down HackRx system")
    if system_state.query_processor:
        try:
            await system_state.query_processor.cleanup()
        except:
            pass  # Don't fail shutdown on cleanup errors


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
    allow_origins=["*"],
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
    Health check endpoint - FAST RESPONSE for Render port detection
    """
    settings = get_settings()
    
    return HealthResponse(
        status="healthy",  # Always healthy for Render port detection
        version="1.0.0",
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        environment=settings.environment,
        system_initialization=system_state.get_status()
    )


@app.get("/")
async def root():
    """
    Root endpoint with system information
    """
    return {
        "message": "HackRx 6.0 Intelligent Query-Retrieval System",
        "status": "operational",
        "system_initialization": system_state.get_status(),
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "hackrx": "/hackrx/run"
        }
    }


@app.get("/status")
async def status_check():
    """
    Detailed status endpoint for monitoring initialization
    """
    return {
        "system": system_state.get_status(),
        "ready_for_processing": system_state.initialization_complete,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    }


@app.post("/hackrx/run", response_model=HackRXResponse)
async def hackrx_run(request: HackRXRequest):
    """
    Main HackRx endpoint for document processing and question answering
    
    This endpoint ensures system is fully initialized before processing requests.
    First request may take longer due to model loading.
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
        
        # Get initialized processor (may wait for initialization)
        try:
            query_processor = await system_state.get_processor()
        except RuntimeError as e:
            if "timeout" in str(e).lower():
                raise HTTPException(
                    status_code=503, 
                    detail="System is still initializing. Please try again in a few moments."
                )
            elif "initialization failed" in str(e).lower():
                raise HTTPException(
                    status_code=503,
                    detail="System initialization failed. Please contact support."
                )
            else:
                raise HTTPException(status_code=503, detail=str(e))
        
        # Process the request
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
    
    # Use PORT environment variable for Render deployment
    port = int(os.environ.get("PORT", 8000))
    
    logger.info(f"🚀 Starting development server on port {port}")
    
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
        access_log=True
    )