# src/api/middleware.py
# HackRx 6.0 - API Middleware for Authentication and Logging

import time
from typing import Callable
from fastapi import Request, Response, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from ..utils.config import get_settings
from ..utils.logger import get_logger, log_api_request

logger = get_logger(__name__)


class AuthMiddleware(BaseHTTPMiddleware):
    """
    Authentication middleware to validate Bearer tokens
    """
    
    def __init__(self, app):
        super().__init__(app)
        self.settings = get_settings()
        self.protected_paths = ["/hackrx/run"]  # Paths that require authentication
        self.public_paths = ["/health", "/", "/docs", "/redoc", "/openapi.json"]
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request and validate authentication if required
        
        Args:
            request: Incoming HTTP request
            call_next: Next middleware/handler in chain
        
        Returns:
            HTTP response
        """
        path = request.url.path
        
        # Skip authentication for public paths
        if path in self.public_paths or path.startswith("/docs") or path.startswith("/redoc"):
            return await call_next(request)
        
        # Check if path requires authentication
        if any(path.startswith(protected) for protected in self.protected_paths):
            auth_header = request.headers.get("Authorization")
            
            if not auth_header:
                logger.warning(f"Missing Authorization header for {path}")
                return JSONResponse(
                    status_code=401,
                    content={"error": "Authorization header required", "status_code": 401}
                )
            
            # Validate Bearer token format
            if not auth_header.startswith("Bearer "):
                logger.warning(f"Invalid Authorization format for {path}")
                return JSONResponse(
                    status_code=401,
                    content={"error": "Invalid authorization format. Use 'Bearer <token>'", "status_code": 401}
                )
            
            # Extract and validate token
            token = auth_header[7:]  # Remove "Bearer " prefix
            
            if token != self.settings.auth_token:
                logger.warning(f"Invalid token for {path}: {token[:10]}...")
                return JSONResponse(
                    status_code=401,  
                    content={"error": "Invalid authorization token", "status_code": 401}
                )
            
            logger.debug(f"Authentication successful for {path}")
        
        # Continue to next middleware/handler
        return await call_next(request)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Request logging middleware to track API usage and performance
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Log request details and response metrics
        
        Args:
            request: Incoming HTTP request  
            call_next: Next middleware/handler in chain
        
        Returns:
            HTTP response with timing headers
        """
        start_time = time.time()
        
        # Extract request details
        method = request.method
        path = request.url.path
        query_params = str(request.query_params) if request.query_params else ""
        user_agent = request.headers.get("user-agent", "")
        client_ip = self._get_client_ip(request)
        
        # Log request start (for long-running requests)
        if path.startswith("/hackrx"):
            logger.info(f"🔄 {method} {path} started from {client_ip}")
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate response time
            response_time = time.time() - start_time
            
            # Add timing header to response
            response.headers["X-Response-Time"] = f"{response_time:.3f}s"
            response.headers["X-Request-ID"] = str(hash(f"{start_time}{client_ip}{path}"))
            
            # Log API request
            log_api_request(
                endpoint=path,
                method=method,
                status_code=response.status_code,
                response_time=response_time,
                user_agent=user_agent if len(user_agent) < 100 else user_agent[:100] + "..."
            )
            
            # Additional logging for main endpoint
            if path.startswith("/hackrx"):
                if response.status_code == 200:
                    logger.info(f"✅ {method} {path} completed successfully in {response_time:.3f}s")
                else:
                    logger.error(f"❌ {method} {path} failed with status {response.status_code} in {response_time:.3f}s")
            
            return response
            
        except Exception as e:
            response_time = time.time() - start_time
            
            logger.error(f"❌ {method} {path} failed after {response_time:.3f}s: {e}")
            
            # Return error response
            return JSONResponse(
                status_code=500,
                content={"error": "Internal server error", "status_code": 500},
                headers={
                    "X-Response-Time": f"{response_time:.3f}s",
                    "X-Request-ID": str(hash(f"{start_time}{client_ip}{path}"))
                }
            )
    
    def _get_client_ip(self, request: Request) -> str:
        """
        Extract client IP address from request
        
        Args:
            request: HTTP request
        
        Returns:
            Client IP address
        """
        # Check for forwarded headers (from reverse proxy)
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        # Fallback to direct client
        if hasattr(request, "client") and request.client:
            return request.client.host
        
        return "unknown"


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Simple rate limiting middleware (optional for production)
    """
    
    def __init__(self, app, max_requests: int = 100, time_window: int = 60):
        super().__init__(app)
        self.max_requests = max_requests
        self.time_window = time_window
        self.client_requests = {}  # {client_ip: [timestamp, ...]}
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Check rate limits before processing request
        
        Args:
            request: Incoming HTTP request
            call_next: Next middleware/handler in chain
        
        Returns:
            HTTP response or rate limit error
        """
        client_ip = self._get_client_ip(request)
        current_time = time.time()
        
        # Clean old requests
        if client_ip in self.client_requests:
            self.client_requests[client_ip] = [
                timestamp for timestamp in self.client_requests[client_ip]
                if current_time - timestamp < self.time_window
            ]
        else:
            self.client_requests[client_ip] = []
        
        # Check rate limit
        if len(self.client_requests[client_ip]) >= self.max_requests:
            logger.warning(f"Rate limit exceeded for {client_ip}")
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "status_code": 429,
                    "retry_after": self.time_window
                },
                headers={"Retry-After": str(self.time_window)}
            )
        
        # Record this request
        self.client_requests[client_ip].append(current_time)
        
        return await call_next(request)
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request"""
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        if hasattr(request, "client") and request.client:
            return request.client.host
        
        return "unknown"