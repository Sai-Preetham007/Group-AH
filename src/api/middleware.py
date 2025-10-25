"""
Middleware for Medical Knowledge RAG Chatbot API
"""
import time
import logging
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Callable
import uuid

logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Logging middleware for API requests"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate request ID
        request_id = str(uuid.uuid4())
        
        # Log request
        start_time = time.time()
        logger.info(f"Request {request_id}: {request.method} {request.url}")
        
        # Process request
        response = await call_next(request)
        
        # Log response
        process_time = time.time() - start_time
        logger.info(f"Request {request_id}: {response.status_code} - {process_time:.3f}s")
        
        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = str(process_time)
        
        return response


class SecurityMiddleware(BaseHTTPMiddleware):
    """Security middleware for medical API"""
    
    def __init__(self, app, max_request_size: int = 1024 * 1024):  # 1MB
        super().__init__(app)
        self.max_request_size = max_request_size
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Check request size
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.max_request_size:
            logger.warning(f"Request too large: {content_length} bytes")
            return Response(
                content="Request too large",
                status_code=413,
                headers={"X-Request-ID": str(uuid.uuid4())}
            )
        
        # Add security headers
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        
        return response


class MedicalSafetyMiddleware(BaseHTTPMiddleware):
    """Medical safety middleware for content validation"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Check for medical safety in request body
        if request.method in ["POST", "PUT", "PATCH"]:
            # This would implement medical content validation
            # For now, just pass through
            pass
        
        response = await call_next(request)
        
        # Add medical disclaimer to responses
        if hasattr(response, 'body') and response.body:
            # Add disclaimer header
            response.headers["X-Medical-Disclaimer"] = "This information is for educational purposes only"
        
        return response
