"""
AI QA Agent - Custom Exception Framework
Comprehensive exception hierarchy with error handling utilities.
"""

import logging
import traceback
from typing import Dict, Any, Optional, List
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from datetime import datetime


class QAAgentException(Exception):
    """Base exception for all AI QA Agent errors."""
    
    def __init__(
        self,
        message: str,
        error_code: str = None,
        details: Dict[str, Any] = None,
        cause: Exception = None,
        user_message: str = None
    ):
        """Initialize QA Agent exception."""
        self.message = message
        self.error_code = error_code or self.__class__.__name__.upper()
        self.details = details or {}
        self.cause = cause
        self.user_message = user_message or message
        self.timestamp = datetime.utcnow()
        
        super().__init__(message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API responses."""
        return {
            "error": self.__class__.__name__,
            "message": self.user_message,
            "error_code": self.error_code,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details
        }
    
    def __str__(self) -> str:
        """String representation with error code."""
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message


class ConfigurationError(QAAgentException):
    """Raised when there's a configuration issue."""
    pass


class DatabaseError(QAAgentException):
    """Raised when database operations fail."""
    pass


class AnalysisError(QAAgentException):
    """Raised when code analysis fails."""
    pass


class GenerationError(QAAgentException):
    """Raised when test generation fails."""
    pass


class ValidationError(QAAgentException):
    """Raised when test validation fails."""
    pass


class AIProviderError(QAAgentException):
    """Raised when AI provider communication fails."""
    
    def __init__(
        self,
        message: str,
        provider: str,
        model: str = None,
        error_code: str = None,
        details: Dict[str, Any] = None,
        cause: Exception = None,
        user_message: str = None
    ):
        """Initialize AI provider error."""
        self.provider = provider
        self.model = model
        
        details = details or {}
        details.update({
            "provider": provider,
            "model": model
        })
        
        super().__init__(
            message=message,
            error_code=error_code,
            details=details,
            cause=cause,
            user_message=user_message or f"AI service temporarily unavailable"
        )


class RepositoryError(QAAgentException):
    """Raised when repository processing fails."""
    pass


class TaskError(QAAgentException):
    """Raised when background task execution fails."""
    pass


class RateLimitError(QAAgentException):
    """Raised when rate limits are exceeded."""
    
    def __init__(
        self,
        message: str,
        service: str = None,
        retry_after: int = None,
        error_code: str = None,
        details: Dict[str, Any] = None,
        cause: Exception = None,
        user_message: str = None
    ):
        """Initialize rate limit error."""
        self.service = service
        self.retry_after = retry_after
        
        details = details or {}
        details.update({
            "service": service,
            "retry_after": retry_after
        })
        
        super().__init__(
            message=message,
            error_code=error_code,
            details=details,
            cause=cause,
            user_message=user_message or f"Rate limit exceeded. Please try again in {retry_after or 'a few'} seconds."
        )


class AuthenticationError(QAAgentException):
    """Raised when authentication fails."""
    pass


class TimeoutError(QAAgentException):
    """Raised when operations timeout."""
    pass


# Error handling utilities
class ErrorHandler:
    """Centralized error handling and reporting utilities."""
    
    @staticmethod
    def log_error(
        logger: logging.Logger,
        error: Exception,
        context: Dict[str, Any] = None,
        level: int = logging.ERROR
    ):
        """Log error with context and stack trace."""
        context = context or {}
        
        if isinstance(error, QAAgentException):
            logger.log(
                level,
                f"QA Agent Error: {error.message}",
                extra={
                    "error_code": error.error_code,
                    "error_type": error.__class__.__name__,
                    "details": error.details,
                    "timestamp": error.timestamp.isoformat(),
                    **context
                },
                exc_info=True
            )
        else:
            logger.log(
                level,
                f"Unexpected Error: {str(error)}",
                extra={
                    "error_type": error.__class__.__name__,
                    **context
                },
                exc_info=True
            )
    
    @staticmethod
    def create_error_response(
        error: Exception,
        status_code: int = 500,
        include_details: bool = False
    ) -> Dict[str, Any]:
        """Create standardized error response."""
        if isinstance(error, QAAgentException):
            response = error.to_dict()
        else:
            response = {
                "error": "InternalServerError",
                "message": "An unexpected error occurred",
                "error_code": "INTERNAL_ERROR",
                "timestamp": datetime.utcnow().isoformat(),
                "details": {}
            }
        
        response["status_code"] = status_code
        
        if not include_details:
            response.pop("details", None)
            if "traceback" in response:
                response.pop("traceback")
        
        return response


# FastAPI exception handlers
logger = logging.getLogger(__name__)


async def qa_agent_exception_handler(request: Request, exc: QAAgentException) -> JSONResponse:
    """Global exception handler for QA Agent exceptions."""
    from .logging import ErrorHandler
    
    ErrorHandler.log_error(
        logger,
        exc,
        context={
            "url": str(request.url),
            "method": request.method,
            "client_ip": request.client.host if request.client else None
        }
    )
    
    # Determine appropriate status code
    status_code_map = {
        ConfigurationError: 500,
        DatabaseError: 500,
        AnalysisError: 422,
        GenerationError: 422,
        ValidationError: 400,
        AIProviderError: 502,
        RepositoryError: 422,
        TaskError: 500,
        RateLimitError: 429,
        AuthenticationError: 401,
        TimeoutError: 408,
    }
    
    status_code = status_code_map.get(type(exc), 500)
    
    response_data = ErrorHandler.create_error_response(
        exc,
        status_code=status_code,
        include_details=False
    )
    
    return JSONResponse(
        status_code=status_code,
        content=response_data
    )


async def validation_exception_handler(request: Request, exc: ValidationError) -> JSONResponse:
    """Specific handler for validation errors."""
    from .logging import ErrorHandler
    
    ErrorHandler.log_error(
        logger,
        exc,
        context={
            "url": str(request.url),
            "method": request.method
        },
        level=logging.WARNING
    )
    
    response_data = ErrorHandler.create_error_response(
        exc,
        status_code=400,
        include_details=True
    )
    
    return JSONResponse(
        status_code=400,
        content=response_data
    )


async def rate_limit_exception_handler(request: Request, exc: RateLimitError) -> JSONResponse:
    """Specific handler for rate limit errors."""
    from .logging import ErrorHandler
    
    ErrorHandler.log_error(
        logger,
        exc,
        context={
            "url": str(request.url),
            "method": request.method
        },
        level=logging.WARNING
    )
    
    response_data = ErrorHandler.create_error_response(
        exc,
        status_code=429,
        include_details=False
    )
    
    headers = {}
    if exc.retry_after:
        headers["Retry-After"] = str(exc.retry_after)
    
    return JSONResponse(
        status_code=429,
        content=response_data,
        headers=headers
    )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handler for unexpected exceptions."""
    from .logging import ErrorHandler
    
    ErrorHandler.log_error(
        logger,
        exc,
        context={
            "url": str(request.url),
            "method": request.method,
            "client_ip": request.client.host if request.client else None
        }
    )
    
    response_data = {
        "error": "InternalServerError",
        "message": "An unexpected error occurred",
        "error_code": "INTERNAL_ERROR",
        "timestamp": datetime.utcnow().isoformat(),
        "status_code": 500
    }
    
    return JSONResponse(
        status_code=500,
        content=response_data
    )
