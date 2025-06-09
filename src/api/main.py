"""
AI QA Agent - FastAPI Application with Organized Routing Structure
Comprehensive API setup with health monitoring, error handling, and modular routing.
"""

import logging
from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import HTMLResponse
import time
from pathlib import Path

from src.core.config import get_settings, Settings
from src.core.logging import get_logger, setup_logging, PerformanceTimer, log_api_request
from src.core.database import get_db, db_manager
from src.core.exceptions import (
    QAAgentException,
    ValidationError,
    RateLimitError,
    qa_agent_exception_handler,
    validation_exception_handler,
    rate_limit_exception_handler,
    general_exception_handler
)

# Import routers
from .routes.health import router as health_router
from .routes.analysis import router as analysis_router
from .routes.generation import router as generation_router

# Setup logging
setup_logging()
logger = get_logger(__name__)


class RequestLoggingMiddleware:
    """Middleware to log all API requests with timing and context."""
    
    def __init__(self, app: FastAPI):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        """Process request with logging."""
        if scope["type"] == "http":
            request = Request(scope, receive)
            start_time = time.time()
            
            # Extract request info
            method = request.method
            path = request.url.path
            client_ip = request.client.host if request.client else "unknown"
            
            # Process request
            async def send_wrapper(message):
                if message["type"] == "http.response.start":
                    # Log request completion
                    duration_ms = (time.time() - start_time) * 1000
                    status_code = message["status"]
                    
                    # Log with context
                    log_api_request(
                        logger=logger,
                        method=method,
                        path=path,
                        status_code=status_code,
                        duration_ms=duration_ms,
                        user_id=None
                    )
                
                await send(message)
            
            await self.app(scope, receive, send_wrapper)
        else:
            await self.app(scope, receive, send)


def create_application() -> FastAPI:
    """Create and configure the FastAPI application with all components."""
    settings = get_settings()
    
    # Create FastAPI app with comprehensive configuration
    app = FastAPI(
        title="AI QA Agent API",
        description="""
        ## Intelligent Test Generation API
        
        The AI QA Agent automatically analyzes codebases and generates comprehensive test suites using advanced AI and code analysis techniques.
        
        ### Key Features
        - **Code Analysis**: AST parsing and complexity analysis
        - **AI Test Generation**: Multiple AI providers (OpenAI, Anthropic)
        - **Test Validation**: Syntax and execution validation
        - **Real-time Progress**: WebSocket-based progress tracking
        - **Quality Scoring**: Comprehensive test quality assessment
        
        ### Getting Started
        1. Use `/api/v1/analysis/repository` to analyze a codebase
        2. Monitor progress with `/api/v1/analysis/{session_id}/status`
        3. Generate tests with `/api/v1/generation/generate`
        4. Validate results with `/api/v1/validation/validate`
        
        ### Support
        - **Health Checks**: `/health/*` endpoints for monitoring
        - **Documentation**: This interactive documentation
        - **Status**: Check system status and configuration
        """,
        version="1.0.0",
        contact={
            "name": "AI QA Agent",
            "url": "https://github.com/yourusername/ai-qa-agent",
        },
        license_info={
            "name": "MIT",
            "url": "https://opensource.org/licenses/MIT",
        },
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
        openapi_url="/openapi.json" if settings.debug else None,
        servers=[
            {
                "url": f"http://{settings.host}:{settings.port}",
                "description": f"Development server ({settings.environment})"
            }
        ] if settings.debug else []
    )
    
    # Add security middleware
    if settings.is_production():
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=settings.allowed_hosts
        )
    
    # Add compression middleware
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Add CORS middleware
    cors_config = settings.get_cors_config()
    app.add_middleware(CORSMiddleware, **cors_config)
    
    # Add request logging middleware
    app.middleware("http")(RequestLoggingMiddleware(app))
    
    # Add exception handlers
    app.add_exception_handler(QAAgentException, qa_agent_exception_handler)
    app.add_exception_handler(ValidationError, validation_exception_handler)
    app.add_exception_handler(RateLimitError, rate_limit_exception_handler)
    app.add_exception_handler(Exception, general_exception_handler)
    
    # Setup static files and templates
    static_dir = Path("src/web/static")
    template_dir = Path("src/web/templates")
    
    # Create directories if they don't exist
    static_dir.mkdir(parents=True, exist_ok=True)
    template_dir.mkdir(parents=True, exist_ok=True)
    
    # Mount static files
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    
    # Include routers with comprehensive configuration
    app.include_router(
        health_router,
        prefix="/health",
        tags=["Health & Monitoring"],
        responses={
            200: {"description": "Service is healthy"},
            503: {"description": "Service is unhealthy"},
            429: {"description": "Too many requests"}
        }
    )
    
    app.include_router(
        analysis_router,
        prefix="/api/v1/analysis",
        tags=["Code Analysis"],
        responses={
            400: {"description": "Invalid request"},
            422: {"description": "Analysis failed"},
            429: {"description": "Rate limit exceeded"}
        }
    )
    
    app.include_router(
        generation_router,
        prefix="/api/v1/generation",
        tags=["Test Generation"],
        responses={
            400: {"description": "Invalid request"},
            422: {"description": "Generation failed"},
            502: {"description": "AI provider error"}
        }
    )
    
    # Root endpoint with enhanced landing page
    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def root(request: Request):
        """Enhanced landing page with system information."""
        settings = get_settings()
        
        # Get system status
        try:
            db_healthy = db_manager.health_check()
            ai_config = settings.get_ai_config()
            
            system_info = {
                "environment": settings.environment,
                "debug": settings.debug,
                "database_status": "connected" if db_healthy else "disconnected",
                "ai_providers": {
                    "openai": "configured" if ai_config["openai"]["available"] else "not configured",
                    "anthropic": "configured" if ai_config["anthropic"]["available"] else "not configured"
                },
                "version": "1.0.0"
            }
        except Exception as e:
            logger.error(f"Error getting system info: {e}")
            system_info = {"error": "Unable to load system information"}
        
        # Load and return the enhanced HTML template
        with open("src/web/templates/index.html", "r") as f:
            return f.read()
    
    logger.info("FastAPI application created and configured successfully")
    return app


# Create the application instance
app = create_application()


@app.on_event("startup")
async def startup_event():
    """Execute on application startup."""
    logger.info("🚀 AI QA Agent API starting up...")
    
    try:
        # Initialize database tables
        with PerformanceTimer("database_initialization", logger):
            db_manager.create_tables()
            
        if db_manager.health_check():
            logger.info("✅ Database connection verified")
        else:
            logger.warning("⚠️ Database health check failed")
        
        # Validate AI configuration
        settings = get_settings()
        try:
            settings.validate_ai_config()
            logger.info("✅ AI configuration validated")
        except Exception as e:
            logger.warning(f"⚠️ AI configuration validation failed: {e}")
        
        logger.info("🎉 API startup completed successfully")
        
    except Exception as e:
        logger.error(f"❌ API startup failed: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Execute on application shutdown."""
    logger.info("🛑 AI QA Agent API shutting down...")
    logger.info("✅ API shutdown completed successfully")


if __name__ == "__main__":
    import uvicorn
    
    settings = get_settings()
    logger.info(f"🚀 Starting AI QA Agent API server...")
    
    uvicorn.run(
        "src.api.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level=settings.log_level.lower(),
        workers=1 if settings.reload else settings.workers,
        access_log=True
    )
