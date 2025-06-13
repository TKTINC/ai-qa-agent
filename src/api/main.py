"""
FastAPI Application - AI QA Agent
Main application with comprehensive middleware and routing
"""

import time
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
from contextlib import asynccontextmanager

from src.core.config import get_settings
from src.core.database import init_db
from src.core.logging import get_logger, log_request_response
from src.core.exceptions import QAAgentException
from src.api.routes import health, analysis, generation

logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    logger.info("🚀 AI QA Agent starting up...")
    
    # Initialize database
    try:
        init_db()
        logger.info("✅ Database initialized successfully")
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        raise
    
    logger.info("🎉 AI QA Agent startup complete!")
    
    yield
    
    # Shutdown
    logger.info("🛑 AI QA Agent shutting down...")
    logger.info("👋 AI QA Agent shutdown complete!")

def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    settings = get_settings()
    
    app = FastAPI(
        title="AI QA Agent",
        description="Intelligent test generation system with advanced code analysis",
        version="1.0.0",
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
        lifespan=lifespan
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Request logging middleware
    @app.middleware("http")
    async def logging_middleware(request: Request, call_next):
        start_time = time.time()
        
        # Log request
        logger.info(f"📥 {request.method} {request.url.path}")
        
        try:
            response = await call_next(request)
            
            # Log response
            process_time = time.time() - start_time
            logger.info(
                f"📤 {request.method} {request.url.path} "
                f"-> {response.status_code} ({process_time:.3f}s)"
            )
            
            return response
            
        except Exception as e:
            process_time = time.time() - start_time
            logger.error(
                f"💥 {request.method} {request.url.path} "
                f"-> ERROR ({process_time:.3f}s): {str(e)}"
            )
            raise
    
    # Global exception handlers
    @app.exception_handler(QAAgentException)
    async def qa_agent_exception_handler(request: Request, exc: QAAgentException):
        logger.error(f"QA Agent error: {exc.message}")
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.message,
                "error_code": exc.error_code,
                "detail": exc.detail
            }
        )
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        logger.warning(f"Validation error: {exc.errors()}")
        return JSONResponse(
            status_code=422,
            content={
                "error": "Validation error",
                "details": exc.errors()
            }
        )
    
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        logger.warning(f"HTTP exception: {exc.status_code} - {exc.detail}")
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.detail}
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unexpected error: {str(exc)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "detail": "An unexpected error occurred"
            }
        )
    
    # Include routers
    app.include_router(health.router)
    app.include_router(analysis.router)
    app.include_router(generation.router)
    
    # Enhanced root endpoint with interactive dashboard
    @app.get("/", response_class=HTMLResponse)
    async def root():
        """Interactive dashboard for AI QA Agent"""
        return HTMLResponse(content=f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>AI QA Agent - Intelligent Test Generation</title>
            <script src="https://cdn.tailwindcss.com"></script>
            <script src="https://unpkg.com/htmx.org@1.9.10"></script>
            <style>
                .status-indicator {{
                    animation: pulse 2s infinite;
                }}
                @keyframes pulse {{
                    0%, 100% {{ opacity: 1; }}
                    50% {{ opacity: 0.5; }}
                }}
                .gradient-bg {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                }}
            </style>
        </head>
        <body class="bg-gray-50">
            <!-- Header -->
            <div class="gradient-bg text-white p-6">
                <div class="max-w-6xl mx-auto">
                    <h1 class="text-4xl font-bold mb-2">🤖 AI QA Agent</h1>
                    <p class="text-xl opacity-90">Intelligent Test Generation with Advanced Code Analysis</p>
                    <div class="mt-4 text-sm opacity-75">
                        Sprint 1.4: Analysis API Integration • Production Ready
                    </div>
                </div>
            </div>
            
            <!-- Main Content -->
            <div class="max-w-6xl mx-auto p-6">
                <!-- System Status -->
                <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
                    <div class="bg-white rounded-lg shadow-md p-6">
                        <h3 class="text-lg font-semibold mb-4 flex items-center">
                            <span class="w-3 h-3 bg-green-500 rounded-full mr-2 status-indicator"></span>
                            System Health
                        </h3>
                        <div id="health-status" hx-get="/health/detailed" hx-trigger="load, every 30s" 
                             class="text-sm text-gray-600">
                            Loading...
                        </div>
                    </div>
                    
                    <div class="bg-white rounded-lg shadow-md p-6">
                        <h3 class="text-lg font-semibold mb-4 flex items-center">
                            <span class="w-3 h-3 bg-blue-500 rounded-full mr-2 status-indicator"></span>
                            Analysis Engine
                        </h3>
                        <div id="analysis-status" hx-get="/api/v1/analysis/status" hx-trigger="load, every 30s"
                             class="text-sm text-gray-600">
                            Loading...
                        </div>
                    </div>
                    
                    <div class="bg-white rounded-lg shadow-md p-6">
                        <h3 class="text-lg font-semibold mb-4 flex items-center">
                            <span class="w-3 h-3 bg-purple-500 rounded-full mr-2 status-indicator"></span>
                            AI Generation
                        </h3>
                        <div id="generation-status" hx-get="/api/v1/generation/status" hx-trigger="load, every 30s"
                             class="text-sm text-gray-600">
                            Loading...
                        </div>
                    </div>
                </div>
                
                <!-- Analysis Capabilities -->
                <div class="bg-white rounded-lg shadow-md p-6 mb-8">
                    <h2 class="text-2xl font-bold mb-6 text-center">🔍 Advanced Code Analysis Capabilities</h2>
                    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                        <div class="text-center p-4">
                            <div class="text-3xl mb-2">🌳</div>
                            <h3 class="font-semibold mb-2">AST Parsing</h3>
                            <p class="text-sm text-gray-600">Multi-language syntax tree analysis with complexity metrics</p>
                        </div>
                        <div class="text-center p-4">
                            <div class="text-3xl mb-2">📊</div>
                            <h3 class="font-semibold mb-2">Repository Analysis</h3>
                            <p class="text-sm text-gray-600">Project-wide structure and architecture pattern detection</p>
                        </div>
                        <div class="text-center p-4">
                            <div class="text-3xl mb-2">🧠</div>
                            <h3 class="font-semibold mb-2">ML Pattern Detection</h3>
                            <p class="text-sm text-gray-600">AI-powered clustering, anomaly detection, and design patterns</p>
                        </div>
                        <div class="text-center p-4">
                            <div class="text-3xl mb-2">🕸️</div>
                            <h3 class="font-semibold mb-2">Graph Analysis</h3>
                            <p class="text-sm text-gray-600">Dependency graphs, centrality analysis, and architectural insights</p>
                        </div>
                    </div>
                </div>
                
                <!-- API Endpoints -->
                <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
                    <div class="bg-white rounded-lg shadow-md p-6">
                        <h3 class="text-xl font-semibold mb-4">🔧 Analysis API Endpoints</h3>
                        <div class="space-y-2 text-sm">
                            <div class="flex items-center">
                                <span class="bg-green-100 text-green-800 px-2 py-1 rounded mr-2">POST</span>
                                <code>/api/v1/analysis/repository</code>
                            </div>
                            <div class="flex items-center">
                                <span class="bg-green-100 text-green-800 px-2 py-1 rounded mr-2">POST</span>
                                <code>/api/v1/analysis/upload</code>
                            </div>
                            <div class="flex items-center">
                                <span class="bg-blue-100 text-blue-800 px-2 py-1 rounded mr-2">GET</span>
                                <code>/api/v1/analysis/sessions</code>
                            </div>
                            <div class="flex items-center">
                                <span class="bg-blue-100 text-blue-800 px-2 py-1 rounded mr-2">GET</span>
                                <code>/api/v1/analysis/sessions/{{id}}/components</code>
                            </div>
                            <div class="flex items-center">
                                <span class="bg-blue-100 text-blue-800 px-2 py-1 rounded mr-2">GET</span>
                                <code>/api/v1/analysis/sessions/{{id}}/patterns</code>
                            </div>
                            <div class="flex items-center">
                                <span class="bg-blue-100 text-blue-800 px-2 py-1 rounded mr-2">GET</span>
                                <code>/api/v1/analysis/sessions/{{id}}/quality</code>
                            </div>
                        </div>
                    </div>
                    
                    <div class="bg-white rounded-lg shadow-md p-6">
                        <h3 class="text-xl font-semibold mb-4">📊 Analysis Features</h3>
                        <ul class="space-y-2 text-sm">
                            <li class="flex items-center">
                                <span class="w-2 h-2 bg-green-500 rounded-full mr-2"></span>
                                Real-time progress tracking with background tasks
                            </li>
                            <li class="flex items-center">
                                <span class="w-2 h-2 bg-green-500 rounded-full mr-2"></span>
                                Archive upload support (.zip, .tar.gz, .tar)
                            </li>
                            <li class="flex items-center">
                                <span class="w-2 h-2 bg-green-500 rounded-full
