"""
FastAPI Main Application
AI QA Agent - Enhanced Sprint 1.4
"""
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
import time

from src.core.config import get_settings
from src.core.logging import get_logger, setup_logging
from src.core.exceptions import QAAgentError
from src.api.routes import health, analysis, chat

# Initialize logging
setup_logging()
logger = get_logger(__name__)

# Get settings
settings = get_settings()

# Create FastAPI app
app = FastAPI(
    title="AI QA Agent API",
    description="Intelligent test generation and code analysis with conversational AI",
    version="1.4.0",
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Error handlers
@app.exception_handler(QAAgentError)
async def qa_agent_exception_handler(request: Request, exc: QAAgentError):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "error_code": exc.error_code}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

# Include routers
app.include_router(health.router)
app.include_router(analysis.router)
app.include_router(chat.router)

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "AI QA Agent API",
        "version": "1.4.0",
        "features": [
            "Code Analysis",
            "Pattern Detection", 
            "Conversational AI",
            "Background Tasks",
            "Real-time Updates"
        ],
        "docs": "/docs" if settings.debug else "Contact administrator"
    }

# Health check for load balancers
@app.get("/ping")
async def ping():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info"
    )

# Sprint 2.4 - Agent API Routes
from .routes.agent.conversation import router as agent_conversation_router

# Include agent routes
app.include_router(agent_conversation_router, tags=["Agent System"])

# Import web routes
from src.web.routes.agent_interface import router as web_router

# Include web router
app.include_router(web_router)

# Import analytics routes
from src.web.routes.analytics_routes import router as analytics_router

# Include analytics router
app.include_router(analytics_router)

# Import demo routes
from src.web.routes.demo_routes import router as demo_router

# Include demo router
app.include_router(demo_router)
