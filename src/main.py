"""
AI QA Agent - Main Application Entry Point
Imports and runs the FastAPI application from the API module.
"""

from src.api.main import app

if __name__ == "__main__":
    import uvicorn
    from src.core.config import get_settings
    
    settings = get_settings()
    
    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level=settings.log_level.lower(),
        workers=1 if settings.reload else settings.workers
    )
