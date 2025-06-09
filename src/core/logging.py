"""
AI QA Agent - Logging Configuration System
Comprehensive logging setup with structured output and multiple handlers.
"""

import logging
import logging.config
import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import traceback

from .config import get_settings


class JsonFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "process": record.process,
            "thread": record.thread,
            "thread_name": record.threadName,
        }
        
        # Add exception information if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields from LoggerAdapter or extra parameter
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
        
        # Add common extra attributes
        extra_attrs = [
            'user_id', 'session_id', 'request_id', 'task_id', 
            'component_id', 'analysis_id', 'operation'
        ]
        for attr in extra_attrs:
            if hasattr(record, attr):
                log_entry[attr] = getattr(record, attr)
        
        return json.dumps(log_entry, default=str)


class ContextLoggerAdapter(logging.LoggerAdapter):
    """Logger adapter that adds contextual information to log records."""
    
    def __init__(self, logger: logging.Logger, extra: Dict[str, Any] = None):
        """Initialize with context information."""
        self.extra = extra or {}
        super().__init__(logger, self.extra)
    
    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple:
        """Add extra context to log record."""
        if 'extra' not in kwargs:
            kwargs['extra'] = {}
        kwargs['extra'].update(self.extra)
        kwargs['extra']['extra_fields'] = kwargs['extra']
        return msg, kwargs
    
    def update_context(self, **kwargs):
        """Update the context information."""
        self.extra.update(kwargs)


def setup_logging() -> None:
    """Configure application logging with multiple handlers and formatters."""
    settings = get_settings()
    
    # Create logs directory
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Define log levels
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
    
    # Logging configuration dictionary
    logging_config: Dict[str, Any] = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            },
            "detailed": {
                "format": (
                    "%(asctime)s - %(name)s - %(levelname)s - "
                    "%(funcName)s:%(lineno)d - %(message)s"
                ),
                "datefmt": "%Y-%m-%d %H:%M:%S"
            },
            "json": {
                "()": JsonFormatter,
            },
            "performance": {
                "format": (
                    "%(asctime)s - PERF - %(name)s - %(message)s - "
                    "%(duration)s ms"
                ),
                "datefmt": "%Y-%m-%d %H:%M:%S"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": log_level,
                "formatter": "detailed" if settings.debug else "default",
                "stream": sys.stdout
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "INFO",
                "formatter": "detailed",
                "filename": str(logs_dir / "app.log"),
                "mode": "a",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5
            },
            "json_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "INFO",
                "formatter": "json",
                "filename": str(logs_dir / "app.json"),
                "mode": "a",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5
            },
            "error_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "ERROR",
                "formatter": "detailed",
                "filename": str(logs_dir / "error.log"),
                "mode": "a",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 10
            }
        },
        "loggers": {
            "": {  # Root logger
                "level": log_level,
                "handlers": ["console", "file", "error_file"],
                "propagate": False
            },
            "ai_qa_agent": {  # Application logger
                "level": log_level,
                "handlers": ["console", "file", "json_file", "error_file"],
                "propagate": False
            },
            "uvicorn": {
                "level": "INFO",
                "handlers": ["console", "file"],
                "propagate": False
            },
            "sqlalchemy.engine": {
                "level": "WARNING" if not settings.debug else "INFO",
                "handlers": ["file"],
                "propagate": False
            }
        }
    }
    
    # Apply configuration
    logging.config.dictConfig(logging_config)
    
    # Log startup information
    logger = get_logger(__name__)
    logger.info(f"Logging configured - Level: {settings.log_level}")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Debug mode: {settings.debug}")
    logger.info(f"Log files location: {logs_dir.absolute()}")


def get_logger(name: str, context: Dict[str, Any] = None) -> ContextLoggerAdapter:
    """Get a context-aware logger for the specified module."""
    base_logger = logging.getLogger(f"ai_qa_agent.{name}")
    return ContextLoggerAdapter(base_logger, context or {})


class PerformanceTimer:
    """Context manager for measuring and logging operation performance."""
    
    def __init__(self, operation: str, logger: Optional[ContextLoggerAdapter] = None, 
                 min_duration_ms: float = 0):
        """Initialize performance timer."""
        self.operation = operation
        self.logger = logger or get_logger("performance")
        self.min_duration_ms = min_duration_ms
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        """Start timing."""
        import time
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End timing and log if needed."""
        import time
        self.end_time = time.perf_counter()
        duration_ms = (self.end_time - self.start_time) * 1000
        
        if duration_ms >= self.min_duration_ms:
            self.logger.info(
                f"Operation completed: {self.operation}",
                extra={"duration": f"{duration_ms:.2f}"}
            )
    
    @property
    def duration_ms(self) -> Optional[float]:
        """Get operation duration in milliseconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time) * 1000
        return None


def log_api_request(logger: ContextLoggerAdapter, method: str, path: str, 
                   status_code: int, duration_ms: float, user_id: str = None):
    """Log API request with standard format."""
    logger.info(
        f"{method} {path} - {status_code} ({duration_ms:.2f}ms)",
        extra={
            "request_method": method,
            "request_path": path,
            "status_code": status_code,
            "duration_ms": duration_ms,
            "user_id": user_id
        }
    )


# Initialize logging on import if not already configured
if not logging.getLogger().handlers:
    setup_logging()
