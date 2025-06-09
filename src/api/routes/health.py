"""
AI QA Agent - Health Check and Monitoring Endpoints
Comprehensive health monitoring with system metrics and database connectivity.
"""

import time
import logging
import psutil
import platform
from typing import Dict, Any, List
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import text

from src.core.database import get_db, db_manager, AnalysisSession, GeneratedTest, TaskStatus
from src.core.config import get_settings, Settings
from src.core.logging import get_logger, PerformanceTimer
from src.core.exceptions import DatabaseError, ConfigurationError

router = APIRouter()
logger = get_logger(__name__)

# Track startup time for uptime calculation
startup_time = time.time()


@router.get("/")
async def health_check(
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings)
) -> Dict[str, Any]:
    """Basic health check endpoint."""
    try:
        with PerformanceTimer("health_check_basic", logger, min_duration_ms=100):
            # Quick database connectivity check
            try:
                db.execute(text("SELECT 1"))
                database_status = "connected"
            except Exception as e:
                logger.error(f"Database health check failed: {e}")
                database_status = "disconnected"
            
            # Determine overall status
            overall_status = "healthy" if database_status == "connected" else "degraded"
            
            response = {
                "status": overall_status,
                "timestamp": time.time(),
                "environment": settings.environment,
                "version": "1.0.0",
                "database": database_status
            }
            
            logger.info(f"Health check completed - Status: {overall_status}")
            return response
            
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Health check failed"
        )


@router.get("/detailed")
async def detailed_health_check(
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings)
) -> Dict[str, Any]:
    """Detailed health check with comprehensive system metrics."""
    start_time = time.time()
    
    try:
        with PerformanceTimer("health_check_detailed", logger, min_duration_ms=200):
            # Database health with timing
            db_start_time = time.time()
            try:
                db.execute(text("SELECT 1"))
                db_response_time = (time.time() - db_start_time) * 1000
                database_health = {
                    "status": "healthy",
                    "response_time_ms": round(db_response_time, 2)
                }
            except Exception as e:
                logger.error(f"Database health check failed: {e}")
                database_health = {
                    "status": "unhealthy",
                    "error": str(e)
                }
            
            # System metrics collection
            try:
                # CPU metrics
                cpu_percent = psutil.cpu_percent(interval=0.1)
                
                # Memory metrics
                memory = psutil.virtual_memory()
                
                # Disk metrics
                disk = psutil.disk_usage('/')
                
                system_metrics = {
                    "cpu_percent": round(cpu_percent, 1),
                    "memory_percent": round(memory.percent, 1),
                    "memory_used_gb": round(memory.used / (1024**3), 2),
                    "memory_total_gb": round(memory.total / (1024**3), 2),
                    "disk_percent": round(disk.percent, 1),
                    "disk_used_gb": round(disk.used / (1024**3), 2),
                    "disk_total_gb": round(disk.total / (1024**3), 2)
                }
                
            except Exception as e:
                logger.error(f"System metrics collection failed: {e}")
                system_metrics = {
                    "cpu_percent": 0,
                    "memory_percent": 0,
                    "memory_used_gb": 0,
                    "memory_total_gb": 0,
                    "disk_percent": 0,
                    "disk_used_gb": 0,
                    "disk_total_gb": 0
                }
            
            # AI provider status
            ai_config = settings.get_ai_config()
            ai_providers = {
                "openai": {
                    "available": ai_config["openai"]["available"],
                    "configured": bool(settings.openai_api_key),
                    "status": "ready" if ai_config["openai"]["available"] else "not_configured"
                },
                "anthropic": {
                    "available": ai_config["anthropic"]["available"],
                    "configured": bool(settings.anthropic_api_key),
                    "status": "ready" if ai_config["anthropic"]["available"] else "not_configured"
                },
                "default_provider": ai_config["default_provider"]
            }
            
            # Application configuration (safe subset)
            app_config = {
                "environment": settings.environment,
                "debug": settings.debug,
                "log_level": settings.log_level,
                "max_upload_size_mb": round(settings.max_upload_size / (1024*1024), 1),
                "supported_languages": settings.supported_languages,
                "max_files_per_repo": settings.max_files_per_repo,
                "analysis_timeout_seconds": settings.analysis_timeout,
                "ai_timeout_seconds": settings.ai_timeout,
                "max_concurrent_tasks": settings.max_concurrent_tasks
            }
            
            # Calculate response time and overall status
            response_time = round((time.time() - start_time) * 1000, 2)
            overall_status = "healthy" if database_health["status"] == "healthy" else "degraded"
            
            response = {
                "status": overall_status,
                "timestamp": time.time(),
                "response_time_ms": response_time,
                "version": "1.0.0",
                "environment": settings.environment,
                "database": database_health,
                "system": system_metrics,
                "ai_providers": ai_providers,
                "configuration": app_config
            }
            
            logger.info(f"Detailed health check completed - Status: {overall_status}, Response time: {response_time}ms")
            return response
            
    except Exception as e:
        logger.error(f"Detailed health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Detailed health check failed"
        )


@router.get("/ready")
async def readiness_check(
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings)
) -> Dict[str, Any]:
    """Kubernetes-style readiness probe."""
    try:
        with PerformanceTimer("readiness_check", logger):
            checks = []
            overall_ready = True
            
            # Database readiness
            try:
                db.execute(text("SELECT 1"))
                # Test a more complex query to ensure full database functionality
                db.execute(text("SELECT COUNT(*) FROM analysis_sessions LIMIT 1"))
                checks.append({
                    "component": "database",
                    "status": "ready"
                })
            except Exception as e:
                checks.append({
                    "component": "database",
                    "status": "not_ready",
                    "error": str(e)
                })
                overall_ready = False
            
            # AI provider readiness (at least one must be configured)
            ai_config = settings.get_ai_config()
            if ai_config["openai"]["available"] or ai_config["anthropic"]["available"]:
                checks.append({
                    "component": "ai_providers",
                    "status": "ready"
                })
            else:
                checks.append({
                    "component": "ai_providers",
                    "status": "not_ready",
                    "error": "No AI providers configured"
                })
                overall_ready = False
            
            # File system readiness
            try:
                import tempfile
                import os
                
                # Test temporary file creation
                with tempfile.NamedTemporaryFile(delete=True) as temp_file:
                    temp_file.write(b"readiness_test")
                    temp_file.flush()
                
                # Test logs directory access
                logs_dir = "logs"
                if not os.path.exists(logs_dir):
                    os.makedirs(logs_dir, exist_ok=True)
                
                checks.append({
                    "component": "filesystem",
                    "status": "ready"
                })
            except Exception as e:
                checks.append({
                    "component": "filesystem",
                    "status": "not_ready",
                    "error": str(e)
                })
                overall_ready = False
            
            # Configuration readiness
            try:
                if settings.secret_key and len(settings.secret_key) >= 32:
                    config_status = "ready"
                    config_error = None
                else:
                    config_status = "not_ready"
                    config_error = "Invalid secret key configuration"
                    overall_ready = False
                
                checks.append({
                    "component": "configuration",
                    "status": config_status,
                    "error": config_error
                })
            except Exception as e:
                checks.append({
                    "component": "configuration",
                    "status": "not_ready",
                    "error": str(e)
                })
                overall_ready = False
            
            response = {
                "ready": overall_ready,
                "timestamp": time.time(),
                "checks": checks
            }
            
            # Log readiness status
            if overall_ready:
                logger.info("Readiness check passed - Application ready to serve requests")
            else:
                failed_components = [check["component"] for check in checks if check["status"] == "not_ready"]
                logger.warning(f"Readiness check failed - Components not ready: {failed_components}")
            
            return response
            
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Readiness check failed"
        )


@router.get("/metrics")
async def metrics_endpoint(
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings)
) -> Dict[str, Any]:
    """Application metrics endpoint for monitoring systems."""
    try:
        with PerformanceTimer("metrics_collection", logger):
            # Business metrics from database
            try:
                # Analysis session metrics
                total_sessions = db.query(AnalysisSession).count()
                completed_sessions = db.query(AnalysisSession).filter(
                    AnalysisSession.status == "completed"
                ).count()
                
                # Test generation metrics
                total_tests = db.query(GeneratedTest).count()
                valid_tests = db.query(GeneratedTest).filter(
                    GeneratedTest.is_valid == True
                ).count()
                
                # Task metrics
                active_tasks = db.query(TaskStatus).filter(
                    TaskStatus.status.in_(["pending", "running"])
                ).count()
                
                # Calculate success rates
                session_success_rate = (completed_sessions / total_sessions * 100) if total_sessions > 0 else 0
                test_success_rate = (valid_tests / total_tests * 100) if total_tests > 0 else 0
                
                business_metrics = {
                    "total_analysis_sessions": total_sessions,
                    "completed_sessions": completed_sessions,
                    "session_success_rate": round(session_success_rate, 2),
                    "total_tests_generated": total_tests,
                    "valid_tests_generated": valid_tests,
                    "test_success_rate": round(test_success_rate, 2)
                }
                
            except Exception as e:
                logger.error(f"Business metrics collection failed: {e}")
                business_metrics = {
                    "total_analysis_sessions": 0,
                    "completed_sessions": 0,
                    "session_success_rate": 0,
                    "total_tests_generated": 0,
                    "valid_tests_generated": 0,
                    "test_success_rate": 0
                }
            
            # System metrics
            try:
                current_time = time.time()
                uptime_seconds = current_time - startup_time
                
                system_metrics = {
                    "uptime_seconds": round(uptime_seconds, 2),
                    "python_version": platform.python_version(),
                    "process_id": 1
                }
                
            except Exception as e:
                logger.error(f"System metrics collection failed: {e}")
                system_metrics = {
                    "uptime_seconds": 0,
                    "python_version": "unknown",
                    "process_id": 0
                }
            
            response = {
                "timestamp": time.time(),
                "business_metrics": business_metrics,
                "system_metrics": system_metrics
            }
            
            logger.info("Metrics collection completed successfully")
            return response
            
    except Exception as e:
        logger.error(f"Metrics collection failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Metrics collection failed"
        )


@router.get("/live")
async def liveness_check() -> Dict[str, Any]:
    """Kubernetes-style liveness probe."""
    return {
        "alive": True,
        "timestamp": time.time(),
        "uptime_seconds": round(time.time() - startup_time, 2)
    }


@router.get("/startup")
async def startup_check(
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Kubernetes-style startup probe."""
    try:
        # Check if critical components are initialized
        db.execute(text("SELECT 1"))
        
        return {
            "started": True,
            "timestamp": time.time(),
            "startup_time": startup_time,
            "initialization_duration_seconds": round(time.time() - startup_time, 2)
        }
    except Exception as e:
        logger.error(f"Startup check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Application not fully started"
        )
