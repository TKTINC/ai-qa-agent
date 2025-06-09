"""
AI QA Agent - Response Models for API Endpoints
Comprehensive Pydantic models for all API responses with validation and documentation.
"""

import time
from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from enum import Enum


class HealthStatus(str, Enum):
    """Health status enumeration."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class ComponentStatus(str, Enum):
    """Component readiness status."""
    READY = "ready"
    NOT_READY = "not_ready"


class TaskStatus(str, Enum):
    """Background task status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# Base response classes
class BaseResponse(BaseModel):
    """Base response model with common fields."""
    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat(),
        },
        use_enum_values=True
    )
    
    timestamp: float = Field(
        default_factory=time.time,
        description="Response timestamp in Unix epoch seconds"
    )


# Health check response models
class HealthCheckResponse(BaseResponse):
    """Basic health check response."""
    status: HealthStatus = Field(description="Overall health status")
    environment: str = Field(description="Application environment")
    version: str = Field(description="Application version")
    database: str = Field(description="Database connection status")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": 1686123456.789,
                "environment": "development",
                "version": "1.0.0",
                "database": "connected"
            }
        }


class SystemMetrics(BaseModel):
    """System resource metrics."""
    cpu_percent: float = Field(ge=0, le=100, description="CPU usage percentage")
    memory_percent: float = Field(ge=0, le=100, description="Memory usage percentage")
    memory_used_gb: float = Field(ge=0, description="Used memory in GB")
    memory_total_gb: float = Field(ge=0, description="Total memory in GB")
    disk_percent: float = Field(ge=0, le=100, description="Disk usage percentage")
    disk_used_gb: float = Field(ge=0, description="Used disk space in GB")
    disk_total_gb: float = Field(ge=0, description="Total disk space in GB")


class DatabaseHealth(BaseModel):
    """Database health information."""
    status: str = Field(description="Database status")
    response_time_ms: Optional[float] = Field(
        None, 
        ge=0, 
        description="Database response time in milliseconds"
    )
    error: Optional[str] = Field(None, description="Error message if unhealthy")


class DetailedHealthResponse(BaseResponse):
    """Detailed health check response with comprehensive metrics."""
    status: HealthStatus = Field(description="Overall health status")
    response_time_ms: float = Field(ge=0, description="Health check response time")
    version: str = Field(description="Application version")
    environment: str = Field(description="Application environment")
    database: DatabaseHealth = Field(description="Database health details")
    system: SystemMetrics = Field(description="System resource metrics")
    ai_providers: Dict[str, Any] = Field(description="AI provider availability")
    configuration: Dict[str, Any] = Field(description="Application configuration summary")


class ReadinessCheck(BaseModel):
    """Individual component readiness check."""
    component: str = Field(description="Component name")
    status: ComponentStatus = Field(description="Component readiness status")
    error: Optional[str] = Field(None, description="Error message if not ready")


class ReadinessResponse(BaseResponse):
    """Kubernetes-style readiness probe response."""
    ready: bool = Field(description="Overall readiness status")
    checks: List[ReadinessCheck] = Field(description="Individual component checks")


class BusinessMetrics(BaseModel):
    """Business-related metrics."""
    total_analysis_sessions: int = Field(ge=0, description="Total analysis sessions")
    completed_sessions: int = Field(ge=0, description="Successfully completed sessions")
    session_success_rate: float = Field(ge=0, le=100, description="Session success rate percentage")
    total_tests_generated: int = Field(ge=0, description="Total tests generated")
    valid_tests_generated: int = Field(ge=0, description="Valid tests generated")
    test_success_rate: float = Field(ge=0, le=100, description="Test validation success rate")


class SystemMetricsBasic(BaseModel):
    """Basic system metrics for metrics endpoint."""
    uptime_seconds: float = Field(ge=0, description="Application uptime in seconds")
    python_version: str = Field(description="Python version")
    process_id: int = Field(description="Process ID")


class MetricsResponse(BaseResponse):
    """Application metrics response for monitoring systems."""
    business_metrics: BusinessMetrics = Field(description="Business performance metrics")
    system_metrics: SystemMetricsBasic = Field(description="System performance metrics")


# Error response models
class ErrorResponse(BaseResponse):
    """Standard error response."""
    error: str = Field(description="Error type")
    message: str = Field(description="Human-readable error message")
    error_code: Optional[str] = Field(None, description="Application-specific error code")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")


# Analysis response models
class AnalysisSessionResponse(BaseResponse):
    """Analysis session response."""
    id: str = Field(description="Session identifier")
    repository_name: str = Field(description="Repository name")
    repository_url: Optional[str] = Field(None, description="Repository URL")
    total_files: int = Field(ge=0, description="Total files in repository")
    analyzed_files: int = Field(ge=0, description="Files successfully analyzed")
    language_distribution: Dict[str, int] = Field(description="Programming language distribution")
    status: str = Field(description="Analysis status")
    progress_percentage: int = Field(ge=0, le=100, description="Analysis progress")
    created_at: datetime = Field(description="Session creation time")
    completed_at: Optional[datetime] = Field(None, description="Session completion time")


class CodeComponentResponse(BaseResponse):
    """Code component response."""
    id: str = Field(description="Component identifier")
    name: str = Field(description="Component name")
    component_type: str = Field(description="Component type (function, class, method)")
    file_path: str = Field(description="File path")
    complexity: int = Field(ge=1, description="Cyclomatic complexity")
    lines_of_code: int = Field(ge=0, description="Lines of code")
    is_testable: bool = Field(description="Whether component is suitable for testing")


class GeneratedTestResponse(BaseResponse):
    """Generated test response."""
    id: str = Field(description="Test identifier")
    test_name: str = Field(description="Test function name")
    test_type: str = Field(description="Test type (unit, integration, etc.)")
    confidence_score: float = Field(ge=0, le=1, description="AI confidence score")
    quality_score: float = Field(ge=0, le=1, description="Test quality score")
    is_valid: bool = Field(description="Whether test passes syntax validation")
    is_executable: bool = Field(description="Whether test can be executed")
    explanation: Optional[str] = Field(None, description="AI explanation of test purpose")
