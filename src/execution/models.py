"""
Execution Models

Pydantic models for execution contexts, results, and monitoring data.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class ExecutionContext(BaseModel):
    """Context for test execution"""
    language: str = Field(..., description="Programming language")
    framework: Optional[str] = Field(None, description="Testing framework")
    tests: str = Field(..., description="Test code to execute")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context")


class ExecutionRequest(BaseModel):
    """Request for test execution"""
    tests: str
    language: Optional[str] = "python"
    framework: Optional[str] = None
    dependencies: Optional[List[str]] = None
    timeout: Optional[int] = 120
    context: Optional[Dict[str, Any]] = None


class ExecutionResponse(BaseModel):
    """Response from test execution"""
    success: bool
    total_tests: int
    passed_tests: int
    failed_tests: int
    execution_time: float
    interpretation: str
    insights: List[str]
    recommendations: List[str]
    monitoring_data: Dict[str, Any]


class ExecutionFeedback(BaseModel):
    """Feedback on execution results"""
    execution_id: str
    user_satisfaction: Optional[float] = Field(None, ge=0.0, le=1.0)
    results_helpful: Optional[bool] = None
    interpretation_accurate: Optional[bool] = None
    performance_acceptable: Optional[bool] = None
    suggestions_valuable: Optional[bool] = None
    comments: Optional[str] = None
    user_id: Optional[str] = None
