"""
Validation Models

Simple models for validation contexts and results.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime


class ValidationContext:
    """Context for validation operations"""
    def __init__(self, code: str, language: str = "python", validation_type: str = "all", 
                 user_preferences: Dict[str, Any] = None, agent_context: Dict[str, Any] = None,
                 project_context: Optional[Dict[str, Any]] = None):
        self.code = code
        self.language = language
        self.validation_type = validation_type
        self.user_preferences = user_preferences or {}
        self.agent_context = agent_context or {}
        self.project_context = project_context


class ValidationRequest:
    """Request for validation"""
    def __init__(self, code: str, language: Optional[str] = "python", validation_type: Optional[str] = "all",
                 context: Optional[Dict[str, Any]] = None, user_id: Optional[str] = None,
                 session_id: Optional[str] = None):
        self.code = code
        self.language = language
        self.validation_type = validation_type
        self.context = context
        self.user_id = user_id
        self.session_id = session_id


class ValidationResponse:
    """Response from validation tool"""
    def __init__(self, success: bool, validation_passed: bool, issues_count: int,
                 critical_issues_count: int, reasoning: str, confidence: float,
                 suggestions: List[str], duration: float, agent_message: str):
        self.success = success
        self.validation_passed = validation_passed
        self.issues_count = issues_count
        self.critical_issues_count = critical_issues_count
        self.reasoning = reasoning
        self.confidence = confidence
        self.suggestions = suggestions
        self.duration = duration
        self.agent_message = agent_message
