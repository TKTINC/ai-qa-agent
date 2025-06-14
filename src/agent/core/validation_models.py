"""
Validation Models for Agent Integration

Simple models for validation contexts and results that integrate with the agent system.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime

# Simple model classes without pydantic dependency
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
    """Request for validation from agents"""
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


class UserFeedback:
    """User feedback on validation results"""
    def __init__(self, validation_id: str, satisfaction_score: Optional[float] = None,
                 found_helpful: Optional[bool] = None, too_strict: Optional[bool] = None,
                 too_lenient: Optional[bool] = None, missed_issues: Optional[List[str]] = None,
                 false_positives: Optional[List[str]] = None, comments: Optional[str] = None,
                 user_id: Optional[str] = None):
        self.validation_id = validation_id
        self.satisfaction_score = satisfaction_score
        self.found_helpful = found_helpful
        self.too_strict = too_strict
        self.too_lenient = too_lenient
        self.missed_issues = missed_issues
        self.false_positives = false_positives
        self.comments = comments
        self.user_id = user_id


class LearningInsight:
    """Learning insight from validation outcomes"""
    def __init__(self, insight_type: str, description: str, confidence: float,
                 evidence_count: int, user_impact: str, recommendation: str):
        self.insight_type = insight_type
        self.description = description
        self.confidence = confidence
        self.evidence_count = evidence_count
        self.user_impact = user_impact
        self.recommendation = recommendation
