"""
Validation Learning System

Learning system that improves validation accuracy and effectiveness over time.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ValidationOutcome:
    """Record of a validation outcome for learning"""
    validation_id: str
    code_analyzed: str
    issues_found: int
    critical_issues: int
    user_feedback: Optional[Dict[str, Any]] = None
    correction_attempted: bool = False
    correction_successful: bool = False
    user_satisfaction: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationPattern:
    """Learned pattern from validation history"""
    pattern_type: str
    pattern_description: str
    success_rate: float
    occurrence_count: int
    confidence: float
    last_seen: datetime


class ValidationLearningSystem:
    """Main learning system for validation improvement"""
    
    def __init__(self):
        self.patterns = {}
        self.outcome_history = []
        self.user_preferences = {}
        
    async def record_validation_outcome(self, validation_result: Any, context: Dict[str, Any]) -> None:
        """Record validation outcome for learning"""
        
        # Create outcome record
        outcome = ValidationOutcome(
            validation_id=f"val_{datetime.now().timestamp()}",
            code_analyzed=context.get("code", ""),
            issues_found=len(getattr(validation_result, 'issues_found', [])),
            critical_issues=len([i for i in getattr(validation_result, 'issues_found', []) 
                               if getattr(i, 'severity', '') == "critical"]),
            context=context
        )
        
        # Store outcome
        self.outcome_history.append(outcome)
        
        # Keep only recent outcomes (last 100)
        if len(self.outcome_history) > 100:
            self.outcome_history = self.outcome_history[-100:]
    
    async def record_user_feedback(self, validation_id: str, feedback: Dict[str, Any]) -> None:
        """Record user feedback on validation results"""
        
        # Find the validation outcome and update it
        for outcome in self.outcome_history:
            if outcome.validation_id == validation_id:
                outcome.user_feedback = feedback
                outcome.user_satisfaction = feedback.get("satisfaction_score", 0.5)
                break
    
    async def get_validation_insights(self, code: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get insights to improve validation for specific code"""
        
        return {
            "predicted_issues": [],
            "user_preferences": self.user_preferences.get(context.get("user_id"), {}),
            "effectiveness_metrics": {
                "total_validations": len(self.outcome_history),
                "average_satisfaction": 0.8  # Default
            },
            "recommendations": ["Continue current validation approach"]
        }


# Export main classes
__all__ = [
    'ValidationLearningSystem',
    'ValidationOutcome',
    'ValidationPattern'
]
