"""
Execution Learning System

Learning system that improves execution strategies and result interpretation
based on execution outcomes, performance metrics, and user feedback.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ExecutionOutcome:
    """Record of an execution outcome for learning"""
    execution_id: str
    language: str
    framework: str
    tests_executed: int
    success_rate: float
    execution_time: float
    memory_usage_mb: float
    user_feedback: Optional[Dict[str, Any]] = None
    performance_issues: List[str] = field(default_factory=list)
    interpretation_accuracy: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    context: Dict[str, Any] = field(default_factory=dict)


class ExecutionLearningSystem:
    """Main learning system for execution improvement"""
    
    def __init__(self):
        self.execution_history = []
        self.user_preferences = {}  # user_id -> preferences
        self.execution_feedback = []
        
    async def record_execution_outcome(self, execution_result: Any, context: Dict[str, Any]) -> None:
        """Record execution outcome for learning"""
        
        # Extract execution metrics
        outcome = ExecutionOutcome(
            execution_id=f"exec_{datetime.now().timestamp()}",
            language=context.get("language", "python"),
            framework=context.get("framework", "unknown"),
            tests_executed=getattr(execution_result.execution_result, 'total_tests', 0),
            success_rate=self._calculate_success_rate(execution_result.execution_result),
            execution_time=execution_result.monitoring_data.get('duration', 0.0),
            memory_usage_mb=execution_result.monitoring_data.get('peak_memory_mb', 0.0),
            context=context
        )
        
        # Store outcome
        self.execution_history.append(outcome)
        
        # Keep only recent history (last 100 executions)
        if len(self.execution_history) > 100:
            self.execution_history = self.execution_history[-100:]
    
    def _calculate_success_rate(self, execution_result) -> float:
        """Calculate success rate from execution result"""
        if not hasattr(execution_result, 'total_tests') or execution_result.total_tests == 0:
            return 0.0
        
        return execution_result.passed_tests / execution_result.total_tests
    
    async def record_user_feedback(self, execution_id: str, feedback: Dict[str, Any]) -> None:
        """Record user feedback on execution results"""
        
        # Store feedback
        self.execution_feedback.append({
            "execution_id": execution_id,
            "feedback": feedback,
            "timestamp": datetime.now()
        })
        
        # Update user preferences
        user_id = feedback.get("user_id")
        if user_id:
            await self._update_user_preferences(user_id, feedback)
    
    async def _update_user_preferences(self, user_id: str, feedback: Dict[str, Any]):
        """Update user preferences based on feedback"""
        
        if user_id not in self.user_preferences:
            self.user_preferences[user_id] = {
                "performance_sensitivity": "medium",
                "detail_preference": "balanced",
                "framework_preferences": [],
                "execution_timeout_preference": 60
            }
        
        prefs = self.user_preferences[user_id]
        
        # Update based on feedback
        if feedback.get("performance_too_slow"):
            prefs["performance_sensitivity"] = "high"
        elif feedback.get("performance_acceptable"):
            prefs["performance_sensitivity"] = "medium"
        
        if feedback.get("framework_worked_well"):
            framework = feedback.get("framework")
            if framework and framework not in prefs["framework_preferences"]:
                prefs["framework_preferences"].append(framework)
    
    async def get_execution_recommendations(self, language: str, framework: str, test_count: int, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get recommendations for execution"""
        
        # Basic recommendations
        recommendations = {
            "estimated_time_seconds": 5.0,
            "estimated_memory_mb": 50.0,
            "predicted_success_rate": 0.8,
            "confidence": 0.3,
            "optimization_suggestions": [],
            "user_customizations": []
        }
        
        # Apply user preferences if available
        user_prefs = self.user_preferences.get(user_id, {})
        
        if user_prefs.get("performance_sensitivity") == "high":
            recommendations["user_customizations"].append("Performance monitoring enabled")
        
        return recommendations
    
    async def analyze_execution_trends(self, time_period_days: int = 30) -> Dict[str, Any]:
        """Analyze execution trends over time"""
        cutoff_date = datetime.now() - timedelta(days=time_period_days)
        recent_executions = [e for e in self.execution_history if e.timestamp > cutoff_date]
        
        if not recent_executions:
            return {"error": "No recent execution data available"}
        
        # Calculate basic statistics
        avg_success_rate = sum(e.success_rate for e in recent_executions) / len(recent_executions)
        avg_execution_time = sum(e.execution_time for e in recent_executions) / len(recent_executions)
        
        return {
            "period_days": time_period_days,
            "total_executions": len(recent_executions),
            "average_success_rate": avg_success_rate,
            "average_execution_time": avg_execution_time,
            "performance_trend": "stable"
        }
    
    async def get_learning_insights(self) -> Dict[str, Any]:
        """Get insights about learning progress"""
        total_executions = len(self.execution_history)
        
        return {
            "total_executions_analyzed": total_executions,
            "user_engagement": len(self.user_preferences),
            "learning_effectiveness": 0.7 if total_executions > 10 else 0.3
        }


# Export main classes
__all__ = [
    'ExecutionLearningSystem',
    'ExecutionOutcome'
]
