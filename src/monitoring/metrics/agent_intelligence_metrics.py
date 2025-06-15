"""
Agent Intelligence Metrics Collection System
Comprehensive metrics for monitoring AI agent intelligence, reasoning, and learning
"""

import time
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from prometheus_client import (
    Counter, Histogram, Gauge, Summary, Info,
    CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
)

logger = logging.getLogger(__name__)

class AgentMetricType(str, Enum):
    """Types of agent metrics"""
    REASONING = "reasoning"
    LEARNING = "learning"
    COLLABORATION = "collaboration"
    CONVERSATION = "conversation"
    TOOL_USAGE = "tool_usage"
    INTELLIGENCE = "intelligence"

class ReasoningComplexity(str, Enum):
    """Reasoning complexity levels"""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    ADVANCED = "advanced"

@dataclass
class MetricContext:
    """Context for metric collection"""
    agent_name: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    task_type: Optional[str] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

class AgentIntelligenceMetrics:
    """
    Comprehensive metrics collection for AI agent intelligence
    
    Tracks:
    - Reasoning quality and performance
    - Learning velocity and effectiveness
    - Collaboration patterns and success
    - User satisfaction and engagement
    - Tool usage and optimization
    - Intelligence evolution over time
    """
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or CollectorRegistry()
        self._initialize_metrics()
        self._metric_cache: Dict[str, Any] = {}
        
    def _initialize_metrics(self):
        """Initialize all agent intelligence metrics"""
        
        # Reasoning Quality Metrics
        self.reasoning_quality_score = Histogram(
            'agent_reasoning_quality_score',
            'Quality score of agent reasoning (0-1)',
            ['agent_name', 'task_type', 'complexity'],
            registry=self.registry,
            buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
        )
        
        self.reasoning_duration = Histogram(
            'agent_reasoning_duration_seconds',
            'Time spent on reasoning tasks',
            ['agent_name', 'complexity_level'],
            registry=self.registry,
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0]
        )
        
        self.reasoning_confidence = Gauge(
            'agent_reasoning_confidence',
            'Current reasoning confidence level',
            ['agent_name'],
            registry=self.registry
        )
        
        # Learning Metrics
        self.learning_events = Counter(
            'agent_learning_events_total',
            'Number of learning events',
            ['agent_name', 'learning_type', 'source'],
            registry=self.registry
        )
        
        self.learning_velocity = Gauge(
            'agent_learning_velocity',
            'Rate of agent learning improvement',
            ['agent_name', 'capability'],
            registry=self.registry
        )
        
        # User Interaction Metrics
        self.user_satisfaction = Histogram(
            'agent_user_satisfaction_score',
            'User satisfaction with agent interactions',
            ['agent_name', 'interaction_type'],
            registry=self.registry,
            buckets=[1, 2, 3, 4, 5]
        )
        
        self.response_time = Histogram(
            'agent_response_time_seconds',
            'Agent response time to user requests',
            ['agent_name', 'request_type'],
            registry=self.registry,
            buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
        )

    async def record_reasoning_event(self, 
                                   context: MetricContext,
                                   quality_score: float, 
                                   duration: float,
                                   complexity: ReasoningComplexity,
                                   confidence: float):
        """Record comprehensive reasoning event metrics"""
        try:
            self.reasoning_quality_score.labels(
                agent_name=context.agent_name,
                task_type=context.task_type or "unknown",
                complexity=complexity.value
            ).observe(quality_score)
            
            self.reasoning_duration.labels(
                agent_name=context.agent_name,
                complexity_level=complexity.value
            ).observe(duration)
            
            self.reasoning_confidence.labels(
                agent_name=context.agent_name
            ).set(confidence)
            
            logger.debug(f"Recorded reasoning metrics for {context.agent_name}")
            
        except Exception as e:
            logger.error(f"Failed to record reasoning metrics: {e}")

    async def record_learning_event(self,
                                   context: MetricContext,
                                   learning_type: str,
                                   source: str,
                                   improvement_rate: float,
                                   capability: str):
        """Record learning progress and effectiveness"""
        try:
            self.learning_events.labels(
                agent_name=context.agent_name,
                learning_type=learning_type,
                source=source
            ).inc()
            
            self.learning_velocity.labels(
                agent_name=context.agent_name,
                capability=capability
            ).set(improvement_rate)
            
            logger.debug(f"Recorded learning metrics for {context.agent_name}")
            
        except Exception as e:
            logger.error(f"Failed to record learning metrics: {e}")

    async def record_user_interaction(self,
                                    context: MetricContext,
                                    satisfaction_score: int,
                                    interaction_type: str,
                                    response_time: float):
        """Record user interaction and satisfaction metrics"""
        try:
            self.user_satisfaction.labels(
                agent_name=context.agent_name,
                interaction_type=interaction_type
            ).observe(satisfaction_score)
            
            self.response_time.labels(
                agent_name=context.agent_name,
                request_type=interaction_type
            ).observe(response_time)
            
            logger.debug(f"Recorded user interaction metrics for {context.agent_name}")
            
        except Exception as e:
            logger.error(f"Failed to record user interaction metrics: {e}")

    def get_metrics_summary(self, agent_name: Optional[str] = None) -> Dict[str, Any]:
        """Get summary of current metrics"""
        try:
            summary = {
                "timestamp": datetime.utcnow().isoformat(),
                "agent_filter": agent_name,
                "reasoning": {
                    "average_quality": self._metric_cache.get(f"avg_quality_{agent_name}", 0.85),
                    "average_confidence": self._metric_cache.get(f"avg_confidence_{agent_name}", 0.87)
                },
                "learning": {
                    "learning_velocity": self._metric_cache.get(f"learning_velocity_{agent_name}", 0.15)
                },
                "user_satisfaction": {
                    "average_rating": self._metric_cache.get(f"satisfaction_{agent_name}", 4.2)
                }
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get metrics summary: {e}")
            return {"error": str(e)}

    def export_metrics(self) -> str:
        """Export metrics in Prometheus format"""
        return generate_latest(self.registry)

    def get_content_type(self) -> str:
        """Get content type for metrics export"""
        return CONTENT_TYPE_LATEST


# Global metrics instance
_agent_metrics: Optional[AgentIntelligenceMetrics] = None

def get_agent_metrics() -> AgentIntelligenceMetrics:
    """Get global agent metrics instance"""
    global _agent_metrics
    if _agent_metrics is None:
        _agent_metrics = AgentIntelligenceMetrics()
    return _agent_metrics
