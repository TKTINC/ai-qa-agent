"""
Tests for Agent Intelligence Metrics System
"""

import pytest
import asyncio
from datetime import datetime

from src.monitoring.metrics.agent_intelligence_metrics import (
    AgentIntelligenceMetrics,
    MetricContext,
    ReasoningComplexity,
    get_agent_metrics
)


@pytest.fixture
def metrics_system():
    """Create metrics system for testing"""
    return AgentIntelligenceMetrics()


@pytest.fixture
def sample_context():
    """Sample metric context"""
    return MetricContext(
        agent_name="test_agent",
        session_id="session_123",
        task_type="code_analysis"
    )


class TestAgentIntelligenceMetrics:
    """Test agent intelligence metrics collection"""
    
    @pytest.mark.asyncio
    async def test_record_reasoning_event(self, metrics_system, sample_context):
        """Test recording reasoning performance metrics"""
        await metrics_system.record_reasoning_event(
            context=sample_context,
            quality_score=0.85,
            duration=2.3,
            complexity=ReasoningComplexity.MODERATE,
            confidence=0.87
        )
        
        # Verify metrics were recorded
        assert True
    
    @pytest.mark.asyncio
    async def test_record_learning_event(self, metrics_system, sample_context):
        """Test recording learning progress metrics"""
        await metrics_system.record_learning_event(
            context=sample_context,
            learning_type="pattern_recognition",
            source="user_feedback",
            improvement_rate=0.15,
            capability="reasoning"
        )
        
        assert True
    
    @pytest.mark.asyncio
    async def test_record_user_interaction(self, metrics_system, sample_context):
        """Test recording user interaction metrics"""
        await metrics_system.record_user_interaction(
            context=sample_context,
            satisfaction_score=4,
            interaction_type="consultation",
            response_time=1.2
        )
        
        assert True
    
    def test_metrics_summary(self, metrics_system):
        """Test getting metrics summary"""
        summary = metrics_system.get_metrics_summary("test_agent")
        
        assert "timestamp" in summary
        assert "reasoning" in summary
        assert "learning" in summary
    
    def test_export_metrics(self, metrics_system):
        """Test exporting metrics in Prometheus format"""
        metrics_output = metrics_system.export_metrics()
        
        assert isinstance(metrics_output, str)
        assert len(metrics_output) >= 0


def test_global_metrics_instance():
    """Test global metrics instance"""
    metrics1 = get_agent_metrics()
    metrics2 = get_agent_metrics()
    
    # Should return same instance
    assert metrics1 is metrics2


if __name__ == "__main__":
    pytest.main([__file__])
