"""
Tests for Intelligence Dashboard Components
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import json

from src.web.dashboards.intelligence_dashboard import (
    IntelligenceDashboard, AgentMetricsCollector, IntelligenceVisualizer,
    IntelligenceMetrics, AgentPerformanceMetrics
)

class TestAgentMetricsCollector:
    """Test agent metrics collection functionality"""
    
    @pytest.fixture
    def metrics_collector(self):
        return AgentMetricsCollector()
    
    @pytest.mark.asyncio
    async def test_collect_intelligence_metrics(self, metrics_collector):
        """Test collecting intelligence metrics"""
        with patch.object(metrics_collector.visualization_service, 'get_system_intelligence_metrics') as mock_system:
            mock_system.return_value = {"system_health_score": 0.85}
            
            # Mock activity data
            metrics_collector.visualization_service.agent_activities = [
                Mock(
                    timestamp=datetime.now(),
                    activity_type="reasoning",
                    confidence=0.9,
                    agent_name="test_agent"
                ),
                Mock(
                    timestamp=datetime.now() - timedelta(hours=2),
                    activity_type="collaboration", 
                    confidence=0.85,
                    agent_name="test_agent"
                )
            ]
            
            # Mock learning events
            metrics_collector.visualization_service.learning_events = [
                Mock(
                    timestamp=datetime.now(),
                    learning_type="improvement",
                    impact_score=0.8,
                    agent_name="test_agent"
                )
            ]
            
            metrics = await metrics_collector.collect_intelligence_metrics()
            
            assert isinstance(metrics, IntelligenceMetrics)
            assert 0 <= metrics.reasoning_quality_avg <= 1
            assert metrics.learning_velocity >= 0
            assert 0 <= metrics.collaboration_effectiveness <= 1
            assert metrics.user_satisfaction > 0
            assert 0 <= metrics.system_health_score <= 1
    
    @pytest.mark.asyncio
    async def test_collect_agent_performance_metrics(self, metrics_collector):
        """Test collecting individual agent performance metrics"""
        agent_name = "test_architect"
        
        # Mock activity data for specific agent
        metrics_collector.visualization_service.agent_activities = [
            Mock(
                timestamp=datetime.now(),
                agent_name=agent_name,
                activity_type="reasoning",
                confidence=0.92
            ),
            Mock(
                timestamp=datetime.now() - timedelta(hours=1),
                agent_name=agent_name,
                activity_type="collaboration",
                confidence=0.88
            )
        ]
        
        # Mock learning events for specific agent
        metrics_collector.visualization_service.learning_events = [
            Mock(
                timestamp=datetime.now(),
                agent_name=agent_name,
                learning_type="improvement",
                impact_score=0.85
            )
        ]
        
        metrics = await metrics_collector.collect_agent_performance_metrics(agent_name)
        
        assert isinstance(metrics, AgentPerformanceMetrics)
        assert metrics.agent_name == agent_name
        assert 0 <= metrics.performance_score <= 1
        assert 0 <= metrics.reasoning_quality <= 1
        assert metrics.learning_rate >= 0
        assert 0 <= metrics.collaboration_score <= 1
        assert 0 <= metrics.task_completion_rate <= 1
        assert 0 <= metrics.user_rating <= 5.0

class TestIntelligenceVisualizer:
    """Test intelligence visualization functionality"""
    
    @pytest.fixture
    def visualizer(self):
        return IntelligenceVisualizer()
    
    def test_create_intelligence_overview_chart(self, visualizer):
        """Test creating intelligence overview radar chart"""
        metrics = IntelligenceMetrics(
            timestamp=datetime.now(),
            reasoning_quality_avg=0.92,
            learning_velocity=0.15,
            collaboration_effectiveness=0.88,
            user_satisfaction=0.96,
            problem_solving_accuracy=0.89,
            capability_improvement_rate=0.75,
            system_health_score=0.85
        )
        
        chart_json = visualizer.create_intelligence_overview_chart(metrics)
        
        assert isinstance(chart_json, str)
        chart_data = json.loads(chart_json)
        assert "data" in chart_data
        assert "layout" in chart_data
        assert chart_data["layout"]["title"]["text"] == "🧠 Agent Intelligence Overview"
    
    def test_create_learning_velocity_chart(self, visualizer):
        """Test creating learning velocity chart"""
        chart_json = visualizer.create_learning_velocity_chart("24h")
        
        assert isinstance(chart_json, str)
        chart_data = json.loads(chart_json)
        assert "data" in chart_data
        assert "layout" in chart_data
        assert "📈 Learning Velocity Trends" in chart_data["layout"]["title"]["text"]
    
    def test_create_agent_performance_comparison(self, visualizer):
        """Test creating agent performance comparison chart"""
        agent_metrics = [
            AgentPerformanceMetrics(
                agent_name="test_architect",
                performance_score=0.92,
                reasoning_quality=0.89,
                learning_rate=0.15,
                collaboration_score=0.88,
                task_completion_rate=0.95,
                user_rating=4.6,
                improvement_velocity=0.12
            ),
            AgentPerformanceMetrics(
                agent_name="code_reviewer",
                performance_score=0.87,
                reasoning_quality=0.85,
                learning_rate=0.12,
                collaboration_score=0.82,
                task_completion_rate=0.91,
                user_rating=4.3,
                improvement_velocity=0.10
            )
        ]
        
        chart_json = visualizer.create_agent_performance_comparison(agent_metrics)
        
        assert isinstance(chart_json, str)
        chart_data = json.loads(chart_json)
        assert "data" in chart_data
        assert len(chart_data["data"]) >= 2  # At least 2 traces
        assert "🤖 Agent Performance Comparison" in chart_data["layout"]["title"]["text"]
    
    def test_create_collaboration_network(self, visualizer):
        """Test creating collaboration network visualization"""
        chart_json = visualizer.create_collaboration_network()
        
        assert isinstance(chart_json, str)
        chart_data = json.loads(chart_json)
        assert "data" in chart_data
        assert "layout" in chart_data
        assert "🤝 Agent Collaboration Network" in chart_data["layout"]["title"]["text"]
    
    def test_create_user_satisfaction_trends(self, visualizer):
        """Test creating user satisfaction trends chart"""
        chart_json = visualizer.create_user_satisfaction_trends("7d")
        
        assert isinstance(chart_json, str)
        chart_data = json.loads(chart_json)
        assert "data" in chart_data
        assert "layout" in chart_data
        assert "😊 User Satisfaction Trends" in chart_data["layout"]["title"]["text"]

class TestIntelligenceDashboard:
    """Test main intelligence dashboard functionality"""
    
    @pytest.fixture
    def dashboard(self):
        return IntelligenceDashboard()
    
    @pytest.mark.asyncio
    async def test_render_intelligence_overview(self, dashboard):
        """Test rendering complete intelligence overview"""
        with patch.object(dashboard.metrics_collector, 'collect_intelligence_metrics') as mock_intel, \
             patch.object(dashboard.metrics_collector, 'collect_agent_performance_metrics') as mock_agent:
            
            # Mock intelligence metrics
            mock_intel.return_value = IntelligenceMetrics(
                timestamp=datetime.now(),
                reasoning_quality_avg=0.92,
                learning_velocity=0.15,
                collaboration_effectiveness=0.88,
                user_satisfaction=0.96,
                problem_solving_accuracy=0.89,
                capability_improvement_rate=0.75,
                system_health_score=0.85
            )
            
            # Mock agent metrics
            mock_agent.return_value = AgentPerformanceMetrics(
                agent_name="test_architect",
                performance_score=0.92,
                reasoning_quality=0.89,
                learning_rate=0.15,
                collaboration_score=0.88,
                task_completion_rate=0.95,
                user_rating=4.6,
                improvement_velocity=0.12
            )
            
            # Mock insights
            with patch.object(dashboard.real_time_monitor, 'generate_intelligence_insights') as mock_insights:
                mock_insights.return_value = [
                    {
                        "type": "positive",
                        "category": "learning",
                        "title": "High Learning Velocity",
                        "description": "Excellent learning performance"
                    }
                ]
                
                # Mock live data
                with patch.object(dashboard.real_time_monitor, 'get_live_intelligence_data') as mock_live:
                    mock_live.return_value = {
                        "timestamp": datetime.now().isoformat(),
                        "status": "operational"
                    }
                    
                    overview = await dashboard.render_intelligence_overview("24h")
                    
                    assert "overview" in overview
                    assert "learning" in overview
                    assert "performance" in overview
                    assert "collaboration" in overview
                    assert "satisfaction" in overview
                    assert "live_data" in overview
                    assert "timestamp" in overview
                    
                    # Check overview section
                    assert "metrics" in overview["overview"]
                    assert "chart" in overview["overview"]
                    assert "insights" in overview["overview"]
                    
                    # Check learning section
                    assert "chart" in overview["learning"]
                    assert "velocity" in overview["learning"]
                    assert "trend" in overview["learning"]
                    
                    # Check performance section
                    assert "chart" in overview["performance"]
                    assert "agents" in overview["performance"]
                    assert "top_performer" in overview["performance"]

if __name__ == "__main__":
    pytest.main([__file__])
