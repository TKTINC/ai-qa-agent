"""
Tests for learning analytics API.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient

from src.api.routes.learning.learning_analytics import router
from src.agent.learning.learning_engine import AgentLearningEngine


@pytest.fixture
def client():
    """Create test client"""
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


@pytest.fixture
def mock_learning_engine():
    """Mock learning engine"""
    engine = MagicMock()
    engine.agent_capabilities = {
        "test_architect": {
            "problem_solving_score": 0.85,
            "learning_velocity": 0.7,
            "user_communication": 0.9
        }
    }
    engine.learning_history = [
        {
            "timestamp": "2025-06-13T10:00:00",
            "agent": "test_architect",
            "learning_score": 0.8,
            "patterns_learned": 3
        }
    ]
    return engine


class TestLearningAnalyticsAPI:
    """Test learning analytics API endpoints"""
    
    def test_get_agent_performance(self, client, mock_learning_engine):
        """Test agent performance endpoint"""
        
        with patch('src.api.routes.learning.learning_analytics.experience_tracker') as mock_tracker:
            # Setup mock
            mock_analysis = MagicMock()
            mock_analysis.total_experiences = 25
            mock_analysis.success_patterns = [{"confidence": 0.8}]
            mock_analysis.optimization_opportunities = []
            mock_analysis.recommendations = ["Continue current approach"]
            
            mock_tracker.analyze_experience_patterns = AsyncMock(return_value=mock_analysis)
            
            with patch('src.api.routes.learning.learning_analytics.learning_engine', mock_learning_engine):
                response = client.get("/api/v1/learning/agents/test_architect/performance")
                
                assert response.status_code == 200
                data = response.json()
                
                assert data["agent_name"] == "test_architect"
                assert data["performance_score"] == 0.85
                assert data["experience_count"] == 25
                assert len(data["areas_for_improvement"]) > 0
    
    def test_get_user_personalization(self, client):
        """Test user personalization endpoint"""
        
        with patch('src.api.routes.learning.learning_analytics.feedback_processor') as mock_processor:
            # Setup mock user preferences
            mock_processor.user_preferences = {
                "user_123": {
                    "communication_style": "technical",
                    "feedback_history": [
                        {"timestamp": "2025-06-13T10:00:00", "rating": 4.5},
                        {"timestamp": "2025-06-13T11:00:00", "rating": 4.8}
                    ]
                }
            }
            
            response = client.get("/api/v1/learning/user/user_123/personalization")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["user_id"] == "user_123"
            assert data["communication_style"] == "technical"
            assert data["personalization_accuracy"] > 0
            assert data["interaction_count"] == 2
    
    def test_submit_learning_feedback(self, client):
        """Test learning feedback submission"""
        
        feedback_data = {
            "session_id": "session_123",
            "agent_name": "test_architect",
            "feedback_type": "interaction",
            "satisfaction_rating": 4.5,
            "feedback_text": "Great job on the analysis!",
            "specific_improvements": ["Add more examples"],
            "context": {"task_type": "code_analysis"}
        }
        
        with patch('src.api.routes.learning.learning_analytics.feedback_processor') as mock_processor:
            # Setup mock feedback processing
            mock_insights = MagicMock()
            mock_insights.feedback_id = "fb_123"
            mock_insights.sentiment_category = "positive"
            mock_insights.improvement_suggestions = ["Continue current approach"]
            mock_insights.preference_updates = {"communication_style": "detailed"}
            mock_insights.specific_insights = ["User appreciated thoroughness"]
            mock_insights.recommendations = ["Maintain current quality level"]
            
            mock_processor.process_immediate_feedback = AsyncMock(return_value=mock_insights)
            
            response = client.post("/api/v1/learning/feedback", json=feedback_data)
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["feedback_processed"] is True
            assert data["feedback_id"] == "fb_123"
            assert data["sentiment_detected"] == "positive"
            assert len(data["immediate_improvements"]) > 0
    
    def test_get_agent_intelligence_metrics(self, client, mock_learning_engine):
        """Test agent intelligence metrics endpoint"""
        
        with patch('src.api.routes.learning.learning_analytics.learning_engine', mock_learning_engine), \
             patch('src.api.routes.learning.learning_analytics.feedback_processor') as mock_processor:
            
            # Setup mocks
            mock_learning_engine.get_learning_insights = AsyncMock(return_value={
                "average_learning_score": 0.85,
                "learning_velocity": 0.7,
                "quality_trend": "improving",
                "total_interactions": 150,
                "top_learning_agents": [{"agent_name": "test_architect", "average_score": 0.9}]
            })
            
            mock_processor.get_user_satisfaction_trends = AsyncMock(return_value={
                "overall_satisfaction": 4.2
            })
            
            response = client.get("/api/v1/learning/analytics/agent-intelligence")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["overall_intelligence_score"] == 0.85
            assert data["learning_velocity"] == 0.7
            assert data["trend_direction"] == "improving"
            assert data["confidence_level"] > 0
    
    def test_get_session_learning_insights(self, client, mock_learning_engine):
        """Test session learning insights endpoint"""
        
        with patch('src.api.routes.learning.learning_analytics.learning_engine', mock_learning_engine):
            # Update mock learning history to include session data
            mock_learning_engine.learning_history = [
                {
                    "interaction_id": "session_123_interaction_1",
                    "timestamp": "2025-06-13T10:00:00",
                    "agent": "test_architect",
                    "learning_score": 0.8,
                    "patterns_learned": 3,
                    "capabilities_updated": 2
                },
                {
                    "interaction_id": "session_123_interaction_2",
                    "timestamp": "2025-06-13T10:30:00",
                    "agent": "test_architect",
                    "learning_score": 0.9,
                    "patterns_learned": 4,
                    "capabilities_updated": 1
                }
            ]
            
            response = client.get("/api/v1/learning/analytics/learning-insights/session_123")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["session_id"] == "session_123"
            assert data["learning_events"] == 2
            assert data["total_patterns_learned"] == 7
            assert data["average_learning_quality"] == 0.85
            assert len(data["learning_timeline"]) == 2
    
    def test_get_improvement_opportunities(self, client, mock_learning_engine):
        """Test improvement opportunities endpoint"""
        
        with patch('src.api.routes.learning.learning_analytics.learning_engine', mock_learning_engine), \
             patch('src.api.routes.learning.learning_analytics.feedback_processor') as mock_processor:
            
            # Setup mocks
            mock_learning_engine.get_learning_insights = AsyncMock(return_value={
                "improvement_opportunities": [
                    "Enhance response clarity",
                    "Improve tool selection accuracy"
                ]
            })
            
            mock_improvement_areas = MagicMock()
            mock_improvement_areas.communication_improvements = ["Improve explanation clarity"]
            mock_improvement_areas.technical_improvements = ["Optimize processing speed"]
            mock_improvement_areas.process_improvements = ["Streamline workflow"]
            mock_improvement_areas.user_experience_improvements = ["Reduce response time"]
            mock_improvement_areas.priority_ranking = ["Improve explanation clarity", "Optimize processing speed"]
            mock_improvement_areas.estimated_impact = {"Improve explanation clarity": 0.8}
            
            mock_processor.feedback_history = [MagicMock() for _ in range(50)]  # Mock feedback history
            mock_processor.identify_improvement_opportunities = AsyncMock(return_value=mock_improvement_areas)
            
            response = client.get("/api/v1/learning/analytics/improvement-opportunities")
            
            assert response.status_code == 200
            data = response.json()
            
            assert "improvement_opportunities" in data
            assert "analysis_info" in data
            assert data["analysis_info"]["analysis_period"] == "30 days"
            assert len(data["actionable_recommendations"]) > 0


class TestLearningAnalyticsIntegration:
    """Integration tests for learning analytics"""
    
    @pytest.mark.asyncio
    async def test_complete_analytics_workflow(self):
        """Test complete analytics workflow"""
        
        # This would test the full workflow from learning event to analytics display
        # For now, we'll test the key components work together
        
        from src.agent.learning.learning_engine import AgentLearningEngine
        from src.analytics.dashboards.learning_dashboard import LearningDashboard
        
        # Create instances
        learning_engine = AgentLearningEngine()
        dashboard = LearningDashboard()
        
        # Add some test data
        learning_engine.agent_capabilities = {
            "test_agent": {
                "problem_solving_score": 0.85,
                "learning_velocity": 0.7,
                "user_communication": 0.9
            }
        }
        
        # Generate dashboard data
        dashboard_data = await dashboard.generate_dashboard_data("24h")
        
        # Verify dashboard data structure
        assert "overview" in dashboard_data
        assert "agent_performance" in dashboard_data
        assert "system_health" in dashboard_data
        
        # Verify metrics are reasonable
        assert dashboard_data["overview"]["agents_active"] == 1
        assert dashboard_data["system_health"]["active_agents"] == 1
