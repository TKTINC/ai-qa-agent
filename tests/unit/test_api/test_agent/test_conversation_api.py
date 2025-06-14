"""
Test Agent Conversation API
"""

import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock

from src.api.main import app
from src.api.routes.agent.conversation import ConversationRequest, agent_system


class TestAgentConversationAPI:
    
    def setup_method(self):
        """Setup test client"""
        self.client = TestClient(app)
    
    @patch('src.api.routes.agent.conversation.agent_system')
    def test_agent_conversation_endpoint(self, mock_agent_system):
        """Test basic agent conversation endpoint"""
        # Mock agent system response
        mock_response = Mock()
        mock_response.content = "This is a test response from the agent system"
        mock_response.session_id = "test_session"
        mock_response.confidence = 0.9
        mock_response.recommendations = ["Test recommendation"]
        mock_response.follow_up_questions = ["Test follow-up?"]
        mock_response.reasoning_steps = []
        mock_response.metadata = {"specialists_consulted": ["test_architect"]}
        mock_response.timestamp = "2025-06-13T10:00:00"
        
        mock_agent_system.system_initialized = True
        mock_agent_system.handle_complex_request = AsyncMock(return_value=mock_response)
        
        # Test request
        request_data = {
            "message": "Help me test my Python application",
            "session_id": "test_session",
            "user_profile": {"expertise_level": "intermediate"}
        }
        
        response = self.client.post("/api/v1/agent/conversation", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["response"] == "This is a test response from the agent system"
        assert data["session_id"] == "test_session"
        assert data["confidence"] == 0.9
        assert "test_architect" in data["agents_involved"]

    @patch('src.api.routes.agent.conversation.agent_system')
    def test_agent_status_endpoint(self, mock_agent_system):
        """Test agent status endpoint"""
        mock_status = {
            "system_initialized": True,
            "specialists": {
                "test_architect": {"availability": "available"},
                "code_reviewer": {"availability": "available"}
            },
            "active_collaborations": 2,
            "system_performance": {
                "avg_response_time_minutes": 2.5,
                "total_consultations": 150
            }
        }
        
        mock_agent_system.system_initialized = True
        mock_agent_system.get_system_status = AsyncMock(return_value=mock_status)
        
        response = self.client.get("/api/v1/agent/status")
        
        assert response.status_code == 200
        data = response.json()
        assert data["system_initialized"] is True
        assert len(data["available_specialists"]) == 2
        assert data["active_collaborations"] == 2

    @patch('src.api.routes.agent.conversation.agent_system')
    def test_specialist_profiles_endpoint(self, mock_agent_system):
        """Test specialist profiles endpoint"""
        mock_specialist = Mock()
        mock_capability = Mock()
        mock_capability.name = "test_strategy_design"
        mock_capability.description = "Design test strategies"
        mock_capability.confidence_level = 0.95
        mock_capability.experience_count = 50
        mock_capability.success_rate = 0.92
        
        mock_profile = Mock()
        mock_profile.agent_name = "test_architect"
        mock_profile.specialization = "Test Architecture"
        mock_profile.expertise_domains = ["testing", "strategy"]
        mock_profile.capabilities = [mock_capability]
        mock_profile.performance_metrics = {"success_rate": 0.9}
        mock_profile.availability_status = "available"
        
        mock_specialist.get_specialist_profile.return_value = mock_profile
        
        mock_agent_system.system_initialized = True
        mock_agent_system.specialists = {"test_architect": mock_specialist}
        
        response = self.client.get("/api/v1/agent/specialists")
        
        assert response.status_code == 200
        data = response.json()
        assert "specialists" in data
        assert "test_architect" in data["specialists"]
        specialist_data = data["specialists"]["test_architect"]
        assert specialist_data["name"] == "test_architect"
        assert specialist_data["specialization"] == "Test Architecture"
        assert len(specialist_data["capabilities"]) == 1

    def test_conversation_context_endpoint(self):
        """Test setting conversation context"""
        context_data = {
            "user_preferences": {"expertise_level": "expert"},
            "project_context": {"language": "python", "framework": "django"}
        }
        
        response = self.client.post(
            "/api/v1/agent/conversation/context?session_id=test_session",
            json=context_data
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["session_id"] == "test_session"

    def test_conversation_insights_endpoint(self):
        """Test getting conversation insights"""
        response = self.client.get("/api/v1/agent/conversation/test_session/insights")
        
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "test_session"
        assert "user_satisfaction_predicted" in data
        assert "agents_involved" in data

    @patch('src.api.routes.agent.conversation.agent_system')
    def test_error_handling(self, mock_agent_system):
        """Test API error handling"""
        mock_agent_system.handle_complex_request = AsyncMock(
            side_effect=Exception("Test error")
        )
        mock_agent_system.system_initialized = True
        
        request_data = {
            "message": "Test message",
            "session_id": "test_session"
        }
        
        response = self.client.post("/api/v1/agent/conversation", json=request_data)
        
        assert response.status_code == 500
        assert "Conversation failed" in response.json()["detail"]

    def test_request_validation(self):
        """Test request validation"""
        # Missing required fields
        invalid_request = {"message": "Test"}  # Missing session_id
        
        response = self.client.post("/api/v1/agent/conversation", json=invalid_request)
        
        assert response.status_code == 422  # Validation error
