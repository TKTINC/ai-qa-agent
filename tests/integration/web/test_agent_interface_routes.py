"""
Integration tests for Agent Interface Web Routes
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch

from fastapi.testclient import TestClient

from src.api.main import app
from src.web.routes.agent_interface import ConversationRequest, FeedbackRequest

class TestAgentInterfaceRoutes:
    """Test agent interface web routes"""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    def test_agent_chat_interface_page(self, client):
        """Test main chat interface page loads"""
        response = client.get("/web/")
        
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "AI QA Agent" in response.text
        assert "Conversational Interface" in response.text
    
    @patch('src.web.routes.agent_interface.chat_interface')
    def test_agent_conversation_endpoint(self, mock_chat_interface, client):
        """Test agent conversation HTTP endpoint"""
        # Mock the chat interface response
        mock_response = {
            "response": {
                "text": "Hello! I can help you with testing strategies and code analysis.",
                "agents_involved": ["test_architect"],
                "confidence": 0.9
            },
            "session_id": "test_session_123",
            "timestamp": datetime.now().isoformat(),
            "agents_involved": ["test_architect"],
            "confidence": 0.9
        }
        
        mock_chat_interface.handle_user_message = AsyncMock(return_value=mock_response)
        
        # Test conversation request
        request_data = {
            "message": "Help me create tests for my authentication system",
            "session_id": "test_session_123",
            "user_profile": {"expertise": "intermediate"}
        }
        
        response = client.post("/web/api/v1/agent/conversation", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["session_id"] == "test_session_123"
        assert "response" in data
        assert "agents_involved" in data
        assert "confidence" in data
        
        # Verify mock was called correctly
        mock_chat_interface.handle_user_message.assert_called_once_with(
            "Help me create tests for my authentication system",
            "test_session_123",
            {"expertise": "intermediate"}
        )
    
    def test_agent_conversation_invalid_request(self, client):
        """Test agent conversation with invalid request"""
        # Missing required fields
        request_data = {
            "message": "Test message"
            # Missing session_id
        }
        
        response = client.post("/web/api/v1/agent/conversation", json=request_data)
        assert response.status_code == 422  # Validation error
    
    @patch('src.web.routes.agent_interface.agent_visualization_service')
    def test_agent_status_endpoint(self, mock_visualization_service, client):
        """Test agent status endpoint"""
        # Mock visualization service responses
        mock_visualization_service.get_system_intelligence_metrics = AsyncMock(return_value={
            "activity_metrics": {"last_hour": 15, "last_day": 120},
            "learning_metrics": {"events_last_hour": 5, "events_last_day": 45},
            "system_health_score": 0.94
        })
        
        mock_visualization_service.get_agent_performance_summary = AsyncMock(return_value={
            "agent_name": "test_architect",
            "performance_score": 0.92,
            "total_activities": 25,
            "average_confidence": 0.89
        })
        
        mock_visualization_service.get_live_activity_feed = AsyncMock(return_value=[
            {
                "timestamp": datetime.now().isoformat(),
                "agent_name": "test_architect",
                "activity_type": "reasoning",
                "description": "Analyzed code complexity",
                "confidence": 0.9
            }
        ])
        
        response = client.get("/web/api/v1/agent/status")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "operational"
        assert "intelligence_metrics" in data
        assert "agent_performances" in data
        assert "activity_feed" in data
        assert "capabilities" in data
        
        # Check capabilities
        capabilities = data["capabilities"]
        assert capabilities["multi_agent_collaboration"] is True
        assert capabilities["real_time_reasoning"] is True
        assert capabilities["learning_enabled"] is True

if __name__ == "__main__":
    pytest.main([__file__])
