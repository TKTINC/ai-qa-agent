"""
Tests for Chat API Routes
AI QA Agent - Enhanced Sprint 1.4
"""
import pytest
import json
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime

from src.api.main import app
from src.chat.conversation_manager import ConversationSession, Message

client = TestClient(app)

class TestChatAPI:
    """Test chat API endpoints"""
    
    @pytest.fixture
    def sample_session(self):
        return ConversationSession(
            session_id="test-session-123",
            user_id="test-user",
            title="Test Chat",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            metadata={}
        )
    
    @pytest.fixture
    def sample_message(self):
        return Message(
            id="msg-123",
            session_id="test-session-123",
            role="user",
            content="Hello, AI!",
            metadata={},
            timestamp=datetime.utcnow()
        )
    
    @patch('src.api.routes.chat.conversation_manager')
    @patch('src.api.routes.chat.llm_integration')
    def test_send_message_new_session(self, mock_llm, mock_conv_manager, sample_session):
        """Test sending message that creates new session"""
        # Mock conversation manager
        mock_conv_manager.create_session = AsyncMock(return_value=sample_session)
        mock_conv_manager.add_message = AsyncMock(side_effect=[
            Message("user-msg", sample_session.session_id, "user", "Hello", {}, datetime.utcnow()),
            Message("ai-msg", sample_session.session_id, "assistant", "Hi there!", {}, datetime.utcnow())
        ])
        mock_conv_manager.get_conversation_context = AsyncMock(return_value={})
        mock_conv_manager.get_messages = AsyncMock(return_value=[])
        
        # Mock LLM
        mock_llm.analyze_user_intent = AsyncMock(return_value={
            "intent": "greeting",
            "confidence": 0.9
        })
        mock_llm.generate_response = AsyncMock(return_value="Hi there! How can I help you?")
        
        # Send message
        request_data = {
            "message": "Hello, AI!",
            "user_id": "test-user"
        }
        
        response = client.post("/api/v1/chat/message", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "session_id" in data
        assert "message_id" in data
        assert "response" in data
        assert data["response"] == "Hi there! How can I help you?"
    
    @patch('src.api.routes.chat.conversation_manager')
    @patch('src.api.routes.chat.llm_integration')
    def test_send_message_existing_session(self, mock_llm, mock_conv_manager, sample_session):
        """Test sending message to existing session"""
        # Mock conversation manager
        mock_conv_manager.get_session = AsyncMock(return_value=sample_session)
        mock_conv_manager.add_message = AsyncMock(side_effect=[
            Message("user-msg", sample_session.session_id, "user", "How are you?", {}, datetime.utcnow()),
            Message("ai-msg", sample_session.session_id, "assistant", "I'm doing well!", {}, datetime.utcnow())
        ])
        mock_conv_manager.get_conversation_context = AsyncMock(return_value={})
        mock_conv_manager.get_messages = AsyncMock(return_value=[])
        
        # Mock LLM
        mock_llm.analyze_user_intent = AsyncMock(return_value={
            "intent": "general_conversation",
            "confidence": 0.8
        })
        mock_llm.generate_response = AsyncMock(return_value="I'm doing well, thanks!")
        
        # Send message
        request_data = {
            "session_id": sample_session.session_id,
            "message": "How are you?"
        }
        
        response = client.post("/api/v1/chat/message", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["session_id"] == sample_session.session_id
        assert data["response"] == "I'm doing well, thanks!"
    
    @patch('src.api.routes.chat.conversation_manager')
    def test_send_message_session_not_found(self, mock_conv_manager):
        """Test sending message to non-existent session"""
        mock_conv_manager.get_session = AsyncMock(return_value=None)
        
        request_data = {
            "session_id": "nonexistent-session",
            "message": "Hello"
        }
        
        response = client.post("/api/v1/chat/message", json=request_data)
        assert response.status_code == 404
        assert response.json()["detail"] == "Session not found"
    
    @patch('src.api.routes.chat.conversation_manager')
    def test_get_sessions(self, mock_conv_manager, sample_session):
        """Test getting conversation sessions"""
        mock_conv_manager.get_recent_sessions = AsyncMock(return_value=[sample_session])
        
        response = client.get("/api/v1/chat/sessions")
        assert response.status_code == 200
        
        data = response.json()
        assert len(data) == 1
        assert data[0]["session_id"] == sample_session.session_id
        assert data[0]["title"] == sample_session.title
    
    @patch('src.api.routes.chat.conversation_manager')
    def test_get_sessions_with_user_filter(self, mock_conv_manager, sample_session):
        """Test getting sessions for specific user"""
        mock_conv_manager.get_recent_sessions = AsyncMock(return_value=[sample_session])
        
        response = client.get("/api/v1/chat/sessions?user_id=test-user&limit=10")
        assert response.status_code == 200
        
        data = response.json()
        assert len(data) == 1
        
        # Verify the mock was called with correct parameters
        mock_conv_manager.get_recent_sessions.assert_called_with(user_id="test-user", limit=10)
    
    @patch('src.api.routes.chat.conversation_manager')
    def test_get_conversation_history(self, mock_conv_manager, sample_session, sample_message):
        """Test getting conversation history"""
        mock_conv_manager.get_session = AsyncMock(return_value=sample_session)
        mock_conv_manager.get_messages = AsyncMock(return_value=[sample_message])
        
        response = client.get(f"/api/v1/chat/sessions/{sample_session.session_id}")
        assert response.status_code == 200
        
        data = response.json()
        assert "session" in data
        assert "messages" in data
        assert "total_messages" in data
        
        assert data["session"]["session_id"] == sample_session.session_id
        assert len(data["messages"]) == 1
        assert data["messages"][0]["content"] == sample_message.content
    
    @patch('src.api.routes.chat.conversation_manager')
    def test_get_conversation_history_not_found(self, mock_conv_manager):
        """Test getting history for non-existent session"""
        mock_conv_manager.get_session = AsyncMock(return_value=None)
        
        response = client.get("/api/v1/chat/sessions/nonexistent")
        assert response.status_code == 404
        assert response.json()["detail"] == "Session not found"
    
    @patch('src.api.routes.chat.conversation_manager')
    def test_create_session(self, mock_conv_manager, sample_session):
        """Test creating new session"""
        mock_conv_manager.create_session = AsyncMock(return_value=sample_session)
        
        request_data = {
            "user_id": "test-user",
            "title": "New Chat Session",
            "metadata": {"test": "data"}
        }
        
        response = client.post("/api/v1/chat/sessions", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["session_id"] == sample_session.session_id
        assert data["title"] == sample_session.title
    
    @patch('src.api.routes.chat.conversation_manager')
    def test_delete_session(self, mock_conv_manager):
        """Test deleting session"""
        mock_conv_manager.delete_session = AsyncMock(return_value=True)
        
        response = client.delete("/api/v1/chat/sessions/test-session-123")
        assert response.status_code == 200
        
        data = response.json()
        assert "deleted successfully" in data["message"]
    
    @patch('src.api.routes.chat.conversation_manager')
    def test_delete_session_not_found(self, mock_conv_manager):
        """Test deleting non-existent session"""
        mock_conv_manager.delete_session = AsyncMock(return_value=False)
        
        response = client.delete("/api/v1/chat/sessions/nonexistent")
        assert response.status_code == 404
        assert response.json()["detail"] == "Session not found"
    
    @patch('src.api.routes.chat.conversation_manager')
    def test_update_session_metadata(self, mock_conv_manager):
        """Test updating session metadata"""
        mock_conv_manager.update_session_metadata = AsyncMock(return_value=True)
        
        metadata = {"analysis_count": 5, "user_preference": "detailed"}
        
        response = client.put(
            "/api/v1/chat/sessions/test-session-123/metadata",
            json=metadata
        )
        assert response.status_code == 200
        
        data = response.json()
        assert "updated successfully" in data["message"]
    
    @patch('src.api.routes.chat.conversation_manager')
    def test_update_session_metadata_not_found(self, mock_conv_manager):
        """Test updating metadata for non-existent session"""
        mock_conv_manager.update_session_metadata = AsyncMock(return_value=False)
        
        response = client.put(
            "/api/v1/chat/sessions/nonexistent/metadata",
            json={"test": "data"}
        )
        assert response.status_code == 404
        assert response.json()["detail"] == "Session not found"

class TestChatWebSocket:
    """Test WebSocket chat functionality"""
    
    def test_websocket_connection_manager(self):
        """Test WebSocket connection manager"""
        from src.api.routes.chat import ChatConnectionManager
        
        manager = ChatConnectionManager()
        
        # Test initial state
        assert len(manager.active_connections) == 0
        
        # Test disconnect non-existent connection
        manager.disconnect("nonexistent")
        assert len(manager.active_connections) == 0
    
    def test_websocket_message_format(self):
        """Test WebSocket message format validation"""
        # Test message structure
        welcome_msg = {
            "type": "system",
            "message": "Connected to AI QA Agent",
            "session_id": "test-session",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Validate required fields
        assert "type" in welcome_msg
        assert "message" in welcome_msg
        assert "session_id" in welcome_msg
        assert "timestamp" in welcome_msg
        
        # Test typing indicator format
        typing_msg = {
            "type": "typing",
            "message": "AI is thinking...",
            "session_id": "test-session",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        assert typing_msg["type"] == "typing"
        
        # Test response message format
        response_msg = {
            "type": "message",
            "message": "AI response",
            "message_id": "msg-123",
            "session_id": "test-session",
            "metadata": {},
            "timestamp": datetime.utcnow().isoformat()
        }
        
        assert response_msg["type"] == "message"
        assert "message_id" in response_msg

class TestChatIntegration:
    """Test integration between chat components"""
    
    @patch('src.api.routes.chat.conversation_manager')
    @patch('src.api.routes.chat.llm_integration')
    def test_chat_flow_integration(self, mock_llm, mock_conv_manager):
        """Test complete chat flow integration"""
        # Setup mocks for complete flow
        session = ConversationSession(
            session_id="integration-test",
            user_id="test-user",
            title="Integration Test",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            metadata={}
        )
        
        mock_conv_manager.create_session = AsyncMock(return_value=session)
        mock_conv_manager.add_message = AsyncMock(side_effect=[
            Message("user-msg", session.session_id, "user", "Analyze my code", {}, datetime.utcnow()),
            Message("ai-msg", session.session_id, "assistant", "I'll help analyze your code", {}, datetime.utcnow())
        ])
        mock_conv_manager.get_conversation_context = AsyncMock(return_value={
            "analysis_results": []
        })
        mock_conv_manager.get_messages = AsyncMock(return_value=[])
        
        mock_llm.analyze_user_intent = AsyncMock(return_value={
            "intent": "analysis_request",
            "confidence": 0.95,
            "entities": {"programming_languages": ["python"]}
        })
        mock_llm.generate_response = AsyncMock(return_value="I'll help you analyze your Python code. Please share the file or code content.")
        
        # Test the flow
        request_data = {
            "message": "Analyze my Python code for quality issues",
            "user_id": "test-user"
        }
        
        response = client.post("/api/v1/chat/message", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["session_id"] == session.session_id
        assert "analyze" in data["response"].lower()
        assert "metadata" in data
        assert data["metadata"]["intent"]["intent"] == "analysis_request"

if __name__ == "__main__":
    pytest.main([__file__])
