"""
Tests for Conversation Manager
AI QA Agent - Enhanced Sprint 1.4
"""
import pytest
import asyncio
from datetime import datetime
from unittest.mock import patch, MagicMock

from src.chat.conversation_manager import (
    ConversationManager, Message, ConversationSession
)

class TestConversationManager:
    """Test conversation management functionality"""
    
    @pytest.fixture
    def conversation_manager(self):
        return ConversationManager()
    
    @pytest.mark.asyncio
    async def test_create_session(self, conversation_manager):
        """Test creating a new conversation session"""
        session = await conversation_manager.create_session(
            user_id="test-user",
            title="Test Session"
        )
        
        assert session.session_id is not None
        assert session.user_id == "test-user"
        assert session.title == "Test Session"
        assert session.message_count == 0
        assert isinstance(session.created_at, datetime)
    
    @pytest.mark.asyncio
    async def test_add_message(self, conversation_manager):
        """Test adding messages to a session"""
        # Create session
        session = await conversation_manager.create_session()
        
        # Add user message
        user_message = await conversation_manager.add_message(
            session.session_id,
            "user",
            "Hello, I need help with testing",
            {"intent": "help_request"}
        )
        
        assert user_message.session_id == session.session_id
        assert user_message.role == "user"
        assert user_message.content == "Hello, I need help with testing"
        assert user_message.metadata["intent"] == "help_request"
        
        # Add assistant message
        ai_message = await conversation_manager.add_message(
            session.session_id,
            "assistant",
            "I'd be happy to help you with testing!"
        )
        
        assert ai_message.role == "assistant"
        assert ai_message.session_id == session.session_id
    
    @pytest.mark.asyncio
    async def test_get_messages(self, conversation_manager):
        """Test retrieving messages from a session"""
        session = await conversation_manager.create_session()
        
        # Add several messages
        messages = []
        for i in range(5):
            msg = await conversation_manager.add_message(
                session.session_id,
                "user" if i % 2 == 0 else "assistant",
                f"Message {i}"
            )
            messages.append(msg)
        
        # Get all messages
        retrieved = await conversation_manager.get_messages(session.session_id)
        assert len(retrieved) == 5
        
        # Test limit
        limited = await conversation_manager.get_messages(session.session_id, limit=3)
        assert len(limited) == 3
        
        # Test offset
        offset_messages = await conversation_manager.get_messages(
            session.session_id, 
            limit=2, 
            offset=1
        )
        assert len(offset_messages) == 2
    
    @pytest.mark.asyncio
    async def test_get_conversation_context(self, conversation_manager):
        """Test getting conversation context"""
        session = await conversation_manager.create_session(
            title="Context Test",
            metadata={"test": "data"}
        )
        
        # Add some messages
        await conversation_manager.add_message(session.session_id, "user", "Hello")
        await conversation_manager.add_message(session.session_id, "assistant", "Hi there!")
        
        context = await conversation_manager.get_conversation_context(session.session_id)
        
        assert "session" in context
        assert "messages" in context
        assert "message_count" in context
        assert "analysis_results" in context
        
        assert context["session"]["title"] == "Context Test"
        assert context["session"]["metadata"]["test"] == "data"
        assert len(context["messages"]) == 2
    
    @pytest.mark.asyncio
    async def test_update_session_metadata(self, conversation_manager):
        """Test updating session metadata"""
        session = await conversation_manager.create_session()
        
        # Update metadata
        success = await conversation_manager.update_session_metadata(
            session.session_id,
            {"analysis_count": 3, "user_preference": "detailed"}
        )
        assert success
        
        # Verify update
        updated_session = await conversation_manager.get_session(session.session_id)
        assert updated_session.metadata["analysis_count"] == 3
        assert updated_session.metadata["user_preference"] == "detailed"
    
    @pytest.mark.asyncio
    async def test_delete_session(self, conversation_manager):
        """Test deleting a session"""
        session = await conversation_manager.create_session()
        await conversation_manager.add_message(session.session_id, "user", "Test message")
        
        # Delete session
        success = await conversation_manager.delete_session(session.session_id)
        assert success
        
        # Verify deletion
        deleted_session = await conversation_manager.get_session(session.session_id)
        assert deleted_session is None
    
    @pytest.mark.asyncio
    async def test_get_recent_sessions(self, conversation_manager):
        """Test getting recent sessions"""
        # Create multiple sessions
        sessions = []
        for i in range(3):
            session = await conversation_manager.create_session(
                user_id="test-user",
                title=f"Session {i}"
            )
            sessions.append(session)
        
        # Get recent sessions
        recent = await conversation_manager.get_recent_sessions(user_id="test-user")
        assert len(recent) == 3
        
        # Should be sorted by updated_at descending
        assert recent[0].title == "Session 2"  # Most recent
    
    @pytest.mark.asyncio
    async def test_message_serialization(self):
        """Test message serialization and deserialization"""
        message = Message(
            id="test-id",
            session_id="test-session",
            role="user",
            content="Test content",
            metadata={"test": "value"},
            timestamp=datetime.utcnow()
        )
        
        # Serialize
        message_dict = message.to_dict()
        assert message_dict["id"] == "test-id"
        assert message_dict["role"] == "user"
        assert message_dict["content"] == "Test content"
        
        # Deserialize
        restored = Message.from_dict(message_dict)
        assert restored.id == message.id
        assert restored.role == message.role
        assert restored.content == message.content
        assert restored.metadata == message.metadata
    
    @pytest.mark.asyncio
    async def test_session_serialization(self):
        """Test session serialization and deserialization"""
        session = ConversationSession(
            session_id="test-session",
            user_id="test-user",
            title="Test Session",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            metadata={"test": "data"}
        )
        
        # Serialize
        session_dict = session.to_dict()
        assert session_dict["session_id"] == "test-session"
        assert session_dict["title"] == "Test Session"
        
        # Deserialize
        restored = ConversationSession.from_dict(session_dict)
        assert restored.session_id == session.session_id
        assert restored.title == session.title
        assert restored.metadata == session.metadata

if __name__ == "__main__":
    pytest.main([__file__])
