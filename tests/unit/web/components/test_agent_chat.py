"""
Tests for Agent Chat Interface Components
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch

from src.web.components.agent_chat import (
    AgentChatInterface, ConversationManager, ReasoningVisualizationManager,
    AgentVisualization, CollaborationEvent
)

class TestConversationManager:
    """Test conversation management functionality"""
    
    @pytest.fixture
    def conversation_manager(self):
        return ConversationManager()
    
    def test_initial_agent_setup(self, conversation_manager):
        """Test initial agent configuration"""
        assert len(conversation_manager.active_agents) == 5
        
        # Check test architect agent
        test_architect = conversation_manager.active_agents["test_architect"]
        assert test_architect.agent_name == "Test Architect"
        assert test_architect.status == "ready"
        assert test_architect.avatar == "🏗️"
        assert test_architect.confidence == 0.0
    
    @pytest.mark.asyncio
    async def test_update_agent_status(self, conversation_manager):
        """Test updating agent status"""
        await conversation_manager.update_agent_status(
            "test_architect", "active", 0.95, "Analyzing code complexity"
        )
        
        agent = conversation_manager.active_agents["test_architect"]
        assert agent.status == "active"
        assert agent.confidence == 0.95
        assert agent.current_task == "Analyzing code complexity"
    
    @pytest.mark.asyncio
    async def test_add_collaboration_event(self, conversation_manager):
        """Test adding collaboration events"""
        await conversation_manager.add_collaboration_event(
            "test_architect", ["code_reviewer", "performance_analyst"],
            "multi_agent_analysis", "Collaborating on code review"
        )
        
        assert len(conversation_manager.collaboration_events) == 1
        event = conversation_manager.collaboration_events[0]
        assert event.primary_agent == "test_architect"
        assert "code_reviewer" in event.collaborating_agents
        assert event.collaboration_type == "multi_agent_analysis"
    
    @pytest.mark.asyncio
    async def test_collaboration_event_limit(self, conversation_manager):
        """Test collaboration event limit (keep only recent 10)"""
        # Add 15 events
        for i in range(15):
            await conversation_manager.add_collaboration_event(
                "test_architect", ["code_reviewer"],
                "test_collaboration", f"Event {i}"
            )
        
        # Should only keep last 10
        assert len(conversation_manager.collaboration_events) == 10
        # Check that most recent events are kept
        assert conversation_manager.collaboration_events[-1].message == "Event 14"

class TestReasoningVisualizationManager:
    """Test reasoning visualization functionality"""
    
    @pytest.fixture
    def reasoning_manager(self):
        return ReasoningVisualizationManager()
    
    @pytest.mark.asyncio
    async def test_start_reasoning_session(self, reasoning_manager):
        """Test starting a reasoning session"""
        session_id = "test_session_1"
        await reasoning_manager.start_reasoning_session(
            session_id, "test_architect", "Analyzing code complexity"
        )
        
        assert session_id in reasoning_manager.active_reasoning_sessions
        session = reasoning_manager.active_reasoning_sessions[session_id]
        assert session["agent_name"] == "test_architect"
        assert session["status"] == "active"
        assert len(session["steps"]) == 1
    
    @pytest.mark.asyncio
    async def test_add_reasoning_step(self, reasoning_manager):
        """Test adding reasoning steps"""
        session_id = "test_session_1"
        await reasoning_manager.start_reasoning_session(
            session_id, "test_architect", "Initial thought"
        )
        
        await reasoning_manager.add_reasoning_step(
            session_id, "think", "Analyzing the problem...", 
            confidence=0.8
        )
        
        session = reasoning_manager.active_reasoning_sessions[session_id]
        assert len(session["steps"]) == 2
        
        step = session["steps"][-1]
        assert step["step_type"] == "think"
        assert step["thought"] == "Analyzing the problem..."
        assert step["confidence"] == 0.8
    
    @pytest.mark.asyncio
    async def test_complete_reasoning_session(self, reasoning_manager):
        """Test completing a reasoning session"""
        session_id = "test_session_1"
        await reasoning_manager.start_reasoning_session(
            session_id, "test_architect", "Initial thought"
        )
        
        await reasoning_manager.complete_reasoning_session(
            session_id, "Task completed successfully"
        )
        
        # Session should be moved to history
        assert session_id not in reasoning_manager.active_reasoning_sessions
        assert len(reasoning_manager.reasoning_history) == 1
        
        completed_session = reasoning_manager.reasoning_history[0]
        assert completed_session["status"] == "completed"
        assert completed_session["outcome"] == "Task completed successfully"
    
    @pytest.mark.asyncio
    async def test_reasoning_history_limit(self, reasoning_manager):
        """Test reasoning history limit (keep only recent 20)"""
        # Create and complete 25 sessions
        for i in range(25):
            session_id = f"test_session_{i}"
            await reasoning_manager.start_reasoning_session(
                session_id, "test_architect", f"Task {i}"
            )
            await reasoning_manager.complete_reasoning_session(
                session_id, f"Completed task {i}"
            )
        
        # Should only keep last 20
        assert len(reasoning_manager.reasoning_history) == 20

class TestAgentChatInterface:
    """Test main agent chat interface"""
    
    @pytest.fixture
    def chat_interface(self):
        return AgentChatInterface()
    
    @pytest.mark.asyncio
    async def test_requires_collaboration_detection(self, chat_interface):
        """Test detection of messages requiring collaboration"""
        # Messages that should require collaboration
        collaborative_messages = [
            "I need performance and security analysis",
            "Can you do a comprehensive review of my code?",
            "I want complete analysis of multiple aspects",
            "Please review architecture and quality together"
        ]
        
        for message in collaborative_messages:
            result = await chat_interface._requires_collaboration(message)
            assert result, f"Message should require collaboration: {message}"
        
        # Messages that should not require collaboration
        single_agent_messages = [
            "How do I write unit tests?",
            "Can you help with documentation?",
            "What's the best way to optimize this function?"
        ]
        
        for message in single_agent_messages:
            result = await chat_interface._requires_collaboration(message)
            assert not result, f"Message should not require collaboration: {message}"
    
    @pytest.mark.asyncio
    async def test_select_agents_for_collaboration(self, chat_interface):
        """Test agent selection for collaboration"""
        message = "I need testing strategy, performance optimization, and security review"
        agents = await chat_interface._select_agents_for_collaboration(message)
        
        expected_agents = ["test_architect", "performance_analyst", "security_specialist"]
        assert all(agent in agents for agent in expected_agents)
        assert len(agents) <= 3  # Should limit to 3 agents
    
    @pytest.mark.asyncio
    async def test_select_best_agent(self, chat_interface):
        """Test single agent selection"""
        test_cases = [
            ("My code is running slowly", "performance_analyst"),
            ("I'm worried about security vulnerabilities", "security_specialist"),
            ("Can you review my code quality?", "code_reviewer"),
            ("Help me write documentation", "documentation_expert"),
            ("I need testing help", "test_architect")
        ]
        
        for message, expected_agent in test_cases:
            selected_agent = await chat_interface._select_best_agent(message)
            assert selected_agent == expected_agent, f"Wrong agent for '{message}'"
    
    @pytest.mark.asyncio
    async def test_get_agent_contribution(self, chat_interface):
        """Test getting agent contributions"""
        message = "Help me test my authentication system"
        
        # Test different agent contributions
        agents = ["test_architect", "performance_analyst", "security_specialist"]
        
        for agent in agents:
            contribution = await chat_interface._get_agent_contribution(
                agent, message, "test_session"
            )
            
            assert "response" in contribution
            assert "summary" in contribution
            assert "confidence" in contribution
            assert "reasoning" in contribution
            
            # Check response is agent-specific
            assert len(contribution["response"]) > 50
            assert 0.8 <= contribution["confidence"] <= 1.0
    
    @pytest.mark.asyncio
    async def test_synthesize_collaborative_response(self, chat_interface):
        """Test synthesizing multiple agent responses"""
        agent_responses = {
            "test_architect": {
                "response": "Test strategy recommendation...",
                "summary": "Testing strategy",
                "confidence": 0.92
            },
            "security_specialist": {
                "response": "Security analysis results...",
                "summary": "Security assessment",
                "confidence": 0.94
            }
        }
        
        message = "Help me with secure testing"
        result = await chat_interface._synthesize_collaborative_response(
            agent_responses, message
        )
        
        assert "Test Architect" in result
        assert "Security Specialist" in result
        assert "Coordinated Recommendation" in result
        assert len(result) > 200  # Should be substantial response
    
    @pytest.mark.asyncio
    async def test_websocket_connection_management(self, chat_interface):
        """Test WebSocket connection management"""
        # Mock WebSocket
        mock_websocket = Mock()
        mock_websocket.accept = AsyncMock()
        mock_websocket.send_json = AsyncMock()
        
        session_id = "test_session"
        
        # Test connection
        await chat_interface.connect_websocket(mock_websocket, session_id)
        
        assert session_id in chat_interface.websocket_connections
        assert chat_interface.websocket_connections[session_id] == mock_websocket
        mock_websocket.accept.assert_called_once()
        mock_websocket.send_json.assert_called_once()
        
        # Test disconnection
        await chat_interface.disconnect_websocket(session_id)
        assert session_id not in chat_interface.websocket_connections
    
    @pytest.mark.asyncio
    async def test_get_conversation_context(self, chat_interface):
        """Test getting conversation context"""
        session_id = "test_session"
        
        # Add some test data
        await chat_interface.conversation_manager.add_collaboration_event(
            "test_architect", ["code_reviewer"], "test_collaboration", "Test event"
        )
        
        context = await chat_interface.get_conversation_context(session_id)
        
        assert context["session_id"] == session_id
        assert "active_agents" in context
        assert "recent_collaborations" in context
        assert "reasoning_sessions" in context
        
        # Check active agents
        assert len(context["active_agents"]) == 5
        
        # Check recent collaborations
        assert len(context["recent_collaborations"]) == 1
        assert context["recent_collaborations"][0]["primary_agent"] == "test_architect"

if __name__ == "__main__":
    pytest.main([__file__])
