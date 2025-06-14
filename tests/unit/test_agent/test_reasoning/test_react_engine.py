"""
Test ReAct Engine
"""

import pytest
import asyncio
from unittest.mock import Mock, patch
from datetime import datetime

from src.agent.reasoning.react_engine import ReActReasoner
from src.agent.core.models import (
    AgentState, Message, Goal, ReasoningStep, ReasoningType,
    TaskStatus, UserProfile
)


class TestReActReasoner:
    
    @pytest.fixture
    def reasoner(self):
        """Create ReAct reasoner instance"""
        return ReActReasoner()
    
    @pytest.fixture
    def agent_state(self):
        """Create test agent state"""
        return AgentState(session_id="test_session")
    
    @pytest.fixture
    def available_tools(self):
        """Available tools for testing"""
        return ["code_analyzer", "test_generator", "explanation"]

    @pytest.mark.asyncio
    async def test_reason_and_act_basic(self, reasoner, agent_state, available_tools):
        """Test basic ReAct cycle"""
        user_input = "Help me analyze my code"
        
        response = await reasoner.reason_and_act(user_input, agent_state, available_tools)
        
        assert response.content is not None
        assert len(response.content) > 0
        assert response.session_id == agent_state.session_id
        assert response.confidence > 0
        assert len(response.reasoning_steps) >= 4  # Should have multiple reasoning steps
        
        # Check reasoning step types
        step_types = [step.type for step in response.reasoning_steps]
        assert ReasoningType.OBSERVATION in step_types
        assert ReasoningType.THOUGHT in step_types
        assert ReasoningType.PLAN in step_types
        assert ReasoningType.ACTION in step_types

    @pytest.mark.asyncio
    async def test_observation_step(self, reasoner, agent_state, available_tools):
        """Test observation step generation"""
        user_input = "I need help with testing"
        
        observation = await reasoner._observe_situation(user_input, agent_state, available_tools)
        
        assert observation.type == ReasoningType.OBSERVATION
        assert "testing" in observation.content.lower()
        assert observation.confidence > 0.5
        assert "available tools" in observation.content.lower()

    @pytest.mark.asyncio
    async def test_thought_generation(self, reasoner, agent_state, available_tools):
        """Test thought generation step"""
        user_input = "Analyze my complex authentication system"
        observation = ReasoningStep(
            type=ReasoningType.OBSERVATION,
            content="User wants analysis",
            confidence=0.9
        )
        
        thought = await reasoner._generate_thoughts(user_input, agent_state, observation)
        
        assert thought.type == ReasoningType.THOUGHT
        assert "complex" in thought.content.lower() or "analysis" in thought.content.lower()
        assert thought.confidence > 0.5

    @pytest.mark.asyncio
    async def test_plan_creation(self, reasoner, agent_state, available_tools):
        """Test plan creation step"""
        user_input = "Generate tests for my API"
        thought = ReasoningStep(
            type=ReasoningType.THOUGHT,
            content="User wants test generation",
            confidence=0.8
        )
        
        plan = await reasoner._create_plan(user_input, agent_state, thought)
        
        assert plan.type == ReasoningType.PLAN
        assert "plan" in plan.content.lower() or "step" in plan.content.lower()
        assert plan.confidence > 0.5

    @pytest.mark.asyncio
    async def test_action_execution(self, reasoner, agent_state, available_tools):
        """Test action execution step"""
        user_input = "Help me understand testing"
        plan = ReasoningStep(
            type=ReasoningType.PLAN,
            content="Execute explanation plan",
            confidence=0.8,
            context={"plan_steps": ["explain", "provide examples"]}
        )
        
        action = await reasoner._execute_actions(user_input, agent_state, plan, available_tools)
        
        assert action.type == ReasoningType.ACTION
        assert len(action.context.get("actions", [])) > 0
        assert action.confidence > 0.5

    @pytest.mark.asyncio
    async def test_reflection_step(self, reasoner, agent_state, available_tools):
        """Test reflection step"""
        user_input = "Test request"
        reasoning_steps = [
            ReasoningStep(type=ReasoningType.OBSERVATION, content="Observed", confidence=0.9),
            ReasoningStep(type=ReasoningType.THOUGHT, content="Thought", confidence=0.8),
            ReasoningStep(type=ReasoningType.PLAN, content="Planned", confidence=0.85),
            ReasoningStep(type=ReasoningType.ACTION, content="Acted", confidence=0.9)
        ]
        
        reflection = await reasoner._reflect_on_outcome(user_input, agent_state, reasoning_steps)
        
        assert reflection.type == ReasoningType.REFLECTION
        assert "confidence" in reflection.content.lower() or "reasoning" in reflection.content.lower()
        assert reflection.confidence > 0.5

    @pytest.mark.asyncio
    async def test_response_generation_analysis(self, reasoner, agent_state, available_tools):
        """Test response generation for analysis requests"""
        user_input = "Please analyze my code structure"
        reasoning_steps = []
        
        response = await reasoner._generate_response(user_input, agent_state, reasoning_steps)
        
        assert "analyze" in response.lower()
        assert len(response) > 50  # Should be substantial response

    @pytest.mark.asyncio
    async def test_response_generation_explanation(self, reasoner, agent_state, available_tools):
        """Test response generation for explanation requests"""
        user_input = "Can you explain unit testing best practices?"
        reasoning_steps = []
        
        response = await reasoner._generate_response(user_input, agent_state, reasoning_steps)
        
        assert "explain" in response.lower() or "testing" in response.lower()
        assert len(response) > 50

    @pytest.mark.asyncio
    async def test_response_generation_help(self, reasoner, agent_state, available_tools):
        """Test response generation for help requests"""
        user_input = "I need help with my project"
        reasoning_steps = []
        
        response = await reasoner._generate_response(user_input, agent_state, reasoning_steps)
        
        assert "help" in response.lower()
        assert len(response) > 50

    @pytest.mark.asyncio
    async def test_suggestions_generation(self, reasoner, agent_state, available_tools):
        """Test suggestions generation"""
        reasoning_steps = [
            ReasoningStep(type=ReasoningType.THOUGHT, content="Good reasoning", confidence=0.9)
        ]
        
        suggestions = await reasoner._generate_suggestions(agent_state, reasoning_steps)
        
        assert isinstance(suggestions, list)
        assert len(suggestions) > 0
        assert all(isinstance(suggestion, str) for suggestion in suggestions)

    @pytest.mark.asyncio
    async def test_follow_ups_analysis(self, reasoner, agent_state, available_tools):
        """Test follow-up generation for analysis requests"""
        user_input = "Analyze my code"
        
        follow_ups = await reasoner._generate_follow_ups(user_input, agent_state)
        
        assert isinstance(follow_ups, list)
        assert len(follow_ups) <= 3
        assert any("code" in followup.lower() or "analysis" in followup.lower() for followup in follow_ups)

    @pytest.mark.asyncio
    async def test_follow_ups_testing(self, reasoner, agent_state, available_tools):
        """Test follow-up generation for testing requests"""
        user_input = "I want to improve my tests"
        
        follow_ups = await reasoner._generate_follow_ups(user_input, agent_state)
        
        assert isinstance(follow_ups, list)
        assert len(follow_ups) <= 3
        assert any("test" in followup.lower() for followup in follow_ups)

    def test_extract_tools_used(self, reasoner):
        """Test tool extraction from reasoning steps"""
        reasoning_steps = [
            ReasoningStep(type=ReasoningType.ACTION, content="Action 1", confidence=0.8, tools_used=["tool1"]),
            ReasoningStep(type=ReasoningType.ACTION, content="Action 2", confidence=0.8, tools_used=["tool2", "tool3"]),
            ReasoningStep(type=ReasoningType.THOUGHT, content="Thought", confidence=0.8, tools_used=[])
        ]
        
        tools = reasoner._extract_tools_used(reasoning_steps)
        
        assert set(tools) == {"tool1", "tool2", "tool3"}

    def test_calculate_confidence(self, reasoner):
        """Test confidence calculation"""
        reasoning_steps = [
            ReasoningStep(type=ReasoningType.OBSERVATION, content="Step 1", confidence=0.9),
            ReasoningStep(type=ReasoningType.THOUGHT, content="Step 2", confidence=0.8),
            ReasoningStep(type=ReasoningType.PLAN, content="Step 3", confidence=0.85),
            ReasoningStep(type=ReasoningType.ACTION, content="Step 4", confidence=0.9),
            ReasoningStep(type=ReasoningType.REFLECTION, content="Step 5", confidence=0.8)
        ]
        
        confidence = reasoner._calculate_confidence(reasoning_steps)
        
        # Should be average of confidences, boosted for complete cycle
        expected = (0.9 + 0.8 + 0.85 + 0.9 + 0.8) / 5 * 1.1
        assert abs(confidence - min(1.0, expected)) < 0.01

    def test_calculate_confidence_empty(self, reasoner):
        """Test confidence calculation with no steps"""
        confidence = reasoner._calculate_confidence([])
        assert confidence == 0.5

    @pytest.mark.asyncio
    async def test_conversation_context_influence(self, reasoner, agent_state, available_tools):
        """Test that conversation context influences reasoning"""
        # Add previous conversation
        agent_state.add_message(Message(role="user", content="I'm working on a Python project"))
        agent_state.add_message(Message(role="assistant", content="Great! I can help with Python development"))
        
        user_input = "Now I need help with testing"
        
        response = await reasoner.reason_and_act(user_input, agent_state, available_tools)
        
        # Should reference previous context
        observation_step = next(step for step in response.reasoning_steps if step.type == ReasoningType.OBSERVATION)
        assert "message" in observation_step.content.lower() or "context" in observation_step.content.lower()

    @pytest.mark.asyncio 
    async def test_goal_influence_on_reasoning(self, reasoner, agent_state, available_tools):
        """Test that current goal influences reasoning"""
        goal = Goal(description="Improve code quality through better testing")
        agent_state.current_goal = goal
        
        user_input = "What should I do next?"
        
        response = await reasoner.reason_and_act(user_input, agent_state, available_tools)
        
        # Should reference the current goal
        observation_step = next(step for step in response.reasoning_steps if step.type == ReasoningType.OBSERVATION)
        assert "goal" in observation_step.content.lower()

    @pytest.mark.asyncio
    async def test_user_preferences_adaptation(self, reasoner, available_tools):
        """Test adaptation to user preferences"""
        agent_state = AgentState(
            session_id="test_session",
            user_preferences={
                "expertise_level": "expert",
                "communication_style": "direct"
            }
        )
        
        user_input = "Explain testing strategies"
        
        response = await reasoner.reason_and_act(user_input, agent_state, available_tools)
        
        # Should consider user expertise level in reasoning
        thought_step = next(step for step in response.reasoning_steps if step.type == ReasoningType.THOUGHT)
        assert "expert" in thought_step.content.lower() or "expertise" in thought_step.content.lower()

    @pytest.mark.asyncio
    async def test_error_handling(self, reasoner, agent_state, available_tools):
        """Test error handling in reasoning process"""
        # Test with problematic input
        user_input = ""
        
        # Should handle gracefully without crashing
        response = await reasoner.reason_and_act(user_input, agent_state, available_tools)
        
        assert response is not None
        assert response.content is not None
        assert response.confidence >= 0
