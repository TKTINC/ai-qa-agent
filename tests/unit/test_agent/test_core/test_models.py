"""
Test Agent Core Models
"""

import pytest
from datetime import datetime, timedelta
from src.agent.core.models import (
    Message, Goal, ReasoningStep, AgentState, UserProfile,
    AgentResponse, TaskPlan, TaskStatus, ReasoningType
)


class TestMessage:
    def test_message_creation(self):
        """Test message creation with defaults"""
        message = Message(role="user", content="Hello")
        
        assert message.role == "user"
        assert message.content == "Hello"
        assert message.id is not None
        assert isinstance(message.timestamp, datetime)
        assert message.metadata == {}

    def test_message_with_metadata(self):
        """Test message creation with metadata"""
        metadata = {"source": "test", "confidence": 0.9}
        message = Message(role="assistant", content="Response", metadata=metadata)
        
        assert message.metadata == metadata


class TestGoal:
    def test_goal_creation(self):
        """Test goal creation with defaults"""
        goal = Goal(description="Test goal")
        
        assert goal.description == "Test goal"
        assert goal.priority == 1
        assert goal.status == TaskStatus.PENDING
        assert isinstance(goal.created_at, datetime)
        assert goal.target_completion is None
        assert goal.context == {}
        assert goal.sub_goals == []

    def test_goal_with_all_fields(self):
        """Test goal creation with all fields"""
        target_time = datetime.utcnow() + timedelta(hours=2)
        context = {"source": "user_input"}
        
        goal = Goal(
            description="Complex goal",
            priority=3,
            target_completion=target_time,
            context=context
        )
        
        assert goal.description == "Complex goal"
        assert goal.priority == 3
        assert goal.target_completion == target_time
        assert goal.context == context


class TestReasoningStep:
    def test_reasoning_step_creation(self):
        """Test reasoning step creation"""
        step = ReasoningStep(
            type=ReasoningType.THOUGHT,
            content="Thinking about the problem",
            confidence=0.8
        )
        
        assert step.type == ReasoningType.THOUGHT
        assert step.content == "Thinking about the problem"
        assert step.confidence == 0.8
        assert isinstance(step.timestamp, datetime)
        assert step.context == {}
        assert step.tools_used == []
        assert step.duration_ms is None

    def test_reasoning_step_with_tools(self):
        """Test reasoning step with tools"""
        step = ReasoningStep(
            type=ReasoningType.ACTION,
            content="Using tools",
            confidence=0.9,
            tools_used=["code_analyzer", "test_generator"]
        )
        
        assert step.tools_used == ["code_analyzer", "test_generator"]


class TestAgentState:
    def test_agent_state_creation(self):
        """Test agent state creation"""
        state = AgentState(session_id="test_session")
        
        assert state.session_id == "test_session"
        assert state.current_goal is None
        assert state.conversation_context == []
        assert state.reasoning_history == []
        assert state.active_tools == []
        assert state.user_preferences == {}
        assert state.session_memory == {}
        assert isinstance(state.created_at, datetime)
        assert isinstance(state.last_updated, datetime)

    def test_add_message(self):
        """Test adding message to agent state"""
        state = AgentState(session_id="test_session")
        message = Message(role="user", content="Hello")
        
        initial_time = state.last_updated
        state.add_message(message)
        
        assert len(state.conversation_context) == 1
        assert state.conversation_context[0] == message
        assert state.last_updated > initial_time

    def test_add_reasoning_step(self):
        """Test adding reasoning step to agent state"""
        state = AgentState(session_id="test_session")
        step = ReasoningStep(
            type=ReasoningType.THOUGHT,
            content="Test thought",
            confidence=0.8
        )
        
        initial_time = state.last_updated
        state.add_reasoning_step(step)
        
        assert len(state.reasoning_history) == 1
        assert state.reasoning_history[0] == step
        assert state.last_updated > initial_time

    def test_get_recent_context(self):
        """Test getting recent conversation context"""
        state = AgentState(session_id="test_session")
        
        # Add multiple messages
        for i in range(15):
            message = Message(role="user" if i % 2 == 0 else "assistant", content=f"Message {i}")
            state.add_message(message)
        
        recent = state.get_recent_context(5)
        assert len(recent) == 5
        assert recent[-1].content == "Message 14"
        assert recent[0].content == "Message 10"


class TestUserProfile:
    def test_user_profile_creation(self):
        """Test user profile creation"""
        profile = UserProfile(user_id="test_user")
        
        assert profile.user_id == "test_user"
        assert profile.expertise_level == "intermediate"
        assert profile.communication_style == "balanced"
        assert profile.preferred_tools == []
        assert profile.domain_knowledge == {}
        assert profile.learning_goals == []
        assert profile.successful_patterns == []
        assert isinstance(profile.created_at, datetime)

    def test_user_profile_with_preferences(self):
        """Test user profile with preferences"""
        domain_knowledge = {"python": 0.8, "testing": 0.6}
        preferred_tools = ["pytest", "black"]
        learning_goals = ["TDD", "CI/CD"]
        
        profile = UserProfile(
            user_id="expert_user",
            expertise_level="expert",
            communication_style="direct",
            preferred_tools=preferred_tools,
            domain_knowledge=domain_knowledge,
            learning_goals=learning_goals
        )
        
        assert profile.expertise_level == "expert"
        assert profile.communication_style == "direct"
        assert profile.preferred_tools == preferred_tools
        assert profile.domain_knowledge == domain_knowledge
        assert profile.learning_goals == learning_goals


class TestAgentResponse:
    def test_agent_response_creation(self):
        """Test agent response creation"""
        response = AgentResponse(
            content="Test response",
            session_id="test_session"
        )
        
        assert response.content == "Test response"
        assert response.session_id == "test_session"
        assert response.reasoning_steps == []
        assert response.tools_used == []
        assert response.confidence == 0.8
        assert response.suggestions == []
        assert response.follow_up_questions == []
        assert isinstance(response.timestamp, datetime)
        assert response.metadata == {}

    def test_agent_response_with_reasoning(self):
        """Test agent response with reasoning steps"""
        reasoning_steps = [
            ReasoningStep(
                type=ReasoningType.THOUGHT,
                content="Thinking",
                confidence=0.9
            )
        ]
        
        response = AgentResponse(
            content="Detailed response",
            reasoning_steps=reasoning_steps,
            tools_used=["analyzer"],
            confidence=0.95,
            suggestions=["Try this"],
            follow_up_questions=["What about this?"],
            session_id="test_session"
        )
        
        assert len(response.reasoning_steps) == 1
        assert response.tools_used == ["analyzer"]
        assert response.confidence == 0.95
        assert response.suggestions == ["Try this"]
        assert response.follow_up_questions == ["What about this?"]


class TestTaskPlan:
    def test_task_plan_creation(self):
        """Test task plan creation"""
        goal = Goal(description="Test goal")
        
        plan = TaskPlan(goal=goal)
        
        assert plan.goal == goal
        assert plan.steps == []
        assert plan.estimated_duration is None
        assert plan.required_tools == []
        assert plan.success_criteria == []
        assert isinstance(plan.created_at, datetime)
        assert plan.status == TaskStatus.PENDING

    def test_task_plan_with_steps(self):
        """Test task plan with steps"""
        goal = Goal(description="Complex goal")
        steps = [
            {"step": 1, "action": "analyze"},
            {"step": 2, "action": "generate"}
        ]
        required_tools = ["analyzer", "generator"]
        success_criteria = ["Analysis complete", "Generation successful"]
        
        plan = TaskPlan(
            goal=goal,
            steps=steps,
            estimated_duration=30,
            required_tools=required_tools,
            success_criteria=success_criteria
        )
        
        assert plan.steps == steps
        assert plan.estimated_duration == 30
        assert plan.required_tools == required_tools
        assert plan.success_criteria == success_criteria
