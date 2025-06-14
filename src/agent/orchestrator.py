"""
Agent Orchestrator
Central coordination system for the AI QA Agent
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from .core.models import (
    AgentState, Message, Goal, AgentResponse, 
    UserProfile, TaskStatus, ReasoningStep
)
from .reasoning.react_engine import ReActReasoner
from .planning.task_planner import TaskPlanner
from .planning.goal_manager import GoalManager
from .memory.conversation_memory import ConversationMemory
from ..core.exceptions import AgentError
from ..chat.conversation_manager import ConversationManager


logger = logging.getLogger(__name__)


class QAAgentOrchestrator:
    """
    Central orchestrator for the AI QA Agent system.
    
    Coordinates between reasoning, planning, goal management, and tool usage
    to provide intelligent assistance to users.
    """

    def __init__(self):
        self.reasoner = ReActReasoner()
        self.planner = TaskPlanner()
        self.goal_manager = GoalManager()
        self.memory = ConversationMemory()
        self.tool_manager = None  # Will be initialized in Sprint 2.2
        
        # Integration with existing conversation system
        self.conversation_manager = ConversationManager()
        
        # Agent configuration
        self.max_session_duration = 3600  # 1 hour
        self.max_reasoning_depth = 10
        self.response_timeout = 30  # seconds

    async def process_user_request(
        self,
        user_input: str,
        session_id: str,
        user_profile: Optional[UserProfile] = None
    ) -> AgentResponse:
        """
        Process a user request using the full agent intelligence system
        
        Args:
            user_input: The user's message or request
            session_id: Conversation session identifier
            user_profile: User's profile and preferences
            
        Returns:
            AgentResponse with reasoning and recommendations
        """
        start_time = datetime.utcnow()

        try:
            # Get or create agent state for this session
            agent_state = await self._get_agent_state(session_id, user_profile)
            
            # Add user message to conversation context
            user_message = Message(role="user", content=user_input)
            agent_state.add_message(user_message)

            # Extract potential goals from user input
            potential_goals = await self.goal_manager.extract_goals_from_input(
                user_input, agent_state
            )

            # Update current goal if needed
            if potential_goals:
                await self._update_current_goal(agent_state, potential_goals)

            # Get available tools (placeholder for Sprint 2.2)
            available_tools = await self._get_available_tools(agent_state)

            # Execute ReAct reasoning cycle
            response = await self.reasoner.reason_and_act(
                user_input, agent_state, available_tools
            )

            # Update agent state with response
            assistant_message = Message(role="assistant", content=response.content)
            agent_state.add_message(assistant_message)

            # Save updated state
            await self._save_agent_state(agent_state)

            # Learn from this interaction
            await self._learn_from_interaction(agent_state, response, user_input)

            duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            logger.info(f"Processed user request in {duration_ms}ms")

            return response

        except Exception as e:
            logger.error(f"Error processing user request: {str(e)}")
            raise AgentError(f"Failed to process request: {str(e)}")

    async def _get_agent_state(
        self,
        session_id: str,
        user_profile: Optional[UserProfile]
    ) -> AgentState:
        """
        Get or create agent state for a session
        """
        # Try to load existing state from memory
        existing_state = await self.memory.get_session_state(session_id)
        
        if existing_state:
            # Update user preferences if profile provided
            if user_profile:
                existing_state.user_preferences.update({
                    "expertise_level": user_profile.expertise_level,
                    "communication_style": user_profile.communication_style,
                    "preferred_tools": user_profile.preferred_tools,
                    "domain_knowledge": user_profile.domain_knowledge
                })
            return existing_state

        # Create new agent state
        state = AgentState(
            session_id=session_id,
            user_preferences=self._extract_user_preferences(user_profile) if user_profile else {}
        )

        return state

    def _extract_user_preferences(self, user_profile: UserProfile) -> Dict[str, Any]:
        """Extract user preferences from profile"""
        return {
            "expertise_level": user_profile.expertise_level,
            "communication_style": user_profile.communication_style,
            "preferred_tools": user_profile.preferred_tools,
            "domain_knowledge": user_profile.domain_knowledge,
            "learning_goals": user_profile.learning_goals
        }

    async def _update_current_goal(
        self,
        agent_state: AgentState,
        potential_goals: List[Goal]
    ) -> None:
        """
        Update the current goal based on new potential goals
        """
        if not potential_goals:
            return

        # Prioritize goals
        prioritized_goals = await self.goal_manager.prioritize_goals(potential_goals)
        highest_priority_goal = prioritized_goals[0]

        # Set as current goal if no current goal or higher priority
        if (not agent_state.current_goal or 
            highest_priority_goal.priority < agent_state.current_goal.priority):
            await self.goal_manager.set_current_goal(agent_state, highest_priority_goal)

    async def _get_available_tools(self, agent_state: AgentState) -> List[str]:
        """
        Get list of available tools for this session
        (Placeholder for Sprint 2.2 tool system)
        """
        # Basic tools that are always available
        basic_tools = [
            "conversation",
            "explanation",
            "analysis_request",
            "help_provider"
        ]

        # Add tools based on user preferences
        if "code" in str(agent_state.user_preferences.get("domain_knowledge", {})):
            basic_tools.extend(["code_analyzer", "test_generator"])

        return basic_tools

    async def _save_agent_state(self, agent_state: AgentState) -> None:
        """
        Save agent state to memory
        """
        await self.memory.save_session_state(agent_state)

    async def _learn_from_interaction(
        self,
        agent_state: AgentState,
        response: AgentResponse,
        user_input: str
    ) -> None:
        """
        Learn from this interaction to improve future responses
        (Foundation for Sprint 2.3 learning system)
        """
        # Extract learning opportunities
        learning_data = {
            "user_input": user_input,
            "response_confidence": response.confidence,
            "reasoning_steps": len(response.reasoning_steps),
            "tools_used": response.tools_used,
            "session_context": len(agent_state.conversation_context)
        }

        # Store learning data for future analysis
        agent_state.session_memory["learning_history"] = (
            agent_state.session_memory.get("learning_history", []) + [learning_data]
        )

        # Update user preferences based on interaction
        await self._update_user_preferences(agent_state, user_input, response)

    async def _update_user_preferences(
        self,
        agent_state: AgentState,
        user_input: str,
        response: AgentResponse
    ) -> None:
        """
        Update user preferences based on interaction patterns
        """
        # Track communication style preferences
        if response.confidence > 0.8 and len(response.reasoning_steps) > 3:
            # User seems to appreciate detailed reasoning
            current_style = agent_state.user_preferences.get("communication_style", "balanced")
            if current_style == "direct":
                agent_state.user_preferences["communication_style"] = "balanced"
            elif current_style == "balanced":
                agent_state.user_preferences["communication_style"] = "detailed"

        # Track domain interests
        domain_keywords = {
            "testing": ["test", "testing", "coverage", "unit test"],
            "quality": ["quality", "clean", "maintainable", "refactor"],
            "security": ["security", "vulnerability", "authentication", "authorization"],
            "performance": ["performance", "optimization", "speed", "memory"]
        }

        user_domains = agent_state.user_preferences.get("domain_knowledge", {})
        for domain, keywords in domain_keywords.items():
            if any(keyword in user_input.lower() for keyword in keywords):
                current_level = user_domains.get(domain, 0.5)
                user_domains[domain] = min(1.0, current_level + 0.1)

        agent_state.user_preferences["domain_knowledge"] = user_domains

    async def get_session_insights(self, session_id: str) -> Dict[str, Any]:
        """
        Get insights about a conversation session
        """
        agent_state = await self.memory.get_session_state(session_id)
        if not agent_state:
            return {"error": "Session not found"}

        insights = {
            "session_id": session_id,
            "created_at": agent_state.created_at,
            "last_updated": agent_state.last_updated,
            "message_count": len(agent_state.conversation_context),
            "reasoning_steps": len(agent_state.reasoning_history),
            "current_goal": agent_state.current_goal.description if agent_state.current_goal else None,
            "user_preferences": agent_state.user_preferences,
            "session_duration": (datetime.utcnow() - agent_state.created_at).total_seconds()
        }

        # Add reasoning quality insights
        if agent_state.reasoning_history:
            avg_confidence = sum(step.confidence for step in agent_state.reasoning_history) / len(agent_state.reasoning_history)
            insights["avg_reasoning_confidence"] = avg_confidence

        # Add learning insights
        learning_history = agent_state.session_memory.get("learning_history", [])
        if learning_history:
            insights["learning_events"] = len(learning_history)
            insights["avg_response_confidence"] = sum(
                event["response_confidence"] for event in learning_history
            ) / len(learning_history)

        return insights

    async def suggest_next_actions(self, session_id: str) -> List[str]:
        """
        Suggest next actions for the user based on conversation context
        """
        agent_state = await self.memory.get_session_state(session_id)
        if not agent_state:
            return ["Start a new conversation"]

        suggestions = []

        # Suggest based on current goal
        if agent_state.current_goal:
            if agent_state.current_goal.status == TaskStatus.EXECUTING:
                suggestions.append(f"Continue working on: {agent_state.current_goal.description}")
            elif agent_state.current_goal.status == TaskStatus.COMPLETED:
                suggestions.append("Set a new goal for our conversation")

        # Suggest based on conversation patterns
        recent_messages = agent_state.get_recent_context(3)
        if recent_messages:
            last_topics = [msg.content.lower() for msg in recent_messages[-2:]]
            
            if any("analyze" in topic for topic in last_topics):
                suggestions.append("Share code for analysis")
                suggestions.append("Ask for specific analysis insights")
            
            if any("test" in topic for topic in last_topics):
                suggestions.append("Request test generation")
                suggestions.append("Ask about testing best practices")

        # Default suggestions
        if not suggestions:
            suggestions.extend([
                "Ask me to analyze your code",
                "Request help with testing strategy",
                "Ask for explanation of testing concepts",
                "Share a specific coding challenge"
            ])

        return suggestions[:4]  # Limit to 4 suggestions
