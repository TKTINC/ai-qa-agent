"""
Goal Management System
Manages user goals, priorities, and objectives across conversation sessions
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from ..core.models import Goal, TaskStatus, AgentState, UserProfile
from ...core.exceptions import AgentError


logger = logging.getLogger(__name__)


class GoalManager:
    """
    Manages user goals and objectives, tracking progress and priorities
    across conversation sessions.
    """

    def __init__(self):
        self.max_active_goals = 5
        self.goal_timeout_hours = 24
        self.priority_levels = {1: "Critical", 2: "High", 3: "Medium", 4: "Low", 5: "Optional"}

    async def create_goal(
        self,
        description: str,
        priority: int = 3,
        context: Optional[Dict[str, Any]] = None,
        target_completion: Optional[datetime] = None
    ) -> Goal:
        """
        Create a new goal from user input
        
        Args:
            description: Goal description
            priority: Priority level (1-5, 1 is highest)
            context: Additional context about the goal
            target_completion: When the goal should be completed
            
        Returns:
            Created Goal object
        """
        if priority < 1 or priority > 5:
            raise AgentError("Priority must be between 1 (highest) and 5 (lowest)")

        goal = Goal(
            description=description,
            priority=priority,
            context=context or {},
            target_completion=target_completion
        )

        logger.info(f"Created goal: {description} (Priority: {self.priority_levels[priority]})")
        return goal

    async def extract_goals_from_input(
        self,
        user_input: str,
        agent_state: AgentState
    ) -> List[Goal]:
        """
        Extract potential goals from user input using natural language understanding
        """
        goals = []
        input_lower = user_input.lower()

        # Look for explicit goal statements
        goal_indicators = [
            "i want to", "i need to", "help me", "i'd like to",
            "my goal is", "i'm trying to", "can you help me"
        ]

        # Look for action-oriented requests
        action_patterns = [
            ("analyze", "Analyze code for quality and testing opportunities"),
            ("improve", "Improve code quality and test coverage"),
            ("generate", "Generate comprehensive test suite"),
            ("review", "Review code for issues and improvements"),
            ("optimize", "Optimize code performance and maintainability"),
            ("explain", "Understand testing concepts and best practices"),
            ("fix", "Fix identified issues and problems"),
            ("test", "Create effective testing strategy")
        ]

        # Extract explicit goals
        for indicator in goal_indicators:
            if indicator in input_lower:
                # Extract the part after the indicator as potential goal
                parts = input_lower.split(indicator, 1)
                if len(parts) > 1:
                    goal_text = parts[1].strip()
                    if len(goal_text) > 5:  # Minimum meaningful goal length
                        priority = self._determine_priority_from_text(goal_text)
                        goal = await self.create_goal(
                            description=goal_text.capitalize(),
                            priority=priority,
                            context={"source": "explicit_statement", "original_input": user_input}
                        )
                        goals.append(goal)

        # Extract implicit goals from action patterns
        for action, goal_template in action_patterns:
            if action in input_lower:
                priority = self._determine_priority_from_text(user_input)
                context = {
                    "source": "action_pattern",
                    "action": action,
                    "original_input": user_input
                }
                
                # Customize goal based on context
                if "code" in input_lower or "file" in input_lower:
                    context["involves_code"] = True
                if "urgent" in input_lower or "asap" in input_lower:
                    priority = min(priority, 2)  # Increase priority

                goal = await self.create_goal(
                    description=goal_template,
                    priority=priority,
                    context=context
                )
                goals.append(goal)
                break  # Only create one goal per input to avoid duplicates

        return goals

    def _determine_priority_from_text(self, text: str) -> int:
        """
        Determine goal priority based on text content
        """
        text_lower = text.lower()
        
        # High priority indicators
        if any(word in text_lower for word in ["urgent", "critical", "asap", "immediately", "emergency"]):
            return 1
        
        # Medium-high priority indicators
        if any(word in text_lower for word in ["important", "soon", "quickly", "priority"]):
            return 2
        
        # Low priority indicators
        if any(word in text_lower for word in ["when possible", "eventually", "sometime", "optional"]):
            return 4
        
        # Default to medium priority
        return 3

    async def set_current_goal(self, agent_state: AgentState, goal: Goal) -> None:
        """
        Set the current active goal for the agent
        """
        previous_goal = agent_state.current_goal
        agent_state.current_goal = goal
        goal.status = TaskStatus.EXECUTING

        if previous_goal:
            logger.info(f"Switched from goal '{previous_goal.description}' to '{goal.description}'")
        else:
            logger.info(f"Set current goal: '{goal.description}'")

    async def complete_goal(self, goal: Goal, agent_state: AgentState) -> Dict[str, Any]:
        """
        Mark a goal as completed and provide completion summary
        """
        goal.status = TaskStatus.COMPLETED
        completion_time = datetime.utcnow()
        
        # Clear current goal if this was the active one
        if agent_state.current_goal and agent_state.current_goal.id == goal.id:
            agent_state.current_goal = None

        completion_summary = {
            "goal_id": goal.id,
            "description": goal.description,
            "completion_time": completion_time,
            "duration": completion_time - goal.created_at,
            "priority": goal.priority,
            "success": True
        }

        logger.info(f"Goal completed: '{goal.description}' (Duration: {completion_summary['duration']})")
        return completion_summary

    async def prioritize_goals(self, goals: List[Goal], user_profile: Optional[UserProfile] = None) -> List[Goal]:
        """
        Prioritize goals based on priority, urgency, and user profile
        """
        def priority_score(goal: Goal) -> tuple:
            # Primary sort by priority (lower number = higher priority)
            primary = goal.priority
            
            # Secondary sort by creation time (newer first for same priority)
            secondary = -goal.created_at.timestamp()
            
            # Tertiary sort by target completion (sooner first)
            tertiary = 0
            if goal.target_completion:
                time_to_deadline = (goal.target_completion - datetime.utcnow()).total_seconds()
                tertiary = time_to_deadline
            
            return (primary, secondary, tertiary)

        sorted_goals = sorted(goals, key=priority_score)
        
        # Apply user profile preferences if available
        if user_profile:
            sorted_goals = await self._apply_user_preferences(sorted_goals, user_profile)

        return sorted_goals

    async def _apply_user_preferences(self, goals: List[Goal], user_profile: UserProfile) -> List[Goal]:
        """
        Adjust goal prioritization based on user preferences
        """
        # Boost priority for goals matching user's domain knowledge
        for goal in goals:
            goal_text = goal.description.lower()
            
            # Check if goal aligns with user's strong domains
            for domain, knowledge_level in user_profile.domain_knowledge.items():
                if domain.lower() in goal_text and knowledge_level > 0.7:
                    # User is knowledgeable in this area, might want to focus here
                    if goal.priority > 2:
                        goal.priority = max(1, goal.priority - 1)
                        
            # Check if goal aligns with learning goals
            for learning_goal in user_profile.learning_goals:
                if any(word in goal_text for word in learning_goal.lower().split()):
                    # This goal supports user's learning objectives
                    if goal.priority > 2:
                        goal.priority = max(1, goal.priority - 1)

        return goals

    async def suggest_next_goal(
        self,
        agent_state: AgentState,
        user_profile: Optional[UserProfile] = None,
        recent_goals: Optional[List[Goal]] = None
    ) -> Optional[Goal]:
        """
        Suggest the next goal to work on based on context and history
        """
        suggestions = []

        # If no current goal, suggest based on conversation context
        if not agent_state.current_goal:
            recent_messages = agent_state.get_recent_context(3)
            if recent_messages:
                # Analyze recent conversation for goal opportunities
                last_message = recent_messages[-1].content.lower()
                
                if "analyze" in last_message and "code" in last_message:
                    suggestions.append(await self.create_goal(
                        "Perform comprehensive code analysis",
                        priority=2,
                        context={"source": "conversation_analysis"}
                    ))
                
                elif "test" in last_message:
                    suggestions.append(await self.create_goal(
                        "Develop comprehensive testing strategy",
                        priority=2,
                        context={"source": "conversation_analysis"}
                    ))

        # Suggest based on user profile if available
        if user_profile:
            for learning_goal in user_profile.learning_goals:
                suggestions.append(await self.create_goal(
                    f"Learn about {learning_goal}",
                    priority=3,
                    context={"source": "user_learning_goals"}
                ))

        # Suggest based on successful patterns
        if recent_goals:
            completed_goals = [g for g in recent_goals if g.status == TaskStatus.COMPLETED]
            if completed_goals:
                # Find patterns in successful goals
                common_themes = self._extract_goal_themes(completed_goals)
                for theme in common_themes:
                    suggestions.append(await self.create_goal(
                        f"Continue working on {theme} improvements",
                        priority=3,
                        context={"source": "successful_patterns", "theme": theme}
                    ))

        # Return highest priority suggestion
        if suggestions:
            prioritized = await self.prioritize_goals(suggestions, user_profile)
            return prioritized[0]

        return None

    def _extract_goal_themes(self, goals: List[Goal]) -> List[str]:
        """
        Extract common themes from a list of goals
        """
        themes = []
        theme_keywords = {
            "testing": ["test", "testing", "coverage"],
            "quality": ["quality", "improve", "optimization"],
            "analysis": ["analyze", "review", "examine"],
            "learning": ["learn", "understand", "explain"]
        }

        for theme, keywords in theme_keywords.items():
            theme_count = sum(
                1 for goal in goals 
                if any(keyword in goal.description.lower() for keyword in keywords)
            )
            if theme_count >= 2:  # At least 2 goals with this theme
                themes.append(theme)

        return themes

    async def track_goal_progress(self, goal: Goal) -> Dict[str, Any]:
        """
        Track progress on a specific goal
        """
        progress_info = {
            "goal_id": goal.id,
            "description": goal.description,
            "status": goal.status,
            "priority": goal.priority,
            "created_at": goal.created_at,
            "elapsed_time": datetime.utcnow() - goal.created_at,
            "progress_percentage": self._calculate_progress_percentage(goal)
        }

        if goal.target_completion:
            time_remaining = goal.target_completion - datetime.utcnow()
            progress_info["time_remaining"] = time_remaining
            progress_info["on_track"] = time_remaining.total_seconds() > 0

        return progress_info

    def _calculate_progress_percentage(self, goal: Goal) -> int:
        """
        Calculate estimated progress percentage for a goal
        """
        if goal.status == TaskStatus.COMPLETED:
            return 100
        elif goal.status == TaskStatus.FAILED or goal.status == TaskStatus.CANCELLED:
            return 0
        elif goal.status == TaskStatus.EXECUTING:
            # Estimate based on time elapsed and typical goal duration
            elapsed = datetime.utcnow() - goal.created_at
            estimated_duration = timedelta(hours=2)  # Default estimate
            
            if goal.target_completion:
                estimated_duration = goal.target_completion - goal.created_at
            
            progress = min(90, int((elapsed.total_seconds() / estimated_duration.total_seconds()) * 100))
            return progress
        else:
            return 0

    async def cleanup_old_goals(self, goals: List[Goal]) -> List[Goal]:
        """
        Clean up old or stale goals
        """
        current_time = datetime.utcnow()
        active_goals = []

        for goal in goals:
            # Remove goals older than timeout
            if (current_time - goal.created_at).total_seconds() > (self.goal_timeout_hours * 3600):
                if goal.status not in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                    goal.status = TaskStatus.CANCELLED
                    logger.info(f"Cancelled stale goal: '{goal.description}'")
            else:
                active_goals.append(goal)

        return active_goals
