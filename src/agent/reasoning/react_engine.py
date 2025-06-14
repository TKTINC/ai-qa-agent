"""
ReAct (Reasoning + Acting) Engine
Implements the ReAct pattern for agent reasoning and action
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from ..core.models import (
    AgentState, ReasoningStep, ReasoningType, Message, 
    AgentResponse, Goal, TaskStatus
)
from ...core.exceptions import AgentError


logger = logging.getLogger(__name__)


class ReActReasoner:
    """
    Implements the ReAct (Reasoning + Acting) pattern for intelligent agent behavior.
    
    The ReAct pattern involves:
    1. Observation: Understanding the current situation
    2. Thought: Reasoning about what to do next
    3. Plan: Creating an approach to achieve the goal
    4. Action: Executing planned actions
    5. Reflection: Learning from the outcome
    """

    def __init__(self):
        self.reasoning_timeout = 30  # seconds
        self.max_reasoning_steps = 20
        self.confidence_threshold = 0.7

    async def reason_and_act(
        self,
        user_input: str,
        agent_state: AgentState,
        available_tools: List[str]
    ) -> AgentResponse:
        """
        Execute complete ReAct cycle for user input
        
        Args:
            user_input: The user's message or request
            agent_state: Current agent state and context
            available_tools: List of available tool names
            
        Returns:
            AgentResponse with reasoning steps and final response
        """
        start_time = datetime.utcnow()
        reasoning_steps = []

        try:
            # Step 1: Observe the situation
            observation_step = await self._observe_situation(
                user_input, agent_state, available_tools
            )
            reasoning_steps.append(observation_step)
            agent_state.add_reasoning_step(observation_step)

            # Step 2: Think about the request
            thought_step = await self._generate_thoughts(
                user_input, agent_state, observation_step
            )
            reasoning_steps.append(thought_step)
            agent_state.add_reasoning_step(thought_step)

            # Step 3: Plan the approach
            plan_step = await self._create_plan(
                user_input, agent_state, thought_step
            )
            reasoning_steps.append(plan_step)
            agent_state.add_reasoning_step(plan_step)

            # Step 4: Execute actions
            action_step = await self._execute_actions(
                user_input, agent_state, plan_step, available_tools
            )
            reasoning_steps.append(action_step)
            agent_state.add_reasoning_step(action_step)

            # Step 5: Reflect on results
            reflection_step = await self._reflect_on_outcome(
                user_input, agent_state, reasoning_steps
            )
            reasoning_steps.append(reflection_step)
            agent_state.add_reasoning_step(reflection_step)

            # Generate final response
            response_content = await self._generate_response(
                user_input, agent_state, reasoning_steps
            )

            # Calculate overall confidence
            confidence = self._calculate_confidence(reasoning_steps)

            # Generate suggestions and follow-ups
            suggestions = await self._generate_suggestions(agent_state, reasoning_steps)
            follow_ups = await self._generate_follow_ups(user_input, agent_state)

            duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            logger.info(f"ReAct reasoning completed in {duration_ms}ms with confidence {confidence}")

            return AgentResponse(
                content=response_content,
                reasoning_steps=reasoning_steps,
                tools_used=self._extract_tools_used(reasoning_steps),
                confidence=confidence,
                suggestions=suggestions,
                follow_up_questions=follow_ups,
                session_id=agent_state.session_id,
                metadata={
                    "reasoning_duration_ms": duration_ms,
                    "steps_count": len(reasoning_steps),
                    "reasoning_pattern": "ReAct"
                }
            )

        except Exception as e:
            logger.error(f"Error in ReAct reasoning: {str(e)}")
            raise AgentError(f"Reasoning failed: {str(e)}")

    async def _observe_situation(
        self,
        user_input: str,
        agent_state: AgentState,
        available_tools: List[str]
    ) -> ReasoningStep:
        """
        Observe and analyze the current situation
        """
        start_time = datetime.utcnow()

        # Analyze user input and context
        observations = []

        # Analyze user request
        if "test" in user_input.lower():
            observations.append("User is asking about testing-related topics")
        if "error" in user_input.lower() or "bug" in user_input.lower():
            observations.append("User may have encountered an issue")
        if "help" in user_input.lower():
            observations.append("User is requesting assistance")

        # Analyze conversation context
        recent_messages = agent_state.get_recent_context(5)
        if recent_messages:
            observations.append(f"Conversation context: {len(recent_messages)} recent messages")

        # Analyze current goal
        if agent_state.current_goal:
            observations.append(f"Active goal: {agent_state.current_goal.description}")

        # Analyze available tools
        observations.append(f"Available tools: {', '.join(available_tools)}")

        observation_content = "Situation Analysis:\n" + "\n".join(f"- {obs}" for obs in observations)

        duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)

        return ReasoningStep(
            type=ReasoningType.OBSERVATION,
            content=observation_content,
            confidence=0.9,
            context={"observations": observations, "input_analysis": user_input},
            duration_ms=duration_ms
        )

    async def _generate_thoughts(
        self,
        user_input: str,
        agent_state: AgentState,
        observation: ReasoningStep
    ) -> ReasoningStep:
        """
        Generate thoughts about how to approach the request
        """
        start_time = datetime.utcnow()

        thoughts = []

        # Analyze the complexity of the request
        if len(user_input.split()) > 20:
            thoughts.append("This is a complex request that may require multiple steps")
        else:
            thoughts.append("This appears to be a straightforward request")

        # Consider the best approach
        if "analyze" in user_input.lower():
            thoughts.append("User wants analysis - I should use code analysis tools")
        elif "generate" in user_input.lower() or "create" in user_input.lower():
            thoughts.append("User wants generation - I should plan a creation approach")
        elif "explain" in user_input.lower():
            thoughts.append("User wants explanation - I should provide educational content")

        # Consider user expertise level
        expertise_level = agent_state.user_preferences.get("expertise_level", "intermediate")
        thoughts.append(f"User expertise level: {expertise_level} - adjust response accordingly")

        # Consider tools needed
        if "code" in user_input.lower():
            thoughts.append("Will likely need code analysis tools")

        thought_content = "Reasoning Process:\n" + "\n".join(f"- {thought}" for thought in thoughts)

        duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)

        return ReasoningStep(
            type=ReasoningType.THOUGHT,
            content=thought_content,
            confidence=0.85,
            context={"thoughts": thoughts, "approach_analysis": True},
            duration_ms=duration_ms
        )

    async def _create_plan(
        self,
        user_input: str,
        agent_state: AgentState,
        thought_step: ReasoningStep
    ) -> ReasoningStep:
        """
        Create a plan for executing the request
        """
        start_time = datetime.utcnow()

        plan_steps = []

        # Determine primary action
        if "analyze" in user_input.lower():
            plan_steps.extend([
                "1. Identify what needs to be analyzed",
                "2. Select appropriate analysis tools",
                "3. Execute analysis with progress tracking",
                "4. Interpret results for user",
                "5. Provide actionable recommendations"
            ])
        elif "explain" in user_input.lower():
            plan_steps.extend([
                "1. Identify what needs explanation",
                "2. Gather relevant context and examples",
                "3. Structure explanation for user's expertise level",
                "4. Provide clear, actionable information"
            ])
        else:
            plan_steps.extend([
                "1. Clarify user's specific needs",
                "2. Determine best approach to help",
                "3. Execute solution with user feedback",
                "4. Validate outcome meets user expectations"
            ])

        plan_content = "Execution Plan:\n" + "\n".join(plan_steps)

        duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)

        return ReasoningStep(
            type=ReasoningType.PLAN,
            content=plan_content,
            confidence=0.8,
            context={"plan_steps": plan_steps, "execution_strategy": "step_by_step"},
            duration_ms=duration_ms
        )

    async def _execute_actions(
        self,
        user_input: str,
        agent_state: AgentState,
        plan_step: ReasoningStep,
        available_tools: List[str]
    ) -> ReasoningStep:
        """
        Execute the planned actions
        """
        start_time = datetime.utcnow()

        actions_taken = []
        tools_used = []

        # Simulate action execution based on plan
        plan_context = plan_step.context.get("plan_steps", [])

        if "analyze" in user_input.lower() and "code_analyzer" in available_tools:
            actions_taken.append("Prepared code analysis tool for execution")
            tools_used.append("code_analyzer")

        if "explain" in user_input.lower():
            actions_taken.append("Gathered explanatory context and examples")

        # Always prepare response
        actions_taken.append("Formulated comprehensive response based on analysis")

        action_content = "Actions Executed:\n" + "\n".join(f"- {action}" for action in actions_taken)

        duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)

        return ReasoningStep(
            type=ReasoningType.ACTION,
            content=action_content,
            confidence=0.85,
            context={"actions": actions_taken},
            tools_used=tools_used,
            duration_ms=duration_ms
        )

    async def _reflect_on_outcome(
        self,
        user_input: str,
        agent_state: AgentState,
        reasoning_steps: List[ReasoningStep]
    ) -> ReasoningStep:
        """
        Reflect on the reasoning process and outcome
        """
        start_time = datetime.utcnow()

        reflections = []

        # Evaluate reasoning quality
        total_confidence = sum(step.confidence for step in reasoning_steps)
        avg_confidence = total_confidence / len(reasoning_steps) if reasoning_steps else 0
        reflections.append(f"Average reasoning confidence: {avg_confidence:.2f}")

        # Evaluate completeness
        if len(reasoning_steps) >= 4:
            reflections.append("Completed full ReAct cycle (Observe → Think → Plan → Act)")
        else:
            reflections.append("Abbreviated reasoning cycle due to simple request")

        # Evaluate tool usage
        tools_used = self._extract_tools_used(reasoning_steps)
        if tools_used:
            reflections.append(f"Successfully utilized tools: {', '.join(tools_used)}")
        else:
            reflections.append("No tools required for this request")

        # Consider improvements
        if avg_confidence < 0.7:
            reflections.append("Lower confidence - may need clarification or additional context")
        else:
            reflections.append("High confidence in reasoning and planned response")

        reflection_content = "Reasoning Reflection:\n" + "\n".join(f"- {ref}" for ref in reflections)

        duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)

        return ReasoningStep(
            type=ReasoningType.REFLECTION,
            content=reflection_content,
            confidence=avg_confidence,
            context={"reflections": reflections, "quality_assessment": avg_confidence},
            duration_ms=duration_ms
        )

    async def _generate_response(
        self,
        user_input: str,
        agent_state: AgentState,
        reasoning_steps: List[ReasoningStep]
    ) -> str:
        """
        Generate the final response based on reasoning
        """
        # Extract key insights from reasoning
        observations = next((step for step in reasoning_steps if step.type == ReasoningType.OBSERVATION), None)
        thoughts = next((step for step in reasoning_steps if step.type == ReasoningType.THOUGHT), None)
        plan = next((step for step in reasoning_steps if step.type == ReasoningType.PLAN), None)
        actions = next((step for step in reasoning_steps if step.type == ReasoningType.ACTION), None)

        # Build response based on request type
        if "explain" in user_input.lower():
            response = self._generate_explanation_response(user_input, agent_state)
        elif "analyze" in user_input.lower():
            response = self._generate_analysis_response(user_input, agent_state)
        elif "help" in user_input.lower():
            response = self._generate_help_response(user_input, agent_state)
        else:
            response = self._generate_general_response(user_input, agent_state)

        return response

    def _generate_explanation_response(self, user_input: str, agent_state: AgentState) -> str:
        """Generate educational/explanatory response"""
        return f"""I'd be happy to help explain that! Based on your question, I can see you're looking for a clear understanding of the topic.

Let me break this down in a way that's helpful for your expertise level. I'll provide both the conceptual understanding and practical applications so you can apply this knowledge effectively.

Would you like me to start with the fundamentals, or would you prefer to focus on specific aspects that are most relevant to your current work?"""

    def _generate_analysis_response(self, user_input: str, agent_state: AgentState) -> str:
        """Generate analysis-focused response"""
        return f"""I'm ready to help you analyze your code! I can perform comprehensive analysis including:

• **Code Structure Analysis**: Functions, classes, complexity metrics
• **Quality Assessment**: Testability scoring, maintainability insights  
• **Pattern Detection**: Design patterns, architectural insights
• **Test Recommendations**: Priority areas for test coverage

To get started, you can either:
1. Share specific code you'd like me to analyze
2. Point me to a repository or file path
3. Describe what specific aspects you'd like me to focus on

What would be most helpful for your current needs?"""

    def _generate_help_response(self, user_input: str, agent_state: AgentState) -> str:
        """Generate helpful assistance response"""
        return f"""I'm here to help! I'm an AI QA agent that specializes in code analysis, test generation, and software quality improvement.

Here's what I can assist you with:

• **Code Analysis**: Understand complexity, quality, and structure
• **Test Strategy**: Design comprehensive testing approaches
• **Quality Improvement**: Identify areas for enhancement
• **Best Practices**: Share testing and development recommendations

What specific challenge are you working on? I can provide targeted assistance based on your needs."""

    def _generate_general_response(self, user_input: str, agent_state: AgentState) -> str:
        """Generate general conversational response"""
        return f"""I understand you'd like assistance with your software quality and testing needs. I'm equipped to help with various aspects of code analysis and test generation.

Based on your message, I can provide guidance on best practices, analyze specific code patterns, or help you develop a testing strategy that fits your project requirements.

What specific area would you like to explore together?"""

    async def _generate_suggestions(
        self,
        agent_state: AgentState,
        reasoning_steps: List[ReasoningStep]
    ) -> List[str]:
        """Generate helpful suggestions based on reasoning"""
        suggestions = []

        # Suggest based on conversation context
        if not agent_state.current_goal:
            suggestions.append("Consider setting a specific goal for our conversation")

        # Suggest based on reasoning quality
        avg_confidence = sum(step.confidence for step in reasoning_steps) / len(reasoning_steps)
        if avg_confidence < 0.7:
            suggestions.append("Feel free to provide more context for better assistance")

        # Suggest next steps
        suggestions.append("Ask me to analyze specific code or explain testing concepts")
        suggestions.append("Share your current challenges for targeted recommendations")

        return suggestions

    async def _generate_follow_ups(self, user_input: str, agent_state: AgentState) -> List[str]:
        """Generate relevant follow-up questions"""
        follow_ups = []

        if "analyze" in user_input.lower():
            follow_ups.extend([
                "What specific aspects of the code are you most concerned about?",
                "Are there particular quality metrics you'd like me to focus on?",
                "Would you like me to suggest specific testing strategies?"
            ])
        elif "test" in user_input.lower():
            follow_ups.extend([
                "What testing framework are you currently using?",
                "Are you looking for unit tests, integration tests, or both?",
                "What's your current test coverage goal?"
            ])
        else:
            follow_ups.extend([
                "What's your main goal for improving code quality?",
                "Are you working on a specific project or codebase?",
                "What's your experience level with testing and quality practices?"
            ])

        return follow_ups[:3]  # Limit to 3 follow-ups

    def _extract_tools_used(self, reasoning_steps: List[ReasoningStep]) -> List[str]:
        """Extract list of tools used during reasoning"""
        tools = set()
        for step in reasoning_steps:
            tools.update(step.tools_used)
        return list(tools)

    def _calculate_confidence(self, reasoning_steps: List[ReasoningStep]) -> float:
        """Calculate overall confidence based on reasoning steps"""
        if not reasoning_steps:
            return 0.5

        total_confidence = sum(step.confidence for step in reasoning_steps)
        avg_confidence = total_confidence / len(reasoning_steps)

        # Boost confidence if we completed full ReAct cycle
        if len(reasoning_steps) >= 5:
            avg_confidence = min(1.0, avg_confidence * 1.1)

        return round(avg_confidence, 3)
