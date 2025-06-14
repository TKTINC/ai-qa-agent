"""
Task Planning System
Creates and manages execution plans for user goals
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from ..core.models import Goal, TaskPlan, TaskStatus, AgentState
from ...core.exceptions import AgentError


logger = logging.getLogger(__name__)


class TaskPlanner:
    """
    Intelligent task planning system that breaks down complex goals
    into actionable steps and manages execution strategies.
    """

    def __init__(self):
        self.max_plan_steps = 10
        self.default_step_duration = 5  # minutes
        self.planning_timeout = 15  # seconds

    async def create_plan(
        self,
        goal: Goal,
        agent_state: AgentState,
        available_tools: List[str]
    ) -> TaskPlan:
        """
        Create a detailed execution plan for a goal
        
        Args:
            goal: The goal to plan for
            agent_state: Current agent state and context
            available_tools: Available tools for execution
            
        Returns:
            TaskPlan with detailed steps and requirements
        """
        start_time = datetime.utcnow()

        try:
            # Analyze goal complexity and requirements
            complexity_analysis = await self._analyze_goal_complexity(goal)
            
            # Break down goal into actionable steps
            plan_steps = await self._decompose_goal(goal, complexity_analysis, available_tools)
            
            # Estimate duration and required tools
            estimated_duration = self._estimate_duration(plan_steps)
            required_tools = self._identify_required_tools(plan_steps, available_tools)
            
            # Define success criteria
            success_criteria = await self._define_success_criteria(goal, plan_steps)
            
            # Create the plan
            plan = TaskPlan(
                goal=goal,
                steps=plan_steps,
                estimated_duration=estimated_duration,
                required_tools=required_tools,
                success_criteria=success_criteria
            )

            duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            logger.info(f"Created plan for goal '{goal.description}' in {duration_ms}ms")

            return plan

        except Exception as e:
            logger.error(f"Error creating plan for goal '{goal.description}': {str(e)}")
            raise AgentError(f"Planning failed: {str(e)}")

    async def _analyze_goal_complexity(self, goal: Goal) -> Dict[str, Any]:
        """
        Analyze the complexity and requirements of a goal
        """
        complexity_analysis = {
            "complexity_score": 1,  # 1-5 scale
            "requires_analysis": False,
            "requires_generation": False,
            "requires_explanation": False,
            "requires_multiple_steps": False,
            "domain_specific": False
        }

        description = goal.description.lower()

        # Analyze keywords to understand requirements
        if any(word in description for word in ["analyze", "review", "examine", "inspect"]):
            complexity_analysis["requires_analysis"] = True
            complexity_analysis["complexity_score"] += 1

        if any(word in description for word in ["generate", "create", "build", "write"]):
            complexity_analysis["requires_generation"] = True
            complexity_analysis["complexity_score"] += 1

        if any(word in description for word in ["explain", "teach", "show", "guide"]):
            complexity_analysis["requires_explanation"] = True

        if any(word in description for word in ["comprehensive", "complete", "thorough", "detailed"]):
            complexity_analysis["requires_multiple_steps"] = True
            complexity_analysis["complexity_score"] += 1

        if any(word in description for word in ["security", "performance", "architecture", "design patterns"]):
            complexity_analysis["domain_specific"] = True
            complexity_analysis["complexity_score"] += 1

        # Cap complexity score
        complexity_analysis["complexity_score"] = min(5, complexity_analysis["complexity_score"])

        return complexity_analysis

    async def _decompose_goal(
        self,
        goal: Goal,
        complexity_analysis: Dict[str, Any],
        available_tools: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Break down a goal into actionable steps
        """
        steps = []
        description = goal.description.lower()

        # Step 1: Always start with understanding/clarification
        steps.append({
            "step_number": 1,
            "title": "Understand Requirements",
            "description": "Clarify and validate the specific requirements",
            "action_type": "clarification",
            "estimated_minutes": 2,
            "tools_needed": [],
            "success_criteria": ["Requirements clearly understood", "Scope defined"]
        })

        # Step 2: Analysis steps if needed
        if complexity_analysis["requires_analysis"]:
            if "code" in description and "code_analyzer" in available_tools:
                steps.append({
                    "step_number": len(steps) + 1,
                    "title": "Code Analysis",
                    "description": "Perform comprehensive code analysis",
                    "action_type": "analysis",
                    "estimated_minutes": 5,
                    "tools_needed": ["code_analyzer"],
                    "success_criteria": ["Code structure analyzed", "Quality metrics calculated"]
                })

            steps.append({
                "step_number": len(steps) + 1,
                "title": "Analysis Interpretation",
                "description": "Interpret analysis results and identify key insights",
                "action_type": "interpretation",
                "estimated_minutes": 3,
                "tools_needed": [],
                "success_criteria": ["Results interpreted", "Key insights identified"]
            })

        # Step 3: Generation steps if needed
        if complexity_analysis["requires_generation"]:
            steps.append({
                "step_number": len(steps) + 1,
                "title": "Solution Generation",
                "description": "Generate solution based on analysis and requirements",
                "action_type": "generation",
                "estimated_minutes": 7,
                "tools_needed": ["test_generator"] if "test_generator" in available_tools else [],
                "success_criteria": ["Solution generated", "Quality validated"]
            })

        # Step 4: Explanation steps if needed
        if complexity_analysis["requires_explanation"]:
            steps.append({
                "step_number": len(steps) + 1,
                "title": "Explanation and Education",
                "description": "Provide clear explanation with educational value",
                "action_type": "explanation",
                "estimated_minutes": 4,
                "tools_needed": [],
                "success_criteria": ["Concepts explained clearly", "Educational value provided"]
            })

        # Step 5: Always end with validation and recommendations
        steps.append({
            "step_number": len(steps) + 1,
            "title": "Validation and Recommendations",
            "description": "Validate solution and provide actionable recommendations",
            "action_type": "validation",
            "estimated_minutes": 3,
            "tools_needed": [],
            "success_criteria": ["Solution validated", "Next steps recommended"]
        })

        return steps

    def _estimate_duration(self, plan_steps: List[Dict[str, Any]]) -> int:
        """
        Estimate total duration for plan execution
        """
        total_minutes = sum(step.get("estimated_minutes", self.default_step_duration) for step in plan_steps)
        
        # Add buffer for complex plans
        if len(plan_steps) > 5:
            total_minutes = int(total_minutes * 1.2)

        return total_minutes

    def _identify_required_tools(
        self,
        plan_steps: List[Dict[str, Any]],
        available_tools: List[str]
    ) -> List[str]:
        """
        Identify all tools required for plan execution
        """
        required_tools = set()
        
        for step in plan_steps:
            tools_needed = step.get("tools_needed", [])
            for tool in tools_needed:
                if tool in available_tools:
                    required_tools.add(tool)
        
        return list(required_tools)

    async def _define_success_criteria(
        self,
        goal: Goal,
        plan_steps: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Define success criteria for the overall plan
        """
        criteria = []

        # Extract success criteria from individual steps
        for step in plan_steps:
            step_criteria = step.get("success_criteria", [])
            criteria.extend(step_criteria)

        # Add overall goal-specific criteria
        description = goal.description.lower()
        
        if "analyze" in description:
            criteria.append("Comprehensive analysis completed")
        
        if "improve" in description or "optimize" in description:
            criteria.append("Improvement recommendations provided")
        
        if "test" in description:
            criteria.append("Testing strategy established")

        # Always include user satisfaction
        criteria.append("User requirements met")
        criteria.append("Clear next steps provided")

        # Remove duplicates while preserving order
        unique_criteria = []
        seen = set()
        for criterion in criteria:
            if criterion not in seen:
                unique_criteria.append(criterion)
                seen.add(criterion)

        return unique_criteria

    async def adapt_plan(
        self,
        plan: TaskPlan,
        new_information: Dict[str, Any],
        agent_state: AgentState
    ) -> TaskPlan:
        """
        Adapt an existing plan based on new information or changing requirements
        """
        logger.info(f"Adapting plan for task {plan.task_id}")

        # Create adapted plan with updated steps
        adapted_steps = []
        
        for step in plan.steps:
            # Check if step needs modification
            if self._step_needs_adaptation(step, new_information):
                adapted_step = await self._adapt_step(step, new_information)
                adapted_steps.append(adapted_step)
            else:
                adapted_steps.append(step)

        # Update plan with adapted steps
        plan.steps = adapted_steps
        plan.estimated_duration = self._estimate_duration(adapted_steps)
        
        logger.info(f"Plan adapted with {len(adapted_steps)} steps")
        return plan

    def _step_needs_adaptation(self, step: Dict[str, Any], new_information: Dict[str, Any]) -> bool:
        """
        Determine if a step needs adaptation based on new information
        """
        # Check if new requirements affect this step
        if new_information.get("scope_change") and step["action_type"] in ["analysis", "generation"]:
            return True
        
        # Check if tool availability changed
        if new_information.get("tool_changes") and step.get("tools_needed"):
            return True
        
        return False

    async def _adapt_step(self, step: Dict[str, Any], new_information: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adapt a specific step based on new information
        """
        adapted_step = step.copy()

        # Adjust based on scope changes
        if new_information.get("scope_change"):
            if new_information["scope_change"] == "expanded":
                adapted_step["estimated_minutes"] = int(adapted_step["estimated_minutes"] * 1.5)
                adapted_step["description"] += " (expanded scope)"
            elif new_information["scope_change"] == "reduced":
                adapted_step["estimated_minutes"] = int(adapted_step["estimated_minutes"] * 0.7)
                adapted_step["description"] += " (focused scope)"

        # Adjust based on tool changes
        if new_information.get("tool_changes"):
            available_tools = new_information["tool_changes"].get("available", [])
            adapted_step["tools_needed"] = [
                tool for tool in adapted_step.get("tools_needed", [])
                if tool in available_tools
            ]

        return adapted_step

    async def validate_plan_feasibility(
        self,
        plan: TaskPlan,
        available_tools: List[str],
        time_constraints: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Validate if a plan is feasible given current constraints
        """
        validation_result = {
            "feasible": True,
            "issues": [],
            "recommendations": [],
            "confidence": 1.0
        }

        # Check tool availability
        missing_tools = []
        for tool in plan.required_tools:
            if tool not in available_tools:
                missing_tools.append(tool)

        if missing_tools:
            validation_result["feasible"] = False
            validation_result["issues"].append(f"Missing required tools: {', '.join(missing_tools)}")
            validation_result["recommendations"].append("Ensure all required tools are available")
            validation_result["confidence"] *= 0.7

        # Check time constraints
        if time_constraints and plan.estimated_duration > time_constraints:
            validation_result["issues"].append(f"Plan duration ({plan.estimated_duration}m) exceeds constraint ({time_constraints}m)")
            validation_result["recommendations"].append("Consider reducing scope or extending time limit")
            validation_result["confidence"] *= 0.8

        # Check plan complexity
        if len(plan.steps) > self.max_plan_steps:
            validation_result["issues"].append(f"Plan has too many steps ({len(plan.steps)} > {self.max_plan_steps})")
            validation_result["recommendations"].append("Break down into sub-goals or simplify approach")
            validation_result["confidence"] *= 0.6

        return validation_result
