"""
Code Reviewer Agent
Specializes in code quality and review processes
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from .base_specialist import SpecialistAgent
from ..communication.models import ConsultationRequest, ConsultationResponse, AgentCapability
from ..tools.base_tool import ToolParameters
from ...core.exceptions import AgentError


logger = logging.getLogger(__name__)


class CodeReviewAgent(SpecialistAgent):
    """
    Specialist agent focused on code quality and review processes.
    
    Expertise areas:
    - Code quality assessment
    - Code review best practices
    - Refactoring recommendations
    - Code maintainability analysis
    - Technical debt identification
    """

    def __init__(self):
        super().__init__(
            name="code_reviewer",
            specialization="Code Quality & Review",
            expertise_domains=[
                "code_quality", "code_review", "refactoring", 
                "maintainability", "technical_debt", "code_standards"
            ]
        )

    async def _initialize_capabilities(self) -> None:
        """Initialize Code Reviewer specific capabilities"""
        await super()._initialize_capabilities()
        
        specialist_capabilities = [
            AgentCapability(
                name="code_quality_assessment",
                description="Assess code quality and identify improvement areas",
                confidence_level=0.94
            ),
            AgentCapability(
                name="refactoring_recommendations",
                description="Provide specific refactoring recommendations",
                confidence_level=0.91
            ),
            AgentCapability(
                name="technical_debt_analysis",
                description="Identify and prioritize technical debt",
                confidence_level=0.89
            ),
            AgentCapability(
                name="code_standards_enforcement",
                description="Ensure adherence to coding standards",
                confidence_level=0.93
            ),
            AgentCapability(
                name="maintainability_improvement",
                description="Improve code maintainability and readability",
                confidence_level=0.90
            )
        ]
        self.capabilities.extend(specialist_capabilities)

    async def _setup_specialist_tools(self) -> None:
        """Set up Code Reviewer specific tools"""
        from ..tools.analysis.code_analysis_tool import CodeAnalysisTool
        analysis_tool = CodeAnalysisTool()
        await self.tool_manager.register_tool(analysis_tool)

    async def analyze_problem(self, problem: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze code quality problems and provide improvement recommendations
        
        Args:
            problem: Problem description
            context: Analysis context including code and quality concerns
            
        Returns:
            Analysis with code quality recommendations
        """
        start_time = datetime.utcnow()
        
        try:
            analysis_result = {
                "analysis_type": "code_quality_review",
                "quality_assessment": await self._assess_code_quality(problem, context),
                "issue_identification": await self._identify_quality_issues(problem, context),
                "refactoring_opportunities": await self._identify_refactoring_opportunities(context),
                "technical_debt_analysis": await self._analyze_technical_debt(context),
                "improvement_priorities": await self._prioritize_improvements(context),
                "action_plan": await self._create_improvement_action_plan(context)
            }
            
            await self.update_capability("code_quality_assessment", True,
                                       (datetime.utcnow() - start_time).total_seconds())
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Code Reviewer analysis failed: {str(e)}")
            await self.update_capability("code_quality_assessment", False,
                                       (datetime.utcnow() - start_time).total_seconds())
            raise AgentError(f"Code quality analysis failed: {str(e)}")

    async def _assess_code_quality(self, problem: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall code quality"""
        # Use code analysis tool if available
        if "code_content" in context or "file_path" in context:
            # Execute code analysis
            tool_params = ToolParameters(
                parameters={
                    "analysis_type": "quality",
                    "code_content": context.get("code_content"),
                    "file_path": context.get("file_path")
                },
                session_id="code_review_session"
            )
            
            tools = await self.tool_manager.select_tools("analyze code quality", None)
            if tools:
                result = await self.tool_manager.execute_tool(tools[0], tool_params)
                if result.success:
                    quality_data = result.data.get("quality_metrics", {})
                    return {
                        "overall_grade": quality_data.get("quality_grade", "C"),
                        "complexity_score": quality_data.get("avg_complexity", 5),
                        "maintainability": quality_data.get("avg_testability_score", 0.5),
                        "technical_debt_indicators": self._extract_debt_indicators(quality_data),
                        "improvement_potential": quality_data.get("improvement_potential", True)
                    }
        
        # Fallback to text-based analysis
        return self._assess_quality_from_description(problem, context)

    def _assess_quality_from_description(self, problem: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Assess quality based on problem description"""
        quality_indicators = {
            "complexity_concerns": any(word in problem.lower() for word in 
                                     ["complex", "complicated", "hard to understand"]),
            "maintainability_concerns": any(word in problem.lower() for word in
                                          ["maintain", "modify", "change", "update"]),
            "readability_concerns": any(word in problem.lower() for word in
                                      ["readable", "understand", "confusing"]),
            "performance_concerns": any(word in problem.lower() for word in
                                      ["slow", "performance", "optimize"])
        }
        
        return {
            "overall_grade": "D" if sum(quality_indicators.values()) > 2 else "C",
            "primary_concerns": [k for k, v in quality_indicators.items() if v],
            "assessment_confidence": 0.7,
            "needs_detailed_analysis": True
        }

    async def _identify_quality_issues(self, problem: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify specific code quality issues"""
        issues = []
        
        # Extract issues from problem description
        problem_lower = problem.lower()
        
        if "complex" in problem_lower or "complicated" in problem_lower:
            issues.append({
                "type": "complexity",
                "severity": "medium",
                "description": "High complexity identified",
                "impact": "Difficult to understand and maintain",
                "recommendation": "Break down into smaller, focused functions"
            })
            
        if "duplicate" in problem_lower or "repeated" in problem_lower:
            issues.append({
                "type": "duplication",
                "severity": "medium",
                "description": "Code duplication detected",
                "impact": "Increases maintenance burden",
                "recommendation": "Extract common functionality into reusable components"
            })
            
        if "long" in problem_lower or "large" in problem_lower:
            issues.append({
                "type": "size",
                "severity": "low",
                "description": "Large code units detected",
                "impact": "Reduced readability and testability",
                "recommendation": "Split into smaller, focused units"
            })
        
        return issues

    async def _identify_refactoring_opportunities(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify refactoring opportunities"""
        opportunities = [
            {
                "type": "extract_method",
                "description": "Extract repeated code patterns into methods",
                "benefit": "Reduced duplication and improved maintainability",
                "effort": "low",
                "priority": "medium"
            },
            {
                "type": "simplify_conditionals",
                "description": "Simplify complex conditional statements",
                "benefit": "Improved readability and reduced cognitive load",
                "effort": "low",
                "priority": "high"
            },
            {
                "type": "improve_naming",
                "description": "Use more descriptive variable and function names",
                "benefit": "Enhanced code self-documentation",
                "effort": "low",
                "priority": "medium"
            }
        ]
        
        return opportunities

    async def provide_consultation(self, request: ConsultationRequest) -> ConsultationResponse:
        """
        Provide expert consultation on code quality
        
        Args:
            request: Consultation request
            
        Returns:
            Expert response with code quality recommendations
        """
        start_time = datetime.utcnow()
        
        try:
            question = request.question.lower()
            
            if "quality" in question:
                response = await self._consult_on_quality(request)
            elif "refactor" in question:
                response = await self._consult_on_refactoring(request)
            elif "review" in question:
                response = await self._consult_on_review_process(request)
            elif "debt" in question:
                response = await self._consult_on_technical_debt(request)
            else:
                response = await self._provide_general_code_consultation(request)
            
            consultation_response = ConsultationResponse(
                consultation_id=request.id,
                responding_agent=self.name,
                response=response["answer"],
                confidence=response["confidence"],
                recommendations=response["recommendations"],
                follow_up_questions=response["follow_ups"]
            )
            
            await self.update_capability("consultation", True,
                                       (datetime.utcnow() - start_time).total_seconds())
            
            return consultation_response
            
        except Exception as e:
            logger.error(f"Code Reviewer consultation failed: {str(e)}")
            await self.update_capability("consultation", False,
                                       (datetime.utcnow() - start_time).total_seconds())
            
            return ConsultationResponse(
                consultation_id=request.id,
                responding_agent=self.name,
                response=f"I encountered an issue providing consultation: {str(e)}",
                confidence=0.0
            )

    async def _consult_on_quality(self, request: ConsultationRequest) -> Dict[str, Any]:
        """Provide consultation on code quality"""
        return {
            "answer": "Code quality can be improved through systematic review of complexity, maintainability, and readability. I recommend starting with automated quality metrics, then focusing on high-impact areas like reducing complexity and improving naming conventions.",
            "confidence": 0.93,
            "recommendations": [
                "Implement automated code quality checks",
                "Focus on reducing cyclomatic complexity",
                "Improve variable and function naming",
                "Add comprehensive documentation",
                "Establish coding standards and guidelines"
            ],
            "follow_ups": [
                "What specific quality issues are you experiencing?",
                "Do you have existing quality metrics or tools?",
                "What is your team's experience with code reviews?"
            ]
        }

    async def collaborate_on_task(self, task: str, collaboration_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Collaborate with other agents on code quality tasks
        
        Args:
            task: Collaboration task description
            collaboration_context: Context including other agents and shared data
            
        Returns:
            Code Reviewer's contribution to the collaborative effort
        """
        contribution = {
            "agent": self.name,
            "specialization_applied": "code_quality_review",
            "quality_analysis": await self._analyze_quality_for_collaboration(task, collaboration_context),
            "review_recommendations": await self._generate_review_recommendations(task, collaboration_context),
            "integration_suggestions": await self._suggest_quality_integration(collaboration_context)
        }
        
        return contribution

    async def _analyze_quality_for_collaboration(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze task from code quality perspective"""
        return {
            "quality_implications": self._identify_quality_implications(task),
            "review_checkpoints": self._define_review_checkpoints(task),
            "quality_gates": self._recommend_quality_gates(task),
            "collaboration_standards": self._suggest_collaboration_standards(context)
        }

    def _identify_quality_implications(self, task: str) -> List[str]:
        """Identify quality implications of a task"""
        implications = []
        task_lower = task.lower()
        
        if "implement" in task_lower or "develop" in task_lower:
            implications.extend([
                "Code standards compliance required",
                "Peer review process needed",
                "Quality metrics monitoring essential"
            ])
            
        if "refactor" in task_lower or "improve" in task_lower:
            implications.extend([
                "Before/after quality comparison needed",
                "Regression testing required",
                "Documentation updates necessary"
            ])
            
        return implications
