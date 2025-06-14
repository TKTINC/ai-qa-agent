"""
Performance Analyst Agent
Specializes in performance testing and optimization
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from .base_specialist import SpecialistAgent
from ..communication.models import ConsultationRequest, ConsultationResponse, AgentCapability
from ...core.exceptions import AgentError


logger = logging.getLogger(__name__)


class PerformanceAnalyst(SpecialistAgent):
    """
    Specialist agent focused on performance analysis and optimization.
    
    Expertise areas:
    - Performance testing strategies
    - Performance bottleneck identification
    - Load testing and capacity planning
    - Performance optimization recommendations
    - Monitoring and observability
    """

    def __init__(self):
        super().__init__(
            name="performance_analyst",
            specialization="Performance Analysis & Optimization",
            expertise_domains=[
                "performance_testing", "load_testing", "capacity_planning",
                "performance_optimization", "monitoring", "scalability"
            ]
        )

    async def _initialize_capabilities(self) -> None:
        """Initialize Performance Analyst specific capabilities"""
        await super()._initialize_capabilities()
        
        specialist_capabilities = [
            AgentCapability(
                name="performance_bottleneck_analysis",
                description="Identify and analyze performance bottlenecks",
                confidence_level=0.93
            ),
            AgentCapability(
                name="load_testing_strategy",
                description="Design comprehensive load testing strategies",
                confidence_level=0.91
            ),
            AgentCapability(
                name="capacity_planning",
                description="Plan system capacity and scaling requirements",
                confidence_level=0.88
            ),
            AgentCapability(
                name="performance_optimization",
                description="Recommend specific performance optimizations",
                confidence_level=0.90
            ),
            AgentCapability(
                name="monitoring_strategy",
                description="Design performance monitoring and alerting",
                confidence_level=0.87
            )
        ]
        self.capabilities.extend(specialist_capabilities)

    async def analyze_problem(self, problem: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze performance problems and provide optimization recommendations
        
        Args:
            problem: Problem description
            context: Analysis context including performance data and requirements
            
        Returns:
            Analysis with performance optimization recommendations
        """
        start_time = datetime.utcnow()
        
        try:
            analysis_result = {
                "analysis_type": "performance_analysis",
                "performance_assessment": await self._assess_performance_problem(problem, context),
                "bottleneck_analysis": await self._identify_bottlenecks(problem, context),
                "optimization_recommendations": await self._recommend_optimizations(problem, context),
                "testing_strategy": await self._design_performance_testing(problem, context),
                "monitoring_recommendations": await self._recommend_monitoring(context),
                "capacity_planning": await self._analyze_capacity_needs(context)
            }
            
            await self.update_capability("performance_bottleneck_analysis", True,
                                       (datetime.utcnow() - start_time).total_seconds())
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Performance Analyst analysis failed: {str(e)}")
            await self.update_capability("performance_bottleneck_analysis", False,
                                       (datetime.utcnow() - start_time).total_seconds())
            raise AgentError(f"Performance analysis failed: {str(e)}")

    async def _assess_performance_problem(self, problem: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Assess the performance problem"""
        problem_lower = problem.lower()
        
        performance_indicators = {
            "slow_response": any(word in problem_lower for word in ["slow", "delay", "timeout"]),
            "high_load": any(word in problem_lower for word in ["load", "traffic", "users"]),
            "memory_issues": any(word in problem_lower for word in ["memory", "ram", "leak"]),
            "cpu_issues": any(word in problem_lower for word in ["cpu", "processing", "compute"]),
            "database_issues": any(word in problem_lower for word in ["database", "query", "db"])
        }
        
        severity = "high" if sum(performance_indicators.values()) > 2 else "medium" if sum(performance_indicators.values()) > 0 else "low"
        
        return {
            "problem_type": self._classify_performance_problem(problem),
            "severity": severity,
            "indicators": performance_indicators,
            "impact_assessment": self._assess_performance_impact(problem, context),
            "urgency": self._determine_urgency(problem, context)
        }

    def _classify_performance_problem(self, problem: str) -> str:
        """Classify the type of performance problem"""
        problem_lower = problem.lower()
        
        if any(word in problem_lower for word in ["response", "latency", "slow"]):
            return "response_time"
        elif any(word in problem_lower for word in ["load", "traffic", "concurrent"]):
            return "throughput"
        elif any(word in problem_lower for word in ["memory", "ram", "leak"]):
            return "memory_usage"
        elif any(word in problem_lower for word in ["cpu", "processing"]):
            return "cpu_utilization"
        elif any(word in problem_lower for word in ["database", "query"]):
            return "database_performance"
        else:
            return "general_performance"

    async def provide_consultation(self, request: ConsultationRequest) -> ConsultationResponse:
        """
        Provide expert consultation on performance
        
        Args:
            request: Consultation request
            
        Returns:
            Expert response with performance recommendations
        """
        start_time = datetime.utcnow()
        
        try:
            question = request.question.lower()
            
            if "bottleneck" in question or "slow" in question:
                response = await self._consult_on_bottlenecks(request)
            elif "load" in question or "testing" in question:
                response = await self._consult_on_load_testing(request)
            elif "optimize" in question or "improve" in question:
                response = await self._consult_on_optimization(request)
            elif "monitor" in question:
                response = await self._consult_on_monitoring(request)
            else:
                response = await self._provide_general_performance_consultation(request)
            
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
            logger.error(f"Performance Analyst consultation failed: {str(e)}")
            await self.update_capability("consultation", False,
                                       (datetime.utcnow() - start_time).total_seconds())
            
            return ConsultationResponse(
                consultation_id=request.id,
                responding_agent=self.name,
                response=f"I encountered an issue providing consultation: {str(e)}",
                confidence=0.0
            )

    async def _consult_on_bottlenecks(self, request: ConsultationRequest) -> Dict[str, Any]:
        """Provide consultation on performance bottlenecks"""
        return {
            "answer": "To identify performance bottlenecks, I recommend starting with application profiling to understand where time is spent, monitoring resource utilization (CPU, memory, I/O), and analyzing request patterns. Focus on the slowest operations first as they typically provide the highest optimization impact.",
            "confidence": 0.92,
            "recommendations": [
                "Implement application performance monitoring (APM)",
                "Profile code to identify slow functions and queries",
                "Monitor system resources during peak usage",
                "Analyze database query performance",
                "Check for memory leaks and inefficient algorithms"
            ],
            "follow_ups": [
                "What specific performance issues are you experiencing?",
                "Do you have performance monitoring tools in place?",
                "What are your current response time targets?"
            ]
        }

    async def collaborate_on_task(self, task: str, collaboration_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Collaborate with other agents on performance-related tasks
        
        Args:
            task: Collaboration task description
            collaboration_context: Context including other agents and shared data
            
        Returns:
            Performance Analyst's contribution to the collaborative effort
        """
        contribution = {
            "agent": self.name,
            "specialization_applied": "performance_analysis",
            "performance_considerations": await self._analyze_performance_implications(task, collaboration_context),
            "testing_recommendations": await self._recommend_performance_testing(task, collaboration_context),
            "optimization_opportunities": await self._identify_optimization_opportunities(task, collaboration_context)
        }
        
        return contribution

    async def _analyze_performance_implications(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance implications of a task"""
        return {
            "performance_impact": self._assess_task_performance_impact(task),
            "scalability_concerns": self._identify_scalability_concerns(task),
            "resource_requirements": self._estimate_resource_requirements(task, context),
            "monitoring_needs": self._identify_monitoring_needs(task)
        }

    def _assess_task_performance_impact(self, task: str) -> List[str]:
        """Assess potential performance impact of a task"""
        impacts = []
        task_lower = task.lower()
        
        if "database" in task_lower or "query" in task_lower:
            impacts.extend([
                "Database query performance impact",
                "Connection pool utilization",
                "Index optimization opportunities"
            ])
            
        if "api" in task_lower or "service" in task_lower:
            impacts.extend([
                "API response time impact",
                "Concurrent request handling",
                "Service dependency performance"
            ])
            
        if "cache" in task_lower:
            impacts.extend([
                "Cache hit ratio optimization",
                "Memory usage considerations",
                "Cache invalidation strategy"
            ])
            
        return impacts
