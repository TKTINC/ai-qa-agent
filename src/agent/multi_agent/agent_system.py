"""
Multi-Agent System Orchestrator
Coordinates multiple specialist agents for collaborative problem-solving
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from ..orchestrator import QAAgentOrchestrator
from ..specialists.test_architect import TestArchitectAgent
from ..specialists.code_reviewer import CodeReviewAgent
from ..specialists.performance_analyst import PerformanceAnalyst
from ..communication.models import (
    AgentMessage, Collaboration, ConsultationRequest, 
    CollaborationType, MessageType, KnowledgeShare
)
from ..core.models import AgentState, AgentResponse, Goal
from ...core.exceptions import AgentError


logger = logging.getLogger(__name__)


class QAAgentSystem:
    """
    Complete multi-agent system for AI-powered QA assistance.
    
    Coordinates between a central orchestrator and specialist agents
    to provide comprehensive, collaborative problem-solving.
    """

    def __init__(self):
        # Core orchestrator
        self.coordinator = QAAgentOrchestrator()
        
        # Specialist agents
        self.specialists = {
            'test_architect': TestArchitectAgent(),
            'code_reviewer': CodeReviewAgent(),
            'performance_analyst': PerformanceAnalyst()
        }
        
        # Collaboration management
        self.active_collaborations: Dict[str, Collaboration] = {}
        self.collaboration_manager = None  # Will be initialized
        
        # System state
        self.system_initialized = False
        self.message_queue: List[AgentMessage] = []
        
    async def initialize_system(self) -> bool:
        """
        Initialize the complete multi-agent system
        
        Returns:
            True if initialization successful
        """
        try:
            # Initialize all specialist agents
            for name, agent in self.specialists.items():
                success = await agent.initialize_specialist()
                if not success:
                    logger.error(f"Failed to initialize {name}")
                    return False
                logger.info(f"Initialized specialist: {name}")
            
            # Initialize collaboration manager
            from ..collaboration.collaboration_manager import CollaborationManager
            self.collaboration_manager = CollaborationManager(self.specialists)
            
            self.system_initialized = True
            logger.info("Multi-agent system initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize multi-agent system: {str(e)}")
            return False

    async def handle_complex_request(
        self,
        user_request: str,
        session_id: str,
        user_profile: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        """
        Handle complex requests using multi-agent collaboration
        
        Args:
            user_request: User's request or question
            session_id: Conversation session identifier
            user_profile: User profile and preferences
            
        Returns:
            Comprehensive response from agent collaboration
        """
        start_time = datetime.utcnow()
        
        try:
            if not self.system_initialized:
                await self.initialize_system()
            
            # Analyze request complexity and determine approach
            approach = await self._determine_approach(user_request)
            
            if approach == "single_agent":
                # Use coordinator for simple requests
                return await self.coordinator.process_user_request(
                    user_request, session_id, user_profile
                )
            
            elif approach == "multi_agent_consultation":
                # Multi-agent consultation for complex requests
                return await self._handle_multi_agent_consultation(
                    user_request, session_id, user_profile
                )
            
            elif approach == "collaborative_analysis":
                # Full collaborative analysis for very complex requests
                return await self._handle_collaborative_analysis(
                    user_request, session_id, user_profile
                )
            
            else:
                # Default to coordinator
                return await self.coordinator.process_user_request(
                    user_request, session_id, user_profile
                )
                
        except Exception as e:
            logger.error(f"Error handling complex request: {str(e)}")
            raise AgentError(f"Multi-agent processing failed: {str(e)}")

    async def _determine_approach(self, user_request: str) -> str:
        """
        Determine the best approach for handling the request
        
        Args:
            user_request: User's request
            
        Returns:
            Approach strategy: single_agent, multi_agent_consultation, or collaborative_analysis
        """
        request_lower = user_request.lower()
        
        # Indicators for different approaches
        complexity_indicators = [
            "comprehensive", "complete", "thorough", "detailed",
            "analyze and", "review and", "both", "multiple"
        ]
        
        multi_domain_indicators = [
            "performance and quality", "testing and review", "architecture and optimization",
            "strategy and implementation", "design and test"
        ]
        
        collaboration_indicators = [
            "team", "collaborate", "work together", "consensus", "compare approaches"
        ]
        
        # Count indicators
        complexity_score = sum(1 for indicator in complexity_indicators if indicator in request_lower)
        multi_domain_score = sum(1 for indicator in multi_domain_indicators if indicator in request_lower)
        collaboration_score = sum(1 for indicator in collaboration_indicators if indicator in request_lower)
        
        # Determine approach
        if collaboration_score > 0 or (complexity_score >= 2 and multi_domain_score >= 1):
            return "collaborative_analysis"
        elif complexity_score >= 1 or multi_domain_score >= 1:
            return "multi_agent_consultation"
        else:
            return "single_agent"

    async def _handle_multi_agent_consultation(
        self,
        user_request: str,
        session_id: str,
        user_profile: Optional[Dict[str, Any]]
    ) -> AgentResponse:
        """
        Handle request through multi-agent consultation
        """
        # Identify relevant specialists
        relevant_specialists = await self._identify_relevant_specialists(user_request)
        
        consultations = []
        
        # Get consultations from each relevant specialist
        for specialist_name in relevant_specialists:
            specialist = self.specialists[specialist_name]
            
            consultation_request = ConsultationRequest(
                requesting_agent="coordinator",
                specialist_agent=specialist_name,
                question=user_request,
                context={"session_id": session_id, "user_profile": user_profile or {}}
            )
            
            try:
                consultation_response = await specialist.provide_consultation(consultation_request)
                consultations.append({
                    "specialist": specialist_name,
                    "response": consultation_response,
                    "specialization": specialist.specialization
                })
            except Exception as e:
                logger.warning(f"Consultation with {specialist_name} failed: {str(e)}")
        
        # Synthesize consultations into unified response
        return await self._synthesize_consultations(consultations, user_request, session_id)

    async def _handle_collaborative_analysis(
        self,
        user_request: str,
        session_id: str,
        user_profile: Optional[Dict[str, Any]]
    ) -> AgentResponse:
        """
        Handle request through full collaborative analysis
        """
        # Create collaboration session
        collaboration = Collaboration(
            collaboration_type=CollaborationType.CONSENSUS,
            participating_agents=list(self.specialists.keys()),
            coordinator_agent="coordinator",
            objective=user_request,
            context={"session_id": session_id, "user_profile": user_profile or {}}
        )
        
        self.active_collaborations[collaboration.id] = collaboration
        
        try:
            # Get contributions from all specialists
            contributions = []
            
            for specialist_name, specialist in self.specialists.items():
                try:
                    contribution = await specialist.collaborate_on_task(
                        user_request,
                        {
                            "collaboration_id": collaboration.id,
                            "participating_agents": collaboration.participating_agents,
                            "shared_workspace": collaboration.shared_workspace
                        }
                    )
                    contributions.append(contribution)
                    
                    # Add to shared workspace
                    collaboration.shared_workspace[specialist_name] = contribution
                    
                except Exception as e:
                    logger.warning(f"Collaboration with {specialist_name} failed: {str(e)}")
            
            # Synthesize collaborative contributions
            return await self._synthesize_collaboration(contributions, collaboration, user_request, session_id)
            
        finally:
            # Mark collaboration as completed
            collaboration.completed_at = datetime.utcnow()
            collaboration.status = "completed"

    async def _identify_relevant_specialists(self, user_request: str) -> List[str]:
        """
        Identify which specialists are relevant for the request
        """
        request_lower = user_request.lower()
        relevant_specialists = []
        
        # Test Architect indicators
        if any(word in request_lower for word in [
            "test", "testing", "strategy", "architecture", "coverage", "framework"
        ]):
            relevant_specialists.append("test_architect")
        
        # Code Reviewer indicators
        if any(word in request_lower for word in [
            "quality", "review", "refactor", "maintain", "clean", "debt", "standard"
        ]):
            relevant_specialists.append("code_reviewer")
        
        # Performance Analyst indicators
        if any(word in request_lower for word in [
            "performance", "speed", "optimize", "slow", "bottleneck", "load", "scale"
        ]):
            relevant_specialists.append("performance_analyst")
        
        # If no specific indicators, include all for comprehensive analysis
        if not relevant_specialists:
            relevant_specialists = list(self.specialists.keys())
        
        return relevant_specialists

    async def _synthesize_consultations(
        self,
        consultations: List[Dict[str, Any]],
        user_request: str,
        session_id: str
    ) -> AgentResponse:
        """
        Synthesize multiple specialist consultations into a unified response
        """
        if not consultations:
            return AgentResponse(
                content="I apologize, but I wasn't able to get specialist input on your request.",
                session_id=session_id,
                confidence=0.2
            )
        
        # Aggregate insights
        all_recommendations = []
        all_follow_ups = []
        total_confidence = 0
        
        response_parts = [
            f"I've consulted with {len(consultations)} specialist{'s' if len(consultations) > 1 else ''} to provide you with comprehensive guidance:\n"
        ]
        
        for consultation in consultations:
            specialist = consultation["specialist"]
            response_obj = consultation["response"]
            specialization = consultation["specialization"]
            
            response_parts.append(f"\n**{specialization} Perspective:**")
            response_parts.append(response_obj.response)
            
            all_recommendations.extend(response_obj.recommendations)
            all_follow_ups.extend(response_obj.follow_up_questions)
            total_confidence += response_obj.confidence
        
        # Add synthesized recommendations
        if all_recommendations:
            unique_recommendations = list(set(all_recommendations))[:5]  # Top 5 unique
            response_parts.append(f"\n**Key Recommendations:**")
            for i, rec in enumerate(unique_recommendations, 1):
                response_parts.append(f"{i}. {rec}")
        
        avg_confidence = total_confidence / len(consultations)
        
        return AgentResponse(
            content="\n".join(response_parts),
            session_id=session_id,
            confidence=avg_confidence,
            recommendations=all_recommendations[:10],  # Top 10
            follow_up_questions=all_follow_ups[:5],  # Top 5
            metadata={
                "consultation_type": "multi_specialist",
                "specialists_consulted": [c["specialist"] for c in consultations],
                "synthesis_confidence": avg_confidence
            }
        )

    async def _synthesize_collaboration(
        self,
        contributions: List[Dict[str, Any]],
        collaboration: Collaboration,
        user_request: str,
        session_id: str
    ) -> AgentResponse:
        """
        Synthesize collaborative contributions into a unified response
        """
        if not contributions:
            return AgentResponse(
                content="I apologize, but the collaborative analysis didn't produce results.",
                session_id=session_id,
                confidence=0.1
            )
        
        response_parts = [
            f"I've conducted a comprehensive collaborative analysis with {len(contributions)} specialist agents. Here's our unified assessment:\n"
        ]
        
        # Organize contributions by type
        analyses = []
        recommendations = []
        strategies = []
        
        for contribution in contributions:
            agent_name = contribution.get("agent", "unknown")
            specialization = contribution.get("specialization_applied", "general")
            
            response_parts.append(f"\n**{agent_name.replace('_', ' ').title()} Analysis:**")
            
            if "analysis" in contribution:
                analysis_content = contribution["analysis"]
                if isinstance(analysis_content, dict):
                    for key, value in analysis_content.items():
                        if isinstance(value, list):
                            response_parts.append(f"• {key.replace('_', ' ').title()}: {', '.join(value[:3])}")
                        else:
                            response_parts.append(f"• {key.replace('_', ' ').title()}: {value}")
                else:
                    response_parts.append(str(analysis_content))
            
            if "recommendations" in contribution:
                recommendations.extend(contribution["recommendations"])
        
        # Add unified recommendations
        if recommendations:
            unique_recommendations = list(set(recommendations))[:7]
            response_parts.append(f"\n**Unified Team Recommendations:**")
            for i, rec in enumerate(unique_recommendations, 1):
                response_parts.append(f"{i}. {rec}")
        
        # Calculate collaboration confidence
        collaboration_confidence = 0.9  # High confidence for collaborative analysis
        
        return AgentResponse(
            content="\n".join(response_parts),
            session_id=session_id,
            confidence=collaboration_confidence,
            recommendations=recommendations[:15],
            follow_up_questions=[
                "Would you like me to elaborate on any specific specialist's analysis?",
                "Should we dive deeper into any particular recommendation?",
                "Would you like to discuss implementation strategies for these suggestions?"
            ],
            metadata={
                "collaboration_type": "comprehensive_analysis",
                "collaboration_id": collaboration.id,
                "specialists_involved": [c.get("agent") for c in contributions],
                "collaborative_confidence": collaboration_confidence
            }
        )

    async def get_system_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system status
        
        Returns:
            System status including agent availability and performance
        """
        status = {
            "system_initialized": self.system_initialized,
            "specialists": {},
            "active_collaborations": len(self.active_collaborations),
            "system_performance": await self._get_system_performance()
        }
        
        # Get status of each specialist
        for name, agent in self.specialists.items():
            profile = agent.get_specialist_profile()
            status["specialists"][name] = {
                "availability": profile.availability_status,
                "capabilities": len(profile.capabilities),
                "performance_metrics": profile.performance_metrics,
                "collaboration_history": len(profile.collaboration_history)
            }
        
        return status

    async def _get_system_performance(self) -> Dict[str, Any]:
        """Get system-wide performance metrics"""
        total_consultations = 0
        total_collaborations = 0
        avg_response_time = 0
        
        for agent in self.specialists.values():
            metrics = agent.performance_metrics
            total_consultations += metrics.get("consultation_count", 0)
            total_collaborations += len(agent.collaboration_history)
            avg_response_time += metrics.get("response_time_avg", 0)
        
        specialist_count = len(self.specialists)
        avg_response_time = avg_response_time / specialist_count if specialist_count > 0 else 0
        
        return {
            "total_consultations": total_consultations,
            "total_collaborations": total_collaborations,
            "avg_response_time_minutes": round(avg_response_time, 2),
            "system_uptime": "active" if self.system_initialized else "inactive"
        }

    async def get_collaboration_insights(self, collaboration_id: str) -> Optional[Dict[str, Any]]:
        """
        Get insights about a specific collaboration
        
        Args:
            collaboration_id: Collaboration identifier
            
        Returns:
            Collaboration insights or None if not found
        """
        collaboration = self.active_collaborations.get(collaboration_id)
        if not collaboration:
            return None
        
        return {
            "collaboration_id": collaboration_id,
            "type": collaboration.collaboration_type,
            "participating_agents": collaboration.participating_agents,
            "objective": collaboration.objective,
            "status": collaboration.status,
            "duration": (
                (collaboration.completed_at or datetime.utcnow()) - collaboration.started_at
            ).total_seconds(),
            "shared_workspace_size": len(collaboration.shared_workspace),
            "message_count": len(collaboration.messages)
        }
