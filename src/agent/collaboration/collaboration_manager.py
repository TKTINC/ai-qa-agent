"""
Collaboration Manager
Manages agent collaboration sessions and coordination
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from ..communication.models import (
    Collaboration, CollaborationType, AgentMessage, MessageType,
    KnowledgeShare, ConsultationRequest
)
from ...core.exceptions import AgentError


logger = logging.getLogger(__name__)


class CollaborationManager:
    """
    Manages collaboration sessions between agents.
    
    Coordinates different types of collaboration patterns including
    sequential, parallel, consensus building, and peer review.
    """

    def __init__(self, specialists: Dict[str, Any]):
        self.specialists = specialists
        self.active_collaborations: Dict[str, Collaboration] = {}
        self.collaboration_history: List[Collaboration] = []
        self.message_routing = MessageRoutingManager()

    async def start_collaboration(
        self,
        collaboration_type: CollaborationType,
        participating_agents: List[str],
        objective: str,
        context: Dict[str, Any] = None
    ) -> Collaboration:
        """
        Start a new collaboration session
        
        Args:
            collaboration_type: Type of collaboration to conduct
            participating_agents: List of agent names to involve
            objective: Collaboration objective
            context: Additional context for collaboration
            
        Returns:
            Collaboration session object
        """
        collaboration = Collaboration(
            collaboration_type=collaboration_type,
            participating_agents=participating_agents,
            coordinator_agent="collaboration_manager",
            objective=objective,
            context=context or {}
        )
        
        self.active_collaborations[collaboration.id] = collaboration
        
        logger.info(f"Started {collaboration_type} collaboration with {len(participating_agents)} agents")
        
        # Notify participating agents
        await self._notify_collaboration_start(collaboration)
        
        return collaboration

    async def _notify_collaboration_start(self, collaboration: Collaboration) -> None:
        """Notify agents about collaboration start"""
        for agent_name in collaboration.participating_agents:
            if agent_name in self.specialists:
                message = AgentMessage(
                    sender_agent="collaboration_manager",
                    recipient_agent=agent_name,
                    message_type=MessageType.COLLABORATION_REQUEST,
                    content=f"Collaboration started: {collaboration.objective}",
                    data={
                        "collaboration": collaboration.model_dump(),
                        "role": "participant"
                    }
                )
                
                # Route message to agent
                await self.message_routing.route_message(message, self.specialists)

    async def coordinate_sequential_collaboration(
        self,
        collaboration: Collaboration,
        task: str
    ) -> Dict[str, Any]:
        """
        Coordinate sequential collaboration where agents work one after another
        """
        results = []
        accumulated_context = collaboration.context.copy()
        
        for i, agent_name in enumerate(collaboration.participating_agents):
            if agent_name not in self.specialists:
                logger.warning(f"Agent {agent_name} not available for collaboration")
                continue
            
            agent = self.specialists[agent_name]
            
            # Prepare collaboration context with previous results
            collab_context = {
                **accumulated_context,
                "collaboration_id": collaboration.id,
                "step": i + 1,
                "previous_results": results,
                "next_agents": collaboration.participating_agents[i+1:]
            }
            
            try:
                contribution = await agent.collaborate_on_task(task, collab_context)
                results.append(contribution)
                
                # Update shared workspace
                collaboration.shared_workspace[f"{agent_name}_contribution"] = contribution
                
                # Share insights with next agent
                if i < len(collaboration.participating_agents) - 1:
                    await self._share_insights_with_next_agent(
                        collaboration, agent_name, contribution, i + 1
                    )
                
            except Exception as e:
                logger.error(f"Error in sequential collaboration with {agent_name}: {str(e)}")
                results.append({"error": str(e), "agent": agent_name})
        
        return {
            "collaboration_type": "sequential",
            "results": results,
            "summary": await self._summarize_sequential_results(results)
        }

    async def coordinate_parallel_collaboration(
        self,
        collaboration: Collaboration,
        task: str
    ) -> Dict[str, Any]:
        """
        Coordinate parallel collaboration where agents work simultaneously
        """
        # Prepare tasks for all agents
        agent_tasks = []
        
        for agent_name in collaboration.participating_agents:
            if agent_name in self.specialists:
                agent = self.specialists[agent_name]
                collab_context = {
                    **collaboration.context,
                    "collaboration_id": collaboration.id,
                    "collaboration_mode": "parallel",
                    "other_agents": [a for a in collaboration.participating_agents if a != agent_name]
                }
                
                agent_tasks.append(agent.collaborate_on_task(task, collab_context))
            else:
                agent_tasks.append(self._create_error_result(agent_name, "Agent not available"))
        
        # Execute all tasks in parallel
        try:
            results = await asyncio.gather(*agent_tasks, return_exceptions=True)
            
            # Process results and handle exceptions
            processed_results = []
            for i, result in enumerate(results):
                agent_name = collaboration.participating_agents[i]
                if isinstance(result, Exception):
                    processed_results.append({
                        "error": str(result),
                        "agent": agent_name
                    })
                else:
                    processed_results.append(result)
                    collaboration.shared_workspace[f"{agent_name}_contribution"] = result
            
            return {
                "collaboration_type": "parallel",
                "results": processed_results,
                "summary": await self._summarize_parallel_results(processed_results)
            }
            
        except Exception as e:
            logger.error(f"Error in parallel collaboration: {str(e)}")
            raise AgentError(f"Parallel collaboration failed: {str(e)}")

    async def coordinate_consensus_collaboration(
        self,
        collaboration: Collaboration,
        task: str
    ) -> Dict[str, Any]:
        """
        Coordinate consensus-building collaboration
        """
        # Phase 1: Initial contributions
        initial_contributions = await self.coordinate_parallel_collaboration(collaboration, task)
        
        # Phase 2: Review and feedback
        feedback_round = await self._conduct_feedback_round(
            collaboration, initial_contributions["results"]
        )
        
        # Phase 3: Consensus building
        consensus = await self._build_consensus(
            collaboration, initial_contributions["results"], feedback_round
        )
        
        return {
            "collaboration_type": "consensus",
            "initial_contributions": initial_contributions,
            "feedback_round": feedback_round,
            "consensus": consensus,
            "summary": await self._summarize_consensus_results(consensus)
        }

    async def _conduct_feedback_round(
        self,
        collaboration: Collaboration,
        initial_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Conduct feedback round where agents review each other's work"""
        feedback_tasks = []
        
        for i, agent_name in enumerate(collaboration.participating_agents):
            if agent_name not in self.specialists:
                continue
                
            agent = self.specialists[agent_name]
            
            # Get other agents' contributions for review
            other_contributions = [
                result for j, result in enumerate(initial_results) 
                if j != i and not result.get("error")
            ]
            
            if other_contributions:
                feedback_context = {
                    **collaboration.context,
                    "collaboration_id": collaboration.id,
                    "phase": "feedback",
                    "contributions_to_review": other_contributions,
                    "own_contribution": initial_results[i] if i < len(initial_results) else None
                }
                
                # Request feedback from agent
                feedback_task = self._get_agent_feedback(agent, feedback_context)
                feedback_tasks.append(feedback_task)
        
        try:
            feedback_results = await asyncio.gather(*feedback_tasks, return_exceptions=True)
            return [r for r in feedback_results if not isinstance(r, Exception)]
        except Exception as e:
            logger.error(f"Error in feedback round: {str(e)}")
            return []

    async def _get_agent_feedback(self, agent, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get feedback from an agent on other contributions"""
        # This would be implemented with a specific feedback method
        # For now, simulate feedback generation
        return {
            "agent": agent.name,
            "feedback_type": "peer_review",
            "comments": f"Feedback from {agent.specialization}",
            "suggestions": ["Consider additional validation", "Good approach overall"],
            "agreement_level": 0.8
        }

    async def _build_consensus(
        self,
        collaboration: Collaboration,
        initial_results: List[Dict[str, Any]],
        feedback: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Build consensus from contributions and feedback"""
        # Analyze agreement levels
        agreement_scores = [f.get("agreement_level", 0.5) for f in feedback]
        avg_agreement = sum(agreement_scores) / len(agreement_scores) if agreement_scores else 0.5
        
        # Synthesize consensus
        consensus = {
            "agreement_level": avg_agreement,
            "consensus_reached": avg_agreement > 0.7,
            "key_points": self._extract_consensus_points(initial_results, feedback),
            "areas_of_disagreement": self._identify_disagreements(feedback),
            "final_recommendations": self._synthesize_recommendations(initial_results, feedback)
        }
        
        return consensus

    def _extract_consensus_points(
        self,
        results: List[Dict[str, Any]],
        feedback: List[Dict[str, Any]]
    ) -> List[str]:
        """Extract points of consensus from results and feedback"""
        # Simple implementation - would be more sophisticated in practice
        consensus_points = [
            "All agents agree on the importance of comprehensive testing",
            "Quality assurance is a key priority across all recommendations",
            "Performance considerations should be integrated throughout"
        ]
        return consensus_points

    def _identify_disagreements(self, feedback: List[Dict[str, Any]]) -> List[str]:
        """Identify areas where agents disagree"""
        disagreements = []
        for f in feedback:
            if f.get("agreement_level", 1.0) < 0.6:
                disagreements.append(f"Disagreement noted by {f.get('agent', 'unknown')}")
        return disagreements

    def _synthesize_recommendations(
        self,
        results: List[Dict[str, Any]],
        feedback: List[Dict[str, Any]]
    ) -> List[str]:
        """Synthesize final recommendations from all inputs"""
        recommendations = []
        
        # Extract recommendations from results
        for result in results:
            if "recommendations" in result:
                recommendations.extend(result["recommendations"])
        
        # Add feedback-based recommendations
        for f in feedback:
            if "suggestions" in f:
                recommendations.extend(f["suggestions"])
        
        # Remove duplicates and return top recommendations
        unique_recommendations = list(set(recommendations))
        return unique_recommendations[:10]

    async def _summarize_sequential_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarize sequential collaboration results"""
        successful_steps = sum(1 for r in results if not r.get("error"))
        return {
            "total_steps": len(results),
            "successful_steps": successful_steps,
            "success_rate": successful_steps / len(results) if results else 0,
            "final_outcome": "completed" if successful_steps == len(results) else "partial"
        }

    async def _summarize_parallel_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarize parallel collaboration results"""
        successful_agents = sum(1 for r in results if not r.get("error"))
        return {
            "total_agents": len(results),
            "successful_agents": successful_agents,
            "success_rate": successful_agents / len(results) if results else 0,
            "collaboration_efficiency": "high" if successful_agents / len(results) > 0.8 else "medium"
        }

    async def _summarize_consensus_results(self, consensus: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize consensus collaboration results"""
        return {
            "consensus_achieved": consensus.get("consensus_reached", False),
            "agreement_level": consensus.get("agreement_level", 0),
            "consensus_quality": "strong" if consensus.get("agreement_level", 0) > 0.8 else "moderate",
            "key_outcomes": len(consensus.get("final_recommendations", []))
        }

    async def _create_error_result(self, agent_name: str, error_message: str) -> Dict[str, Any]:
        """Create error result for failed agent"""
        return {
            "agent": agent_name,
            "error": error_message,
            "timestamp": datetime.utcnow().isoformat()
        }


class MessageRoutingManager:
    """Manages message routing between agents"""
    
    async def route_message(self, message: AgentMessage, specialists: Dict[str, Any]) -> bool:
        """Route message to appropriate agent"""
        try:
            recipient_agent = specialists.get(message.recipient_agent)
            if recipient_agent:
                await recipient_agent.process_message(message)
                return True
            else:
                logger.warning(f"Recipient agent {message.recipient_agent} not found")
                return False
        except Exception as e:
            logger.error(f"Error routing message: {str(e)}")
            return False
