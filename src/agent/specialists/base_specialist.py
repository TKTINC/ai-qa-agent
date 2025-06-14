"""
Base Specialist Agent
Foundation for all specialist agent implementations
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from abc import ABC, abstractmethod

from ..core.models import AgentState, AgentResponse, Goal, ReasoningStep
from ..reasoning.react_engine import ReActReasoner
from ..tools.tool_manager import ToolManager
from ..communication.models import (
    AgentMessage, ConsultationRequest, ConsultationResponse,
    KnowledgeShare, AgentCapability, SpecialistProfile, MessageType
)
from ...core.exceptions import AgentError


logger = logging.getLogger(__name__)


class SpecialistAgent(ABC):
    """
    Abstract base class for specialist agents.
    
    All specialist agents inherit from this class and implement
    domain-specific reasoning and problem-solving capabilities.
    """

    def __init__(self, name: str, specialization: str, expertise_domains: List[str]):
        self.name = name
        self.specialization = specialization
        self.expertise_domains = expertise_domains
        
        # Core components
        self.reasoner = ReActReasoner()
        self.tool_manager = ToolManager()
        
        # Specialist capabilities
        self.capabilities: List[AgentCapability] = []
        self.performance_metrics: Dict[str, float] = {}
        self.collaboration_history: List[str] = []
        
        # Communication
        self.message_queue: List[AgentMessage] = []
        self.active_collaborations: Dict[str, Any] = {}
        
        # State
        self.availability_status = "available"
        self.current_tasks: List[str] = []
        self.learning_history: List[Dict[str, Any]] = []

    @abstractmethod
    async def analyze_problem(self, problem: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a problem within the agent's domain of expertise
        
        Args:
            problem: Problem description
            context: Additional context and data
            
        Returns:
            Analysis results with insights and recommendations
        """
        pass

    @abstractmethod
    async def provide_consultation(self, request: ConsultationRequest) -> ConsultationResponse:
        """
        Provide expert consultation on a specific question
        
        Args:
            request: Consultation request from another agent
            
        Returns:
            Expert consultation response
        """
        pass

    @abstractmethod
    async def collaborate_on_task(self, task: str, collaboration_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Collaborate with other agents on a shared task
        
        Args:
            task: Task description
            collaboration_context: Context including other agents and shared data
            
        Returns:
            Collaboration contribution
        """
        pass

    async def initialize_specialist(self) -> bool:
        """
        Initialize the specialist agent
        
        Returns:
            True if initialization successful
        """
        try:
            # Initialize base capabilities
            await self._initialize_capabilities()
            
            # Set up tools specific to this specialist
            await self._setup_specialist_tools()
            
            # Initialize performance tracking
            self._initialize_performance_metrics()
            
            logger.info(f"Specialist agent {self.name} initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize specialist {self.name}: {str(e)}")
            return False

    async def _initialize_capabilities(self) -> None:
        """Initialize agent capabilities (override in subclasses)"""
        # Base capabilities that all specialists have
        base_capabilities = [
            AgentCapability(
                name="consultation",
                description="Provide expert consultation in specialization area",
                confidence_level=0.9
            ),
            AgentCapability(
                name="collaboration",
                description="Collaborate effectively with other agents",
                confidence_level=0.8
            ),
            AgentCapability(
                name="knowledge_sharing",
                description="Share knowledge and insights with team",
                confidence_level=0.85
            )
        ]
        self.capabilities.extend(base_capabilities)

    async def _setup_specialist_tools(self) -> None:
        """Set up tools specific to this specialist (override in subclasses)"""
        pass

    def _initialize_performance_metrics(self) -> None:
        """Initialize performance tracking metrics"""
        self.performance_metrics = {
            "consultation_success_rate": 1.0,
            "collaboration_effectiveness": 0.8,
            "response_time_avg": 5.0,  # minutes
            "knowledge_sharing_frequency": 0.0,
            "user_satisfaction_avg": 0.9
        }

    async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """
        Process incoming message from another agent
        
        Args:
            message: Incoming message
            
        Returns:
            Response message if applicable
        """
        try:
            self.message_queue.append(message)
            
            if message.message_type == MessageType.CONSULTATION:
                # Handle consultation request
                consultation_data = message.data.get("consultation_request")
                if consultation_data:
                    request = ConsultationRequest(**consultation_data)
                    response = await self.provide_consultation(request)
                    
                    return AgentMessage(
                        sender_agent=self.name,
                        recipient_agent=message.sender_agent,
                        message_type=MessageType.RESULT_SHARING,
                        content=f"Consultation response: {response.response}",
                        data={"consultation_response": response.model_dump()}
                    )
                    
            elif message.message_type == MessageType.KNOWLEDGE_SHARE:
                # Process shared knowledge
                await self._process_shared_knowledge(message)
                
            elif message.message_type == MessageType.COLLABORATION_REQUEST:
                # Handle collaboration request
                return await self._handle_collaboration_request(message)
                
            return None
            
        except Exception as e:
            logger.error(f"Error processing message in {self.name}: {str(e)}")
            return None

    async def _process_shared_knowledge(self, message: AgentMessage) -> None:
        """Process knowledge shared by another agent"""
        knowledge_data = message.data.get("knowledge_share")
        if knowledge_data:
            knowledge = KnowledgeShare(**knowledge_data)
            
            # Store and integrate the knowledge
            self.learning_history.append({
                "type": "knowledge_received",
                "source": knowledge.sharing_agent,
                "knowledge_type": knowledge.knowledge_type,
                "content": knowledge.content,
                "confidence": knowledge.confidence,
                "received_at": datetime.utcnow()
            })
            
            logger.info(f"{self.name} received knowledge: {knowledge.title}")

    async def _handle_collaboration_request(self, message: AgentMessage) -> AgentMessage:
        """Handle request to collaborate"""
        collaboration_data = message.data.get("collaboration")
        
        if self.availability_status == "available":
            response_content = f"{self.name} accepts collaboration request"
            response_data = {"accepted": True, "capabilities": [cap.model_dump() for cap in self.capabilities]}
        else:
            response_content = f"{self.name} is currently {self.availability_status}"
            response_data = {"accepted": False, "status": self.availability_status}
        
        return AgentMessage(
            sender_agent=self.name,
            recipient_agent=message.sender_agent,
            message_type=MessageType.STATUS_UPDATE,
            content=response_content,
            data=response_data
        )

    async def share_knowledge(self, knowledge: KnowledgeShare, target_agents: List[str]) -> None:
        """
        Share knowledge with other agents
        
        Args:
            knowledge: Knowledge to share
            target_agents: List of agent names to share with
        """
        knowledge.sharing_agent = self.name
        knowledge.receiving_agents = target_agents
        
        # This would typically send through a communication manager
        # For now, we'll log the knowledge sharing
        logger.info(f"{self.name} sharing knowledge '{knowledge.title}' with {target_agents}")
        
        # Update performance metrics
        self.performance_metrics["knowledge_sharing_frequency"] += 1

    async def update_capability(self, capability_name: str, success: bool, execution_time: float) -> None:
        """
        Update capability based on usage outcome
        
        Args:
            capability_name: Name of capability used
            success: Whether the capability was used successfully
            execution_time: Time taken for execution
        """
        for cap in self.capabilities:
            if cap.name == capability_name:
                cap.experience_count += 1
                cap.last_used = datetime.utcnow()
                
                # Update success rate with weighted average
                if cap.experience_count == 1:
                    cap.success_rate = 1.0 if success else 0.0
                else:
                    weight = 0.1
                    new_success = 1.0 if success else 0.0
                    cap.success_rate = (1 - weight) * cap.success_rate + weight * new_success
                
                break

    def get_specialist_profile(self) -> SpecialistProfile:
        """
        Get detailed profile of this specialist
        
        Returns:
            SpecialistProfile with capabilities and metrics
        """
        return SpecialistProfile(
            agent_name=self.name,
            specialization=self.specialization,
            capabilities=self.capabilities,
            expertise_domains=self.expertise_domains,
            collaboration_history=self.collaboration_history,
            performance_metrics=self.performance_metrics,
            availability_status=self.availability_status
        )

    async def suggest_improvements(self) -> List[str]:
        """
        Suggest improvements based on performance analysis
        
        Returns:
            List of improvement suggestions
        """
        suggestions = []
        
        # Analyze performance metrics
        if self.performance_metrics.get("consultation_success_rate", 1.0) < 0.8:
            suggestions.append("Consider additional training in consultation techniques")
        
        if self.performance_metrics.get("response_time_avg", 0) > 10:
            suggestions.append("Work on reducing response time for better collaboration")
        
        if self.performance_metrics.get("knowledge_sharing_frequency", 0) < 5:
            suggestions.append("Increase knowledge sharing to help team learning")
        
        # Analyze capability performance
        for cap in self.capabilities:
            if cap.success_rate < 0.7 and cap.experience_count > 5:
                suggestions.append(f"Improve {cap.name} capability through focused practice")
        
        return suggestions

    async def learn_from_interaction(self, interaction_data: Dict[str, Any]) -> None:
        """
        Learn from interactions to improve performance
        
        Args:
            interaction_data: Data about the interaction and its outcome
        """
        self.learning_history.append({
            "type": "interaction_learning",
            "data": interaction_data,
            "learned_at": datetime.utcnow()
        })
        
        # Update performance metrics based on interaction
        if "success" in interaction_data:
            # Update relevant metrics
            interaction_type = interaction_data.get("type", "general")
            if interaction_type == "consultation":
                current_rate = self.performance_metrics.get("consultation_success_rate", 1.0)
                success = interaction_data["success"]
                weight = 0.1
                new_rate = (1 - weight) * current_rate + weight * (1.0 if success else 0.0)
                self.performance_metrics["consultation_success_rate"] = new_rate
