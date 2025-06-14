#!/bin/bash
# Setup Script for Sprint 2.3: Multi-Agent Architecture & Collaboration
# AI QA Agent - Sprint 2.3

set -e
echo "🚀 Setting up Sprint 2.3: Multi-Agent Architecture & Collaboration..."

# Check prerequisites (Sprint 2.2 completed)
if [ ! -f "src/agent/tools/tool_manager.py" ]; then
    echo "❌ Error: Sprint 2.2 must be completed first"
    echo "Missing: src/agent/tools/tool_manager.py"
    exit 1
fi

if [ ! -f "src/agent/tools/analysis/code_analysis_tool.py" ]; then
    echo "❌ Error: Sprint 2.2 must be completed first"
    echo "Missing: src/agent/tools/analysis/code_analysis_tool.py"
    exit 1
fi

# Install new dependencies with pip3 (macOS compatible)
echo "📦 Installing new dependencies..."
pip3 install networkx==3.2.1
pip3 install matplotlib==3.8.2
pip3 install plotly==5.17.0
pip3 install websockets==12.0
pip3 install aioredis==2.0.1

# Create multi-agent directory structure
echo "📁 Creating multi-agent directory structure..."
mkdir -p src/agent/multi_agent
mkdir -p src/agent/specialists
mkdir -p src/agent/communication
mkdir -p src/agent/collaboration
mkdir -p tests/unit/test_agent/test_multi_agent
mkdir -p tests/unit/test_agent/test_specialists
mkdir -p tests/unit/test_agent/test_communication

# Create multi-agent __init__.py files
echo "📄 Creating multi-agent __init__.py files..."
cat > src/agent/multi_agent/__init__.py << 'EOF'
"""
AI QA Agent - Multi-Agent System
Sprint 2.3: Multi-Agent Architecture & Collaboration

This module implements the multi-agent system including:
- Multi-agent coordination and communication
- Specialist agent implementations
- Collaborative problem-solving patterns
- Cross-agent knowledge sharing
- Agent teamwork and consensus building
"""

from .agent_system import QAAgentSystem
from .collaboration_manager import CollaborationManager

__all__ = [
    'QAAgentSystem',
    'CollaborationManager'
]
EOF

cat > src/agent/specialists/__init__.py << 'EOF'
"""Specialist agent implementations"""
EOF

cat > src/agent/communication/__init__.py << 'EOF'
"""Agent communication components"""
EOF

cat > src/agent/collaboration/__init__.py << 'EOF'
"""Agent collaboration components"""
EOF

# Create Agent Communication Models
echo "📄 Creating src/agent/communication/models.py..."
cat > src/agent/communication/models.py << 'EOF'
"""
Agent Communication Models
Defines data structures for inter-agent communication and collaboration
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field
import uuid


class MessageType(str, Enum):
    """Types of inter-agent messages"""
    CONSULTATION = "consultation"
    COLLABORATION_REQUEST = "collaboration_request"
    KNOWLEDGE_SHARE = "knowledge_share"
    TASK_DELEGATION = "task_delegation"
    RESULT_SHARING = "result_sharing"
    CONSENSUS_BUILDING = "consensus_building"
    STATUS_UPDATE = "status_update"


class CollaborationType(str, Enum):
    """Types of agent collaboration"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONSENSUS = "consensus"
    PEER_REVIEW = "peer_review"
    EXPERT_CONSULTATION = "expert_consultation"


class AgentRole(str, Enum):
    """Roles that agents can play"""
    COORDINATOR = "coordinator"
    SPECIALIST = "specialist"
    REVIEWER = "reviewer"
    CONSULTANT = "consultant"
    EXECUTOR = "executor"


class AgentMessage(BaseModel):
    """Message between agents"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    sender_agent: str
    recipient_agent: str
    message_type: MessageType
    content: str
    data: Dict[str, Any] = Field(default_factory=dict)
    context: Dict[str, Any] = Field(default_factory=dict)
    priority: int = Field(default=3, ge=1, le=5)
    requires_response: bool = False
    response_deadline: Optional[datetime] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class Collaboration(BaseModel):
    """Collaboration session between agents"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    collaboration_type: CollaborationType
    participating_agents: List[str]
    coordinator_agent: str
    objective: str
    context: Dict[str, Any] = Field(default_factory=dict)
    messages: List[AgentMessage] = Field(default_factory=list)
    shared_workspace: Dict[str, Any] = Field(default_factory=dict)
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    status: str = "active"  # active, completed, failed, cancelled


class ConsultationRequest(BaseModel):
    """Request for specialist consultation"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    requesting_agent: str
    specialist_agent: str
    question: str
    context: Dict[str, Any] = Field(default_factory=dict)
    urgency: int = Field(default=3, ge=1, le=5)
    expected_response_time: Optional[int] = None  # minutes
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ConsultationResponse(BaseModel):
    """Response to consultation request"""
    consultation_id: str
    responding_agent: str
    response: str
    confidence: float = Field(ge=0.0, le=1.0)
    recommendations: List[str] = Field(default_factory=list)
    additional_data: Dict[str, Any] = Field(default_factory=dict)
    follow_up_questions: List[str] = Field(default_factory=list)
    responded_at: datetime = Field(default_factory=datetime.utcnow)


class KnowledgeShare(BaseModel):
    """Knowledge sharing between agents"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    sharing_agent: str
    receiving_agents: List[str]
    knowledge_type: str
    title: str
    content: str
    data: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(ge=0.0, le=1.0)
    source: str  # experience, analysis, learning, external
    shared_at: datetime = Field(default_factory=datetime.utcnow)


class AgentCapability(BaseModel):
    """Agent capability description"""
    name: str
    description: str
    confidence_level: float = Field(ge=0.0, le=1.0)
    experience_count: int = 0
    success_rate: float = Field(ge=0.0, le=1.0, default=1.0)
    last_used: Optional[datetime] = None


class SpecialistProfile(BaseModel):
    """Profile of a specialist agent"""
    agent_name: str
    specialization: str
    capabilities: List[AgentCapability]
    expertise_domains: List[str]
    collaboration_history: List[str] = Field(default_factory=list)
    performance_metrics: Dict[str, float] = Field(default_factory=dict)
    preferred_collaboration_types: List[CollaborationType] = Field(default_factory=list)
    availability_status: str = "available"  # available, busy, offline
EOF

# Create Base Specialist Agent
echo "📄 Creating src/agent/specialists/base_specialist.py..."
cat > src/agent/specialists/base_specialist.py << 'EOF'
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
EOF

# Create Test Architect Agent
echo "📄 Creating src/agent/specialists/test_architect.py..."
cat > src/agent/specialists/test_architect.py << 'EOF'
"""
Test Architect Agent
Specializes in test strategy and architecture design
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


class TestArchitectAgent(SpecialistAgent):
    """
    Specialist agent focused on test architecture and strategy design.
    
    Expertise areas:
    - Test strategy development
    - Test architecture design
    - Test coverage optimization
    - Testing framework selection
    - Quality assurance planning
    """

    def __init__(self):
        super().__init__(
            name="test_architect",
            specialization="Test Architecture & Strategy",
            expertise_domains=[
                "test_strategy", "test_architecture", "test_coverage",
                "testing_frameworks", "quality_assurance", "test_planning"
            ]
        )

    async def _initialize_capabilities(self) -> None:
        """Initialize Test Architect specific capabilities"""
        await super()._initialize_capabilities()
        
        specialist_capabilities = [
            AgentCapability(
                name="test_strategy_design",
                description="Design comprehensive testing strategies for projects",
                confidence_level=0.95
            ),
            AgentCapability(
                name="test_architecture_planning",
                description="Plan test architecture and organization",
                confidence_level=0.92
            ),
            AgentCapability(
                name="coverage_optimization",
                description="Optimize test coverage for maximum effectiveness",
                confidence_level=0.90
            ),
            AgentCapability(
                name="framework_selection",
                description="Select optimal testing frameworks and tools",
                confidence_level=0.88
            ),
            AgentCapability(
                name="quality_metrics_design",
                description="Design quality metrics and success criteria",
                confidence_level=0.87
            )
        ]
        self.capabilities.extend(specialist_capabilities)

    async def _setup_specialist_tools(self) -> None:
        """Set up Test Architect specific tools"""
        # Register code analysis tool for architecture assessment
        from ..tools.analysis.code_analysis_tool import CodeAnalysisTool
        analysis_tool = CodeAnalysisTool()
        await self.tool_manager.register_tool(analysis_tool)

    async def analyze_problem(self, problem: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze testing-related problems and provide architectural insights
        
        Args:
            problem: Problem description
            context: Analysis context including code, requirements, constraints
            
        Returns:
            Analysis with testing strategy recommendations
        """
        start_time = datetime.utcnow()
        
        try:
            analysis_result = {
                "analysis_type": "test_architecture",
                "problem_assessment": await self._assess_testing_problem(problem, context),
                "architecture_recommendations": await self._design_test_architecture(problem, context),
                "strategy_recommendations": await self._develop_test_strategy(problem, context),
                "coverage_analysis": await self._analyze_coverage_requirements(problem, context),
                "framework_recommendations": await self._recommend_testing_frameworks(context),
                "quality_metrics": await self._define_quality_metrics(problem, context),
                "implementation_plan": await self._create_implementation_plan(problem, context)
            }
            
            # Update capability usage
            await self.update_capability("test_strategy_design", True, 
                                       (datetime.utcnow() - start_time).total_seconds())
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Test Architect analysis failed: {str(e)}")
            await self.update_capability("test_strategy_design", False,
                                       (datetime.utcnow() - start_time).total_seconds())
            raise AgentError(f"Test architecture analysis failed: {str(e)}")

    async def _assess_testing_problem(self, problem: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Assess the testing problem and identify key challenges"""
        assessment = {
            "problem_type": self._classify_testing_problem(problem),
            "complexity_level": self._assess_complexity(problem, context),
            "key_challenges": self._identify_testing_challenges(problem, context),
            "constraints": context.get("constraints", []),
            "current_state": context.get("current_testing_state", {})
        }
        
        return assessment

    def _classify_testing_problem(self, problem: str) -> str:
        """Classify the type of testing problem"""
        problem_lower = problem.lower()
        
        if any(word in problem_lower for word in ["legacy", "no tests", "untested"]):
            return "legacy_code_testing"
        elif any(word in problem_lower for word in ["performance", "load", "speed"]):
            return "performance_testing"
        elif any(word in problem_lower for word in ["security", "vulnerability", "auth"]):
            return "security_testing"
        elif any(word in problem_lower for word in ["integration", "api", "service"]):
            return "integration_testing"
        elif any(word in problem_lower for word in ["ui", "frontend", "user interface"]):
            return "ui_testing"
        else:
            return "general_testing_strategy"

    def _assess_complexity(self, problem: str, context: Dict[str, Any]) -> str:
        """Assess the complexity level of the testing challenge"""
        complexity_indicators = 0
        
        # Check for complexity indicators
        if context.get("codebase_size", 0) > 10000:  # lines of code
            complexity_indicators += 1
        
        if len(context.get("dependencies", [])) > 10:
            complexity_indicators += 1
            
        if context.get("has_legacy_code", False):
            complexity_indicators += 1
            
        if context.get("multiple_environments", False):
            complexity_indicators += 1
            
        if complexity_indicators >= 3:
            return "high"
        elif complexity_indicators >= 2:
            return "medium"
        else:
            return "low"

    def _identify_testing_challenges(self, problem: str, context: Dict[str, Any]) -> List[str]:
        """Identify specific testing challenges"""
        challenges = []
        
        # Analyze problem description for challenges
        problem_lower = problem.lower()
        
        if "no tests" in problem_lower or "untested" in problem_lower:
            challenges.append("Lack of existing test coverage")
            
        if "legacy" in problem_lower:
            challenges.append("Legacy code dependencies and coupling")
            
        if "complex" in problem_lower or "complicated" in problem_lower:
            challenges.append("High system complexity")
            
        if "time" in problem_lower or "deadline" in problem_lower:
            challenges.append("Time constraints for implementation")
            
        # Analyze context for additional challenges
        if context.get("team_experience", "intermediate") == "beginner":
            challenges.append("Limited team testing experience")
            
        if not context.get("ci_cd_available", True):
            challenges.append("Lack of automated testing infrastructure")
            
        return challenges

    async def _design_test_architecture(self, problem: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Design test architecture recommendations"""
        architecture = {
            "test_layers": self._recommend_test_layers(context),
            "test_organization": self._recommend_test_organization(context),
            "test_data_strategy": self._design_test_data_strategy(context),
            "environment_strategy": self._design_environment_strategy(context),
            "automation_strategy": self._design_automation_strategy(context)
        }
        
        return architecture

    def _recommend_test_layers(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Recommend test layer architecture"""
        layers = {
            "unit_tests": {
                "coverage_target": "80-90%",
                "focus": "Individual components and functions",
                "tools": ["pytest", "unittest", "jest"],
                "priority": "high"
            },
            "integration_tests": {
                "coverage_target": "60-70%",
                "focus": "Component interactions and API endpoints",
                "tools": ["pytest", "postman", "supertest"],
                "priority": "high"
            },
            "end_to_end_tests": {
                "coverage_target": "20-30%",
                "focus": "Critical user journeys and workflows",
                "tools": ["selenium", "cypress", "playwright"],
                "priority": "medium"
            }
        }
        
        # Adjust based on context
        if context.get("api_heavy", False):
            layers["api_tests"] = {
                "coverage_target": "90%+",
                "focus": "API contracts and data validation",
                "tools": ["postman", "rest-assured", "tavern"],
                "priority": "high"
            }
            
        return layers

    def _recommend_test_organization(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Recommend test organization structure"""
        return {
            "structure": "mirror_source_structure",
            "naming_convention": "test_[function_name] or [ClassName]Test",
            "file_organization": {
                "unit_tests": "tests/unit/",
                "integration_tests": "tests/integration/",
                "e2e_tests": "tests/e2e/",
                "fixtures": "tests/fixtures/",
                "utilities": "tests/utils/"
            },
            "test_categories": [
                "smoke_tests", "regression_tests", "performance_tests", "security_tests"
            ]
        }

    async def _develop_test_strategy(self, problem: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Develop comprehensive test strategy"""
        strategy = {
            "approach": self._select_testing_approach(problem, context),
            "priorities": self._prioritize_testing_areas(context),
            "phases": self._plan_testing_phases(context),
            "success_criteria": self._define_success_criteria(context),
            "risk_mitigation": self._identify_testing_risks(context)
        }
        
        return strategy

    def _select_testing_approach(self, problem: str, context: Dict[str, Any]) -> str:
        """Select the most appropriate testing approach"""
        if "legacy" in problem.lower():
            return "characterization_testing_first"
        elif context.get("new_project", False):
            return "tdd_approach"
        elif context.get("has_some_tests", False):
            return "gap_analysis_and_enhancement"
        else:
            return "risk_based_testing"

    async def provide_consultation(self, request: ConsultationRequest) -> ConsultationResponse:
        """
        Provide expert consultation on testing architecture
        
        Args:
            request: Consultation request
            
        Returns:
            Expert response with testing recommendations
        """
        start_time = datetime.utcnow()
        
        try:
            question = request.question.lower()
            context = request.context
            
            if "strategy" in question:
                response = await self._consult_on_strategy(request)
            elif "architecture" in question:
                response = await self._consult_on_architecture(request)
            elif "coverage" in question:
                response = await self._consult_on_coverage(request)
            elif "framework" in question or "tool" in question:
                response = await self._consult_on_tools(request)
            else:
                response = await self._provide_general_consultation(request)
            
            consultation_response = ConsultationResponse(
                consultation_id=request.id,
                responding_agent=self.name,
                response=response["answer"],
                confidence=response["confidence"],
                recommendations=response["recommendations"],
                follow_up_questions=response["follow_ups"]
            )
            
            # Update performance metrics
            await self.update_capability("consultation", True,
                                       (datetime.utcnow() - start_time).total_seconds())
            
            return consultation_response
            
        except Exception as e:
            logger.error(f"Test Architect consultation failed: {str(e)}")
            await self.update_capability("consultation", False,
                                       (datetime.utcnow() - start_time).total_seconds())
            
            return ConsultationResponse(
                consultation_id=request.id,
                responding_agent=self.name,
                response=f"I encountered an issue providing consultation: {str(e)}",
                confidence=0.0
            )

    async def _consult_on_strategy(self, request: ConsultationRequest) -> Dict[str, Any]:
        """Provide consultation on testing strategy"""
        return {
            "answer": "For test strategy, I recommend starting with risk assessment to identify critical areas, then implementing a layered testing approach with unit tests forming the foundation, integration tests covering component interactions, and end-to-end tests for critical user journeys.",
            "confidence": 0.92,
            "recommendations": [
                "Begin with risk-based test prioritization",
                "Implement testing pyramid structure",
                "Focus on high-value, high-risk areas first",
                "Establish clear testing standards and guidelines"
            ],
            "follow_ups": [
                "What is your current test coverage percentage?",
                "Do you have any existing testing infrastructure?",
                "What are your main quality concerns?"
            ]
        }

    async def _consult_on_architecture(self, request: ConsultationRequest) -> Dict[str, Any]:
        """Provide consultation on test architecture"""
        return {
            "answer": "Test architecture should mirror your application architecture while maintaining clear separation of concerns. I recommend organizing tests by type (unit, integration, e2e), using consistent naming conventions, and implementing shared utilities for common testing patterns.",
            "confidence": 0.90,
            "recommendations": [
                "Mirror source code structure in test organization",
                "Separate test types into distinct directories",
                "Create reusable test utilities and fixtures",
                "Implement test data management strategy"
            ],
            "follow_ups": [
                "What testing frameworks are you currently using?",
                "How is your application architecture organized?",
                "Do you need help with test data management?"
            ]
        }

    async def collaborate_on_task(self, task: str, collaboration_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Collaborate with other agents on testing-related tasks
        
        Args:
            task: Collaboration task description
            collaboration_context: Context including other agents and shared data
            
        Returns:
            Test Architect's contribution to the collaborative effort
        """
        contribution = {
            "agent": self.name,
            "specialization_applied": "test_architecture",
            "analysis": await self._analyze_collaborative_task(task, collaboration_context),
            "recommendations": await self._generate_collaborative_recommendations(task, collaboration_context),
            "test_strategy": await self._contribute_test_strategy(task, collaboration_context),
            "coordination_suggestions": await self._suggest_coordination(collaboration_context)
        }
        
        return contribution

    async def _analyze_collaborative_task(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the collaborative task from testing perspective"""
        return {
            "testing_implications": self._identify_testing_implications(task),
            "quality_considerations": self._identify_quality_considerations(task),
            "integration_points": self._identify_integration_points(task, context),
            "risk_assessment": self._assess_testing_risks_for_task(task)
        }

    def _identify_testing_implications(self, task: str) -> List[str]:
        """Identify testing implications of a task"""
        implications = []
        task_lower = task.lower()
        
        if "api" in task_lower:
            implications.extend([
                "API contract testing needed",
                "Request/response validation required",
                "Error handling scenarios to test"
            ])
            
        if "database" in task_lower:
            implications.extend([
                "Data integrity testing required",
                "Transaction testing needed",
                "Performance impact on queries"
            ])
            
        if "security" in task_lower:
            implications.extend([
                "Security test scenarios required",
                "Authentication/authorization testing",
                "Input validation and sanitization tests"
            ])
            
        return implications

    async def _generate_collaborative_recommendations(self, task: str, context: Dict[str, Any]) -> List[str]:
        """Generate recommendations for collaborative work"""
        recommendations = [
            "Establish testing checkpoints throughout development",
            "Define clear quality gates for deliverables",
            "Implement continuous testing in CI/CD pipeline"
        ]
        
        # Add context-specific recommendations
        other_agents = context.get("participating_agents", [])
        
        if "code_reviewer" in other_agents:
            recommendations.append("Coordinate with Code Reviewer on testability criteria")
            
        if "performance_analyst" in other_agents:
            recommendations.append("Align testing strategy with performance benchmarks")
            
        if "security_specialist" in other_agents:
            recommendations.append("Integrate security testing into overall test strategy")
            
        return recommendations
EOF

# Create Code Reviewer Agent
echo "📄 Creating src/agent/specialists/code_reviewer.py..."
cat > src/agent/specialists/code_reviewer.py << 'EOF'
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
EOF

# Create Performance Analyst Agent
echo "📄 Creating src/agent/specialists/performance_analyst.py..."
cat > src/agent/specialists/performance_analyst.py << 'EOF'
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
EOF

# Create comprehensive test files
echo "📄 Creating comprehensive test files for multi-agent system..."

cat > tests/unit/test_agent/test_multi_agent/__init__.py << 'EOF'
"""Tests for multi-agent system"""
EOF

cat > tests/unit/test_agent/test_specialists/__init__.py << 'EOF'
"""Tests for specialist agents"""
EOF

cat > tests/unit/test_agent/test_communication/__init__.py << 'EOF'
"""Tests for agent communication"""
EOF

cat > tests/unit/test_agent/test_specialists/test_test_architect.py << 'EOF'
"""
Test Test Architect Agent
"""

import pytest
import asyncio
from datetime import datetime

from src.agent.specialists.test_architect import TestArchitectAgent
from src.agent.communication.models import ConsultationRequest, AgentCapability


class TestTestArchitectAgent:
    
    @pytest.fixture
    async def test_architect(self):
        """Create Test Architect agent instance"""
        agent = TestArchitectAgent()
        await agent.initialize_specialist()
        return agent
    
    def test_initialization(self, test_architect):
        """Test Test Architect initialization"""
        assert test_architect.name == "test_architect"
        assert test_architect.specialization == "Test Architecture & Strategy"
        assert "test_strategy" in test_architect.expertise_domains
        assert len(test_architect.capabilities) > 3
        
        # Check for specific capabilities
        capability_names = [cap.name for cap in test_architect.capabilities]
        assert "test_strategy_design" in capability_names
        assert "test_architecture_planning" in capability_names

    @pytest.mark.asyncio
    async def test_analyze_problem_legacy_code(self, test_architect):
        """Test analysis of legacy code testing problem"""
        problem = "I have a legacy codebase with no tests and need to add comprehensive testing"
        context = {
            "codebase_size": 15000,
            "has_legacy_code": True,
            "team_experience": "intermediate"
        }
        
        result = await test_architect.analyze_problem(problem, context)
        
        assert result["analysis_type"] == "test_architecture"
        assert "problem_assessment" in result
        assert "architecture_recommendations" in result
        assert "strategy_recommendations" in result
        
        # Check problem assessment
        problem_assessment = result["problem_assessment"]
        assert problem_assessment["problem_type"] == "legacy_code_testing"
        assert "Lack of existing test coverage" in problem_assessment["key_challenges"]

    @pytest.mark.asyncio
    async def test_analyze_problem_performance_testing(self, test_architect):
        """Test analysis of performance testing problem"""
        problem = "Need to implement performance testing for our API"
        context = {
            "api_heavy": True,
            "expected_load": 1000,
            "current_testing_state": {}
        }
        
        result = await test_architect.analyze_problem(problem, context)
        
        assert result["analysis_type"] == "test_architecture"
        problem_assessment = result["problem_assessment"]
        assert problem_assessment["problem_type"] == "performance_testing"

    @pytest.mark.asyncio
    async def test_provide_consultation_strategy(self, test_architect):
        """Test consultation on testing strategy"""
        request = ConsultationRequest(
            requesting_agent="user_agent",
            specialist_agent="test_architect",
            question="What testing strategy should I use for a new microservices project?",
            context={"project_type": "microservices", "team_size": 5}
        )
        
        response = await test_architect.provide_consultation(request)
        
        assert response.responding_agent == "test_architect"
        assert response.confidence > 0.8
        assert len(response.recommendations) > 0
        assert "strategy" in response.response.lower()

    @pytest.mark.asyncio
    async def test_provide_consultation_architecture(self, test_architect):
        """Test consultation on test architecture"""
        request = ConsultationRequest(
            requesting_agent="user_agent",
            specialist_agent="test_architect",
            question="How should I organize my test architecture?",
            context={"application_type": "web_api"}
        )
        
        response = await test_architect.provide_consultation(request)
        
        assert response.responding_agent == "test_architect"
        assert response.confidence > 0.8
        assert "architecture" in response.response.lower()
        assert len(response.follow_up_questions) > 0

    @pytest.mark.asyncio
    async def test_collaborate_on_task(self, test_architect):
        """Test collaboration on a task"""
        task = "Design comprehensive testing approach for e-commerce API"
        collaboration_context = {
            "participating_agents": ["test_architect", "code_reviewer", "performance_analyst"],
            "shared_data": {"api_endpoints": 15, "user_base": 10000}
        }
        
        contribution = await test_architect.collaborate_on_task(task, collaboration_context)
        
        assert contribution["agent"] == "test_architect"
        assert contribution["specialization_applied"] == "test_architecture"
        assert "analysis" in contribution
        assert "recommendations" in contribution
        assert "test_strategy" in contribution

    def test_classify_testing_problem(self, test_architect):
        """Test problem classification"""
        # Test different problem types
        assert test_architect._classify_testing_problem("legacy code with no tests") == "legacy_code_testing"
        assert test_architect._classify_testing_problem("performance testing needed") == "performance_testing"
        assert test_architect._classify_testing_problem("security vulnerabilities") == "security_testing"
        assert test_architect._classify_testing_problem("API integration testing") == "integration_testing"

    def test_assess_complexity(self, test_architect):
        """Test complexity assessment"""
        # High complexity context
        high_complexity_context = {
            "codebase_size": 20000,
            "dependencies": ["dep1", "dep2", "dep3", "dep4", "dep5"],
            "has_legacy_code": True,
            "multiple_environments": True
        }
        assert test_architect._assess_complexity("complex problem", high_complexity_context) == "high"
        
        # Low complexity context
        low_complexity_context = {
            "codebase_size": 1000,
            "dependencies": ["dep1"],
            "has_legacy_code": False
        }
        assert test_architect._assess_complexity("simple problem", low_complexity_context) == "low"

    def test_recommend_test_layers(self, test_architect):
        """Test test layer recommendations"""
        context = {"api_heavy": True}
        layers = test_architect._recommend_test_layers(context)
        
        assert "unit_tests" in layers
        assert "integration_tests" in layers
        assert "end_to_end_tests" in layers
        assert "api_tests" in layers  # Should be added for API-heavy projects
        
        # Check priorities
        assert layers["unit_tests"]["priority"] == "high"
        assert layers["api_tests"]["priority"] == "high"

    @pytest.mark.asyncio
    async def test_capability_updates(self, test_architect):
        """Test capability update mechanism"""
        initial_count = None
        for cap in test_architect.capabilities:
            if cap.name == "test_strategy_design":
                initial_count = cap.experience_count
                break
        
        # Simulate successful capability usage
        await test_architect.update_capability("test_strategy_design", True, 2.5)
        
        for cap in test_architect.capabilities:
            if cap.name == "test_strategy_design":
                assert cap.experience_count == initial_count + 1
                assert cap.last_used is not None
                break

    def test_specialist_profile(self, test_architect):
        """Test specialist profile generation"""
        profile = test_architect.get_specialist_profile()
        
        assert profile.agent_name == "test_architect"
        assert profile.specialization == "Test Architecture & Strategy"
        assert len(profile.capabilities) > 0
        assert len(profile.expertise_domains) > 0
        assert profile.availability_status == "available"

    @pytest.mark.asyncio
    async def test_suggest_improvements(self, test_architect):
        """Test improvement suggestions"""
        # Simulate poor performance
        test_architect.performance_metrics["consultation_success_rate"] = 0.6
        test_architect.performance_metrics["response_time_avg"] = 15.0
        
        suggestions = await test_architect.suggest_improvements()
        
        assert len(suggestions) > 0
        suggestion_text = " ".join(suggestions).lower()
        assert "consultation" in suggestion_text or "response time" in suggestion_text
EOF

# Update requirements.txt
echo "📄 Updating requirements.txt..."
cat >> requirements.txt << 'EOF'

# Sprint 2.3 - Multi-Agent Architecture & Collaboration
networkx==3.2.1
matplotlib==3.8.2
plotly==5.17.0
websockets==12.0
aioredis==2.0.1
EOF

# Run verification tests
echo "🧪 Running tests to verify Sprint 2.3 implementation..."
python3 -m pytest tests/unit/test_agent/test_specialists/ -v

# Run functional verification
echo "🔍 Testing basic functionality..."
python3 -c "
import asyncio
from src.agent.specialists.test_architect import TestArchitectAgent
from src.agent.specialists.code_reviewer import CodeReviewAgent
from src.agent.specialists.performance_analyst import PerformanceAnalyst
from src.agent.communication.models import ConsultationRequest

async def test_sprint_2_3():
    print('Testing Sprint 2.3 Multi-Agent System...')
    
    # Test specialist agents
    test_architect = TestArchitectAgent()
    code_reviewer = CodeReviewAgent()
    performance_analyst = PerformanceAnalyst()
    
    # Initialize agents
    await test_architect.initialize_specialist()
    await code_reviewer.initialize_specialist()
    await performance_analyst.initialize_specialist()
    
    print(f'✅ Test Architect initialized: {test_architect.name}')
    print(f'✅ Code Reviewer initialized: {code_reviewer.name}')
    print(f'✅ Performance Analyst initialized: {performance_analyst.name}')
    
    # Test specialist capabilities
    ta_profile = test_architect.get_specialist_profile()
    print(f'✅ Test Architect capabilities: {len(ta_profile.capabilities)}')
    
    # Test consultation
    consultation_request = ConsultationRequest(
        requesting_agent='coordinator',
        specialist_agent='test_architect',
        question='What testing strategy should I use for my Python API?'
    )
    
    response = await test_architect.provide_consultation(consultation_request)
    print(f'✅ Consultation response: {response.confidence} confidence')
    
    # Test problem analysis
    problem = 'I need to improve test coverage for legacy code'
    context = {'codebase_size': 10000, 'has_legacy_code': True}
    
    analysis = await test_architect.analyze_problem(problem, context)
    print(f'✅ Problem analysis: {analysis[\"analysis_type\"]}')
    
    # Test collaboration
    task = 'Design comprehensive testing for e-commerce platform'
    collab_context = {'participating_agents': ['test_architect', 'code_reviewer']}
    
    contribution = await test_architect.collaborate_on_task(task, collab_context)
    print(f'✅ Collaboration contribution: {contribution[\"agent\"]}')
    
    print('🎉 Sprint 2.3 verification successful!')

asyncio.run(test_sprint_2_3())
"

echo "✅ Sprint 2.3: Multi-Agent Architecture & Collaboration setup complete!"
echo ""
echo "📋 Summary of Sprint 2.3 Implementation:"
echo "• Multi-Agent Communication Models with consultation and collaboration patterns"
echo "• Base Specialist Agent framework with capability tracking and learning"
echo "• Test Architect Agent with comprehensive testing strategy expertise"
echo "• Code Reviewer Agent with quality assessment and improvement recommendations"
echo "• Performance Analyst Agent with bottleneck identification and optimization"
echo "• Agent collaboration patterns with knowledge sharing and consensus building"
echo "• Comprehensive specialist capabilities with performance tracking"
echo "• Cross-agent consultation and communication protocols"
echo ""
echo "🔄 Ready for Sprint 2.4: Agent APIs & Conversational Interfaces"
echo "Run this setup script to implement Sprint 2.3, then let me know when you're ready for Sprint 2.4!"
