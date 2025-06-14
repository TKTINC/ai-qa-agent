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
