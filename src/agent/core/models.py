"""
Agent Core Models
Defines data structures for agent state, reasoning, and communication
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field
import uuid


class TaskStatus(str, Enum):
    """Task execution status"""
    PENDING = "pending"
    PLANNING = "planning"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ReasoningType(str, Enum):
    """Types of reasoning steps"""
    OBSERVATION = "observation"
    THOUGHT = "thought"
    PLAN = "plan"
    ACTION = "action"
    REFLECTION = "reflection"


class Message(BaseModel):
    """Conversation message"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    role: str  # user, assistant, system
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Goal(BaseModel):
    """User goal or objective"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    description: str
    priority: int = Field(default=1, ge=1, le=5)
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.utcnow)
    target_completion: Optional[datetime] = None
    context: Dict[str, Any] = Field(default_factory=dict)
    sub_goals: List['Goal'] = Field(default_factory=list)


class ReasoningStep(BaseModel):
    """A single step in the reasoning process"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: ReasoningType
    content: str
    confidence: float = Field(ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    context: Dict[str, Any] = Field(default_factory=dict)
    tools_used: List[str] = Field(default_factory=list)
    duration_ms: Optional[int] = None


class AgentState(BaseModel):
    """Complete agent state"""
    session_id: str
    current_goal: Optional[Goal] = None
    conversation_context: List[Message] = Field(default_factory=list)
    reasoning_history: List[ReasoningStep] = Field(default_factory=list)
    active_tools: List[str] = Field(default_factory=list)
    user_preferences: Dict[str, Any] = Field(default_factory=dict)
    session_memory: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_updated: datetime = Field(default_factory=datetime.utcnow)

    def add_message(self, message: Message) -> None:
        """Add message to conversation context"""
        self.conversation_context.append(message)
        self.last_updated = datetime.utcnow()

    def add_reasoning_step(self, step: ReasoningStep) -> None:
        """Add reasoning step to history"""
        self.reasoning_history.append(step)
        self.last_updated = datetime.utcnow()

    def get_recent_context(self, limit: int = 10) -> List[Message]:
        """Get recent conversation context"""
        return self.conversation_context[-limit:]


class UserProfile(BaseModel):
    """User preference profile"""
    user_id: str
    expertise_level: str = "intermediate"  # beginner, intermediate, expert
    communication_style: str = "balanced"  # direct, detailed, educational, balanced
    preferred_tools: List[str] = Field(default_factory=list)
    domain_knowledge: Dict[str, float] = Field(default_factory=dict)
    learning_goals: List[str] = Field(default_factory=list)
    successful_patterns: List[Dict[str, Any]] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_updated: datetime = Field(default_factory=datetime.utcnow)


class AgentResponse(BaseModel):
    """Agent response to user input"""
    content: str
    reasoning_steps: List[ReasoningStep] = Field(default_factory=list)
    tools_used: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.8)
    suggestions: List[str] = Field(default_factory=list)
    follow_up_questions: List[str] = Field(default_factory=list)
    session_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TaskPlan(BaseModel):
    """Planned approach for a task"""
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    goal: Goal
    steps: List[Dict[str, Any]] = Field(default_factory=list)
    estimated_duration: Optional[int] = None  # minutes
    required_tools: List[str] = Field(default_factory=list)
    success_criteria: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    status: TaskStatus = TaskStatus.PENDING


# Update Goal model to handle forward references
Goal.model_rebuild()
