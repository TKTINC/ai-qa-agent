"""
Base Tool Interface
Defines the interface that all agent tools must implement
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field
import uuid


class ToolType(str, Enum):
    """Types of tools available to agents"""
    ANALYSIS = "analysis"
    GENERATION = "generation"
    EXECUTION = "execution"
    VALIDATION = "validation"
    COMMUNICATION = "communication"
    UTILITY = "utility"


class ToolCapability(str, Enum):
    """Specific capabilities that tools can provide"""
    CODE_ANALYSIS = "code_analysis"
    TEST_GENERATION = "test_generation"
    CODE_EXECUTION = "code_execution"
    QUALITY_ASSESSMENT = "quality_assessment"
    PATTERN_DETECTION = "pattern_detection"
    DOCUMENTATION = "documentation"
    FILE_MANAGEMENT = "file_management"
    API_INTEGRATION = "api_integration"


class ToolResult(BaseModel):
    """Result from tool execution"""
    tool_name: str
    success: bool
    data: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    execution_time_ms: Optional[int] = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    error_message: Optional[str] = None
    recommendations: List[str] = Field(default_factory=list)
    next_steps: List[str] = Field(default_factory=list)


class ToolParameters(BaseModel):
    """Parameters for tool execution"""
    parameters: Dict[str, Any] = Field(default_factory=dict)
    context: Dict[str, Any] = Field(default_factory=dict)
    options: Dict[str, Any] = Field(default_factory=dict)
    session_id: str
    agent_context: Optional[Dict[str, Any]] = Field(default_factory=dict)


class ToolDescription(BaseModel):
    """Tool description and capabilities"""
    name: str
    description: str
    tool_type: ToolType
    capabilities: List[ToolCapability]
    required_parameters: List[str] = Field(default_factory=list)
    optional_parameters: List[str] = Field(default_factory=list)
    supported_languages: List[str] = Field(default_factory=list)
    estimated_duration_ms: Optional[int] = None
    complexity_level: int = Field(default=1, ge=1, le=5)
    dependencies: List[str] = Field(default_factory=list)


class AgentTool(ABC):
    """
    Abstract base class for all agent tools.
    
    All tools must implement this interface to be used by the agent system.
    """

    def __init__(self, name: str, description: str, tool_type: ToolType):
        self.name = name
        self.description = description
        self.tool_type = tool_type
        self.capabilities: List[ToolCapability] = []
        self.supported_languages: List[str] = []
        self.dependencies: List[str] = []
        
        # Tool state
        self.is_initialized = False
        self.last_used = None
        self.usage_count = 0
        self.success_rate = 1.0
        
        # Performance tracking
        self.avg_execution_time_ms = 0
        self.total_execution_time_ms = 0

    @abstractmethod
    async def execute(self, parameters: ToolParameters) -> ToolResult:
        """
        Execute the tool with given parameters
        
        Args:
            parameters: Tool execution parameters
            
        Returns:
            ToolResult with execution outcome
        """
        pass

    @abstractmethod
    def can_handle(self, task: str, context: Dict[str, Any] = None) -> float:
        """
        Determine if this tool can handle a given task
        
        Args:
            task: Task description
            context: Additional context about the task
            
        Returns:
            Confidence score (0.0 to 1.0) that this tool can handle the task
        """
        pass

    async def initialize(self) -> bool:
        """
        Initialize the tool (override if needed)
        
        Returns:
            True if initialization successful
        """
        self.is_initialized = True
        return True

    async def cleanup(self) -> None:
        """
        Cleanup tool resources (override if needed)
        """
        pass

    def get_description(self) -> ToolDescription:
        """
        Get detailed tool description
        
        Returns:
            ToolDescription with tool capabilities and requirements
        """
        return ToolDescription(
            name=self.name,
            description=self.description,
            tool_type=self.tool_type,
            capabilities=self.capabilities,
            supported_languages=self.supported_languages,
            estimated_duration_ms=self.avg_execution_time_ms or 1000,
            dependencies=self.dependencies
        )

    def update_performance_metrics(self, execution_time_ms: int, success: bool) -> None:
        """
        Update tool performance metrics
        
        Args:
            execution_time_ms: Time taken for execution
            success: Whether execution was successful
        """
        self.usage_count += 1
        self.last_used = datetime.utcnow()
        self.total_execution_time_ms += execution_time_ms
        self.avg_execution_time_ms = self.total_execution_time_ms // self.usage_count
        
        # Update success rate with weighted average
        if self.usage_count == 1:
            self.success_rate = 1.0 if success else 0.0
        else:
            weight = 0.1  # Weight for new measurements
            new_success = 1.0 if success else 0.0
            self.success_rate = (1 - weight) * self.success_rate + weight * new_success

    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get tool performance statistics
        
        Returns:
            Dictionary with performance metrics
        """
        return {
            "name": self.name,
            "usage_count": self.usage_count,
            "success_rate": self.success_rate,
            "avg_execution_time_ms": self.avg_execution_time_ms,
            "last_used": self.last_used,
            "is_initialized": self.is_initialized
        }
