"""
Test Base Tool Interface
"""

import pytest
import asyncio
from unittest.mock import Mock, patch
from datetime import datetime

from src.agent.tools.base_tool import (
    AgentTool, ToolResult, ToolParameters, ToolDescription, 
    ToolType, ToolCapability
)


class MockTool(AgentTool):
    """Mock tool for testing"""
    
    def __init__(self):
        super().__init__("mock_tool", "Test tool", ToolType.ANALYSIS)
        self.capabilities = [ToolCapability.CODE_ANALYSIS]
        
    async def execute(self, parameters: ToolParameters) -> ToolResult:
        await asyncio.sleep(0.01)  # Simulate work
        return ToolResult(
            tool_name=self.name,
            success=True,
            data={"result": "test"},
            confidence=0.9
        )
    
    def can_handle(self, task: str, context=None) -> float:
        return 0.8 if "test" in task.lower() else 0.1


class TestToolResult:
    def test_tool_result_creation(self):
        """Test tool result creation"""
        result = ToolResult(
            tool_name="test_tool",
            success=True,
            data={"key": "value"},
            confidence=0.95
        )
        
        assert result.tool_name == "test_tool"
        assert result.success is True
        assert result.data == {"key": "value"}
        assert result.confidence == 0.95
        assert result.metadata == {}
        assert result.error_message is None
        assert result.recommendations == []

    def test_tool_result_with_error(self):
        """Test tool result with error"""
        result = ToolResult(
            tool_name="failed_tool",
            success=False,
            error_message="Something went wrong",
            confidence=0.0
        )
        
        assert result.success is False
        assert result.error_message == "Something went wrong"
        assert result.confidence == 0.0


class TestToolParameters:
    def test_tool_parameters_creation(self):
        """Test tool parameters creation"""
        params = ToolParameters(
            parameters={"param1": "value1"},
            context={"context_key": "context_value"},
            session_id="test_session"
        )
        
        assert params.parameters == {"param1": "value1"}
        assert params.context == {"context_key": "context_value"}
        assert params.session_id == "test_session"
        assert params.options == {}
        assert params.agent_context == {}


class TestAgentTool:
    
    @pytest.fixture
    def mock_tool(self):
        """Create mock tool instance"""
        return MockTool()
    
    def test_tool_initialization(self, mock_tool):
        """Test tool initialization"""
        assert mock_tool.name == "mock_tool"
        assert mock_tool.description == "Test tool"
        assert mock_tool.tool_type == ToolType.ANALYSIS
        assert mock_tool.capabilities == [ToolCapability.CODE_ANALYSIS]
        assert mock_tool.is_initialized is False
        assert mock_tool.usage_count == 0
        assert mock_tool.success_rate == 1.0

    @pytest.mark.asyncio
    async def test_tool_initialization_method(self, mock_tool):
        """Test tool initialize method"""
        result = await mock_tool.initialize()
        
        assert result is True
        assert mock_tool.is_initialized is True

    @pytest.mark.asyncio 
    async def test_tool_execution(self, mock_tool):
        """Test tool execution"""
        parameters = ToolParameters(
            parameters={"test": "value"},
            session_id="test_session"
        )
        
        result = await mock_tool.execute(parameters)
        
        assert isinstance(result, ToolResult)
        assert result.tool_name == "mock_tool"
        assert result.success is True
        assert result.confidence == 0.9
        assert result.data == {"result": "test"}

    def test_can_handle_method(self, mock_tool):
        """Test can_handle method"""
        # Should handle test-related tasks well
        confidence = mock_tool.can_handle("run test analysis")
        assert confidence == 0.8
        
        # Should not handle unrelated tasks well
        confidence = mock_tool.can_handle("generate documentation")
        assert confidence == 0.1

    def test_get_description(self, mock_tool):
        """Test get_description method"""
        description = mock_tool.get_description()
        
        assert isinstance(description, ToolDescription)
        assert description.name == "mock_tool"
        assert description.description == "Test tool"
        assert description.tool_type == ToolType.ANALYSIS
        assert description.capabilities == [ToolCapability.CODE_ANALYSIS]

    def test_performance_metrics_update(self, mock_tool):
        """Test performance metrics update"""
        initial_usage = mock_tool.usage_count
        initial_success_rate = mock_tool.success_rate
        
        # Update with successful execution
        mock_tool.update_performance_metrics(1000, True)
        
        assert mock_tool.usage_count == initial_usage + 1
        assert mock_tool.last_used is not None
        assert mock_tool.avg_execution_time_ms == 1000
        assert mock_tool.success_rate >= initial_success_rate

    def test_performance_stats(self, mock_tool):
        """Test performance stats retrieval"""
        mock_tool.update_performance_metrics(500, True)
        
        stats = mock_tool.get_performance_stats()
        
        assert stats["name"] == "mock_tool"
        assert stats["usage_count"] == 1
        assert stats["avg_execution_time_ms"] == 500
        assert stats["success_rate"] == 1.0
        assert stats["is_initialized"] is False  # Not initialized yet

    @pytest.mark.asyncio
    async def test_cleanup_method(self, mock_tool):
        """Test cleanup method"""
        # Should not raise any exceptions
        await mock_tool.cleanup()


class TestToolDescription:
    def test_tool_description_creation(self):
        """Test tool description creation"""
        description = ToolDescription(
            name="test_tool",
            description="A test tool",
            tool_type=ToolType.GENERATION,
            capabilities=[ToolCapability.TEST_GENERATION, ToolCapability.CODE_ANALYSIS],
            required_parameters=["input_file"],
            optional_parameters=["format", "output_dir"],
            supported_languages=["python", "javascript"],
            estimated_duration_ms=2000,
            complexity_level=3
        )
        
        assert description.name == "test_tool"
        assert description.description == "A test tool"
        assert description.tool_type == ToolType.GENERATION
        assert len(description.capabilities) == 2
        assert ToolCapability.TEST_GENERATION in description.capabilities
        assert description.required_parameters == ["input_file"]
        assert description.optional_parameters == ["format", "output_dir"]
        assert description.supported_languages == ["python", "javascript"]
        assert description.estimated_duration_ms == 2000
        assert description.complexity_level == 3
        assert description.dependencies == []

    def test_tool_description_defaults(self):
        """Test tool description with defaults"""
        description = ToolDescription(
            name="minimal_tool",
            description="Minimal tool",
            tool_type=ToolType.UTILITY,
            capabilities=[]
        )
        
        assert description.required_parameters == []
        assert description.optional_parameters == []
        assert description.supported_languages == []
        assert description.estimated_duration_ms is None
        assert description.complexity_level == 1
        assert description.dependencies == []
