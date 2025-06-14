"""
Test Tool Manager
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from src.agent.tools.tool_manager import ToolManager, ChainResult, ToolSelectionStrategy
from src.agent.tools.base_tool import AgentTool, ToolResult, ToolParameters, ToolType, ToolCapability
from src.agent.core.models import AgentState


class MockAnalysisTool(AgentTool):
    """Mock analysis tool for testing"""
    
    def __init__(self):
        super().__init__("mock_analyzer", "Mock analysis tool", ToolType.ANALYSIS)
        self.capabilities = [ToolCapability.CODE_ANALYSIS]
        
    async def execute(self, parameters: ToolParameters) -> ToolResult:
        await asyncio.sleep(0.01)
        return ToolResult(
            tool_name=self.name,
            success=True,
            data={"analysis": "complete"},
            confidence=0.9
        )
    
    def can_handle(self, task: str, context=None) -> float:
        return 0.9 if "analyze" in task.lower() else 0.2


class MockGenerationTool(AgentTool):
    """Mock generation tool for testing"""
    
    def __init__(self):
        super().__init__("mock_generator", "Mock generation tool", ToolType.GENERATION)
        self.capabilities = [ToolCapability.TEST_GENERATION]
        
    async def execute(self, parameters: ToolParameters) -> ToolResult:
        await asyncio.sleep(0.02)
        return ToolResult(
            tool_name=self.name,
            success=True,
            data={"generated": "tests"},
            confidence=0.8
        )
    
    def can_handle(self, task: str, context=None) -> float:
        return 0.8 if "generate" in task.lower() else 0.1


class MockFailingTool(AgentTool):
    """Mock tool that fails for testing error handling"""
    
    def __init__(self):
        super().__init__("mock_failing", "Mock failing tool", ToolType.UTILITY)
        
    async def execute(self, parameters: ToolParameters) -> ToolResult:
        raise Exception("Tool execution failed")
    
    def can_handle(self, task: str, context=None) -> float:
        return 0.5


class TestToolManager:
    
    @pytest.fixture
    def tool_manager(self):
        """Create tool manager instance"""
        return ToolManager()
    
    @pytest.fixture
    def agent_state(self):
        """Create test agent state"""
        return AgentState(session_id="test_session")
    
    @pytest.fixture
    def tool_parameters(self):
        """Create test tool parameters"""
        return ToolParameters(
            parameters={"test_param": "value"},
            session_id="test_session"
        )

    @pytest.mark.asyncio
    async def test_register_tool(self, tool_manager):
        """Test tool registration"""
        tool = MockAnalysisTool()
        
        result = await tool_manager.register_tool(tool)
        
        assert result is True
        assert "mock_analyzer" in tool_manager.tools
        assert tool_manager.tools["mock_analyzer"] == tool
        assert tool.is_initialized is True

    @pytest.mark.asyncio
    async def test_unregister_tool(self, tool_manager):
        """Test tool unregistration"""
        tool = MockAnalysisTool()
        await tool_manager.register_tool(tool)
        
        result = await tool_manager.unregister_tool("mock_analyzer")
        
        assert result is True
        assert "mock_analyzer" not in tool_manager.tools

    @pytest.mark.asyncio
    async def test_unregister_nonexistent_tool(self, tool_manager):
        """Test unregistering non-existent tool"""
        result = await tool_manager.unregister_tool("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_select_tools_best_match(self, tool_manager, agent_state):
        """Test tool selection with best match strategy"""
        analysis_tool = MockAnalysisTool()
        generation_tool = MockGenerationTool()
        
        await tool_manager.register_tool(analysis_tool)
        await tool_manager.register_tool(generation_tool)
        
        # Task that should prefer analysis tool
        selected = await tool_manager.select_tools(
            "analyze my code",
            agent_state,
            strategy=ToolSelectionStrategy.BEST_MATCH
        )
        
        assert len(selected) > 0
        assert selected[0] == analysis_tool  # Should be first due to higher confidence

    @pytest.mark.asyncio
    async def test_select_tools_no_tools(self, tool_manager, agent_state):
        """Test tool selection with no registered tools"""
        selected = await tool_manager.select_tools("any task", agent_state)
        assert selected == []

    @pytest.mark.asyncio
    async def test_select_tools_no_capable_tools(self, tool_manager, agent_state):
        """Test tool selection when no tools can handle the task"""
        tool = MockAnalysisTool()
        # Override can_handle to return 0 confidence
        tool.can_handle = lambda task, context=None: 0.0
        
        await tool_manager.register_tool(tool)
        
        selected = await tool_manager.select_tools("unhandleable task", agent_state)
        assert selected == []

    @pytest.mark.asyncio
    async def test_execute_tool_success(self, tool_manager, tool_parameters):
        """Test successful tool execution"""
        tool = MockAnalysisTool()
        await tool.initialize()
        
        result = await tool_manager.execute_tool(tool, tool_parameters)
        
        assert isinstance(result, ToolResult)
        assert result.success is True
        assert result.tool_name == "mock_analyzer"
        assert result.execution_time_ms is not None
        assert result.execution_time_ms > 0

    @pytest.mark.asyncio
    async def test_execute_tool_failure(self, tool_manager, tool_parameters):
        """Test tool execution failure"""
        tool = MockFailingTool()
        await tool.initialize()
        
        result = await tool_manager.execute_tool(tool, tool_parameters)
        
        assert isinstance(result, ToolResult)
        assert result.success is False
        assert result.error_message is not None
        assert "Tool execution failed" in result.error_message

    @pytest.mark.asyncio
    async def test_execute_tool_timeout(self, tool_manager, tool_parameters):
        """Test tool execution timeout"""
        tool = MockAnalysisTool()
        
        # Mock execute to take longer than timeout
        async def slow_execute(params):
            await asyncio.sleep(2)  # Longer than default timeout
            return ToolResult(tool_name=tool.name, success=True)
        
        tool.execute = slow_execute
        await tool.initialize()
        
        # Set short timeout for testing
        tool_manager.tool_timeout_seconds = 0.1
        
        result = await tool_manager.execute_tool(tool, tool_parameters)
        
        assert result.success is False
        assert "timed out" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_execute_tool_chain_sequential(self, tool_manager, tool_parameters):
        """Test sequential tool chain execution"""
        analysis_tool = MockAnalysisTool()
        generation_tool = MockGenerationTool()
        
        await analysis_tool.initialize()
        await generation_tool.initialize()
        
        tools = [analysis_tool, generation_tool]
        
        result = await tool_manager.execute_tool_chain(
            tools, tool_parameters, "sequential"
        )
        
        assert isinstance(result, ChainResult)
        assert result.success is True
        assert len(result.results) == 2
        assert result.total_execution_time_ms > 0
        assert result.chain_confidence > 0

    @pytest.mark.asyncio
    async def test_execute_tool_chain_parallel(self, tool_manager, tool_parameters):
        """Test parallel tool chain execution"""
        analysis_tool = MockAnalysisTool()
        generation_tool = MockGenerationTool()
        
        await analysis_tool.initialize()
        await generation_tool.initialize()
        
        tools = [analysis_tool, generation_tool]
        
        result = await tool_manager.execute_tool_chain(
            tools, tool_parameters, "parallel"
        )
        
        assert isinstance(result, ChainResult)
        assert result.success is True
        assert len(result.results) == 2

    @pytest.mark.asyncio
    async def test_execute_tool_chain_with_failure(self, tool_manager, tool_parameters):
        """Test tool chain execution with one tool failing"""
        analysis_tool = MockAnalysisTool()
        failing_tool = MockFailingTool()
        
        await analysis_tool.initialize()
        await failing_tool.initialize()
        
        tools = [analysis_tool, failing_tool]
        
        result = await tool_manager.execute_tool_chain(
            tools, tool_parameters, "sequential"
        )
        
        assert isinstance(result, ChainResult)
        # Should still be successful because one tool succeeded
        assert result.success is True
        assert len(result.results) == 2
        assert result.results[0].success is True
        assert result.results[1].success is False

    @pytest.mark.asyncio
    async def test_get_available_tools(self, tool_manager):
        """Test getting available tools"""
        tool1 = MockAnalysisTool()
        tool2 = MockGenerationTool()
        
        await tool_manager.register_tool(tool1)
        await tool_manager.register_tool(tool2)
        
        descriptions = await tool_manager.get_available_tools()
        
        assert len(descriptions) == 2
        tool_names = [desc.name for desc in descriptions]
        assert "mock_analyzer" in tool_names
        assert "mock_generator" in tool_names

    @pytest.mark.asyncio
    async def test_get_tool_performance_stats(self, tool_manager, tool_parameters):
        """Test getting tool performance statistics"""
        tool = MockAnalysisTool()
        await tool_manager.register_tool(tool)
        
        # Execute tool to generate stats
        await tool_manager.execute_tool(tool, tool_parameters)
        
        stats = await tool_manager.get_tool_performance_stats()
        
        assert stats["total_tools"] == 1
        assert "mock_analyzer" in stats["tools"]
        assert stats["tools"]["mock_analyzer"]["usage_count"] == 1
        assert stats["overall_stats"]["total_usage"] == 1

    @pytest.mark.asyncio
    async def test_suggest_tool_improvements(self, tool_manager):
        """Test tool improvement suggestions"""
        tool = MockAnalysisTool()
        
        # Simulate poor performance
        tool.success_rate = 0.5
        tool.avg_execution_time_ms = 15000
        tool.usage_count = 0
        
        await tool_manager.register_tool(tool)
        
        suggestions = await tool_manager.suggest_tool_improvements()
        
        assert len(suggestions) > 0
        suggestion_text = " ".join(suggestions).lower()
        assert "success rate" in suggestion_text or "slow" in suggestion_text or "unused" in suggestion_text

    def test_extract_context_dict(self, tool_manager, agent_state):
        """Test context extraction from agent state"""
        agent_state.user_preferences = {"expertise_level": "expert"}
        agent_state.active_tools = ["tool1", "tool2"]
        
        context = tool_manager._extract_context_dict(agent_state)
        
        assert context["session_id"] == "test_session"
        assert context["user_preferences"] == {"expertise_level": "expert"}
        assert context["recent_tools"] == ["tool1", "tool2"]
        assert context["conversation_length"] == 0

    def test_determine_max_tools(self, tool_manager, agent_state):
        """Test determining maximum tools for a task"""
        # Simple task
        max_tools = tool_manager._determine_max_tools("simple task", agent_state)
        assert max_tools >= 1
        
        # Complex task
        max_tools_complex = tool_manager._determine_max_tools(
            "comprehensive detailed analysis", agent_state
        )
        assert max_tools_complex > max_tools

    @pytest.mark.asyncio
    async def test_tool_selection_strategies(self, tool_manager, agent_state):
        """Test different tool selection strategies"""
        # Create tools with different performance characteristics
        fast_tool = MockAnalysisTool()
        fast_tool.avg_execution_time_ms = 100
        fast_tool.success_rate = 0.7
        
        reliable_tool = MockGenerationTool()
        reliable_tool.avg_execution_time_ms = 1000
        reliable_tool.success_rate = 0.95
        
        await tool_manager.register_tool(fast_tool)
        await tool_manager.register_tool(reliable_tool)
        
        # Test fastest strategy
        fastest = await tool_manager.select_tools(
            "analyze and generate", agent_state, strategy=ToolSelectionStrategy.FASTEST
        )
        
        # Test most reliable strategy
        reliable = await tool_manager.select_tools(
            "analyze and generate", agent_state, strategy=ToolSelectionStrategy.MOST_RELIABLE
        )
        
        # Should select different tools based on strategy
        assert len(fastest) > 0
        assert len(reliable) > 0
