"""
Tool Manager
Manages tool selection, execution, and coordination for agent operations
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from .base_tool import AgentTool, ToolResult, ToolParameters, ToolDescription, ToolCapability
from ..core.models import AgentState, ReasoningStep, ReasoningType
from ...core.exceptions import AgentError


logger = logging.getLogger(__name__)


class ChainResult(BaseModel):
    """Result from chaining multiple tools"""
    success: bool
    results: List[ToolResult]
    total_execution_time_ms: int
    chain_confidence: float
    final_data: Dict[str, Any] = Field(default_factory=dict)
    recommendations: List[str] = Field(default_factory=list)


class ToolSelectionStrategy(str, Enum):
    """Strategies for tool selection"""
    BEST_MATCH = "best_match"
    FASTEST = "fastest"
    MOST_RELIABLE = "most_reliable"
    COMPREHENSIVE = "comprehensive"


class ToolManager:
    """
    Manages tool lifecycle, selection, and execution for agent operations.
    
    Provides intelligent tool selection based on task requirements,
    context awareness, and performance history.
    """

    def __init__(self):
        self.tools: Dict[str, AgentTool] = {}
        self.tool_chains: Dict[str, List[str]] = {}
        self.selection_strategy = ToolSelectionStrategy.BEST_MATCH
        self.max_concurrent_tools = 5
        self.tool_timeout_seconds = 60

    async def register_tool(self, tool: AgentTool) -> bool:
        """
        Register a tool with the manager
        
        Args:
            tool: Tool instance to register
            
        Returns:
            True if registration successful
        """
        try:
            await tool.initialize()
            self.tools[tool.name] = tool
            logger.info(f"Registered tool: {tool.name}")
            return True
        except Exception as e:
            logger.error(f"Failed to register tool {tool.name}: {str(e)}")
            return False

    async def unregister_tool(self, tool_name: str) -> bool:
        """
        Unregister a tool from the manager
        
        Args:
            tool_name: Name of tool to unregister
            
        Returns:
            True if unregistration successful
        """
        if tool_name in self.tools:
            try:
                await self.tools[tool_name].cleanup()
                del self.tools[tool_name]
                logger.info(f"Unregistered tool: {tool_name}")
                return True
            except Exception as e:
                logger.error(f"Failed to unregister tool {tool_name}: {str(e)}")
                return False
        return False

    async def select_tools(
        self,
        task: str,
        context: AgentState,
        capabilities: Optional[List[ToolCapability]] = None,
        strategy: Optional[ToolSelectionStrategy] = None
    ) -> List[AgentTool]:
        """
        Intelligently select tools for a given task
        
        Args:
            task: Task description
            context: Agent state and context
            capabilities: Required capabilities
            strategy: Selection strategy to use
            
        Returns:
            List of selected tools in execution order
        """
        if not self.tools:
            logger.warning("No tools available for selection")
            return []

        strategy = strategy or self.selection_strategy
        
        # Get tool confidence scores for the task
        tool_scores = []
        for tool_name, tool in self.tools.items():
            try:
                confidence = tool.can_handle(task, self._extract_context_dict(context))
                if confidence > 0:
                    tool_scores.append((tool, confidence))
            except Exception as e:
                logger.warning(f"Error evaluating tool {tool_name}: {str(e)}")

        if not tool_scores:
            logger.warning(f"No tools can handle task: {task}")
            return []

        # Apply selection strategy
        if strategy == ToolSelectionStrategy.BEST_MATCH:
            # Sort by confidence score
            tool_scores.sort(key=lambda x: x[1], reverse=True)
        elif strategy == ToolSelectionStrategy.FASTEST:
            # Sort by average execution time
            tool_scores.sort(key=lambda x: x[0].avg_execution_time_ms)
        elif strategy == ToolSelectionStrategy.MOST_RELIABLE:
            # Sort by success rate
            tool_scores.sort(key=lambda x: x[0].success_rate, reverse=True)
        elif strategy == ToolSelectionStrategy.COMPREHENSIVE:
            # Complex scoring considering multiple factors
            def comprehensive_score(tool_confidence_pair):
                tool, confidence = tool_confidence_pair
                # Weight factors: confidence (50%), reliability (30%), speed (20%)
                speed_score = 1.0 - min(1.0, tool.avg_execution_time_ms / 10000)
                return (confidence * 0.5 + tool.success_rate * 0.3 + speed_score * 0.2)
            
            tool_scores.sort(key=comprehensive_score, reverse=True)

        # Filter by required capabilities if specified
        if capabilities:
            filtered_tools = []
            for tool, confidence in tool_scores:
                tool_capabilities = tool.capabilities
                if any(cap in tool_capabilities for cap in capabilities):
                    filtered_tools.append((tool, confidence))
            tool_scores = filtered_tools

        # Return top tools (limit based on task complexity)
        max_tools = self._determine_max_tools(task, context)
        selected_tools = [tool for tool, _ in tool_scores[:max_tools]]
        
        logger.info(f"Selected {len(selected_tools)} tools for task: {task}")
        return selected_tools

    async def execute_tool(
        self,
        tool: AgentTool,
        parameters: ToolParameters
    ) -> ToolResult:
        """
        Execute a single tool with error handling and performance tracking
        
        Args:
            tool: Tool to execute
            parameters: Execution parameters
            
        Returns:
            ToolResult with execution outcome
        """
        start_time = datetime.utcnow()
        
        try:
            # Execute tool with timeout
            result = await asyncio.wait_for(
                tool.execute(parameters),
                timeout=self.tool_timeout_seconds
            )
            
            execution_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            result.execution_time_ms = execution_time
            
            # Update tool performance metrics
            tool.update_performance_metrics(execution_time, result.success)
            
            logger.info(f"Tool {tool.name} executed successfully in {execution_time}ms")
            return result

        except asyncio.TimeoutError:
            execution_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            tool.update_performance_metrics(execution_time, False)
            
            error_result = ToolResult(
                tool_name=tool.name,
                success=False,
                error_message=f"Tool execution timed out after {self.tool_timeout_seconds}s",
                execution_time_ms=execution_time
            )
            
            logger.error(f"Tool {tool.name} timed out after {self.tool_timeout_seconds}s")
            return error_result

        except Exception as e:
            execution_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            tool.update_performance_metrics(execution_time, False)
            
            error_result = ToolResult(
                tool_name=tool.name,
                success=False,
                error_message=str(e),
                execution_time_ms=execution_time
            )
            
            logger.error(f"Tool {tool.name} execution failed: {str(e)}")
            return error_result

    async def execute_tool_chain(
        self,
        tools: List[AgentTool],
        parameters: ToolParameters,
        chain_strategy: str = "sequential"
    ) -> ChainResult:
        """
        Execute multiple tools in a coordinated chain
        
        Args:
            tools: List of tools to execute
            parameters: Base parameters for execution
            chain_strategy: How to chain tools ("sequential", "parallel", "conditional")
            
        Returns:
            ChainResult with combined results
        """
        start_time = datetime.utcnow()
        results = []
        accumulated_data = {}
        
        try:
            if chain_strategy == "sequential":
                # Execute tools one after another, passing data forward
                for i, tool in enumerate(tools):
                    # Update parameters with accumulated data
                    chain_params = parameters.model_copy()
                    chain_params.context.update(accumulated_data)
                    chain_params.context["chain_position"] = i
                    chain_params.context["previous_results"] = results
                    
                    result = await self.execute_tool(tool, chain_params)
                    results.append(result)
                    
                    if result.success:
                        accumulated_data.update(result.data)
                    else:
                        logger.warning(f"Tool {tool.name} failed in chain, continuing...")

            elif chain_strategy == "parallel":
                # Execute tools concurrently
                tasks = [
                    self.execute_tool(tool, parameters) 
                    for tool in tools
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Convert exceptions to error results
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        results[i] = ToolResult(
                            tool_name=tools[i].name,
                            success=False,
                            error_message=str(result)
                        )

            total_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            
            # Calculate overall chain confidence
            successful_results = [r for r in results if r.success]
            if successful_results:
                chain_confidence = sum(r.confidence for r in successful_results) / len(successful_results)
            else:
                chain_confidence = 0.0

            # Generate recommendations based on all results
            all_recommendations = []
            for result in results:
                all_recommendations.extend(result.recommendations)

            return ChainResult(
                success=len(successful_results) > 0,
                results=results,
                total_execution_time_ms=total_time,
                chain_confidence=chain_confidence,
                final_data=accumulated_data,
                recommendations=list(set(all_recommendations))  # Remove duplicates
            )

        except Exception as e:
            total_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            logger.error(f"Tool chain execution failed: {str(e)}")
            
            return ChainResult(
                success=False,
                results=results,
                total_execution_time_ms=total_time,
                chain_confidence=0.0,
                final_data=accumulated_data
            )

    def _extract_context_dict(self, agent_state: AgentState) -> Dict[str, Any]:
        """
        Extract relevant context from agent state for tool evaluation
        """
        return {
            "session_id": agent_state.session_id,
            "current_goal": agent_state.current_goal.description if agent_state.current_goal else None,
            "conversation_length": len(agent_state.conversation_context),
            "user_preferences": agent_state.user_preferences,
            "recent_tools": agent_state.active_tools,
            "reasoning_history_length": len(agent_state.reasoning_history)
        }

    def _determine_max_tools(self, task: str, context: AgentState) -> int:
        """
        Determine maximum number of tools to use based on task complexity
        """
        base_count = 1
        
        # Increase based on task complexity indicators
        complexity_keywords = ["comprehensive", "detailed", "thorough", "complete", "analyze"]
        if any(keyword in task.lower() for keyword in complexity_keywords):
            base_count += 1
        
        # Increase based on conversation context
        if len(context.conversation_context) > 5:
            base_count += 1
        
        # Increase if user has expressed preferences for detailed responses
        if context.user_preferences.get("communication_style") == "detailed":
            base_count += 1
        
        return min(base_count, self.max_concurrent_tools)

    async def get_available_tools(self) -> List[ToolDescription]:
        """
        Get descriptions of all available tools
        
        Returns:
            List of tool descriptions
        """
        descriptions = []
        for tool in self.tools.values():
            descriptions.append(tool.get_description())
        return descriptions

    async def get_tool_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics for all tools
        
        Returns:
            Dictionary with tool performance data
        """
        stats = {
            "total_tools": len(self.tools),
            "tools": {},
            "overall_stats": {
                "total_usage": 0,
                "avg_success_rate": 0.0,
                "avg_execution_time_ms": 0.0
            }
        }
        
        total_usage = 0
        total_success_rate = 0.0
        total_avg_time = 0.0
        
        for tool in self.tools.values():
            tool_stats = tool.get_performance_stats()
            stats["tools"][tool.name] = tool_stats
            
            total_usage += tool_stats["usage_count"]
            total_success_rate += tool_stats["success_rate"]
            total_avg_time += tool_stats["avg_execution_time_ms"]
        
        if self.tools:
            stats["overall_stats"]["total_usage"] = total_usage
            stats["overall_stats"]["avg_success_rate"] = total_success_rate / len(self.tools)
            stats["overall_stats"]["avg_execution_time_ms"] = total_avg_time / len(self.tools)
        
        return stats

    async def suggest_tool_improvements(self) -> List[str]:
        """
        Suggest improvements for tool performance
        
        Returns:
            List of improvement suggestions
        """
        suggestions = []
        
        for tool in self.tools.values():
            stats = tool.get_performance_stats()
            
            if stats["success_rate"] < 0.8:
                suggestions.append(f"Tool '{tool.name}' has low success rate ({stats['success_rate']:.2f})")
            
            if stats["avg_execution_time_ms"] > 10000:
                suggestions.append(f"Tool '{tool.name}' is slow (avg {stats['avg_execution_time_ms']}ms)")
            
            if stats["usage_count"] == 0:
                suggestions.append(f"Tool '{tool.name}' is unused - consider removal or improvement")
        
        return suggestions
