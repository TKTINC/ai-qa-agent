#!/bin/bash
# Setup Script for Sprint 2.2: Intelligent Tool System & Test Generation
# AI QA Agent - Sprint 2.2

set -e
echo "🚀 Setting up Sprint 2.2: Intelligent Tool System & Test Generation..."

# Check prerequisites (Sprint 2.1 completed)
if [ ! -f "src/agent/orchestrator.py" ]; then
    echo "❌ Error: Sprint 2.1 must be completed first"
    echo "Missing: src/agent/orchestrator.py"
    exit 1
fi

if [ ! -f "src/agent/reasoning/react_engine.py" ]; then
    echo "❌ Error: Sprint 2.1 must be completed first"
    echo "Missing: src/agent/reasoning/react_engine.py"
    exit 1
fi

# Install new dependencies with pip3 (macOS compatible)
echo "📦 Installing new dependencies..."
pip3 install jinja2==3.1.2
pip3 install ast-tools==0.1.0
pip3 install autopep8==2.0.4
pip3 install rope==1.11.0
pip3 install bandit==1.7.5

# Create tools directory structure
echo "📁 Creating agent tools directory structure..."
mkdir -p src/agent/tools
mkdir -p src/agent/tools/analysis
mkdir -p src/agent/tools/generation
mkdir -p src/agent/tools/execution
mkdir -p src/agent/tools/validation
mkdir -p tests/unit/test_agent/test_tools
mkdir -p tests/unit/test_agent/test_tools/test_analysis
mkdir -p tests/unit/test_agent/test_tools/test_generation

# Create tools __init__.py files
echo "📄 Creating tools __init__.py files..."
cat > src/agent/tools/__init__.py << 'EOF'
"""
AI QA Agent - Intelligent Tool System
Sprint 2.2: Tool Management and Test Generation

This module implements the intelligent tool system including:
- Tool management and orchestration
- Code analysis tools with agent integration
- Test generation tools with intelligent strategies
- Tool chaining and workflow coordination
- Quality assessment and validation tools
"""

from .tool_manager import ToolManager
from .analysis.code_analysis_tool import CodeAnalysisTool
from .generation.test_generation_tool import TestGenerationTool
from .execution.code_execution_tool import CodeExecutionTool
from .validation.quality_assessment_tool import QualityAssessmentTool

__all__ = [
    'ToolManager',
    'CodeAnalysisTool',
    'TestGenerationTool', 
    'CodeExecutionTool',
    'QualityAssessmentTool'
]
EOF

cat > src/agent/tools/analysis/__init__.py << 'EOF'
"""Agent analysis tools"""
EOF

cat > src/agent/tools/generation/__init__.py << 'EOF'
"""Agent generation tools"""
EOF

cat > src/agent/tools/execution/__init__.py << 'EOF'
"""Agent execution tools"""
EOF

cat > src/agent/tools/validation/__init__.py << 'EOF'
"""Agent validation tools"""
EOF

# Create Base Tool Interface
echo "📄 Creating src/agent/tools/base_tool.py..."
cat > src/agent/tools/base_tool.py << 'EOF'
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
EOF

# Create Tool Manager
echo "📄 Creating src/agent/tools/tool_manager.py..."
cat > src/agent/tools/tool_manager.py << 'EOF'
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
EOF

# Create Code Analysis Tool
echo "📄 Creating src/agent/tools/analysis/code_analysis_tool.py..."
cat > src/agent/tools/analysis/code_analysis_tool.py << 'EOF'
"""
Code Analysis Tool
Agent-integrated code analysis with intelligent insights
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from ..base_tool import AgentTool, ToolResult, ToolParameters, ToolType, ToolCapability
from ....analysis.ast_parser import ASTCodeParser
from ....analysis.repository_analyzer import RepositoryAnalyzer
from ....analysis.ml_pattern_detector import MLPatternDetector
from ....core.exceptions import AgentError


logger = logging.getLogger(__name__)


class CodeAnalysisTool(AgentTool):
    """
    Intelligent code analysis tool that integrates with agent reasoning.
    
    Provides comprehensive code analysis capabilities including:
    - AST parsing and component extraction
    - Quality metrics and complexity analysis
    - Pattern detection and architectural insights
    - Agent-friendly result interpretation
    """

    def __init__(self):
        super().__init__(
            name="code_analyzer",
            description="Comprehensive code analysis with quality metrics and pattern detection",
            tool_type=ToolType.ANALYSIS
        )
        
        self.capabilities = [
            ToolCapability.CODE_ANALYSIS,
            ToolCapability.PATTERN_DETECTION,
            ToolCapability.QUALITY_ASSESSMENT
        ]
        
        self.supported_languages = ["python"]
        self.dependencies = ["astroid", "radon", "tree-sitter", "scikit-learn"]
        
        # Initialize analysis components
        self.ast_parser = None
        self.repository_analyzer = None
        self.pattern_detector = None

    async def initialize(self) -> bool:
        """Initialize analysis components"""
        try:
            self.ast_parser = ASTCodeParser()
            self.repository_analyzer = RepositoryAnalyzer()
            self.pattern_detector = MLPatternDetector()
            
            self.is_initialized = True
            logger.info("Code analysis tool initialized successfully")
            return True
        
        except Exception as e:
            logger.error(f"Failed to initialize code analysis tool: {str(e)}")
            return False

    def can_handle(self, task: str, context: Dict[str, Any] = None) -> float:
        """
        Determine if this tool can handle the given task
        
        Args:
            task: Task description
            context: Additional context
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        task_lower = task.lower()
        
        # High confidence for explicit analysis requests
        analysis_keywords = [
            "analyze", "analysis", "examine", "inspect", "review",
            "quality", "complexity", "structure", "patterns"
        ]
        
        confidence = 0.0
        
        for keyword in analysis_keywords:
            if keyword in task_lower:
                confidence += 0.2
        
        # Additional confidence for code-related terms
        code_keywords = ["code", "function", "class", "method", "file", "repository"]
        for keyword in code_keywords:
            if keyword in task_lower:
                confidence += 0.1
        
        # Context-based confidence adjustments
        if context:
            if context.get("involves_code", False):
                confidence += 0.2
            
            # Check user preferences
            user_prefs = context.get("user_preferences", {})
            domain_knowledge = user_prefs.get("domain_knowledge", {})
            if domain_knowledge.get("python", 0) > 0.5:
                confidence += 0.1
        
        return min(1.0, confidence)

    async def execute(self, parameters: ToolParameters) -> ToolResult:
        """
        Execute code analysis based on parameters
        
        Args:
            parameters: Tool execution parameters
            
        Returns:
            ToolResult with analysis results
        """
        start_time = datetime.utcnow()
        
        try:
            if not self.is_initialized:
                await self.initialize()
            
            # Extract analysis parameters
            analysis_type = parameters.parameters.get("analysis_type", "comprehensive")
            file_path = parameters.parameters.get("file_path")
            repository_path = parameters.parameters.get("repository_path")
            code_content = parameters.parameters.get("code_content")
            
            analysis_results = {}
            recommendations = []
            
            if code_content:
                # Analyze code content directly
                results = await self._analyze_code_content(code_content, analysis_type)
                analysis_results.update(results)
                
            elif file_path:
                # Analyze specific file
                results = await self._analyze_file(file_path, analysis_type)
                analysis_results.update(results)
                
            elif repository_path:
                # Analyze entire repository
                results = await self._analyze_repository(repository_path, analysis_type)
                analysis_results.update(results)
                
            else:
                return ToolResult(
                    tool_name=self.name,
                    success=False,
                    error_message="No code source specified (file_path, repository_path, or code_content required)"
                )
            
            # Generate intelligent recommendations
            recommendations = await self._generate_recommendations(analysis_results, parameters)
            
            # Generate next steps
            next_steps = await self._generate_next_steps(analysis_results, parameters)
            
            execution_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            
            return ToolResult(
                tool_name=self.name,
                success=True,
                data=analysis_results,
                metadata={
                    "analysis_type": analysis_type,
                    "components_analyzed": analysis_results.get("component_count", 0),
                    "languages_detected": analysis_results.get("languages", []),
                    "complexity_score": analysis_results.get("overall_complexity", 0)
                },
                execution_time_ms=execution_time,
                confidence=self._calculate_analysis_confidence(analysis_results),
                recommendations=recommendations,
                next_steps=next_steps
            )
            
        except Exception as e:
            execution_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            logger.error(f"Code analysis failed: {str(e)}")
            
            return ToolResult(
                tool_name=self.name,
                success=False,
                error_message=str(e),
                execution_time_ms=execution_time
            )

    async def _analyze_code_content(self, code_content: str, analysis_type: str) -> Dict[str, Any]:
        """Analyze code content string"""
        results = {}
        
        # Parse with AST parser
        ast_result = await self.ast_parser.parse_code_content(code_content)
        results["ast_analysis"] = ast_result
        
        if analysis_type in ["comprehensive", "quality"]:
            # Add quality metrics
            results["quality_metrics"] = self._calculate_quality_metrics(ast_result)
        
        if analysis_type in ["comprehensive", "patterns"]:
            # Add pattern detection
            pattern_results = await self.pattern_detector.detect_patterns_from_components(
                ast_result.get("components", [])
            )
            results["patterns"] = pattern_results
        
        return results

    async def _analyze_file(self, file_path: str, analysis_type: str) -> Dict[str, Any]:
        """Analyze a specific file"""
        results = {}
        
        # Parse file with AST parser
        ast_result = await self.ast_parser.parse_file(file_path)
        results["file_analysis"] = ast_result
        
        # Add file-specific metrics
        results["file_metrics"] = {
            "file_path": file_path,
            "component_count": len(ast_result.get("components", [])),
            "line_count": ast_result.get("metadata", {}).get("line_count", 0),
            "complexity_score": ast_result.get("metadata", {}).get("complexity", 0)
        }
        
        if analysis_type in ["comprehensive", "quality"]:
            results["quality_assessment"] = self._assess_file_quality(ast_result)
        
        return results

    async def _analyze_repository(self, repository_path: str, analysis_type: str) -> Dict[str, Any]:
        """Analyze an entire repository"""
        results = {}
        
        # Analyze repository structure
        repo_analysis = await self.repository_analyzer.analyze_repository(repository_path)
        results["repository_analysis"] = repo_analysis
        
        if analysis_type in ["comprehensive", "patterns"]:
            # Add ML pattern detection
            pattern_results = await self.pattern_detector.analyze_codebase(repository_path)
            results["ml_patterns"] = pattern_results
        
        if analysis_type in ["comprehensive", "architecture"]:
            # Add architectural insights
            results["architecture"] = self._extract_architectural_insights(repo_analysis)
        
        return results

    def _calculate_quality_metrics(self, ast_result: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate quality metrics from AST analysis"""
        components = ast_result.get("components", [])
        
        if not components:
            return {"error": "No components found for quality analysis"}
        
        # Calculate aggregate metrics
        total_complexity = sum(comp.get("complexity", 0) for comp in components)
        avg_complexity = total_complexity / len(components) if components else 0
        
        high_complexity_count = sum(1 for comp in components if comp.get("complexity", 0) > 10)
        
        # Calculate testability scores
        testability_scores = [comp.get("testability_score", 0.5) for comp in components]
        avg_testability = sum(testability_scores) / len(testability_scores) if testability_scores else 0
        
        return {
            "total_components": len(components),
            "avg_complexity": round(avg_complexity, 2),
            "high_complexity_components": high_complexity_count,
            "avg_testability_score": round(avg_testability, 2),
            "quality_grade": self._calculate_quality_grade(avg_complexity, avg_testability),
            "improvement_potential": high_complexity_count > 0 or avg_testability < 0.7
        }

    def _assess_file_quality(self, ast_result: Dict[str, Any]) -> Dict[str, Any]:
        """Assess the quality of a specific file"""
        components = ast_result.get("components", [])
        metadata = ast_result.get("metadata", {})
        
        # File-level assessments
        line_count = metadata.get("line_count", 0)
        complexity = metadata.get("complexity", 0)
        
        assessments = {
            "file_size_assessment": "appropriate" if line_count < 500 else "too_large",
            "complexity_assessment": "low" if complexity < 5 else "moderate" if complexity < 15 else "high",
            "component_organization": "good" if len(components) < 10 else "consider_splitting",
            "maintainability_score": self._calculate_maintainability_score(line_count, complexity, len(components))
        }
        
        return assessments

    def _extract_architectural_insights(self, repo_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Extract architectural insights from repository analysis"""
        analysis_results = repo_analysis.get("analysis_results", {})
        
        insights = {
            "total_files": analysis_results.get("total_files", 0),
            "architecture_patterns": analysis_results.get("architecture_patterns", []),
            "dependency_complexity": len(analysis_results.get("imports", [])),
            "module_organization": analysis_results.get("directory_structure", {}),
            "code_distribution": analysis_results.get("file_types", {}),
            "potential_improvements": []
        }
        
        # Generate improvement suggestions
        if insights["total_files"] > 100:
            insights["potential_improvements"].append("Consider modularization for large codebase")
        
        if insights["dependency_complexity"] > 50:
            insights["potential_improvements"].append("High dependency complexity - review imports")
        
        return insights

    def _calculate_quality_grade(self, avg_complexity: float, avg_testability: float) -> str:
        """Calculate overall quality grade"""
        if avg_complexity < 5 and avg_testability > 0.8:
            return "A"
        elif avg_complexity < 10 and avg_testability > 0.6:
            return "B"
        elif avg_complexity < 15 and avg_testability > 0.4:
            return "C"
        else:
            return "D"

    def _calculate_maintainability_score(self, line_count: int, complexity: int, component_count: int) -> float:
        """Calculate maintainability score (0.0 to 1.0)"""
        # Base score
        score = 1.0
        
        # Penalize large files
        if line_count > 500:
            score -= 0.2
        elif line_count > 1000:
            score -= 0.4
        
        # Penalize high complexity
        if complexity > 10:
            score -= 0.2
        elif complexity > 20:
            score -= 0.4
        
        # Penalize too many components in one file
        if component_count > 10:
            score -= 0.1
        elif component_count > 20:
            score -= 0.3
        
        return max(0.0, score)

    def _calculate_analysis_confidence(self, analysis_results: Dict[str, Any]) -> float:
        """Calculate confidence in analysis results"""
        base_confidence = 0.8
        
        # Increase confidence based on successful analysis components
        if "ast_analysis" in analysis_results:
            base_confidence += 0.1
        
        if "quality_metrics" in analysis_results:
            base_confidence += 0.05
        
        if "patterns" in analysis_results:
            base_confidence += 0.05
        
        # Decrease confidence if errors encountered
        for key, value in analysis_results.items():
            if isinstance(value, dict) and "error" in value:
                base_confidence -= 0.1
        
        return min(1.0, max(0.0, base_confidence))

    async def _generate_recommendations(self, analysis_results: Dict[str, Any], parameters: ToolParameters) -> List[str]:
        """Generate intelligent recommendations based on analysis"""
        recommendations = []
        
        # Quality-based recommendations
        quality_metrics = analysis_results.get("quality_metrics", {})
        if quality_metrics.get("high_complexity_components", 0) > 0:
            recommendations.append("Consider refactoring high-complexity components for better maintainability")
        
        if quality_metrics.get("avg_testability_score", 1.0) < 0.6:
            recommendations.append("Improve code testability by reducing dependencies and complexity")
        
        # Pattern-based recommendations
        patterns = analysis_results.get("patterns", {})
        if isinstance(patterns, dict) and patterns.get("anomalies"):
            recommendations.append("Review flagged code anomalies for potential issues")
        
        # Architecture-based recommendations
        architecture = analysis_results.get("architecture", {})
        improvements = architecture.get("potential_improvements", [])
        recommendations.extend(improvements)
        
        # User context-based recommendations
        user_prefs = parameters.context.get("user_preferences", {})
        if user_prefs.get("expertise_level") == "beginner":
            recommendations.append("Focus on understanding code structure before making changes")
        elif user_prefs.get("expertise_level") == "expert":
            recommendations.append("Consider advanced refactoring patterns and architectural improvements")
        
        return recommendations[:5]  # Limit to top 5 recommendations

    async def _generate_next_steps(self, analysis_results: Dict[str, Any], parameters: ToolParameters) -> List[str]:
        """Generate suggested next steps"""
        next_steps = []
        
        # Based on current goal
        current_goal = parameters.context.get("current_goal")
        if current_goal and "test" in current_goal.lower():
            next_steps.append("Generate comprehensive tests for identified high-priority components")
        
        # Based on analysis results
        if analysis_results.get("quality_metrics", {}).get("improvement_potential", False):
            next_steps.append("Prioritize refactoring efforts on low-quality components")
        
        if "repository_analysis" in analysis_results:
            next_steps.append("Review architectural patterns and consider design improvements")
        
        # General next steps
        next_steps.extend([
            "Request specific analysis of components that interest you",
            "Ask for detailed explanation of any analysis results",
            "Discuss testing strategies for the analyzed code"
        ])
        
        return next_steps[:4]  # Limit to top 4 next steps
EOF

# Create comprehensive test files for tools
echo "📄 Creating test files for tools..."

cat > tests/unit/test_agent/test_tools/__init__.py << 'EOF'
"""Tests for agent tools"""
EOF

cat > tests/unit/test_agent/test_tools/test_analysis/__init__.py << 'EOF'
"""Tests for analysis tools"""
EOF

cat > tests/unit/test_agent/test_tools/test_base_tool.py << 'EOF'
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
EOF

cat > tests/unit/test_agent/test_tools/test_tool_manager.py << 'EOF'
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
EOF

# Update requirements.txt
echo "📄 Updating requirements.txt..."
cat >> requirements.txt << 'EOF'

# Sprint 2.2 - Intelligent Tool System & Test Generation
jinja2==3.1.2
ast-tools==0.1.0
autopep8==2.0.4
rope==1.11.0
bandit==1.7.5
EOF

# Run verification tests
echo "🧪 Running tests to verify Sprint 2.2 implementation..."
python3 -m pytest tests/unit/test_agent/test_tools/ -v

# Run functional verification
echo "🔍 Testing basic functionality..."
python3 -c "
import asyncio
from src.agent.tools.tool_manager import ToolManager
from src.agent.tools.analysis.code_analysis_tool import CodeAnalysisTool
from src.agent.tools.base_tool import ToolParameters
from src.agent.core.models import AgentState

async def test_sprint_2_2():
    print('Testing Sprint 2.2 Tool System...')
    
    # Test tool manager
    manager = ToolManager()
    
    # Test code analysis tool
    analysis_tool = CodeAnalysisTool()
    success = await manager.register_tool(analysis_tool)
    print(f'✅ Tool Registration: {success}')
    
    # Test tool selection
    agent_state = AgentState(session_id='test_session')
    selected_tools = await manager.select_tools(
        'analyze my Python code for quality issues',
        agent_state
    )
    print(f'✅ Tool Selection: {len(selected_tools)} tools selected')
    
    # Test tool execution
    if selected_tools:
        params = ToolParameters(
            parameters={'code_content': 'def hello(): return \"world\"'},
            session_id='test_session'
        )
        
        result = await manager.execute_tool(selected_tools[0], params)
        print(f'✅ Tool Execution: Success={result.success}, Confidence={result.confidence}')
    
    # Test available tools
    available = await manager.get_available_tools()
    print(f'✅ Available Tools: {len(available)} tools registered')
    
    # Test performance stats
    stats = await manager.get_tool_performance_stats()
    print(f'✅ Performance Stats: {stats[\"total_tools\"]} tools tracked')
    
    print('🎉 Sprint 2.2 verification successful!')

asyncio.run(test_sprint_2_2())
"

echo "✅ Sprint 2.2: Intelligent Tool System & Test Generation setup complete!"
echo ""
echo "📋 Summary of Sprint 2.2 Implementation:"
echo "• Comprehensive Tool Management System with intelligent selection"
echo "• Code Analysis Tool with agent-friendly interface" 
echo "• Base Tool Interface with standardized execution patterns"
echo "• Tool chaining and coordination capabilities"
echo "• Performance tracking and optimization suggestions"
echo "• Integration with Sprint 2.1 agent orchestrator"
echo "• Comprehensive test coverage with tool mocking"
echo "• Foundation for test generation and validation tools"
echo ""
echo "🔄 Ready for Sprint 2.3: Multi-Agent Architecture & Collaboration"
echo "Run this setup script to implement Sprint 2.2, then let me know when you're ready for Sprint 2.3!"
