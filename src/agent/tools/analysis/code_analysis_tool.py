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
