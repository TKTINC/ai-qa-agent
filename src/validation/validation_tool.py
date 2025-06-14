"""
Standalone Validation Tool

Intelligent validation tool that can work independently and be integrated with agents later.
"""

import asyncio
import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ValidationIssue:
    """Represents a validation issue found in code"""
    issue_type: str
    severity: str  # critical, major, minor, info
    message: str
    line_number: Optional[int] = None
    column: Optional[int] = None
    file_path: Optional[str] = None
    suggestion: Optional[str] = None
    confidence: float = 1.0


@dataclass
class ReasonedValidationResult:
    """Validation result with reasoning"""
    validation_passed: bool
    issues_found: List[ValidationIssue]
    reasoning: str
    severity_assessment: str
    improvement_suggestions: List[str]
    confidence: float
    learning_insights: List[str]
    validation_duration: float
    tools_used: List[str] = field(default_factory=list)
    
    def to_agent_message(self) -> str:
        """Convert to natural language for agent communication"""
        if self.validation_passed:
            return f"✅ Validation passed with {self.confidence:.1%} confidence. {self.reasoning}"
        else:
            issues_summary = f"{len(self.issues_found)} issues found"
            critical_issues = len([i for i in self.issues_found if i.severity == 'critical'])
            if critical_issues > 0:
                issues_summary += f" ({critical_issues} critical)"
            return f"❌ Validation failed: {issues_summary}. {self.reasoning}"


@dataclass
class ToolResult:
    """Tool execution result"""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    message: Optional[str] = None


@dataclass
class CorrectedCode:
    """Result of automatic code correction"""
    original_code: str
    corrected_code: str
    corrections_made: List[str]
    confidence: float
    still_has_issues: bool
    remaining_issues: List[ValidationIssue] = field(default_factory=list)


class SyntaxValidator:
    """Validates code syntax for multiple languages"""
    
    async def validate_python_syntax(self, code: str) -> List[ValidationIssue]:
        """Validate Python syntax"""
        issues = []
        try:
            compile(code, '<string>', 'exec')
        except SyntaxError as e:
            issues.append(ValidationIssue(
                issue_type="syntax_error",
                severity="critical",
                message=f"Python syntax error: {e.msg}",
                line_number=e.lineno,
                column=e.offset,
                suggestion="Fix syntax error before proceeding"
            ))
        except Exception as e:
            issues.append(ValidationIssue(
                issue_type="compilation_error",
                severity="critical",
                message=f"Compilation error: {str(e)}",
                suggestion="Check code structure and syntax"
            ))
        return issues
    
    async def validate_javascript_syntax(self, code: str) -> List[ValidationIssue]:
        """Validate JavaScript syntax (basic check)"""
        issues = []
        # Basic JS syntax checks
        if code.count('{') != code.count('}'):
            issues.append(ValidationIssue(
                issue_type="brace_mismatch",
                severity="critical",
                message="Mismatched braces in JavaScript code",
                suggestion="Ensure all opening braces have matching closing braces"
            ))
        if code.count('(') != code.count(')'):
            issues.append(ValidationIssue(
                issue_type="parenthesis_mismatch",
                severity="critical",
                message="Mismatched parentheses in JavaScript code",
                suggestion="Ensure all opening parentheses have matching closing parentheses"
            ))
        return issues


class SemanticValidator:
    """Validates code semantics and best practices"""
    
    async def validate_python_semantics(self, code: str) -> List[ValidationIssue]:
        """Validate Python semantic issues"""
        issues = []
        
        lines = code.split('\n')
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            
            # Check for unused imports (simple heuristic)
            if stripped.startswith('import ') or stripped.startswith('from '):
                import_name = stripped.split()[-1].split('.')[0]
                if import_name not in code[len(line):]:
                    issues.append(ValidationIssue(
                        issue_type="unused_import",
                        severity="minor",
                        message=f"Potentially unused import: {import_name}",
                        line_number=i,
                        suggestion="Remove unused imports to improve code clarity"
                    ))
            
            # Check for bare except clauses
            if 'except:' in stripped and 'except Exception:' not in stripped:
                issues.append(ValidationIssue(
                    issue_type="bare_except",
                    severity="major",
                    message="Bare except clause catches all exceptions",
                    line_number=i,
                    suggestion="Use specific exception types or 'except Exception:'"
                ))
        
        return issues
    
    async def validate_test_semantics(self, code: str) -> List[ValidationIssue]:
        """Validate test-specific semantic issues"""
        issues = []
        
        # Check for test function naming
        if 'def ' in code and not any(line.strip().startswith('def test_') for line in code.split('\n')):
            issues.append(ValidationIssue(
                issue_type="test_naming",
                severity="major",
                message="Test functions should start with 'test_'",
                suggestion="Rename test functions to follow 'test_' naming convention"
            ))
        
        # Check for assertions in tests
        if 'def test_' in code and 'assert' not in code:
            issues.append(ValidationIssue(
                issue_type="missing_assertions",
                severity="critical",
                message="Test function has no assertions",
                suggestion="Add assert statements to verify expected behavior"
            ))
        
        return issues


class QualityAssessor:
    """Assesses code quality and testing effectiveness"""
    
    async def assess_test_quality(self, code: str) -> Dict[str, Any]:
        """Assess the quality of test code"""
        quality_metrics = {
            'readability_score': 0.0,
            'coverage_potential': 0.0,
            'assertion_count': 0,
            'edge_case_coverage': 0.0,
            'maintainability': 0.0
        }
        
        lines = code.split('\n')
        
        # Count assertions
        assertion_count = sum(1 for line in lines if 'assert' in line.strip())
        quality_metrics['assertion_count'] = assertion_count
        
        # Readability score based on line length and complexity
        long_lines = sum(1 for line in lines if len(line) > 100)
        avg_line_length = sum(len(line) for line in lines) / max(len(lines), 1)
        readability = max(0, 1.0 - (long_lines / max(len(lines), 1)) - (avg_line_length - 50) / 100)
        quality_metrics['readability_score'] = min(1.0, readability)
        
        # Edge case coverage heuristic
        edge_case_indicators = ['None', 'empty', 'zero', 'negative', 'invalid', 'error', 'exception']
        edge_case_count = sum(1 for indicator in edge_case_indicators 
                             if indicator.lower() in code.lower())
        quality_metrics['edge_case_coverage'] = min(1.0, edge_case_count / 5.0)
        
        # Maintainability based on documentation and structure
        has_docstring = '"""' in code or "'''" in code
        has_comments = '#' in code
        maintainability = 0.5 + (0.3 if has_docstring else 0) + (0.2 if has_comments else 0)
        quality_metrics['maintainability'] = maintainability
        
        return quality_metrics


class StandaloneValidationTool:
    """Standalone validation tool that doesn't depend on agent system"""
    
    def __init__(self):
        self.name = "validation_tool"
        self.description = "Validate code syntax, semantics, and quality with intelligent analysis"
        self.parameters = {
            "code": "Code content to validate",
            "language": "Programming language (python, javascript, etc.)",
            "validation_type": "Type of validation (syntax, semantic, quality, all)",
            "context": "Additional context for validation"
        }
        self.syntax_validator = SyntaxValidator()
        self.semantic_validator = SemanticValidator()
        self.quality_assessor = QualityAssessor()
    
    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        """Execute validation with reasoning"""
        start_time = datetime.now()
        
        try:
            code = parameters.get("code", "")
            language = parameters.get("language", "python").lower()
            validation_type = parameters.get("validation_type", "all")
            context = parameters.get("context", {})
            
            # Perform validation with reasoning
            result = await self.validate_with_reasoning(code, language, validation_type, context)
            
            duration = (datetime.now() - start_time).total_seconds()
            result.validation_duration = duration
            
            return ToolResult(
                success=True,
                data={
                    "validation_result": result,
                    "agent_message": result.to_agent_message(),
                    "quality_metrics": await self.quality_assessor.assess_test_quality(code) if 'test' in validation_type else {},
                    "duration": duration
                },
                message=result.to_agent_message()
            )
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return ToolResult(
                success=False,
                error=str(e),
                message=f"Validation failed due to error: {e}"
            )
    
    async def validate_with_reasoning(self, code: str, language: str, validation_type: str, context: Dict[str, Any]) -> ReasonedValidationResult:
        """Validate code with reasoning about results and improvement suggestions"""
        issues = []
        tools_used = []
        
        # Step 1: Syntax validation
        if validation_type in ["syntax", "all"]:
            if language == "python":
                syntax_issues = await self.syntax_validator.validate_python_syntax(code)
                issues.extend(syntax_issues)
                tools_used.append("python_syntax_validator")
            elif language == "javascript":
                syntax_issues = await self.syntax_validator.validate_javascript_syntax(code)
                issues.extend(syntax_issues)
                tools_used.append("javascript_syntax_validator")
        
        # Step 2: Semantic validation
        if validation_type in ["semantic", "all"]:
            if language == "python":
                semantic_issues = await self.semantic_validator.validate_python_semantics(code)
                issues.extend(semantic_issues)
                tools_used.append("python_semantic_validator")
                
                # Additional test-specific validation
                if "test" in code.lower() or "def test_" in code:
                    test_issues = await self.semantic_validator.validate_test_semantics(code)
                    issues.extend(test_issues)
                    tools_used.append("test_semantic_validator")
        
        # Step 3: Generate reasoning
        critical_issues = [i for i in issues if i.severity == "critical"]
        major_issues = [i for i in issues if i.severity == "major"]
        
        if critical_issues:
            reasoning = f"Found {len(critical_issues)} critical issues that prevent code execution. " \
                       f"Primary concern: {critical_issues[0].message}"
            severity_assessment = "Critical - requires immediate attention"
            validation_passed = False
        elif major_issues:
            reasoning = f"Found {len(major_issues)} major issues affecting code quality. " \
                       f"Most significant: {major_issues[0].message}"
            severity_assessment = "Major - should be addressed"
            validation_passed = False
        elif issues:
            reasoning = f"Found {len(issues)} minor issues that could be improved. " \
                       f"Overall code quality is acceptable."
            severity_assessment = "Minor - improvements recommended"
            validation_passed = True
        else:
            reasoning = "No validation issues found. Code meets quality standards."
            severity_assessment = "Excellent - no issues detected"
            validation_passed = True
        
        # Generate improvement suggestions
        suggestions = []
        for issue in issues[:3]:  # Top 3 issues
            if issue.suggestion:
                suggestions.append(issue.suggestion)
        
        # Generate learning insights
        learning_insights = []
        if language == "python" and any("test_" in issue.message for issue in issues):
            learning_insights.append("User's code involves Python testing patterns")
        
        # Calculate confidence based on validation completeness
        confidence = 0.9 if len(tools_used) >= 2 else 0.7
        if validation_type == "all":
            confidence = min(1.0, confidence + 0.1)
        
        return ReasonedValidationResult(
            validation_passed=validation_passed,
            issues_found=issues,
            reasoning=reasoning,
            severity_assessment=severity_assessment,
            improvement_suggestions=suggestions,
            confidence=confidence,
            learning_insights=learning_insights,
            validation_duration=0.0,  # Will be set by caller
            tools_used=tools_used
        )
    
    async def self_correct(self, validation_result: ReasonedValidationResult, original_code: str) -> CorrectedCode:
        """Attempt to automatically fix validation issues"""
        corrected_code = original_code
        corrections_made = []
        
        for issue in validation_result.issues_found:
            if issue.severity in ["critical", "major"]:
                if issue.issue_type == "unused_import" and issue.line_number:
                    # Remove unused import
                    lines = corrected_code.split('\n')
                    if 0 <= issue.line_number - 1 < len(lines):
                        lines[issue.line_number - 1] = ""
                        corrected_code = '\n'.join(lines)
                        corrections_made.append(f"Removed unused import at line {issue.line_number}")
                
                elif issue.issue_type == "bare_except":
                    # Replace bare except with Exception
                    corrected_code = corrected_code.replace("except:", "except Exception:")
                    corrections_made.append("Replaced bare except with 'except Exception:'")
        
        confidence = 0.8 if corrections_made else 0.3
        
        return CorrectedCode(
            original_code=original_code,
            corrected_code=corrected_code,
            corrections_made=corrections_made,
            confidence=confidence,
            still_has_issues=False,  # Simplified for now
            remaining_issues=[]
        )
    
    def can_handle(self, task: str) -> float:
        """Return confidence score 0-1 for handling validation tasks"""
        validation_keywords = [
            "validate", "check", "verify", "syntax", "quality", 
            "test", "lint", "format", "error", "issue"
        ]
        
        task_lower = task.lower()
        matches = sum(1 for keyword in validation_keywords if keyword in task_lower)
        
        # Higher confidence for specific validation requests
        if "validation" in task_lower or "validate" in task_lower:
            return min(1.0, 0.8 + matches * 0.05)
        
        return min(1.0, matches * 0.15)


# Export classes for use by other modules
__all__ = [
    'StandaloneValidationTool',
    'ReasonedValidationResult', 
    'ValidationIssue',
    'CorrectedCode',
    'SyntaxValidator',
    'SemanticValidator',
    'QualityAssessor'
]
