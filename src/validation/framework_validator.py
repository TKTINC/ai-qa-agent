"""
Multi-Framework Validation

Support for validating tests across different testing frameworks.
"""

import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

# Use ValidationIssue from validation_tool
from .validation_tool import ValidationIssue


@dataclass
class FrameworkInfo:
    """Information about a detected testing framework"""
    name: str
    version: Optional[str]
    confidence: float
    indicators: List[str]
    best_practices: List[str]


@dataclass
class FrameworkValidationResult:
    """Result of framework-specific validation"""
    framework: FrameworkInfo
    validation_passed: bool
    framework_issues: List[ValidationIssue]
    best_practice_violations: List[ValidationIssue]
    framework_specific_suggestions: List[str]
    configuration_recommendations: List[str]


class PytestValidator:
    """Validator for pytest framework"""
    
    def detect_framework(self, code: str, file_path: Optional[str] = None) -> FrameworkInfo:
        """Detect pytest usage"""
        indicators = []
        confidence = 0.0
        
        # Check for pytest imports
        if "import pytest" in code or "from pytest" in code:
            indicators.append("pytest import found")
            confidence += 0.4
        
        # Check for pytest decorators
        pytest_decorators = ["@pytest.fixture", "@pytest.mark", "@pytest.parametrize"]
        for decorator in pytest_decorators:
            if decorator in code:
                indicators.append(f"{decorator} decorator found")
                confidence += 0.3
        
        # Check for pytest-specific patterns
        if "def test_" in code and ("assert " in code):
            indicators.append("pytest test function pattern")
            confidence += 0.2
        
        confidence = min(1.0, confidence)
        
        return FrameworkInfo(
            name="pytest",
            version=None,
            confidence=confidence,
            indicators=indicators,
            best_practices=self.get_best_practices()
        )
    
    async def validate_framework_usage(self, code: str) -> List[ValidationIssue]:
        """Validate pytest-specific patterns"""
        issues = []
        
        # Check for proper assertion usage
        if "def test_" in code and "assert " not in code:
            issues.append(ValidationIssue(
                issue_type="missing_assertions",
                severity="major",
                message="Test function lacks assertions - tests should verify expected behavior",
                suggestion="Add assert statements to verify the expected outcomes"
            ))
        
        return issues
    
    def get_best_practices(self) -> List[str]:
        """Get pytest best practices"""
        return [
            "Use descriptive test function names starting with 'test_'",
            "Use fixtures for test setup and teardown",
            "Use parametrize for testing multiple inputs",
            "Keep tests isolated and independent"
        ]


class MultiFrameworkValidator:
    """Main validator that handles multiple testing frameworks"""
    
    def __init__(self):
        self.validators = {
            "pytest": PytestValidator()
        }
    
    async def validate_with_framework_detection(self, code: str, file_path: Optional[str] = None) -> FrameworkValidationResult:
        """Validate code with automatic framework detection"""
        
        # Try to detect pytest first
        pytest_validator = self.validators["pytest"]
        framework_info = pytest_validator.detect_framework(code, file_path)
        
        if framework_info.confidence > 0.3:
            # Use pytest validation
            framework_issues = await pytest_validator.validate_framework_usage(code)
            
            return FrameworkValidationResult(
                framework=framework_info,
                validation_passed=len(framework_issues) == 0,
                framework_issues=framework_issues,
                best_practice_violations=[],
                framework_specific_suggestions=framework_info.best_practices,
                configuration_recommendations=["Create pytest.ini for configuration"]
            )
        
        # No framework detected
        return FrameworkValidationResult(
            framework=FrameworkInfo(
                name="unknown",
                version=None,
                confidence=0.0,
                indicators=[],
                best_practices=[]
            ),
            validation_passed=True,
            framework_issues=[],
            best_practice_violations=[],
            framework_specific_suggestions=["Consider using a testing framework like pytest"],
            configuration_recommendations=[]
        )


# Export main classes
__all__ = [
    'MultiFrameworkValidator',
    'FrameworkValidationResult',
    'FrameworkInfo',
    'PytestValidator'
]
