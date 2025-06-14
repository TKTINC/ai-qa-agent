"""
Intelligent Validation Engine

Validation engine that learns and improves over time, providing context-aware validation
that considers user goals and preferences.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from pathlib import Path

from ..agent.core.models import UserProfile, ConversationContext
from ..agent.tools.validation_tool import ValidationIssue, ReasonedValidationResult
from ..core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class ValidationContext:
    """Context for intelligent validation"""
    code: str
    language: str
    validation_type: str
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    agent_context: Dict[str, Any] = field(default_factory=dict)
    project_context: Optional[Dict[str, Any]] = None
    historical_patterns: List[str] = field(default_factory=list)


@dataclass
class ContextualValidation:
    """Validation result with contextual awareness"""
    base_validation: ReasonedValidationResult
    context_adjustments: List[str]
    user_specific_insights: List[str]
    project_specific_recommendations: List[str]
    learning_applied: List[str]
    confidence_adjustment: float = 0.0


@dataclass
class ValidationExplanation:
    """User-friendly explanation of validation results"""
    summary: str
    detailed_explanations: List[str]
    educational_content: List[str]
    actionable_steps: List[str]
    related_concepts: List[str]


@dataclass
class ImprovementSuggestion:
    """Specific improvement suggestion"""
    category: str
    description: str
    impact: str  # high, medium, low
    effort: str  # easy, moderate, complex
    code_example: Optional[str] = None
    learning_resource: Optional[str] = None


class ContextAwareValidator:
    """Validator that adapts to user context and preferences"""
    
    def __init__(self):
        self.user_patterns = {}  # Cache user-specific patterns
        self.project_patterns = {}  # Cache project-specific patterns
        
    async def adapt_validation_to_user(self, 
                                     validation_result: ReasonedValidationResult,
                                     user_profile: UserProfile) -> ContextualValidation:
        """Adapt validation results to user expertise and preferences"""
        context_adjustments = []
        user_insights = []
        
        # Adjust based on user expertise level
        if user_profile.expertise_level == "beginner":
            # Filter out minor issues for beginners, focus on critical ones
            critical_issues = [i for i in validation_result.issues_found 
                             if i.severity in ["critical", "major"]]
            if len(critical_issues) < len(validation_result.issues_found):
                context_adjustments.append(
                    f"Filtered {len(validation_result.issues_found) - len(critical_issues)} "
                    f"minor issues to focus on critical problems"
                )
                validation_result.issues_found = critical_issues
            
            user_insights.append("Focus on fixing critical issues first")
            
        elif user_profile.expertise_level == "expert":
            # Show all issues and advanced suggestions
            user_insights.append("All validation issues shown including minor optimizations")
            
        # Adapt to preferred frameworks
        if user_profile.preferred_frameworks:
            framework_specific = []
            for framework in user_profile.preferred_frameworks:
                if framework.lower() == "pytest" and "test_" in validation_result.reasoning:
                    framework_specific.append("Consider using pytest fixtures for setup")
                elif framework.lower() == "unittest" and "test_" in validation_result.reasoning:
                    framework_specific.append("Consider using unittest.TestCase for structure")
            
            if framework_specific:
                user_insights.extend(framework_specific)
                context_adjustments.append(f"Added {framework} specific recommendations")
        
        # Adapt to communication style
        if user_profile.communication_style == "detailed":
            user_insights.append("Detailed technical explanations available on request")
        elif user_profile.communication_style == "concise":
            # Shorten explanations
            validation_result.reasoning = validation_result.reasoning[:100] + "..." \
                if len(validation_result.reasoning) > 100 else validation_result.reasoning
            context_adjustments.append("Condensed explanations for concise communication style")
        
        return ContextualValidation(
            base_validation=validation_result,
            context_adjustments=context_adjustments,
            user_specific_insights=user_insights,
            project_specific_recommendations=[],
            learning_applied=[],
            confidence_adjustment=0.1 if user_profile.expertise_level == "expert" else 0.0
        )
    
    async def apply_project_context(self,
                                  validation: ContextualValidation,
                                  project_context: Dict[str, Any]) -> ContextualValidation:
        """Apply project-specific context to validation"""
        project_recommendations = []
        
        # Apply project coding standards
        if "coding_standards" in project_context:
            standards = project_context["coding_standards"]
            if standards.get("max_line_length"):
                max_length = standards["max_line_length"]
                long_lines = sum(1 for line in validation.base_validation.reasoning.split('\n') 
                               if len(line) > max_length)
                if long_lines > 0:
                    project_recommendations.append(
                        f"Project standard: Keep lines under {max_length} characters"
                    )
            
            if standards.get("require_docstrings", False):
                project_recommendations.append("Project requires docstrings for all functions")
        
        # Apply project testing patterns
        if "testing_patterns" in project_context:
            patterns = project_context["testing_patterns"]
            if patterns.get("preferred_assertion_style"):
                style = patterns["preferred_assertion_style"]
                project_recommendations.append(f"Project prefers {style} assertion style")
        
        validation.project_specific_recommendations = project_recommendations
        return validation


class LearningEnhancedValidator:
    """Validator that learns from user feedback and outcomes"""
    
    def __init__(self):
        self.validation_history = []
        self.user_feedback_patterns = {}
        self.success_patterns = {}
        
    async def learn_from_feedback(self,
                                validation_id: str,
                                user_feedback: Dict[str, Any],
                                outcome_success: bool):
        """Learn from user feedback on validation results"""
        learning_entry = {
            "validation_id": validation_id,
            "feedback": user_feedback,
            "success": outcome_success,
            "timestamp": datetime.now()
        }
        
        self.validation_history.append(learning_entry)
        
        # Extract patterns from feedback
        if user_feedback.get("too_verbose"):
            self._record_pattern("verbosity", "reduce", user_feedback.get("user_id"))
        
        if user_feedback.get("missed_important_issue"):
            self._record_pattern("sensitivity", "increase", user_feedback.get("user_id"))
        
        if user_feedback.get("too_many_minor_issues"):
            self._record_pattern("minor_filtering", "increase", user_feedback.get("user_id"))
    
    def _record_pattern(self, pattern_type: str, adjustment: str, user_id: str):
        """Record learning pattern"""
        if user_id not in self.user_feedback_patterns:
            self.user_feedback_patterns[user_id] = {}
        
        if pattern_type not in self.user_feedback_patterns[user_id]:
            self.user_feedback_patterns[user_id][pattern_type] = []
        
        self.user_feedback_patterns[user_id][pattern_type].append({
            "adjustment": adjustment,
            "timestamp": datetime.now()
        })
    
    async def apply_learned_patterns(self,
                                   validation: ContextualValidation,
                                   user_id: str) -> ContextualValidation:
        """Apply learned patterns to improve validation"""
        learning_applied = []
        
        if user_id in self.user_feedback_patterns:
            patterns = self.user_feedback_patterns[user_id]
            
            # Apply verbosity adjustments
            if "verbosity" in patterns:
                recent_feedback = [p for p in patterns["verbosity"] 
                                 if p["timestamp"] > datetime.now() - timedelta(days=30)]
                if recent_feedback and recent_feedback[-1]["adjustment"] == "reduce":
                    # Reduce explanation length
                    original_length = len(validation.base_validation.reasoning)
                    validation.base_validation.reasoning = validation.base_validation.reasoning[:200]
                    learning_applied.append(f"Reduced explanation length based on user preference")
            
            # Apply sensitivity adjustments
            if "sensitivity" in patterns:
                recent_feedback = [p for p in patterns["sensitivity"]
                                 if p["timestamp"] > datetime.now() - timedelta(days=30)]
                if recent_feedback and recent_feedback[-1]["adjustment"] == "increase":
                    # Add more thorough checking
                    validation.user_specific_insights.append(
                        "Applied enhanced checking based on previous feedback"
                    )
                    learning_applied.append("Increased validation sensitivity")
        
        validation.learning_applied = learning_applied
        return validation


class IntelligentValidator:
    """Main intelligent validation engine that learns and improves over time"""
    
    def __init__(self):
        self.context_validator = ContextAwareValidator()
        self.learning_validator = LearningEnhancedValidator()
        self.explanation_generator = ExplanationGenerator()
        
    async def validate_with_context(self,
                                   code: str,
                                   test_purpose: str,
                                   user_preferences: UserProfile,
                                   conversation_context: ConversationContext) -> ContextualValidation:
        """Context-aware validation that considers user goals and preferences"""
        
        # Import here to avoid circular imports
        from ..agent.tools.validation_tool import ValidationTool
        
        # Create base validation
        validation_tool = ValidationTool()
        
        # Determine validation type based on test purpose
        validation_type = "all"
        if "syntax" in test_purpose.lower():
            validation_type = "syntax"
        elif "quality" in test_purpose.lower():
            validation_type = "quality"
        
        # Execute base validation
        result = await validation_tool.execute({
            "code": code,
            "language": "python",  # Default, could be detected
            "validation_type": validation_type,
            "context": {
                "user_preferences": user_preferences.__dict__ if hasattr(user_preferences, '__dict__') else {},
                "agent_context": conversation_context.__dict__ if hasattr(conversation_context, '__dict__') else {}
            }
        })
        
        base_validation = result.data["validation_result"]
        
        # Apply context-aware adaptations
        contextual_validation = await self.context_validator.adapt_validation_to_user(
            base_validation, user_preferences
        )
        
        # Apply learning from previous interactions
        user_id = getattr(user_preferences, 'user_id', 'anonymous')
        contextual_validation = await self.learning_validator.apply_learned_patterns(
            contextual_validation, user_id
        )
        
        return contextual_validation
    
    async def explain_validation_results(self, 
                                       result: ContextualValidation) -> ValidationExplanation:
        """Provide clear explanations of validation issues for user education"""
        return await self.explanation_generator.generate_explanation(result)
    
    async def suggest_improvements(self, 
                                 result: ContextualValidation) -> List[ImprovementSuggestion]:
        """Generate specific, actionable improvement recommendations"""
        suggestions = []
        
        for issue in result.base_validation.issues_found[:5]:  # Top 5 issues
            suggestion = ImprovementSuggestion(
                category=issue.issue_type,
                description=issue.message,
                impact=self._determine_impact(issue.severity),
                effort=self._determine_effort(issue.issue_type),
                code_example=self._generate_code_example(issue),
                learning_resource=self._suggest_learning_resource(issue.issue_type)
            )
            suggestions.append(suggestion)
        
        return suggestions
    
    def _determine_impact(self, severity: str) -> str:
        """Determine impact level from severity"""
        mapping = {
            "critical": "high",
            "major": "medium", 
            "minor": "low",
            "info": "low"
        }
        return mapping.get(severity, "medium")
    
    def _determine_effort(self, issue_type: str) -> str:
        """Determine effort required to fix issue"""
        easy_fixes = ["unused_import", "print_in_test", "bare_except"]
        complex_fixes = ["syntax_error", "missing_assertions"]
        
        if issue_type in easy_fixes:
            return "easy"
        elif issue_type in complex_fixes:
            return "complex"
        else:
            return "moderate"
    
    def _generate_code_example(self, issue: ValidationIssue) -> Optional[str]:
        """Generate code example for fixing the issue"""
        examples = {
            "unused_import": "# Remove unused imports\n# from unused_module import something  # Remove this",
            "bare_except": "# Replace bare except\ntry:\n    risky_operation()\nexcept Exception as e:  # Better than 'except:'\n    handle_error(e)",
            "missing_assertions": "def test_function():\n    result = function_to_test()\n    assert result == expected_value  # Add meaningful assertions"
        }
        return examples.get(issue.issue_type)
    
    def _suggest_learning_resource(self, issue_type: str) -> Optional[str]:
        """Suggest learning resources for the issue type"""
        resources = {
            "syntax_error": "Python syntax documentation: https://docs.python.org/3/tutorial/",
            "unused_import": "PEP 8 Style Guide: https://pep8.org/",
            "missing_assertions": "Python testing guide: https://docs.python.org/3/library/unittest.html"
        }
        return resources.get(issue_type)


class ExplanationGenerator:
    """Generates user-friendly explanations of validation results"""
    
    async def generate_explanation(self, validation: ContextualValidation) -> ValidationExplanation:
        """Generate comprehensive explanation of validation results"""
        base = validation.base_validation
        
        # Generate summary
        if base.validation_passed:
            summary = f"✅ Validation successful! {len(base.issues_found)} minor improvements available."
        else:
            critical_count = len([i for i in base.issues_found if i.severity == "critical"])
            summary = f"❌ Validation found {len(base.issues_found)} issues ({critical_count} critical)."
        
        # Generate detailed explanations
        detailed = []
        for issue in base.issues_found[:3]:  # Top 3 issues
            explanation = f"**{issue.issue_type.replace('_', ' ').title()}**: {issue.message}"
            if issue.suggestion:
                explanation += f"\n  💡 Suggestion: {issue.suggestion}"
            detailed.append(explanation)
        
        # Generate educational content
        educational = []
        if any("test_" in issue.message for issue in base.issues_found):
            educational.append("Testing best practices: Always include meaningful assertions in test functions")
        
        if any("syntax" in issue.issue_type for issue in base.issues_found):
            educational.append("Syntax errors prevent code from running and should be fixed first")
        
        # Generate actionable steps
        actionable = []
        for suggestion in base.improvement_suggestions[:3]:
            actionable.append(f"• {suggestion}")
        
        # Add context-specific actions
        for insight in validation.user_specific_insights:
            actionable.append(f"• {insight}")
        
        # Related concepts
        related = []
        if "test" in base.reasoning.lower():
            related.extend(["Test-driven development", "Unit testing", "Code coverage"])
        
        if "quality" in base.reasoning.lower():
            related.extend(["Code quality metrics", "Static analysis", "Linting"])
        
        return ValidationExplanation(
            summary=summary,
            detailed_explanations=detailed,
            educational_content=educational,
            actionable_steps=actionable,
            related_concepts=related
        )


# Export main classes
__all__ = [
    'IntelligentValidator',
    'ContextualValidation',
    'ValidationExplanation', 
    'ImprovementSuggestion',
    'ValidationContext'
]
