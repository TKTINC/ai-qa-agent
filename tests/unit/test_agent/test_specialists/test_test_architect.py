"""
Test Test Architect Agent
"""

import pytest
import asyncio
from datetime import datetime

from src.agent.specialists.test_architect import TestArchitectAgent
from src.agent.communication.models import ConsultationRequest, AgentCapability


class TestTestArchitectAgent:
    
    @pytest.fixture
    async def test_architect(self):
        """Create Test Architect agent instance"""
        agent = TestArchitectAgent()
        await agent.initialize_specialist()
        return agent
    
    def test_initialization(self, test_architect):
        """Test Test Architect initialization"""
        assert test_architect.name == "test_architect"
        assert test_architect.specialization == "Test Architecture & Strategy"
        assert "test_strategy" in test_architect.expertise_domains
        assert len(test_architect.capabilities) > 3
        
        # Check for specific capabilities
        capability_names = [cap.name for cap in test_architect.capabilities]
        assert "test_strategy_design" in capability_names
        assert "test_architecture_planning" in capability_names

    @pytest.mark.asyncio
    async def test_analyze_problem_legacy_code(self, test_architect):
        """Test analysis of legacy code testing problem"""
        problem = "I have a legacy codebase with no tests and need to add comprehensive testing"
        context = {
            "codebase_size": 15000,
            "has_legacy_code": True,
            "team_experience": "intermediate"
        }
        
        result = await test_architect.analyze_problem(problem, context)
        
        assert result["analysis_type"] == "test_architecture"
        assert "problem_assessment" in result
        assert "architecture_recommendations" in result
        assert "strategy_recommendations" in result
        
        # Check problem assessment
        problem_assessment = result["problem_assessment"]
        assert problem_assessment["problem_type"] == "legacy_code_testing"
        assert "Lack of existing test coverage" in problem_assessment["key_challenges"]

    @pytest.mark.asyncio
    async def test_analyze_problem_performance_testing(self, test_architect):
        """Test analysis of performance testing problem"""
        problem = "Need to implement performance testing for our API"
        context = {
            "api_heavy": True,
            "expected_load": 1000,
            "current_testing_state": {}
        }
        
        result = await test_architect.analyze_problem(problem, context)
        
        assert result["analysis_type"] == "test_architecture"
        problem_assessment = result["problem_assessment"]
        assert problem_assessment["problem_type"] == "performance_testing"

    @pytest.mark.asyncio
    async def test_provide_consultation_strategy(self, test_architect):
        """Test consultation on testing strategy"""
        request = ConsultationRequest(
            requesting_agent="user_agent",
            specialist_agent="test_architect",
            question="What testing strategy should I use for a new microservices project?",
            context={"project_type": "microservices", "team_size": 5}
        )
        
        response = await test_architect.provide_consultation(request)
        
        assert response.responding_agent == "test_architect"
        assert response.confidence > 0.8
        assert len(response.recommendations) > 0
        assert "strategy" in response.response.lower()

    @pytest.mark.asyncio
    async def test_provide_consultation_architecture(self, test_architect):
        """Test consultation on test architecture"""
        request = ConsultationRequest(
            requesting_agent="user_agent",
            specialist_agent="test_architect",
            question="How should I organize my test architecture?",
            context={"application_type": "web_api"}
        )
        
        response = await test_architect.provide_consultation(request)
        
        assert response.responding_agent == "test_architect"
        assert response.confidence > 0.8
        assert "architecture" in response.response.lower()
        assert len(response.follow_up_questions) > 0

    @pytest.mark.asyncio
    async def test_collaborate_on_task(self, test_architect):
        """Test collaboration on a task"""
        task = "Design comprehensive testing approach for e-commerce API"
        collaboration_context = {
            "participating_agents": ["test_architect", "code_reviewer", "performance_analyst"],
            "shared_data": {"api_endpoints": 15, "user_base": 10000}
        }
        
        contribution = await test_architect.collaborate_on_task(task, collaboration_context)
        
        assert contribution["agent"] == "test_architect"
        assert contribution["specialization_applied"] == "test_architecture"
        assert "analysis" in contribution
        assert "recommendations" in contribution
        assert "test_strategy" in contribution

    def test_classify_testing_problem(self, test_architect):
        """Test problem classification"""
        # Test different problem types
        assert test_architect._classify_testing_problem("legacy code with no tests") == "legacy_code_testing"
        assert test_architect._classify_testing_problem("performance testing needed") == "performance_testing"
        assert test_architect._classify_testing_problem("security vulnerabilities") == "security_testing"
        assert test_architect._classify_testing_problem("API integration testing") == "integration_testing"

    def test_assess_complexity(self, test_architect):
        """Test complexity assessment"""
        # High complexity context
        high_complexity_context = {
            "codebase_size": 20000,
            "dependencies": ["dep1", "dep2", "dep3", "dep4", "dep5"],
            "has_legacy_code": True,
            "multiple_environments": True
        }
        assert test_architect._assess_complexity("complex problem", high_complexity_context) == "high"
        
        # Low complexity context
        low_complexity_context = {
            "codebase_size": 1000,
            "dependencies": ["dep1"],
            "has_legacy_code": False
        }
        assert test_architect._assess_complexity("simple problem", low_complexity_context) == "low"

    def test_recommend_test_layers(self, test_architect):
        """Test test layer recommendations"""
        context = {"api_heavy": True}
        layers = test_architect._recommend_test_layers(context)
        
        assert "unit_tests" in layers
        assert "integration_tests" in layers
        assert "end_to_end_tests" in layers
        assert "api_tests" in layers  # Should be added for API-heavy projects
        
        # Check priorities
        assert layers["unit_tests"]["priority"] == "high"
        assert layers["api_tests"]["priority"] == "high"

    @pytest.mark.asyncio
    async def test_capability_updates(self, test_architect):
        """Test capability update mechanism"""
        initial_count = None
        for cap in test_architect.capabilities:
            if cap.name == "test_strategy_design":
                initial_count = cap.experience_count
                break
        
        # Simulate successful capability usage
        await test_architect.update_capability("test_strategy_design", True, 2.5)
        
        for cap in test_architect.capabilities:
            if cap.name == "test_strategy_design":
                assert cap.experience_count == initial_count + 1
                assert cap.last_used is not None
                break

    def test_specialist_profile(self, test_architect):
        """Test specialist profile generation"""
        profile = test_architect.get_specialist_profile()
        
        assert profile.agent_name == "test_architect"
        assert profile.specialization == "Test Architecture & Strategy"
        assert len(profile.capabilities) > 0
        assert len(profile.expertise_domains) > 0
        assert profile.availability_status == "available"

    @pytest.mark.asyncio
    async def test_suggest_improvements(self, test_architect):
        """Test improvement suggestions"""
        # Simulate poor performance
        test_architect.performance_metrics["consultation_success_rate"] = 0.6
        test_architect.performance_metrics["response_time_avg"] = 15.0
        
        suggestions = await test_architect.suggest_improvements()
        
        assert len(suggestions) > 0
        suggestion_text = " ".join(suggestions).lower()
        assert "consultation" in suggestion_text or "response time" in suggestion_text
