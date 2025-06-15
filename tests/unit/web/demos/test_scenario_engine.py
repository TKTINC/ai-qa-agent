"""
Tests for Demo Scenario Engine
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch

from src.web.demos.scenario_engine import (
    DemoOrchestrator, ScenarioLibrary, NarrativeGenerator,
    DemoType, AudienceType, DemoStep, DemoScenario, DemoExecution
)

class TestScenarioLibrary:
    """Test scenario library functionality"""
    
    @pytest.fixture
    def scenario_library(self):
        return ScenarioLibrary()
    
    def test_scenario_initialization(self, scenario_library):
        """Test that scenarios are properly initialized"""
        scenarios = scenario_library.list_scenarios()
        
        assert len(scenarios) >= 4  # At least 4 demo types
        assert any(s.demo_type == DemoType.LEGACY_RESCUE for s in scenarios)
        assert any(s.demo_type == DemoType.DEBUGGING_SESSION for s in scenarios)
        assert any(s.demo_type == DemoType.AI_TEACHING for s in scenarios)
        assert any(s.demo_type == DemoType.CUSTOM_EXPLORATION for s in scenarios)
    
    def test_get_scenario(self, scenario_library):
        """Test getting specific scenario"""
        scenario = scenario_library.get_scenario(DemoType.LEGACY_RESCUE.value)
        
        assert scenario is not None
        assert scenario.scenario_id == "legacy_rescue_mission"
        assert scenario.title == "Legacy Code Rescue Mission"
        assert len(scenario.steps) > 0
        assert len(scenario.learning_objectives) > 0
    
    def test_legacy_rescue_scenario_structure(self, scenario_library):
        """Test legacy rescue scenario has proper structure"""
        scenario = scenario_library.get_scenario(DemoType.LEGACY_RESCUE.value)
        
        assert scenario.duration_minutes == 8
        assert scenario.complexity_level == "Intermediate"
        assert scenario.target_audience == AudienceType.TECHNICAL
        
        # Check that steps have required fields
        for step in scenario.steps:
            assert step.step_number > 0
            assert step.title
            assert step.user_input
            assert step.expected_response
            assert step.agent_reasoning
            assert isinstance(step.tools_used, list)
            assert isinstance(step.collaboration_agents, list)
            assert isinstance(step.learning_points, list)

class TestDemoOrchestrator:
    """Test demo orchestrator functionality"""
    
    @pytest.fixture
    def demo_orchestrator(self):
        return DemoOrchestrator()
    
    @pytest.mark.asyncio
    async def test_start_demo(self, demo_orchestrator):
        """Test starting a demo"""
        session_id = "test_session_123"
        demo_type = DemoType.LEGACY_RESCUE.value
        audience_type = AudienceType.TECHNICAL.value
        
        demo_execution = await demo_orchestrator.start_demo(demo_type, audience_type, session_id)
        
        assert isinstance(demo_execution, DemoExecution)
        assert demo_execution.session_id == session_id
        assert demo_execution.current_step == 0
        assert demo_execution.scenario.demo_type == DemoType.LEGACY_RESCUE
        assert session_id in demo_orchestrator.active_demos
    
    @pytest.mark.asyncio
    async def test_execute_demo_step(self, demo_orchestrator):
        """Test executing a demo step"""
        session_id = "test_session_456"
        
        # Start demo first
        await demo_orchestrator.start_demo(
            DemoType.LEGACY_RESCUE.value, 
            AudienceType.TECHNICAL.value, 
            session_id
        )
        
        # Execute first step
        result = await demo_orchestrator.execute_demo_step(session_id)
        
        assert result["status"] == "success"
        assert "step_info" in result
        assert "response" in result
        assert "interaction" in result
        assert "progress" in result
        
        # Check progress
        progress = result["progress"]
        assert progress["current_step"] == 1
        assert progress["completion_percentage"] > 0
    
    @pytest.mark.asyncio
    async def test_get_demo_status(self, demo_orchestrator):
        """Test getting demo status"""
        session_id = "test_session_789"
        
        # Start demo
        await demo_orchestrator.start_demo(
            DemoType.AI_TEACHING.value,
            AudienceType.EDUCATIONAL.value,
            session_id
        )
        
        # Get status
        status = await demo_orchestrator.get_demo_status(session_id)
        
        assert status["status"] == "active"
        assert "scenario" in status
        assert "progress" in status
        assert status["scenario"]["title"] == "AI Testing Tutor"
    
    @pytest.mark.asyncio
    async def test_end_demo(self, demo_orchestrator):
        """Test ending a demo"""
        session_id = "test_session_end"
        
        # Start and execute some steps
        await demo_orchestrator.start_demo(
            DemoType.DEBUGGING_SESSION.value,
            AudienceType.TECHNICAL.value,
            session_id
        )
        
        await demo_orchestrator.execute_demo_step(session_id)
        await demo_orchestrator.execute_demo_step(session_id)
        
        # End demo
        summary = await demo_orchestrator.end_demo(session_id)
        
        assert summary["status"] == "completed"
        assert "scenario_title" in summary
        assert "total_duration_minutes" in summary
        assert "steps_completed" in summary
        assert "learning_objectives_covered" in summary
        assert session_id not in demo_orchestrator.active_demos

class TestNarrativeGenerator:
    """Test narrative generator functionality"""
    
    @pytest.fixture
    def narrative_generator(self):
        return NarrativeGenerator()
    
    @pytest.fixture
    def sample_scenario(self):
        return DemoScenario(
            scenario_id="test_scenario",
            title="Test Scenario",
            description="A test scenario",
            demo_type=DemoType.LEGACY_RESCUE,
            target_audience=AudienceType.TECHNICAL,
            duration_minutes=5,
            complexity_level="Test",
            learning_objectives=["Test objective 1", "Test objective 2"],
            steps=[],
            setup_requirements={},
            success_metrics={}
        )
    
    @pytest.mark.asyncio
    async def test_create_business_narrative(self, narrative_generator, sample_scenario):
        """Test creating business-focused narrative"""
        narrative = await narrative_generator.create_demo_narrative(
            sample_scenario, 
            AudienceType.BUSINESS
        )
        
        assert "opening" in narrative
        assert "value_proposition" in narrative
        assert "roi_focus" in narrative
        assert "success_metrics" in narrative
        
        # Business narrative should focus on ROI and efficiency
        assert "ROI" in narrative["value_proposition"] or "efficiency" in narrative["value_proposition"]
    
    @pytest.mark.asyncio
    async def test_create_technical_narrative(self, narrative_generator, sample_scenario):
        """Test creating technical narrative"""
        narrative = await narrative_generator.create_demo_narrative(
            sample_scenario,
            AudienceType.TECHNICAL
        )
        
        assert "opening" in narrative
        assert "technical_focus" in narrative
        assert "innovation_highlights" in narrative
        assert "learning_outcomes" in narrative
        
        # Technical narrative should focus on implementation details
        assert "technical" in narrative["technical_focus"].lower()
    
    @pytest.mark.asyncio
    async def test_create_educational_narrative(self, narrative_generator, sample_scenario):
        """Test creating educational narrative"""
        narrative = await narrative_generator.create_demo_narrative(
            sample_scenario,
            AudienceType.EDUCATIONAL
        )
        
        assert "opening" in narrative
        assert "learning_approach" in narrative
        assert "skill_development" in narrative
        assert "confidence_building" in narrative
        
        # Educational narrative should focus on learning
        assert "learning" in narrative["learning_approach"].lower()
    
    @pytest.mark.asyncio
    async def test_create_executive_narrative(self, narrative_generator, sample_scenario):
        """Test creating executive narrative"""
        narrative = await narrative_generator.create_demo_narrative(
            sample_scenario,
            AudienceType.EXECUTIVE
        )
        
        assert "opening" in narrative
        assert "strategic_value" in narrative
        assert "competitive_advantage" in narrative
        assert "business_impact" in narrative
        
        # Executive narrative should focus on strategic value
        assert "strategic" in narrative["strategic_value"].lower()

if __name__ == "__main__":
    pytest.main([__file__])
