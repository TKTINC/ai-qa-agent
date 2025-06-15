"""
Demo Scenario Engine
Orchestrates compelling demonstration scenarios that showcase the full capabilities
of the AI agent system through realistic, engaging use cases.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging
from dataclasses import dataclass, asdict
from enum import Enum
import random

from src.web.services.agent_visualization import agent_visualization_service

# Import agent components with fallbacks
try:
    from src.agent.orchestrator import QAAgentOrchestrator
    from src.agent.multi_agent.agent_system import QAAgentSystem
except ImportError:
    class MockQAAgentOrchestrator:
        async def process_user_request(self, user_input: str, session_id: str) -> str:
            return f"Mock response to: {user_input}"
    
    class MockQAAgentSystem:
        async def handle_complex_request(self, user_request: str, session_id: str) -> dict:
            return {"response": f"Mock multi-agent response to: {user_request}"}
    
    QAAgentOrchestrator = MockQAAgentOrchestrator
    QAAgentSystem = MockQAAgentSystem

logger = logging.getLogger(__name__)

class DemoType(Enum):
    LEGACY_RESCUE = "legacy_rescue"
    DEBUGGING_SESSION = "debugging_session"
    AI_TEACHING = "ai_teaching"
    CUSTOM_EXPLORATION = "custom_exploration"

class AudienceType(Enum):
    TECHNICAL = "technical"
    BUSINESS = "business"
    EDUCATIONAL = "educational"
    EXECUTIVE = "executive"

@dataclass
class DemoStep:
    """Individual step in a demo scenario"""
    step_number: int
    title: str
    description: str
    user_input: str
    expected_response: str
    agent_reasoning: str
    tools_used: List[str]
    collaboration_agents: List[str]
    duration_seconds: int
    learning_points: List[str]

@dataclass
class DemoScenario:
    """Complete demo scenario definition"""
    scenario_id: str
    title: str
    description: str
    demo_type: DemoType
    target_audience: AudienceType
    duration_minutes: int
    complexity_level: str
    learning_objectives: List[str]
    steps: List[DemoStep]
    setup_requirements: Dict[str, Any]
    success_metrics: Dict[str, Any]

@dataclass
class DemoExecution:
    """Runtime demo execution state"""
    scenario: DemoScenario
    current_step: int
    start_time: datetime
    session_id: str
    audience_interactions: List[Dict[str, Any]]
    performance_metrics: Dict[str, Any]
    real_time_feedback: List[str]

class ScenarioLibrary:
    """Library of pre-built demo scenarios"""
    
    def __init__(self):
        self.scenarios = {}
        self._initialize_scenarios()
    
    def _initialize_scenarios(self):
        """Initialize all demo scenarios"""
        self.scenarios[DemoType.LEGACY_RESCUE.value] = self._create_legacy_rescue_scenario()
        self.scenarios[DemoType.DEBUGGING_SESSION.value] = self._create_debugging_scenario()
        self.scenarios[DemoType.AI_TEACHING.value] = self._create_teaching_scenario()
        self.scenarios[DemoType.CUSTOM_EXPLORATION.value] = self._create_custom_scenario()
    
    def _create_legacy_rescue_scenario(self) -> DemoScenario:
        """Create the Legacy Code Rescue Mission scenario"""
        
        steps = [
            DemoStep(
                step_number=1,
                title="User Presents Challenge",
                description="User describes inherited legacy codebase problem",
                user_input="I inherited this 10,000-line Python codebase with zero tests. The previous developer left no documentation. I need to add comprehensive testing before we can make any changes. Where do I even start?",
                expected_response="I understand this is a challenging situation! Let me help you create a strategic approach to testing this legacy codebase. I'll analyze the code structure, identify critical components, and prioritize testing efforts for maximum impact.",
                agent_reasoning="User has a complex legacy code problem requiring strategic planning, risk assessment, and systematic approach. Need to engage Test Architect for strategy and Code Reviewer for quality assessment.",
                tools_used=["repository_analyzer", "complexity_analyzer", "dependency_mapper"],
                collaboration_agents=["Test Architect", "Code Reviewer"],
                duration_seconds=45,
                learning_points=[
                    "Legacy code requires strategic, not random, testing approach",
                    "AI can quickly analyze and prioritize 10,000+ lines of code",
                    "Multi-agent collaboration provides comprehensive perspective"
                ]
            ),
            DemoStep(
                step_number=2,
                title="Intelligent Code Analysis",
                description="Agents analyze codebase structure and identify critical components",
                user_input="Here's the repository structure. Please analyze it and tell me what you find.",
                expected_response="I've analyzed your codebase and found several key insights:\n\n🔍 **Architecture Analysis:**\n- 3 core modules with high interconnectivity\n- User authentication system (critical - needs testing first)\n- Payment processing module (high risk)\n- 15 utility functions used across multiple modules\n\n📊 **Complexity Hotspots:**\n- `UserManager.authenticate()` - Complexity: 12 (very high)\n- `PaymentProcessor.process_payment()` - Complexity: 8 (high)\n- `DataValidator.validate_input()` - Complexity: 6 (medium-high)\n\n🎯 **Recommended Testing Priority:**\n1. Authentication system (security critical)\n2. Payment processing (business critical)\n3. Core utility functions (widespread impact)",
                agent_reasoning="Test Architect identifies critical paths, Code Reviewer assesses quality issues, analysis tools provide complexity metrics. Prioritizing by risk and business impact.",
                tools_used=["ast_parser", "graph_analyzer", "pattern_detector"],
                collaboration_agents=["Test Architect", "Code Reviewer", "Security Specialist"],
                duration_seconds=90,
                learning_points=[
                    "AI can rapidly analyze large codebases for testing priorities",
                    "Security and business critical components identified automatically",
                    "Complexity metrics guide testing effort allocation"
                ]
            ),
            DemoStep(
                step_number=3,
                title="Strategic Test Planning",
                description="Agents collaborate to create comprehensive testing strategy",
                user_input="That analysis is incredibly helpful! Can you create a detailed testing plan that my team can follow?",
                expected_response="Absolutely! I'll create a comprehensive 4-week testing strategy:\n\n📋 **Week 1: Foundation & Critical Path**\n- Set up testing infrastructure (pytest, coverage tools)\n- Test authentication system (15 test cases)\n- Create integration tests for login flow\n- Target: 80% coverage on auth module\n\n📋 **Week 2: Business Critical Components**\n- Payment processing comprehensive testing\n- Edge cases and error scenarios\n- Security testing for financial data\n- Target: 90% coverage on payment module\n\n📋 **Week 3: Core Utilities & Integration**\n- Test all utility functions\n- Cross-module integration testing\n- Performance testing for critical paths\n- Target: 75% overall codebase coverage\n\n📋 **Week 4: Edge Cases & Refinement**\n- Stress testing and boundary conditions\n- User scenario testing\n- Documentation and test maintenance setup\n- Target: 85% overall coverage with quality metrics\n\n🎯 **Success Metrics:**\n- Code coverage: 85%+\n- Critical path coverage: 95%+\n- Zero high-severity security gaps\n- Documented test procedures for team",
                agent_reasoning="Test Architect creates strategic timeline, Security Specialist ensures security coverage, Performance Analyst adds performance considerations. Collaboration produces comprehensive plan.",
                tools_used=["test_strategy_planner", "coverage_analyzer", "timeline_generator"],
                collaboration_agents=["Test Architect", "Security Specialist", "Performance Analyst"],
                duration_seconds=120,
                learning_points=[
                    "AI creates actionable, timeline-based testing strategies",
                    "Multi-agent collaboration ensures comprehensive coverage",
                    "Strategy balances thoroughness with practical timeline constraints"
                ]
            ),
            DemoStep(
                step_number=4,
                title="Automated Test Generation",
                description="Agents generate actual test code for immediate use",
                user_input="This plan looks great! Can you generate some actual test code to get us started?",
                expected_response="I'll generate comprehensive test code for your authentication system to get you started immediately:\n\n```python\n# test_authentication.py\nimport pytest\nfrom unittest.mock import Mock, patch\nfrom src.auth.user_manager import UserManager\nfrom src.auth.exceptions import AuthenticationError\n\nclass TestUserAuthentication:\n    def setup_method(self):\n        self.user_manager = UserManager()\n        self.valid_user = {'username': 'testuser', 'password': 'secure123'}\n    \n    def test_valid_authentication_success(self):\n        \"\"\"Test successful authentication with valid credentials\"\"\"\n        result = self.user_manager.authenticate(\n            self.valid_user['username'], \n            self.valid_user['password']\n        )\n        assert result.is_authenticated is True\n        assert result.user_id is not None\n    \n    def test_invalid_password_fails(self):\n        \"\"\"Test authentication fails with wrong password\"\"\"\n        with pytest.raises(AuthenticationError):\n            self.user_manager.authenticate(\n                self.valid_user['username'], \n                'wrongpassword'\n            )\n    \n    # ... 12 more comprehensive test cases\n```\n\nI've also generated integration tests, security tests, and performance benchmarks. Would you like me to show you the payment processing tests next?",
                agent_reasoning="Test generation tool creates realistic, comprehensive test code. Code Reviewer ensures quality. Security Specialist adds security test cases. Generated code is production-ready.",
                tools_used=["test_generator", "code_quality_checker", "security_test_generator"],
                collaboration_agents=["Test Architect", "Code Reviewer", "Security Specialist"],
                duration_seconds=75,
                learning_points=[
                    "AI generates production-ready test code, not just suggestions",
                    "Tests include edge cases, security scenarios, and error handling",
                    "Generated code follows best practices and is immediately usable"
                ]
            ),
            DemoStep(
                step_number=5,
                title="Success Validation & Next Steps",
                description="Agents validate approach and provide ongoing guidance",
                user_input="This is exactly what we needed! How do we maintain this quality as we continue?",
                expected_response="Excellent! I'm glad this approach works for your team. Here's how to maintain and improve your testing quality:\n\n🔄 **Continuous Improvement Process:**\n- Run daily test suite with coverage reports\n- Weekly test review with quality metrics\n- Monthly strategy updates based on code changes\n\n📈 **Quality Monitoring:**\n- Code coverage tracking (target: maintain 85%+)\n- Test execution time monitoring\n- Automated quality gates in CI/CD\n\n🤝 **Team Collaboration:**\n- Test-driven development for new features\n- Peer review process for test code\n- Regular knowledge sharing sessions\n\n🔮 **Future Enhancements:**\n- Automated test generation for new code\n- AI-powered test maintenance\n- Continuous quality assessment\n\nI can continue to help your team as you expand the codebase. Just let me know when you need assistance with new modules or advanced testing scenarios!",
                agent_reasoning="Providing long-term success guidance, establishing maintenance processes, setting up continuous improvement. Ensuring user can sustain progress independently while offering ongoing support.",
                tools_used=["process_optimizer", "quality_metrics_tracker"],
                collaboration_agents=["Test Architect", "Documentation Expert"],
                duration_seconds=60,
                learning_points=[
                    "AI provides ongoing strategic guidance, not just one-time solutions",
                    "Establishes sustainable processes for long-term success",
                    "Balances immediate needs with future scalability"
                ]
            )
        ]
        
        return DemoScenario(
            scenario_id="legacy_rescue_mission",
            title="Legacy Code Rescue Mission",
            description="Watch AI agents tackle a realistic 10,000-line untested legacy codebase with strategic analysis, collaborative planning, and automated test generation",
            demo_type=DemoType.LEGACY_RESCUE,
            target_audience=AudienceType.TECHNICAL,
            duration_minutes=8,
            complexity_level="Intermediate",
            learning_objectives=[
                "Strategic approach to legacy code testing",
                "AI-powered code analysis and prioritization",
                "Multi-agent collaboration for comprehensive solutions",
                "Automated test generation with production-ready quality",
                "Sustainable testing processes for ongoing success"
            ],
            steps=steps,
            setup_requirements={
                "sample_codebase": "large_legacy_python_project",
                "complexity_data": "realistic_metrics",
                "collaboration_enabled": True
            },
            success_metrics={
                "audience_engagement": "90%+",
                "learning_objective_achievement": "85%+",
                "technical_accuracy": "95%+",
                "practical_applicability": "90%+"
            }
        )
    
    def _create_debugging_scenario(self) -> DemoScenario:
        """Create Real-Time Debugging Session scenario"""
        
        steps = [
            DemoStep(
                step_number=1,
                title="Production Crisis Alert",
                description="User reports critical production performance issue",
                user_input="We have a production emergency! Our user dashboard is loading 20+ seconds instead of under 2 seconds. Users are complaining and some are timing out. Can you help us debug this quickly?",
                expected_response="I understand this is urgent! Let me immediately engage our Performance Analyst and Code Reviewer to investigate this production issue. We'll work systematically to identify the bottleneck and provide both immediate fixes and preventive testing.",
                agent_reasoning="Production emergency requiring immediate response. Need Performance Analyst for bottleneck identification, Code Reviewer for code quality issues, and systematic debugging approach.",
                tools_used=["performance_profiler", "log_analyzer", "bottleneck_detector"],
                collaboration_agents=["Performance Analyst", "Code Reviewer"],
                duration_seconds=30,
                learning_points=[
                    "AI agents can respond immediately to production emergencies",
                    "Systematic approach prevents panic-driven debugging",
                    "Multi-agent collaboration provides comprehensive analysis"
                ]
            ),
            DemoStep(
                step_number=2,
                title="Rapid Problem Diagnosis",
                description="Agents quickly identify performance bottlenecks",
                user_input="Here are the error logs and recent code changes. What do you see?",
                expected_response="🚨 **Immediate Findings:**\n\nPerformance Analyst identified:\n- Database query in UserDashboard.load_data() executing 1,247 times per page load\n- N+1 query problem in user.get_recent_activities()\n- Missing database index on frequently queried user_activities.timestamp\n\nCode Reviewer found:\n- Recent commit added inefficient loop in activity processing\n- Caching layer bypassed in dashboard controller\n- Memory leak in session management\n\n⚡ **Immediate Action Plan:**\n1. Add database index (2-minute fix, 60% improvement)\n2. Fix N+1 query with single JOIN (5-minute fix, 80% improvement)\n3. Re-enable caching layer (1-minute fix, 40% improvement)\n\n🎯 **Expected Result:** 2-second load time restored within 10 minutes",
                agent_reasoning="Performance Analyst uses profiling tools to identify database issues, Code Reviewer analyzes recent changes for regressions. Collaboration produces ranked list of fixes by impact and effort.",
                tools_used=["query_analyzer", "performance_profiler", "code_diff_analyzer", "caching_inspector"],
                collaboration_agents=["Performance Analyst", "Code Reviewer", "Test Architect"],
                duration_seconds=90,
                learning_points=[
                    "AI rapidly diagnoses complex performance issues",
                    "Provides prioritized fixes based on impact vs effort",
                    "Identifies both immediate fixes and root causes"
                ]
            ),
            DemoStep(
                step_number=3,
                title="Automated Test Generation for Prevention",
                description="While fixing, agents generate tests to prevent future issues",
                user_input="Those fixes worked perfectly! Load time is back to 1.8 seconds. How do we prevent this from happening again?",
                expected_response="Excellent! Now let's ensure this never happens again. I'm generating comprehensive performance tests:\n\n```python\n# test_dashboard_performance.py\nimport pytest\nimport time\nfrom unittest.mock import patch\nfrom src.dashboard.user_dashboard import UserDashboard\n\nclass TestDashboardPerformance:\n    def test_dashboard_load_time_under_2_seconds(self):\n        \"\"\"Critical: Dashboard must load under 2 seconds\"\"\"\n        start_time = time.time()\n        dashboard = UserDashboard(user_id='test_user')\n        result = dashboard.load_data()\n        load_time = time.time() - start_time\n        \n        assert load_time < 2.0, f\"Dashboard load time {load_time:.2f}s exceeds 2s limit\"\n        assert result.data is not None\n    \n    def test_database_query_count_optimized(self):\n        \"\"\"Ensure N+1 query problem doesn't return\"\"\"\n        with patch('src.db.connection.execute') as mock_execute:\n            dashboard = UserDashboard(user_id='test_user')\n            dashboard.load_data()\n            \n            # Should be 3 queries max: user, activities, metadata\n            assert mock_execute.call_count <= 3, f\"Too many DB queries: {mock_execute.call_count}\"\n```\n\n🔒 **Performance Monitoring Setup:**\n- Automated performance tests in CI/CD\n- Real-time monitoring alerts for >2s load times\n- Weekly performance regression testing\n- Database query optimization validation\n\nYour production issue is resolved AND protected against future occurrences!",
                agent_reasoning="Test Architect creates performance-focused tests, Performance Analyst adds monitoring recommendations, Security Specialist ensures performance tests don't compromise security. Focus on prevention through testing.",
                tools_used=["performance_test_generator", "monitoring_setup_tool", "ci_cd_integrator"],
                collaboration_agents=["Test Architect", "Performance Analyst", "Security Specialist"],
                duration_seconds=105,
                learning_points=[
                    "AI generates preventive tests, not just reactive fixes",
                    "Performance testing integrated into development workflow",
                    "Monitoring and alerts prevent future production issues"
                ]
            )
        ]
        
        return DemoScenario(
            scenario_id="debugging_session",
            title="Real-Time Production Debugging",
            description="Experience AI agents collaborating to solve a critical production performance issue with systematic diagnosis, rapid fixes, and preventive testing",
            demo_type=DemoType.DEBUGGING_SESSION,
            target_audience=AudienceType.TECHNICAL,
            duration_minutes=6,
            complexity_level="Advanced",
            learning_objectives=[
                "Emergency response protocols for production issues",
                "Systematic debugging methodology",
                "Performance bottleneck identification and resolution",
                "Preventive testing for performance regression",
                "Real-time multi-agent collaboration under pressure"
            ],
            steps=steps,
            setup_requirements={
                "production_scenario": "performance_degradation",
                "monitoring_data": "realistic_logs",
                "time_pressure": True
            },
            success_metrics={
                "problem_resolution_time": "<10 minutes",
                "audience_engagement": "95%+",
                "technical_accuracy": "98%+",
                "preventive_value": "90%+"
            }
        )
    
    def _create_teaching_scenario(self) -> DemoScenario:
        """Create AI Teaching Assistant scenario"""
        
        steps = [
            DemoStep(
                step_number=1,
                title="Learning Request",
                description="User asks for personalized testing education",
                user_input="I'm a junior developer and I'm really struggling with testing. I know I should write tests, but I don't know where to start, what to test, or how to write good tests. Can you teach me while we work on my actual code?",
                expected_response="I'd love to help you learn testing! Let's start with your actual code so you can learn through hands-on practice. Testing becomes much clearer when you see it applied to real problems you're solving.\n\nFirst, can you show me a function or class you're working on? We'll start simple and build your confidence step by step. I'll explain not just HOW to write tests, but WHY each test matters and how it helps you write better code.",
                agent_reasoning="Educational mode activated. Need to adapt communication style for junior developer, provide hands-on learning, explain concepts clearly, build confidence through success.",
                tools_used=["educational_adapter", "skill_level_assessor", "learning_path_generator"],
                collaboration_agents=["Test Architect", "Documentation Expert"],
                duration_seconds=45,
                learning_points=[
                    "AI adapts teaching style to user's experience level",
                    "Hands-on learning with real code is more effective",
                    "Building confidence through incremental success"
                ]
            ),
            DemoStep(
                step_number=2,
                title="Interactive Code Analysis",
                description="Agent teaches testing concepts through user's actual code",
                user_input="Here's a simple calculator function I wrote. How would I test this?\n\n```python\ndef calculate_discount(price, discount_percent):\n    if discount_percent > 100:\n        return 0\n    discount = price * (discount_percent / 100)\n    return price - discount\n```",
                expected_response="Perfect! This is a great function to learn testing with. Let me teach you by walking through how I think about testing:\n\n🎯 **Step 1: What should this function do?**\n- Take a price and discount percentage\n- Return the discounted price\n- Handle edge cases (like >100% discount)\n\n🧪 **Step 2: What scenarios should we test?**\n- Normal case: $100 with 20% discount = $80\n- Edge case: 0% discount = original price\n- Edge case: 100% discount = $0\n- Error case: >100% discount (your code returns 0)\n- Edge case: negative numbers?\n\n📝 **Step 3: Let's write your first test together:**\n\n```python\ndef test_normal_discount():\n    # Arrange: Set up your test data\n    price = 100\n    discount_percent = 20\n    \n    # Act: Call the function\n    result = calculate_discount(price, discount_percent)\n    \n    # Assert: Check if it's correct\n    assert result == 80\n```\n\n🎉 **You just learned the Arrange-Act-Assert pattern!** This is the foundation of all good tests. Ready to try writing the next test yourself?",
                agent_reasoning="Teaching through concrete example, introducing AAA pattern, encouraging active participation. Building understanding through guided discovery rather than lecturing.",
                tools_used=["code_analyzer", "test_pattern_explainer", "interactive_tutor"],
                collaboration_agents=["Test Architect", "Documentation Expert"],
                duration_seconds=120,
                learning_points=[
                    "Testing concepts taught through concrete examples",
                    "Arrange-Act-Assert pattern as foundation",
                    "Interactive learning builds deeper understanding"
                ]
            ),
            DemoStep(
                step_number=3,
                title="Guided Practice",
                description="User practices with AI guidance and feedback",
                user_input="Let me try writing a test for the edge case where discount is 0%:\n\n```python\ndef test_zero_discount():\n    price = 50\n    discount_percent = 0\n    result = calculate_discount(price, discount_percent)\n    assert result == 50\n```",
                expected_response="🎉 **Excellent work!** You just wrote a perfect test! Let me highlight what you did right:\n\n✅ **Great Test Structure:**\n- Clear, descriptive test name (`test_zero_discount`)\n- Used Arrange-Act-Assert pattern correctly\n- Tested an important edge case\n- Simple, focused test that checks one thing\n\n🎯 **Learning Moment:**\nNotice how this test would catch a bug if someone accidentally wrote `return price * discount_percent` instead of the correct calculation. Edge cases like 0% often reveal bugs!\n\n📚 **Next Level Concepts:**\nNow that you understand basic testing, let's explore:\n1. **Test-Driven Development (TDD):** Write tests BEFORE code\n2. **Testing negative cases:** What if someone passes negative numbers?\n3. **Error handling:** Should your function raise exceptions for invalid inputs?\n\nWant to try writing a test for negative discount percentages? What do you think should happen?",
                agent_reasoning="Providing positive reinforcement, highlighting what was done correctly, introducing next-level concepts gradually. Encouraging critical thinking about edge cases and error handling.",
                tools_used=["feedback_generator", "concept_progressor", "critical_thinking_prompter"],
                collaboration_agents=["Test Architect", "Documentation Expert"],
                duration_seconds=90,
                learning_points=[
                    "Positive reinforcement builds confidence and motivation",
                    "Highlighting correct patterns reinforces learning",
                    "Gradual introduction of advanced concepts"
                ]
            ),
            DemoStep(
                step_number=4,
                title="Advanced Concepts Integration",
                description="Agent introduces more sophisticated testing concepts",
                user_input="I think negative discount percentages should raise an exception. Here's my test:\n\n```python\ndef test_negative_discount_raises_error():\n    with pytest.raises(ValueError):\n        calculate_discount(100, -10)\n```",
                expected_response="🚀 **Outstanding!** You just leveled up to advanced testing concepts:\n\n✅ **You discovered:**\n- Exception testing with `pytest.raises()`\n- Input validation is important for robust code\n- Tests can drive better API design\n\n🔄 **Test-Driven Development in Action:**\nYour test is actually pointing out that your original function needs improvement! This is TDD - tests drive code improvement:\n\n```python\ndef calculate_discount(price, discount_percent):\n    if discount_percent < 0:\n        raise ValueError(\"Discount percentage cannot be negative\")\n    if discount_percent > 100:\n        return 0\n    discount = price * (discount_percent / 100)\n    return price - discount\n```\n\n🎓 **You've now learned:**\n1. ✅ Basic test structure (AAA pattern)\n2. ✅ Edge case testing\n3. ✅ Exception testing\n4. ✅ How tests improve code design\n5. ✅ Test-driven development mindset\n\n🎯 **Ready for Real Projects:**\nYou now have the foundation to test any function! Would you like to practice with a more complex example, or shall we explore integration testing where multiple functions work together?",
                agent_reasoning="Celebrating advanced learning achievement, connecting concepts to real-world development practices, offering progression paths. User has demonstrated mastery of fundamentals.",
                tools_used=["mastery_assessor", "concept_connector", "progression_planner"],
                collaboration_agents=["Test Architect", "Documentation Expert"],
                duration_seconds=105,
                learning_points=[
                    "Tests drive better code design and API decisions",
                    "Exception testing is crucial for robust applications",
                    "TDD mindset: tests first, implementation second"
                ]
            )
        ]
        
        return DemoScenario(
            scenario_id="ai_teaching_assistant",
            title="AI Testing Tutor",
            description="Experience personalized, adaptive learning where AI teaches testing concepts through hands-on practice with the user's actual code",
            demo_type=DemoType.AI_TEACHING,
            target_audience=AudienceType.EDUCATIONAL,
            duration_minutes=10,
            complexity_level="Beginner",
            learning_objectives=[
                "Understanding fundamental testing concepts",
                "Arrange-Act-Assert pattern mastery",
                "Edge case identification and testing",
                "Exception testing and error handling",
                "Test-driven development introduction",
                "Building confidence through hands-on practice"
            ],
            steps=steps,
            setup_requirements={
                "educational_mode": True,
                "adaptive_communication": True,
                "hands_on_learning": True
            },
            success_metrics={
                "learning_comprehension": "90%+",
                "confidence_building": "85%+",
                "practical_application": "95%+",
                "concept_retention": "88%+"
            }
        )
    
    def _create_custom_scenario(self) -> DemoScenario:
        """Create Custom Exploration scenario"""
        
        steps = [
            DemoStep(
                step_number=1,
                title="Custom Scenario Setup",
                description="User defines their specific testing challenge",
                user_input="I want to explore how your agents would handle my specific situation: [Custom user input]",
                expected_response="I'd be happy to help with your specific scenario! Let me understand your situation and engage the most appropriate agents to provide you with the best assistance.",
                agent_reasoning="Adaptive scenario that responds to user's specific needs. Route to appropriate specialist agents based on the type of challenge presented.",
                tools_used=["scenario_analyzer", "agent_router", "custom_response_generator"],
                collaboration_agents=["Dynamic based on user needs"],
                duration_seconds=60,
                learning_points=[
                    "AI agents can adapt to any testing scenario",
                    "Dynamic agent selection based on problem type",
                    "Flexible problem-solving approach"
                ]
            )
        ]
        
        return DemoScenario(
            scenario_id="custom_exploration",
            title="Custom Agent Exploration",
            description="Explore agent capabilities with your own testing scenarios and challenges",
            demo_type=DemoType.CUSTOM_EXPLORATION,
            target_audience=AudienceType.TECHNICAL,
            duration_minutes=999,  # Unlimited
            complexity_level="Any Level",
            learning_objectives=[
                "Demonstrate agent adaptability",
                "Showcase problem-solving flexibility",
                "Explore specific user interests"
            ],
            steps=steps,
            setup_requirements={
                "user_input_required": True,
                "dynamic_adaptation": True
            },
            success_metrics={
                "user_satisfaction": "90%+",
                "problem_resolution": "85%+",
                "demonstration_value": "80%+"
            }
        )
    
    def get_scenario(self, demo_type: str) -> Optional[DemoScenario]:
        """Get scenario by type"""
        return self.scenarios.get(demo_type)
    
    def list_scenarios(self) -> List[DemoScenario]:
        """List all available scenarios"""
        return list(self.scenarios.values())

class DemoOrchestrator:
    """Orchestrates demo execution and audience interaction"""
    
    def __init__(self):
        self.scenario_library = ScenarioLibrary()
        self.agent_orchestrator = QAAgentOrchestrator()
        self.agent_system = QAAgentSystem()
        self.active_demos: Dict[str, DemoExecution] = {}
    
    async def start_demo(self, demo_type: str, audience_type: str, session_id: str) -> DemoExecution:
        """Start a demo scenario"""
        scenario = self.scenario_library.get_scenario(demo_type)
        if not scenario:
            raise ValueError(f"Unknown demo type: {demo_type}")
        
        demo_execution = DemoExecution(
            scenario=scenario,
            current_step=0,
            start_time=datetime.now(),
            session_id=session_id,
            audience_interactions=[],
            performance_metrics={},
            real_time_feedback=[]
        )
        
        self.active_demos[session_id] = demo_execution
        
        logger.info(f"Started demo '{scenario.title}' for session {session_id}")
        return demo_execution
    
    async def execute_demo_step(self, session_id: str, step_number: Optional[int] = None) -> Dict[str, Any]:
        """Execute a specific demo step"""
        if session_id not in self.active_demos:
            raise ValueError(f"No active demo for session {session_id}")
        
        demo = self.active_demos[session_id]
        
        if step_number is not None:
            demo.current_step = step_number
        
        if demo.current_step >= len(demo.scenario.steps):
            return {"status": "completed", "message": "Demo completed successfully!"}
        
        current_step = demo.scenario.steps[demo.current_step]
        
        # Simulate agent response based on scenario
        if demo.scenario.demo_type == DemoType.CUSTOM_EXPLORATION:
            # For custom scenarios, use actual agent system
            agent_response = await self.agent_system.handle_complex_request(
                current_step.user_input, session_id
            )
            response_text = agent_response.get("response", current_step.expected_response)
        else:
            # For scripted scenarios, use expected responses
            response_text = current_step.expected_response
        
        # Record interaction
        interaction = {
            "step": demo.current_step + 1,
            "timestamp": datetime.now().isoformat(),
            "user_input": current_step.user_input,
            "agent_response": response_text,
            "reasoning": current_step.agent_reasoning,
            "tools_used": current_step.tools_used,
            "agents_involved": current_step.collaboration_agents,
            "learning_points": current_step.learning_points
        }
        
        demo.audience_interactions.append(interaction)
        demo.current_step += 1
        
        return {
            "status": "success",
            "step_info": asdict(current_step),
            "response": response_text,
            "interaction": interaction,
            "progress": {
                "current_step": demo.current_step,
                "total_steps": len(demo.scenario.steps),
                "completion_percentage": (demo.current_step / len(demo.scenario.steps)) * 100
            }
        }
    
    async def get_demo_status(self, session_id: str) -> Dict[str, Any]:
        """Get current demo status"""
        if session_id not in self.active_demos:
            return {"status": "not_found", "message": "No active demo found"}
        
        demo = self.active_demos[session_id]
        elapsed_time = datetime.now() - demo.start_time
        
        return {
            "status": "active",
            "scenario": {
                "title": demo.scenario.title,
                "description": demo.scenario.description,
                "duration_minutes": demo.scenario.duration_minutes,
                "complexity": demo.scenario.complexity_level
            },
            "progress": {
                "current_step": demo.current_step + 1,
                "total_steps": len(demo.scenario.steps),
                "completion_percentage": (demo.current_step / len(demo.scenario.steps)) * 100,
                "elapsed_minutes": elapsed_time.total_seconds() / 60
            },
            "learning_objectives": demo.scenario.learning_objectives,
            "interactions_count": len(demo.audience_interactions)
        }
    
    async def end_demo(self, session_id: str) -> Dict[str, Any]:
        """End demo and provide summary"""
        if session_id not in self.active_demos:
            return {"status": "not_found", "message": "No active demo found"}
        
        demo = self.active_demos[session_id]
        end_time = datetime.now()
        total_duration = end_time - demo.start_time
        
        summary = {
            "status": "completed",
            "scenario_title": demo.scenario.title,
            "total_duration_minutes": total_duration.total_seconds() / 60,
            "steps_completed": len(demo.audience_interactions),
            "learning_objectives_covered": demo.scenario.learning_objectives,
            "key_learning_points": [
                point for interaction in demo.audience_interactions 
                for point in interaction.get("learning_points", [])
            ],
            "agents_demonstrated": list(set([
                agent for interaction in demo.audience_interactions
                for agent in interaction.get("agents_involved", [])
            ])),
            "tools_showcased": list(set([
                tool for interaction in demo.audience_interactions
                for tool in interaction.get("tools_used", [])
            ]))
        }
        
        # Clean up active demo
        del self.active_demos[session_id]
        
        return summary

class NarrativeGenerator:
    """Generates compelling narratives for demo scenarios"""
    
    def __init__(self):
        pass
    
    async def create_demo_narrative(self, scenario: DemoScenario, audience: AudienceType) -> Dict[str, Any]:
        """Create engaging storyline for demo scenario"""
        
        if audience == AudienceType.BUSINESS:
            return self._create_business_narrative(scenario)
        elif audience == AudienceType.TECHNICAL:
            return self._create_technical_narrative(scenario)
        elif audience == AudienceType.EDUCATIONAL:
            return self._create_educational_narrative(scenario)
        elif audience == AudienceType.EXECUTIVE:
            return self._create_executive_narrative(scenario)
        else:
            return self._create_general_narrative(scenario)
    
    def _create_business_narrative(self, scenario: DemoScenario) -> Dict[str, Any]:
        """Business-focused narrative emphasizing ROI and efficiency"""
        return {
            "opening": f"Today we'll demonstrate how AI agents can solve real business challenges. This {scenario.title} scenario shows measurable improvements in development efficiency and code quality.",
            "value_proposition": "See how AI agents reduce manual testing time by 40-60% while improving quality and reducing production bugs.",
            "roi_focus": "Focus on cost savings, faster delivery, and risk reduction through intelligent automation.",
            "success_metrics": ["Development speed increase", "Quality improvement", "Cost reduction", "Risk mitigation"]
        }
    
    def _create_technical_narrative(self, scenario: DemoScenario) -> Dict[str, Any]:
        """Technical narrative focusing on implementation and capabilities"""
        return {
            "opening": f"Let's explore the technical sophistication of our AI agent system through {scenario.title}. You'll see advanced reasoning, collaboration, and intelligent tool usage.",
            "technical_focus": "Deep-dive into agent reasoning, ReAct patterns, multi-agent collaboration, and intelligent tool orchestration.",
            "innovation_highlights": "Novel application of AI to software testing challenges with production-ready results.",
            "learning_outcomes": scenario.learning_objectives
        }
    
    def _create_educational_narrative(self, scenario: DemoScenario) -> Dict[str, Any]:
        """Educational narrative for learning and skill development"""
        return {
            "opening": f"Welcome to an interactive learning experience with {scenario.title}. We'll learn testing concepts through hands-on practice with AI guidance.",
            "learning_approach": "Adaptive, personalized learning that builds understanding through practical application.",
            "skill_development": "Progressive skill building from basics to advanced concepts with immediate feedback.",
            "confidence_building": "Learn through success and guided discovery rather than trial and error."
        }
    
    def _create_executive_narrative(self, scenario: DemoScenario) -> Dict[str, Any]:
        """Executive narrative focusing on strategic value and competitive advantage"""
        return {
            "opening": f"This {scenario.title} demonstration showcases how AI agents provide strategic competitive advantages in software development.",
            "strategic_value": "Demonstrates innovation leadership, talent attraction, and operational excellence through AI adoption.",
            "competitive_advantage": "First-mover advantage in AI-powered development tools and quality assurance.",
            "business_impact": "Measurable improvements in speed, quality, and developer productivity."
        }
    
    def _create_general_narrative(self, scenario: DemoScenario) -> Dict[str, Any]:
        """General narrative suitable for mixed audiences"""
        return {
            "opening": f"Experience the power of AI agents through {scenario.title}. See how artificial intelligence can solve real-world software testing challenges.",
            "universal_appeal": "Demonstrates practical AI application with immediate, tangible benefits.",
            "accessibility": "Complex AI made accessible and understandable through practical demonstration.",
            "broad_impact": "Applicable across industries, team sizes, and technical backgrounds."
        }

# Global demo engine instance
demo_scenario_engine = DemoOrchestrator()
narrative_generator = NarrativeGenerator()
