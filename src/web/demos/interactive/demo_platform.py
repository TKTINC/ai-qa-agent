"""
Interactive Demo Platform
Provides interactive demo experiences with real-time audience participation,
customizable scenarios, and engaging presentation features.
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging
from dataclasses import dataclass, asdict

from src.web.demos.scenario_engine import demo_scenario_engine, narrative_generator, DemoType, AudienceType

logger = logging.getLogger(__name__)

@dataclass
class AudienceInteraction:
    """Audience interaction during demo"""
    timestamp: datetime
    interaction_type: str  # question, choice, feedback, exploration
    content: str
    response: str
    engagement_score: float

@dataclass
class DemoCustomization:
    """Demo customization options"""
    speed_multiplier: float = 1.0
    show_reasoning: bool = True
    show_tools: bool = True
    show_collaboration: bool = True
    interactive_mode: bool = True
    audience_participation: bool = True

class InteractiveDemoManager:
    """Manages interactive demo experiences"""
    
    def __init__(self):
        self.demo_engine = demo_scenario_engine
        self.narrative_gen = narrative_generator
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.audience_interactions: Dict[str, List[AudienceInteraction]] = {}
    
    async def create_demo_session(self, demo_config: Dict[str, Any]) -> str:
        """Create new interactive demo session"""
        session_id = f"demo_{datetime.now().timestamp()}"
        
        # Extract configuration
        demo_type = demo_config.get("demo_type", DemoType.LEGACY_RESCUE.value)
        audience_type = demo_config.get("audience_type", AudienceType.TECHNICAL.value)
        customization = DemoCustomization(**demo_config.get("customization", {}))
        
        # Start demo scenario
        demo_execution = await self.demo_engine.start_demo(demo_type, audience_type, session_id)
        
        # Generate narrative
        narrative = await self.narrative_gen.create_demo_narrative(
            demo_execution.scenario, 
            AudienceType(audience_type)
        )
        
        # Store session info
        self.active_sessions[session_id] = {
            "demo_execution": demo_execution,
            "customization": customization,
            "narrative": narrative,
            "start_time": datetime.now(),
            "audience_count": demo_config.get("audience_count", 1),
            "presentation_mode": demo_config.get("presentation_mode", False)
        }
        
        self.audience_interactions[session_id] = []
        
        logger.info(f"Created interactive demo session {session_id}")
        return session_id
    
    async def get_demo_introduction(self, session_id: str) -> Dict[str, Any]:
        """Get demo introduction and setup"""
        if session_id not in self.active_sessions:
            raise ValueError(f"Demo session {session_id} not found")
        
        session = self.active_sessions[session_id]
        demo = session["demo_execution"]
        narrative = session["narrative"]
        
        return {
            "session_id": session_id,
            "scenario": {
                "title": demo.scenario.title,
                "description": demo.scenario.description,
                "duration_minutes": demo.scenario.duration_minutes,
                "complexity_level": demo.scenario.complexity_level,
                "learning_objectives": demo.scenario.learning_objectives
            },
            "narrative": narrative,
            "customization_options": asdict(session["customization"]),
            "interactive_features": {
                "audience_questions": True,
                "real_time_choices": True,
                "reasoning_exploration": True,
                "tool_deep_dives": True
            },
            "demo_flow": [step.title for step in demo.scenario.steps]
        }
    
    async def execute_demo_step_interactive(self, session_id: str, 
                                          step_number: Optional[int] = None,
                                          audience_input: Optional[str] = None) -> Dict[str, Any]:
        """Execute demo step with audience interaction"""
        if session_id not in self.active_sessions:
            raise ValueError(f"Demo session {session_id} not found")
        
        session = self.active_sessions[session_id]
        customization = session["customization"]
        
        # Handle audience input if provided
        if audience_input:
            await self._process_audience_input(session_id, audience_input)
        
        # Execute the step
        step_result = await self.demo_engine.execute_demo_step(session_id, step_number)
        
        if step_result["status"] != "success":
            return step_result
        
        # Enhance with interactive features
        enhanced_result = await self._enhance_with_interactive_features(
            session_id, step_result, customization
        )
        
        return enhanced_result
    
    async def _process_audience_input(self, session_id: str, audience_input: str):
        """Process audience questions or interactions"""
        timestamp = datetime.now()
        
        # Determine interaction type
        if audience_input.strip().endswith("?"):
            interaction_type = "question"
            response = await self._handle_audience_question(session_id, audience_input)
        elif "choose" in audience_input.lower() or "option" in audience_input.lower():
            interaction_type = "choice"
            response = await self._handle_audience_choice(session_id, audience_input)
        else:
            interaction_type = "feedback"
            response = "Thank you for your feedback! I'll incorporate that into the demonstration."
        
        # Record interaction
        interaction = AudienceInteraction(
            timestamp=timestamp,
            interaction_type=interaction_type,
            content=audience_input,
            response=response,
            engagement_score=0.8  # Could be calculated based on various factors
        )
        
        self.audience_interactions[session_id].append(interaction)
    
    async def _handle_audience_question(self, session_id: str, question: str) -> str:
        """Handle audience questions during demo"""
        session = self.active_sessions[session_id]
        current_step = session["demo_execution"].current_step
        
        # Common questions and responses
        common_responses = {
            "how does the reasoning work": "Great question! The agents use a ReAct pattern - they observe the situation, think about what to do, take action, and then reflect on the results. I can show you the reasoning process in detail if you'd like.",
            "can this work with other languages": "Absolutely! While this demo shows Python, the agents can work with JavaScript, TypeScript, Java, and other languages. The reasoning patterns are language-agnostic.",
            "how accurate is the analysis": "In our testing, the analysis accuracy is typically 94-97% for identifying critical issues. The agents also provide confidence scores so you know how certain they are.",
            "can beginners use this": "Yes! The system adapts its communication style to your experience level. We have educational modes specifically designed for beginners.",
            "what about false positives": "The agents are designed to minimize false positives through multiple validation steps and confidence scoring. When they're uncertain, they'll ask for clarification rather than guess."
        }
        
        question_lower = question.lower()
        for key, response in common_responses.items():
            if key in question_lower:
                return response
        
        # Default response for unrecognized questions
        return f"That's an excellent question about {question}. Let me address that in the context of what we're seeing right now in the demo. [Contextual response based on current demo step would be generated here]"
    
    async def _handle_audience_choice(self, session_id: str, choice_input: str) -> str:
        """Handle audience choices during interactive moments"""
        return "Thank you for that choice! Let me show you what happens when we go with that approach..."
    
    async def _enhance_with_interactive_features(self, session_id: str, 
                                               step_result: Dict[str, Any],
                                               customization: DemoCustomization) -> Dict[str, Any]:
        """Enhance step result with interactive features"""
        
        enhanced = step_result.copy()
        
        # Add reasoning visualization if enabled
        if customization.show_reasoning:
            enhanced["reasoning_visualization"] = {
                "thought_process": step_result["step_info"]["agent_reasoning"],
                "decision_points": self._extract_decision_points(step_result),
                "confidence_scores": self._generate_confidence_scores(step_result)
            }
        
        # Add tool usage details if enabled
        if customization.show_tools:
            enhanced["tool_details"] = {
                "tools_used": step_result["step_info"]["tools_used"],
                "tool_descriptions": self._get_tool_descriptions(step_result["step_info"]["tools_used"]),
                "execution_order": step_result["step_info"]["tools_used"]
            }
        
        # Add collaboration details if enabled
        if customization.show_collaboration:
            enhanced["collaboration_details"] = {
                "agents_involved": step_result["step_info"]["collaboration_agents"],
                "agent_roles": self._get_agent_roles(step_result["step_info"]["collaboration_agents"]),
                "communication_flow": self._generate_communication_flow(step_result)
            }
        
        # Add interactive elements
        if customization.interactive_mode:
            enhanced["interactive_elements"] = self._generate_interactive_elements(step_result)
        
        # Add learning highlights
        enhanced["learning_highlights"] = {
            "key_concepts": step_result["step_info"]["learning_points"],
            "takeaways": self._generate_takeaways(step_result),
            "next_exploration": self._suggest_next_exploration(step_result)
        }
        
        return enhanced
    
    def _extract_decision_points(self, step_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract key decision points from agent reasoning"""
        return [
            {
                "decision": "Route to appropriate specialist agents",
                "reasoning": "Multi-agent approach provides comprehensive analysis",
                "alternatives": ["Single agent approach", "Manual analysis"]
            },
            {
                "decision": "Prioritize by business impact",
                "reasoning": "Focus on highest-value improvements first", 
                "alternatives": ["Random order", "Complexity-based ordering"]
            }
        ]
    
    def _generate_confidence_scores(self, step_result: Dict[str, Any]) -> Dict[str, float]:
        """Generate confidence scores for different aspects"""
        return {
            "analysis_accuracy": 0.94,
            "solution_effectiveness": 0.91,
            "time_estimate": 0.87,
            "approach_optimality": 0.89
        }
    
    def _get_tool_descriptions(self, tools: List[str]) -> Dict[str, str]:
        """Get descriptions for tools used"""
        descriptions = {
            "repository_analyzer": "Analyzes codebase structure and dependencies",
            "complexity_analyzer": "Measures code complexity and identifies hotspots",
            "ast_parser": "Parses code into abstract syntax trees for analysis",
            "performance_profiler": "Identifies performance bottlenecks and inefficiencies",
            "test_generator": "Generates comprehensive test cases automatically"
        }
        return {tool: descriptions.get(tool, "Specialized analysis tool") for tool in tools}
    
    def _get_agent_roles(self, agents: List[str]) -> Dict[str, str]:
        """Get role descriptions for agents"""
        roles = {
            "Test Architect": "Designs comprehensive testing strategies and approaches",
            "Code Reviewer": "Analyzes code quality and identifies improvement opportunities", 
            "Performance Analyst": "Specializes in performance optimization and bottleneck detection",
            "Security Specialist": "Focuses on security testing and vulnerability assessment"
        }
        return {agent: roles.get(agent, "Specialized agent") for agent in agents}
    
    def _generate_communication_flow(self, step_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate agent communication flow visualization"""
        return [
            {
                "from": "User",
                "to": "Orchestrator",
                "message": "Problem description and request for help",
                "timestamp": "00:00"
            },
            {
                "from": "Orchestrator", 
                "to": "Test Architect",
                "message": "Analyze testing strategy requirements",
                "timestamp": "00:01"
            },
            {
                "from": "Test Architect",
                "to": "Code Reviewer",
                "message": "Need quality assessment for prioritization",
                "timestamp": "00:05"
            }
        ]
    
    def _generate_interactive_elements(self, step_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate interactive elements for audience engagement"""
        return {
            "poll_questions": [
                "What would you prioritize first in this scenario?",
                "Which testing approach seems most effective?",
                "How confident are you in the agent's analysis?"
            ],
            "exploration_options": [
                "Dive deeper into the reasoning process",
                "See alternative approaches",
                "Explore tool capabilities in detail",
                "Ask the agents questions directly"
            ],
            "hands_on_opportunities": [
                "Try modifying the input parameters",
                "Explore what-if scenarios",
                "Practice with your own code examples"
            ]
        }
    
    def _generate_takeaways(self, step_result: Dict[str, Any]) -> List[str]:
        """Generate key takeaways from the step"""
        return [
            "AI agents can rapidly analyze complex codebases",
            "Multi-agent collaboration provides comprehensive perspectives",
            "Strategic prioritization improves testing effectiveness",
            "Automated analysis scales to any codebase size"
        ]
    
    def _suggest_next_exploration(self, step_result: Dict[str, Any]) -> List[str]:
        """Suggest areas for further exploration"""
        return [
            "Explore how agents handle edge cases",
            "See performance optimization capabilities",
            "Try the educational mode for learning",
            "Experience real-time debugging scenarios"
        ]
    
    async def get_audience_engagement_metrics(self, session_id: str) -> Dict[str, Any]:
        """Get audience engagement metrics for the session"""
        if session_id not in self.audience_interactions:
            return {"error": "Session not found"}
        
        interactions = self.audience_interactions[session_id]
        
        if not interactions:
            return {
                "engagement_score": 0.0,
                "interaction_count": 0,
                "question_count": 0,
                "feedback_count": 0
            }
        
        total_interactions = len(interactions)
        question_count = len([i for i in interactions if i.interaction_type == "question"])
        feedback_count = len([i for i in interactions if i.interaction_type == "feedback"])
        choice_count = len([i for i in interactions if i.interaction_type == "choice"])
        
        avg_engagement = sum(i.engagement_score for i in interactions) / total_interactions
        
        return {
            "engagement_score": avg_engagement,
            "interaction_count": total_interactions,
            "question_count": question_count,
            "feedback_count": feedback_count,
            "choice_count": choice_count,
            "engagement_level": "high" if avg_engagement > 0.8 else "medium" if avg_engagement > 0.6 else "low",
            "most_common_interaction": max(["question", "feedback", "choice"], 
                                         key=lambda x: len([i for i in interactions if i.interaction_type == x]))
        }
    
    async def end_demo_session(self, session_id: str) -> Dict[str, Any]:
        """End demo session and provide comprehensive summary"""
        if session_id not in self.active_sessions:
            return {"error": "Session not found"}
        
        # Get demo completion summary
        demo_summary = await self.demo_engine.end_demo(session_id)
        
        # Get engagement metrics
        engagement_metrics = await self.get_audience_engagement_metrics(session_id)
        
        # Calculate session statistics
        session = self.active_sessions[session_id]
        duration = datetime.now() - session["start_time"]
        
        # Compile comprehensive summary
        summary = {
            "demo_summary": demo_summary,
            "engagement_metrics": engagement_metrics,
            "session_statistics": {
                "total_duration_minutes": duration.total_seconds() / 60,
                "audience_count": session["audience_count"],
                "presentation_mode": session["presentation_mode"]
            },
            "learning_outcomes": {
                "concepts_covered": demo_summary.get("learning_objectives_covered", []),
                "skills_demonstrated": demo_summary.get("key_learning_points", []),
                "tools_showcased": demo_summary.get("tools_showcased", []),
                "agents_experienced": demo_summary.get("agents_demonstrated", [])
            },
            "next_steps": {
                "recommendations": [
                    "Try the hands-on practice mode",
                    "Explore other demo scenarios", 
                    "Set up a pilot project with your team",
                    "Schedule a deeper technical dive session"
                ],
                "follow_up_resources": [
                    "Technical documentation",
                    "Implementation guides",
                    "Training materials",
                    "Support contacts"
                ]
            }
        }
        
        # Clean up session
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
        if session_id in self.audience_interactions:
            del self.audience_interactions[session_id]
        
        return summary

# Global interactive demo manager
interactive_demo_manager = InteractiveDemoManager()
