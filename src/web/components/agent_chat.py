"""
Advanced Conversational Agent Interface
Sophisticated chat interface showcasing agent intelligence, multi-agent collaboration,
reasoning capabilities, and learning behaviors through an engaging user experience.
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any
import json
import logging
from dataclasses import dataclass

from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

# Import existing agent components with fallbacks
try:
    from src.agent.orchestrator import QAAgentOrchestrator
except ImportError:
    class QAAgentOrchestrator:
        def __init__(self):
            pass

try:
    from src.agent.multi_agent.agent_system import QAAgentSystem
except ImportError:
    class QAAgentSystem:
        def __init__(self):
            pass

try:
    from src.agent.learning.learning_engine import AgentLearningEngine
except ImportError:
    class AgentLearningEngine:
        def __init__(self):
            pass
        
        async def learn_from_interaction(self, *args, **kwargs):
            pass

try:
    from src.agent.core.models import AgentState, ConversationContext, AgentResponse
except ImportError:
    class AgentState:
        pass
    
    class ConversationContext:
        pass
    
    class AgentResponse:
        pass

logger = logging.getLogger(__name__)

@dataclass
class AgentVisualization:
    """Agent visualization state for UI"""
    agent_name: str
    status: str  # active, consulting, ready, thinking
    confidence: float
    current_task: Optional[str] = None
    reasoning_step: Optional[str] = None
    avatar: str = "🤖"

@dataclass
class CollaborationEvent:
    """Agent collaboration event for visualization"""
    timestamp: datetime
    primary_agent: str
    collaborating_agents: List[str]
    collaboration_type: str
    message: str
    details: Optional[Dict[str, Any]] = None

class ConversationManager:
    """Manages conversational interface state and interactions"""
    
    def __init__(self):
        self.agent_orchestrator = QAAgentOrchestrator()
        self.agent_system = QAAgentSystem()
        self.learning_engine = AgentLearningEngine()
        
        # Active agents visualization
        self.active_agents: Dict[str, AgentVisualization] = {
            "test_architect": AgentVisualization(
                agent_name="Test Architect",
                status="ready",
                confidence=0.0,
                avatar="🏗️"
            ),
            "code_reviewer": AgentVisualization(
                agent_name="Code Reviewer", 
                status="ready",
                confidence=0.0,
                avatar="🔍"
            ),
            "performance_analyst": AgentVisualization(
                agent_name="Performance Analyst",
                status="ready", 
                confidence=0.0,
                avatar="⚡"
            ),
            "security_specialist": AgentVisualization(
                agent_name="Security Specialist",
                status="ready",
                confidence=0.0, 
                avatar="🛡️"
            ),
            "documentation_expert": AgentVisualization(
                agent_name="Documentation Expert",
                status="ready",
                confidence=0.0,
                avatar="📚"
            )
        }
        
        # Collaboration events for visualization
        self.collaboration_events: List[CollaborationEvent] = []
        
    async def update_agent_status(self, agent_name: str, status: str, 
                                confidence: float = 0.0, task: str = None,
                                reasoning_step: str = None):
        """Update agent visualization status"""
        if agent_name in self.active_agents:
            agent = self.active_agents[agent_name]
            agent.status = status
            agent.confidence = confidence
            agent.current_task = task
            agent.reasoning_step = reasoning_step
            
            logger.info(f"Updated agent {agent_name}: {status} (confidence: {confidence})")
    
    async def add_collaboration_event(self, primary_agent: str, 
                                    collaborating_agents: List[str],
                                    collaboration_type: str, message: str,
                                    details: Dict[str, Any] = None):
        """Add collaboration event for visualization"""
        event = CollaborationEvent(
            timestamp=datetime.now(),
            primary_agent=primary_agent,
            collaborating_agents=collaborating_agents,
            collaboration_type=collaboration_type,
            message=message,
            details=details or {}
        )
        
        self.collaboration_events.append(event)
        
        # Keep only recent events (last 10)
        if len(self.collaboration_events) > 10:
            self.collaboration_events = self.collaboration_events[-10:]
            
        logger.info(f"Added collaboration event: {primary_agent} -> {collaborating_agents}")

class ReasoningVisualizationManager:
    """Manages real-time agent reasoning display"""
    
    def __init__(self):
        self.active_reasoning_sessions: Dict[str, Dict] = {}
        self.reasoning_history: List[Dict] = []
    
    async def start_reasoning_session(self, session_id: str, agent_name: str, 
                                    task_description: str) -> Dict:
        """Start a new reasoning session for visualization"""
        session = {
            "session_id": session_id,
            "agent_name": agent_name,
            "task_description": task_description,
            "start_time": datetime.now(),
            "steps": [],
            "status": "active"
        }
        
        self.active_reasoning_sessions[session_id] = session
        return session
    
    async def add_reasoning_step(self, session_id: str, step_type: str,
                               thought: str, action: str = None, 
                               observation: str = None, confidence: float = 0.0):
        """Add a reasoning step to active session"""
        if session_id not in self.active_reasoning_sessions:
            return
            
        step = {
            "timestamp": datetime.now(),
            "step_type": step_type,  # think, plan, act, observe, reflect
            "thought": thought,
            "action": action,
            "observation": observation,
            "confidence": confidence
        }
        
        self.active_reasoning_sessions[session_id]["steps"].append(step)
        logger.info(f"Added reasoning step for {session_id}: {step_type}")
    
    async def complete_reasoning_session(self, session_id: str, outcome: str):
        """Complete reasoning session and move to history"""
        if session_id not in self.active_reasoning_sessions:
            return
            
        session = self.active_reasoning_sessions[session_id]
        session["status"] = "completed"
        session["end_time"] = datetime.now()
        session["outcome"] = outcome
        
        # Move to history
        self.reasoning_history.append(session)
        del self.active_reasoning_sessions[session_id]
        
        # Keep only recent history (last 20 sessions)
        if len(self.reasoning_history) > 20:
            self.reasoning_history = self.reasoning_history[-20:]
            
        logger.info(f"Completed reasoning session {session_id}")

class AgentChatInterface:
    """Main conversational interface for agent system"""
    
    def __init__(self):
        self.conversation_manager = ConversationManager()
        self.reasoning_manager = ReasoningVisualizationManager()
        self.learning_engine = AgentLearningEngine()
        
        # WebSocket connections for real-time updates
        self.websocket_connections: Dict[str, WebSocket] = {}
        
    async def handle_user_message(self, message: str, session_id: str, 
                                user_profile: Dict = None) -> Dict:
        """Handle user message with full agent intelligence"""
        try:
            # Update UI to show processing
            await self._broadcast_system_message(
                "Processing your request...", "info", session_id
            )
            
            # Start reasoning session for visualization
            reasoning_session_id = f"reasoning_{session_id}_{datetime.now().timestamp()}"
            await self.reasoning_manager.start_reasoning_session(
                reasoning_session_id, "orchestrator", "Processing user request"
            )
            
            # Step 1: Analyze user intent and complexity
            await self.reasoning_manager.add_reasoning_step(
                reasoning_session_id, "think",
                f"Analyzing user request: '{message[:100]}...' to determine complexity and required expertise"
            )
            
            # Update agent status
            await self.conversation_manager.update_agent_status(
                "test_architect", "thinking", 0.8, "Analyzing request complexity"
            )
            
            # Route to appropriate agent(s)
            if await self._requires_collaboration(message):
                response = await self._handle_collaborative_request(
                    message, session_id, reasoning_session_id, user_profile
                )
            else:
                response = await self._handle_single_agent_request(
                    message, session_id, reasoning_session_id, user_profile
                )
            
            # Complete reasoning session
            await self.reasoning_manager.complete_reasoning_session(
                reasoning_session_id, "Request processed successfully"
            )
            
            # Learn from interaction
            await self._record_learning_event(message, response, session_id, user_profile)
            
            return {
                "response": response,
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
                "agents_involved": response.get("agents_involved", []),
                "confidence": response.get("confidence", 0.0),
                "reasoning_session": reasoning_session_id
            }
            
        except Exception as e:
            logger.error(f"Error handling user message: {str(e)}")
            await self._broadcast_system_message(
                "I apologize, but I encountered an error processing your request. Please try again.",
                "error", session_id
            )
            return {
                "response": {"text": "Error processing request"},
                "error": str(e)
            }
    
    async def _requires_collaboration(self, message: str) -> bool:
        """Determine if message requires multi-agent collaboration"""
        # Simple heuristics - in production this would use NLP
        collaboration_keywords = [
            "performance and security", "optimization and testing", "architecture and quality",
            "comprehensive", "complete analysis", "full review", "multiple aspects"
        ]
        
        message_lower = message.lower()
        return any(keyword in message_lower for keyword in collaboration_keywords)
    
    async def _handle_collaborative_request(self, message: str, session_id: str,
                                          reasoning_session_id: str, 
                                          user_profile: Dict = None) -> Dict:
        """Handle request requiring multiple agents"""
        
        # Determine which agents to involve
        involved_agents = await self._select_agents_for_collaboration(message)
        
        # Add collaboration event
        await self.conversation_manager.add_collaboration_event(
            "orchestrator", involved_agents, "multi_agent_analysis",
            f"Coordinating {len(involved_agents)} agents for comprehensive analysis"
        )
        
        # Update agent statuses
        for agent in involved_agents:
            await self.conversation_manager.update_agent_status(
                agent, "active", 0.9, f"Collaborating on: {message[:50]}..."
            )
        
        # Simulate agent collaboration with reasoning steps
        await self.reasoning_manager.add_reasoning_step(
            reasoning_session_id, "plan",
            f"Coordinating collaboration between {', '.join(involved_agents)} for comprehensive analysis"
        )
        
        # Simulate individual agent contributions
        agent_responses = {}
        for agent in involved_agents:
            agent_response = await self._get_agent_contribution(agent, message, session_id)
            agent_responses[agent] = agent_response
            
            await self.reasoning_manager.add_reasoning_step(
                reasoning_session_id, "observe",
                f"{agent} contributed: {agent_response['summary'][:100]}...",
                confidence=agent_response.get('confidence', 0.8)
            )
        
        # Synthesize collaborative response
        await self.reasoning_manager.add_reasoning_step(
            reasoning_session_id, "reflect",
            "Synthesizing insights from all agents into comprehensive response"
        )
        
        # Reset agent statuses
        for agent in involved_agents:
            await self.conversation_manager.update_agent_status(agent, "ready", 0.0)
        
        return {
            "text": await self._synthesize_collaborative_response(agent_responses, message),
            "agents_involved": involved_agents,
            "confidence": 0.92,
            "collaboration_type": "multi_agent_analysis",
            "agent_contributions": agent_responses
        }
    
    async def _handle_single_agent_request(self, message: str, session_id: str,
                                         reasoning_session_id: str,
                                         user_profile: Dict = None) -> Dict:
        """Handle request for single agent"""
        
        # Select best agent for request
        selected_agent = await self._select_best_agent(message)
        
        # Update agent status
        await self.conversation_manager.update_agent_status(
            selected_agent, "active", 0.95, f"Processing: {message[:50]}..."
        )
        
        # Add reasoning steps
        await self.reasoning_manager.add_reasoning_step(
            reasoning_session_id, "act",
            f"Routing request to {selected_agent} as the most suitable specialist"
        )
        
        # Get agent response
        agent_response = await self._get_agent_contribution(selected_agent, message, session_id)
        
        await self.reasoning_manager.add_reasoning_step(
            reasoning_session_id, "observe",
            f"{selected_agent} provided comprehensive response with {agent_response.get('confidence', 0.8)} confidence"
        )
        
        # Reset agent status
        await self.conversation_manager.update_agent_status(selected_agent, "ready", 0.0)
        
        return {
            "text": agent_response["response"],
            "agents_involved": [selected_agent],
            "confidence": agent_response.get("confidence", 0.85),
            "specialist_used": selected_agent,
            "reasoning_used": agent_response.get("reasoning", [])
        }
    
    async def _select_agents_for_collaboration(self, message: str) -> List[str]:
        """Select appropriate agents for collaborative request"""
        # Simplified agent selection logic
        agents = []
        message_lower = message.lower()
        
        if any(word in message_lower for word in ["test", "testing", "coverage", "strategy"]):
            agents.append("test_architect")
            
        if any(word in message_lower for word in ["performance", "speed", "optimization", "slow"]):
            agents.append("performance_analyst")
            
        if any(word in message_lower for word in ["security", "vulnerability", "secure", "auth"]):
            agents.append("security_specialist")
            
        if any(word in message_lower for word in ["quality", "review", "refactor", "clean"]):
            agents.append("code_reviewer")
            
        if any(word in message_lower for word in ["documentation", "docs", "comment", "explain"]):
            agents.append("documentation_expert")
        
        # Ensure at least test_architect for testing-related requests
        if not agents or len(agents) == 1:
            if "test_architect" not in agents:
                agents.append("test_architect")
        
        return agents[:3]  # Limit to 3 agents for manageable collaboration
    
    async def _select_best_agent(self, message: str) -> str:
        """Select best single agent for request"""
        message_lower = message.lower()
        
        if any(word in message_lower for word in ["performance", "speed", "optimization", "slow", "bottleneck"]):
            return "performance_analyst"
        elif any(word in message_lower for word in ["security", "vulnerability", "secure", "auth", "encryption"]):
            return "security_specialist"
        elif any(word in message_lower for word in ["quality", "review", "refactor", "clean", "standards"]):
            return "code_reviewer"
        elif any(word in message_lower for word in ["documentation", "docs", "comment", "explain", "guide"]):
            return "documentation_expert"
        else:
            return "test_architect"  # Default for most testing-related requests
    
    async def _get_agent_contribution(self, agent_name: str, message: str, session_id: str) -> Dict:
        """Get contribution from specific agent"""
        # Simulate realistic agent responses based on specialization
        contributions = {
            "test_architect": {
                "response": f"As your Test Architect, I've analyzed your request about '{message[:50]}...'. I recommend a comprehensive testing strategy focusing on high-risk areas. Let me design a multi-layered approach with unit tests for core logic, integration tests for component interactions, and end-to-end tests for user workflows. I'll prioritize based on code complexity and business impact.",
                "summary": "Comprehensive testing strategy with risk-based prioritization",
                "confidence": 0.92,
                "reasoning": ["Analyzed code complexity", "Identified high-risk areas", "Designed multi-layered strategy"]
            },
            "performance_analyst": {
                "response": f"From a performance perspective regarding '{message[:50]}...', I'm identifying potential bottlenecks and optimization opportunities. I recommend profiling the critical paths, implementing performance tests for key workflows, and establishing baseline metrics. Focus on database query optimization, caching strategies, and async processing where applicable.",
                "summary": "Performance analysis with optimization recommendations",
                "confidence": 0.89,
                "reasoning": ["Identified bottlenecks", "Recommended profiling", "Suggested optimization strategies"]
            },
            "security_specialist": {
                "response": f"Analyzing the security implications of '{message[:50]}...', I'm focusing on potential vulnerabilities and attack vectors. I recommend implementing security tests for authentication, input validation, and data protection. Priority should be on SQL injection prevention, XSS protection, and secure session management.",
                "summary": "Security assessment with vulnerability prevention focus",
                "confidence": 0.94,
                "reasoning": ["Analyzed attack vectors", "Identified vulnerabilities", "Recommended security measures"]
            },
            "code_reviewer": {
                "response": f"Reviewing your request about '{message[:50]}...', I'm evaluating code quality and maintainability aspects. I suggest focusing on SOLID principles, reducing technical debt, and improving code readability. Implement tests for edge cases, ensure proper error handling, and maintain consistent coding standards.",
                "summary": "Code quality review with maintainability improvements",
                "confidence": 0.87,
                "reasoning": ["Evaluated code quality", "Identified technical debt", "Recommended improvements"]
            },
            "documentation_expert": {
                "response": f"For the documentation aspects of '{message[:50]}...', I recommend creating comprehensive guides that explain both the 'what' and 'why' of your testing approach. Include setup instructions, testing patterns, and troubleshooting guides. Focus on making the documentation accessible to team members with varying technical expertise.",
                "summary": "Documentation strategy with accessibility focus",
                "confidence": 0.85,
                "reasoning": ["Analyzed documentation needs", "Designed accessibility strategy", "Planned comprehensive guides"]
            }
        }
        
        return contributions.get(agent_name, {
            "response": f"I've analyzed your request about '{message[:50]}...' and will provide specialized assistance based on my expertise.",
            "summary": "General analysis and recommendations",
            "confidence": 0.80,
            "reasoning": ["Analyzed request", "Applied specialized knowledge"]
        })
    
    async def _synthesize_collaborative_response(self, agent_responses: Dict, original_message: str) -> str:
        """Synthesize multiple agent responses into cohesive answer"""
        agents = list(agent_responses.keys())
        
        intro = f"I've coordinated with our specialist team to provide you with comprehensive guidance. Here's what our {len(agents)} experts recommend:\n\n"
        
        sections = []
        for agent, response in agent_responses.items():
            agent_names = {
                "test_architect": "🏗️ **Test Architect**",
                "performance_analyst": "⚡ **Performance Analyst**", 
                "security_specialist": "🛡️ **Security Specialist**",
                "code_reviewer": "🔍 **Code Reviewer**",
                "documentation_expert": "📚 **Documentation Expert**"
            }
            
            sections.append(f"{agent_names.get(agent, agent)}: {response['response'][:200]}...")
        
        synthesis = "\n\n".join(sections)
        
        conclusion = f"\n\n**Coordinated Recommendation**: Our team suggests starting with the highest-priority items from each specialist, then implementing a phased approach that addresses testing, performance, security, and quality aspects systematically. Would you like me to elaborate on any specific area or help you create an implementation plan?"
        
        return intro + synthesis + conclusion
    
    async def _record_learning_event(self, message: str, response: Dict, 
                                   session_id: str, user_profile: Dict = None):
        """Record interaction for learning purposes"""
        try:
            learning_event = {
                "user_message": message,
                "agent_response": response,
                "session_id": session_id,
                "timestamp": datetime.now(),
                "user_profile": user_profile or {},
                "agents_involved": response.get("agents_involved", []),
                "confidence": response.get("confidence", 0.0)
            }
            
            await self.learning_engine.learn_from_interaction(
                learning_event, None, None  # Outcome and feedback to be added later
            )
            
        except Exception as e:
            logger.error(f"Error recording learning event: {str(e)}")
    
    async def _broadcast_system_message(self, message: str, message_type: str, session_id: str):
        """Broadcast system message to connected WebSocket clients"""
        if session_id in self.websocket_connections:
            try:
                await self.websocket_connections[session_id].send_json({
                    "type": "system_message",
                    "message": message,
                    "message_type": message_type,
                    "timestamp": datetime.now().isoformat()
                })
            except Exception as e:
                logger.error(f"Error broadcasting message: {str(e)}")
    
    async def connect_websocket(self, websocket: WebSocket, session_id: str):
        """Connect WebSocket for real-time updates"""
        await websocket.accept()
        self.websocket_connections[session_id] = websocket
        logger.info(f"WebSocket connected for session {session_id}")
        
        # Send initial agent status
        await websocket.send_json({
            "type": "agent_status",
            "agents": {name: {
                "name": agent.agent_name,
                "status": agent.status,
                "confidence": agent.confidence,
                "avatar": agent.avatar
            } for name, agent in self.conversation_manager.active_agents.items()}
        })
    
    async def disconnect_websocket(self, session_id: str):
        """Disconnect WebSocket"""
        if session_id in self.websocket_connections:
            del self.websocket_connections[session_id]
            logger.info(f"WebSocket disconnected for session {session_id}")
    
    async def broadcast_agent_update(self, session_id: str):
        """Broadcast agent status update to WebSocket"""
        if session_id in self.websocket_connections:
            try:
                await self.websocket_connections[session_id].send_json({
                    "type": "agent_update",
                    "agents": {name: {
                        "name": agent.agent_name,
                        "status": agent.status,
                        "confidence": agent.confidence,
                        "current_task": agent.current_task,
                        "reasoning_step": agent.reasoning_step,
                        "avatar": agent.avatar
                    } for name, agent in self.conversation_manager.active_agents.items()},
                    "timestamp": datetime.now().isoformat()
                })
            except Exception as e:
                logger.error(f"Error broadcasting agent update: {str(e)}")
    
    async def broadcast_collaboration_event(self, session_id: str, event: CollaborationEvent):
        """Broadcast collaboration event to WebSocket"""
        if session_id in self.websocket_connections:
            try:
                await self.websocket_connections[session_id].send_json({
                    "type": "collaboration_event",
                    "event": {
                        "timestamp": event.timestamp.isoformat(),
                        "primary_agent": event.primary_agent,
                        "collaborating_agents": event.collaborating_agents,
                        "collaboration_type": event.collaboration_type,
                        "message": event.message,
                        "details": event.details
                    }
                })
            except Exception as e:
                logger.error(f"Error broadcasting collaboration event: {str(e)}")

    async def get_conversation_context(self, session_id: str) -> Dict:
        """Get full conversation context for session"""
        return {
            "session_id": session_id,
            "active_agents": {name: {
                "name": agent.agent_name,
                "status": agent.status,
                "confidence": agent.confidence,
                "current_task": agent.current_task,
                "avatar": agent.avatar
            } for name, agent in self.conversation_manager.active_agents.items()},
            "recent_collaborations": [
                {
                    "timestamp": event.timestamp.isoformat(),
                    "primary_agent": event.primary_agent,
                    "collaborating_agents": event.collaborating_agents,
                    "message": event.message
                } for event in self.conversation_manager.collaboration_events[-5:]
            ],
            "reasoning_sessions": [
                {
                    "session_id": session["session_id"],
                    "agent_name": session["agent_name"],
                    "status": session["status"],
                    "steps_count": len(session["steps"])
                } for session in self.reasoning_manager.active_reasoning_sessions.values()
            ]
        }
