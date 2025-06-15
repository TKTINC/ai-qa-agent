"""
Agent Reasoning Tracing System (Simplified)
Basic tracing for agent reasoning processes
"""

import time
import logging
import asyncio
from typing import Dict, List, Optional, Any, AsyncContextManager
from datetime import datetime
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

# Try to import OpenTelemetry, fallback to basic logging if not available
try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    TRACING_AVAILABLE = True
except ImportError:
    TRACING_AVAILABLE = False
    print("⚠️  OpenTelemetry not available, using basic logging for tracing")

logger = logging.getLogger(__name__)

@dataclass
class ReasoningStep:
    """Detailed reasoning step for tracing"""
    step_id: str
    step_type: str  # observe, think, plan, act, reflect
    description: str
    confidence: float
    duration: float
    tools_used: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass
class ReasoningSession:
    """Complete reasoning session with all steps"""
    session_id: str
    agent_name: str
    task_type: str
    user_request: str
    start_time: datetime
    end_time: Optional[datetime] = None
    steps: List[ReasoningStep] = field(default_factory=list)
    success: bool = False

class AgentReasoningTracer:
    """
    Basic agent reasoning tracer with optional OpenTelemetry integration
    """
    
    def __init__(self, service_name: str = "qa-agent-reasoning"):
        self.service_name = service_name
        self.tracer = None
        self._active_sessions: Dict[str, ReasoningSession] = {}
        
        if TRACING_AVAILABLE:
            self._initialize_tracing()
        else:
            logger.info("Using basic logging tracer (OpenTelemetry not available)")
        
    def _initialize_tracing(self):
        """Initialize OpenTelemetry tracing if available"""
        try:
            if TRACING_AVAILABLE:
                # Set up basic tracer provider
                tracer_provider = TracerProvider()
                trace.set_tracer_provider(tracer_provider)
                
                # Create tracer
                self.tracer = trace.get_tracer(
                    instrumenting_module_name=__name__,
                    instrumenting_library_version="1.0.0"
                )
                
                logger.info(f"OpenTelemetry tracer initialized for {self.service_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenTelemetry tracer: {e}")
            self.tracer = None

    @asynccontextmanager
    async def trace_reasoning_session(self, 
                                    session_id: str,
                                    agent_name: str,
                                    task_type: str,
                                    user_request: str) -> AsyncContextManager[ReasoningSession]:
        """
        Trace a complete reasoning session
        """
        # Create reasoning session
        session = ReasoningSession(
            session_id=session_id,
            agent_name=agent_name,
            task_type=task_type,
            user_request=user_request,
            start_time=datetime.utcnow()
        )
        
        self._active_sessions[session_id] = session
        
        # Start span if OpenTelemetry is available
        span = None
        if self.tracer:
            span = self.tracer.start_span(
                f"agent_reasoning_session",
                attributes={
                    "agent.name": agent_name,
                    "agent.session_id": session_id,
                    "agent.task_type": task_type,
                }
            )
        
        try:
            logger.info(f"Started reasoning session {session_id} for agent {agent_name}")
            yield session
            
            # Mark session as successful
            session.success = True
            session.end_time = datetime.utcnow()
            
            if span:
                span.set_attribute("reasoning.success", True)
                span.set_attribute("reasoning.steps_count", len(session.steps))
            
            logger.info(f"Reasoning session {session_id} completed successfully")
            
        except Exception as e:
            session.success = False
            session.end_time = datetime.utcnow()
            
            if span:
                span.set_attribute("reasoning.success", False)
                span.record_exception(e)
            
            logger.error(f"Reasoning session {session_id} failed: {e}")
            raise
            
        finally:
            if span:
                span.end()
            
            # Clean up active session
            if session_id in self._active_sessions:
                del self._active_sessions[session_id]

    async def trace_reasoning_step(self,
                                 session_id: str,
                                 step_type: str,
                                 description: str,
                                 confidence: float,
                                 tools_used: Optional[List[str]] = None) -> ReasoningStep:
        """
        Trace an individual reasoning step within a session
        """
        if session_id not in self._active_sessions:
            logger.warning(f"No active session found for {session_id}")
            return None
        
        session = self._active_sessions[session_id]
        step_start = time.time()
        
        # Generate step ID
        step_id = f"{session_id}_step_{len(session.steps) + 1}"
        
        try:
            # Simulate step execution time
            await asyncio.sleep(0.01)
            
            step_duration = time.time() - step_start
            
            # Create reasoning step
            step = ReasoningStep(
                step_id=step_id,
                step_type=step_type,
                description=description,
                confidence=confidence,
                duration=step_duration,
                tools_used=tools_used or []
            )
            
            # Add step to session
            session.steps.append(step)
            
            logger.debug(f"Traced reasoning step {step_id}: {step_type}")
            return step
            
        except Exception as e:
            logger.error(f"Error tracing reasoning step {step_id}: {e}")
            raise

    def get_session_trace_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get trace summary for a reasoning session"""
        if session_id not in self._active_sessions:
            return None
        
        session = self._active_sessions[session_id]
        
        return {
            "session_id": session_id,
            "agent_name": session.agent_name,
            "task_type": session.task_type,
            "start_time": session.start_time.isoformat(),
            "end_time": session.end_time.isoformat() if session.end_time else None,
            "steps_count": len(session.steps),
            "success": session.success,
            "reasoning_steps": [
                {
                    "step_id": step.step_id,
                    "step_type": step.step_type,
                    "description": step.description,
                    "confidence": step.confidence,
                    "duration": step.duration
                }
                for step in session.steps
            ]
        }


# Global tracer instance
_reasoning_tracer: Optional[AgentReasoningTracer] = None

def get_reasoning_tracer() -> AgentReasoningTracer:
    """Get global reasoning tracer instance"""
    global _reasoning_tracer
    if _reasoning_tracer is None:
        _reasoning_tracer = AgentReasoningTracer()
    return _reasoning_tracer
