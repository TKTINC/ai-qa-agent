"""
Agent Visualization Service
Handles real-time visualization of agent activities, reasoning, and collaboration
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json
import logging
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class AgentActivity:
    """Represents an agent activity for visualization"""
    timestamp: datetime
    agent_name: str
    activity_type: str  # reasoning, collaboration, tool_usage, learning
    description: str
    confidence: float
    details: Dict[str, Any]
    duration: Optional[float] = None

@dataclass
class ReasoningStep:
    """Represents a reasoning step for visualization"""
    step_number: int
    step_type: str  # observe, think, plan, act, reflect
    thought: str
    action: Optional[str] = None
    observation: Optional[str] = None
    confidence: float = 0.0
    timestamp: datetime = None
    duration: Optional[float] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class LearningEvent:
    """Represents a learning event for visualization"""
    timestamp: datetime
    agent_name: str
    learning_type: str  # preference, pattern, feedback, improvement
    description: str
    impact_score: float
    details: Dict[str, Any]

class AgentVisualizationService:
    """Service for managing agent visualization data"""
    
    def __init__(self):
        self.agent_activities: List[AgentActivity] = []
        self.reasoning_sessions: Dict[str, List[ReasoningStep]] = {}
        self.learning_events: List[LearningEvent] = []
        self.agent_performance_cache: Dict[str, Dict] = {}
        
        # WebSocket subscribers for real-time updates
        self.websocket_subscribers: Dict[str, List] = {}
        
    async def record_agent_activity(self, agent_name: str, activity_type: str,
                                  description: str, confidence: float,
                                  details: Dict[str, Any] = None) -> str:
        """Record an agent activity"""
        activity = AgentActivity(
            timestamp=datetime.now(),
            agent_name=agent_name,
            activity_type=activity_type,
            description=description,
            confidence=confidence,
            details=details or {}
        )
        
        self.agent_activities.append(activity)
        
        # Keep only recent activities (last 100)
        if len(self.agent_activities) > 100:
            self.agent_activities = self.agent_activities[-100:]
        
        # Broadcast to subscribers
        await self._broadcast_activity_update(activity)
        
        activity_id = f"{agent_name}_{int(activity.timestamp.timestamp())}"
        logger.info(f"Recorded activity {activity_id}: {description}")
        return activity_id
    
    async def start_reasoning_session(self, session_id: str, agent_name: str,
                                    initial_thought: str) -> None:
        """Start a new reasoning session"""
        initial_step = ReasoningStep(
            step_number=1,
            step_type="observe",
            thought=initial_thought,
            confidence=0.0
        )
        
        self.reasoning_sessions[session_id] = [initial_step]
        
        await self.record_agent_activity(
            agent_name, "reasoning", 
            f"Started reasoning session: {initial_thought[:50]}...",
            0.8, {"session_id": session_id}
        )
        
        logger.info(f"Started reasoning session {session_id} for {agent_name}")
    
    async def add_reasoning_step(self, session_id: str, step_type: str,
                               thought: str, action: str = None,
                               observation: str = None, confidence: float = 0.0) -> None:
        """Add a step to existing reasoning session"""
        if session_id not in self.reasoning_sessions:
            logger.warning(f"Reasoning session {session_id} not found")
            return
        
        steps = self.reasoning_sessions[session_id]
        step_number = len(steps) + 1
        
        step = ReasoningStep(
            step_number=step_number,
            step_type=step_type,
            thought=thought,
            action=action,
            observation=observation,
            confidence=confidence
        )
        
        steps.append(step)
        
        # Broadcast reasoning update
        await self._broadcast_reasoning_update(session_id, step)
        
        logger.info(f"Added reasoning step {step_number} to session {session_id}: {step_type}")
    
    async def complete_reasoning_session(self, session_id: str, outcome: str) -> None:
        """Complete a reasoning session"""
        if session_id not in self.reasoning_sessions:
            return
        
        steps = self.reasoning_sessions[session_id]
        final_step = ReasoningStep(
            step_number=len(steps) + 1,
            step_type="reflect",
            thought=f"Session completed: {outcome}",
            confidence=1.0
        )
        
        steps.append(final_step)
        
        # Calculate session duration
        if steps:
            duration = (final_step.timestamp - steps[0].timestamp).total_seconds()
            for step in steps:
                if step.duration is None:
                    step.duration = duration / len(steps)
        
        await self._broadcast_reasoning_completion(session_id, outcome)
        
        logger.info(f"Completed reasoning session {session_id}: {outcome}")
    
    async def record_learning_event(self, agent_name: str, learning_type: str,
                                  description: str, impact_score: float,
                                  details: Dict[str, Any] = None) -> None:
        """Record a learning event"""
        event = LearningEvent(
            timestamp=datetime.now(),
            agent_name=agent_name,
            learning_type=learning_type,
            description=description,
            impact_score=impact_score,
            details=details or {}
        )
        
        self.learning_events.append(event)
        
        # Keep only recent events (last 50)
        if len(self.learning_events) > 50:
            self.learning_events = self.learning_events[-50:]
        
        # Also record as activity
        await self.record_agent_activity(
            agent_name, "learning", description, impact_score,
            {"learning_type": learning_type, **details}
        )
        
        await self._broadcast_learning_update(event)
        
        logger.info(f"Recorded learning event for {agent_name}: {description}")
    
    async def get_agent_performance_summary(self, agent_name: str,
                                          time_window: timedelta = timedelta(hours=24)) -> Dict:
        """Get performance summary for an agent"""
        cutoff_time = datetime.now() - time_window
        
        # Filter recent activities
        recent_activities = [
            activity for activity in self.agent_activities
            if activity.agent_name == agent_name and activity.timestamp >= cutoff_time
        ]
        
        # Filter recent learning events
        recent_learning = [
            event for event in self.learning_events
            if event.agent_name == agent_name and event.timestamp >= cutoff_time
        ]
        
        # Calculate performance metrics
        total_activities = len(recent_activities)
        avg_confidence = sum(a.confidence for a in recent_activities) / max(total_activities, 1)
        learning_count = len(recent_learning)
        avg_learning_impact = sum(e.impact_score for e in recent_learning) / max(learning_count, 1)
        
        # Activity type breakdown
        activity_types = {}
        for activity in recent_activities:
            activity_types[activity.activity_type] = activity_types.get(activity.activity_type, 0) + 1
        
        # Learning type breakdown
        learning_types = {}
        for event in recent_learning:
            learning_types[event.learning_type] = learning_types.get(event.learning_type, 0) + 1
        
        summary = {
            "agent_name": agent_name,
            "time_window_hours": time_window.total_seconds() / 3600,
            "total_activities": total_activities,
            "average_confidence": round(avg_confidence, 3),
            "learning_events": learning_count,
            "average_learning_impact": round(avg_learning_impact, 3),
            "activity_breakdown": activity_types,
            "learning_breakdown": learning_types,
            "performance_score": round((avg_confidence + avg_learning_impact) / 2, 3),
            "last_activity": recent_activities[-1].timestamp.isoformat() if recent_activities else None
        }
        
        # Cache the summary
        self.agent_performance_cache[agent_name] = summary
        
        return summary
    
    async def get_live_activity_feed(self, limit: int = 20) -> List[Dict]:
        """Get recent agent activities for live feed"""
        recent_activities = sorted(
            self.agent_activities[-limit:],
            key=lambda x: x.timestamp,
            reverse=True
        )
        
        return [
            {
                "timestamp": activity.timestamp.isoformat(),
                "agent_name": activity.agent_name,
                "activity_type": activity.activity_type,
                "description": activity.description,
                "confidence": activity.confidence,
                "details": activity.details
            }
            for activity in recent_activities
        ]
    
    async def get_reasoning_session_data(self, session_id: str) -> Optional[Dict]:
        """Get reasoning session data for visualization"""
        if session_id not in self.reasoning_sessions:
            return None
        
        steps = self.reasoning_sessions[session_id]
        
        return {
            "session_id": session_id,
            "step_count": len(steps),
            "start_time": steps[0].timestamp.isoformat() if steps else None,
            "latest_step": steps[-1].step_type if steps else None,
            "average_confidence": sum(s.confidence for s in steps) / max(len(steps), 1),
            "steps": [
                {
                    "step_number": step.step_number,
                    "step_type": step.step_type,
                    "thought": step.thought,
                    "action": step.action,
                    "observation": step.observation,
                    "confidence": step.confidence,
                    "timestamp": step.timestamp.isoformat(),
                    "duration": step.duration
                }
                for step in steps
            ]
        }
    
    async def get_system_intelligence_metrics(self) -> Dict:
        """Get overall system intelligence metrics"""
        now = datetime.now()
        last_hour = now - timedelta(hours=1)
        last_day = now - timedelta(days=1)
        
        # Activities in last hour and day
        hour_activities = [a for a in self.agent_activities if a.timestamp >= last_hour]
        day_activities = [a for a in self.agent_activities if a.timestamp >= last_day]
        
        # Learning events in last hour and day
        hour_learning = [e for e in self.learning_events if e.timestamp >= last_hour]
        day_learning = [e for e in self.learning_events if e.timestamp >= last_day]
        
        # Agent activity summary
        agent_activity_count = {}
        for activity in day_activities:
            agent_activity_count[activity.agent_name] = agent_activity_count.get(activity.agent_name, 0) + 1
        
        # Average confidence trends
        hour_confidence = sum(a.confidence for a in hour_activities) / max(len(hour_activities), 1)
        day_confidence = sum(a.confidence for a in day_activities) / max(len(day_activities), 1)
        
        return {
            "timestamp": now.isoformat(),
            "activity_metrics": {
                "last_hour": len(hour_activities),
                "last_day": len(day_activities),
                "average_confidence_hour": round(hour_confidence, 3),
                "average_confidence_day": round(day_confidence, 3)
            },
            "learning_metrics": {
                "events_last_hour": len(hour_learning),
                "events_last_day": len(day_learning),
                "average_impact_hour": round(sum(e.impact_score for e in hour_learning) / max(len(hour_learning), 1), 3),
                "average_impact_day": round(sum(e.impact_score for e in day_learning) / max(len(day_learning), 1), 3)
            },
            "agent_activity_distribution": agent_activity_count,
            "active_reasoning_sessions": len(self.reasoning_sessions),
            "system_health_score": round((day_confidence + (len(day_learning) / 10)) / 2, 3)
        }
    
    async def subscribe_to_updates(self, subscriber_id: str, websocket) -> None:
        """Subscribe to real-time visualization updates"""
        if subscriber_id not in self.websocket_subscribers:
            self.websocket_subscribers[subscriber_id] = []
        
        self.websocket_subscribers[subscriber_id].append(websocket)
        logger.info(f"Added WebSocket subscriber {subscriber_id}")
    
    async def unsubscribe_from_updates(self, subscriber_id: str, websocket) -> None:
        """Unsubscribe from real-time updates"""
        if subscriber_id in self.websocket_subscribers:
            if websocket in self.websocket_subscribers[subscriber_id]:
                self.websocket_subscribers[subscriber_id].remove(websocket)
            
            if not self.websocket_subscribers[subscriber_id]:
                del self.websocket_subscribers[subscriber_id]
        
        logger.info(f"Removed WebSocket subscriber {subscriber_id}")
    
    async def _broadcast_activity_update(self, activity: AgentActivity) -> None:
        """Broadcast activity update to all subscribers"""
        message = {
            "type": "activity_update",
            "data": {
                "timestamp": activity.timestamp.isoformat(),
                "agent_name": activity.agent_name,
                "activity_type": activity.activity_type,
                "description": activity.description,
                "confidence": activity.confidence,
                "details": activity.details
            }
        }
        
        await self._broadcast_to_subscribers(message)
    
    async def _broadcast_reasoning_update(self, session_id: str, step: ReasoningStep) -> None:
        """Broadcast reasoning step update"""
        message = {
            "type": "reasoning_update",
            "data": {
                "session_id": session_id,
                "step": {
                    "step_number": step.step_number,
                    "step_type": step.step_type,
                    "thought": step.thought,
                    "action": step.action,
                    "observation": step.observation,
                    "confidence": step.confidence,
                    "timestamp": step.timestamp.isoformat()
                }
            }
        }
        
        await self._broadcast_to_subscribers(message)
    
    async def _broadcast_reasoning_completion(self, session_id: str, outcome: str) -> None:
        """Broadcast reasoning session completion"""
        message = {
            "type": "reasoning_completion",
            "data": {
                "session_id": session_id,
                "outcome": outcome,
                "timestamp": datetime.now().isoformat()
            }
        }
        
        await self._broadcast_to_subscribers(message)
    
    async def _broadcast_learning_update(self, event: LearningEvent) -> None:
        """Broadcast learning event update"""
        message = {
            "type": "learning_update",
            "data": {
                "timestamp": event.timestamp.isoformat(),
                "agent_name": event.agent_name,
                "learning_type": event.learning_type,
                "description": event.description,
                "impact_score": event.impact_score,
                "details": event.details
            }
        }
        
        await self._broadcast_to_subscribers(message)
    
    async def _broadcast_to_subscribers(self, message: Dict) -> None:
        """Broadcast message to all WebSocket subscribers"""
        for subscriber_id, websockets in self.websocket_subscribers.items():
            for websocket in websockets[:]:  # Create copy to avoid modification during iteration
                try:
                    await websocket.send_json(message)
                except Exception as e:
                    logger.error(f"Error broadcasting to subscriber {subscriber_id}: {str(e)}")
                    # Remove failed WebSocket
                    websockets.remove(websocket)

# Global instance
agent_visualization_service = AgentVisualizationService()
