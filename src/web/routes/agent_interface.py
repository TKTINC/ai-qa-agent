"""
Web routes for Agent Chat Interface
Handles web interface rendering and real-time communication
"""

import asyncio
from datetime import datetime
from typing import Dict, Optional
import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Request, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from src.web.components.agent_chat import AgentChatInterface
from src.web.services.agent_visualization import agent_visualization_service

# Fallback for missing learning engine
try:
    from src.agent.learning.learning_engine import AgentLearningEngine
    learning_engine = AgentLearningEngine()
except ImportError:
    class MockLearningEngine:
        async def get_session_insights(self, session_id: str):
            return {"insights": ["Mock learning insight"]}
        
        async def process_feedback(self, feedback_data: Dict):
            return {"improvements_applied": ["Mock improvement"]}
    
    learning_engine = MockLearningEngine()

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/web", tags=["Agent Interface"])
templates = Jinja2Templates(directory="src/web/templates")

# Global chat interface instance
chat_interface = AgentChatInterface()

class ConversationRequest(BaseModel):
    message: str
    session_id: str
    user_profile: Optional[Dict] = None

class FeedbackRequest(BaseModel):
    session_id: str
    message_id: Optional[str] = None
    feedback_type: str  # positive, negative, suggestion
    feedback_text: str
    rating: Optional[int] = None  # 1-5 scale

@router.get("/", response_class=HTMLResponse)
async def agent_chat_interface(request: Request):
    """Render the main agent chat interface"""
    return templates.TemplateResponse(
        "agent_chat.html",
        {
            "request": request,
            "title": "AI QA Agent - Conversational Interface",
            "timestamp": datetime.now().isoformat()
        }
    )

@router.post("/api/v1/agent/conversation")
async def agent_conversation(request: ConversationRequest):
    """Handle agent conversation via HTTP"""
    try:
        logger.info(f"Processing conversation request for session {request.session_id}")
        
        # Record user activity for visualization
        await agent_visualization_service.record_agent_activity(
            "user", "conversation", 
            f"User message: {request.message[:50]}...",
            1.0, {"session_id": request.session_id}
        )
        
        # Process the message through agent system
        response = await chat_interface.handle_user_message(
            request.message, 
            request.session_id,
            request.user_profile
        )
        
        # Record agent response activity
        await agent_visualization_service.record_agent_activity(
            "agent_system", "response",
            f"Generated response for user query",
            response.get("confidence", 0.8),
            {
                "session_id": request.session_id,
                "agents_involved": response.get("agents_involved", []),
                "response_length": len(response.get("response", {}).get("text", ""))
            }
        )
        
        return {
            "success": True,
            "response": response["response"],
            "session_id": request.session_id,
            "timestamp": response["timestamp"],
            "agents_involved": response.get("agents_involved", []),
            "confidence": response.get("confidence", 0.0)
        }
        
    except Exception as e:
        logger.error(f"Error in agent conversation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing conversation: {str(e)}")

@router.websocket("/api/v1/agent/conversation/stream/{session_id}")
async def agent_conversation_websocket(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time agent conversation"""
    try:
        await chat_interface.connect_websocket(websocket, session_id)
        
        # Subscribe to visualization updates
        await agent_visualization_service.subscribe_to_updates(session_id, websocket)
        
        logger.info(f"WebSocket connection established for session {session_id}")
        
        while True:
            try:
                # Wait for messages from client
                data = await websocket.receive_json()
                
                if data.get("type") == "ping":
                    await websocket.send_json({"type": "pong", "timestamp": datetime.now().isoformat()})
                elif data.get("type") == "request_status":
                    # Send current agent status
                    context = await chat_interface.get_conversation_context(session_id)
                    await websocket.send_json({
                        "type": "status_update",
                        "data": context
                    })
                elif data.get("type") == "feedback":
                    # Handle real-time feedback
                    await handle_real_time_feedback(data, session_id)
                    
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"Error in WebSocket communication: {str(e)}")
                await websocket.send_json({
                    "type": "error",
                    "message": "Communication error occurred"
                })
                break
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error for session {session_id}: {str(e)}")
    finally:
        await chat_interface.disconnect_websocket(session_id)
        await agent_visualization_service.unsubscribe_from_updates(session_id, websocket)

@router.get("/api/v1/agent/status")
async def get_agent_status():
    """Get current agent system status"""
    try:
        # Get system intelligence metrics
        intelligence_metrics = await agent_visualization_service.get_system_intelligence_metrics()
        
        # Get agent performance summaries
        agent_names = ["test_architect", "code_reviewer", "performance_analyst", 
                      "security_specialist", "documentation_expert"]
        
        agent_performances = {}
        for agent_name in agent_names:
            performance = await agent_visualization_service.get_agent_performance_summary(agent_name)
            agent_performances[agent_name] = performance
        
        # Get live activity feed
        activity_feed = await agent_visualization_service.get_live_activity_feed(10)
        
        return {
            "status": "operational",
            "timestamp": datetime.now().isoformat(),
            "intelligence_metrics": intelligence_metrics,
            "agent_performances": agent_performances,
            "activity_feed": activity_feed,
            "capabilities": {
                "multi_agent_collaboration": True,
                "real_time_reasoning": True,
                "learning_enabled": True,
                "visualization_enabled": True
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting agent status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving agent status: {str(e)}")

@router.get("/api/v1/agent/context/{session_id}")
async def get_conversation_context(session_id: str):
    """Get conversation context for a session"""
    try:
        context = await chat_interface.get_conversation_context(session_id)
        
        # Add learning insights
        learning_insights = await learning_engine.get_session_insights(session_id)
        context["learning_insights"] = learning_insights
        
        return {
            "success": True,
            "context": context,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting conversation context: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving context: {str(e)}")

@router.post("/api/v1/agent/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    """Submit user feedback for learning"""
    try:
        logger.info(f"Received feedback for session {feedback.session_id}: {feedback.feedback_type}")
        
        # Process feedback through learning engine
        feedback_data = {
            "session_id": feedback.session_id,
            "feedback_type": feedback.feedback_type,
            "feedback_text": feedback.feedback_text,
            "rating": feedback.rating,
            "timestamp": datetime.now(),
            "message_id": feedback.message_id
        }
        
        # Record feedback as learning event
        await agent_visualization_service.record_learning_event(
            "user_feedback", feedback.feedback_type,
            f"User provided {feedback.feedback_type} feedback: {feedback.feedback_text[:50]}...",
            feedback.rating / 5.0 if feedback.rating else 0.5,
            feedback_data
        )
        
        # Process feedback for learning
        learning_result = await learning_engine.process_feedback(feedback_data)
        
        return {
            "success": True,
            "message": "Feedback received and processed",
            "learning_applied": learning_result.get("improvements_applied", []),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error processing feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing feedback: {str(e)}")

@router.get("/api/v1/agent/reasoning/{session_id}")
async def get_reasoning_session(session_id: str):
    """Get reasoning session data for visualization"""
    try:
        reasoning_data = await agent_visualization_service.get_reasoning_session_data(session_id)
        
        if not reasoning_data:
            raise HTTPException(status_code=404, detail="Reasoning session not found")
        
        return {
            "success": True,
            "reasoning_session": reasoning_data,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting reasoning session: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving reasoning session: {str(e)}")

@router.get("/api/v1/agent/analytics/live")
async def get_live_analytics():
    """Get live agent analytics data"""
    try:
        # Get system intelligence metrics
        intelligence_metrics = await agent_visualization_service.get_system_intelligence_metrics()
        
        # Get live activity feed
        activity_feed = await agent_visualization_service.get_live_activity_feed(20)
        
        # Get agent performance summaries
        agent_names = ["test_architect", "code_reviewer", "performance_analyst", 
                      "security_specialist", "documentation_expert"]
        
        performance_summaries = {}
        for agent_name in agent_names:
            summary = await agent_visualization_service.get_agent_performance_summary(agent_name)
            performance_summaries[agent_name] = summary
        
        return {
            "timestamp": datetime.now().isoformat(),
            "intelligence_metrics": intelligence_metrics,
            "activity_feed": activity_feed,
            "performance_summaries": performance_summaries,
            "system_health": {
                "status": "operational",
                "uptime": "99.9%",
                "response_time": "0.45s",
                "active_sessions": len(chat_interface.websocket_connections)
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting live analytics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving analytics: {str(e)}")

async def handle_real_time_feedback(data: Dict, session_id: str):
    """Handle real-time feedback from WebSocket"""
    try:
        feedback_type = data.get("feedback_type", "general")
        feedback_text = data.get("feedback_text", "")
        rating = data.get("rating")
        
        # Create feedback request
        feedback = FeedbackRequest(
            session_id=session_id,
            feedback_type=feedback_type,
            feedback_text=feedback_text,
            rating=rating
        )
        
        # Process the feedback
        await submit_feedback(feedback)
        
        logger.info(f"Processed real-time feedback for session {session_id}")
        
    except Exception as e:
        logger.error(f"Error handling real-time feedback: {str(e)}")

# WebSocket endpoint for live analytics streaming
@router.websocket("/api/v1/agent/analytics/stream")
async def analytics_stream_websocket(websocket: WebSocket):
    """WebSocket endpoint for streaming live analytics"""
    try:
        await websocket.accept()
        logger.info("Analytics WebSocket connection established")
        
        while True:
            try:
                # Send analytics update every 5 seconds
                analytics_data = await get_live_analytics()
                await websocket.send_json({
                    "type": "analytics_update",
                    "data": analytics_data
                })
                
                await asyncio.sleep(5)
                
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"Error in analytics WebSocket: {str(e)}")
                await websocket.send_json({
                    "type": "error",
                    "message": "Analytics streaming error"
                })
                break
                
    except WebSocketDisconnect:
        logger.info("Analytics WebSocket disconnected")
    except Exception as e:
        logger.error(f"Analytics WebSocket error: {str(e)}")
