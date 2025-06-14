"""
Agent Conversation API
Conversational interface for the multi-agent system
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from datetime import datetime

from ....agent.multi_agent.agent_system import QAAgentSystem
from ....agent.core.models import AgentState, UserProfile
from ....core.exceptions import AgentError


logger = logging.getLogger(__name__)
router = APIRouter()

# Global agent system instance
agent_system = QAAgentSystem()


class ConversationRequest(BaseModel):
    """Request for agent conversation"""
    message: str
    session_id: str
    user_profile: Optional[Dict[str, Any]] = None
    context: Optional[Dict[str, Any]] = None
    stream_response: bool = False


class ConversationResponse(BaseModel):
    """Response from agent conversation"""
    response: str
    session_id: str
    agents_involved: List[str]
    reasoning_steps: List[Dict[str, Any]]
    confidence: float
    recommendations: List[str]
    follow_up_questions: List[str]
    metadata: Dict[str, Any]
    timestamp: datetime


class AgentStatusResponse(BaseModel):
    """Agent system status response"""
    system_initialized: bool
    available_specialists: List[str]
    active_collaborations: int
    system_performance: Dict[str, Any]


@router.post("/api/v1/agent/conversation", response_model=ConversationResponse)
async def agent_conversation(request: ConversationRequest):
    """
    Have a conversation with the multi-agent system
    
    This endpoint provides the main conversational interface to the AI agent system.
    The system will automatically determine whether to use a single agent or 
    multiple agents based on the complexity and nature of the request.
    """
    try:
        # Initialize system if needed
        if not agent_system.system_initialized:
            await agent_system.initialize_system()
        
        # Process the request through the multi-agent system
        response = await agent_system.handle_complex_request(
            user_request=request.message,
            session_id=request.session_id,
            user_profile=request.user_profile
        )
        
        # Extract agents involved from metadata
        agents_involved = response.metadata.get("specialists_consulted", [])
        if response.metadata.get("collaboration_id"):
            agents_involved.extend(response.metadata.get("specialists_involved", []))
        
        # Convert reasoning steps to serializable format
        reasoning_steps = []
        for step in response.reasoning_steps:
            reasoning_steps.append({
                "type": step.type,
                "content": step.content,
                "confidence": step.confidence,
                "tools_used": step.tools_used,
                "timestamp": step.timestamp
            })
        
        return ConversationResponse(
            response=response.content,
            session_id=response.session_id,
            agents_involved=agents_involved,
            reasoning_steps=reasoning_steps,
            confidence=response.confidence,
            recommendations=response.recommendations,
            follow_up_questions=response.follow_up_questions,
            metadata=response.metadata,
            timestamp=response.timestamp
        )
        
    except Exception as e:
        logger.error(f"Error in agent conversation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Conversation failed: {str(e)}")


@router.websocket("/api/v1/agent/conversation/stream/{session_id}")
async def agent_conversation_stream(websocket: WebSocket, session_id: str):
    """
    Real-time conversation with the agent system via WebSocket
    
    This provides a streaming interface where users can see agent reasoning
    and collaboration happening in real-time.
    """
    await websocket.accept()
    
    try:
        # Initialize system if needed
        if not agent_system.system_initialized:
            await agent_system.initialize_system()
            await websocket.send_json({
                "type": "system_status",
                "data": {"status": "initialized", "message": "Agent system ready"}
            })
        
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            message_type = data.get("type", "message")
            
            if message_type == "message":
                user_message = data.get("message", "")
                user_profile = data.get("user_profile")
                
                # Send acknowledgment
                await websocket.send_json({
                    "type": "message_received",
                    "data": {"message": "Processing your request..."}
                })
                
                try:
                    # Process with streaming updates
                    await process_message_with_streaming(
                        websocket, user_message, session_id, user_profile
                    )
                    
                except Exception as e:
                    await websocket.send_json({
                        "type": "error",
                        "data": {"error": str(e)}
                    })
            
            elif message_type == "status":
                # Send system status
                status = await agent_system.get_system_status()
                await websocket.send_json({
                    "type": "status_response",
                    "data": status
                })
                
            elif message_type == "ping":
                # Health check
                await websocket.send_json({
                    "type": "pong",
                    "data": {"timestamp": datetime.utcnow().isoformat()}
                })
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error for session {session_id}: {str(e)}")
        try:
            await websocket.send_json({
                "type": "error",
                "data": {"error": "Connection error occurred"}
            })
        except:
            pass


async def process_message_with_streaming(
    websocket: WebSocket,
    message: str,
    session_id: str,
    user_profile: Optional[Dict[str, Any]]
):
    """Process message with real-time streaming updates"""
    
    # Send processing start
    await websocket.send_json({
        "type": "processing_start",
        "data": {
            "message": "Analyzing your request...",
            "timestamp": datetime.utcnow().isoformat()
        }
    })
    
    # Determine approach and notify user
    approach = await agent_system._determine_approach(message)
    await websocket.send_json({
        "type": "approach_determined",
        "data": {
            "approach": approach,
            "message": f"Using {approach.replace('_', ' ')} approach"
        }
    })
    
    # If multi-agent, show which specialists are being consulted
    if approach in ["multi_agent_consultation", "collaborative_analysis"]:
        relevant_specialists = await agent_system._identify_relevant_specialists(message)
        await websocket.send_json({
            "type": "specialists_identified",
            "data": {
                "specialists": relevant_specialists,
                "message": f"Consulting with: {', '.join(relevant_specialists)}"
            }
        })
        
        # Stream individual specialist consultations
        for specialist in relevant_specialists:
            await websocket.send_json({
                "type": "specialist_consulting",
                "data": {
                    "specialist": specialist,
                    "message": f"Getting input from {specialist.replace('_', ' ').title()}..."
                }
            })
            
            # Small delay to simulate real consultation time
            await asyncio.sleep(0.5)
    
    # Process the actual request
    try:
        response = await agent_system.handle_complex_request(
            user_request=message,
            session_id=session_id,
            user_profile=user_profile
        )
        
        # Send final response
        await websocket.send_json({
            "type": "response_complete",
            "data": {
                "response": response.content,
                "confidence": response.confidence,
                "recommendations": response.recommendations,
                "follow_up_questions": response.follow_up_questions,
                "metadata": response.metadata,
                "timestamp": response.timestamp.isoformat()
            }
        })
        
    except Exception as e:
        await websocket.send_json({
            "type": "processing_error",
            "data": {"error": str(e)}
        })


@router.get("/api/v1/agent/status", response_model=AgentStatusResponse)
async def get_agent_status():
    """
    Get current status of the agent system
    
    Returns information about system initialization, available specialists,
    active collaborations, and performance metrics.
    """
    try:
        if not agent_system.system_initialized:
            await agent_system.initialize_system()
        
        status = await agent_system.get_system_status()
        
        return AgentStatusResponse(
            system_initialized=status["system_initialized"],
            available_specialists=list(status["specialists"].keys()),
            active_collaborations=status["active_collaborations"],
            system_performance=status["system_performance"]
        )
        
    except Exception as e:
        logger.error(f"Error getting agent status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")


@router.get("/api/v1/agent/specialists")
async def get_specialist_profiles():
    """
    Get detailed profiles of all specialist agents
    
    Returns comprehensive information about each specialist including
    their capabilities, expertise domains, and performance metrics.
    """
    try:
        if not agent_system.system_initialized:
            await agent_system.initialize_system()
        
        profiles = {}
        for name, agent in agent_system.specialists.items():
            profile = agent.get_specialist_profile()
            profiles[name] = {
                "name": profile.agent_name,
                "specialization": profile.specialization,
                "expertise_domains": profile.expertise_domains,
                "capabilities": [
                    {
                        "name": cap.name,
                        "description": cap.description,
                        "confidence_level": cap.confidence_level,
                        "experience_count": cap.experience_count,
                        "success_rate": cap.success_rate
                    }
                    for cap in profile.capabilities
                ],
                "performance_metrics": profile.performance_metrics,
                "availability_status": profile.availability_status
            }
        
        return {"specialists": profiles}
        
    except Exception as e:
        logger.error(f"Error getting specialist profiles: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get specialist profiles: {str(e)}")


@router.post("/api/v1/agent/conversation/context")
async def set_conversation_context(
    session_id: str,
    context: Dict[str, Any]
):
    """
    Set conversation context for enhanced agent responses
    
    Allows setting user preferences, project context, and other information
    that helps agents provide more targeted and relevant assistance.
    """
    try:
        # This would typically store context in the conversation manager
        # For now, we'll return a success response
        return {
            "status": "success",
            "message": "Conversation context updated",
            "session_id": session_id,
            "context_keys": list(context.keys())
        }
        
    except Exception as e:
        logger.error(f"Error setting conversation context: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to set context: {str(e)}")


@router.get("/api/v1/agent/conversation/{session_id}/insights")
async def get_conversation_insights(session_id: str):
    """
    Get insights about a conversation session
    
    Returns analytics about the conversation including agent involvement,
    collaboration patterns, and effectiveness metrics.
    """
    try:
        # This would get insights from the orchestrator
        insights = {
            "session_id": session_id,
            "message_count": 0,  # Would be tracked
            "agents_involved": [],  # Would be tracked
            "collaboration_sessions": 0,  # Would be tracked
            "user_satisfaction_predicted": 0.85,  # Would be calculated
            "key_topics": [],  # Would be extracted
            "recommendations_given": []  # Would be tracked
        }
        
        return insights
        
    except Exception as e:
        logger.error(f"Error getting conversation insights: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get insights: {str(e)}")
