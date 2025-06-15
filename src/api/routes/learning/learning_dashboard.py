"""
Real-time learning dashboard API endpoints.
"""

import asyncio
import json
from typing import Dict, List, Optional, Any, AsyncGenerator
from datetime import datetime, timedelta
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Depends, Query
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse
import numpy as np

from ....agent.learning.learning_engine import AgentLearningEngine
from ....streaming.learning_stream import LearningStreamManager
from ....analytics.dashboards.learning_dashboard import LearningDashboard
from ....core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/learning", tags=["Learning Dashboard"])

# Global instances
learning_engine = AgentLearningEngine()
stream_manager = LearningStreamManager()
dashboard = LearningDashboard()


@router.websocket("/live-analytics")
async def live_learning_analytics(websocket: WebSocket):
    """Stream real-time learning analytics and agent improvements"""
    
    await websocket.accept()
    client_id = f"client_{datetime.now().timestamp()}"
    
    try:
        # Register client for learning updates
        await stream_manager.register_client(client_id, websocket)
        
        # Send initial analytics data
        initial_data = await _get_initial_analytics_data()
        await websocket.send_json({
            "type": "initial_data",
            "data": initial_data,
            "timestamp": datetime.now().isoformat()
        })
        
        # Start streaming updates
        while True:
            try:
                # Wait for client messages (keep-alive, requests, etc.)
                message = await websocket.receive_text()
                
                if message == "ping":
                    await websocket.send_json({"type": "pong"})
                elif message.startswith("request_"):
                    # Handle specific data requests
                    request_type = message.split("_", 1)[1]
                    response_data = await _handle_data_request(request_type)
                    await websocket.send_json({
                        "type": f"response_{request_type}",
                        "data": response_data,
                        "timestamp": datetime.now().isoformat()
                    })
                    
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"Error in live analytics stream: {str(e)}")
                await websocket.send_json({
                    "type": "error",
                    "message": str(e),
                    "timestamp": datetime.now().isoformat()
                })
                
    except WebSocketDisconnect:
        logger.info(f"Client {client_id} disconnected from live analytics")
    except Exception as e:
        logger.error(f"Error in live analytics websocket: {str(e)}")
    finally:
        # Unregister client
        await stream_manager.unregister_client(client_id)


@router.get("/stream/learning-events")
async def stream_learning_events():
    """Server-Sent Events stream for learning events"""
    
    async def event_generator():
        """Generate learning events as SSE"""
        
        while True:
            try:
                # Get recent learning events
                recent_events = await _get_recent_learning_events()
                
                for event in recent_events:
                    yield {
                        "event": "learning_event",
                        "data": json.dumps(event)
                    }
                
                # Wait before next batch
                await asyncio.sleep(5)
                
            except Exception as e:
                logger.error(f"Error in learning events stream: {str(e)}")
                yield {
                    "event": "error",
                    "data": json.dumps({"error": str(e)})
                }
    
    return EventSourceResponse(event_generator())


@router.get("/dashboard/data")
async def get_dashboard_data(
    time_range: str = Query("24h", description="Time range for dashboard data"),
    refresh_rate: int = Query(30, description="Refresh rate in seconds")
) -> Dict[str, Any]:
    """Get comprehensive dashboard data for learning analytics"""
    
    try:
        # Generate dashboard data
        dashboard_data = await dashboard.generate_dashboard_data(time_range)
        
        # Add real-time metadata
        dashboard_data["metadata"] = {
            "generated_at": datetime.now().isoformat(),
            "time_range": time_range,
            "refresh_rate": refresh_rate,
            "data_freshness": "real-time"
        }
        
        logger.info(f"Generated dashboard data for {time_range} range")
        return dashboard_data
        
    except Exception as e:
        logger.error(f"Error generating dashboard data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating dashboard: {str(e)}")


@router.get("/charts/learning-velocity")
async def get_learning_velocity_chart(
    agents: Optional[List[str]] = Query(None, description="Filter by agent names"),
    time_period: str = Query("7d", description="Time period for analysis")
) -> Dict[str, Any]:
    """Generate learning velocity chart data"""
    
    try:
        chart_data = await dashboard.create_learning_velocity_chart(
            agents=agents,
            time_period=time_period
        )
        
        return {
            "chart_type": "learning_velocity",
            "data": chart_data,
            "metadata": {
                "agents_included": agents or ["all"],
                "time_period": time_period,
                "generated_at": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Error generating learning velocity chart: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating chart: {str(e)}")


@router.get("/charts/agent-performance")
async def get_agent_performance_chart(
    metrics: List[str] = Query(["success_rate", "satisfaction", "learning_velocity"], 
                              description="Metrics to include"),
    comparison_mode: str = Query("agents", description="Compare by: agents, time_periods, metrics")
) -> Dict[str, Any]:
    """Generate agent performance comparison chart"""
    
    try:
        chart_data = await dashboard.create_agent_performance_chart(
            metrics=metrics,
            comparison_mode=comparison_mode
        )
        
        return {
            "chart_type": "agent_performance",
            "data": chart_data,
            "metadata": {
                "metrics": metrics,
                "comparison_mode": comparison_mode,
                "generated_at": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Error generating agent performance chart: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating chart: {str(e)}")


@router.get("/charts/satisfaction-trends")
async def get_satisfaction_trends_chart(
    granularity: str = Query("daily", description="Time granularity: hourly, daily, weekly"),
    include_predictions: bool = Query(True, description="Include satisfaction predictions")
) -> Dict[str, Any]:
    """Generate user satisfaction trends chart"""
    
    try:
        chart_data = await dashboard.create_satisfaction_trends_chart(
            granularity=granularity,
            include_predictions=include_predictions
        )
        
        return {
            "chart_type": "satisfaction_trends",
            "data": chart_data,
            "metadata": {
                "granularity": granularity,
                "includes_predictions": include_predictions,
                "generated_at": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Error generating satisfaction trends chart: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating chart: {str(e)}")


async def _get_initial_analytics_data() -> Dict[str, Any]:
    """Get initial analytics data for new websocket connections"""
    
    # Get current learning insights
    learning_insights = await learning_engine.get_learning_insights("24h")
    
    # Get agent performance summary
    agent_summary = {}
    for agent_name in learning_engine.agent_capabilities.keys():
        capabilities = learning_engine.agent_capabilities[agent_name]
        agent_summary[agent_name] = {
            "performance_score": capabilities.get('problem_solving_score', 0.7),
            "learning_velocity": capabilities.get('learning_velocity', 0.5),
            "user_satisfaction": capabilities.get('user_communication', 0.7)
        }
    
    return {
        "learning_insights": learning_insights,
        "agent_summary": agent_summary,
        "system_health": {
            "total_agents": len(learning_engine.agent_capabilities),
            "active_learning": True,
            "last_update": datetime.now().isoformat()
        }
    }


async def _handle_data_request(request_type: str) -> Dict[str, Any]:
    """Handle specific data requests from websocket clients"""
    
    if request_type == "agent_details":
        return {
            "agents": learning_engine.agent_capabilities,
            "total_count": len(learning_engine.agent_capabilities)
        }
    elif request_type == "recent_learning":
        return {
            "recent_events": learning_engine.learning_history[-10:],
            "total_events": len(learning_engine.learning_history)
        }
    elif request_type == "performance_summary":
        # Calculate system-wide performance metrics
        if learning_engine.agent_capabilities:
            scores = [caps.get('problem_solving_score', 0.7) for caps in learning_engine.agent_capabilities.values()]
            return {
                "average_performance": np.mean(scores),
                "performance_std": np.std(scores),
                "top_performer": max(learning_engine.agent_capabilities.items(), 
                                   key=lambda x: x[1].get('problem_solving_score', 0))[0],
                "total_agents": len(scores)
            }
        else:
            return {"message": "No agent performance data available"}
    else:
        return {"error": f"Unknown request type: {request_type}"}


async def _get_recent_learning_events() -> List[Dict[str, Any]]:
    """Get recent learning events for SSE streaming"""
    
    # Get recent learning history
    recent_events = learning_engine.learning_history[-5:]  # Last 5 events
    
    formatted_events = []
    for event in recent_events:
        formatted_events.append({
            "event_id": event.get('interaction_id', 'unknown'),
            "agent": event.get('agent', 'unknown'),
            "learning_score": event.get('learning_score', 0),
            "patterns_learned": event.get('patterns_learned', 0),
            "timestamp": event.get('timestamp'),
            "event_type": "learning_update"
        })
    
    return formatted_events
