"""
Analytics Web Routes
Handles analytics dashboard rendering, real-time data streaming, 
and interactive chart generation for agent intelligence visualization.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, Optional, List
import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Request, Query, HTTPException, Response
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from src.web.dashboards.intelligence_dashboard import intelligence_dashboard
from src.web.analytics.real_time_analytics import real_time_analytics, predictive_analytics

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/web/analytics", tags=["Analytics"])

# Templates setup - handle missing templates gracefully
try:
    templates = Jinja2Templates(directory="src/web/templates")
except Exception:
    templates = None

class AnalyticsRequest(BaseModel):
    time_range: str = "24h"
    agents: Optional[List[str]] = None
    metrics: Optional[List[str]] = None

class PredictionRequest(BaseModel):
    agent_name: Optional[str] = None
    forecast_hours: int = 24
    prediction_type: str = "performance"

@router.get("/", response_class=HTMLResponse)
async def analytics_dashboard(request: Request):
    """Render the main analytics dashboard"""
    if templates is None:
        return HTMLResponse("""
        <!DOCTYPE html>
        <html>
        <head><title>Analytics Dashboard</title></head>
        <body>
        <h1>🚧 Analytics Dashboard Coming Soon</h1>
        <p>The analytics dashboard template will be available after completing the full setup.</p>
        <p>In the meantime, you can access the API endpoints:</p>
        <ul>
            <li><a href="/web/analytics/api/intelligence-overview">/api/intelligence-overview</a></li>
            <li><a href="/web/analytics/api/live-metrics">/api/live-metrics</a></li>
        </ul>
        </body>
        </html>
        """)
    
    return templates.TemplateResponse(
        "analytics/dashboard.html",
        {
            "request": request,
            "title": "AI Agent Analytics Dashboard",
            "timestamp": datetime.now().isoformat()
        }
    )

@router.get("/api/intelligence-overview")
async def get_intelligence_overview(time_range: str = Query("24h", regex="^(1h|6h|24h|7d|30d)$")):
    """Get comprehensive intelligence overview data"""
    try:
        logger.info(f"Generating intelligence overview for time range: {time_range}")
        
        overview_data = await intelligence_dashboard.render_intelligence_overview(time_range)
        
        return {
            "success": True,
            "data": overview_data,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error generating intelligence overview: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating overview: {str(e)}")

@router.get("/api/live-metrics")
async def get_live_metrics():
    """Get current live metrics"""
    try:
        metrics = await real_time_analytics.process_live_metrics()
        alerts = await real_time_analytics.check_alerts(metrics)
        
        # Convert metrics to serializable format
        metrics_data = {
            name: {
                "metric_name": metric.metric_name,
                "value": metric.value,
                "timestamp": metric.timestamp.isoformat(),
                "unit": metric.unit,
                "category": metric.category,
                "trend": metric.trend
            }
            for name, metric in metrics.items()
        }
        
        return {
            "success": True,
            "metrics": metrics_data,
            "alerts": alerts,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting live metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting metrics: {str(e)}")

@router.get("/api/metric-trends/{metric_name}")
async def get_metric_trends(metric_name: str, time_window: str = Query("1h", regex="^(1h|6h|24h)$")):
    """Get trend data for specific metric"""
    try:
        trends = await real_time_analytics.get_metric_trends(metric_name, time_window)
        
        if "error" in trends:
            raise HTTPException(status_code=404, detail=trends["error"])
        
        return {
            "success": True,
            "trends": trends,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting metric trends: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting trends: {str(e)}")

@router.post("/api/predictions")
async def get_predictions(request: PredictionRequest):
    """Get predictive analytics and forecasts"""
    try:
        predictions = {}
        
        if request.prediction_type == "satisfaction":
            satisfaction_pred = await predictive_analytics.predict_user_satisfaction(request.forecast_hours)
            predictions["user_satisfaction"] = satisfaction_pred
        
        if request.prediction_type == "performance" and request.agent_name:
            performance_pred = await intelligence_dashboard.analytics_engine.predict_agent_performance(
                request.agent_name, request.forecast_hours // 24 or 1
            )
            predictions["agent_performance"] = performance_pred
        
        if request.prediction_type == "all":
            # Get all available predictions
            satisfaction_pred = await predictive_analytics.predict_user_satisfaction(request.forecast_hours)
            predictions["user_satisfaction"] = satisfaction_pred
            
            if request.agent_name:
                performance_pred = await intelligence_dashboard.analytics_engine.predict_agent_performance(
                    request.agent_name, request.forecast_hours // 24 or 1
                )
                predictions["agent_performance"] = performance_pred
        
        return {
            "success": True,
            "predictions": predictions,
            "forecast_hours": request.forecast_hours,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error generating predictions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating predictions: {str(e)}")

@router.get("/api/agent-recommendations/{agent_name}")
async def get_agent_recommendations(agent_name: str):
    """Get improvement recommendations for specific agent"""
    try:
        recommendations = await predictive_analytics.recommend_agent_improvements(agent_name)
        
        if "error" in recommendations:
            raise HTTPException(status_code=404, detail=recommendations["error"])
        
        return {
            "success": True,
            "recommendations": recommendations,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting agent recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting recommendations: {str(e)}")

@router.get("/api/learning-analysis")
async def get_learning_analysis():
    """Get comprehensive learning pattern analysis"""
    try:
        learning_patterns = await intelligence_dashboard.analytics_engine.analyze_learning_patterns()
        
        return {
            "success": True,
            "analysis": learning_patterns,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error analyzing learning patterns: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing learning: {str(e)}")

@router.get("/api/system-health")
async def get_system_health():
    """Get comprehensive system health metrics"""
    try:
        # Get live intelligence data
        live_data = await intelligence_dashboard.real_time_monitor.get_live_intelligence_data()
        
        # Get insights
        insights = await intelligence_dashboard.real_time_monitor.generate_intelligence_insights()
        
        # Calculate health score components
        intelligence_metrics = live_data["intelligence_metrics"]
        
        health_components = {
            "reasoning_quality": intelligence_metrics["reasoning_quality_avg"],
            "learning_velocity": min(1.0, intelligence_metrics["learning_velocity"] / 0.3),  # Normalize
            "collaboration": intelligence_metrics["collaboration_effectiveness"],
            "user_satisfaction": intelligence_metrics["user_satisfaction"],
            "system_stability": intelligence_metrics["system_health_score"]
        }
        
        overall_health = sum(health_components.values()) / len(health_components)
        
        return {
            "success": True,
            "overall_health": overall_health,
            "health_components": health_components,
            "insights": insights,
            "status": "excellent" if overall_health > 0.9 else "good" if overall_health > 0.8 else "needs_attention",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting system health: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting health metrics: {str(e)}")

@router.websocket("/ws/live-analytics")
async def live_analytics_websocket(websocket: WebSocket):
    """WebSocket for real-time analytics streaming"""
    try:
        await websocket.accept()
        await real_time_analytics.subscribe_to_updates(websocket)
        
        logger.info("Live analytics WebSocket connected")
        
        # Send initial data
        initial_data = await real_time_analytics.process_live_metrics()
        await websocket.send_json({
            "type": "initial_data",
            "metrics": {
                name: {
                    "metric_name": metric.metric_name,
                    "value": metric.value,
                    "timestamp": metric.timestamp.isoformat(),
                    "unit": metric.unit,
                    "category": metric.category,
                    "trend": metric.trend
                }
                for name, metric in initial_data.items()
            },
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep connection alive and send periodic updates
        while True:
            try:
                # Send live metrics every 5 seconds
                metrics = await real_time_analytics.process_live_metrics()
                alerts = await real_time_analytics.check_alerts(metrics)
                
                await real_time_analytics.broadcast_update({
                    "metrics": {
                        name: {
                            "metric_name": metric.metric_name,
                            "value": metric.value,
                            "timestamp": metric.timestamp.isoformat(),
                            "unit": metric.unit,
                            "category": metric.category,
                            "trend": metric.trend
                        }
                        for name, metric in metrics.items()
                    },
                    "alerts": alerts
                })
                
                await asyncio.sleep(5)
                
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"Error in live analytics stream: {str(e)}")
                await websocket.send_json({
                    "type": "error",
                    "message": "Analytics streaming error"
                })
                break
                
    except WebSocketDisconnect:
        logger.info("Live analytics WebSocket disconnected")
    except Exception as e:
        logger.error(f"Live analytics WebSocket error: {str(e)}")
    finally:
        await real_time_analytics.unsubscribe_from_updates(websocket)

@router.get("/api/export/dashboard-data")
async def export_dashboard_data(format: str = Query("json", regex="^(json|csv)$"), 
                               time_range: str = Query("24h")):
    """Export dashboard data in various formats"""
    try:
        # Get comprehensive dashboard data
        overview_data = await intelligence_dashboard.render_intelligence_overview(time_range)
        live_data = await real_time_analytics.process_live_metrics()
        
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "time_range": time_range,
            "intelligence_overview": overview_data,
            "live_metrics": {
                name: {
                    "metric_name": metric.metric_name,
                    "value": metric.value,
                    "timestamp": metric.timestamp.isoformat(),
                    "unit": metric.unit,
                    "category": metric.category,
                    "trend": metric.trend
                }
                for name, metric in live_data.items()
            }
        }
        
        if format == "json":
            return JSONResponse(content=export_data)
        elif format == "csv":
            # Convert to CSV format (simplified)
            import pandas as pd
            
            # Create DataFrame from metrics
            metrics_df = pd.DataFrame([
                {
                    "metric_name": metric.metric_name,
                    "value": metric.value,
                    "timestamp": metric.timestamp.isoformat(),
                    "unit": metric.unit,
                    "category": metric.category,
                    "trend": metric.trend
                }
                for metric in live_data.values()
            ])
            
            csv_content = metrics_df.to_csv(index=False)
            
            return Response(
                content=csv_content,
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename=analytics_{time_range}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"}
            )
        
    except Exception as e:
        logger.error(f"Error exporting dashboard data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error exporting data: {str(e)}")

# Chart generation endpoints
@router.get("/api/charts/intelligence-radar")
async def get_intelligence_radar_chart():
    """Generate intelligence radar chart data"""
    try:
        intelligence_metrics = await intelligence_dashboard.metrics_collector.collect_intelligence_metrics()
        chart_json = intelligence_dashboard.visualizer.create_intelligence_overview_chart(intelligence_metrics)
        
        return JSONResponse(content=json.loads(chart_json))
        
    except Exception as e:
        logger.error(f"Error generating radar chart: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating chart: {str(e)}")

@router.get("/api/charts/learning-velocity")
async def get_learning_velocity_chart(time_range: str = Query("24h")):
    """Generate learning velocity chart data"""
    try:
        chart_json = intelligence_dashboard.visualizer.create_learning_velocity_chart(time_range)
        
        return JSONResponse(content=json.loads(chart_json))
        
    except Exception as e:
        logger.error(f"Error generating learning velocity chart: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating chart: {str(e)}")

@router.get("/api/charts/agent-performance")
async def get_agent_performance_chart():
    """Generate agent performance comparison chart"""
    try:
        # Get agent metrics
        agent_names = ["test_architect", "code_reviewer", "performance_analyst", 
                      "security_specialist", "documentation_expert"]
        
        agent_metrics = []
        for agent_name in agent_names:
            metrics = await intelligence_dashboard.metrics_collector.collect_agent_performance_metrics(agent_name)
            agent_metrics.append(metrics)
        
        chart_json = intelligence_dashboard.visualizer.create_agent_performance_comparison(agent_metrics)
        
        return JSONResponse(content=json.loads(chart_json))
        
    except Exception as e:
        logger.error(f"Error generating performance chart: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating chart: {str(e)}")

@router.get("/api/charts/collaboration-network")
async def get_collaboration_network_chart():
    """Generate collaboration network chart data"""
    try:
        chart_json = intelligence_dashboard.visualizer.create_collaboration_network()
        
        return JSONResponse(content=json.loads(chart_json))
        
    except Exception as e:
        logger.error(f"Error generating collaboration chart: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating chart: {str(e)}")

@router.get("/api/charts/satisfaction-trends")
async def get_satisfaction_trends_chart(time_range: str = Query("7d")):
    """Generate user satisfaction trends chart"""
    try:
        chart_json = intelligence_dashboard.visualizer.create_user_satisfaction_trends(time_range)
        
        return JSONResponse(content=json.loads(chart_json))
        
    except Exception as e:
        logger.error(f"Error generating satisfaction chart: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating chart: {str(e)}")
