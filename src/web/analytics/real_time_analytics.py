"""
Real-Time Analytics Service
Handles real-time data processing, streaming analytics, and live dashboard updates
for agent intelligence monitoring and visualization.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, AsyncGenerator
import logging
from dataclasses import dataclass, asdict
import numpy as np

from src.web.services.agent_visualization import agent_visualization_service
from src.web.dashboards.intelligence_dashboard import intelligence_dashboard

logger = logging.getLogger(__name__)

@dataclass
class LiveMetric:
    """Live metric data point"""
    metric_name: str
    value: float
    timestamp: datetime
    unit: str
    category: str
    trend: str  # up, down, stable

@dataclass
class AlertCondition:
    """Alert condition definition"""
    metric_name: str
    threshold: float
    operator: str  # gt, lt, eq
    severity: str  # info, warning, critical
    message_template: str

class RealTimeAnalyticsEngine:
    """Real-time analytics processing engine"""
    
    def __init__(self):
        self.visualization_service = agent_visualization_service
        self.dashboard = intelligence_dashboard
        
        # Live metrics storage
        self.live_metrics: Dict[str, List[LiveMetric]] = {}
        self.metric_history_limit = 100
        
        # Alert conditions
        self.alert_conditions = [
            AlertCondition(
                "reasoning_quality", 0.7, "lt", "warning",
                "Reasoning quality below {threshold}: {value:.2f}"
            ),
            AlertCondition(
                "learning_velocity", 0.3, "gt", "info", 
                "High learning velocity detected: {value:.2f} events/hour"
            ),
            AlertCondition(
                "user_satisfaction", 0.8, "lt", "critical",
                "User satisfaction critically low: {value:.1%}"
            ),
            AlertCondition(
                "system_health", 0.75, "lt", "warning",
                "System health requires attention: {value:.1%}"
            )
        ]
        
        # WebSocket subscribers
        self.subscribers: List[Any] = []
    
    async def process_live_metrics(self) -> Dict[str, LiveMetric]:
        """Process and update live metrics"""
        
        # Get intelligence data
        live_data = await self.dashboard.real_time_monitor.get_live_intelligence_data()
        intelligence_metrics = live_data["intelligence_metrics"]
        
        # Create live metrics
        metrics = {}
        
        # Reasoning quality
        reasoning_metric = LiveMetric(
            metric_name="reasoning_quality",
            value=intelligence_metrics["reasoning_quality_avg"],
            timestamp=datetime.now(),
            unit="score",
            category="intelligence",
            trend=self._calculate_trend("reasoning_quality", intelligence_metrics["reasoning_quality_avg"])
        )
        metrics["reasoning_quality"] = reasoning_metric
        self._store_metric(reasoning_metric)
        
        # Learning velocity
        learning_metric = LiveMetric(
            metric_name="learning_velocity", 
            value=intelligence_metrics["learning_velocity"],
            timestamp=datetime.now(),
            unit="events/hour",
            category="learning",
            trend=self._calculate_trend("learning_velocity", intelligence_metrics["learning_velocity"])
        )
        metrics["learning_velocity"] = learning_metric
        self._store_metric(learning_metric)
        
        # User satisfaction
        satisfaction_metric = LiveMetric(
            metric_name="user_satisfaction",
            value=intelligence_metrics["user_satisfaction"],
            timestamp=datetime.now(),
            unit="percentage",
            category="experience",
            trend=self._calculate_trend("user_satisfaction", intelligence_metrics["user_satisfaction"])
        )
        metrics["user_satisfaction"] = satisfaction_metric
        self._store_metric(satisfaction_metric)
        
        # System health
        health_metric = LiveMetric(
            metric_name="system_health",
            value=intelligence_metrics["system_health_score"], 
            timestamp=datetime.now(),
            unit="percentage",
            category="system",
            trend=self._calculate_trend("system_health", intelligence_metrics["system_health_score"])
        )
        metrics["system_health"] = health_metric
        self._store_metric(health_metric)
        
        # Collaboration effectiveness
        collaboration_metric = LiveMetric(
            metric_name="collaboration_effectiveness",
            value=intelligence_metrics["collaboration_effectiveness"],
            timestamp=datetime.now(),
            unit="score",
            category="collaboration", 
            trend=self._calculate_trend("collaboration_effectiveness", intelligence_metrics["collaboration_effectiveness"])
        )
        metrics["collaboration_effectiveness"] = collaboration_metric
        self._store_metric(collaboration_metric)
        
        return metrics
    
    def _calculate_trend(self, metric_name: str, current_value: float) -> str:
        """Calculate trend direction for metric"""
        if metric_name not in self.live_metrics or len(self.live_metrics[metric_name]) < 2:
            return "stable"
        
        recent_values = [m.value for m in self.live_metrics[metric_name][-5:]]
        if len(recent_values) < 2:
            return "stable"
        
        # Simple trend calculation
        avg_recent = np.mean(recent_values[:-1])
        
        if current_value > avg_recent * 1.05:
            return "up"
        elif current_value < avg_recent * 0.95:
            return "down"
        else:
            return "stable"
    
    def _store_metric(self, metric: LiveMetric):
        """Store metric in history"""
        if metric.metric_name not in self.live_metrics:
            self.live_metrics[metric.metric_name] = []
        
        self.live_metrics[metric.metric_name].append(metric)
        
        # Limit history
        if len(self.live_metrics[metric.metric_name]) > self.metric_history_limit:
            self.live_metrics[metric.metric_name] = self.live_metrics[metric.metric_name][-self.metric_history_limit:]
    
    async def check_alerts(self, metrics: Dict[str, LiveMetric]) -> List[Dict[str, Any]]:
        """Check alert conditions and generate alerts"""
        
        alerts = []
        
        for condition in self.alert_conditions:
            if condition.metric_name in metrics:
                metric = metrics[condition.metric_name]
                
                # Check condition
                triggered = False
                if condition.operator == "gt" and metric.value > condition.threshold:
                    triggered = True
                elif condition.operator == "lt" and metric.value < condition.threshold:
                    triggered = True
                elif condition.operator == "eq" and abs(metric.value - condition.threshold) < 0.01:
                    triggered = True
                
                if triggered:
                    alert = {
                        "id": f"{condition.metric_name}_{datetime.now().timestamp()}",
                        "severity": condition.severity,
                        "metric": condition.metric_name,
                        "message": condition.message_template.format(
                            threshold=condition.threshold,
                            value=metric.value
                        ),
                        "timestamp": datetime.now().isoformat(),
                        "category": metric.category
                    }
                    alerts.append(alert)
        
        return alerts
    
    async def get_metric_trends(self, metric_name: str, time_window: str = "1h") -> Dict[str, Any]:
        """Get trend data for specific metric"""
        
        if metric_name not in self.live_metrics:
            return {"error": f"Metric {metric_name} not found"}
        
        # Filter by time window
        cutoff_time = datetime.now()
        if time_window == "1h":
            cutoff_time -= timedelta(hours=1)
        elif time_window == "6h":
            cutoff_time -= timedelta(hours=6)
        elif time_window == "24h":
            cutoff_time -= timedelta(hours=24)
        
        filtered_metrics = [
            m for m in self.live_metrics[metric_name]
            if m.timestamp >= cutoff_time
        ]
        
        if not filtered_metrics:
            return {"error": "No data in time window"}
        
        values = [m.value for m in filtered_metrics]
        times = [m.timestamp.isoformat() for m in filtered_metrics]
        
        # Calculate trend statistics
        if len(values) > 1:
            trend_slope = np.polyfit(range(len(values)), values, 1)[0]
            trend_direction = "increasing" if trend_slope > 0 else "decreasing" if trend_slope < 0 else "stable"
        else:
            trend_direction = "stable"
        
        return {
            "metric_name": metric_name,
            "time_window": time_window,
            "data_points": len(values),
            "values": values,
            "timestamps": times,
            "current_value": values[-1] if values else 0,
            "min_value": min(values) if values else 0,
            "max_value": max(values) if values else 0,
            "avg_value": np.mean(values) if values else 0,
            "trend_direction": trend_direction,
            "volatility": np.std(values) if len(values) > 1 else 0
        }
    
    async def subscribe_to_updates(self, websocket):
        """Subscribe websocket to real-time updates"""
        self.subscribers.append(websocket)
        logger.info(f"Added analytics subscriber, total: {len(self.subscribers)}")
    
    async def unsubscribe_from_updates(self, websocket):
        """Unsubscribe websocket from updates"""
        if websocket in self.subscribers:
            self.subscribers.remove(websocket)
            logger.info(f"Removed analytics subscriber, total: {len(self.subscribers)}")
    
    async def broadcast_update(self, data: Dict[str, Any]):
        """Broadcast update to all subscribers"""
        if not self.subscribers:
            return
        
        message = {
            "type": "analytics_update",
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
        
        # Send to all subscribers
        failed_subscribers = []
        for subscriber in self.subscribers:
            try:
                await subscriber.send_json(message)
            except Exception as e:
                logger.error(f"Failed to send to subscriber: {e}")
                failed_subscribers.append(subscriber)
        
        # Remove failed subscribers
        for failed in failed_subscribers:
            self.subscribers.remove(failed)

class PredictiveAnalytics:
    """Predictive analytics for forecasting agent behavior"""
    
    def __init__(self):
        self.real_time_engine = RealTimeAnalyticsEngine()
    
    async def predict_user_satisfaction(self, forecast_hours: int = 24) -> Dict[str, Any]:
        """Predict user satisfaction trends"""
        
        # Get historical satisfaction data
        satisfaction_history = self.real_time_engine.live_metrics.get("user_satisfaction", [])
        
        if len(satisfaction_history) < 5:
            return {"error": "Insufficient data for prediction"}
        
        # Extract values and create time series
        values = [m.value for m in satisfaction_history[-20:]]  # Last 20 data points
        
        # Simple moving average prediction
        window_size = min(5, len(values))
        recent_avg = np.mean(values[-window_size:])
        
        # Calculate trend
        if len(values) >= 2:
            trend = (values[-1] - values[-2]) * 0.1  # Dampened trend
        else:
            trend = 0
        
        # Generate predictions
        predictions = []
        current_value = recent_avg
        
        for hour in range(forecast_hours):
            # Add some realistic variation
            noise = np.random.normal(0, 0.02)
            current_value = max(0.5, min(1.0, current_value + trend + noise))
            predictions.append(current_value)
        
        return {
            "current_satisfaction": values[-1] if values else 0.9,
            "predicted_values": predictions,
            "forecast_hours": forecast_hours,
            "trend": "increasing" if trend > 0 else "decreasing" if trend < 0 else "stable",
            "confidence": min(0.9, 0.7 + len(values) * 0.01),
            "recommendations": self._generate_satisfaction_recommendations(recent_avg, trend)
        }
    
    def _generate_satisfaction_recommendations(self, current_avg: float, trend: float) -> List[str]:
        """Generate recommendations based on satisfaction trends"""
        
        recommendations = []
        
        if current_avg < 0.8:
            recommendations.append("Focus on improving response quality and accuracy")
            recommendations.append("Consider additional user feedback collection")
        
        if trend < 0:
            recommendations.append("Investigate recent changes that may have impacted satisfaction")
            recommendations.append("Increase monitoring frequency for early issue detection")
        
        if current_avg > 0.95 and trend > 0:
            recommendations.append("Satisfaction is excellent - maintain current practices")
            recommendations.append("Consider documenting successful patterns for replication")
        
        return recommendations
    
    async def recommend_agent_improvements(self, agent_name: str) -> Dict[str, Any]:
        """Generate improvement recommendations for specific agent"""
        
        # Get agent performance data from dashboard
        try:
            agent_metrics = await intelligence_dashboard.metrics_collector.collect_agent_performance_metrics(agent_name)
        except Exception as e:
            return {"error": f"Failed to get agent metrics: {str(e)}"}
        
        recommendations = []
        priority_areas = []
        
        # Analyze performance areas
        if agent_metrics.reasoning_quality < 0.8:
            recommendations.append("Improve reasoning algorithms and confidence calculation")
            priority_areas.append("reasoning")
        
        if agent_metrics.learning_rate < 0.1:
            recommendations.append("Increase learning opportunities and feedback processing")
            priority_areas.append("learning")
        
        if agent_metrics.collaboration_score < 0.7:
            recommendations.append("Enhance collaboration protocols and communication")
            priority_areas.append("collaboration")
        
        if agent_metrics.performance_score > 0.9:
            recommendations.append("Performance is excellent - focus on knowledge sharing")
        
        return {
            "agent_name": agent_name,
            "current_score": agent_metrics.performance_score,
            "recommendations": recommendations,
            "priority_areas": priority_areas,
            "improvement_potential": max(0, 0.95 - agent_metrics.performance_score),
            "next_review": (datetime.now() + timedelta(days=7)).isoformat()
        }

# Global analytics engine
real_time_analytics = RealTimeAnalyticsEngine()
predictive_analytics = PredictiveAnalytics()
