"""
Learning analytics dashboard data generator.
"""

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import numpy as np

from ...agent.learning.learning_engine import AgentLearningEngine
from ...agent.learning.experience_tracker import ExperienceTracker
from ...agent.learning.feedback_processor import FeedbackProcessor
from ...core.logging import get_logger

logger = get_logger(__name__)


class LearningDashboard:
    """Generate data for learning analytics dashboards"""
    
    def __init__(self):
        self.learning_engine = AgentLearningEngine()
        self.experience_tracker = ExperienceTracker()
        self.feedback_processor = FeedbackProcessor()
    
    async def generate_dashboard_data(self, time_range: str = "24h") -> Dict[str, Any]:
        """Generate comprehensive dashboard data"""
        
        # Get learning insights
        learning_insights = await self.learning_engine.get_learning_insights(time_range)
        
        # Get agent performance data
        agent_performance = await self._get_agent_performance_summary()
        
        # Get user satisfaction trends
        satisfaction_trends = await self.feedback_processor.get_user_satisfaction_trends(time_window=time_range)
        
        # Get learning velocity data
        learning_velocity = await self._get_learning_velocity_data(time_range)
        
        # Get improvement opportunities
        improvement_opportunities = await self._get_improvement_opportunities()
        
        dashboard_data = {
            "overview": {
                "total_interactions": learning_insights.get('total_interactions', 0),
                "average_learning_score": learning_insights.get('average_learning_score', 0.7),
                "learning_velocity": learning_insights.get('learning_velocity', 0.5),
                "agents_active": len(self.learning_engine.agent_capabilities),
                "quality_trend": learning_insights.get('quality_trend', 'stable')
            },
            "agent_performance": agent_performance,
            "satisfaction_trends": satisfaction_trends,
            "learning_velocity": learning_velocity,
            "improvement_opportunities": improvement_opportunities,
            "recent_achievements": await self._get_recent_achievements(),
            "system_health": await self._get_system_health_metrics()
        }
        
        return dashboard_data
    
    async def create_learning_velocity_chart(self, 
                                           agents: Optional[List[str]] = None,
                                           time_period: str = "7d") -> Dict[str, Any]:
        """Create learning velocity visualization data"""
        
        # Filter agents if specified
        target_agents = agents or list(self.learning_engine.agent_capabilities.keys())
        
        chart_data = {
            "labels": [],
            "datasets": []
        }
        
        # Generate time series data
        end_date = datetime.now()
        if time_period == "7d":
            start_date = end_date - timedelta(days=7)
            date_range = [start_date + timedelta(days=i) for i in range(8)]
        elif time_period == "30d":
            start_date = end_date - timedelta(days=30)
            date_range = [start_date + timedelta(days=i*7) for i in range(5)]  # Weekly points
        else:
            start_date = end_date - timedelta(hours=24)
            date_range = [start_date + timedelta(hours=i*4) for i in range(7)]  # 4-hour intervals
        
        chart_data["labels"] = [date.strftime("%Y-%m-%d") for date in date_range]
        
        # Generate data for each agent
        for agent_name in target_agents:
            capabilities = self.learning_engine.agent_capabilities.get(agent_name, {})
            base_velocity = capabilities.get('learning_velocity', 0.5)
            
            # Simulate velocity changes over time (in production, this would be real data)
            velocity_data = []
            for i, date in enumerate(date_range):
                # Add some realistic variation
                variation = np.sin(i * 0.5) * 0.1 + np.random.normal(0, 0.05)
                velocity = max(0, min(1, base_velocity + variation))
                velocity_data.append(round(velocity, 3))
            
            chart_data["datasets"].append({
                "label": agent_name,
                "data": velocity_data,
                "backgroundColor": self._get_agent_color(agent_name),
                "borderColor": self._get_agent_color(agent_name),
                "fill": False
            })
        
        return chart_data
    
    async def create_agent_performance_chart(self, 
                                           metrics: List[str],
                                           comparison_mode: str = "agents") -> Dict[str, Any]:
        """Create agent performance comparison chart"""
        
        chart_data = {
            "labels": [],
            "datasets": []
        }
        
        if comparison_mode == "agents":
            # Compare agents across metrics
            agent_names = list(self.learning_engine.agent_capabilities.keys())
            chart_data["labels"] = agent_names
            
            for metric in metrics:
                metric_data = []
                for agent_name in agent_names:
                    capabilities = self.learning_engine.agent_capabilities.get(agent_name, {})
                    value = self._get_metric_value(capabilities, metric)
                    metric_data.append(value)
                
                chart_data["datasets"].append({
                    "label": metric.replace("_", " ").title(),
                    "data": metric_data,
                    "backgroundColor": self._get_metric_color(metric)
                })
        
        return chart_data
    
    async def create_satisfaction_trends_chart(self, 
                                             granularity: str = "daily",
                                             include_predictions: bool = True) -> Dict[str, Any]:
        """Create user satisfaction trends chart"""
        
        # Get satisfaction data
        satisfaction_data = await self.feedback_processor.get_user_satisfaction_trends(time_window="30d")
        
        chart_data = {
            "labels": [],
            "datasets": []
        }
        
        if satisfaction_data.get('daily_averages'):
            daily_data = satisfaction_data['daily_averages']
            
            # Sort by date
            sorted_dates = sorted(daily_data.keys())
            chart_data["labels"] = sorted_dates
            
            # Historical satisfaction data
            satisfaction_values = [daily_data[date] for date in sorted_dates]
            
            chart_data["datasets"].append({
                "label": "User Satisfaction",
                "data": satisfaction_values,
                "backgroundColor": "rgba(54, 162, 235, 0.2)",
                "borderColor": "rgba(54, 162, 235, 1)",
                "fill": True
            })
            
            # Add predictions if requested
            if include_predictions and len(satisfaction_values) > 3:
                # Simple linear trend prediction
                x = np.arange(len(satisfaction_values))
                y = np.array(satisfaction_values)
                z = np.polyfit(x, y, 1)
                
                # Predict next 7 days
                future_dates = []
                future_values = []
                for i in range(1, 8):
                    future_date = datetime.strptime(sorted_dates[-1], "%Y-%m-%d") + timedelta(days=i)
                    future_dates.append(future_date.strftime("%Y-%m-%d"))
                    future_value = z[0] * (len(satisfaction_values) + i - 1) + z[1]
                    future_values.append(max(1, min(5, future_value)))  # Clamp to 1-5 range
                
                chart_data["labels"].extend(future_dates)
                
                # Add prediction dataset
                prediction_data = [None] * len(satisfaction_values) + future_values
                chart_data["datasets"].append({
                    "label": "Predicted Satisfaction",
                    "data": prediction_data,
                    "backgroundColor": "rgba(255, 206, 86, 0.2)",
                    "borderColor": "rgba(255, 206, 86, 1)",
                    "borderDash": [5, 5],
                    "fill": False
                })
        
        return chart_data
    
    async def _get_agent_performance_summary(self) -> Dict[str, Any]:
        """Get summary of agent performance metrics"""
        
        agent_summary = {}
        
        for agent_name, capabilities in self.learning_engine.agent_capabilities.items():
            agent_summary[agent_name] = {
                "performance_score": capabilities.get('problem_solving_score', 0.7),
                "learning_velocity": capabilities.get('learning_velocity', 0.5),
                "user_satisfaction": capabilities.get('user_communication', 0.7),
                "efficiency_score": capabilities.get('tool_usage_efficiency', 0.7),
                "consistency_score": capabilities.get('consistency_score', 0.7),
                "improvement_trend": "improving"  # Would be calculated from historical data
            }
        
        return agent_summary
    
    async def _get_learning_velocity_data(self, time_range: str) -> Dict[str, Any]:
        """Get learning velocity data"""
        
        learning_insights = await self.learning_engine.get_learning_insights(time_range)
        
        return {
            "overall_velocity": learning_insights.get('learning_velocity', 0.5),
            "velocity_trend": learning_insights.get('quality_trend', 'stable'),
            "top_learning_agents": learning_insights.get('top_learning_agents', []),
            "velocity_distribution": {
                "high": len([a for a in self.learning_engine.agent_capabilities.values() 
                           if a.get('learning_velocity', 0) > 0.7]),
                "medium": len([a for a in self.learning_engine.agent_capabilities.values() 
                             if 0.4 <= a.get('learning_velocity', 0) <= 0.7]),
                "low": len([a for a in self.learning_engine.agent_capabilities.values() 
                          if a.get('learning_velocity', 0) < 0.4])
            }
        }
    
    async def _get_improvement_opportunities(self) -> Dict[str, Any]:
        """Get improvement opportunities data"""
        
        learning_insights = await self.learning_engine.get_learning_insights("30d")
        improvement_opps = learning_insights.get('improvement_opportunities', [])
        
        return {
            "total_opportunities": len(improvement_opps),
            "high_priority": len([opp for opp in improvement_opps if "critical" in opp.lower()]),
            "categories": {
                "communication": len([opp for opp in improvement_opps if "communication" in opp.lower()]),
                "technical": len([opp for opp in improvement_opps if "technical" in opp.lower()]),
                "process": len([opp for opp in improvement_opps if "process" in opp.lower()])
            },
            "recent_opportunities": improvement_opps[:5]  # Top 5 recent
        }
    
    async def _get_recent_achievements(self) -> List[Dict[str, Any]]:
        """Get recent learning achievements"""
        
        achievements = []
        
        # Analyze recent learning history for achievements
        recent_learning = self.learning_engine.learning_history[-10:]
        
        for entry in recent_learning:
            if entry.get('learning_score', 0) > 0.9:
                achievements.append({
                    "type": "high_quality_learning",
                    "agent": entry.get('agent', 'unknown'),
                    "description": f"Achieved exceptional learning quality score of {entry.get('learning_score', 0):.2f}",
                    "timestamp": entry.get('timestamp'),
                    "impact": "high"
                })
        
        # Add capability improvement achievements
        for agent_name, capabilities in self.learning_engine.agent_capabilities.items():
            if capabilities.get('learning_velocity', 0) > 0.8:
                achievements.append({
                    "type": "learning_velocity_milestone",
                    "agent": agent_name,
                    "description": f"Reached high learning velocity of {capabilities.get('learning_velocity', 0):.2f}",
                    "timestamp": datetime.now().isoformat(),
                    "impact": "medium"
                })
        
        return achievements[:5]  # Return top 5 achievements
    
    async def _get_system_health_metrics(self) -> Dict[str, Any]:
        """Get system health metrics"""
        
        total_agents = len(self.learning_engine.agent_capabilities)
        active_learning = len(self.learning_engine.learning_history) > 0
        
        # Calculate health score
        if total_agents > 0:
            avg_performance = np.mean([
                caps.get('problem_solving_score', 0.7) 
                for caps in self.learning_engine.agent_capabilities.values()
            ])
            health_score = avg_performance
        else:
            health_score = 0.5
        
        return {
            "health_score": health_score,
            "status": "healthy" if health_score > 0.7 else "needs_attention" if health_score > 0.5 else "critical",
            "active_agents": total_agents,
            "learning_active": active_learning,
            "last_learning_event": self.learning_engine.learning_history[-1].get('timestamp') if self.learning_engine.learning_history else None,
            "total_learning_events": len(self.learning_engine.learning_history)
        }
    
    def _get_agent_color(self, agent_name: str) -> str:
        """Get color for agent in charts"""
        colors = [
            "rgba(255, 99, 132, 1)",   # Red
            "rgba(54, 162, 235, 1)",   # Blue
            "rgba(255, 205, 86, 1)",   # Yellow
            "rgba(75, 192, 192, 1)",   # Green
            "rgba(153, 102, 255, 1)",  # Purple
            "rgba(255, 159, 64, 1)"    # Orange
        ]
        
        # Use hash of agent name to get consistent color
        agent_hash = hash(agent_name) % len(colors)
        return colors[agent_hash]
    
    def _get_metric_color(self, metric: str) -> str:
        """Get color for metric in charts"""
        metric_colors = {
            "success_rate": "rgba(75, 192, 192, 0.6)",
            "satisfaction": "rgba(54, 162, 235, 0.6)",
            "learning_velocity": "rgba(255, 205, 86, 0.6)",
            "efficiency": "rgba(153, 102, 255, 0.6)",
            "consistency": "rgba(255, 159, 64, 0.6)"
        }
        
        return metric_colors.get(metric, "rgba(128, 128, 128, 0.6)")
    
    def _get_metric_value(self, capabilities: Dict[str, Any], metric: str) -> float:
        """Get metric value from agent capabilities"""
        
        metric_mapping = {
            "success_rate": "problem_solving_score",
            "satisfaction": "user_communication",
            "learning_velocity": "learning_velocity",
            "efficiency": "tool_usage_efficiency",
            "consistency": "consistency_score"
        }
        
        capability_key = metric_mapping.get(metric, metric)
        return capabilities.get(capability_key, 0.7)
