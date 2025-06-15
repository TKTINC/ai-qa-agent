"""
Agent Intelligence Dashboard
Comprehensive dashboard showcasing agent intelligence and capabilities through
advanced analytics, real-time monitoring, and compelling visualizations.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import json
import logging
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict

# Visualization libraries
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.utils

from src.web.services.agent_visualization import agent_visualization_service, AgentActivity, LearningEvent

# Fallback imports
try:
    from src.agent.learning.learning_engine import AgentLearningEngine
    learning_engine = AgentLearningEngine()
except ImportError:
    class MockLearningEngine:
        async def get_learning_analytics(self, time_window: str):
            return {"learning_rate": 0.15, "improvement_areas": ["testing", "performance"]}
    learning_engine = MockLearningEngine()

logger = logging.getLogger(__name__)

@dataclass
class IntelligenceMetrics:
    """Intelligence metrics for dashboard display"""
    timestamp: datetime
    reasoning_quality_avg: float
    learning_velocity: float
    collaboration_effectiveness: float
    user_satisfaction: float
    problem_solving_accuracy: float
    capability_improvement_rate: float
    system_health_score: float

@dataclass
class AgentPerformanceMetrics:
    """Individual agent performance metrics"""
    agent_name: str
    performance_score: float
    reasoning_quality: float
    learning_rate: float
    collaboration_score: float
    task_completion_rate: float
    user_rating: float
    improvement_velocity: float

class AgentMetricsCollector:
    """Collect and process agent intelligence metrics"""
    
    def __init__(self):
        self.visualization_service = agent_visualization_service
        self.learning_engine = learning_engine
    
    async def collect_intelligence_metrics(self, time_window: timedelta = timedelta(hours=24)) -> IntelligenceMetrics:
        """Collect comprehensive intelligence metrics"""
        cutoff_time = datetime.now() - time_window
        
        # Get system intelligence metrics
        system_metrics = await self.visualization_service.get_system_intelligence_metrics()
        
        # Calculate reasoning quality
        activities = self.visualization_service.agent_activities
        recent_activities = [a for a in activities if a.timestamp >= cutoff_time]
        
        reasoning_activities = [a for a in recent_activities if a.activity_type == "reasoning"]
        reasoning_quality = sum(a.confidence for a in reasoning_activities) / max(len(reasoning_activities), 1)
        
        # Calculate learning velocity
        learning_events = self.visualization_service.learning_events
        recent_learning = [e for e in learning_events if e.timestamp >= cutoff_time]
        learning_velocity = len(recent_learning) / max(time_window.total_seconds() / 3600, 1)  # events per hour
        
        # Calculate collaboration effectiveness
        collaboration_activities = [a for a in recent_activities if a.activity_type == "collaboration"]
        collaboration_effectiveness = sum(a.confidence for a in collaboration_activities) / max(len(collaboration_activities), 1)
        
        # Calculate user satisfaction (mock for demo)
        user_satisfaction = 0.962  # High satisfaction score
        
        # Calculate problem solving accuracy
        response_activities = [a for a in recent_activities if a.activity_type == "response"]
        problem_solving_accuracy = sum(a.confidence for a in response_activities) / max(len(response_activities), 1)
        
        # Calculate capability improvement rate
        improvement_events = [e for e in recent_learning if e.learning_type == "improvement"]
        capability_improvement_rate = sum(e.impact_score for e in improvement_events) / max(len(improvement_events), 1)
        
        # System health score from existing metrics
        system_health_score = system_metrics.get("system_health_score", 0.85)
        
        return IntelligenceMetrics(
            timestamp=datetime.now(),
            reasoning_quality_avg=reasoning_quality,
            learning_velocity=learning_velocity,
            collaboration_effectiveness=collaboration_effectiveness,
            user_satisfaction=user_satisfaction,
            problem_solving_accuracy=problem_solving_accuracy,
            capability_improvement_rate=capability_improvement_rate,
            system_health_score=system_health_score
        )
    
    async def collect_agent_performance_metrics(self, agent_name: str, 
                                              time_window: timedelta = timedelta(hours=24)) -> AgentPerformanceMetrics:
        """Collect performance metrics for specific agent"""
        cutoff_time = datetime.now() - time_window
        
        # Get agent activities
        activities = self.visualization_service.agent_activities
        agent_activities = [a for a in activities if a.agent_name == agent_name and a.timestamp >= cutoff_time]
        
        # Calculate metrics
        total_activities = len(agent_activities)
        
        # Performance score (average confidence)
        performance_score = sum(a.confidence for a in agent_activities) / max(total_activities, 1)
        
        # Reasoning quality
        reasoning_activities = [a for a in agent_activities if a.activity_type == "reasoning"]
        reasoning_quality = sum(a.confidence for a in reasoning_activities) / max(len(reasoning_activities), 1)
        
        # Learning rate
        learning_events = [e for e in self.visualization_service.learning_events 
                          if e.agent_name == agent_name and e.timestamp >= cutoff_time]
        learning_rate = len(learning_events) / max(time_window.total_seconds() / 3600, 1)
        
        # Collaboration score
        collaboration_activities = [a for a in agent_activities if a.activity_type == "collaboration"]
        collaboration_score = sum(a.confidence for a in collaboration_activities) / max(len(collaboration_activities), 1)
        
        # Task completion rate (mock)
        task_completion_rate = min(0.95, performance_score + 0.1)
        
        # User rating (mock based on performance)
        user_rating = min(5.0, performance_score * 5.0)
        
        # Improvement velocity
        improvement_events = [e for e in learning_events if e.learning_type == "improvement"]
        improvement_velocity = sum(e.impact_score for e in improvement_events) / max(len(improvement_events), 1)
        
        return AgentPerformanceMetrics(
            agent_name=agent_name,
            performance_score=performance_score,
            reasoning_quality=reasoning_quality,
            learning_rate=learning_rate,
            collaboration_score=collaboration_score,
            task_completion_rate=task_completion_rate,
            user_rating=user_rating,
            improvement_velocity=improvement_velocity
        )

class IntelligenceVisualizer:
    """Create visualizations for agent intelligence data"""
    
    def __init__(self):
        self.metrics_collector = AgentMetricsCollector()
    
    def create_intelligence_overview_chart(self, metrics: IntelligenceMetrics) -> str:
        """Create comprehensive intelligence overview chart"""
        
        # Create radar chart for intelligence metrics
        categories = [
            'Reasoning Quality',
            'Learning Velocity', 
            'Collaboration',
            'User Satisfaction',
            'Problem Solving',
            'Capability Growth',
            'System Health'
        ]
        
        values = [
            metrics.reasoning_quality_avg,
            metrics.learning_velocity / 2.0,  # Normalize to 0-1 scale
            metrics.collaboration_effectiveness,
            metrics.user_satisfaction,
            metrics.problem_solving_accuracy,
            metrics.capability_improvement_rate,
            metrics.system_health_score
        ]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Current Intelligence',
            line_color='rgb(50, 171, 96)',
            fillcolor='rgba(50, 171, 96, 0.3)'
        ))
        
        # Add benchmark line (target performance)
        benchmark = [0.9, 0.8, 0.85, 0.95, 0.88, 0.7, 0.9]
        fig.add_trace(go.Scatterpolar(
            r=benchmark,
            theta=categories,
            fill=None,
            name='Target Performance',
            line_color='rgb(255, 193, 7)',
            line_dash='dash'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            title={
                'text': '🧠 Agent Intelligence Overview',
                'x': 0.5,
                'font': {'size': 20}
            },
            font=dict(size=12),
            height=500,
            margin=dict(t=80, b=40, l=40, r=40)
        )
        
        return fig.to_json()
    
    def create_learning_velocity_chart(self, time_range: str = "24h") -> str:
        """Create learning velocity trend chart"""
        
        # Generate sample time series data
        now = datetime.now()
        if time_range == "24h":
            times = [now - timedelta(hours=i) for i in range(24, 0, -1)]
            learning_rates = [
                0.1 + 0.05 * np.sin(i/4) + np.random.normal(0, 0.02) 
                for i in range(24)
            ]
        elif time_range == "7d":
            times = [now - timedelta(days=i) for i in range(7, 0, -1)]
            learning_rates = [
                0.15 + 0.03 * np.sin(i/2) + np.random.normal(0, 0.01) 
                for i in range(7)
            ]
        else:  # 30d
            times = [now - timedelta(days=i) for i in range(30, 0, -1)]
            learning_rates = [
                0.12 + 0.02 * np.sin(i/5) + np.random.normal(0, 0.005) 
                for i in range(30)
            ]
        
        fig = go.Figure()
        
        # Add learning velocity line
        fig.add_trace(go.Scatter(
            x=times,
            y=learning_rates,
            mode='lines+markers',
            name='Learning Velocity',
            line=dict(color='rgb(99, 110, 250)', width=3),
            marker=dict(size=6),
            hovertemplate='<b>%{x}</b><br>Learning Rate: %{y:.3f}<extra></extra>'
        ))
        
        # Add trend line
        z = np.polyfit(range(len(learning_rates)), learning_rates, 1)
        trend_line = np.poly1d(z)
        fig.add_trace(go.Scatter(
            x=times,
            y=trend_line(range(len(learning_rates))),
            mode='lines',
            name='Trend',
            line=dict(color='rgb(255, 99, 132)', width=2, dash='dash'),
            hovertemplate='<b>Trend</b><br>%{y:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title={
                'text': f'📈 Learning Velocity Trends ({time_range})',
                'x': 0.5,
                'font': {'size': 18}
            },
            xaxis_title='Time',
            yaxis_title='Learning Rate (events/hour)',
            height=400,
            margin=dict(t=60, b=40, l=60, r=40),
            hovermode='x unified'
        )
        
        return fig.to_json()
    
    def create_agent_performance_comparison(self, agent_metrics: List[AgentPerformanceMetrics]) -> str:
        """Create agent performance comparison chart"""
        
        agent_names = [m.agent_name for m in agent_metrics]
        performance_scores = [m.performance_score for m in agent_metrics]
        learning_rates = [m.learning_rate for m in agent_metrics]
        collaboration_scores = [m.collaboration_score for m in agent_metrics]
        
        # Create grouped bar chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Performance Score',
            x=agent_names,
            y=performance_scores,
            marker_color='rgb(55, 83, 109)',
            hovertemplate='<b>%{x}</b><br>Performance: %{y:.3f}<extra></extra>'
        ))
        
        fig.add_trace(go.Bar(
            name='Learning Rate',
            x=agent_names,
            y=[rate/2 for rate in learning_rates],  # Normalize to 0-1 scale
            marker_color='rgb(26, 118, 255)',
            hovertemplate='<b>%{x}</b><br>Learning Rate: %{customdata:.3f}<extra></extra>',
            customdata=learning_rates
        ))
        
        fig.add_trace(go.Bar(
            name='Collaboration Score',
            x=agent_names,
            y=collaboration_scores,
            marker_color='rgb(50, 171, 96)',
            hovertemplate='<b>%{x}</b><br>Collaboration: %{y:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title={
                'text': '🤖 Agent Performance Comparison',
                'x': 0.5,
                'font': {'size': 18}
            },
            xaxis_title='Agents',
            yaxis_title='Score',
            barmode='group',
            height=400,
            margin=dict(t=60, b=40, l=60, r=40),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig.to_json()
    
    def create_collaboration_network(self, time_window: timedelta = timedelta(hours=24)) -> str:
        """Create collaboration network visualization"""
        
        # Generate sample collaboration data
        agents = ['Test Architect', 'Code Reviewer', 'Performance Analyst', 'Security Specialist', 'Documentation Expert']
        
        # Create network data
        collaborations = [
            ('Test Architect', 'Code Reviewer', 45),
            ('Test Architect', 'Security Specialist', 32),
            ('Performance Analyst', 'Code Reviewer', 28),
            ('Security Specialist', 'Code Reviewer', 25),
            ('Test Architect', 'Performance Analyst', 22),
            ('Documentation Expert', 'Test Architect', 18),
            ('Performance Analyst', 'Security Specialist', 15),
            ('Documentation Expert', 'Code Reviewer', 12)
        ]
        
        # Create network graph
        edge_x = []
        edge_y = []
        edge_info = []
        
        # Position agents in circle
        import math
        n_agents = len(agents)
        positions = {}
        for i, agent in enumerate(agents):
            angle = 2 * math.pi * i / n_agents
            x = math.cos(angle)
            y = math.sin(angle)
            positions[agent] = (x, y)
        
        # Create edges
        for source, target, weight in collaborations:
            x0, y0 = positions[source]
            x1, y1 = positions[target]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_info.append(f"{source} ↔ {target}: {weight} collaborations")
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Create nodes
        node_x = [positions[agent][0] for agent in agents]
        node_y = [positions[agent][1] for agent in agents]
        
        # Calculate node sizes based on collaboration frequency
        node_sizes = []
        for agent in agents:
            total_collaborations = sum(weight for source, target, weight in collaborations 
                                     if source == agent or target == agent)
            node_sizes.append(20 + total_collaborations/2)
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=agents,
            textposition="middle center",
            hovertext=[f"{agent}<br>Total Collaborations: {size-20:.0f}" for agent, size in zip(agents, node_sizes)],
            marker=dict(
                size=node_sizes,
                color=[
                    'rgb(255, 99, 132)',   # Test Architect
                    'rgb(54, 162, 235)',   # Code Reviewer  
                    'rgb(255, 205, 86)',   # Performance Analyst
                    'rgb(75, 192, 192)',   # Security Specialist
                    'rgb(153, 102, 255)'   # Documentation Expert
                ],
                line=dict(width=2, color='rgb(50, 50, 50)')
            )
        )
        
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title={
                               'text': '🤝 Agent Collaboration Network',
                               'x': 0.5,
                               'font': {'size': 18}
                           },
                           titlefont_size=16,
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=40,l=40,r=40,t=80),
                           annotations=[ dict(
                               text="Node size represents collaboration frequency",
                               showarrow=False,
                               xref="paper", yref="paper",
                               x=0.005, y=-0.002,
                               xanchor='left', yanchor='bottom',
                               font=dict(color='rgb(150,150,150)', size=12)
                           )],
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           height=500
                       ))
        
        return fig.to_json()
    
    def create_user_satisfaction_trends(self, time_range: str = "7d") -> str:
        """Create user satisfaction trend chart"""
        
        # Generate sample satisfaction data
        now = datetime.now()
        if time_range == "24h":
            times = [now - timedelta(hours=i) for i in range(24, 0, -1)]
            satisfaction = [
                4.6 + 0.3 * np.sin(i/6) + np.random.normal(0, 0.1) 
                for i in range(24)
            ]
        elif time_range == "7d":
            times = [now - timedelta(days=i) for i in range(7, 0, -1)]
            satisfaction = [
                4.7 + 0.2 * np.sin(i/2) + np.random.normal(0, 0.05) 
                for i in range(7)
            ]
        else:  # 30d
            times = [now - timedelta(days=i) for i in range(30, 0, -1)]
            satisfaction = [
                4.65 + 0.15 * np.sin(i/7) + np.random.normal(0, 0.03) 
                for i in range(30)
            ]
        
        # Ensure satisfaction stays within 1-5 range
        satisfaction = [max(1, min(5, s)) for s in satisfaction]
        
        fig = go.Figure()
        
        # Add satisfaction line with fill
        fig.add_trace(go.Scatter(
            x=times,
            y=satisfaction,
            mode='lines+markers',
            name='User Satisfaction',
            line=dict(color='rgb(50, 171, 96)', width=3),
            marker=dict(size=6),
            fill='tonexty',
            fillcolor='rgba(50, 171, 96, 0.2)',
            hovertemplate='<b>%{x}</b><br>Rating: %{y:.2f}/5.0<extra></extra>'
        ))
        
        # Add average line
        avg_satisfaction = np.mean(satisfaction)
        fig.add_hline(
            y=avg_satisfaction,
            line_dash="dash",
            line_color="rgb(255, 193, 7)",
            annotation_text=f"Average: {avg_satisfaction:.2f}",
            annotation_position="bottom right"
        )
        
        fig.update_layout(
            title={
                'text': f'😊 User Satisfaction Trends ({time_range})',
                'x': 0.5,
                'font': {'size': 18}
            },
            xaxis_title='Time',
            yaxis_title='Rating (1-5)',
            yaxis=dict(range=[1, 5]),
            height=400,
            margin=dict(t=60, b=40, l=60, r=40),
            hovermode='x unified'
        )
        
        return fig.to_json()

class RealTimeMonitor:
    """Real-time monitoring for intelligence dashboard"""
    
    def __init__(self):
        self.visualization_service = agent_visualization_service
        self.metrics_collector = AgentMetricsCollector()
    
    async def get_live_intelligence_data(self) -> Dict[str, Any]:
        """Get live intelligence data for real-time updates"""
        
        # Get current intelligence metrics
        intelligence_metrics = await self.metrics_collector.collect_intelligence_metrics()
        
        # Get agent performance metrics
        agent_names = ["test_architect", "code_reviewer", "performance_analyst", 
                      "security_specialist", "documentation_expert"]
        
        agent_metrics = []
        for agent_name in agent_names:
            metrics = await self.metrics_collector.collect_agent_performance_metrics(agent_name)
            agent_metrics.append(asdict(metrics))
        
        # Get system status
        system_metrics = await self.visualization_service.get_system_intelligence_metrics()
        
        # Get recent activities
        recent_activities = await self.visualization_service.get_live_activity_feed(10)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "intelligence_metrics": asdict(intelligence_metrics),
            "agent_metrics": agent_metrics,
            "system_metrics": system_metrics,
            "recent_activities": recent_activities,
            "status": "operational"
        }
    
    async def generate_intelligence_insights(self) -> List[Dict[str, Any]]:
        """Generate intelligent insights about system performance"""
        
        insights = []
        
        # Get current metrics
        intelligence_metrics = await self.metrics_collector.collect_intelligence_metrics()
        
        # Learning velocity insight
        if intelligence_metrics.learning_velocity > 0.2:
            insights.append({
                "type": "positive",
                "category": "learning",
                "title": "High Learning Velocity Detected",
                "description": f"Agents are learning at {intelligence_metrics.learning_velocity:.2f} events/hour, indicating rapid adaptation",
                "impact": "high",
                "timestamp": datetime.now().isoformat()
            })
        
        # Collaboration effectiveness insight
        if intelligence_metrics.collaboration_effectiveness > 0.85:
            insights.append({
                "type": "positive", 
                "category": "collaboration",
                "title": "Excellent Agent Collaboration",
                "description": f"Collaboration effectiveness at {intelligence_metrics.collaboration_effectiveness:.1%} shows strong teamwork",
                "impact": "medium",
                "timestamp": datetime.now().isoformat()
            })
        
        # User satisfaction insight
        if intelligence_metrics.user_satisfaction > 0.95:
            insights.append({
                "type": "positive",
                "category": "satisfaction",
                "title": "Outstanding User Satisfaction",
                "description": f"User satisfaction at {intelligence_metrics.user_satisfaction:.1%} exceeds targets",
                "impact": "high",
                "timestamp": datetime.now().isoformat()
            })
        
        # System health insight
        if intelligence_metrics.system_health_score < 0.8:
            insights.append({
                "type": "warning",
                "category": "performance",
                "title": "System Health Attention Needed",
                "description": f"System health at {intelligence_metrics.system_health_score:.1%} is below optimal levels",
                "impact": "medium",
                "timestamp": datetime.now().isoformat()
            })
        
        return insights

class AnalyticsEngine:
    """Advanced analytics engine for deeper insights"""
    
    def __init__(self):
        self.visualization_service = agent_visualization_service
    
    async def predict_agent_performance(self, agent_name: str, forecast_days: int = 7) -> Dict[str, Any]:
        """Predict future agent performance using trend analysis"""
        
        # Get historical performance data
        activities = self.visualization_service.agent_activities
        agent_activities = [a for a in activities if a.agent_name == agent_name]
        
        # Create time series data
        daily_performance = {}
        for activity in agent_activities[-30:]:  # Last 30 activities
            day = activity.timestamp.date()
            if day not in daily_performance:
                daily_performance[day] = []
            daily_performance[day].append(activity.confidence)
        
        # Calculate daily averages
        dates = sorted(daily_performance.keys())
        performance_values = [np.mean(daily_performance[date]) for date in dates]
        
        if len(performance_values) < 3:
            return {"error": "Insufficient data for prediction"}
        
        # Simple linear trend prediction
        x = np.arange(len(performance_values))
        z = np.polyfit(x, performance_values, 1)
        trend = np.poly1d(z)
        
        # Predict future values
        future_x = np.arange(len(performance_values), len(performance_values) + forecast_days)
        predictions = trend(future_x)
        
        # Ensure predictions stay within reasonable bounds
        predictions = np.clip(predictions, 0.0, 1.0)
        
        return {
            "agent_name": agent_name,
            "current_performance": performance_values[-1] if performance_values else 0.8,
            "predicted_performance": predictions.tolist(),
            "trend_direction": "increasing" if z[0] > 0 else "decreasing" if z[0] < 0 else "stable",
            "confidence": min(0.9, max(0.6, 1 - abs(z[0]) * 10)),  # Confidence based on trend stability
            "forecast_days": forecast_days
        }
    
    async def analyze_learning_patterns(self) -> Dict[str, Any]:
        """Analyze learning patterns across agents"""
        
        learning_events = self.visualization_service.learning_events
        
        # Group by learning type
        learning_by_type = {}
        for event in learning_events:
            if event.learning_type not in learning_by_type:
                learning_by_type[event.learning_type] = []
            learning_by_type[event.learning_type].append(event)
        
        # Analyze patterns
        patterns = {}
        for learning_type, events in learning_by_type.items():
            impact_scores = [e.impact_score for e in events]
            patterns[learning_type] = {
                "frequency": len(events),
                "average_impact": np.mean(impact_scores) if impact_scores else 0,
                "trend": "increasing" if len(events) > 5 else "stable",
                "effectiveness": np.mean(impact_scores) if impact_scores else 0
            }
        
        # Identify most effective learning type
        most_effective = max(patterns.keys(), key=lambda x: patterns[x]["effectiveness"]) if patterns else None
        
        return {
            "patterns": patterns,
            "most_effective_type": most_effective,
            "total_learning_events": len(learning_events),
            "learning_diversity": len(patterns),
            "overall_learning_health": "excellent" if len(patterns) > 3 else "good"
        }

class IntelligenceDashboard:
    """Main intelligence dashboard orchestrator"""
    
    def __init__(self):
        self.metrics_collector = AgentMetricsCollector()
        self.visualizer = IntelligenceVisualizer()
        self.real_time_monitor = RealTimeMonitor()
        self.analytics_engine = AnalyticsEngine()
    
    async def render_intelligence_overview(self, time_range: str = "24h") -> Dict[str, Any]:
        """Render complete intelligence overview dashboard"""
        
        # Collect metrics
        intelligence_metrics = await self.metrics_collector.collect_intelligence_metrics()
        
        # Get agent performance metrics
        agent_names = ["test_architect", "code_reviewer", "performance_analyst", 
                      "security_specialist", "documentation_expert"]
        
        agent_metrics = []
        for agent_name in agent_names:
            metrics = await self.metrics_collector.collect_agent_performance_metrics(agent_name)
            agent_metrics.append(metrics)
        
        # Generate visualizations
        overview_chart = self.visualizer.create_intelligence_overview_chart(intelligence_metrics)
        learning_chart = self.visualizer.create_learning_velocity_chart(time_range)
        performance_chart = self.visualizer.create_agent_performance_comparison(agent_metrics)
        collaboration_chart = self.visualizer.create_collaboration_network()
        satisfaction_chart = self.visualizer.create_user_satisfaction_trends(time_range)
        
        # Get insights
        insights = await self.real_time_monitor.generate_intelligence_insights()
        
        # Get live data
        live_data = await self.real_time_monitor.get_live_intelligence_data()
        
        return {
            "overview": {
                "metrics": asdict(intelligence_metrics),
                "chart": overview_chart,
                "insights": insights[:3]  # Top 3 insights
            },
            "learning": {
                "chart": learning_chart,
                "velocity": intelligence_metrics.learning_velocity,
                "trend": "increasing" if intelligence_metrics.learning_velocity > 0.15 else "stable"
            },
            "performance": {
                "chart": performance_chart,
                "agents": [asdict(m) for m in agent_metrics],
                "top_performer": max(agent_metrics, key=lambda x: x.performance_score).agent_name
            },
            "collaboration": {
                "chart": collaboration_chart,
                "effectiveness": intelligence_metrics.collaboration_effectiveness,
                "network_health": "excellent" if intelligence_metrics.collaboration_effectiveness > 0.8 else "good"
            },
            "satisfaction": {
                "chart": satisfaction_chart,
                "current_rating": intelligence_metrics.user_satisfaction * 5.0,
                "trend": "positive"
            },
            "live_data": live_data,
            "timestamp": datetime.now().isoformat()
        }

# Global dashboard instance
intelligence_dashboard = IntelligenceDashboard()
