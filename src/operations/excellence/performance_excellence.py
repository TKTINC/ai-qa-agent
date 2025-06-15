"""
Performance Excellence Monitoring System

This module implements comprehensive performance excellence tracking,
SLA monitoring, and continuous improvement for production AI agent systems.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import numpy as np
from collections import defaultdict, deque
import redis.asyncio as redis

from src.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

class SLAStatus(Enum):
    """SLA compliance status"""
    MEETING = "meeting"
    AT_RISK = "at_risk"
    BREACHED = "breached"
    CRITICAL = "critical"

class PerformanceMetric(Enum):
    """Performance metrics tracked"""
    UPTIME = "uptime"
    RESPONSE_TIME = "response_time"
    ERROR_RATE = "error_rate"
    THROUGHPUT = "throughput"
    AVAILABILITY = "availability"
    REASONING_QUALITY = "reasoning_quality"
    USER_SATISFACTION = "user_satisfaction"
    LEARNING_VELOCITY = "learning_velocity"

@dataclass
class ExcellenceMetrics:
    """Performance excellence metrics snapshot"""
    timestamp: datetime
    uptime_percentage: float
    mean_time_to_recovery: float  # minutes
    error_rate: float  # percentage
    response_time_p95: float  # milliseconds
    throughput_per_second: float
    resource_utilization: float  # percentage
    agent_reasoning_quality: float
    learning_velocity: float
    user_satisfaction: float
    deployment_frequency: float  # per week
    change_failure_rate: float  # percentage
    security_incidents: int
    compliance_score: float

@dataclass
class SLATarget:
    """SLA target definition"""
    metric: PerformanceMetric
    target_value: float
    measurement_window: str  # e.g., "monthly", "weekly", "daily"
    tolerance: float  # acceptable variance
    critical_threshold: float  # threshold for critical alerts
    measurement_unit: str

@dataclass
class SLAComplianceReport:
    """SLA compliance report"""
    reporting_period: str
    overall_compliance: float  # percentage
    metric_compliance: Dict[str, float]
    breached_slas: List[str]
    at_risk_slas: List[str]
    improvement_trends: Dict[str, float]
    recommendations: List[str]
    next_review_date: datetime

@dataclass
class PerformanceInsight:
    """Performance insight with actionable recommendations"""
    insight_type: str
    title: str
    description: str
    impact_assessment: str
    confidence: float
    recommended_actions: List[str]
    estimated_improvement: float
    implementation_effort: str

class PerformanceTracker:
    """Tracks and analyzes performance metrics"""
    
    def __init__(self):
        self.metrics_history = defaultdict(deque)
        self.performance_baselines = {}
        self.trend_analysis_window = 168  # 7 days in hours
        self.redis_client = None
    
    async def initialize(self):
        """Initialize performance tracker"""
        try:
            self.redis_client = redis.Redis(
                host=settings.redis_host,
                port=settings.redis_port,
                decode_responses=True
            )
            await self.redis_client.ping()
            
            # Load historical metrics
            await self._load_historical_metrics()
            
            # Establish performance baselines
            await self._establish_baselines()
            
            logger.info("Performance tracker initialized")
            
        except Exception as e:
            logger.error(f"Performance tracker initialization failed: {e}")
    
    async def record_performance_metric(self, metric: PerformanceMetric, value: float, timestamp: Optional[datetime] = None):
        """Record a performance metric"""
        if timestamp is None:
            timestamp = datetime.now()
        
        # Store in memory with timestamp
        metric_data = {'value': value, 'timestamp': timestamp}
        self.metrics_history[metric.value].append(metric_data)
        
        # Keep only recent data in memory (last 7 days)
        cutoff_time = timestamp - timedelta(days=7)
        while (self.metrics_history[metric.value] and 
               self.metrics_history[metric.value][0]['timestamp'] < cutoff_time):
            self.metrics_history[metric.value].popleft()
        
        # Store in Redis for persistence
        if self.redis_client:
            try:
                await self.redis_client.lpush(
                    f"metrics:{metric.value}",
                    json.dumps(metric_data, default=str)
                )
                # Keep last 10000 metrics per type
                await self.redis_client.ltrim(f"metrics:{metric.value}", 0, 9999)
            except Exception as e:
                logger.error(f"Failed to store metric in Redis: {e}")
    
    async def get_current_performance_snapshot(self) -> ExcellenceMetrics:
        """Get current performance excellence snapshot"""
        try:
            current_time = datetime.now()
            
            # Calculate current metrics
            uptime = await self._calculate_uptime()
            mttr = await self._calculate_mean_time_to_recovery()
            error_rate = await self._calculate_error_rate()
            response_time = await self._calculate_response_time_p95()
            throughput = await self._calculate_throughput()
            resource_util = await self._calculate_resource_utilization()
            reasoning_quality = await self._get_latest_metric(PerformanceMetric.REASONING_QUALITY, 0.90)
            learning_velocity = await self._get_latest_metric(PerformanceMetric.LEARNING_VELOCITY, 0.12)
            user_satisfaction = await self._get_latest_metric(PerformanceMetric.USER_SATISFACTION, 0.85)
            
            # Operational metrics
            deployment_freq = await self._calculate_deployment_frequency()
            change_failure_rate = await self._calculate_change_failure_rate()
            security_incidents = await self._count_security_incidents()
            compliance_score = await self._calculate_compliance_score()
            
            return ExcellenceMetrics(
                timestamp=current_time,
                uptime_percentage=uptime,
                mean_time_to_recovery=mttr,
                error_rate=error_rate,
                response_time_p95=response_time,
                throughput_per_second=throughput,
                resource_utilization=resource_util,
                agent_reasoning_quality=reasoning_quality,
                learning_velocity=learning_velocity,
                user_satisfaction=user_satisfaction,
                deployment_frequency=deployment_freq,
                change_failure_rate=change_failure_rate,
                security_incidents=security_incidents,
                compliance_score=compliance_score
            )
            
        except Exception as e:
            logger.error(f"Failed to get performance snapshot: {e}")
            return await self._get_fallback_metrics()
    
    async def analyze_performance_trends(self, days_back: int = 7) -> Dict[str, Any]:
        """Analyze performance trends over specified period"""
        try:
            trends = {}
            cutoff_time = datetime.now() - timedelta(days=days_back)
            
            for metric_name, metric_history in self.metrics_history.items():
                # Filter to time period
                recent_data = [
                    data for data in metric_history 
                    if data['timestamp'] > cutoff_time
                ]
                
                if len(recent_data) < 2:
                    continue
                
                # Calculate trend
                values = [data['value'] for data in recent_data]
                timestamps = [(data['timestamp'] - cutoff_time).total_seconds() for data in recent_data]
                
                # Simple linear trend
                if len(values) >= 3:
                    trend_slope = np.polyfit(timestamps, values, 1)[0]
                    trend_direction = "improving" if trend_slope > 0 else "declining" if trend_slope < 0 else "stable"
                    
                    # Calculate trend strength
                    correlation = abs(np.corrcoef(timestamps, values)[0, 1]) if len(timestamps) > 1 else 0
                    trend_strength = "strong" if correlation > 0.7 else "moderate" if correlation > 0.4 else "weak"
                    
                    trends[metric_name] = {
                        'direction': trend_direction,
                        'strength': trend_strength,
                        'slope': trend_slope,
                        'current_value': values[-1],
                        'period_start_value': values[0],
                        'change_percentage': ((values[-1] - values[0]) / values[0] * 100) if values[0] != 0 else 0
                    }
            
            return trends
            
        except Exception as e:
            logger.error(f"Trend analysis failed: {e}")
            return {}
    
    async def _calculate_uptime(self) -> float:
        """Calculate system uptime percentage"""
        # Get availability data for last 24 hours
        availability_data = list(self.metrics_history.get(PerformanceMetric.AVAILABILITY.value, []))
        
        if not availability_data:
            return 99.5  # Default assumption
        
        # Calculate uptime from availability metrics
        recent_data = [
            data for data in availability_data 
            if data['timestamp'] > datetime.now() - timedelta(hours=24)
        ]
        
        if recent_data:
            uptime_values = [data['value'] for data in recent_data]
            return np.mean(uptime_values) * 100
        
        return 99.5
    
    async def _calculate_mean_time_to_recovery(self) -> float:
        """Calculate mean time to recovery in minutes"""
        # Simulate MTTR calculation based on incident data
        # In production, this would analyze actual incident resolution times
        
        # Look for error rate spikes and recovery patterns
        error_data = list(self.metrics_history.get(PerformanceMetric.ERROR_RATE.value, []))
        
        if len(error_data) < 10:
            return 15.0  # Default MTTR
        
        # Simple heuristic: time between error spikes and return to normal
        recovery_times = []
        normal_error_rate = 0.02  # 2% normal error rate
        
        in_incident = False
        incident_start = None
        
        for data in error_data[-100:]:  # Last 100 data points
            if data['value'] > normal_error_rate * 2 and not in_incident:
                # Incident started
                in_incident = True
                incident_start = data['timestamp']
            elif data['value'] <= normal_error_rate and in_incident and incident_start:
                # Incident resolved
                recovery_time = (data['timestamp'] - incident_start).total_seconds() / 60
                recovery_times.append(recovery_time)
                in_incident = False
                incident_start = None
        
        if recovery_times:
            return np.mean(recovery_times)
        
        return 15.0
    
    async def _calculate_error_rate(self) -> float:
        """Calculate current error rate"""
        error_data = list(self.metrics_history.get(PerformanceMetric.ERROR_RATE.value, []))
        
        if error_data:
            # Get recent error rate (last hour)
            recent_threshold = datetime.now() - timedelta(hours=1)
            recent_data = [data for data in error_data if data['timestamp'] > recent_threshold]
            
            if recent_data:
                return np.mean([data['value'] for data in recent_data]) * 100
        
        return 0.5  # Default 0.5% error rate
    
    async def _calculate_response_time_p95(self) -> float:
        """Calculate 95th percentile response time"""
        response_data = list(self.metrics_history.get(PerformanceMetric.RESPONSE_TIME.value, []))
        
        if response_data:
            # Get recent response times (last hour)
            recent_threshold = datetime.now() - timedelta(hours=1)
            recent_data = [data for data in response_data if data['timestamp'] > recent_threshold]
            
            if recent_data:
                values = [data['value'] for data in recent_data]
                return np.percentile(values, 95)
        
        return 450.0  # Default P95 response time
    
    async def _calculate_throughput(self) -> float:
        """Calculate requests per second throughput"""
        throughput_data = list(self.metrics_history.get(PerformanceMetric.THROUGHPUT.value, []))
        
        if throughput_data:
            # Get recent throughput (last 10 minutes)
            recent_threshold = datetime.now() - timedelta(minutes=10)
            recent_data = [data for data in throughput_data if data['timestamp'] > recent_threshold]
            
            if recent_data:
                return np.mean([data['value'] for data in recent_data])
        
        return 25.0  # Default 25 RPS
    
    async def _calculate_resource_utilization(self) -> float:
        """Calculate average resource utilization"""
        # Simulate resource utilization calculation
        # In production, this would aggregate CPU, memory, disk, network utilization
        return 72.5  # Default 72.5% utilization
    
    async def _get_latest_metric(self, metric: PerformanceMetric, default: float) -> float:
        """Get latest value for a specific metric"""
        metric_data = list(self.metrics_history.get(metric.value, []))
        
        if metric_data:
            return metric_data[-1]['value']
        
        return default
    
    async def _calculate_deployment_frequency(self) -> float:
        """Calculate deployment frequency per week"""
        # Simulate deployment frequency calculation
        # In production, this would track actual deployments
        return 7.0  # Default 7 deployments per week
    
    async def _calculate_change_failure_rate(self) -> float:
        """Calculate change failure rate percentage"""
        # Simulate change failure rate calculation
        # In production, this would track deployment failures
        return 2.0  # Default 2% change failure rate
    
    async def _count_security_incidents(self) -> int:
        """Count security incidents in current period"""
        # Simulate security incident counting
        # In production, this would query security incident database
        return 0  # Default no incidents
    
    async def _calculate_compliance_score(self) -> float:
        """Calculate overall compliance score"""
        # Simulate compliance score calculation
        # In production, this would aggregate compliance metrics
        return 1.0  # Default 100% compliance
    
    async def _load_historical_metrics(self):
        """Load historical metrics from Redis"""
        if not self.redis_client:
            return
        
        try:
            for metric in PerformanceMetric:
                metric_data = await self.redis_client.lrange(f"metrics:{metric.value}", 0, 999)
                for data_str in metric_data:
                    try:
                        data = json.loads(data_str)
                        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
                        self.metrics_history[metric.value].append(data)
                    except (json.JSONDecodeError, ValueError):
                        continue
        except Exception as e:
            logger.error(f"Failed to load historical metrics: {e}")
    
    async def _establish_baselines(self):
        """Establish performance baselines from historical data"""
        for metric_name, metric_history in self.metrics_history.items():
            if len(metric_history) >= 10:
                values = [data['value'] for data in metric_history]
                self.performance_baselines[metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'p50': np.percentile(values, 50),
                    'p95': np.percentile(values, 95),
                    'p99': np.percentile(values, 99)
                }
    
    async def _get_fallback_metrics(self) -> ExcellenceMetrics:
        """Get fallback metrics when calculation fails"""
        return ExcellenceMetrics(
            timestamp=datetime.now(),
            uptime_percentage=99.5,
            mean_time_to_recovery=15.0,
            error_rate=0.5,
            response_time_p95=450.0,
            throughput_per_second=25.0,
            resource_utilization=70.0,
            agent_reasoning_quality=0.90,
            learning_velocity=0.12,
            user_satisfaction=0.85,
            deployment_frequency=7.0,
            change_failure_rate=2.0,
            security_incidents=0,
            compliance_score=1.0
        )

class SLAMonitor:
    """Monitors SLA compliance and generates reports"""
    
    def __init__(self):
        self.sla_targets = self._initialize_sla_targets()
        self.compliance_history = []
    
    def _initialize_sla_targets(self) -> Dict[str, SLATarget]:
        """Initialize SLA targets for different metrics"""
        return {
            'uptime': SLATarget(
                metric=PerformanceMetric.UPTIME,
                target_value=99.9,
                measurement_window="monthly",
                tolerance=0.1,
                critical_threshold=99.0,
                measurement_unit="percentage"
            ),
            'response_time': SLATarget(
                metric=PerformanceMetric.RESPONSE_TIME,
                target_value=500.0,
                measurement_window="daily",
                tolerance=100.0,
                critical_threshold=1000.0,
                measurement_unit="milliseconds"
            ),
            'error_rate': SLATarget(
                metric=PerformanceMetric.ERROR_RATE,
                target_value=1.0,
                measurement_window="daily",
                tolerance=0.5,
                critical_threshold=5.0,
                measurement_unit="percentage"
            ),
            'reasoning_quality': SLATarget(
                metric=PerformanceMetric.REASONING_QUALITY,
                target_value=0.85,
                measurement_window="weekly",
                tolerance=0.05,
                critical_threshold=0.75,
                measurement_unit="score"
            ),
            'user_satisfaction': SLATarget(
                metric=PerformanceMetric.USER_SATISFACTION,
                target_value=0.90,
                measurement_window="monthly",
                tolerance=0.05,
                critical_threshold=0.80,
                measurement_unit="score"
            )
        }
    
    async def check_sla_compliance(self, current_metrics: ExcellenceMetrics) -> SLAComplianceReport:
        """Check current SLA compliance status"""
        try:
            metric_compliance = {}
            breached_slas = []
            at_risk_slas = []
            
            # Check each SLA target
            for sla_name, sla_target in self.sla_targets.items():
                current_value = await self._get_metric_value(current_metrics, sla_target.metric)
                compliance_status = await self._check_metric_compliance(current_value, sla_target)
                
                # Calculate compliance percentage
                if sla_target.metric in [PerformanceMetric.ERROR_RATE]:
                    # Lower is better for error rate
                    compliance_pct = max(0, (sla_target.target_value - current_value) / sla_target.target_value * 100)
                else:
                    # Higher is better for most metrics
                    compliance_pct = min(100, current_value / sla_target.target_value * 100)
                
                metric_compliance[sla_name] = compliance_pct
                
                # Check compliance status
                if compliance_status == SLAStatus.BREACHED:
                    breached_slas.append(sla_name)
                elif compliance_status == SLAStatus.AT_RISK:
                    at_risk_slas.append(sla_name)
            
            # Calculate overall compliance
            overall_compliance = np.mean(list(metric_compliance.values()))
            
            # Generate improvement trends
            improvement_trends = await self._calculate_improvement_trends()
            
            # Generate recommendations
            recommendations = await self._generate_sla_recommendations(breached_slas, at_risk_slas, metric_compliance)
            
            return SLAComplianceReport(
                reporting_period=f"{datetime.now().strftime('%Y-%m')} (Current)",
                overall_compliance=overall_compliance,
                metric_compliance=metric_compliance,
                breached_slas=breached_slas,
                at_risk_slas=at_risk_slas,
                improvement_trends=improvement_trends,
                recommendations=recommendations,
                next_review_date=datetime.now() + timedelta(days=7)
            )
            
        except Exception as e:
            logger.error(f"SLA compliance check failed: {e}")
            return await self._get_fallback_sla_report()
    
    async def _get_metric_value(self, metrics: ExcellenceMetrics, metric_type: PerformanceMetric) -> float:
        """Get metric value from excellence metrics"""
        mapping = {
            PerformanceMetric.UPTIME: metrics.uptime_percentage,
            PerformanceMetric.RESPONSE_TIME: metrics.response_time_p95,
            PerformanceMetric.ERROR_RATE: metrics.error_rate,
            PerformanceMetric.REASONING_QUALITY: metrics.agent_reasoning_quality,
            PerformanceMetric.USER_SATISFACTION: metrics.user_satisfaction,
            PerformanceMetric.THROUGHPUT: metrics.throughput_per_second
        }
        return mapping.get(metric_type, 0.0)
    
    async def _check_metric_compliance(self, current_value: float, sla_target: SLATarget) -> SLAStatus:
        """Check compliance status for a specific metric"""
        if sla_target.metric == PerformanceMetric.ERROR_RATE:
            # Lower is better for error rate
            if current_value >= sla_target.critical_threshold:
                return SLAStatus.BREACHED
            elif current_value > sla_target.target_value + sla_target.tolerance:
                return SLAStatus.AT_RISK
            else:
                return SLAStatus.MEETING
        else:
            # Higher is better for most metrics
            if current_value < sla_target.critical_threshold:
                return SLAStatus.BREACHED
            elif current_value < sla_target.target_value - sla_target.tolerance:
                return SLAStatus.AT_RISK
            else:
                return SLAStatus.MEETING
    
    async def _calculate_improvement_trends(self) -> Dict[str, float]:
        """Calculate improvement trends for SLA metrics"""
        # Simulate trend calculation
        # In production, this would analyze historical SLA compliance data
        return {
            'uptime': 0.05,  # 0.05% improvement
            'response_time': -2.5,  # 2.5% improvement (negative is good for response time)
            'error_rate': -0.1,  # 0.1% improvement (negative is good for error rate)
            'reasoning_quality': 1.2,  # 1.2% improvement
            'user_satisfaction': 0.8  # 0.8% improvement
        }
    
    async def _generate_sla_recommendations(self, breached: List[str], at_risk: List[str], compliance: Dict[str, float]) -> List[str]:
        """Generate recommendations for SLA improvement"""
        recommendations = []
        
        # Recommendations for breached SLAs
        for sla in breached:
            if sla == 'uptime':
                recommendations.append("CRITICAL: Implement redundancy and failover mechanisms to improve uptime")
            elif sla == 'response_time':
                recommendations.append("CRITICAL: Optimize application performance and consider scaling up")
            elif sla == 'error_rate':
                recommendations.append("CRITICAL: Investigate and fix underlying causes of errors")
            elif sla == 'reasoning_quality':
                recommendations.append("CRITICAL: Review and improve AI model performance")
            elif sla == 'user_satisfaction':
                recommendations.append("CRITICAL: Analyze user feedback and improve user experience")
        
        # Recommendations for at-risk SLAs
        for sla in at_risk:
            if sla == 'uptime':
                recommendations.append("Monitor uptime closely and prepare contingency plans")
            elif sla == 'response_time':
                recommendations.append("Consider performance optimization and caching improvements")
            elif sla == 'error_rate':
                recommendations.append("Increase error monitoring and implement preventive measures")
            elif sla == 'reasoning_quality':
                recommendations.append("Review reasoning algorithms and consider model updates")
            elif sla == 'user_satisfaction':
                recommendations.append("Gather more user feedback and identify improvement areas")
        
        # General recommendations
        if not breached and not at_risk:
            recommendations.append("All SLAs are meeting targets - continue current practices")
            recommendations.append("Consider raising SLA targets to drive further improvement")
        
        recommendations.append("Schedule regular SLA review meetings with stakeholders")
        recommendations.append("Implement automated SLA monitoring and alerting")
        
        return recommendations
    
    async def _get_fallback_sla_report(self) -> SLAComplianceReport:
        """Get fallback SLA report when monitoring fails"""
        return SLAComplianceReport(
            reporting_period="Current (Fallback)",
            overall_compliance=95.0,
            metric_compliance={'uptime': 99.0, 'response_time': 90.0, 'error_rate': 95.0},
            breached_slas=[],
            at_risk_slas=[],
            improvement_trends={},
            recommendations=["Review monitoring system health"],
            next_review_date=datetime.now() + timedelta(days=7)
        )

class InsightGenerator:
    """Generates actionable performance insights"""
    
    def __init__(self):
        self.insight_patterns = self._initialize_insight_patterns()
    
    def _initialize_insight_patterns(self) -> List[Dict[str, Any]]:
        """Initialize patterns for generating insights"""
        return [
            {
                'name': 'response_time_degradation',
                'condition': lambda metrics: metrics.response_time_p95 > 800,
                'insight_type': 'performance_issue',
                'title': 'Response Time Degradation Detected',
                'impact': 'High - User experience affected',
                'actions': [
                    'Analyze slow database queries',
                    'Review application performance profiling',
                    'Consider scaling up resources',
                    'Implement response time optimization'
                ]
            },
            {
                'name': 'high_resource_utilization',
                'condition': lambda metrics: metrics.resource_utilization > 85,
                'insight_type': 'capacity_issue',
                'title': 'High Resource Utilization',
                'impact': 'Medium - System approaching capacity limits',
                'actions': [
                    'Plan capacity scaling',
                    'Optimize resource-intensive processes',
                    'Review auto-scaling policies',
                    'Consider workload balancing'
                ]
            },
            {
                'name': 'reasoning_quality_decline',
                'condition': lambda metrics: metrics.agent_reasoning_quality < 0.85,
                'insight_type': 'quality_issue',
                'title': 'Agent Reasoning Quality Below Target',
                'impact': 'High - Core functionality affected',
                'actions': [
                    'Review reasoning model performance',
                    'Analyze recent reasoning failures',
                    'Consider model retraining',
                    'Update reasoning algorithms'
                ]
            },
            {
                'name': 'user_satisfaction_decline',
                'condition': lambda metrics: metrics.user_satisfaction < 0.80,
                'insight_type': 'user_experience_issue',
                'title': 'User Satisfaction Below Expectations',
                'impact': 'High - User retention at risk',
                'actions': [
                    'Conduct user experience analysis',
                    'Gather detailed user feedback',
                    'Review conversation quality',
                    'Implement user experience improvements'
                ]
            },
            {
                'name': 'excellent_performance',
                'condition': lambda metrics: (metrics.uptime_percentage > 99.9 and 
                                            metrics.response_time_p95 < 300 and 
                                            metrics.agent_reasoning_quality > 0.95),
                'insight_type': 'positive_trend',
                'title': 'Excellent Performance Achievement',
                'impact': 'Positive - System performing exceptionally well',
                'actions': [
                    'Document current best practices',
                    'Share success factors with team',
                    'Consider raising performance targets',
                    'Investigate optimization opportunities'
                ]
            }
        ]
    
    async def generate_performance_insights(self, metrics: ExcellenceMetrics, trends: Dict[str, Any]) -> List[PerformanceInsight]:
        """Generate actionable performance insights"""
        insights = []
        
        try:
            # Check each insight pattern
            for pattern in self.insight_patterns:
                if pattern['condition'](metrics):
                    insight = await self._create_insight_from_pattern(pattern, metrics, trends)
                    insights.append(insight)
            
            # Generate trend-based insights
            trend_insights = await self._generate_trend_insights(trends)
            insights.extend(trend_insights)
            
            # Sort insights by impact and confidence
            insights.sort(key=lambda x: (self._impact_score(x.impact_assessment), x.confidence), reverse=True)
            
            return insights[:10]  # Return top 10 insights
            
        except Exception as e:
            logger.error(f"Insight generation failed: {e}")
            return []
    
    async def _create_insight_from_pattern(self, pattern: Dict[str, Any], metrics: ExcellenceMetrics, trends: Dict[str, Any]) -> PerformanceInsight:
        """Create insight from a pattern match"""
        # Calculate confidence based on how far from normal the metric is
        confidence = 0.8  # Base confidence
        
        # Adjust confidence based on trend strength
        related_trends = [t for t in trends.values() if t.get('strength') == 'strong']
        if related_trends:
            confidence = min(0.95, confidence + 0.1)
        
        # Estimate improvement potential
        improvement_estimate = await self._estimate_improvement_potential(pattern, metrics)
        
        return PerformanceInsight(
            insight_type=pattern['insight_type'],
            title=pattern['title'],
            description=await self._generate_insight_description(pattern, metrics),
            impact_assessment=pattern['impact'],
            confidence=confidence,
            recommended_actions=pattern['actions'],
            estimated_improvement=improvement_estimate,
            implementation_effort=await self._estimate_implementation_effort(pattern)
        )
    
    async def _generate_trend_insights(self, trends: Dict[str, Any]) -> List[PerformanceInsight]:
        """Generate insights based on performance trends"""
        trend_insights = []
        
        for metric_name, trend_data in trends.items():
            if trend_data.get('strength') == 'strong':
                direction = trend_data.get('direction')
                change_pct = trend_data.get('change_percentage', 0)
                
                if direction == 'declining' and abs(change_pct) > 10:
                    insight = PerformanceInsight(
                        insight_type='negative_trend',
                        title=f'Strong Declining Trend in {metric_name.title()}',
                        description=f'{metric_name.title()} has declined by {abs(change_pct):.1f}% over the analysis period',
                        impact_assessment='Medium - Trend requires attention',
                        confidence=0.85,
                        recommended_actions=[
                            f'Investigate root cause of {metric_name} decline',
                            'Implement corrective measures',
                            'Monitor trend closely'
                        ],
                        estimated_improvement=abs(change_pct),
                        implementation_effort='Medium'
                    )
                    trend_insights.append(insight)
                
                elif direction == 'improving' and abs(change_pct) > 15:
                    insight = PerformanceInsight(
                        insight_type='positive_trend',
                        title=f'Strong Improvement in {metric_name.title()}',
                        description=f'{metric_name.title()} has improved by {abs(change_pct):.1f}% over the analysis period',
                        impact_assessment='Positive - Trend should be maintained',
                        confidence=0.85,
                        recommended_actions=[
                            f'Identify factors contributing to {metric_name} improvement',
                            'Document and replicate successful practices',
                            'Consider applying similar improvements to other metrics'
                        ],
                        estimated_improvement=0,  # Already improving
                        implementation_effort='Low'
                    )
                    trend_insights.append(insight)
        
        return trend_insights
    
    async def _generate_insight_description(self, pattern: Dict[str, Any], metrics: ExcellenceMetrics) -> str:
        """Generate detailed description for insight"""
        pattern_name = pattern['name']
        
        if pattern_name == 'response_time_degradation':
            return f"Response time P95 is {metrics.response_time_p95:.0f}ms, significantly above the recommended 500ms threshold. This may impact user experience and satisfaction."
        
        elif pattern_name == 'high_resource_utilization':
            return f"System resource utilization is at {metrics.resource_utilization:.1f}%, approaching the recommended maximum of 80%. Proactive scaling may be needed."
        
        elif pattern_name == 'reasoning_quality_decline':
            return f"Agent reasoning quality is {metrics.agent_reasoning_quality:.2f}, below the target of 0.85. This affects the core AI capabilities of the system."
        
        elif pattern_name == 'user_satisfaction_decline':
            return f"User satisfaction score is {metrics.user_satisfaction:.2f}, below the expected 0.90 threshold. User experience improvements are needed."
        
        elif pattern_name == 'excellent_performance':
            return f"System is performing excellently with {metrics.uptime_percentage:.2f}% uptime, {metrics.response_time_p95:.0f}ms response time, and {metrics.agent_reasoning_quality:.2f} reasoning quality."
        
        return f"Performance insight detected for {pattern_name}"
    
    async def _estimate_improvement_potential(self, pattern: Dict[str, Any], metrics: ExcellenceMetrics) -> float:
        """Estimate potential improvement percentage"""
        pattern_name = pattern['name']
        
        if pattern_name == 'response_time_degradation':
            # Potential to improve back to 500ms target
            current = metrics.response_time_p95
            target = 500.0
            return max(0, (current - target) / current * 100)
        
        elif pattern_name == 'high_resource_utilization':
            # Potential to optimize to 70% utilization
            current = metrics.resource_utilization
            target = 70.0
            return max(0, (current - target) / current * 100)
        
        elif pattern_name == 'reasoning_quality_decline':
            # Potential to improve to 0.95 quality
            current = metrics.agent_reasoning_quality
            target = 0.95
            return max(0, (target - current) / current * 100)
        
        elif pattern_name == 'user_satisfaction_decline':
            # Potential to improve to 0.95 satisfaction
            current = metrics.user_satisfaction
            target = 0.95
            return max(0, (target - current) / current * 100)
        
        return 10.0  # Default 10% improvement potential
    
    async def _estimate_implementation_effort(self, pattern: Dict[str, Any]) -> str:
        """Estimate implementation effort for insight"""
        effort_map = {
            'response_time_degradation': 'Medium',
            'high_resource_utilization': 'Low',
            'reasoning_quality_decline': 'High',
            'user_satisfaction_decline': 'High',
            'excellent_performance': 'Low'
        }
        
        return effort_map.get(pattern['name'], 'Medium')
    
    def _impact_score(self, impact_assessment: str) -> int:
        """Convert impact assessment to numeric score for sorting"""
        if 'High' in impact_assessment:
            return 3
        elif 'Medium' in impact_assessment:
            return 2
        elif 'Positive' in impact_assessment:
            return 1
        else:
            return 0

class ProductionExcellenceMonitor:
    """Main production excellence monitoring system"""
    
    def __init__(self):
        self.performance_tracker = PerformanceTracker()
        self.sla_monitor = SLAMonitor()
        self.insight_generator = InsightGenerator()
        self._initialized = False
    
    async def initialize(self):
        """Initialize production excellence monitoring"""
        if self._initialized:
            return
        
        try:
            await self.performance_tracker.initialize()
            
            self._initialized = True
            logger.info("Production excellence monitor initialized")
            
        except Exception as e:
            logger.error(f"Production excellence monitor initialization failed: {e}")
    
    async def track_excellence_metrics(self) -> ExcellenceMetrics:
        """Track current performance excellence metrics"""
        if not self._initialized:
            await self.initialize()
        
        return await self.performance_tracker.get_current_performance_snapshot()
    
    async def ensure_sla_compliance(self) -> SLAComplianceReport:
        """Ensure SLA compliance and generate report"""
        if not self._initialized:
            await self.initialize()
        
        current_metrics = await self.performance_tracker.get_current_performance_snapshot()
        return await self.sla_monitor.check_sla_compliance(current_metrics)
    
    async def generate_performance_insights(self) -> List[PerformanceInsight]:
        """Generate actionable performance insights"""
        if not self._initialized:
            await self.initialize()
        
        current_metrics = await self.performance_tracker.get_current_performance_snapshot()
        trends = await self.performance_tracker.analyze_performance_trends()
        
        return await self.insight_generator.generate_performance_insights(current_metrics, trends)
    
    async def comprehensive_excellence_report(self) -> Dict[str, Any]:
        """Generate comprehensive production excellence report"""
        try:
            # Get all excellence data
            metrics = await self.track_excellence_metrics()
            sla_report = await self.ensure_sla_compliance()
            insights = await self.generate_performance_insights()
            trends = await self.performance_tracker.analyze_performance_trends()
            
            # Generate executive summary
            executive_summary = await self._generate_executive_summary(metrics, sla_report, insights)
            
            return {
                'executive_summary': executive_summary,
                'current_metrics': metrics,
                'sla_compliance': sla_report,
                'performance_insights': insights,
                'trends_analysis': trends,
                'report_timestamp': datetime.now().isoformat(),
                'next_review_date': (datetime.now() + timedelta(days=7)).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Comprehensive excellence report failed: {e}")
            return {
                'error': str(e),
                'report_timestamp': datetime.now().isoformat()
            }
    
    async def record_performance_metric(self, metric: PerformanceMetric, value: float):
        """Record a performance metric"""
        if not self._initialized:
            await self.initialize()
        
        await self.performance_tracker.record_performance_metric(metric, value)
    
    async def _generate_executive_summary(self, metrics: ExcellenceMetrics, sla_report: SLAComplianceReport, insights: List[PerformanceInsight]) -> Dict[str, Any]:
        """Generate executive summary of production excellence"""
        # Overall health score
        health_components = [
            metrics.uptime_percentage / 100,
            min(1.0, 500 / metrics.response_time_p95),  # Response time score
            max(0, 1.0 - metrics.error_rate / 5),      # Error rate score (5% is terrible)
            metrics.agent_reasoning_quality,
            metrics.user_satisfaction
        ]
        overall_health = np.mean(health_components) * 100
        
        # Count issues by severity
        critical_issues = len([i for i in insights if 'High' in i.impact_assessment and 'issue' in i.insight_type])
        improvement_opportunities = len([i for i in insights if i.insight_type in ['performance_issue', 'capacity_issue']])
        positive_trends = len([i for i in insights if i.insight_type == 'positive_trend'])
        
        return {
            'overall_health_score': overall_health,
            'health_status': 'Excellent' if overall_health >= 95 else 'Good' if overall_health >= 85 else 'Needs Attention',
            'sla_compliance_percentage': sla_report.overall_compliance,
            'critical_issues': critical_issues,
            'improvement_opportunities': improvement_opportunities,
            'positive_trends': positive_trends,
            'key_metrics': {
                'uptime': f"{metrics.uptime_percentage:.2f}%",
                'response_time': f"{metrics.response_time_p95:.0f}ms",
                'error_rate': f"{metrics.error_rate:.2f}%",
                'user_satisfaction': f"{metrics.user_satisfaction:.2f}"
            },
            'top_priority_actions': [insight.recommended_actions[0] for insight in insights[:3] if insight.recommended_actions]
        }

# Global production excellence monitor
excellence_monitor = ProductionExcellenceMonitor()

async def get_excellence_monitor() -> ProductionExcellenceMonitor:
    """Get initialized excellence monitor"""
    if not excellence_monitor._initialized:
        await excellence_monitor.initialize()
    return excellence_monitor
