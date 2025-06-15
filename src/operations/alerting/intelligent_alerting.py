"""
Intelligent Alerting & Incident Response System

This module implements smart alerting with machine learning-based noise reduction,
automated incident classification, and intelligent response orchestration.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
from collections import defaultdict
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import redis.asyncio as redis

from src.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class IncidentStatus(Enum):
    """Incident status types"""
    OPEN = "open"
    INVESTIGATING = "investigating"
    IDENTIFIED = "identified"
    MONITORING = "monitoring"
    RESOLVED = "resolved"

class ResponseAction(Enum):
    """Automated response actions"""
    RESTART_SERVICE = "restart_service"
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    ENABLE_CIRCUIT_BREAKER = "enable_circuit_breaker"
    CLEAR_CACHE = "clear_cache"
    SWITCH_PROVIDER = "switch_provider"
    THROTTLE_REQUESTS = "throttle_requests"
    NOTIFY_TEAM = "notify_team"
    CREATE_INCIDENT = "create_incident"

@dataclass
class SmartAlert:
    """Intelligent alert with context and correlation"""
    alert_id: str
    severity: AlertSeverity
    title: str
    description: str
    source_system: str
    metrics: Dict[str, Any]
    timestamp: datetime
    correlation_id: Optional[str] = None
    predicted_impact: str = "unknown"
    recommended_actions: List[ResponseAction] = field(default_factory=list)
    confidence: float = 0.0
    noise_score: float = 0.0  # Higher score = more likely to be noise
    auto_resolved: bool = False

@dataclass
class Incident:
    """Incident record with automated classification"""
    incident_id: str
    title: str
    description: str
    severity: AlertSeverity
    status: IncidentStatus
    affected_systems: List[str]
    related_alerts: List[str]
    created_at: datetime
    updated_at: datetime
    resolution_time: Optional[datetime] = None
    root_cause: Optional[str] = None
    response_actions_taken: List[str] = field(default_factory=list)
    escalation_level: int = 0

@dataclass
class ResponseResult:
    """Result of automated incident response"""
    incident_id: str
    actions_attempted: List[str]
    actions_successful: List[str]
    actions_failed: List[str]
    resolution_achieved: bool
    time_to_resolution: Optional[timedelta] = None
    escalation_required: bool = False
    follow_up_actions: List[str] = field(default_factory=list)

class AlertPrioritizer:
    """Prioritizes alerts based on impact and urgency"""
    
    def __init__(self):
        self.impact_weights = self._initialize_impact_weights()
        self.urgency_factors = self._initialize_urgency_factors()
        self.business_context = self._initialize_business_context()
    
    def _initialize_impact_weights(self) -> Dict[str, float]:
        """Initialize impact weights for different systems"""
        return {
            'agent_orchestrator': 1.0,      # Highest impact
            'reasoning_engine': 0.95,
            'conversation_manager': 0.9,
            'learning_system': 0.7,
            'monitoring_system': 0.5,
            'logging_system': 0.3
        }
    
    def _initialize_urgency_factors(self) -> Dict[str, float]:
        """Initialize urgency factors for different scenarios"""
        return {
            'user_facing_error': 1.0,       # Highest urgency
            'performance_degradation': 0.8,
            'capacity_warning': 0.6,
            'security_alert': 0.9,
            'configuration_drift': 0.4,
            'maintenance_reminder': 0.2
        }
    
    def _initialize_business_context(self) -> Dict[str, Any]:
        """Initialize business context for prioritization"""
        return {
            'peak_hours': [9, 10, 11, 14, 15, 16],
            'business_days': [0, 1, 2, 3, 4],  # Monday to Friday
            'high_value_users': ['premium_users', 'enterprise_clients'],
            'critical_features': ['agent_conversation', 'test_generation', 'analysis']
        }
    
    async def prioritize_alert(self, alert: SmartAlert) -> Tuple[float, str]:
        """Calculate alert priority score and explanation"""
        try:
            # Base priority from severity
            severity_scores = {
                AlertSeverity.CRITICAL: 1.0,
                AlertSeverity.ERROR: 0.8,
                AlertSeverity.WARNING: 0.6,
                AlertSeverity.INFO: 0.3
            }
            base_score = severity_scores.get(alert.severity, 0.5)
            
            # Impact factor
            impact_score = self.impact_weights.get(alert.source_system, 0.5)
            
            # Urgency factor based on alert type
            urgency_score = await self._calculate_urgency_score(alert)
            
            # Business context factor
            context_score = await self._calculate_context_score(alert)
            
            # Noise reduction factor (inverse of noise score)
            noise_factor = max(0.1, 1.0 - alert.noise_score)
            
            # Calculate final priority
            priority = (base_score * 0.4 + impact_score * 0.3 + urgency_score * 0.2 + context_score * 0.1) * noise_factor
            
            # Generate explanation
            explanation = await self._generate_priority_explanation(
                alert, base_score, impact_score, urgency_score, context_score, noise_factor
            )
            
            return priority, explanation
            
        except Exception as e:
            logger.error(f"Alert prioritization failed: {e}")
            return 0.5, "Default priority due to calculation error"
    
    async def _calculate_urgency_score(self, alert: SmartAlert) -> float:
        """Calculate urgency score based on alert characteristics"""
        urgency = 0.5  # Default urgency
        
        # Check for user-facing errors
        if 'user' in alert.description.lower() or 'conversation' in alert.source_system:
            urgency = max(urgency, self.urgency_factors['user_facing_error'])
        
        # Check for performance issues
        if 'response_time' in alert.metrics or 'latency' in alert.description.lower():
            urgency = max(urgency, self.urgency_factors['performance_degradation'])
        
        # Check for security issues
        if 'security' in alert.description.lower() or 'unauthorized' in alert.description.lower():
            urgency = max(urgency, self.urgency_factors['security_alert'])
        
        # Check for capacity issues
        if 'capacity' in alert.description.lower() or 'resource' in alert.description.lower():
            urgency = max(urgency, self.urgency_factors['capacity_warning'])
        
        return urgency
    
    async def _calculate_context_score(self, alert: SmartAlert) -> float:
        """Calculate business context score"""
        context_score = 0.5  # Default context
        
        current_time = alert.timestamp
        current_hour = current_time.hour
        current_day = current_time.weekday()
        
        # Higher priority during business hours
        if current_hour in self.business_context['peak_hours']:
            context_score += 0.3
        
        # Higher priority on business days
        if current_day in self.business_context['business_days']:
            context_score += 0.2
        
        # Higher priority for critical features
        for feature in self.business_context['critical_features']:
            if feature in alert.source_system or feature in alert.description.lower():
                context_score += 0.3
                break
        
        return min(1.0, context_score)
    
    async def _generate_priority_explanation(self, alert: SmartAlert, base: float, impact: float, urgency: float, context: float, noise: float) -> str:
        """Generate human-readable priority explanation"""
        explanation_parts = []
        
        if base >= 0.8:
            explanation_parts.append(f"Critical/Error severity ({alert.severity.value})")
        
        if impact >= 0.8:
            explanation_parts.append(f"High-impact system ({alert.source_system})")
        
        if urgency >= 0.8:
            explanation_parts.append("User-facing or security issue")
        
        if context >= 0.7:
            explanation_parts.append("During business hours/critical feature")
        
        if noise < 0.5:
            explanation_parts.append("Potential noise - reduced priority")
        
        if not explanation_parts:
            explanation_parts.append("Standard priority based on alert characteristics")
        
        return "; ".join(explanation_parts)

class NoiseReducer:
    """Reduces alert noise using machine learning and pattern recognition"""
    
    def __init__(self):
        self.noise_patterns = self._initialize_noise_patterns()
        self.alert_history = []
        self.clustering_model = None
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def _initialize_noise_patterns(self) -> List[Dict[str, Any]]:
        """Initialize known noise patterns"""
        return [
            {
                'pattern': 'temporary_spike',
                'description': 'Temporary metric spike that self-resolves',
                'duration_threshold': 300,  # 5 minutes
                'noise_score': 0.8
            },
            {
                'pattern': 'recurring_false_positive',
                'description': 'Alert that repeatedly triggers without real issue',
                'frequency_threshold': 5,   # 5 times in hour
                'noise_score': 0.9
            },
            {
                'pattern': 'maintenance_window',
                'description': 'Alert during known maintenance window',
                'noise_score': 0.95
            },
            {
                'pattern': 'test_environment',
                'description': 'Alert from test or development environment',
                'noise_score': 0.7
            }
        ]
    
    async def calculate_noise_score(self, alert: SmartAlert) -> float:
        """Calculate noise score for an alert (0 = signal, 1 = noise)"""
        try:
            noise_score = 0.0
            
            # Check against known noise patterns
            pattern_score = await self._check_noise_patterns(alert)
            noise_score = max(noise_score, pattern_score)
            
            # Analyze alert frequency
            frequency_score = await self._analyze_alert_frequency(alert)
            noise_score = max(noise_score, frequency_score)
            
            # Check for rapid resolution pattern
            resolution_score = await self._check_rapid_resolution_pattern(alert)
            noise_score = max(noise_score, resolution_score)
            
            # Machine learning-based noise detection
            if self.is_trained:
                ml_score = await self._ml_noise_detection(alert)
                noise_score = max(noise_score, ml_score)
            
            return min(1.0, noise_score)
            
        except Exception as e:
            logger.error(f"Noise score calculation failed: {e}")
            return 0.0  # Default to signal, not noise
    
    async def _check_noise_patterns(self, alert: SmartAlert) -> float:
        """Check alert against known noise patterns"""
        max_noise_score = 0.0
        
        for pattern in self.noise_patterns:
            pattern_type = pattern['pattern']
            
            if pattern_type == 'temporary_spike':
                # Check if this is a temporary spike pattern
                if await self._is_temporary_spike(alert, pattern['duration_threshold']):
                    max_noise_score = max(max_noise_score, pattern['noise_score'])
            
            elif pattern_type == 'recurring_false_positive':
                # Check if this alert type has been recurring without resolution
                if await self._is_recurring_false_positive(alert, pattern['frequency_threshold']):
                    max_noise_score = max(max_noise_score, pattern['noise_score'])
            
            elif pattern_type == 'maintenance_window':
                # Check if alert occurred during maintenance window
                if await self._is_maintenance_window(alert.timestamp):
                    max_noise_score = max(max_noise_score, pattern['noise_score'])
            
            elif pattern_type == 'test_environment':
                # Check if alert is from test environment
                if 'test' in alert.source_system.lower() or 'dev' in alert.source_system.lower():
                    max_noise_score = max(max_noise_score, pattern['noise_score'])
        
        return max_noise_score
    
    async def _analyze_alert_frequency(self, alert: SmartAlert) -> float:
        """Analyze alert frequency to detect noise"""
        # Look for similar alerts in the past hour
        current_time = alert.timestamp
        hour_ago = current_time - timedelta(hours=1)
        
        similar_alerts = 0
        for historical_alert in self.alert_history:
            if (historical_alert['timestamp'] > hour_ago and
                historical_alert['source_system'] == alert.source_system and
                historical_alert['title'].lower() == alert.title.lower()):
                similar_alerts += 1
        
        # If more than 10 similar alerts in an hour, likely noise
        if similar_alerts > 10:
            return 0.8
        elif similar_alerts > 5:
            return 0.6
        elif similar_alerts > 3:
            return 0.4
        else:
            return 0.0
    
    async def _check_rapid_resolution_pattern(self, alert: SmartAlert) -> float:
        """Check if alert matches rapid self-resolution pattern"""
        # Look for patterns where this type of alert typically resolves quickly
        similar_resolved_alerts = []
        
        for historical_alert in self.alert_history:
            if (historical_alert.get('source_system') == alert.source_system and
                historical_alert.get('resolved', False) and
                historical_alert.get('resolution_time')):
                similar_resolved_alerts.append(historical_alert['resolution_time'])
        
        if len(similar_resolved_alerts) >= 5:
            avg_resolution_time = np.mean(similar_resolved_alerts)
            if avg_resolution_time < 300:  # Less than 5 minutes average
                return 0.6  # Likely quick self-resolving issue
        
        return 0.0
    
    async def _is_temporary_spike(self, alert: SmartAlert, duration_threshold: int) -> bool:
        """Check if alert represents a temporary spike"""
        # Look for metrics that indicate a spike
        for metric_name, value in alert.metrics.items():
            if isinstance(value, (int, float)):
                # Check if this metric typically returns to normal quickly
                # In production, this would analyze historical metric data
                if 'spike' in alert.description.lower() or 'temporary' in alert.description.lower():
                    return True
        
        return False
    
    async def _is_recurring_false_positive(self, alert: SmartAlert, frequency_threshold: int) -> bool:
        """Check if alert is a recurring false positive"""
        # Count similar unresolved alerts
        similar_unresolved = 0
        
        for historical_alert in self.alert_history[-20:]:  # Last 20 alerts
            if (historical_alert.get('source_system') == alert.source_system and
                historical_alert.get('title', '').lower() == alert.title.lower() and
                not historical_alert.get('resolved', False)):
                similar_unresolved += 1
        
        return similar_unresolved >= frequency_threshold
    
    async def _is_maintenance_window(self, timestamp: datetime) -> bool:
        """Check if timestamp falls within maintenance window"""
        # Maintenance windows typically during off-hours
        hour = timestamp.hour
        day = timestamp.weekday()
        
        # Assume maintenance windows: 2-4 AM on weekends
        if day in [5, 6] and 2 <= hour <= 4:  # Saturday/Sunday 2-4 AM
            return True
        
        return False
    
    async def _ml_noise_detection(self, alert: SmartAlert) -> float:
        """Use machine learning for noise detection"""
        try:
            if not self.is_trained:
                return 0.0
            
            # Extract features from alert
            features = await self._extract_alert_features(alert)
            
            # Use clustering to detect outliers (potential noise)
            features_scaled = self.scaler.transform([features])
            
            # Use trained clustering model to predict if alert is noise
            cluster_label = self.clustering_model.fit_predict(features_scaled)
            
            # If alert is in noise cluster, return high noise score
            if cluster_label[0] == -1:  # Outlier in DBSCAN
                return 0.7
            else:
                return 0.1
                
        except Exception as e:
            logger.error(f"ML noise detection failed: {e}")
            return 0.0
    
    async def _extract_alert_features(self, alert: SmartAlert) -> List[float]:
        """Extract numerical features from alert for ML"""
        features = []
        
        # Severity as number
        severity_map = {AlertSeverity.INFO: 1, AlertSeverity.WARNING: 2, AlertSeverity.ERROR: 3, AlertSeverity.CRITICAL: 4}
        features.append(severity_map.get(alert.severity, 2))
        
        # Hour of day
        features.append(alert.timestamp.hour)
        
        # Day of week
        features.append(alert.timestamp.weekday())
        
        # Source system hash (simplified)
        features.append(hash(alert.source_system) % 1000)
        
        # Number of metrics
        features.append(len(alert.metrics))
        
        # Description length
        features.append(len(alert.description))
        
        # Average metric value (if numeric)
        numeric_values = [v for v in alert.metrics.values() if isinstance(v, (int, float))]
        features.append(np.mean(numeric_values) if numeric_values else 0)
        
        return features
    
    async def train_noise_detection(self, historical_alerts: List[Dict[str, Any]]):
        """Train noise detection model on historical data"""
        try:
            if len(historical_alerts) < 50:
                logger.warning("Insufficient data for noise detection training")
                return
            
            # Extract features from historical alerts
            features = []
            for alert_data in historical_alerts:
                alert_features = await self._extract_historical_alert_features(alert_data)
                features.append(alert_features)
            
            # Standardize features
            features_scaled = self.scaler.fit_transform(features)
            
            # Train DBSCAN clustering model to identify noise patterns
            self.clustering_model = DBSCAN(eps=0.5, min_samples=5)
            self.clustering_model.fit(features_scaled)
            
            self.is_trained = True
            logger.info("Noise detection model trained successfully")
            
        except Exception as e:
            logger.error(f"Noise detection training failed: {e}")
    
    async def _extract_historical_alert_features(self, alert_data: Dict[str, Any]) -> List[float]:
        """Extract features from historical alert data"""
        features = []
        
        # Convert historical data to features similar to current alerts
        features.append(alert_data.get('severity_num', 2))
        features.append(alert_data.get('hour', 12))
        features.append(alert_data.get('day_of_week', 3))
        features.append(hash(alert_data.get('source_system', '')) % 1000)
        features.append(alert_data.get('metric_count', 1))
        features.append(len(alert_data.get('description', '')))
        features.append(alert_data.get('avg_metric_value', 0))
        
        return features

class IncidentPredictor:
    """Predicts potential incidents based on alert patterns"""
    
    def __init__(self):
        self.incident_patterns = self._initialize_incident_patterns()
        self.alert_correlations = {}
    
    def _initialize_incident_patterns(self) -> List[Dict[str, Any]]:
        """Initialize known incident prediction patterns"""
        return [
            {
                'name': 'cascading_failure',
                'description': 'Multiple system alerts indicating cascading failure',
                'pattern': {
                    'alert_count_threshold': 5,
                    'time_window_minutes': 15,
                    'system_diversity': 3
                },
                'incident_probability': 0.85
            },
            {
                'name': 'resource_exhaustion',
                'description': 'Resource utilization alerts leading to system failure',
                'pattern': {
                    'cpu_threshold': 0.9,
                    'memory_threshold': 0.9,
                    'duration_minutes': 10
                },
                'incident_probability': 0.75
            },
            {
                'name': 'external_dependency_failure',
                'description': 'External service failures affecting system',
                'pattern': {
                    'error_rate_threshold': 0.5,
                    'response_time_increase': 5.0,
                    'affected_services': 2
                },
                'incident_probability': 0.7
            }
        ]
    
    async def predict_incident_likelihood(self, recent_alerts: List[SmartAlert]) -> Tuple[float, str, List[str]]:
        """Predict likelihood of incident based on recent alerts"""
        try:
            max_probability = 0.0
            predicted_type = "none"
            contributing_factors = []
            
            for pattern in self.incident_patterns:
                probability = await self._evaluate_incident_pattern(recent_alerts, pattern)
                if probability > max_probability:
                    max_probability = probability
                    predicted_type = pattern['name']
                    contributing_factors = await self._identify_contributing_factors(recent_alerts, pattern)
            
            return max_probability, predicted_type, contributing_factors
            
        except Exception as e:
            logger.error(f"Incident prediction failed: {e}")
            return 0.0, "prediction_error", []
    
    async def _evaluate_incident_pattern(self, alerts: List[SmartAlert], pattern: Dict[str, Any]) -> float:
        """Evaluate how well alerts match an incident pattern"""
        pattern_name = pattern['name']
        pattern_config = pattern['pattern']
        base_probability = pattern['incident_probability']
        
        if pattern_name == 'cascading_failure':
            return await self._evaluate_cascading_failure(alerts, pattern_config, base_probability)
        elif pattern_name == 'resource_exhaustion':
            return await self._evaluate_resource_exhaustion(alerts, pattern_config, base_probability)
        elif pattern_name == 'external_dependency_failure':
            return await self._evaluate_external_dependency_failure(alerts, pattern_config, base_probability)
        
        return 0.0
    
    async def _evaluate_cascading_failure(self, alerts: List[SmartAlert], pattern: Dict[str, Any], base_prob: float) -> float:
        """Evaluate cascading failure pattern"""
        # Check alert count
        if len(alerts) < pattern['alert_count_threshold']:
            return 0.0
        
        # Check time window
        if alerts:
            time_span = (max(alert.timestamp for alert in alerts) - 
                        min(alert.timestamp for alert in alerts)).total_seconds() / 60
            if time_span > pattern['time_window_minutes']:
                return 0.0
        
        # Check system diversity
        unique_systems = set(alert.source_system for alert in alerts)
        if len(unique_systems) < pattern['system_diversity']:
            return 0.0
        
        # Calculate confidence based on how well pattern matches
        alert_count_factor = min(1.0, len(alerts) / (pattern['alert_count_threshold'] * 2))
        system_diversity_factor = min(1.0, len(unique_systems) / pattern['system_diversity'])
        
        return base_prob * alert_count_factor * system_diversity_factor
    
    async def _evaluate_resource_exhaustion(self, alerts: List[SmartAlert], pattern: Dict[str, Any], base_prob: float) -> float:
        """Evaluate resource exhaustion pattern"""
        resource_alerts = []
        
        for alert in alerts:
            # Check for resource-related metrics
            for metric, value in alert.metrics.items():
                if isinstance(value, (int, float)):
                    if ('cpu' in metric.lower() and value > pattern['cpu_threshold']) or \
                       ('memory' in metric.lower() and value > pattern['memory_threshold']):
                        resource_alerts.append(alert)
                        break
        
        if len(resource_alerts) < 2:  # Need at least 2 resource alerts
            return 0.0
        
        # Check duration
        if resource_alerts:
            duration = (max(alert.timestamp for alert in resource_alerts) - 
                       min(alert.timestamp for alert in resource_alerts)).total_seconds() / 60
            if duration >= pattern['duration_minutes']:
                return base_prob * min(1.0, len(resource_alerts) / 3)
        
        return 0.0
    
    async def _evaluate_external_dependency_failure(self, alerts: List[SmartAlert], pattern: Dict[str, Any], base_prob: float) -> float:
        """Evaluate external dependency failure pattern"""
        # Look for error rate and response time issues
        error_alerts = 0
        response_time_alerts = 0
        
        for alert in alerts:
            if 'error' in alert.description.lower() or alert.severity in [AlertSeverity.ERROR, AlertSeverity.CRITICAL]:
                error_alerts += 1
            
            if 'response_time' in alert.description.lower() or 'latency' in alert.description.lower():
                response_time_alerts += 1
        
        if error_alerts >= 2 and response_time_alerts >= 1:
            return base_prob * min(1.0, (error_alerts + response_time_alerts) / 5)
        
        return 0.0
    
    async def _identify_contributing_factors(self, alerts: List[SmartAlert], pattern: Dict[str, Any]) -> List[str]:
        """Identify contributing factors for incident prediction"""
        factors = []
        
        # High-severity alerts
        critical_alerts = [a for a in alerts if a.severity == AlertSeverity.CRITICAL]
        if critical_alerts:
            factors.append(f"{len(critical_alerts)} critical alerts detected")
        
        # Multiple affected systems
        affected_systems = set(alert.source_system for alert in alerts)
        if len(affected_systems) > 2:
            factors.append(f"Multiple systems affected: {', '.join(list(affected_systems)[:3])}")
        
        # Recent alert frequency
        recent_count = len([a for a in alerts if (datetime.now() - a.timestamp).total_seconds() < 600])
        if recent_count > 3:
            factors.append(f"{recent_count} alerts in last 10 minutes")
        
        # Pattern-specific factors
        pattern_name = pattern['name']
        if pattern_name == 'resource_exhaustion':
            factors.append("Resource utilization thresholds exceeded")
        elif pattern_name == 'cascading_failure':
            factors.append("Rapid spread across multiple systems")
        elif pattern_name == 'external_dependency_failure':
            factors.append("External service degradation detected")
        
        return factors

class ResponseOptimizer:
    """Optimizes automated incident response strategies"""
    
    def __init__(self):
        self.response_history = []
        self.success_rates = defaultdict(float)
        self.response_strategies = self._initialize_response_strategies()
    
    def _initialize_response_strategies(self) -> Dict[str, List[ResponseAction]]:
        """Initialize response strategies for different incident types"""
        return {
            'high_cpu_usage': [
                ResponseAction.SCALE_UP,
                ResponseAction.THROTTLE_REQUESTS,
                ResponseAction.RESTART_SERVICE
            ],
            'memory_leak': [
                ResponseAction.RESTART_SERVICE,
                ResponseAction.CLEAR_CACHE,
                ResponseAction.SCALE_UP
            ],
            'database_connection_failure': [
                ResponseAction.RESTART_SERVICE,
                ResponseAction.SWITCH_PROVIDER,
                ResponseAction.ENABLE_CIRCUIT_BREAKER
            ],
            'api_error_rate_spike': [
                ResponseAction.ENABLE_CIRCUIT_BREAKER,
                ResponseAction.THROTTLE_REQUESTS,
                ResponseAction.SWITCH_PROVIDER
            ],
            'external_service_failure': [
                ResponseAction.SWITCH_PROVIDER,
                ResponseAction.ENABLE_CIRCUIT_BREAKER,
                ResponseAction.NOTIFY_TEAM
            ]
        }
    
    async def optimize_incident_response(self, incident: Incident, alerts: List[SmartAlert]) -> List[ResponseAction]:
        """Optimize response actions for an incident"""
        try:
            # Classify incident type
            incident_type = await self._classify_incident(incident, alerts)
            
            # Get base response strategy
            base_actions = self.response_strategies.get(incident_type, [ResponseAction.NOTIFY_TEAM])
            
            # Optimize based on historical success rates
            optimized_actions = await self._optimize_based_on_history(base_actions, incident_type)
            
            # Consider current system state
            contextualized_actions = await self._contextualize_actions(optimized_actions, incident, alerts)
            
            return contextualized_actions
            
        except Exception as e:
            logger.error(f"Response optimization failed: {e}")
            return [ResponseAction.NOTIFY_TEAM, ResponseAction.CREATE_INCIDENT]
    
    async def _classify_incident(self, incident: Incident, alerts: List[SmartAlert]) -> str:
        """Classify incident type based on alerts and symptoms"""
        # Analyze alert descriptions and metrics for classification
        descriptions = [alert.description.lower() for alert in alerts]
        all_text = " ".join(descriptions + [incident.description.lower()])
        
        if 'cpu' in all_text and ('high' in all_text or 'usage' in all_text):
            return 'high_cpu_usage'
        elif 'memory' in all_text and ('leak' in all_text or 'usage' in all_text):
            return 'memory_leak'
        elif 'database' in all_text and ('connection' in all_text or 'timeout' in all_text):
            return 'database_connection_failure'
        elif 'api' in all_text and ('error' in all_text or 'rate' in all_text):
            return 'api_error_rate_spike'
        elif 'external' in all_text or 'dependency' in all_text:
            return 'external_service_failure'
        else:
            return 'unknown_incident'
    
    async def _optimize_based_on_history(self, base_actions: List[ResponseAction], incident_type: str) -> List[ResponseAction]:
        """Optimize actions based on historical success rates"""
        # Sort actions by historical success rate for this incident type
        action_scores = {}
        
        for action in base_actions:
            success_rate = self.success_rates.get(f"{incident_type}_{action.value}", 0.5)
            action_scores[action] = success_rate
        
        # Sort by success rate (highest first)
        optimized_actions = sorted(base_actions, key=lambda x: action_scores[x], reverse=True)
        
        return optimized_actions
    
    async def _contextualize_actions(self, actions: List[ResponseAction], incident: Incident, alerts: List[SmartAlert]) -> List[ResponseAction]:
        """Contextualize actions based on current system state"""
        contextualized = []
        
        for action in actions:
            # Check if action is appropriate for current context
            if await self._is_action_appropriate(action, incident, alerts):
                contextualized.append(action)
        
        # Ensure we always have at least notification action
        if not contextualized:
            contextualized.append(ResponseAction.NOTIFY_TEAM)
        
        return contextualized
    
    async def _is_action_appropriate(self, action: ResponseAction, incident: Incident, alerts: List[SmartAlert]) -> bool:
        """Check if action is appropriate for current context"""
        # Business hours check for disruptive actions
        current_hour = datetime.now().hour
        is_business_hours = 9 <= current_hour <= 17
        
        disruptive_actions = [ResponseAction.RESTART_SERVICE, ResponseAction.SCALE_DOWN]
        
        if action in disruptive_actions and is_business_hours and incident.severity != AlertSeverity.CRITICAL:
            return False
        
        # Check system capacity for scaling actions
        if action == ResponseAction.SCALE_UP:
            # In production, this would check actual system capacity
            return True
        
        # Check if switch provider action is available
        if action == ResponseAction.SWITCH_PROVIDER:
            # In production, this would check if alternative providers are configured
            return True
        
        return True

class IntelligentAlertingSystem:
    """Main intelligent alerting and incident response system"""
    
    def __init__(self):
        self.alert_prioritizer = AlertPrioritizer()
        self.noise_reducer = NoiseReducer()
        self.incident_predictor = IncidentPredictor()
        self.response_optimizer = ResponseOptimizer()
        self.active_incidents = {}
        self.alert_history = []
        self.redis_client = None
    
    async def initialize(self):
        """Initialize intelligent alerting system"""
        try:
            self.redis_client = redis.Redis(
                host=settings.redis_host,
                port=settings.redis_port,
                decode_responses=True
            )
            await self.redis_client.ping()
            
            # Load historical data for training
            await self._load_historical_data()
            
            # Train noise reduction model if enough data
            if len(self.alert_history) >= 50:
                await self.noise_reducer.train_noise_detection(self.alert_history)
            
            logger.info("Intelligent alerting system initialized")
            
        except Exception as e:
            logger.error(f"Intelligent alerting initialization failed: {e}")
    
    async def generate_intelligent_alert(self, 
                                       title: str,
                                       description: str,
                                       source_system: str,
                                       severity: AlertSeverity,
                                       metrics: Dict[str, Any]) -> SmartAlert:
        """Generate intelligent alert with context and correlation"""
        try:
            # Create base alert
            alert = SmartAlert(
                alert_id=f"alert_{int(datetime.now().timestamp())}_{hash(title) % 10000}",
                severity=severity,
                title=title,
                description=description,
                source_system=source_system,
                metrics=metrics,
                timestamp=datetime.now()
            )
            
            # Calculate noise score
            alert.noise_score = await self.noise_reducer.calculate_noise_score(alert)
            
            # Prioritize alert
            priority_score, priority_explanation = await self.alert_prioritizer.prioritize_alert(alert)
            alert.confidence = priority_score
            
            # Correlate with existing alerts
            alert.correlation_id = await self._correlate_alert(alert)
            
            # Predict impact
            alert.predicted_impact = await self._predict_alert_impact(alert)
            
            # Generate recommended actions
            alert.recommended_actions = await self._generate_recommended_actions(alert)
            
            # Store alert
            await self._store_alert(alert)
            
            # Check for incident prediction
            await self._check_incident_prediction(alert)
            
            return alert
            
        except Exception as e:
            logger.error(f"Alert generation failed: {e}")
            # Return basic alert without intelligence
            return SmartAlert(
                alert_id=f"basic_alert_{int(datetime.now().timestamp())}",
                severity=severity,
                title=title,
                description=description,
                source_system=source_system,
                metrics=metrics,
                timestamp=datetime.now()
            )
    
    async def automated_incident_response(self, incident: Incident) -> ResponseResult:
        """Perform automated incident response"""
        try:
            start_time = datetime.now()
            
            # Get related alerts
            related_alerts = await self._get_related_alerts(incident.related_alerts)
            
            # Optimize response strategy
            response_actions = await self.response_optimizer.optimize_incident_response(incident, related_alerts)
            
            # Execute responses
            execution_results = await self._execute_response_actions(response_actions, incident)
            
            # Check if incident is resolved
            resolution_achieved = await self._check_incident_resolution(incident)
            
            # Calculate response time
            response_time = datetime.now() - start_time
            
            # Determine if escalation is needed
            escalation_required = await self._determine_escalation_need(incident, execution_results, resolution_achieved)
            
            # Generate follow-up actions
            follow_up_actions = await self._generate_follow_up_actions(incident, execution_results, resolution_achieved)
            
            # Create response result
            response_result = ResponseResult(
                incident_id=incident.incident_id,
                actions_attempted=[action.value for action in response_actions],
                actions_successful=execution_results['successful'],
                actions_failed=execution_results['failed'],
                resolution_achieved=resolution_achieved,
                time_to_resolution=response_time if resolution_achieved else None,
                escalation_required=escalation_required,
                follow_up_actions=follow_up_actions
            )
            
            # Update incident status
            await self._update_incident_status(incident, response_result)
            
            # Store response for learning
            await self._store_response_result(response_result)
            
            return response_result
            
        except Exception as e:
            logger.error(f"Automated incident response failed: {e}")
            return ResponseResult(
                incident_id=incident.incident_id,
                actions_attempted=[],
                actions_successful=[],
                actions_failed=[f"Response system error: {e}"],
                resolution_achieved=False,
                escalation_required=True,
                follow_up_actions=["Manual investigation required"]
            )
    
    async def _correlate_alert(self, alert: SmartAlert) -> Optional[str]:
        """Correlate alert with existing incidents or alert groups"""
        # Look for similar recent alerts
        recent_threshold = datetime.now() - timedelta(minutes=30)
        
        for historical_alert in self.alert_history:
            if (historical_alert.get('timestamp', datetime.min) > recent_threshold and
                historical_alert.get('source_system') == alert.source_system):
                
                # Check for similar description or metrics
                if await self._alerts_are_related(alert, historical_alert):
                    return historical_alert.get('correlation_id', f"corr_{historical_alert.get('alert_id', 'unknown')}")
        
        # Create new correlation ID
        return f"corr_{alert.alert_id}"
    
    async def _alerts_are_related(self, alert1: SmartAlert, alert2_data: Dict[str, Any]) -> bool:
        """Check if two alerts are related"""
        # Simple similarity check based on description keywords
        desc1_words = set(alert1.description.lower().split())
        desc2_words = set(alert2_data.get('description', '').lower().split())
        
        # If they share significant keywords, they're related
        common_words = desc1_words.intersection(desc2_words)
        similarity = len(common_words) / max(len(desc1_words), len(desc2_words), 1)
        
        return similarity > 0.5
    
    async def _predict_alert_impact(self, alert: SmartAlert) -> str:
        """Predict the potential impact of an alert"""
        impact_factors = []
        
        # Severity-based impact
        if alert.severity == AlertSeverity.CRITICAL:
            impact_factors.append("High user impact expected")
        elif alert.severity == AlertSeverity.ERROR:
            impact_factors.append("Moderate user impact possible")
        
        # System-based impact
        high_impact_systems = ['agent_orchestrator', 'conversation_manager', 'reasoning_engine']
        if alert.source_system in high_impact_systems:
            impact_factors.append("Core system affected")
        
        # Time-based impact
        current_hour = alert.timestamp.hour
        if 9 <= current_hour <= 17:
            impact_factors.append("During business hours")
        
        # Metric-based impact
        for metric_name, value in alert.metrics.items():
            if isinstance(value, (int, float)):
                if 'error_rate' in metric_name.lower() and value > 0.1:
                    impact_factors.append("High error rate detected")
                elif 'response_time' in metric_name.lower() and value > 2000:
                    impact_factors.append("Performance degradation")
        
        if not impact_factors:
            return "Limited impact expected"
        else:
            return "; ".join(impact_factors)
    
    async def _generate_recommended_actions(self, alert: SmartAlert) -> List[ResponseAction]:
        """Generate recommended actions for an alert"""
        actions = []
        
        # Severity-based actions
        if alert.severity == AlertSeverity.CRITICAL:
            actions.append(ResponseAction.CREATE_INCIDENT)
            actions.append(ResponseAction.NOTIFY_TEAM)
        
        # System-specific actions
        if 'memory' in alert.description.lower():
            actions.append(ResponseAction.RESTART_SERVICE)
            actions.append(ResponseAction.CLEAR_CACHE)
        
        if 'cpu' in alert.description.lower():
            actions.append(ResponseAction.SCALE_UP)
            actions.append(ResponseAction.THROTTLE_REQUESTS)
        
        if 'error' in alert.description.lower():
            actions.append(ResponseAction.ENABLE_CIRCUIT_BREAKER)
        
        # Default action for high-noise alerts
        if alert.noise_score > 0.7:
            return []  # No automatic actions for likely noise
        
        return actions[:3]  # Limit to top 3 actions
    
    async def _check_incident_prediction(self, alert: SmartAlert):
        """Check if new alert triggers incident prediction"""
        # Get recent alerts for prediction
        recent_alerts = await self._get_recent_alerts(minutes=30)
        recent_alerts.append(alert)
        
        # Predict incident likelihood
        incident_probability, incident_type, factors = await self.incident_predictor.predict_incident_likelihood(recent_alerts)
        
        # If high probability, create incident
        if incident_probability > 0.7:
            await self._create_predicted_incident(alert, incident_type, incident_probability, factors)
    
    async def _create_predicted_incident(self, trigger_alert: SmartAlert, incident_type: str, probability: float, factors: List[str]):
        """Create incident based on prediction"""
        incident = Incident(
            incident_id=f"incident_{int(datetime.now().timestamp())}",
            title=f"Predicted {incident_type.replace('_', ' ').title()}",
            description=f"Incident predicted with {probability*100:.1f}% confidence. Contributing factors: {'; '.join(factors)}",
            severity=trigger_alert.severity,
            status=IncidentStatus.OPEN,
            affected_systems=[trigger_alert.source_system],
            related_alerts=[trigger_alert.alert_id],
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        self.active_incidents[incident.incident_id] = incident
        
        # Trigger automated response
        await self.automated_incident_response(incident)
    
    async def _execute_response_actions(self, actions: List[ResponseAction], incident: Incident) -> Dict[str, List[str]]:
        """Execute automated response actions"""
        successful = []
        failed = []
        
        for action in actions:
            try:
                success = await self._execute_single_action(action, incident)
                if success:
                    successful.append(action.value)
                    logger.info(f"Successfully executed {action.value} for incident {incident.incident_id}")
                else:
                    failed.append(action.value)
                    logger.warning(f"Failed to execute {action.value} for incident {incident.incident_id}")
            except Exception as e:
                failed.append(f"{action.value}: {e}")
                logger.error(f"Exception executing {action.value}: {e}")
        
        return {'successful': successful, 'failed': failed}
    
    async def _execute_single_action(self, action: ResponseAction, incident: Incident) -> bool:
        """Execute a single response action"""
        try:
            if action == ResponseAction.RESTART_SERVICE:
                # Simulate service restart
                logger.info(f"Restarting services for incident {incident.incident_id}")
                await asyncio.sleep(1)  # Simulate restart time
                return True
            
            elif action == ResponseAction.SCALE_UP:
                # Simulate scaling up
                logger.info(f"Scaling up services for incident {incident.incident_id}")
                await asyncio.sleep(1)  # Simulate scaling time
                return True
            
            elif action == ResponseAction.CLEAR_CACHE:
                # Simulate cache clearing
                logger.info(f"Clearing cache for incident {incident.incident_id}")
                await asyncio.sleep(0.5)  # Simulate cache clear time
                return True
            
            elif action == ResponseAction.ENABLE_CIRCUIT_BREAKER:
                # Simulate circuit breaker activation
                logger.info(f"Enabling circuit breaker for incident {incident.incident_id}")
                return True
            
            elif action == ResponseAction.THROTTLE_REQUESTS:
                # Simulate request throttling
                logger.info(f"Throttling requests for incident {incident.incident_id}")
                return True
            
            elif action == ResponseAction.SWITCH_PROVIDER:
                # Simulate provider switching
                logger.info(f"Switching provider for incident {incident.incident_id}")
                return True
            
            elif action == ResponseAction.NOTIFY_TEAM:
                # Simulate team notification
                logger.info(f"Notifying team about incident {incident.incident_id}")
                return True
            
            elif action == ResponseAction.CREATE_INCIDENT:
                # Incident already exists, mark as successful
                return True
            
            else:
                logger.warning(f"Unknown action: {action}")
                return False
                
        except Exception as e:
            logger.error(f"Action execution failed: {e}")
            return False
    
    async def _check_incident_resolution(self, incident: Incident) -> bool:
        """Check if incident has been resolved"""
        # Simulate resolution check based on recent metrics
        # In production, this would check actual system health
        
        # Simple heuristic: incident is resolved if no new related alerts in last 5 minutes
        recent_threshold = datetime.now() - timedelta(minutes=5)
        recent_related_alerts = [
            alert for alert in self.alert_history[-10:]  # Check last 10 alerts
            if (alert.get('timestamp', datetime.min) > recent_threshold and
                alert.get('source_system') in incident.affected_systems)
        ]
        
        return len(recent_related_alerts) == 0
    
    async def _determine_escalation_need(self, incident: Incident, execution_results: Dict[str, List[str]], resolved: bool) -> bool:
        """Determine if incident needs escalation"""
        # Escalate if critical and not resolved
        if incident.severity == AlertSeverity.CRITICAL and not resolved:
            return True
        
        # Escalate if all actions failed
        if execution_results['successful'] == [] and execution_results['failed'] != []:
            return True
        
        # Escalate if incident is older than 30 minutes and not resolved
        age = datetime.now() - incident.created_at
        if age > timedelta(minutes=30) and not resolved:
            return True
        
        return False
    
    async def _generate_follow_up_actions(self, incident: Incident, execution_results: Dict[str, List[str]], resolved: bool) -> List[str]:
        """Generate follow-up actions based on incident response"""
        follow_ups = []
        
        if resolved:
            follow_ups.extend([
                "Monitor system stability for 30 minutes",
                "Document resolution in incident report",
                "Review incident for prevention opportunities"
            ])
        else:
            follow_ups.extend([
                "Continue monitoring incident progression",
                "Prepare for manual intervention if needed",
                "Gather additional diagnostic information"
            ])
        
        # Add specific follow-ups based on failed actions
        if 'restart_service' in execution_results['failed']:
            follow_ups.append("Investigate why service restart failed")
        
        if 'scale_up' in execution_results['failed']:
            follow_ups.append("Check resource availability and scaling policies")
        
        return follow_ups
    
    async def _update_incident_status(self, incident: Incident, response_result: ResponseResult):
        """Update incident status based on response result"""
        if response_result.resolution_achieved:
            incident.status = IncidentStatus.RESOLVED
            incident.resolution_time = datetime.now()
        elif response_result.escalation_required:
            incident.status = IncidentStatus.INVESTIGATING
            incident.escalation_level += 1
        else:
            incident.status = IncidentStatus.MONITORING
        
        incident.updated_at = datetime.now()
        incident.response_actions_taken.extend(response_result.actions_successful)
    
    async def _store_alert(self, alert: SmartAlert):
        """Store alert for historical analysis"""
        alert_data = {
            'alert_id': alert.alert_id,
            'severity': alert.severity.value,
            'title': alert.title,
            'description': alert.description,
            'source_system': alert.source_system,
            'timestamp': alert.timestamp,
            'noise_score': alert.noise_score,
            'confidence': alert.confidence
        }
        
        self.alert_history.append(alert_data)
        
        # Keep only last 1000 alerts in memory
        if len(self.alert_history) > 1000:
            self.alert_history = self.alert_history[-1000:]
        
        # Store in Redis if available
        if self.redis_client:
            try:
                await self.redis_client.lpush("alert_history", json.dumps(alert_data, default=str))
                await self.redis_client.ltrim("alert_history", 0, 999)  # Keep last 1000
            except Exception as e:
                logger.error(f"Failed to store alert in Redis: {e}")
    
    async def _store_response_result(self, response_result: ResponseResult):
        """Store response result for learning"""
        try:
            result_data = {
                'incident_id': response_result.incident_id,
                'actions_attempted': response_result.actions_attempted,
                'success_rate': len(response_result.actions_successful) / len(response_result.actions_attempted) if response_result.actions_attempted else 0,
                'resolution_achieved': response_result.resolution_achieved,
                'timestamp': datetime.now().isoformat()
            }
            
            if self.redis_client:
                await self.redis_client.lpush("response_history", json.dumps(result_data))
                await self.redis_client.ltrim("response_history", 0, 499)  # Keep last 500
                
        except Exception as e:
            logger.error(f"Failed to store response result: {e}")
    
    async def _get_recent_alerts(self, minutes: int = 30) -> List[SmartAlert]:
        """Get recent alerts for analysis"""
        threshold = datetime.now() - timedelta(minutes=minutes)
        recent = []
        
        for alert_data in self.alert_history:
            alert_time = alert_data.get('timestamp')
            if isinstance(alert_time, str):
                alert_time = datetime.fromisoformat(alert_time)
            
            if alert_time and alert_time > threshold:
                # Convert back to SmartAlert object
                alert = SmartAlert(
                    alert_id=alert_data['alert_id'],
                    severity=AlertSeverity(alert_data['severity']),
                    title=alert_data['title'],
                    description=alert_data['description'],
                    source_system=alert_data['source_system'],
                    metrics={},
                    timestamp=alert_time,
                    noise_score=alert_data.get('noise_score', 0),
                    confidence=alert_data.get('confidence', 0)
                )
                recent.append(alert)
        
        return recent
    
    async def _get_related_alerts(self, alert_ids: List[str]) -> List[SmartAlert]:
        """Get specific alerts by ID"""
        related = []
        
        for alert_data in self.alert_history:
            if alert_data.get('alert_id') in alert_ids:
                alert = SmartAlert(
                    alert_id=alert_data['alert_id'],
                    severity=AlertSeverity(alert_data['severity']),
                    title=alert_data['title'],
                    description=alert_data['description'],
                    source_system=alert_data['source_system'],
                    metrics={},
                    timestamp=alert_data['timestamp'],
                    noise_score=alert_data.get('noise_score', 0),
                    confidence=alert_data.get('confidence', 0)
                )
                related.append(alert)
        
        return related
    
    async def _load_historical_data(self):
        """Load historical alert data for training"""
        try:
            if self.redis_client:
                alert_data = await self.redis_client.lrange("alert_history", 0, 999)
                for data in alert_data:
                    try:
                        alert_dict = json.loads(data)
                        self.alert_history.append(alert_dict)
                    except json.JSONDecodeError:
                        continue
                        
        except Exception as e:
            logger.error(f"Failed to load historical data: {e}")

# Global intelligent alerting system
intelligent_alerting = IntelligentAlertingSystem()

async def get_intelligent_alerting() -> IntelligentAlertingSystem:
    """Get initialized intelligent alerting system"""
    if not hasattr(intelligent_alerting, '_initialized'):
        await intelligent_alerting.initialize()
        intelligent_alerting._initialized = True
    return intelligent_alerting
