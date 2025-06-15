"""
Basic Agent Intelligence Analytics
Simple analytics and trend analysis for agent intelligence
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

# Try to import ML libraries, fallback to basic math if not available
try:
    import numpy as np
    from scipy import stats
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("⚠️  ML libraries not available, using basic analytics")

logger = logging.getLogger(__name__)

class TrendDirection(str, Enum):
    """Trend direction indicators"""
    IMPROVING = "improving"
    DECLINING = "declining"
    STABLE = "stable"
    VOLATILE = "volatile"

@dataclass
class IntelligenceTrend:
    """Intelligence trend analysis result"""
    metric_name: str
    current_value: float
    trend_direction: TrendDirection
    change_rate: float
    confidence: float
    recommendation: str

@dataclass
class PerformancePrediction:
    """Performance prediction result"""
    agent_name: str
    prediction_window: str
    predicted_metrics: Dict[str, float]
    recommendations: List[str]

class IntelligenceAnalytics:
    """
    Basic analytics for agent intelligence monitoring
    """
    
    def __init__(self):
        self._historical_data: Dict[str, List[float]] = {}
        self._baselines: Dict[str, float] = {}
        
    async def analyze_intelligence_trends(self, 
                                        agent_data: Dict[str, List[float]], 
                                        time_window: str = "24h") -> List[IntelligenceTrend]:
        """
        Analyze intelligence trends over time for multiple metrics
        """
        try:
            trends = []
            
            for metric_name, values in agent_data.items():
                if len(values) < 3:  # Need minimum data points
                    continue
                
                trend = await self._analyze_single_metric_trend(metric_name, values)
                if trend:
                    trends.append(trend)
            
            logger.info(f"Analyzed {len(trends)} intelligence trends")
            return trends
            
        except Exception as e:
            logger.error(f"Error analyzing intelligence trends: {e}")
            return []

    async def _analyze_single_metric_trend(self, metric_name: str, values: List[float]) -> Optional[IntelligenceTrend]:
        """Analyze trend for a single metric using basic math"""
        try:
            if len(values) < 3:
                return None
            
            if ML_AVAILABLE:
                # Use scipy for trend analysis if available
                x = np.arange(len(values))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
                change_rate = slope
                confidence = abs(r_value)
            else:
                # Basic trend calculation
                change_rate = (values[-1] - values[0]) / len(values)
                confidence = 0.7  # Default confidence
            
            # Determine trend direction
            if abs(change_rate) < 0.001:
                direction = TrendDirection.STABLE
            elif change_rate > 0:
                direction = TrendDirection.IMPROVING
            else:
                direction = TrendDirection.DECLINING
            
            # Check for volatility
            if ML_AVAILABLE:
                std_dev = np.std(values)
                mean_val = np.mean(values)
            else:
                mean_val = sum(values) / len(values)
                std_dev = (sum((x - mean_val) ** 2 for x in values) / len(values)) ** 0.5
            
            if std_dev > mean_val * 0.3:  # High relative std dev
                direction = TrendDirection.VOLATILE
            
            # Generate recommendation
            recommendation = self._generate_trend_recommendation(metric_name, direction, change_rate, values[-1])
            
            return IntelligenceTrend(
                metric_name=metric_name,
                current_value=values[-1],
                trend_direction=direction,
                change_rate=change_rate,
                confidence=confidence,
                recommendation=recommendation
            )
            
        except Exception as e:
            logger.error(f"Error analyzing trend for {metric_name}: {e}")
            return None

    def _generate_trend_recommendation(self, metric_name: str, direction: TrendDirection, slope: float, current_value: float) -> str:
        """Generate actionable recommendations based on trend analysis"""
        if direction == TrendDirection.IMPROVING:
            return f"{metric_name} is improving (+{slope:.4f}). Continue current strategies."
        elif direction == TrendDirection.DECLINING:
            return f"{metric_name} is declining ({slope:.4f}). Investigate causes and implement improvements."
        elif direction == TrendDirection.VOLATILE:
            return f"{metric_name} shows high volatility. Stabilize environment and monitor closely."
        else:  # STABLE
            if current_value > 0.8:
                return f"{metric_name} is stable at good performance. Maintain current approach."
            else:
                return f"{metric_name} is stable but below optimal. Consider optimization strategies."

    async def predict_agent_performance(self, 
                                      agent_name: str,
                                      historical_metrics: Dict[str, List[float]],
                                      prediction_window: str = "24h") -> PerformancePrediction:
        """
        Basic performance prediction
        """
        try:
            predicted_metrics = {}
            
            # Simple prediction based on recent trends
            for metric_name, values in historical_metrics.items():
                if len(values) >= 3:
                    # Simple linear extrapolation
                    recent_values = values[-3:]
                    if ML_AVAILABLE:
                        prediction = np.mean(recent_values) + (recent_values[-1] - recent_values[0]) / len(recent_values)
                    else:
                        prediction = sum(recent_values) / len(recent_values)
                    
                    predicted_metrics[metric_name] = prediction
            
            # Generate basic recommendations
            recommendations = [
                "Monitor agent performance closely",
                "Continue current optimization strategies",
                "Review system resources if performance declines"
            ]
            
            return PerformancePrediction(
                agent_name=agent_name,
                prediction_window=prediction_window,
                predicted_metrics=predicted_metrics,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Error predicting performance for {agent_name}: {e}")
            return PerformancePrediction(
                agent_name=agent_name,
                prediction_window=prediction_window,
                predicted_metrics={},
                recommendations=["Review prediction system health"]
            )

    async def generate_intelligence_insights(self, 
                                           system_metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate basic intelligence insights from system metrics"""
        try:
            insights = []
            
            # Basic system intelligence insight
            if 'agents' in system_metrics:
                agent_count = len(system_metrics['agents'])
                insights.append({
                    "type": "system_intelligence",
                    "title": "Active Agent Count",
                    "metric": agent_count,
                    "description": f"System has {agent_count} active agents",
                    "recommendation": "System operational"
                })
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating intelligence insights: {e}")
            return [{"type": "error", "description": f"Insight generation failed: {e}"}]


# Global analytics instance
_intelligence_analytics: Optional[IntelligenceAnalytics] = None

def get_intelligence_analytics() -> IntelligenceAnalytics:
    """Get global intelligence analytics instance"""
    global _intelligence_analytics
    if _intelligence_analytics is None:
        _intelligence_analytics = IntelligenceAnalytics()
    return _intelligence_analytics
