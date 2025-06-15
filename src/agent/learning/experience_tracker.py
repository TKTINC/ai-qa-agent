"""
Track and analyze agent experiences for learning.
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import json
import numpy as np

from ..core.models import TaskOutcome, ExperienceRecord, ExperienceAnalysis
from ...core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for experience analysis"""
    success_rate: float
    average_duration: float
    user_satisfaction: float
    efficiency_score: float
    learning_velocity: float
    consistency_score: float


class ExperienceTracker:
    """Track and analyze agent experiences for learning"""
    
    def __init__(self, max_experiences: int = 10000):
        self.max_experiences = max_experiences
        self.experiences: deque = deque(maxlen=max_experiences)
        self.agent_metrics: Dict[str, PerformanceMetrics] = {}
        self.task_patterns: Dict[str, Dict] = defaultdict(dict)
        self.tool_effectiveness: Dict[str, Dict] = defaultdict(dict)
        
    async def record_experience(self,
                              agent: str,
                              task: str,
                              approach: str,
                              tools_used: List[str],
                              outcome: TaskOutcome,
                              user_satisfaction: Optional[float] = None) -> ExperienceRecord:
        """Record detailed experience data for analysis"""
        
        experience = ExperienceRecord(
            experience_id=f"exp_{datetime.now().timestamp()}",
            timestamp=datetime.now(),
            agent_name=agent,
            task_description=task,
            task_type=self._classify_task_type(task),
            approach_used=approach,
            tools_used=tools_used,
            outcome=outcome,
            user_satisfaction=user_satisfaction or 0.7,
            context_factors=await self._extract_context_factors(task, approach, tools_used),
            performance_metrics=await self._calculate_performance_metrics(outcome, user_satisfaction)
        )
        
        # Store experience
        self.experiences.append(experience)
        
        # Update real-time metrics
        await self._update_agent_metrics(agent, experience)
        await self._update_task_patterns(task, experience)
        await self._update_tool_effectiveness(tools_used, experience)
        
        logger.debug(f"Recorded experience for {agent}: {task[:50]}...")
        return experience
    
    async def analyze_experience_patterns(self, agent: str, time_period: str = "7d") -> ExperienceAnalysis:
        """Identify patterns in agent experiences that lead to success"""
        
        # Filter experiences for the agent and time period
        cutoff_time = self._parse_time_period(time_period)
        agent_experiences = [
            exp for exp in self.experiences
            if exp.agent_name == agent and exp.timestamp >= cutoff_time
        ]
        
        if not agent_experiences:
            return ExperienceAnalysis(
                agent_name=agent,
                time_period=time_period,
                total_experiences=0,
                success_patterns=[],
                failure_patterns=[],
                optimization_opportunities=[],
                performance_trends={},
                recommendations=["Insufficient experience data for analysis"]
            )
        
        # Analyze success patterns
        success_patterns = await self._analyze_success_patterns(agent_experiences)
        
        # Analyze failure patterns
        failure_patterns = await self._analyze_failure_patterns(agent_experiences)
        
        # Identify optimization opportunities
        optimization_opportunities = await self._identify_optimization_opportunities(agent_experiences)
        
        # Calculate performance trends
        performance_trends = await self._calculate_performance_trends(agent_experiences)
        
        # Generate recommendations
        recommendations = await self._generate_experience_recommendations(
            success_patterns, failure_patterns, optimization_opportunities
        )
        
        analysis = ExperienceAnalysis(
            agent_name=agent,
            time_period=time_period,
            total_experiences=len(agent_experiences),
            success_patterns=success_patterns,
            failure_patterns=failure_patterns,
            optimization_opportunities=optimization_opportunities,
            performance_trends=performance_trends,
            recommendations=recommendations,
            analysis_timestamp=datetime.now()
        )
        
        logger.info(f"Analyzed {len(agent_experiences)} experiences for {agent}: "
                   f"{len(success_patterns)} success patterns, "
                   f"{len(failure_patterns)} failure patterns")
        
        return analysis
    
    async def get_top_performing_agents(self, metric: str = "overall", limit: int = 5) -> List[Dict[str, Any]]:
        """Get top performing agents based on specified metric"""
        
        if not self.agent_metrics:
            return []
        
        # Define metric extractors
        metric_extractors = {
            "overall": lambda m: (m.success_rate * 0.4 + m.user_satisfaction * 0.3 + 
                                m.efficiency_score * 0.2 + m.consistency_score * 0.1),
            "success_rate": lambda m: m.success_rate,
            "satisfaction": lambda m: m.user_satisfaction,
            "efficiency": lambda m: m.efficiency_score,
            "learning": lambda m: m.learning_velocity,
            "consistency": lambda m: m.consistency_score
        }
        
        if metric not in metric_extractors:
            metric = "overall"
        
        # Calculate scores and sort
        agent_scores = []
        for agent_name, metrics in self.agent_metrics.items():
            score = metric_extractors[metric](metrics)
            agent_scores.append({
                "agent_name": agent_name,
                "score": score,
                "metrics": asdict(metrics),
                "experience_count": len([exp for exp in self.experiences if exp.agent_name == agent_name])
            })
        
        # Sort by score and return top performers
        agent_scores.sort(key=lambda x: x["score"], reverse=True)
        return agent_scores[:limit]
    
    async def get_task_expertise_map(self) -> Dict[str, Dict[str, float]]:
        """Get map of which agents excel at which types of tasks"""
        
        expertise_map = defaultdict(lambda: defaultdict(float))
        
        for experience in self.experiences:
            agent = experience.agent_name
            task_type = experience.task_type
            
            # Calculate expertise score based on success and satisfaction
            success_factor = 1.0 if experience.outcome.success else 0.0
            satisfaction_factor = experience.user_satisfaction / 5.0  # Normalize to 0-1
            expertise_score = (success_factor * 0.7 + satisfaction_factor * 0.3)
            
            # Update running average
            current_score = expertise_map[agent][task_type]
            experience_count = sum(1 for exp in self.experiences 
                                 if exp.agent_name == agent and exp.task_type == task_type)
            
            expertise_map[agent][task_type] = (
                current_score * (experience_count - 1) + expertise_score
            ) / experience_count
        
        return dict(expertise_map)
    
    async def identify_learning_opportunities(self, agent: str) -> List[Dict[str, Any]]:
        """Identify specific learning opportunities for an agent"""
        
        agent_experiences = [exp for exp in self.experiences if exp.agent_name == agent]
        
        if not agent_experiences:
            return []
        
        opportunities = []
        
        # Low success rate tasks
        task_success_rates = defaultdict(list)
        for exp in agent_experiences:
            success = 1.0 if exp.outcome.success else 0.0
            task_success_rates[exp.task_type].append(success)
        
        for task_type, successes in task_success_rates.items():
            success_rate = np.mean(successes)
            if success_rate < 0.7 and len(successes) >= 3:
                opportunities.append({
                    "type": "improve_task_performance",
                    "task_type": task_type,
                    "current_success_rate": success_rate,
                    "experience_count": len(successes),
                    "priority": 1.0 - success_rate,
                    "recommendation": f"Focus on improving {task_type} task performance"
                })
        
        # Tool usage optimization
        tool_usage = defaultdict(lambda: {"total": 0, "successful": 0})
        for exp in agent_experiences:
            for tool in exp.tools_used:
                tool_usage[tool]["total"] += 1
                if exp.outcome.success:
                    tool_usage[tool]["successful"] += 1
        
        for tool, stats in tool_usage.items():
            if stats["total"] >= 3:
                success_rate = stats["successful"] / stats["total"]
                if success_rate < 0.6:
                    opportunities.append({
                        "type": "improve_tool_usage",
                        "tool": tool,
                        "current_success_rate": success_rate,
                        "usage_count": stats["total"],
                        "priority": 0.8 - success_rate,
                        "recommendation": f"Improve effectiveness with {tool} tool"
                    })
        
        # Communication improvement
        recent_satisfaction = [
            exp.user_satisfaction for exp in agent_experiences[-10:]
            if exp.user_satisfaction is not None
        ]
        
        if recent_satisfaction and np.mean(recent_satisfaction) < 3.5:
            opportunities.append({
                "type": "improve_communication",
                "current_satisfaction": np.mean(recent_satisfaction),
                "sample_size": len(recent_satisfaction),
                "priority": 4.0 - np.mean(recent_satisfaction),
                "recommendation": "Focus on improving user communication and satisfaction"
            })
        
        # Sort by priority
        opportunities.sort(key=lambda x: x["priority"], reverse=True)
        return opportunities[:5]  # Top 5 opportunities
    
    def _classify_task_type(self, task: str) -> str:
        """Classify task into type categories"""
        task_lower = task.lower()
        
        if any(keyword in task_lower for keyword in ["test", "testing", "coverage"]):
            return "testing"
        elif any(keyword in task_lower for keyword in ["analyze", "analysis", "review"]):
            return "analysis"
        elif any(keyword in task_lower for keyword in ["fix", "debug", "error", "bug"]):
            return "debugging"
        elif any(keyword in task_lower for keyword in ["generate", "create", "build"]):
            return "generation"
        elif any(keyword in task_lower for keyword in ["optimize", "improve", "performance"]):
            return "optimization"
        elif any(keyword in task_lower for keyword in ["security", "vulnerability", "secure"]):
            return "security"
        else:
            return "general"
    
    async def _extract_context_factors(self, task: str, approach: str, tools_used: List[str]) -> Dict[str, Any]:
        """Extract contextual factors that might influence performance"""
        
        factors = {
            "task_complexity": self._estimate_task_complexity(task),
            "approach_type": self._categorize_approach(approach),
            "tool_count": len(tools_used),
            "specialized_tools": any(tool in ["security_scanner", "performance_profiler"] for tool in tools_used),
            "collaboration_required": "collaborate" in approach.lower() or "team" in approach.lower(),
            "time_of_day": datetime.now().hour,
            "day_of_week": datetime.now().weekday()
        }
        
        return factors
    
    def _estimate_task_complexity(self, task: str) -> float:
        """Estimate task complexity based on description"""
        complexity_indicators = {
            "simple": 0.2, "basic": 0.3, "standard": 0.5, "complex": 0.7,
            "advanced": 0.8, "comprehensive": 0.9, "multi": 0.8, "integration": 0.7
        }
        
        task_lower = task.lower()
        complexity_scores = [score for keyword, score in complexity_indicators.items() 
                           if keyword in task_lower]
        
        if not complexity_scores:
            # Estimate based on length and specific keywords
            base_complexity = min(0.8, len(task.split()) / 20)
            return base_complexity
        
        return max(complexity_scores)
    
    def _categorize_approach(self, approach: str) -> str:
        """Categorize the approach used"""
        approach_lower = approach.lower()
        
        if "step" in approach_lower or "sequential" in approach_lower:
            return "sequential"
        elif "parallel" in approach_lower or "concurrent" in approach_lower:
            return "parallel"
        elif "collaborative" in approach_lower or "team" in approach_lower:
            return "collaborative"
        elif "iterative" in approach_lower or "incremental" in approach_lower:
            return "iterative"
        else:
            return "standard"
    
    async def _calculate_performance_metrics(self, outcome: TaskOutcome, user_satisfaction: Optional[float]) -> Dict[str, float]:
        """Calculate performance metrics for the experience"""
        
        metrics = {
            "success_score": 1.0 if outcome.success else 0.0,
            "efficiency_score": outcome.success_metrics.get("efficiency", 0.5),
            "quality_score": outcome.success_metrics.get("quality", 0.5),
            "satisfaction_score": (user_satisfaction or 3.5) / 5.0,  # Normalize to 0-1
            "duration_score": 1.0 / max(1.0, outcome.success_metrics.get("duration", 1.0) / 300.0)  # Normalize to 5min baseline
        }
        
        return metrics
    
    async def _update_agent_metrics(self, agent: str, experience: ExperienceRecord) -> None:
        """Update running metrics for the agent"""
        
        if agent not in self.agent_metrics:
            self.agent_metrics[agent] = PerformanceMetrics(
                success_rate=0.7,
                average_duration=300.0,
                user_satisfaction=3.5,
                efficiency_score=0.5,
                learning_velocity=0.5,
                consistency_score=0.5
            )
        
        metrics = self.agent_metrics[agent]
        
        # Calculate updates with exponential moving average
        alpha = 0.1  # Learning rate
        
        success = 1.0 if experience.outcome.success else 0.0
        metrics.success_rate = metrics.success_rate * (1 - alpha) + success * alpha
        
        duration = experience.outcome.success_metrics.get("duration", 300.0)
        metrics.average_duration = metrics.average_duration * (1 - alpha) + duration * alpha
        
        satisfaction = experience.user_satisfaction or 3.5
        metrics.user_satisfaction = metrics.user_satisfaction * (1 - alpha) + satisfaction * alpha
        
        efficiency = experience.performance_metrics.get("efficiency_score", 0.5)
        metrics.efficiency_score = metrics.efficiency_score * (1 - alpha) + efficiency * alpha
        
        # Learning velocity based on improvement trends
        metrics.learning_velocity = min(1.0, metrics.learning_velocity + 0.01)
        
        # Consistency score based on variance in recent performance
        recent_experiences = [exp for exp in list(self.experiences)[-10:] if exp.agent_name == agent]
        if len(recent_experiences) >= 3:
            recent_successes = [1.0 if exp.outcome.success else 0.0 for exp in recent_experiences]
            consistency = 1.0 - np.std(recent_successes)
            metrics.consistency_score = metrics.consistency_score * 0.9 + consistency * 0.1
    
    async def _update_task_patterns(self, task: str, experience: ExperienceRecord) -> None:
        """Update patterns for task types"""
        
        task_type = experience.task_type
        
        if task_type not in self.task_patterns:
            self.task_patterns[task_type] = {
                "total_attempts": 0,
                "successful_attempts": 0,
                "average_duration": 300.0,
                "common_approaches": defaultdict(int),
                "effective_tools": defaultdict(float)
            }
        
        patterns = self.task_patterns[task_type]
        patterns["total_attempts"] += 1
        
        if experience.outcome.success:
            patterns["successful_attempts"] += 1
        
        # Update duration
        duration = experience.outcome.success_metrics.get("duration", 300.0)
        patterns["average_duration"] = (
            patterns["average_duration"] * 0.9 + duration * 0.1
        )
        
        # Track approach usage
        patterns["common_approaches"][experience.approach_used] += 1
        
        # Track tool effectiveness
        for tool in experience.tools_used:
            success_factor = 1.0 if experience.outcome.success else 0.0
            current_effectiveness = patterns["effective_tools"].get(tool, 0.5)
            patterns["effective_tools"][tool] = current_effectiveness * 0.8 + success_factor * 0.2
    
    async def _update_tool_effectiveness(self, tools_used: List[str], experience: ExperienceRecord) -> None:
        """Update tool effectiveness metrics"""
        
        success_factor = 1.0 if experience.outcome.success else 0.0
        satisfaction_factor = (experience.user_satisfaction or 3.5) / 5.0
        
        for tool in tools_used:
            if tool not in self.tool_effectiveness:
                self.tool_effectiveness[tool] = {
                    "usage_count": 0,
                    "success_rate": 0.7,
                    "satisfaction_rate": 0.7,
                    "efficiency_score": 0.5
                }
            
            tool_metrics = self.tool_effectiveness[tool]
            tool_metrics["usage_count"] += 1
            
            # Update metrics with exponential moving average
            alpha = 0.1
            tool_metrics["success_rate"] = tool_metrics["success_rate"] * (1 - alpha) + success_factor * alpha
            tool_metrics["satisfaction_rate"] = tool_metrics["satisfaction_rate"] * (1 - alpha) + satisfaction_factor * alpha
            
            efficiency = experience.performance_metrics.get("efficiency_score", 0.5)
            tool_metrics["efficiency_score"] = tool_metrics["efficiency_score"] * (1 - alpha) + efficiency * alpha
    
    def _parse_time_period(self, time_period: str) -> datetime:
        """Parse time period string into cutoff datetime"""
        now = datetime.now()
        
        if time_period == "1h":
            return now - timedelta(hours=1)
        elif time_period == "24h" or time_period == "1d":
            return now - timedelta(days=1)
        elif time_period == "7d" or time_period == "1w":
            return now - timedelta(days=7)
        elif time_period == "30d" or time_period == "1m":
            return now - timedelta(days=30)
        else:
            return now - timedelta(days=7)  # Default to 7 days
    
    async def _analyze_success_patterns(self, experiences: List[ExperienceRecord]) -> List[Dict[str, Any]]:
        """Analyze patterns that lead to success"""
        
        successful_experiences = [exp for exp in experiences if exp.outcome.success]
        
        if not successful_experiences:
            return []
        
        patterns = []
        
        # Analyze successful approaches
        approach_success = defaultdict(int)
        for exp in successful_experiences:
            approach_success[exp.approach_used] += 1
        
        for approach, count in approach_success.items():
            if count >= 2:  # Pattern needs at least 2 occurrences
                success_rate = count / len([exp for exp in experiences if exp.approach_used == approach])
                patterns.append({
                    "type": "successful_approach",
                    "pattern": approach,
                    "success_rate": success_rate,
                    "occurrence_count": count,
                    "confidence": min(1.0, count / 5.0)
                })
        
        # Analyze successful tool combinations
        tool_combinations = defaultdict(int)
        for exp in successful_experiences:
            if len(exp.tools_used) > 1:
                combo = tuple(sorted(exp.tools_used))
                tool_combinations[combo] += 1
        
        for combo, count in tool_combinations.items():
            if count >= 2:
                patterns.append({
                    "type": "successful_tool_combination",
                    "pattern": list(combo),
                    "occurrence_count": count,
                    "confidence": min(1.0, count / 3.0)
                })
        
        return patterns
    
    async def _analyze_failure_patterns(self, experiences: List[ExperienceRecord]) -> List[Dict[str, Any]]:
        """Analyze patterns that lead to failure"""
        
        failed_experiences = [exp for exp in experiences if not exp.outcome.success]
        
        if not failed_experiences:
            return []
        
        patterns = []
        
        # Analyze failure approaches
        approach_failures = defaultdict(int)
        for exp in failed_experiences:
            approach_failures[exp.approach_used] += 1
        
        for approach, count in approach_failures.items():
            if count >= 2:
                failure_rate = count / len([exp for exp in experiences if exp.approach_used == approach])
                if failure_rate > 0.4:  # Only flag high failure rates
                    patterns.append({
                        "type": "problematic_approach",
                        "pattern": approach,
                        "failure_rate": failure_rate,
                        "occurrence_count": count,
                        "recommendation": f"Review and improve {approach} approach"
                    })
        
        # Analyze problematic tool usage
        tool_failures = defaultdict(int)
        tool_totals = defaultdict(int)
        
        for exp in experiences:
            for tool in exp.tools_used:
                tool_totals[tool] += 1
                if not exp.outcome.success:
                    tool_failures[tool] += 1
        
        for tool, failure_count in tool_failures.items():
            if tool_totals[tool] >= 3:  # Need sufficient sample size
                failure_rate = failure_count / tool_totals[tool]
                if failure_rate > 0.5:
                    patterns.append({
                        "type": "problematic_tool",
                        "pattern": tool,
                        "failure_rate": failure_rate,
                        "usage_count": tool_totals[tool],
                        "recommendation": f"Improve {tool} usage or find alternatives"
                    })
        
        return patterns
    
    async def _identify_optimization_opportunities(self, experiences: List[ExperienceRecord]) -> List[Dict[str, Any]]:
        """Identify opportunities for optimization"""
        
        opportunities = []
        
        # Duration optimization
        durations = [exp.outcome.success_metrics.get("duration", 300.0) for exp in experiences]
        if durations:
            avg_duration = np.mean(durations)
            if avg_duration > 600:  # More than 10 minutes average
                opportunities.append({
                    "type": "duration_optimization",
                    "current_average": avg_duration,
                    "target_average": 300.0,
                    "potential_improvement": (avg_duration - 300.0) / avg_duration,
                    "recommendation": "Focus on reducing task completion time"
                })
        
        # Tool efficiency optimization
        tool_usage = defaultdict(list)
        for exp in experiences:
            for tool in exp.tools_used:
                efficiency = exp.performance_metrics.get("efficiency_score", 0.5)
                tool_usage[tool].append(efficiency)
        
        for tool, efficiencies in tool_usage.items():
            if len(efficiencies) >= 3:
                avg_efficiency = np.mean(efficiencies)
                if avg_efficiency < 0.6:
                    opportunities.append({
                        "type": "tool_efficiency_optimization",
                        "tool": tool,
                        "current_efficiency": avg_efficiency,
                        "usage_count": len(efficiencies),
                        "recommendation": f"Improve efficiency with {tool} tool"
                    })
        
        return opportunities
    
    async def _calculate_performance_trends(self, experiences: List[ExperienceRecord]) -> Dict[str, Any]:
        """Calculate performance trends over time"""
        
        if len(experiences) < 5:
            return {"trend": "insufficient_data"}
        
        # Sort by timestamp
        sorted_experiences = sorted(experiences, key=lambda x: x.timestamp)
        
        # Calculate trends for key metrics
        success_rates = []
        satisfaction_scores = []
        efficiency_scores = []
        
        window_size = max(3, len(sorted_experiences) // 5)
        
        for i in range(len(sorted_experiences) - window_size + 1):
            window = sorted_experiences[i:i + window_size]
            
            successes = [1.0 if exp.outcome.success else 0.0 for exp in window]
            success_rates.append(np.mean(successes))
            
            satisfactions = [exp.user_satisfaction or 3.5 for exp in window]
            satisfaction_scores.append(np.mean(satisfactions))
            
            efficiencies = [exp.performance_metrics.get("efficiency_score", 0.5) for exp in window]
            efficiency_scores.append(np.mean(efficiencies))
        
        trends = {}
        
        # Calculate trend direction
        if len(success_rates) >= 2:
            success_trend = "improving" if success_rates[-1] > success_rates[0] else "declining"
            satisfaction_trend = "improving" if satisfaction_scores[-1] > satisfaction_scores[0] else "declining"
            efficiency_trend = "improving" if efficiency_scores[-1] > efficiency_scores[0] else "declining"
            
            trends = {
                "success_rate_trend": success_trend,
                "satisfaction_trend": satisfaction_trend,
                "efficiency_trend": efficiency_trend,
                "overall_trend": "improving" if sum([
                    success_rates[-1] > success_rates[0],
                    satisfaction_scores[-1] > satisfaction_scores[0],
                    efficiency_scores[-1] > efficiency_scores[0]
                ]) >= 2 else "declining"
            }
        
        return trends
    
    async def _generate_experience_recommendations(self,
                                                 success_patterns: List[Dict],
                                                 failure_patterns: List[Dict],
                                                 optimization_opportunities: List[Dict]) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        
        recommendations = []
        
        # Recommendations based on success patterns
        for pattern in success_patterns[:3]:  # Top 3 success patterns
            if pattern["type"] == "successful_approach":
                recommendations.append(
                    f"Continue using '{pattern['pattern']}' approach (success rate: {pattern['success_rate']:.1%})"
                )
            elif pattern["type"] == "successful_tool_combination":
                tools = ", ".join(pattern["pattern"])
                recommendations.append(f"Effective tool combination found: {tools}")
        
        # Recommendations based on failure patterns
        for pattern in failure_patterns[:2]:  # Top 2 failure patterns
            recommendations.append(pattern.get("recommendation", "Address identified failure pattern"))
        
        # Recommendations based on optimization opportunities
        for opportunity in optimization_opportunities[:2]:  # Top 2 opportunities
            recommendations.append(opportunity.get("recommendation", "Consider optimization opportunity"))
        
        if not recommendations:
            recommendations.append("Continue current performance level - no critical issues identified")
        
        return recommendations
