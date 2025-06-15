"""
Central learning system for all agent capabilities.
"""

import asyncio
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import numpy as np
from pathlib import Path

from .experience_tracker import ExperienceTracker
from .feedback_processor import FeedbackProcessor
from .personalization import PersonalizationEngine
from .continuous_improvement import ContinuousImprovementSystem
from .cross_agent_learning import CrossAgentLearning
from .quality_assessment import LearningQualityAssessment
from ..core.models import AgentInteraction, InteractionOutcome, UserFeedback, LearningUpdate, CapabilityUpdate
from ...core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class LearningConfiguration:
    """Configuration for learning system"""
    learning_rate: float = 0.1
    experience_window: int = 1000
    feedback_weight: float = 0.3
    cross_agent_learning: bool = True
    real_time_adaptation: bool = True
    quality_threshold: float = 0.7
    personalization_enabled: bool = True


class AgentLearningEngine:
    """Central learning system for all agent capabilities"""
    
    def __init__(self, config: Optional[LearningConfiguration] = None):
        self.config = config or LearningConfiguration()
        
        # Core learning components
        self.experience_tracker = ExperienceTracker()
        self.feedback_processor = FeedbackProcessor()
        self.personalization_engine = PersonalizationEngine()
        self.continuous_improvement = ContinuousImprovementSystem()
        self.cross_agent_learning = CrossAgentLearning()
        self.quality_assessment = LearningQualityAssessment()
        
        # Learning state
        self.learning_history: List[Dict] = []
        self.agent_capabilities: Dict[str, Dict] = {}
        self.global_patterns: Dict[str, Any] = {}
        self.learning_metrics: Dict[str, float] = {}
        
        # Real-time learning queue
        self.real_time_queue: deque = deque(maxlen=100)
        
        logger.info("Agent Learning Engine initialized")
    
    async def learn_from_interaction(self,
                                   interaction: AgentInteraction,
                                   outcome: InteractionOutcome,
                                   user_feedback: Optional[UserFeedback] = None) -> LearningUpdate:
        """Learn from each agent-user interaction"""
        
        learning_start = datetime.now()
        
        try:
            # 1. Record experience
            experience_record = await self.experience_tracker.record_experience(
                agent=interaction.agent_name,
                task=interaction.task_description,
                approach=interaction.approach_used,
                tools_used=interaction.tools_used,
                outcome=outcome,
                user_satisfaction=user_feedback.satisfaction_rating if user_feedback else None
            )
            
            # 2. Process user feedback if provided
            feedback_insights = None
            if user_feedback:
                feedback_insights = await self.feedback_processor.process_immediate_feedback(
                    feedback=user_feedback,
                    interaction_context=interaction.context
                )
            
            # 3. Extract learning patterns
            learning_patterns = await self._extract_learning_patterns(
                interaction, outcome, user_feedback
            )
            
            # 4. Update agent capabilities
            capability_updates = await self._update_agent_capabilities(
                interaction.agent_name, learning_patterns, outcome
            )
            
            # 5. Apply personalization learning
            personalization_update = None
            if user_feedback and hasattr(user_feedback, 'user_id'):
                personalization_update = await self.personalization_engine.learn_user_preferences(
                    user_id=user_feedback.user_id,
                    interaction=interaction,
                    feedback=user_feedback
                )
            
            # 6. Cross-agent knowledge sharing
            if self.config.cross_agent_learning:
                await self.cross_agent_learning.share_learning_insights(
                    source_agent=interaction.agent_name,
                    learning_patterns=learning_patterns,
                    success_indicators=outcome.success_metrics
                )
            
            # 7. Real-time adaptation
            if self.config.real_time_adaptation:
                await self._apply_real_time_adaptations(learning_patterns)
            
            # 8. Quality assessment
            learning_quality = await self.quality_assessment.assess_learning_quality(
                learning_patterns, feedback_insights, capability_updates
            )
            
            # Create comprehensive learning update
            learning_update = LearningUpdate(
                interaction_id=interaction.interaction_id,
                timestamp=learning_start,
                learning_patterns=learning_patterns,
                capability_updates=capability_updates,
                personalization_update=personalization_update,
                feedback_insights=feedback_insights,
                quality_score=learning_quality.overall_score,
                processing_time=(datetime.now() - learning_start).total_seconds()
            )
            
            # Store in learning history
            self.learning_history.append({
                'timestamp': learning_start.isoformat(),
                'interaction_id': interaction.interaction_id,
                'agent': interaction.agent_name,
                'learning_score': learning_quality.overall_score,
                'patterns_learned': len(learning_patterns),
                'capabilities_updated': len(capability_updates)
            })
            
            # Update metrics
            await self._update_learning_metrics(learning_update)
            
            logger.info(f"Learning completed for interaction {interaction.interaction_id}: "
                       f"quality={learning_quality.overall_score:.3f}, "
                       f"patterns={len(learning_patterns)}, "
                       f"time={learning_update.processing_time:.3f}s")
            
            return learning_update
            
        except Exception as e:
            logger.error(f"Learning from interaction failed: {str(e)}")
            raise
    
    async def improve_agent_capabilities(self, agent_name: str, improvement_areas: List[str]) -> CapabilityUpdate:
        """Continuously improve specific agent capabilities"""
        
        # Get current capability state
        current_capabilities = self.agent_capabilities.get(agent_name, {})
        
        # Analyze improvement opportunities
        improvement_analysis = await self.continuous_improvement.analyze_improvement_opportunities(
            agent_name=agent_name,
            current_capabilities=current_capabilities,
            target_areas=improvement_areas,
            historical_data=self._get_agent_history(agent_name)
        )
        
        # Generate capability improvements
        capability_improvements = await self.continuous_improvement.generate_capability_improvements(
            agent_name=agent_name,
            analysis=improvement_analysis
        )
        
        # Apply improvements
        updated_capabilities = await self._apply_capability_improvements(
            agent_name, capability_improvements
        )
        
        # Validate improvements
        validation_result = await self.quality_assessment.validate_capability_improvements(
            agent_name=agent_name,
            before_capabilities=current_capabilities,
            after_capabilities=updated_capabilities
        )
        
        capability_update = CapabilityUpdate(
            agent_name=agent_name,
            improvement_areas=improvement_areas,
            improvements_applied=capability_improvements,
            validation_result=validation_result,
            timestamp=datetime.now()
        )
        
        logger.info(f"Improved capabilities for {agent_name}: "
                   f"areas={len(improvement_areas)}, "
                   f"improvements={len(capability_improvements)}")
        
        return capability_update
    
    async def get_learning_insights(self, time_window: Optional[str] = "24h") -> Dict[str, Any]:
        """Get comprehensive learning insights"""
        
        # Parse time window
        if time_window == "24h":
            since = datetime.now() - timedelta(hours=24)
        elif time_window == "7d":
            since = datetime.now() - timedelta(days=7)
        elif time_window == "30d":
            since = datetime.now() - timedelta(days=30)
        else:
            since = datetime.now() - timedelta(hours=24)
        
        # Filter recent learning data
        recent_learning = [
            entry for entry in self.learning_history
            if datetime.fromisoformat(entry['timestamp']) >= since
        ]
        
        # Calculate insights
        insights = {
            'time_window': time_window,
            'total_interactions': len(recent_learning),
            'average_learning_score': np.mean([entry['learning_score'] for entry in recent_learning]) if recent_learning else 0,
            'total_patterns_learned': sum(entry['patterns_learned'] for entry in recent_learning),
            'agents_improved': len(set(entry['agent'] for entry in recent_learning)),
            'learning_velocity': len(recent_learning) / 24 if time_window == "24h" else len(recent_learning) / 7,
            'quality_trend': await self._calculate_quality_trend(recent_learning),
            'top_learning_agents': await self._get_top_learning_agents(recent_learning),
            'improvement_opportunities': await self._identify_improvement_opportunities(recent_learning)
        }
        
        return insights
    
    async def predict_interaction_outcome(self, 
                                        planned_interaction: Dict[str, Any]) -> Dict[str, Any]:
        """Predict likely outcome of a planned interaction"""
        
        agent_name = planned_interaction.get('agent_name')
        task_type = planned_interaction.get('task_type')
        user_context = planned_interaction.get('user_context', {})
        
        # Get historical data for similar interactions
        similar_interactions = await self._find_similar_interactions(
            agent_name, task_type, user_context
        )
        
        if not similar_interactions:
            return {
                'confidence': 0.1,
                'predicted_success_rate': 0.7,  # Default assumption
                'expected_duration': 300,  # 5 minutes default
                'recommendations': ['No historical data available for this interaction type']
            }
        
        # Calculate predictions based on historical data
        success_rates = [interaction.get('success_rate', 0.7) for interaction in similar_interactions]
        durations = [interaction.get('duration', 300) for interaction in similar_interactions]
        satisfaction_scores = [interaction.get('satisfaction', 0.7) for interaction in similar_interactions]
        
        prediction = {
            'confidence': min(1.0, len(similar_interactions) / 10),  # Higher confidence with more data
            'predicted_success_rate': np.mean(success_rates),
            'expected_duration': np.mean(durations),
            'predicted_satisfaction': np.mean(satisfaction_scores),
            'uncertainty_range': {
                'success_rate': (np.min(success_rates), np.max(success_rates)),
                'duration': (np.min(durations), np.max(durations)),
                'satisfaction': (np.min(satisfaction_scores), np.max(satisfaction_scores))
            },
            'recommendations': await self._generate_interaction_recommendations(similar_interactions),
            'risk_factors': await self._identify_interaction_risks(similar_interactions),
            'similar_interactions_count': len(similar_interactions)
        }
        
        return prediction
    
    async def _extract_learning_patterns(self,
                                       interaction: AgentInteraction,
                                       outcome: InteractionOutcome,
                                       feedback: Optional[UserFeedback]) -> List[Dict[str, Any]]:
        """Extract learnable patterns from interaction"""
        
        patterns = []
        
        # Success pattern analysis
        if outcome.success:
            patterns.append({
                'type': 'success_pattern',
                'agent': interaction.agent_name,
                'task_type': interaction.task_type,
                'approach': interaction.approach_used,
                'tools': interaction.tools_used,
                'context': interaction.context,
                'success_metrics': outcome.success_metrics,
                'confidence': 0.8
            })
        
        # Tool usage patterns
        if interaction.tools_used:
            for tool in interaction.tools_used:
                patterns.append({
                    'type': 'tool_usage_pattern',
                    'tool': tool,
                    'task_type': interaction.task_type,
                    'success': outcome.success,
                    'effectiveness': outcome.success_metrics.get('tool_effectiveness', {}).get(tool, 0.5),
                    'confidence': 0.6
                })
        
        # User satisfaction patterns
        if feedback and hasattr(feedback, 'satisfaction_rating'):
            patterns.append({
                'type': 'satisfaction_pattern',
                'agent': interaction.agent_name,
                'satisfaction_score': feedback.satisfaction_rating,
                'task_type': interaction.task_type,
                'communication_style': getattr(feedback, 'communication_preference', 'unknown'),
                'confidence': 0.7
            })
        
        # Failure pattern analysis
        if not outcome.success and outcome.failure_reasons:
            patterns.append({
                'type': 'failure_pattern',
                'agent': interaction.agent_name,
                'task_type': interaction.task_type,
                'failure_reasons': outcome.failure_reasons,
                'attempted_approach': interaction.approach_used,
                'confidence': 0.9
            })
        
        # Timing patterns
        if hasattr(interaction, 'duration'):
            patterns.append({
                'type': 'timing_pattern',
                'task_type': interaction.task_type,
                'duration': interaction.duration,
                'complexity': getattr(interaction, 'complexity_score', 0.5),
                'efficiency': outcome.success_metrics.get('efficiency', 0.5),
                'confidence': 0.5
            })
        
        return patterns
    
    async def _update_agent_capabilities(self,
                                       agent_name: str,
                                       learning_patterns: List[Dict],
                                       outcome: InteractionOutcome) -> List[Dict[str, Any]]:
        """Update agent capabilities based on learning patterns"""
        
        if agent_name not in self.agent_capabilities:
            self.agent_capabilities[agent_name] = {
                'problem_solving_score': 0.7,
                'tool_usage_efficiency': 0.7,
                'user_communication': 0.7,
                'learning_velocity': 0.5,
                'specialization_areas': [],
                'success_patterns': [],
                'improvement_areas': []
            }
        
        capabilities = self.agent_capabilities[agent_name]
        updates = []
        
        # Update based on success patterns
        success_patterns = [p for p in learning_patterns if p['type'] == 'success_pattern']
        if success_patterns:
            old_score = capabilities['problem_solving_score']
            capabilities['problem_solving_score'] = min(1.0, capabilities['problem_solving_score'] + 0.01)
            updates.append({
                'capability': 'problem_solving_score',
                'old_value': old_score,
                'new_value': capabilities['problem_solving_score'],
                'reason': f'Success in {len(success_patterns)} interactions'
            })
        
        # Update tool usage efficiency
        tool_patterns = [p for p in learning_patterns if p['type'] == 'tool_usage_pattern']
        if tool_patterns:
            avg_effectiveness = np.mean([p['effectiveness'] for p in tool_patterns])
            old_efficiency = capabilities['tool_usage_efficiency']
            capabilities['tool_usage_efficiency'] = (
                capabilities['tool_usage_efficiency'] * 0.9 + avg_effectiveness * 0.1
            )
            updates.append({
                'capability': 'tool_usage_efficiency',
                'old_value': old_efficiency,
                'new_value': capabilities['tool_usage_efficiency'],
                'reason': f'Tool usage feedback from {len(tool_patterns)} interactions'
            })
        
        # Update communication based on satisfaction
        satisfaction_patterns = [p for p in learning_patterns if p['type'] == 'satisfaction_pattern']
        if satisfaction_patterns:
            avg_satisfaction = np.mean([p['satisfaction_score'] for p in satisfaction_patterns]) / 5.0  # Normalize to 0-1
            old_communication = capabilities['user_communication']
            capabilities['user_communication'] = (
                capabilities['user_communication'] * 0.8 + avg_satisfaction * 0.2
            )
            updates.append({
                'capability': 'user_communication',
                'old_value': old_communication,
                'new_value': capabilities['user_communication'],
                'reason': f'User satisfaction feedback from {len(satisfaction_patterns)} interactions'
            })
        
        # Update learning velocity
        capabilities['learning_velocity'] = min(1.0, capabilities['learning_velocity'] + 0.005)
        
        return updates
    
    async def _apply_real_time_adaptations(self, learning_patterns: List[Dict]) -> None:
        """Apply real-time adaptations based on immediate learning"""
        
        # Add to real-time queue
        self.real_time_queue.append({
            'timestamp': datetime.now(),
            'patterns': learning_patterns
        })
        
        # Apply immediate adaptations for critical patterns
        for pattern in learning_patterns:
            if pattern.get('confidence', 0) > 0.8:
                if pattern['type'] == 'failure_pattern':
                    # Immediate adjustment for failure patterns
                    await self._apply_failure_adaptation(pattern)
                elif pattern['type'] == 'success_pattern':
                    # Reinforce successful patterns
                    await self._reinforce_success_pattern(pattern)
    
    async def _apply_failure_adaptation(self, failure_pattern: Dict) -> None:
        """Apply immediate adaptation for failure patterns"""
        agent_name = failure_pattern['agent']
        task_type = failure_pattern['task_type']
        
        # Reduce confidence in failed approach
        if agent_name in self.agent_capabilities:
            failure_key = f"{task_type}_{failure_pattern.get('attempted_approach', 'unknown')}"
            if 'failure_adaptations' not in self.agent_capabilities[agent_name]:
                self.agent_capabilities[agent_name]['failure_adaptations'] = {}
            
            self.agent_capabilities[agent_name]['failure_adaptations'][failure_key] = {
                'timestamp': datetime.now().isoformat(),
                'avoid_approach': failure_pattern.get('attempted_approach'),
                'failure_reasons': failure_pattern.get('failure_reasons', [])
            }
    
    async def _reinforce_success_pattern(self, success_pattern: Dict) -> None:
        """Reinforce successful interaction patterns"""
        agent_name = success_pattern['agent']
        
        if agent_name in self.agent_capabilities:
            if 'success_patterns' not in self.agent_capabilities[agent_name]:
                self.agent_capabilities[agent_name]['success_patterns'] = []
            
            # Add or update success pattern
            self.agent_capabilities[agent_name]['success_patterns'].append({
                'task_type': success_pattern['task_type'],
                'approach': success_pattern['approach'],
                'tools': success_pattern['tools'],
                'success_score': success_pattern.get('success_metrics', {}).get('overall_score', 0.8),
                'timestamp': datetime.now().isoformat()
            })
            
            # Keep only recent success patterns
            self.agent_capabilities[agent_name]['success_patterns'] = \
                self.agent_capabilities[agent_name]['success_patterns'][-20:]
    
    async def _update_learning_metrics(self, learning_update: LearningUpdate) -> None:
        """Update global learning metrics"""
        
        # Update learning rate metrics
        self.learning_metrics['total_interactions'] = self.learning_metrics.get('total_interactions', 0) + 1
        self.learning_metrics['average_quality'] = (
            self.learning_metrics.get('average_quality', 0.7) * 0.95 + 
            learning_update.quality_score * 0.05
        )
        
        # Update processing time metrics
        self.learning_metrics['average_processing_time'] = (
            self.learning_metrics.get('average_processing_time', 1.0) * 0.9 +
            learning_update.processing_time * 0.1
        )
        
        # Update pattern learning metrics
        pattern_count = len(learning_update.learning_patterns)
        self.learning_metrics['patterns_per_interaction'] = (
            self.learning_metrics.get('patterns_per_interaction', 2.0) * 0.9 +
            pattern_count * 0.1
        )
    
    async def _calculate_quality_trend(self, recent_learning: List[Dict]) -> str:
        """Calculate quality trend from recent learning data"""
        if len(recent_learning) < 5:
            return "insufficient_data"
        
        # Get quality scores over time
        scores = [entry['learning_score'] for entry in recent_learning[-10:]]
        
        # Simple trend calculation
        first_half = np.mean(scores[:len(scores)//2])
        second_half = np.mean(scores[len(scores)//2:])
        
        if second_half > first_half + 0.05:
            return "improving"
        elif second_half < first_half - 0.05:
            return "declining"
        else:
            return "stable"
    
    async def _get_top_learning_agents(self, recent_learning: List[Dict]) -> List[Dict]:
        """Get agents with highest learning performance"""
        agent_scores = defaultdict(list)
        
        for entry in recent_learning:
            agent_scores[entry['agent']].append(entry['learning_score'])
        
        agent_averages = {
            agent: np.mean(scores) 
            for agent, scores in agent_scores.items()
        }
        
        sorted_agents = sorted(agent_averages.items(), key=lambda x: x[1], reverse=True)
        
        return [
            {
                'agent': agent,
                'average_score': score,
                'interaction_count': len(agent_scores[agent])
            }
            for agent, score in sorted_agents[:5]
        ]
    
    async def _identify_improvement_opportunities(self, recent_learning: List[Dict]) -> List[str]:
        """Identify opportunities for system improvement"""
        opportunities = []
        
        # Low average scores
        avg_score = np.mean([entry['learning_score'] for entry in recent_learning]) if recent_learning else 0.7
        if avg_score < 0.6:
            opportunities.append("Overall learning quality below threshold - review learning algorithms")
        
        # Low pattern extraction
        avg_patterns = np.mean([entry['patterns_learned'] for entry in recent_learning]) if recent_learning else 2.0
        if avg_patterns < 1.5:
            opportunities.append("Low pattern extraction rate - enhance pattern recognition")
        
        # Agent performance disparities
        agent_scores = defaultdict(list)
        for entry in recent_learning:
            agent_scores[entry['agent']].append(entry['learning_score'])
        
        if len(agent_scores) > 1:
            agent_averages = [np.mean(scores) for scores in agent_scores.values()]
            if max(agent_averages) - min(agent_averages) > 0.3:
                opportunities.append("High performance variation between agents - balance training")
        
        return opportunities
    
    async def _find_similar_interactions(self, 
                                       agent_name: str,
                                       task_type: str,
                                       user_context: Dict) -> List[Dict]:
        """Find historically similar interactions"""
        # This would typically query a database of past interactions
        # For now, return simulated similar interactions
        return []
    
    async def _generate_interaction_recommendations(self, similar_interactions: List[Dict]) -> List[str]:
        """Generate recommendations based on similar interactions"""
        recommendations = []
        
        if not similar_interactions:
            return ["No historical data available for recommendations"]
        
        # Analyze success factors
        successful_interactions = [i for i in similar_interactions if i.get('success', False)]
        
        if successful_interactions:
            # Find common tools in successful interactions
            common_tools = set.intersection(*[
                set(interaction.get('tools_used', [])) 
                for interaction in successful_interactions
            ])
            
            if common_tools:
                recommendations.append(f"Consider using tools: {', '.join(common_tools)}")
        
        return recommendations
    
    async def _identify_interaction_risks(self, similar_interactions: List[Dict]) -> List[str]:
        """Identify potential risks based on historical data"""
        risks = []
        
        if not similar_interactions:
            return []
        
        # Calculate failure rate
        failure_rate = len([i for i in similar_interactions if not i.get('success', True)]) / len(similar_interactions)
        
        if failure_rate > 0.3:
            risks.append(f"High failure rate ({failure_rate:.1%}) for this interaction type")
        
        # Identify common failure reasons
        failures = [i for i in similar_interactions if not i.get('success', True)]
        if failures:
            failure_reasons = []
            for failure in failures:
                failure_reasons.extend(failure.get('failure_reasons', []))
            
            if failure_reasons:
                common_reasons = set(failure_reasons)
                risks.extend([f"Risk: {reason}" for reason in common_reasons])
        
        return risks
    
    def _get_agent_history(self, agent_name: str) -> List[Dict]:
        """Get historical data for specific agent"""
        return [
            entry for entry in self.learning_history
            if entry.get('agent') == agent_name
        ]
    
    async def _apply_capability_improvements(self, 
                                           agent_name: str,
                                           improvements: List[Dict]) -> Dict[str, Any]:
        """Apply capability improvements to agent"""
        
        if agent_name not in self.agent_capabilities:
            self.agent_capabilities[agent_name] = {}
        
        capabilities = self.agent_capabilities[agent_name]
        
        for improvement in improvements:
            capability = improvement['capability']
            adjustment = improvement['adjustment']
            
            old_value = capabilities.get(capability, 0.5)
            new_value = max(0.0, min(1.0, old_value + adjustment))
            capabilities[capability] = new_value
            
            logger.info(f"Applied improvement to {agent_name}.{capability}: {old_value:.3f} -> {new_value:.3f}")
        
        return capabilities
