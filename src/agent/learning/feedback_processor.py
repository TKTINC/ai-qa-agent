"""
Process and learn from user feedback.
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import numpy as np
import re

from ..core.models import UserFeedback, InteractionContext, FeedbackInsights, OutcomeInsights, ImprovementAreas
from ...core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class FeedbackPattern:
    """Represents a pattern in user feedback"""
    pattern_type: str
    pattern_description: str
    frequency: int
    sentiment: str
    confidence: float
    examples: List[str]
    recommendations: List[str]


class FeedbackProcessor:
    """Process and learn from user feedback"""
    
    def __init__(self):
        self.feedback_history: List[UserFeedback] = []
        self.sentiment_patterns: Dict[str, Dict] = defaultdict(dict)
        self.improvement_tracking: Dict[str, List] = defaultdict(list)
        self.user_preferences: Dict[str, Dict] = defaultdict(dict)
        
        # Sentiment keywords for basic sentiment analysis
        self.positive_keywords = {
            'excellent', 'great', 'good', 'helpful', 'useful', 'clear', 'accurate',
            'fast', 'efficient', 'perfect', 'amazing', 'outstanding', 'thorough',
            'comprehensive', 'detailed', 'precise', 'effective', 'smart'
        }
        
        self.negative_keywords = {
            'bad', 'poor', 'wrong', 'slow', 'confusing', 'unclear', 'useless',
            'incomplete', 'inaccurate', 'terrible', 'awful', 'frustrating',
            'difficult', 'complex', 'overwhelming', 'insufficient', 'inadequate'
        }
        
        self.improvement_keywords = {
            'could', 'should', 'might', 'better', 'improve', 'enhance', 'fix',
            'change', 'modify', 'update', 'add', 'remove', 'simplify', 'clarify'
        }
    
    async def process_immediate_feedback(self,
                                       feedback: UserFeedback,
                                       interaction_context: InteractionContext) -> FeedbackInsights:
        """Process real-time user feedback during conversations"""
        
        # Store feedback
        self.feedback_history.append(feedback)
        
        # Analyze sentiment
        sentiment_analysis = await self._analyze_sentiment(feedback.feedback_text or "")
        
        # Extract specific insights
        specific_insights = await self._extract_specific_insights(feedback, interaction_context)
        
        # Identify immediate improvements
        immediate_improvements = await self._identify_immediate_improvements(feedback, interaction_context)
        
        # Update user preferences
        preference_updates = await self._update_user_preferences(feedback, interaction_context)
        
        # Generate actionable recommendations
        recommendations = await self._generate_immediate_recommendations(
            feedback, sentiment_analysis, specific_insights
        )
        
        feedback_insights = FeedbackInsights(
            feedback_id=getattr(feedback, 'feedback_id', f"fb_{datetime.now().timestamp()}"),
            timestamp=datetime.now(),
            sentiment_score=sentiment_analysis['score'],
            sentiment_category=sentiment_analysis['category'],
            specific_insights=specific_insights,
            improvement_suggestions=immediate_improvements,
            preference_updates=preference_updates,
            recommendations=recommendations,
            confidence=sentiment_analysis['confidence']
        )
        
        # Apply immediate adaptations
        if hasattr(feedback, 'user_id') and feedback.user_id:
            await self._apply_immediate_adaptations(feedback.user_id, feedback_insights, interaction_context)
        
        logger.info(f"Processed immediate feedback: sentiment={sentiment_analysis['category']}, "
                   f"insights={len(specific_insights)}, recommendations={len(recommendations)}")
        
        return feedback_insights
    
    async def process_outcome_feedback(self,
                                     task_id: str,
                                     outcome_rating: float,
                                     detailed_feedback: str) -> OutcomeInsights:
        """Learn from task completion feedback"""
        
        # Analyze outcome feedback
        outcome_analysis = await self._analyze_outcome_feedback(outcome_rating, detailed_feedback)
        
        # Extract success factors
        success_factors = await self._extract_success_factors(outcome_rating, detailed_feedback)
        
        # Identify failure points
        failure_points = await self._extract_failure_points(outcome_rating, detailed_feedback)
        
        # Generate improvement strategies
        improvement_strategies = await self._generate_improvement_strategies(
            outcome_analysis, success_factors, failure_points
        )
        
        # Update outcome tracking
        await self._update_outcome_tracking(task_id, outcome_rating, outcome_analysis)
        
        outcome_insights = OutcomeInsights(
            task_id=task_id,
            outcome_rating=outcome_rating,
            outcome_analysis=outcome_analysis,
            success_factors=success_factors,
            failure_points=failure_points,
            improvement_strategies=improvement_strategies,
            timestamp=datetime.now()
        )
        
        logger.info(f"Processed outcome feedback for {task_id}: rating={outcome_rating}, "
                   f"success_factors={len(success_factors)}, strategies={len(improvement_strategies)}")
        
        return outcome_insights
    
    async def identify_improvement_opportunities(self, feedback_history: List[UserFeedback]) -> ImprovementAreas:
        """Identify systematic areas for agent improvement"""
        
        if not feedback_history:
            return ImprovementAreas(
                communication_improvements=[],
                technical_improvements=[],
                process_improvements=[],
                user_experience_improvements=[],
                priority_ranking=[],
                estimated_impact={}
            )
        
        # Analyze feedback patterns
        patterns = await self._analyze_feedback_patterns(feedback_history)
        
        # Categorize improvement areas
        communication_improvements = await self._identify_communication_improvements(patterns)
        technical_improvements = await self._identify_technical_improvements(patterns)
        process_improvements = await self._identify_process_improvements(patterns)
        ux_improvements = await self._identify_ux_improvements(patterns)
        
        # Calculate priority ranking
        all_improvements = (
            communication_improvements + technical_improvements + 
            process_improvements + ux_improvements
        )
        priority_ranking = await self._calculate_improvement_priority(all_improvements, patterns)
        
        # Estimate impact
        estimated_impact = await self._estimate_improvement_impact(all_improvements, feedback_history)
        
        improvement_areas = ImprovementAreas(
            communication_improvements=communication_improvements,
            technical_improvements=technical_improvements,
            process_improvements=process_improvements,
            user_experience_improvements=ux_improvements,
            priority_ranking=priority_ranking,
            estimated_impact=estimated_impact
        )
        
        logger.info(f"Identified improvement opportunities: "
                   f"comm={len(communication_improvements)}, "
                   f"tech={len(technical_improvements)}, "
                   f"process={len(process_improvements)}, "
                   f"ux={len(ux_improvements)}")
        
        return improvement_areas
    
    async def get_user_satisfaction_trends(self, 
                                         user_id: Optional[str] = None,
                                         time_window: str = "30d") -> Dict[str, Any]:
        """Analyze user satisfaction trends over time"""
        
        # Filter feedback by user and time
        cutoff_time = self._parse_time_window(time_window)
        filtered_feedback = [
            fb for fb in self.feedback_history
            if fb.timestamp >= cutoff_time and (not user_id or getattr(fb, 'user_id', None) == user_id)
        ]
        
        if not filtered_feedback:
            return {"trend": "no_data", "message": "Insufficient feedback data for analysis"}
        
        # Calculate satisfaction trends
        satisfaction_scores = [fb.satisfaction_rating for fb in filtered_feedback]
        timestamps = [fb.timestamp for fb in filtered_feedback]
        
        # Group by time periods for trend analysis
        daily_satisfaction = defaultdict(list)
        for fb in filtered_feedback:
            day_key = fb.timestamp.strftime("%Y-%m-%d")
            daily_satisfaction[day_key].append(fb.satisfaction_rating)
        
        daily_averages = {
            day: np.mean(scores) for day, scores in daily_satisfaction.items()
        }
        
        # Calculate trend direction
        if len(daily_averages) >= 3:
            recent_days = sorted(daily_averages.keys())[-7:]  # Last 7 days
            early_avg = np.mean([daily_averages[day] for day in recent_days[:len(recent_days)//2]])
            late_avg = np.mean([daily_averages[day] for day in recent_days[len(recent_days)//2:]])
            
            if late_avg > early_avg + 0.2:
                trend = "improving"
            elif late_avg < early_avg - 0.2:
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"
        
        return {
            "trend": trend,
            "overall_satisfaction": np.mean(satisfaction_scores),
            "satisfaction_std": np.std(satisfaction_scores),
            "feedback_count": len(filtered_feedback),
            "daily_averages": daily_averages,
            "recent_satisfaction": np.mean(satisfaction_scores[-10:]) if len(satisfaction_scores) >= 10 else np.mean(satisfaction_scores),
            "satisfaction_distribution": {
                "excellent": len([s for s in satisfaction_scores if s >= 4.5]),
                "good": len([s for s in satisfaction_scores if 3.5 <= s < 4.5]),
                "average": len([s for s in satisfaction_scores if 2.5 <= s < 3.5]),
                "poor": len([s for s in satisfaction_scores if s < 2.5])
            }
        }
    
    async def _analyze_sentiment(self, feedback_text: str) -> Dict[str, Any]:
        """Analyze sentiment of feedback text"""
        
        if not feedback_text:
            return {"score": 0.0, "category": "neutral", "confidence": 0.1}
        
        text_lower = feedback_text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        
        positive_count = sum(1 for word in words if word in self.positive_keywords)
        negative_count = sum(1 for word in words if word in self.negative_keywords)
        
        # Calculate sentiment score (-1 to 1)
        total_sentiment_words = positive_count + negative_count
        if total_sentiment_words == 0:
            sentiment_score = 0.0
            confidence = 0.1
        else:
            sentiment_score = (positive_count - negative_count) / total_sentiment_words
            confidence = min(1.0, total_sentiment_words / 5.0)  # Higher confidence with more sentiment words
        
        # Categorize sentiment
        if sentiment_score > 0.3:
            category = "positive"
        elif sentiment_score < -0.3:
            category = "negative"
        else:
            category = "neutral"
        
        return {
            "score": sentiment_score,
            "category": category,
            "confidence": confidence,
            "positive_words": positive_count,
            "negative_words": negative_count
        }
    
    async def _extract_specific_insights(self, 
                                       feedback: UserFeedback,
                                       context: InteractionContext) -> List[str]:
        """Extract specific insights from feedback"""
        
        insights = []
        feedback_text = feedback.feedback_text or ""
        
        # Check for specific mentions of agent capabilities
        if "fast" in feedback_text.lower() or "quick" in feedback_text.lower():
            insights.append("User appreciated response speed")
        
        if "slow" in feedback_text.lower():
            insights.append("User found response too slow")
        
        if "clear" in feedback_text.lower() or "understand" in feedback_text.lower():
            insights.append("User found explanation clear and understandable")
        
        if "confusing" in feedback_text.lower() or "unclear" in feedback_text.lower():
            insights.append("User found explanation confusing or unclear")
        
        if "helpful" in feedback_text.lower() or "useful" in feedback_text.lower():
            insights.append("User found assistance helpful and valuable")
        
        if "complete" in feedback_text.lower() or "thorough" in feedback_text.lower():
            insights.append("User appreciated comprehensive assistance")
        
        if "incomplete" in feedback_text.lower() or "missing" in feedback_text.lower():
            insights.append("User felt assistance was incomplete")
        
        # Context-specific insights
        if hasattr(context, 'task_type'):
            task_type = context.task_type
            if task_type in feedback_text.lower():
                insights.append(f"Feedback specifically mentions {task_type} task performance")
        
        return insights
    
    async def _identify_immediate_improvements(self,
                                             feedback: UserFeedback,
                                             context: InteractionContext) -> List[str]:
        """Identify immediate improvements based on feedback"""
        
        improvements = []
        feedback_text = feedback.feedback_text or ""
        
        # Rating-based improvements
        if feedback.satisfaction_rating < 3.0:
            improvements.append("Address fundamental satisfaction issues")
        elif feedback.satisfaction_rating < 4.0:
            improvements.append("Focus on incremental quality improvements")
        
        # Text-based improvements
        if any(word in feedback_text.lower() for word in ["explain", "clarify", "understand"]):
            improvements.append("Improve explanation clarity and detail")
        
        if any(word in feedback_text.lower() for word in ["faster", "quicker", "speed"]):
            improvements.append("Optimize response time and efficiency")
        
        if any(word in feedback_text.lower() for word in ["example", "demo", "show"]):
            improvements.append("Provide more concrete examples and demonstrations")
        
        if any(word in feedback_text.lower() for word in ["step", "guide", "how"]):
            improvements.append("Offer more step-by-step guidance")
        
        return improvements
    
    async def _update_user_preferences(self,
                                     feedback: UserFeedback,
                                     context: InteractionContext) -> Dict[str, Any]:
        """Update user preferences based on feedback"""
        
        user_id = getattr(feedback, 'user_id', 'default')
        updates = {}
        
        if user_id not in self.user_preferences:
            self.user_preferences[user_id] = {
                'communication_style': 'balanced',
                'detail_level': 'medium',
                'response_speed_preference': 'balanced',
                'example_preference': 'some',
                'feedback_history': []
            }
        
        prefs = self.user_preferences[user_id]
        
        # Store feedback in history
        prefs['feedback_history'].append({
            'timestamp': feedback.timestamp.isoformat(),
            'rating': feedback.satisfaction_rating,
            'text': feedback.feedback_text
        })
        
        # Update preferences based on feedback patterns
        feedback_text = feedback.feedback_text or ""
        
        if "too detailed" in feedback_text.lower():
            prefs['detail_level'] = 'low'
            updates['detail_level'] = 'reduced to low'
        elif "more detail" in feedback_text.lower():
            prefs['detail_level'] = 'high'
            updates['detail_level'] = 'increased to high'
        
        if "too technical" in feedback_text.lower():
            prefs['communication_style'] = 'simple'
            updates['communication_style'] = 'simplified'
        elif "more technical" in feedback_text.lower():
            prefs['communication_style'] = 'technical'
            updates['communication_style'] = 'increased technical depth'
        
        if "example" in feedback_text.lower():
            prefs['example_preference'] = 'many'
            updates['example_preference'] = 'increased examples'
        
        return updates
    
    async def _generate_immediate_recommendations(self,
                                                feedback: UserFeedback,
                                                sentiment_analysis: Dict,
                                                insights: List[str]) -> List[str]:
        """Generate actionable recommendations from feedback"""
        
        recommendations = []
        
        # Sentiment-based recommendations
        if sentiment_analysis['category'] == 'negative':
            recommendations.append("Priority: Address negative feedback immediately")
            recommendations.append("Review interaction approach and identify specific pain points")
        elif sentiment_analysis['category'] == 'positive':
            recommendations.append("Reinforce successful interaction patterns")
        
        # Rating-based recommendations
        if feedback.satisfaction_rating <= 2.0:
            recommendations.append("Critical: Fundamental changes needed to interaction approach")
        elif feedback.satisfaction_rating <= 3.0:
            recommendations.append("Important: Significant improvements needed")
        elif feedback.satisfaction_rating >= 4.5:
            recommendations.append("Maintain current high-quality approach")
        
        # Insight-based recommendations
        for insight in insights:
            if "slow" in insight:
                recommendations.append("Optimize processing speed and response time")
            elif "confusing" in insight:
                recommendations.append("Simplify explanations and improve clarity")
            elif "helpful" in insight:
                recommendations.append("Continue current helpful approach")
        
        return recommendations
    
    async def _apply_immediate_adaptations(self,
                                         user_id: str,
                                         insights: FeedbackInsights,
                                         context: InteractionContext) -> None:
        """Apply immediate adaptations based on feedback"""
        
        # This would integrate with the agent system to immediately adjust behavior
        # For now, we log the adaptations
        
        adaptations = []
        
        if insights.sentiment_category == 'negative':
            adaptations.append("Switch to more careful, detailed communication")
        
        if any("clarity" in rec for rec in insights.recommendations):
            adaptations.append("Increase explanation detail and use more examples")
        
        if any("speed" in rec for rec in insights.recommendations):
            adaptations.append("Prioritize faster response generation")
        
        if adaptations:
            logger.info(f"Applied immediate adaptations for user {user_id}: {adaptations}")
    
    async def _analyze_feedback_patterns(self, feedback_history: List[UserFeedback]) -> List[FeedbackPattern]:
        """Analyze patterns across feedback history"""
        
        patterns = []
        
        # Sentiment patterns
        sentiments = []
        for feedback in feedback_history:
            sentiment = await self._analyze_sentiment(feedback.feedback_text or "")
            sentiments.append(sentiment)
        
        sentiment_distribution = Counter([s['category'] for s in sentiments])
        if sentiment_distribution['negative'] / len(sentiments) > 0.3:
            patterns.append(FeedbackPattern(
                pattern_type="high_negative_sentiment",
                pattern_description="High frequency of negative feedback",
                frequency=sentiment_distribution['negative'],
                sentiment="negative",
                confidence=0.8,
                examples=[fb.feedback_text for fb in feedback_history if fb.feedback_text][:3],
                recommendations=["Focus on improving user satisfaction", "Review interaction approaches"]
            ))
        
        # Rating patterns
        low_ratings = [fb for fb in feedback_history if fb.satisfaction_rating < 3.0]
        if len(low_ratings) / len(feedback_history) > 0.25:
            patterns.append(FeedbackPattern(
                pattern_type="frequent_low_ratings",
                pattern_description="Frequent low satisfaction ratings",
                frequency=len(low_ratings),
                sentiment="negative",
                confidence=0.9,
                examples=[str(fb.satisfaction_rating) for fb in low_ratings[:3]],
                recommendations=["Investigate root causes of dissatisfaction", "Implement quality improvements"]
            ))
        
        return patterns
    
    async def _identify_communication_improvements(self, patterns: List[FeedbackPattern]) -> List[str]:
        """Identify communication-related improvements"""
        
        improvements = []
        
        for pattern in patterns:
            if "clarity" in pattern.pattern_description.lower():
                improvements.append("Improve explanation clarity and simplicity")
            if "technical" in pattern.pattern_description.lower():
                improvements.append("Better adapt technical depth to user expertise")
            if "response" in pattern.pattern_description.lower():
                improvements.append("Enhance response relevance and directness")
        
        return improvements
    
    async def _identify_technical_improvements(self, patterns: List[FeedbackPattern]) -> List[str]:
        """Identify technical improvements"""
        
        improvements = []
        
        for pattern in patterns:
            if "accuracy" in pattern.pattern_description.lower():
                improvements.append("Improve response accuracy and precision")
            if "speed" in pattern.pattern_description.lower():
                improvements.append("Optimize processing speed and response time")
            if "reliability" in pattern.pattern_description.lower():
                improvements.append("Enhance system reliability and consistency")
        
        return improvements
    
    async def _identify_process_improvements(self, patterns: List[FeedbackPattern]) -> List[str]:
        """Identify process-related improvements"""
        
        improvements = []
        
        for pattern in patterns:
            if "workflow" in pattern.pattern_description.lower():
                improvements.append("Streamline interaction workflow")
            if "guidance" in pattern.pattern_description.lower():
                improvements.append("Provide better step-by-step guidance")
        
        return improvements
    
    async def _identify_ux_improvements(self, patterns: List[FeedbackPattern]) -> List[str]:
        """Identify user experience improvements"""
        
        improvements = []
        
        for pattern in patterns:
            if "frustration" in pattern.pattern_description.lower():
                improvements.append("Reduce user frustration points")
            if "satisfaction" in pattern.pattern_description.lower():
                improvements.append("Enhance overall user satisfaction")
        
        return improvements
    
    async def _calculate_improvement_priority(self, 
                                            improvements: List[str],
                                            patterns: List[FeedbackPattern]) -> List[str]:
        """Calculate priority ranking for improvements"""
        
        # Simple priority based on pattern confidence and frequency
        improvement_scores = {}
        
        for improvement in improvements:
            score = 0.0
            for pattern in patterns:
                if any(keyword in improvement.lower() for keyword in pattern.pattern_description.lower().split()):
                    score += pattern.confidence * pattern.frequency
            improvement_scores[improvement] = score
        
        # Sort by score
        sorted_improvements = sorted(improvement_scores.items(), key=lambda x: x[1], reverse=True)
        return [improvement for improvement, _ in sorted_improvements]
    
    async def _estimate_improvement_impact(self, 
                                         improvements: List[str],
                                         feedback_history: List[UserFeedback]) -> Dict[str, float]:
        """Estimate potential impact of improvements"""
        
        impact_estimates = {}
        
        current_avg_rating = np.mean([fb.satisfaction_rating for fb in feedback_history])
        
        for improvement in improvements:
            # Simple heuristic for impact estimation
            if "clarity" in improvement.lower():
                impact_estimates[improvement] = 0.5  # Medium impact
            elif "speed" in improvement.lower():
                impact_estimates[improvement] = 0.3  # Lower impact
            elif "satisfaction" in improvement.lower():
                impact_estimates[improvement] = 0.8  # High impact
            else:
                impact_estimates[improvement] = 0.4  # Default medium impact
        
        return impact_estimates
    
    def _parse_time_window(self, time_window: str) -> datetime:
        """Parse time window string"""
        now = datetime.now()
        
        if time_window == "24h":
            return now - timedelta(hours=24)
        elif time_window == "7d":
            return now - timedelta(days=7)
        elif time_window == "30d":
            return now - timedelta(days=30)
        else:
            return now - timedelta(days=7)  # Default
    
    async def _analyze_outcome_feedback(self, rating: float, feedback_text: str) -> Dict[str, Any]:
        """Analyze outcome-specific feedback"""
        
        analysis = {
            "outcome_category": "success" if rating >= 4.0 else "failure" if rating <= 2.0 else "partial",
            "satisfaction_level": "high" if rating >= 4.0 else "low" if rating <= 2.0 else "medium",
            "text_analysis": await self._analyze_sentiment(feedback_text)
        }
        
        return analysis
    
    async def _extract_success_factors(self, rating: float, feedback_text: str) -> List[str]:
        """Extract factors that contributed to success"""
        
        if rating < 3.0:
            return []
        
        success_factors = []
        text_lower = feedback_text.lower()
        
        success_indicators = {
            "helpful": "Provided helpful assistance",
            "clear": "Clear and understandable communication",
            "fast": "Quick response time",
            "accurate": "Accurate information provided",
            "thorough": "Comprehensive coverage of topic",
            "useful": "Practical and useful guidance"
        }
        
        for indicator, factor in success_indicators.items():
            if indicator in text_lower:
                success_factors.append(factor)
        
        return success_factors
    
    async def _extract_failure_points(self, rating: float, feedback_text: str) -> List[str]:
        """Extract points that led to failure or dissatisfaction"""
        
        if rating > 3.0:
            return []
        
        failure_points = []
        text_lower = feedback_text.lower()
        
        failure_indicators = {
            "slow": "Response time too slow",
            "confusing": "Communication was confusing",
            "wrong": "Incorrect information provided",
            "incomplete": "Incomplete assistance",
            "unhelpful": "Did not provide useful help",
            "difficult": "Made task more difficult"
        }
        
        for indicator, point in failure_indicators.items():
            if indicator in text_lower:
                failure_points.append(point)
        
        return failure_points
    
    async def _generate_improvement_strategies(self,
                                             analysis: Dict,
                                             success_factors: List[str],
                                             failure_points: List[str]) -> List[str]:
        """Generate specific improvement strategies"""
        
        strategies = []
        
        # Strategies based on failure points
        for failure_point in failure_points:
            if "slow" in failure_point:
                strategies.append("Implement response time optimization")
            elif "confusing" in failure_point:
                strategies.append("Improve communication clarity and structure")
            elif "incorrect" in failure_point:
                strategies.append("Enhance accuracy validation and fact-checking")
            elif "incomplete" in failure_point:
                strategies.append("Ensure comprehensive coverage of user requests")
        
        # Strategies to reinforce success factors
        for success_factor in success_factors:
            if "helpful" in success_factor:
                strategies.append("Maintain and enhance helpful assistance approach")
            elif "clear" in success_factor:
                strategies.append("Continue clear communication patterns")
        
        return strategies
    
    async def _update_outcome_tracking(self, task_id: str, rating: float, analysis: Dict) -> None:
        """Update outcome tracking for pattern analysis"""
        
        self.improvement_tracking[task_id].append({
            'timestamp': datetime.now().isoformat(),
            'rating': rating,
            'category': analysis['outcome_category'],
            'satisfaction': analysis['satisfaction_level']
        })
