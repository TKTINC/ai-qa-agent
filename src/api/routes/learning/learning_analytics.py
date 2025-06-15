"""
Learning Analytics API endpoints.
"""

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
import json

from ....agent.learning.learning_engine import AgentLearningEngine
from ....agent.learning.experience_tracker import ExperienceTracker
from ....agent.learning.feedback_processor import FeedbackProcessor
from ....agent.learning.quality_assessment import LearningQualityAssessment
from ....core.logging import get_logger
from ....streaming.learning_stream import LearningStreamManager

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/learning", tags=["Learning Analytics"])

# Global instances (would be dependency injected in production)
learning_engine = AgentLearningEngine()
experience_tracker = ExperienceTracker()
feedback_processor = FeedbackProcessor()
quality_assessment = LearningQualityAssessment()
stream_manager = LearningStreamManager()


class AgentPerformanceResponse(BaseModel):
    """Agent performance metrics response"""
    agent_name: str
    time_period: str
    performance_score: float
    improvement_rate: float
    user_satisfaction_trend: List[float]
    capability_improvements: List[Dict[str, Any]]
    successful_patterns: List[Dict[str, Any]]
    areas_for_improvement: List[str]
    learning_velocity: float
    experience_count: int


class PersonalizationProfileResponse(BaseModel):
    """User personalization profile response"""
    user_id: str
    learned_preferences: Dict[str, Any]
    communication_style: str
    expertise_level: str
    preferred_tools: List[str]
    successful_interaction_patterns: List[Dict[str, Any]]
    personalization_accuracy: float
    interaction_count: int


class LearningFeedbackRequest(BaseModel):
    """Learning feedback submission request"""
    session_id: str
    agent_name: str
    feedback_type: str = Field(..., description="Type of feedback: 'interaction', 'outcome', 'preference'")
    satisfaction_rating: float = Field(..., ge=1.0, le=5.0)
    feedback_text: Optional[str] = None
    specific_improvements: List[str] = []
    context: Dict[str, Any] = {}


class IntelligenceMetricsResponse(BaseModel):
    """Agent intelligence metrics response"""
    overall_intelligence_score: float
    reasoning_quality: float
    learning_velocity: float
    adaptation_speed: float
    collaboration_effectiveness: float
    user_satisfaction: float
    capability_growth_rate: float
    knowledge_retention: float
    trend_direction: str
    confidence_level: float


@router.get("/agents/{agent_name}/performance", response_model=AgentPerformanceResponse)
async def get_agent_performance(
    agent_name: str,
    time_period: str = Query("7d", description="Time period: 1h, 24h, 7d, 30d"),
    include_trends: bool = Query(True, description="Include performance trends")
) -> AgentPerformanceResponse:
    """Track agent performance trends and improvement over time"""
    
    try:
        # Get experience analysis for the agent
        experience_analysis = await experience_tracker.analyze_experience_patterns(
            agent=agent_name,
            time_period=time_period
        )
        
        # Get agent capabilities from learning engine
        agent_capabilities = learning_engine.agent_capabilities.get(agent_name, {})
        
        # Calculate performance metrics
        performance_score = agent_capabilities.get('problem_solving_score', 0.7)
        learning_velocity = agent_capabilities.get('learning_velocity', 0.5)
        
        # Get user satisfaction trends
        satisfaction_trends = []
        if include_trends:
            satisfaction_data = await feedback_processor.get_user_satisfaction_trends(
                time_window=time_period
            )
            satisfaction_trends = list(satisfaction_data.get('daily_averages', {}).values())
        
        # Calculate improvement rate
        improvement_rate = 0.0
        if len(experience_analysis.success_patterns) > 0:
            recent_successes = len([p for p in experience_analysis.success_patterns if p.get('confidence', 0) > 0.7])
            improvement_rate = min(1.0, recent_successes / 10.0)
        
        response = AgentPerformanceResponse(
            agent_name=agent_name,
            time_period=time_period,
            performance_score=performance_score,
            improvement_rate=improvement_rate,
            user_satisfaction_trend=satisfaction_trends,
            capability_improvements=experience_analysis.optimization_opportunities,
            successful_patterns=experience_analysis.success_patterns,
            areas_for_improvement=experience_analysis.recommendations,
            learning_velocity=learning_velocity,
            experience_count=experience_analysis.total_experiences
        )
        
        logger.info(f"Retrieved performance data for {agent_name}: score={performance_score:.3f}")
        return response
        
    except Exception as e:
        logger.error(f"Error retrieving agent performance: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving performance data: {str(e)}")


@router.get("/user/{user_id}/personalization", response_model=PersonalizationProfileResponse)
async def get_user_personalization(
    user_id: str,
    include_patterns: bool = Query(True, description="Include interaction patterns")
) -> PersonalizationProfileResponse:
    """Show how agents have adapted to specific user preferences"""
    
    try:
        # Get user preferences from feedback processor
        user_preferences = feedback_processor.user_preferences.get(user_id, {})
        
        # Get learned preferences from personalization engine
        if hasattr(learning_engine, 'personalization_engine'):
            personalization_data = await learning_engine.personalization_engine.get_user_profile(user_id)
        else:
            personalization_data = {}
        
        # Calculate personalization accuracy
        interaction_history = feedback_processor.user_preferences.get(user_id, {}).get('feedback_history', [])
        accuracy = 0.9  # Default high accuracy
        if len(interaction_history) > 5:
            recent_ratings = [fb.get('rating', 3.5) for fb in interaction_history[-10:]]
            accuracy = min(1.0, sum(1 for r in recent_ratings if r >= 4.0) / len(recent_ratings))
        
        # Extract successful patterns
        successful_patterns = []
        if include_patterns and interaction_history:
            for feedback in interaction_history[-5:]:  # Last 5 interactions
                if feedback.get('rating', 0) >= 4.0:
                    successful_patterns.append({
                        'timestamp': feedback.get('timestamp'),
                        'rating': feedback.get('rating'),
                        'pattern_type': 'high_satisfaction',
                        'confidence': 0.8
                    })
        
        response = PersonalizationProfileResponse(
            user_id=user_id,
            learned_preferences=user_preferences,
            communication_style=user_preferences.get('communication_style', 'balanced'),
            expertise_level=user_preferences.get('expertise_level', 'intermediate'),
            preferred_tools=user_preferences.get('preferred_tools', []),
            successful_interaction_patterns=successful_patterns,
            personalization_accuracy=accuracy,
            interaction_count=len(interaction_history)
        )
        
        logger.info(f"Retrieved personalization data for {user_id}: accuracy={accuracy:.3f}")
        return response
        
    except Exception as e:
        logger.error(f"Error retrieving user personalization: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving personalization data: {str(e)}")


@router.post("/feedback")
async def submit_learning_feedback(
    feedback_request: LearningFeedbackRequest,
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """Allow users to provide feedback that improves agent learning"""
    
    try:
        # Create feedback object
        from ....core.models import UserFeedback
        from ....agent.core.models import InteractionContext
        
        feedback = UserFeedback(
            user_id=feedback_request.session_id,  # Using session_id as user_id for now
            satisfaction_rating=feedback_request.satisfaction_rating,
            feedback_text=feedback_request.feedback_text,
            timestamp=datetime.now()
        )
        
        context = InteractionContext(
            session_id=feedback_request.session_id,
            task_type=feedback_request.feedback_type,
            agent_name=feedback_request.agent_name
        )
        
        # Process feedback immediately
        feedback_insights = await feedback_processor.process_immediate_feedback(
            feedback=feedback,
            interaction_context=context
        )
        
        # Schedule background learning update
        background_tasks.add_task(
            _apply_learning_from_feedback,
            feedback_request.agent_name,
            feedback_insights,
            feedback_request.context
        )
        
        # Prepare response
        response = {
            "feedback_processed": True,
            "feedback_id": feedback_insights.feedback_id,
            "sentiment_detected": feedback_insights.sentiment_category,
            "immediate_improvements": feedback_insights.improvement_suggestions,
            "personalization_updates": len(feedback_insights.preference_updates),
            "learning_insights": feedback_insights.specific_insights,
            "recommendations": feedback_insights.recommendations
        }
        
        logger.info(f"Processed feedback for {feedback_request.agent_name}: "
                   f"sentiment={feedback_insights.sentiment_category}, "
                   f"improvements={len(feedback_insights.improvement_suggestions)}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing learning feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing feedback: {str(e)}")


@router.get("/analytics/agent-intelligence", response_model=IntelligenceMetricsResponse)
async def get_agent_intelligence_metrics(
    time_window: str = Query("7d", description="Analysis time window"),
    agent_filter: Optional[str] = Query(None, description="Filter by specific agent")
) -> IntelligenceMetricsResponse:
    """Overall system intelligence metrics and trends"""
    
    try:
        # Get learning insights from learning engine
        learning_insights = await learning_engine.get_learning_insights(time_window)
        
        # Calculate intelligence metrics
        overall_score = learning_insights.get('average_learning_score', 0.7)
        learning_velocity = learning_insights.get('learning_velocity', 0.5)
        
        # Get quality trends
        quality_trend = learning_insights.get('quality_trend', 'stable')
        trend_direction = quality_trend
        
        # Calculate specific intelligence dimensions
        reasoning_quality = overall_score * 0.9  # Slightly lower than overall
        adaptation_speed = min(1.0, learning_velocity * 2.0)  # Scale learning velocity
        
        # Get collaboration metrics
        top_agents = learning_insights.get('top_learning_agents', [])
        collaboration_effectiveness = 0.8  # Default good collaboration
        if len(top_agents) > 1:
            # Calculate based on agent diversity and performance
            agent_scores = [agent.get('average_score', 0.7) for agent in top_agents]
            collaboration_effectiveness = min(1.0, sum(agent_scores) / len(agent_scores))
        
        # Get user satisfaction
        satisfaction_data = await feedback_processor.get_user_satisfaction_trends(time_window=time_window)
        user_satisfaction = satisfaction_data.get('overall_satisfaction', 3.5) / 5.0  # Normalize to 0-1
        
        # Calculate confidence based on data volume
        total_interactions = learning_insights.get('total_interactions', 0)
        confidence_level = min(1.0, total_interactions / 100.0)  # High confidence with 100+ interactions
        
        response = IntelligenceMetricsResponse(
            overall_intelligence_score=overall_score,
            reasoning_quality=reasoning_quality,
            learning_velocity=learning_velocity,
            adaptation_speed=adaptation_speed,
            collaboration_effectiveness=collaboration_effectiveness,
            user_satisfaction=user_satisfaction,
            capability_growth_rate=learning_insights.get('learning_velocity', 0.15),
            knowledge_retention=0.95,  # High retention rate
            trend_direction=trend_direction,
            confidence_level=confidence_level
        )
        
        logger.info(f"Generated intelligence metrics: overall={overall_score:.3f}, "
                   f"trend={trend_direction}, confidence={confidence_level:.3f}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error generating intelligence metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating metrics: {str(e)}")


@router.get("/analytics/learning-insights/{session_id}")
async def get_session_learning_insights(
    session_id: str,
    include_recommendations: bool = Query(True, description="Include improvement recommendations")
) -> Dict[str, Any]:
    """What the agents learned from a specific session"""
    
    try:
        # Get session-specific learning data
        session_learning = []
        for entry in learning_engine.learning_history:
            if entry.get('interaction_id', '').startswith(session_id):
                session_learning.append(entry)
        
        if not session_learning:
            return {
                "session_id": session_id,
                "learning_events": 0,
                "message": "No learning data found for this session"
            }
        
        # Calculate session insights
        total_patterns = sum(entry.get('patterns_learned', 0) for entry in session_learning)
        avg_quality = sum(entry.get('learning_score', 0) for entry in session_learning) / len(session_learning)
        capabilities_updated = sum(entry.get('capabilities_updated', 0) for entry in session_learning)
        
        # Generate recommendations if requested
        recommendations = []
        if include_recommendations and avg_quality < 0.8:
            recommendations = [
                "Consider providing more specific feedback to improve learning quality",
                "Try engaging with different types of tasks to diversify learning",
                "Provide examples of preferred communication styles"
            ]
        
        insights = {
            "session_id": session_id,
            "learning_events": len(session_learning),
            "total_patterns_learned": total_patterns,
            "average_learning_quality": avg_quality,
            "capabilities_updated": capabilities_updated,
            "learning_timeline": [
                {
                    "timestamp": entry.get('timestamp'),
                    "agent": entry.get('agent'),
                    "patterns_learned": entry.get('patterns_learned', 0),
                    "learning_score": entry.get('learning_score', 0)
                }
                for entry in session_learning
            ],
            "session_summary": f"Learned {total_patterns} patterns across {len(session_learning)} interactions",
            "recommendations": recommendations if include_recommendations else []
        }
        
        logger.info(f"Generated session insights for {session_id}: "
                   f"events={len(session_learning)}, quality={avg_quality:.3f}")
        
        return insights
        
    except Exception as e:
        logger.error(f"Error generating session insights: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating insights: {str(e)}")


@router.get("/analytics/improvement-opportunities")
async def get_improvement_opportunities() -> Dict[str, Any]:
    """Areas where agents could improve based on data analysis"""
    
    try:
        # Get improvement opportunities from learning insights
        learning_insights = await learning_engine.get_learning_insights("30d")  # Last 30 days
        improvement_opps = learning_insights.get('improvement_opportunities', [])
        
        # Get feedback-based improvements
        feedback_history = feedback_processor.feedback_history[-100:]  # Last 100 feedback items
        feedback_improvements = await feedback_processor.identify_improvement_opportunities(feedback_history)
        
        # Combine and prioritize opportunities
        all_opportunities = {
            "system_level": improvement_opps,
            "communication": feedback_improvements.communication_improvements,
            "technical": feedback_improvements.technical_improvements,
            "process": feedback_improvements.process_improvements,
            "user_experience": feedback_improvements.user_experience_improvements,
            "priority_ranking": feedback_improvements.priority_ranking,
            "estimated_impact": feedback_improvements.estimated_impact
        }
        
        # Add analysis metadata
        analysis_info = {
            "analysis_period": "30 days",
            "data_sources": ["learning_insights", "user_feedback", "performance_metrics"],
            "total_opportunities": len(improvement_opps) + len(feedback_improvements.priority_ranking),
            "confidence_level": min(1.0, len(feedback_history) / 50.0),  # Higher confidence with more feedback
            "last_updated": datetime.now().isoformat()
        }
        
        response = {
            "improvement_opportunities": all_opportunities,
            "analysis_info": analysis_info,
            "actionable_recommendations": [
                "Focus on top 3 priority improvements for maximum impact",
                "Implement user experience improvements first for immediate satisfaction gains",
                "Address technical improvements to enhance system reliability",
                "Continuously monitor communication effectiveness"
            ]
        }
        
        logger.info(f"Generated improvement opportunities: "
                   f"total={analysis_info['total_opportunities']}, "
                   f"confidence={analysis_info['confidence_level']:.3f}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error generating improvement opportunities: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating opportunities: {str(e)}")


async def _apply_learning_from_feedback(agent_name: str, feedback_insights: Any, context: Dict[str, Any]) -> None:
    """Background task to apply learning from feedback"""
    
    try:
        # Update agent capabilities based on feedback
        if hasattr(feedback_insights, 'improvement_suggestions') and feedback_insights.improvement_suggestions:
            improvement_areas = [
                suggestion.split(' ')[1] if ' ' in suggestion else 'general'
                for suggestion in feedback_insights.improvement_suggestions[:3]
            ]
            
            await learning_engine.improve_agent_capabilities(agent_name, improvement_areas)
        
        logger.info(f"Applied background learning updates for {agent_name}")
        
    except Exception as e:
        logger.error(f"Error applying background learning: {str(e)}")
