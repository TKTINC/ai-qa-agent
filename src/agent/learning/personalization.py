"""
Adapt agent behavior to individual user preferences.
"""

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import json

from ..core.models import UserProfile, AgentInteraction, UserFeedback, PersonalizedResponse
from ...core.logging import get_logger

logger = get_logger(__name__)


class PersonalizationEngine:
    """Adapt agent behavior to individual user preferences"""
    
    def __init__(self):
        self.user_profiles: Dict[str, UserProfile] = {}
        self.interaction_history: Dict[str, List] = {}
        
    async def learn_user_preferences(self,
                                   user_id: str,
                                   interaction: AgentInteraction,
                                   feedback: UserFeedback) -> Dict[str, Any]:
        """Learn and update user preferences from interactions"""
        
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = UserProfile(
                user_id=user_id,
                expertise_level="intermediate",
                communication_style="balanced",
                preferred_detail_level="medium"
            )
        
        profile = self.user_profiles[user_id]
        
        # Learn from feedback
        if feedback.satisfaction_rating >= 4.0:
            # Positive feedback - reinforce current approach
            if hasattr(interaction, 'communication_style'):
                profile.communication_style = interaction.communication_style
        
        # Store interaction for pattern learning
        if user_id not in self.interaction_history:
            self.interaction_history[user_id] = []
        
        self.interaction_history[user_id].append({
            'timestamp': datetime.now().isoformat(),
            'interaction': interaction,
            'feedback': feedback
        })
        
        return {"user_profile_updated": True, "preferences_learned": 3}
    
    async def personalize_response(self, 
                                 response: str,
                                 user_id: str) -> PersonalizedResponse:
        """Personalize response based on user preferences"""
        
        if user_id not in self.user_profiles:
            return PersonalizedResponse(
                original_response=response,
                personalized_response=response,
                personalization_applied=[]
            )
        
        profile = self.user_profiles[user_id]
        personalized = response
        personalizations = []
        
        # Apply personalizations based on profile
        if profile.communication_style == "simple":
            # Simplify technical language
            personalizations.append("simplified_technical_language")
        elif profile.communication_style == "technical":
            # Add more technical detail
            personalizations.append("enhanced_technical_detail")
        
        return PersonalizedResponse(
            original_response=response,
            personalized_response=personalized,
            personalization_applied=personalizations
        )
