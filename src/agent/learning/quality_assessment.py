"""
Assess and validate learning system effectiveness.
"""

import asyncio
from typing import Dict, List, Optional, Any
import numpy as np

from ...core.logging import get_logger

logger = get_logger(__name__)


class LearningQualityAssessment:
    """Assess and validate learning system effectiveness"""
    
    def __init__(self):
        self.quality_metrics: Dict[str, float] = {}
        
    async def assess_learning_quality(self,
                                    learning_patterns: List[Dict],
                                    feedback_insights: Optional[Dict],
                                    capability_updates: List[Dict]) -> 'LearningQuality':
        """Assess the quality of learning from an interaction"""
        
        # Calculate pattern quality score
        pattern_score = self._assess_pattern_quality(learning_patterns)
        
        # Calculate feedback integration score
        feedback_score = self._assess_feedback_integration(feedback_insights)
        
        # Calculate capability improvement score
        capability_score = self._assess_capability_improvements(capability_updates)
        
        # Overall learning quality
        overall_score = (pattern_score * 0.4 + feedback_score * 0.3 + capability_score * 0.3)
        
        return type('LearningQuality', (), {
            'overall_score': overall_score,
            'pattern_quality': pattern_score,
            'feedback_integration': feedback_score,
            'capability_improvement': capability_score
        })()
    
    def _assess_pattern_quality(self, patterns: List[Dict]) -> float:
        """Assess quality of extracted learning patterns"""
        if not patterns:
            return 0.1
        
        confidences = [p.get('confidence', 0.5) for p in patterns]
        return np.mean(confidences)
    
    def _assess_feedback_integration(self, feedback_insights: Optional[Dict]) -> float:
        """Assess how well feedback was integrated"""
        if not feedback_insights:
            return 0.5
        
        return feedback_insights.get('confidence', 0.5)
    
    def _assess_capability_improvements(self, updates: List[Dict]) -> float:
        """Assess quality of capability improvements"""
        if not updates:
            return 0.3
        
        return min(1.0, len(updates) / 5.0)
    
    async def validate_capability_improvements(self,
                                             agent_name: str,
                                             before_capabilities: Dict,
                                             after_capabilities: Dict) -> Dict[str, Any]:
        """Validate that capability improvements are beneficial"""
        
        validation = {
            "improvements_validated": True,
            "positive_changes": [],
            "concerning_changes": [],
            "overall_improvement": 0.0
        }
        
        total_improvement = 0.0
        change_count = 0
        
        for capability, after_value in after_capabilities.items():
            before_value = before_capabilities.get(capability, 0.5)
            change = after_value - before_value
            
            if abs(change) > 0.01:  # Significant change
                change_count += 1
                total_improvement += change
                
                if change > 0:
                    validation["positive_changes"].append({
                        "capability": capability,
                        "improvement": change
                    })
                else:
                    validation["concerning_changes"].append({
                        "capability": capability,
                        "decline": abs(change)
                    })
        
        if change_count > 0:
            validation["overall_improvement"] = total_improvement / change_count
        
        return validation
