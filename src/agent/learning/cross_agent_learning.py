"""
Enable agents to learn from each other's experiences.
"""

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime

from ...core.logging import get_logger

logger = get_logger(__name__)


class CrossAgentLearning:
    """Enable agents to learn from each other's experiences"""
    
    def __init__(self):
        self.shared_knowledge: Dict[str, Any] = {}
        self.agent_specializations: Dict[str, List[str]] = {}
        
    async def share_learning_insights(self,
                                    source_agent: str,
                                    learning_patterns: List[Dict],
                                    success_indicators: Dict) -> Dict[str, Any]:
        """Share successful approaches between agents"""
        
        sharing_result = {
            "patterns_shared": len(learning_patterns),
            "target_agents": [],
            "knowledge_transferred": []
        }
        
        # Share successful patterns
        for pattern in learning_patterns:
            if pattern.get('confidence', 0) > 0.7:
                pattern_key = f"{pattern['type']}_{pattern.get('task_type', 'general')}"
                
                if pattern_key not in self.shared_knowledge:
                    self.shared_knowledge[pattern_key] = []
                
                self.shared_knowledge[pattern_key].append({
                    'source_agent': source_agent,
                    'pattern': pattern,
                    'timestamp': datetime.now().isoformat(),
                    'success_indicators': success_indicators
                })
                
                sharing_result["knowledge_transferred"].append(pattern_key)
        
        logger.info(f"Shared {len(learning_patterns)} patterns from {source_agent}")
        return sharing_result
