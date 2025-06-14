"""
AI QA Agent - Multi-Agent System
Sprint 2.3: Multi-Agent Architecture & Collaboration

This module implements the multi-agent system including:
- Multi-agent coordination and communication
- Specialist agent implementations
- Collaborative problem-solving patterns
- Cross-agent knowledge sharing
- Agent teamwork and consensus building
"""

from .agent_system import QAAgentSystem
from .collaboration_manager import CollaborationManager

__all__ = [
    'QAAgentSystem',
    'CollaborationManager'
]
