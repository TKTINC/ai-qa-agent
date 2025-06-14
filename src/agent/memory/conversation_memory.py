"""
Conversation Memory System
Manages agent state and conversation context across sessions
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

from ..core.models import AgentState, Message, Goal, ReasoningStep
from ...core.exceptions import AgentError


logger = logging.getLogger(__name__)


class ConversationMemory:
    """
    Manages conversation memory and agent state persistence.
    
    Handles session state storage, context management, and memory optimization
    for efficient agent operation.
    """

    def __init__(self):
        self.memory_store: Dict[str, AgentState] = {}
        self.max_sessions = 100
        self.session_timeout_hours = 24
        self.context_limit = 50  # Maximum messages per session
        
        # Start cleanup task
        asyncio.create_task(self._periodic_cleanup())

    async def save_session_state(self, agent_state: AgentState) -> bool:
        """
        Save agent state to memory
        
        Args:
            agent_state: Agent state to save
            
        Returns:
            True if saved successfully
        """
        try:
            # Update last_updated timestamp
            agent_state.last_updated = datetime.utcnow()
            
            # Optimize state before saving
            optimized_state = await self._optimize_state(agent_state)
            
            # Store in memory
            self.memory_store[agent_state.session_id] = optimized_state
            
            logger.debug(f"Saved state for session {agent_state.session_id}")
            return True

        except Exception as e:
            logger.error(f"Error saving session state: {str(e)}")
            raise AgentError(f"Failed to save session state: {str(e)}")

    async def get_session_state(self, session_id: str) -> Optional[AgentState]:
        """
        Retrieve agent state from memory
        
        Args:
            session_id: Session identifier
            
        Returns:
            AgentState if found, None otherwise
        """
        try:
            state = self.memory_store.get(session_id)
            
            if state:
                # Check if session is still valid
                if await self._is_session_valid(state):
                    logger.debug(f"Retrieved state for session {session_id}")
                    return state
                else:
                    # Remove expired session
                    await self._remove_session(session_id)
                    logger.info(f"Removed expired session {session_id}")
                    return None
            
            return None

        except Exception as e:
            logger.error(f"Error retrieving session state: {str(e)}")
            return None

    async def _optimize_state(self, agent_state: AgentState) -> AgentState:
        """
        Optimize agent state for memory efficiency
        """
        optimized_state = agent_state.model_copy()

        # Limit conversation context
        if len(optimized_state.conversation_context) > self.context_limit:
            # Keep most recent messages and important ones
            recent_messages = optimized_state.conversation_context[-self.context_limit//2:]
            important_messages = await self._extract_important_messages(
                optimized_state.conversation_context[:-self.context_limit//2]
            )
            
            optimized_state.conversation_context = important_messages + recent_messages

        # Limit reasoning history
        if len(optimized_state.reasoning_history) > 100:
            # Keep most recent reasoning steps
            optimized_state.reasoning_history = optimized_state.reasoning_history[-100:]

        # Clean up session memory
        session_memory = optimized_state.session_memory.copy()
        
        # Remove old learning history
        learning_history = session_memory.get("learning_history", [])
        if len(learning_history) > 50:
            session_memory["learning_history"] = learning_history[-50:]
        
        optimized_state.session_memory = session_memory

        return optimized_state

    async def _extract_important_messages(self, messages: List[Message]) -> List[Message]:
        """
        Extract important messages from conversation history
        """
        important_messages = []
        
        for message in messages:
            content_lower = message.content.lower()
            
            # Keep messages with important keywords
            important_keywords = [
                "goal", "objective", "important", "critical",
                "analyze", "generate", "help", "problem",
                "error", "issue", "requirement"
            ]
            
            if any(keyword in content_lower for keyword in important_keywords):
                important_messages.append(message)
            
            # Keep messages with metadata indicating importance
            if message.metadata.get("important", False):
                important_messages.append(message)

        return important_messages

    async def _is_session_valid(self, agent_state: AgentState) -> bool:
        """
        Check if a session is still valid (not expired)
        """
        session_age = datetime.utcnow() - agent_state.created_at
        return session_age.total_seconds() < (self.session_timeout_hours * 3600)

    async def _remove_session(self, session_id: str) -> None:
        """
        Remove a session from memory
        """
        if session_id in self.memory_store:
            del self.memory_store[session_id]
            logger.debug(f"Removed session {session_id} from memory")

    async def _periodic_cleanup(self) -> None:
        """
        Periodic cleanup of expired sessions
        """
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                await self._cleanup_expired_sessions()
            except Exception as e:
                logger.error(f"Error in periodic cleanup: {str(e)}")

    async def _cleanup_expired_sessions(self) -> None:
        """
        Clean up expired sessions from memory
        """
        current_time = datetime.utcnow()
        expired_sessions = []

        for session_id, state in self.memory_store.items():
            session_age = current_time - state.created_at
            if session_age.total_seconds() > (self.session_timeout_hours * 3600):
                expired_sessions.append(session_id)

        for session_id in expired_sessions:
            await self._remove_session(session_id)

        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")

        # Also clean up if we have too many sessions
        if len(self.memory_store) > self.max_sessions:
            await self._cleanup_oldest_sessions()

    async def _cleanup_oldest_sessions(self) -> None:
        """
        Clean up oldest sessions if we exceed the maximum
        """
        sessions_by_age = sorted(
            self.memory_store.items(),
            key=lambda x: x[1].last_updated
        )

        sessions_to_remove = len(self.memory_store) - self.max_sessions + 10
        
        for i in range(sessions_to_remove):
            session_id = sessions_by_age[i][0]
            await self._remove_session(session_id)

        logger.info(f"Cleaned up {sessions_to_remove} oldest sessions")

    async def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get memory usage statistics
        """
        total_sessions = len(self.memory_store)
        total_messages = sum(
            len(state.conversation_context) 
            for state in self.memory_store.values()
        )
        total_reasoning_steps = sum(
            len(state.reasoning_history)
            for state in self.memory_store.values()
        )

        # Calculate session ages
        current_time = datetime.utcnow()
        session_ages = [
            (current_time - state.created_at).total_seconds() / 3600
            for state in self.memory_store.values()
        ]

        avg_session_age = sum(session_ages) / len(session_ages) if session_ages else 0

        return {
            "total_sessions": total_sessions,
            "total_messages": total_messages,
            "total_reasoning_steps": total_reasoning_steps,
            "avg_session_age_hours": round(avg_session_age, 2),
            "memory_limit": self.max_sessions,
            "context_limit": self.context_limit
        }

    async def search_conversations(
        self,
        query: str,
        session_id: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search through conversation history
        """
        results = []
        query_lower = query.lower()

        sessions_to_search = (
            [self.memory_store[session_id]] if session_id and session_id in self.memory_store
            else self.memory_store.values()
        )

        for state in sessions_to_search:
            for message in state.conversation_context:
                if query_lower in message.content.lower():
                    results.append({
                        "session_id": state.session_id,
                        "message_id": message.id,
                        "role": message.role,
                        "content": message.content,
                        "timestamp": message.timestamp,
                        "relevance_score": self._calculate_relevance(query_lower, message.content.lower())
                    })

        # Sort by relevance and timestamp
        results.sort(key=lambda x: (x["relevance_score"], x["timestamp"]), reverse=True)
        
        return results[:limit]

    def _calculate_relevance(self, query: str, content: str) -> float:
        """
        Calculate relevance score for search results
        """
        # Simple relevance calculation based on keyword matching
        query_words = query.split()
        content_words = content.split()
        
        matches = sum(1 for word in query_words if word in content_words)
        return matches / len(query_words) if query_words else 0
