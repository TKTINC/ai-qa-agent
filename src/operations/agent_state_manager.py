"""
Agent State Management System for Production Deployment
Manages agent state persistence across container restarts and scaling events
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum

import redis.asyncio as redis
import aiofiles
from pydantic import BaseModel

from src.core.config import get_settings
from src.agent.core.models import AgentState, ConversationContext
from src.agent.learning.models import LearningState

logger = logging.getLogger(__name__)

class StateType(str, Enum):
    """Types of state that can be managed"""
    AGENT_STATE = "agent_state"
    CONVERSATION = "conversation"
    LEARNING = "learning"
    COLLABORATION = "collaboration"
    REASONING = "reasoning"

class StatePersistenceBackend(str, Enum):
    """Available persistence backends"""
    REDIS = "redis"
    FILE = "file"
    MEMORY = "memory"

@dataclass
class StateSnapshot:
    """Snapshot of agent state at a point in time"""
    agent_id: str
    state_type: StateType
    timestamp: datetime
    data: Dict[str, Any]
    version: str = "1.0"
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StateSnapshot':
        """Create from dictionary"""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)

class AgentStateManager:
    """
    Comprehensive agent state management for production deployment
    
    Handles:
    - State persistence across container restarts
    - Agent state distribution for scaling
    - Learning state synchronization
    - Conversation context preservation
    - Collaboration state management
    """
    
    def __init__(self, backend: StatePersistenceBackend = StatePersistenceBackend.REDIS):
        self.settings = get_settings()
        self.backend = backend
        self.redis_client: Optional[redis.Redis] = None
        self._state_cache: Dict[str, StateSnapshot] = {}
        self._cleanup_interval = 3600  # 1 hour
        
    async def initialize(self) -> None:
        """Initialize the state manager"""
        try:
            if self.backend == StatePersistenceBackend.REDIS:
                await self._initialize_redis()
            elif self.backend == StatePersistenceBackend.FILE:
                await self._initialize_file_storage()
            
            # Start background cleanup task
            asyncio.create_task(self._periodic_cleanup())
            
            logger.info(f"Agent state manager initialized with {self.backend} backend")
            
        except Exception as e:
            logger.error(f"Failed to initialize state manager: {e}")
            raise

    async def _initialize_redis(self) -> None:
        """Initialize Redis connection for state storage"""
        redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/1')
        self.redis_client = redis.from_url(
            redis_url,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
            retry_on_timeout=True
        )
        
        # Test connection
        await self.redis_client.ping()
        logger.info("Redis connection established for agent state management")

    async def _initialize_file_storage(self) -> None:
        """Initialize file-based storage"""
        self.storage_path = os.getenv('AGENT_STATE_PATH', './agent_state')
        os.makedirs(self.storage_path, exist_ok=True)
        logger.info(f"File storage initialized at {self.storage_path}")

    async def save_agent_state(self, agent_id: str, state: AgentState) -> bool:
        """
        Persist agent state for recovery and scaling
        
        Args:
            agent_id: Unique identifier for the agent
            state: Current agent state to persist
            
        Returns:
            bool: True if successfully saved
        """
        try:
            snapshot = StateSnapshot(
                agent_id=agent_id,
                state_type=StateType.AGENT_STATE,
                timestamp=datetime.utcnow(),
                data=state.dict(),
                metadata={
                    'container_id': os.getenv('HOSTNAME', 'unknown'),
                    'version': '1.0'
                }
            )
            
            await self._persist_snapshot(snapshot)
            
            # Update local cache
            self._state_cache[f"{agent_id}:agent_state"] = snapshot
            
            logger.debug(f"Agent state saved for {agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save agent state for {agent_id}: {e}")
            return False

    async def restore_agent_state(self, agent_id: str) -> Optional[AgentState]:
        """
        Restore agent state after container restart
        
        Args:
            agent_id: Unique identifier for the agent
            
        Returns:
            Optional[AgentState]: Restored state or None if not found
        """
        try:
            snapshot = await self._retrieve_snapshot(agent_id, StateType.AGENT_STATE)
            
            if snapshot and snapshot.data:
                # Validate state age (don't restore very old states)
                age = datetime.utcnow() - snapshot.timestamp
                if age > timedelta(hours=24):
                    logger.warning(f"Agent state for {agent_id} is too old ({age}), skipping restore")
                    return None
                
                state = AgentState(**snapshot.data)
                logger.info(f"Agent state restored for {agent_id}")
                return state
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to restore agent state for {agent_id}: {e}")
            return None

    async def save_conversation_context(self, session_id: str, context: ConversationContext) -> bool:
        """Save conversation context for session persistence"""
        try:
            snapshot = StateSnapshot(
                agent_id=session_id,
                state_type=StateType.CONVERSATION,
                timestamp=datetime.utcnow(),
                data=context.dict(),
                metadata={
                    'session_duration': context.get_duration_minutes() if hasattr(context, 'get_duration_minutes') else 0,
                    'message_count': len(context.messages) if hasattr(context, 'messages') else 0
                }
            )
            
            await self._persist_snapshot(snapshot)
            self._state_cache[f"{session_id}:conversation"] = snapshot
            
            logger.debug(f"Conversation context saved for session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save conversation context for {session_id}: {e}")
            return False

    async def restore_conversation_context(self, session_id: str) -> Optional[ConversationContext]:
        """Restore conversation context for session continuity"""
        try:
            snapshot = await self._retrieve_snapshot(session_id, StateType.CONVERSATION)
            
            if snapshot and snapshot.data:
                context = ConversationContext(**snapshot.data)
                logger.info(f"Conversation context restored for session {session_id}")
                return context
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to restore conversation context for {session_id}: {e}")
            return None

    async def distribute_learning_update(self, learning_update: Dict[str, Any]) -> bool:
        """
        Distribute learning updates across agent instances
        
        Args:
            learning_update: Learning data to distribute
            
        Returns:
            bool: True if successfully distributed
        """
        try:
            # Create learning snapshot
            snapshot = StateSnapshot(
                agent_id="global_learning",
                state_type=StateType.LEARNING,
                timestamp=datetime.utcnow(),
                data=learning_update,
                metadata={
                    'distribution_id': f"learning_{datetime.utcnow().timestamp()}",
                    'origin_container': os.getenv('HOSTNAME', 'unknown')
                }
            )
            
            await self._persist_snapshot(snapshot)
            
            # Publish learning update to all instances
            if self.backend == StatePersistenceBackend.REDIS:
                await self._publish_learning_update(learning_update)
            
            logger.info("Learning update distributed to all agent instances")
            return True
            
        except Exception as e:
            logger.error(f"Failed to distribute learning update: {e}")
            return False

    async def _publish_learning_update(self, learning_update: Dict[str, Any]) -> None:
        """Publish learning update via Redis pub/sub"""
        if self.redis_client:
            await self.redis_client.publish(
                'agent_learning_updates',
                json.dumps(learning_update)
            )

    async def get_active_agents(self) -> List[str]:
        """Get list of currently active agent instances"""
        try:
            if self.backend == StatePersistenceBackend.REDIS:
                pattern = "*:agent_state"
                keys = await self.redis_client.keys(pattern)
                return [key.split(':')[0] for key in keys]
            
            return list(self._state_cache.keys())
            
        except Exception as e:
            logger.error(f"Failed to get active agents: {e}")
            return []

    async def cleanup_expired_states(self, max_age_hours: int = 24) -> int:
        """
        Clean up expired state snapshots
        
        Args:
            max_age_hours: Maximum age in hours before cleanup
            
        Returns:
            int: Number of states cleaned up
        """
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
            cleaned_count = 0
            
            if self.backend == StatePersistenceBackend.REDIS:
                cleaned_count = await self._cleanup_redis_states(cutoff_time)
            elif self.backend == StatePersistenceBackend.FILE:
                cleaned_count = await self._cleanup_file_states(cutoff_time)
            else:
                # Memory backend - clean local cache
                keys_to_remove = []
                for key, snapshot in self._state_cache.items():
                    if snapshot.timestamp < cutoff_time:
                        keys_to_remove.append(key)
                
                for key in keys_to_remove:
                    del self._state_cache[key]
                    cleaned_count += 1
            
            logger.info(f"Cleaned up {cleaned_count} expired state snapshots")
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup expired states: {e}")
            return 0

    async def _cleanup_redis_states(self, cutoff_time: datetime) -> int:
        """Clean up expired Redis states"""
        if not self.redis_client:
            return 0
        
        cleaned_count = 0
        pattern = "*:*"
        keys = await self.redis_client.keys(pattern)
        
        for key in keys:
            try:
                state_data = await self.redis_client.get(key)
                if state_data:
                    snapshot_dict = json.loads(state_data)
                    timestamp = datetime.fromisoformat(snapshot_dict['timestamp'])
                    
                    if timestamp < cutoff_time:
                        await self.redis_client.delete(key)
                        cleaned_count += 1
            except Exception:
                # Skip invalid entries
                continue
        
        return cleaned_count

    async def _cleanup_file_states(self, cutoff_time: datetime) -> int:
        """Clean up expired file states"""
        if not hasattr(self, 'storage_path'):
            return 0
        
        cleaned_count = 0
        
        try:
            for filename in os.listdir(self.storage_path):
                if filename.endswith('.json'):
                    filepath = os.path.join(self.storage_path, filename)
                    
                    async with aiofiles.open(filepath, 'r') as f:
                        content = await f.read()
                        snapshot_dict = json.loads(content)
                        timestamp = datetime.fromisoformat(snapshot_dict['timestamp'])
                        
                        if timestamp < cutoff_time:
                            os.remove(filepath)
                            cleaned_count += 1
        except Exception as e:
            logger.error(f"Error during file cleanup: {e}")
        
        return cleaned_count

    async def _persist_snapshot(self, snapshot: StateSnapshot) -> None:
        """Persist snapshot to configured backend"""
        if self.backend == StatePersistenceBackend.REDIS:
            await self._persist_redis(snapshot)
        elif self.backend == StatePersistenceBackend.FILE:
            await self._persist_file(snapshot)
        # Memory backend uses only local cache

    async def _persist_redis(self, snapshot: StateSnapshot) -> None:
        """Persist snapshot to Redis"""
        if self.redis_client:
            key = f"{snapshot.agent_id}:{snapshot.state_type.value}"
            value = json.dumps(snapshot.to_dict())
            
            # Set with expiration (24 hours)
            await self.redis_client.setex(key, 86400, value)

    async def _persist_file(self, snapshot: StateSnapshot) -> None:
        """Persist snapshot to file"""
        if hasattr(self, 'storage_path'):
            filename = f"{snapshot.agent_id}_{snapshot.state_type.value}_{snapshot.timestamp.timestamp()}.json"
            filepath = os.path.join(self.storage_path, filename)
            
            async with aiofiles.open(filepath, 'w') as f:
                await f.write(json.dumps(snapshot.to_dict(), indent=2))

    async def _retrieve_snapshot(self, agent_id: str, state_type: StateType) -> Optional[StateSnapshot]:
        """Retrieve snapshot from configured backend"""
        # Check cache first
        cache_key = f"{agent_id}:{state_type.value}"
        if cache_key in self._state_cache:
            return self._state_cache[cache_key]
        
        # Retrieve from backend
        if self.backend == StatePersistenceBackend.REDIS:
            return await self._retrieve_redis(agent_id, state_type)
        elif self.backend == StatePersistenceBackend.FILE:
            return await self._retrieve_file(agent_id, state_type)
        
        return None

    async def _retrieve_redis(self, agent_id: str, state_type: StateType) -> Optional[StateSnapshot]:
        """Retrieve snapshot from Redis"""
        if self.redis_client:
            key = f"{agent_id}:{state_type.value}"
            value = await self.redis_client.get(key)
            
            if value:
                snapshot_dict = json.loads(value)
                return StateSnapshot.from_dict(snapshot_dict)
        
        return None

    async def _retrieve_file(self, agent_id: str, state_type: StateType) -> Optional[StateSnapshot]:
        """Retrieve most recent snapshot from file"""
        if not hasattr(self, 'storage_path'):
            return None
        
        try:
            # Find most recent file for this agent and state type
            pattern = f"{agent_id}_{state_type.value}_"
            matching_files = []
            
            for filename in os.listdir(self.storage_path):
                if filename.startswith(pattern) and filename.endswith('.json'):
                    matching_files.append(filename)
            
            if not matching_files:
                return None
            
            # Get most recent file
            latest_file = max(matching_files)
            filepath = os.path.join(self.storage_path, latest_file)
            
            async with aiofiles.open(filepath, 'r') as f:
                content = await f.read()
                snapshot_dict = json.loads(content)
                return StateSnapshot.from_dict(snapshot_dict)
        
        except Exception as e:
            logger.error(f"Error retrieving file snapshot: {e}")
            return None

    async def _periodic_cleanup(self) -> None:
        """Periodic cleanup of expired states"""
        while True:
            try:
                await asyncio.sleep(self._cleanup_interval)
                await self.cleanup_expired_states()
            except Exception as e:
                logger.error(f"Error in periodic cleanup: {e}")

    async def get_health_status(self) -> Dict[str, Any]:
        """Get health status of state management system"""
        try:
            status = {
                'backend': self.backend.value,
                'healthy': True,
                'cache_size': len(self._state_cache),
                'timestamp': datetime.utcnow().isoformat()
            }
            
            if self.backend == StatePersistenceBackend.REDIS and self.redis_client:
                await self.redis_client.ping()
                status['redis_connected'] = True
            
            return status
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                'backend': self.backend.value,
                'healthy': False,
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }

# Global state manager instance
_state_manager: Optional[AgentStateManager] = None

async def get_state_manager() -> AgentStateManager:
    """Get global state manager instance"""
    global _state_manager
    
    if _state_manager is None:
        backend = StatePersistenceBackend(
            os.getenv('AGENT_STATE_BACKEND', StatePersistenceBackend.REDIS.value)
        )
        _state_manager = AgentStateManager(backend)
        await _state_manager.initialize()
    
    return _state_manager
