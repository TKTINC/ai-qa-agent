#!/bin/bash
# Setup Script for Sprint 5.1: Agent-Optimized Container Architecture
# AI QA Agent - Sprint 5.1

set -e
echo "🚀 Setting up Sprint 5.1: Agent-Optimized Container Architecture..."

# Check prerequisites (previous sprints completed)
if [ ! -f "src/web/demos/scenario_engine.py" ]; then
    echo "❌ Error: Sprint 4.3 must be completed first"
    exit 1
fi

# Install dependencies with pip3 (macOS compatible)
echo "📦 Installing new dependencies..."
pip3 install docker==6.1.3 kubernetes==28.1.0 pyyaml==6.0.1

# Create Docker directory structure
echo "📁 Creating Docker directory structure..."
mkdir -p docker/agent-system
mkdir -p docker/monitoring
mkdir -p docker/scripts
mkdir -p k8s/agent-system
mkdir -p k8s/monitoring
mkdir -p k8s/configs

# Create agent state management module
echo "📄 Creating src/operations/agent_state_manager.py..."
cat > src/operations/agent_state_manager.py << 'EOF'
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
EOF

# Create operations directory if it doesn't exist
mkdir -p src/operations

# Create operations __init__.py
echo "📄 Creating src/operations/__init__.py..."
cat > src/operations/__init__.py << 'EOF'
"""
Operations module for production deployment and management
"""

from .agent_state_manager import AgentStateManager, get_state_manager

__all__ = [
    'AgentStateManager',
    'get_state_manager'
]
EOF

# Create Docker base image for agents
echo "📄 Creating docker/agent-system/Dockerfile.agent-base..."
cat > docker/agent-system/Dockerfile.agent-base << 'EOF'
# Multi-stage Docker build optimized for AI Agent workloads
# Base stage with common dependencies and optimizations
FROM python:3.11-slim as agent-base

# Set optimal environment variables for agent workloads
ENV PYTHONUNBUFFERED=1
ENV PYTHONOPTIMIZE=2
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONHASHSEED=random

# Agent-specific environment variables
ENV AGENT_REASONING_WORKERS=4
ENV AGENT_MEMORY_LIMIT=2048
ENV CONVERSATION_CACHE_SIZE=1000
ENV LEARNING_BATCH_SIZE=100

# Install system dependencies optimized for AI workloads
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    build-essential \
    curl \
    git \
    libc6-dev \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Create app user for security
RUN useradd --create-home --shell /bin/bash app

# Set working directory
WORKDIR /app

# Install Python dependencies with optimizations
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Install performance libraries for agent workloads
RUN pip install --no-cache-dir \
    uvloop==0.19.0 \
    orjson==3.9.10 \
    asyncpg==0.29.0 \
    redis[hiredis]==5.0.1 \
    aiofiles==23.2.1

# Copy application code
COPY --chown=app:app . .

# Switch to app user
USER app

# Health check for agent systems
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import asyncio; from src.operations.agent_state_manager import get_state_manager; asyncio.run(get_state_manager().get_health_status())" || exit 1

# Default command (to be overridden in derived images)
CMD ["python", "-m", "src.api.main"]
EOF

# Create Agent Orchestrator Dockerfile
echo "📄 Creating docker/agent-system/Dockerfile.orchestrator..."
cat > docker/agent-system/Dockerfile.orchestrator << 'EOF'
# Agent Orchestrator Container - Optimized for reasoning and coordination
FROM agent-base as agent-orchestrator

# Orchestrator-specific optimizations
ENV AGENT_MODE=orchestrator
ENV REASONING_CACHE_SIZE=1000
ENV CONVERSATION_CONTEXT_LIMIT=10000
ENV MAX_CONCURRENT_REASONING=8

# Orchestrator requires more memory for reasoning tasks
ENV AGENT_MEMORY_LIMIT=4096

# Copy orchestrator-specific code
COPY --chown=app:app src/agent/orchestrator.py /app/src/agent/
COPY --chown=app:app src/agent/reasoning/ /app/src/agent/reasoning/
COPY --chown=app:app src/agent/planning/ /app/src/agent/planning/
COPY --chown=app:app src/agent/memory/ /app/src/agent/memory/

# Health check specific to orchestrator
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health/agent')" || exit 1

# Expose agent orchestrator port
EXPOSE 8000

# Start orchestrator service
CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
EOF

# Create Specialist Agents Dockerfile
echo "📄 Creating docker/agent-system/Dockerfile.specialists..."
cat > docker/agent-system/Dockerfile.specialists << 'EOF'
# Specialist Agents Container - Optimized for domain expertise
FROM agent-base as specialist-agents

# Specialist-specific optimizations
ENV AGENT_MODE=specialist
ENV SPECIALIST_POOL_SIZE=5
ENV TOOL_EXECUTION_TIMEOUT=300
ENV DOMAIN_KNOWLEDGE_CACHE=2048

# Copy specialist agent code
COPY --chown=app:app src/agent/specialists/ /app/src/agent/specialists/
COPY --chown=app:app src/agent/tools/ /app/src/agent/tools/
COPY --chown=app:app src/validation/ /app/src/validation/

# Health check for specialists
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "from src.agent.specialists.base_specialist import check_specialist_health; check_specialist_health()" || exit 1

# Expose specialist agent port
EXPOSE 8001

# Start specialist agent pool
CMD ["python", "-m", "src.agent.specialists.specialist_pool"]
EOF

# Create Conversation Manager Dockerfile
echo "📄 Creating docker/agent-system/Dockerfile.conversation..."
cat > docker/agent-system/Dockerfile.conversation << 'EOF'
# Conversation Manager Container - Optimized for real-time communication
FROM agent-base as conversation-manager

# Conversation-specific optimizations
ENV AGENT_MODE=conversation
ENV WEBSOCKET_MAX_CONNECTIONS=1000
ENV CONVERSATION_PERSISTENCE=redis
ENV REAL_TIME_ANALYTICS=enabled
ENV SESSION_TIMEOUT=3600

# Install WebSocket optimizations
RUN pip install --no-cache-dir \
    websockets==12.0 \
    python-socketio==5.10.0

# Copy conversation management code
COPY --chown=app:app src/chat/ /app/src/chat/
COPY --chown=app:app src/web/ /app/src/web/
COPY --chown=app:app src/api/routes/agent/ /app/src/api/routes/agent/

# Health check for conversation manager
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8080/health')" || exit 1

# Expose WebSocket and HTTP ports
EXPOSE 8080 8081

# Start conversation manager with WebSocket support
CMD ["python", "-m", "src.chat.conversation_server"]
EOF

# Create Learning Engine Dockerfile
echo "📄 Creating docker/agent-system/Dockerfile.learning..."
cat > docker/agent-system/Dockerfile.learning << 'EOF'
# Learning Engine Container - Optimized for continuous learning
FROM agent-base as learning-engine

# Learning-specific optimizations
ENV AGENT_MODE=learning
ENV LEARNING_MODE=continuous
ENV PATTERN_ANALYSIS=enabled
ENV CROSS_AGENT_LEARNING=enabled
ENV LEARNING_PERSISTENCE_INTERVAL=300

# Install ML libraries for learning
RUN pip install --no-cache-dir \
    scikit-learn==1.3.2 \
    pandas==2.1.3 \
    numpy==1.24.4 \
    scipy==1.11.4

# Copy learning system code
COPY --chown=app:app src/agent/learning/ /app/src/agent/learning/
COPY --chown=app:app src/analytics/ /app/src/analytics/

# Create learning data volume
VOLUME ["/app/learning_data"]

# Health check for learning engine
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "from src.agent.learning.learning_engine import check_learning_health; check_learning_health()" || exit 1

# Expose learning engine port
EXPOSE 8002

# Start learning engine
CMD ["python", "-m", "src.agent.learning.learning_server"]
EOF

# Create production docker-compose file
echo "📄 Creating docker-compose.agent-production.yml..."
cat > docker-compose.agent-production.yml << 'EOF'
version: '3.8'
services:
  # Core Agent Services
  agent-orchestrator:
    build:
      context: .
      dockerfile: docker/agent-system/Dockerfile.orchestrator
    image: qa-agent/orchestrator:latest
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3
    environment:
      - AGENT_MODE=orchestrator
      - REASONING_CACHE_SIZE=1000
      - CONVERSATION_CONTEXT_LIMIT=10000
      - REDIS_URL=redis://redis-agent-state:6379/0
      - DATABASE_URL=postgresql://agent_user:agent_password@postgres-agent:5432/agent_db
    ports:
      - "8000:8000"
    depends_on:
      - redis-agent-state
      - postgres-agent
    healthcheck:
      test: ["CMD", "python", "-c", "import requests; requests.get('http://localhost:8000/health/agent')"]
      interval: 30s
      timeout: 10s
      retries: 3
    networks:
      - agent-network

  # Specialist Agent Pool
  specialist-agents:
    build:
      context: .
      dockerfile: docker/agent-system/Dockerfile.specialists
    image: qa-agent/specialists:latest
    deploy:
      replicas: 5
      resources:
        limits:
          cpus: '1.5'
          memory: 3G
        reservations:
          cpus: '0.5'
          memory: 1G
    environment:
      - AGENT_MODE=specialist
      - SPECIALIST_TYPES=test_architect,code_reviewer,performance_analyst,security_specialist,documentation_expert
      - REDIS_URL=redis://redis-agent-state:6379/1
      - TOOL_EXECUTION_TIMEOUT=300
    depends_on:
      - redis-agent-state
      - agent-orchestrator
    networks:
      - agent-network

  # Conversation Management
  conversation-manager:
    build:
      context: .
      dockerfile: docker/agent-system/Dockerfile.conversation
    image: qa-agent/conversation:latest
    deploy:
      replicas: 2
      resources:
        limits:
          cpus: '1.0'
          memory: 2G
        reservations:
          cpus: '0.5'
          memory: 1G
    environment:
      - AGENT_MODE=conversation
      - WEBSOCKET_MAX_CONNECTIONS=1000
      - CONVERSATION_PERSISTENCE=redis
      - REAL_TIME_ANALYTICS=enabled
      - REDIS_URL=redis://redis-agent-state:6379/2
    ports:
      - "8080:8080"
      - "8081:8081"
    depends_on:
      - redis-agent-state
    networks:
      - agent-network

  # Agent Learning System
  learning-engine:
    build:
      context: .
      dockerfile: docker/agent-system/Dockerfile.learning
    image: qa-agent/learning:latest
    deploy:
      replicas: 1
      resources:
        limits:
          cpus: '1.0'
          memory: 3G
        reservations:
          cpus: '0.5'
          memory: 1.5G
    environment:
      - AGENT_MODE=learning
      - LEARNING_MODE=continuous
      - PATTERN_ANALYSIS=enabled
      - CROSS_AGENT_LEARNING=enabled
      - REDIS_URL=redis://redis-agent-state:6379/3
    volumes:
      - agent_learning_data:/app/learning_data
    depends_on:
      - redis-agent-state
    networks:
      - agent-network

  # Redis for Agent State and Learning
  redis-agent-state:
    image: redis:7-alpine
    command: redis-server --maxmemory 512mb --maxmemory-policy allkeys-lru --appendonly yes
    deploy:
      replicas: 1
      resources:
        limits:
          cpus: '0.5'
          memory: 1G
        reservations:
          cpus: '0.2'
          memory: 512M
    volumes:
      - redis_agent_data:/data
    ports:
      - "6379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 30s
      timeout: 5s
      retries: 3
    networks:
      - agent-network

  # PostgreSQL for persistent data
  postgres-agent:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=agent_db
      - POSTGRES_USER=agent_user
      - POSTGRES_PASSWORD=agent_password
    volumes:
      - postgres_agent_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U agent_user -d agent_db"]
      interval: 30s
      timeout: 5s
      retries: 3
    networks:
      - agent-network

  # Load Balancer (HAProxy)
  load-balancer:
    image: haproxy:2.8-alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./docker/configs/haproxy.cfg:/usr/local/etc/haproxy/haproxy.cfg:ro
    depends_on:
      - agent-orchestrator
      - conversation-manager
    networks:
      - agent-network

volumes:
  agent_learning_data:
  redis_agent_data:
  postgres_agent_data:

networks:
  agent-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
EOF

# Create Kubernetes deployment manifests
echo "📄 Creating k8s/agent-system/orchestrator-deployment.yaml..."
cat > k8s/agent-system/orchestrator-deployment.yaml << 'EOF'
# Agent Orchestrator Deployment for Kubernetes
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agent-orchestrator
  namespace: qa-agent
  labels:
    app: agent-orchestrator
    component: reasoning
    tier: core
spec:
  replicas: 3
  selector:
    matchLabels:
      app: agent-orchestrator
  template:
    metadata:
      labels:
        app: agent-orchestrator
        component: reasoning
        tier: core
    spec:
      containers:
      - name: orchestrator
        image: qa-agent/orchestrator:latest
        imagePullPolicy: Always
        ports:
        - containerPort: 8000
          name: http
        resources:
          requests:
            cpu: 1000m
            memory: 2Gi
            ephemeral-storage: 1Gi
          limits:
            cpu: 2000m
            memory: 4Gi
            ephemeral-storage: 2Gi
        env:
        - name: AGENT_MODE
          value: "orchestrator"
        - name: AGENT_REASONING_CACHE
          value: "redis://redis-agent-state:6379/0"
        - name: CONVERSATION_PERSISTENCE
          value: "enabled"
        - name: MULTI_AGENT_COORDINATION
          value: "enabled"
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: agent-database-secret
              key: connection-string
        - name: REDIS_URL
          valueFrom:
            configMapKeyRef:
              name: agent-config
              key: redis-url
        livenessProbe:
          httpGet:
            path: /health/live
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 2
        startupProbe:
          httpGet:
            path: /health/startup
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 6
        volumeMounts:
        - name: agent-config
          mountPath: /app/config
          readOnly: true
        - name: agent-logs
          mountPath: /app/logs
      volumes:
      - name: agent-config
        configMap:
          name: agent-config
      - name: agent-logs
        emptyDir: {}
      nodeSelector:
        workload-type: cpu-intensive
      tolerations:
      - key: "workload-type"
        operator: "Equal"
        value: "cpu-intensive"
        effect: "NoSchedule"

---
apiVersion: v1
kind: Service
metadata:
  name: agent-orchestrator
  namespace: qa-agent
  labels:
    app: agent-orchestrator
spec:
  selector:
    app: agent-orchestrator
  ports:
  - name: http
    port: 80
    targetPort: 8000
    protocol: TCP
  type: ClusterIP

---
# Horizontal Pod Autoscaler for Agent Orchestrator
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: agent-orchestrator-hpa
  namespace: qa-agent
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: agent-orchestrator
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  - type: Pods
    pods:
      metric:
        name: agent_conversation_queue_length
      target:
        type: AverageValue
        averageValue: "10"
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 100
        periodSeconds: 15
      - type: Pods
        value: 2
        periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
EOF

# Create specialist agents deployment
echo "📄 Creating k8s/agent-system/specialists-deployment.yaml..."
cat > k8s/agent-system/specialists-deployment.yaml << 'EOF'
# Specialist Agents Deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: specialist-agents
  namespace: qa-agent
  labels:
    app: specialist-agents
    component: expertise
    tier: core
spec:
  replicas: 5
  selector:
    matchLabels:
      app: specialist-agents
  template:
    metadata:
      labels:
        app: specialist-agents
        component: expertise
        tier: core
    spec:
      containers:
      - name: specialists
        image: qa-agent/specialists:latest
        imagePullPolicy: Always
        ports:
        - containerPort: 8001
          name: grpc
        resources:
          requests:
            cpu: 500m
            memory: 1Gi
          limits:
            cpu: 1500m
            memory: 3Gi
        env:
        - name: AGENT_MODE
          value: "specialist"
        - name: SPECIALIST_TYPES
          value: "test_architect,code_reviewer,performance_analyst,security_specialist,documentation_expert"
        - name: REDIS_URL
          valueFrom:
            configMapKeyRef:
              name: agent-config
              key: redis-url
        - name: TOOL_EXECUTION_TIMEOUT
          value: "300"
        livenessProbe:
          exec:
            command:
            - python
            - -c
            - "from src.agent.specialists.base_specialist import check_specialist_health; check_specialist_health()"
          initialDelaySeconds: 30
          periodSeconds: 30
        readinessProbe:
          exec:
            command:
            - python
            - -c
            - "from src.agent.specialists.base_specialist import check_specialist_health; check_specialist_health()"
          initialDelaySeconds: 10
          periodSeconds: 10

---
apiVersion: v1
kind: Service
metadata:
  name: specialist-agents
  namespace: qa-agent
spec:
  selector:
    app: specialist-agents
  ports:
  - name: grpc
    port: 8001
    targetPort: 8001
  type: ClusterIP

---
# HPA for Specialist Agents
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: specialist-agents-hpa
  namespace: qa-agent
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: specialist-agents
  minReplicas: 5
  maxReplicas: 50
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 75
  - type: Pods
    pods:
      metric:
        name: specialist_task_queue_length
      target:
        type: AverageValue
        averageValue: "5"
EOF

# Create conversation manager deployment
echo "📄 Creating k8s/agent-system/conversation-deployment.yaml..."
cat > k8s/agent-system/conversation-deployment.yaml << 'EOF'
# Conversation Manager Deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: conversation-manager
  namespace: qa-agent
  labels:
    app: conversation-manager
    component: communication
    tier: interface
spec:
  replicas: 2
  selector:
    matchLabels:
      app: conversation-manager
  template:
    metadata:
      labels:
        app: conversation-manager
        component: communication
        tier: interface
    spec:
      containers:
      - name: conversation
        image: qa-agent/conversation:latest
        imagePullPolicy: Always
        ports:
        - containerPort: 8080
          name: websocket
        - containerPort: 8081
          name: http
        resources:
          requests:
            cpu: 500m
            memory: 1Gi
          limits:
            cpu: 1000m
            memory: 2Gi
        env:
        - name: AGENT_MODE
          value: "conversation"
        - name: WEBSOCKET_MAX_CONNECTIONS
          value: "1000"
        - name: CONVERSATION_PERSISTENCE
          value: "redis"
        - name: REAL_TIME_ANALYTICS
          value: "enabled"
        - name: REDIS_URL
          valueFrom:
            configMapKeyRef:
              name: agent-config
              key: redis-url
        livenessProbe:
          httpGet:
            path: /health
            port: 8081
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8081
          initialDelaySeconds: 5
          periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: conversation-manager
  namespace: qa-agent
spec:
  selector:
    app: conversation-manager
  ports:
  - name: websocket
    port: 8080
    targetPort: 8080
  - name: http
    port: 8081
    targetPort: 8081
  type: LoadBalancer
EOF

# Create namespace and configs
echo "📄 Creating k8s/configs/namespace.yaml..."
cat > k8s/configs/namespace.yaml << 'EOF'
apiVersion: v1
kind: Namespace
metadata:
  name: qa-agent
  labels:
    name: qa-agent
    tier: application
EOF

# Create ConfigMap
echo "📄 Creating k8s/configs/agent-config.yaml..."
cat > k8s/configs/agent-config.yaml << 'EOF'
apiVersion: v1
kind: ConfigMap
metadata:
  name: agent-config
  namespace: qa-agent
data:
  redis-url: "redis://redis-agent-state:6379"
  log-level: "INFO"
  agent-mode: "production"
  reasoning-cache-size: "1000"
  conversation-context-limit: "10000"
  learning-mode: "continuous"
  pattern-analysis: "enabled"
  cross-agent-learning: "enabled"
EOF

# Create build script
echo "📄 Creating docker/scripts/build_agent_images.sh..."
cat > docker/scripts/build_agent_images.sh << 'EOF'
#!/bin/bash
# Build script for all agent Docker images

set -e

echo "🚀 Building AI QA Agent Docker Images..."

# Build base image
echo "📦 Building agent base image..."
docker build -f docker/agent-system/Dockerfile.agent-base -t qa-agent/base:latest .

# Build orchestrator
echo "📦 Building agent orchestrator..."
docker build -f docker/agent-system/Dockerfile.orchestrator -t qa-agent/orchestrator:latest .

# Build specialists
echo "📦 Building specialist agents..."
docker build -f docker/agent-system/Dockerfile.specialists -t qa-agent/specialists:latest .

# Build conversation manager
echo "📦 Building conversation manager..."
docker build -f docker/agent-system/Dockerfile.conversation -t qa-agent/conversation:latest .

# Build learning engine
echo "📦 Building learning engine..."
docker build -f docker/agent-system/Dockerfile.learning -t qa-agent/learning:latest .

echo "✅ All agent images built successfully!"

# List built images
echo "📋 Built images:"
docker images qa-agent/*
EOF

chmod +x docker/scripts/build_agent_images.sh

# Create deployment script
echo "📄 Creating docker/scripts/deploy_k8s.sh..."
cat > docker/scripts/deploy_k8s.sh << 'EOF'
#!/bin/bash
# Kubernetes deployment script for AI QA Agent

set -e

echo "🚀 Deploying AI QA Agent to Kubernetes..."

# Apply namespace and configs
echo "📝 Creating namespace and configurations..."
kubectl apply -f k8s/configs/namespace.yaml
kubectl apply -f k8s/configs/agent-config.yaml

# Deploy core components
echo "📦 Deploying agent components..."
kubectl apply -f k8s/agent-system/orchestrator-deployment.yaml
kubectl apply -f k8s/agent-system/specialists-deployment.yaml
kubectl apply -f k8s/agent-system/conversation-deployment.yaml

# Wait for deployments
echo "⏳ Waiting for deployments to be ready..."
kubectl wait --for=condition=available --timeout=300s deployment/agent-orchestrator -n qa-agent
kubectl wait --for=condition=available --timeout=300s deployment/specialist-agents -n qa-agent
kubectl wait --for=condition=available --timeout=300s deployment/conversation-manager -n qa-agent

# Show status
echo "📊 Deployment status:"
kubectl get pods -n qa-agent
kubectl get services -n qa-agent

echo "✅ AI QA Agent deployed successfully to Kubernetes!"
EOF

chmod +x docker/scripts/deploy_k8s.sh

# Create comprehensive tests
echo "📄 Creating tests/unit/operations/test_agent_state_manager.py..."
mkdir -p tests/unit/operations
cat > tests/unit/operations/test_agent_state_manager.py << 'EOF'
"""
Tests for Agent State Manager
"""

import asyncio
import json
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from src.operations.agent_state_manager import (
    AgentStateManager, 
    StateSnapshot, 
    StateType, 
    StatePersistenceBackend
)
from src.agent.core.models import AgentState


@pytest.fixture
def mock_redis():
    """Mock Redis client"""
    mock_client = AsyncMock()
    mock_client.ping.return_value = True
    mock_client.setex.return_value = True
    mock_client.get.return_value = None
    mock_client.delete.return_value = 1
    mock_client.keys.return_value = []
    mock_client.publish.return_value = 1
    return mock_client


@pytest.fixture
def sample_agent_state():
    """Sample agent state for testing"""
    return AgentState(
        agent_id="test_agent_001",
        current_task="code_analysis",
        reasoning_context={
            "step": "analysis",
            "confidence": 0.85,
            "tools_used": ["ast_parser", "complexity_analyzer"]
        },
        conversation_context={
            "session_id": "session_123",
            "turn_count": 5,
            "user_preferences": {"style": "detailed"}
        },
        learning_state={
            "patterns_learned": 3,
            "success_rate": 0.92,
            "last_improvement": datetime.utcnow().isoformat()
        }
    )


@pytest.fixture
def sample_state_snapshot():
    """Sample state snapshot for testing"""
    return StateSnapshot(
        agent_id="test_agent_001",
        state_type=StateType.AGENT_STATE,
        timestamp=datetime.utcnow(),
        data={
            "agent_id": "test_agent_001",
            "status": "active",
            "capabilities": ["reasoning", "learning"]
        },
        metadata={"version": "1.0", "container_id": "test_container"}
    )


class TestStateSnapshot:
    """Test StateSnapshot functionality"""
    
    def test_to_dict(self, sample_state_snapshot):
        """Test snapshot serialization"""
        result = sample_state_snapshot.to_dict()
        
        assert result['agent_id'] == "test_agent_001"
        assert result['state_type'] == StateType.AGENT_STATE.value
        assert 'timestamp' in result
        assert isinstance(result['timestamp'], str)
        assert result['data']['agent_id'] == "test_agent_001"
    
    def test_from_dict(self, sample_state_snapshot):
        """Test snapshot deserialization"""
        snapshot_dict = sample_state_snapshot.to_dict()
        restored = StateSnapshot.from_dict(snapshot_dict)
        
        assert restored.agent_id == sample_state_snapshot.agent_id
        assert restored.state_type == sample_state_snapshot.state_type
        assert restored.data == sample_state_snapshot.data


class TestAgentStateManager:
    """Test AgentStateManager functionality"""
    
    @pytest.mark.asyncio
    async def test_initialization_redis(self, mock_redis):
        """Test Redis initialization"""
        with patch('redis.asyncio.from_url', return_value=mock_redis):
            manager = AgentStateManager(StatePersistenceBackend.REDIS)
            await manager.initialize()
            
            assert manager.redis_client == mock_redis
            mock_redis.ping.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_initialization_memory(self):
        """Test memory backend initialization"""
        manager = AgentStateManager(StatePersistenceBackend.MEMORY)
        await manager.initialize()
        
        assert manager.backend == StatePersistenceBackend.MEMORY
        assert manager.redis_client is None
    
    @pytest.mark.asyncio
    async def test_save_agent_state_redis(self, mock_redis, sample_agent_state):
        """Test saving agent state to Redis"""
        with patch('redis.asyncio.from_url', return_value=mock_redis):
            manager = AgentStateManager(StatePersistenceBackend.REDIS)
            await manager.initialize()
            
            result = await manager.save_agent_state("test_agent", sample_agent_state)
            
            assert result is True
            mock_redis.setex.assert_called_once()
            
            # Verify cache update
            cache_key = "test_agent:agent_state"
            assert cache_key in manager._state_cache
    
    @pytest.mark.asyncio
    async def test_restore_agent_state_redis(self, mock_redis, sample_state_snapshot):
        """Test restoring agent state from Redis"""
        # Mock Redis return value
        mock_redis.get.return_value = json.dumps(sample_state_snapshot.to_dict())
        
        with patch('redis.asyncio.from_url', return_value=mock_redis):
            manager = AgentStateManager(StatePersistenceBackend.REDIS)
            await manager.initialize()
            
            restored_state = await manager.restore_agent_state("test_agent_001")
            
            assert restored_state is not None
            mock_redis.get.assert_called_once_with("test_agent_001:agent_state")
    
    @pytest.mark.asyncio
    async def test_restore_agent_state_not_found(self, mock_redis):
        """Test restoring non-existent agent state"""
        mock_redis.get.return_value = None
        
        with patch('redis.asyncio.from_url', return_value=mock_redis):
            manager = AgentStateManager(StatePersistenceBackend.REDIS)
            await manager.initialize()
            
            restored_state = await manager.restore_agent_state("nonexistent_agent")
            
            assert restored_state is None
    
    @pytest.mark.asyncio
    async def test_distribute_learning_update(self, mock_redis):
        """Test distributing learning updates"""
        learning_update = {
            "pattern": "user_preference_pytest",
            "confidence": 0.9,
            "applicable_agents": ["test_architect", "code_reviewer"]
        }
        
        with patch('redis.asyncio.from_url', return_value=mock_redis):
            manager = AgentStateManager(StatePersistenceBackend.REDIS)
            await manager.initialize()
            
            result = await manager.distribute_learning_update(learning_update)
            
            assert result is True
            mock_redis.setex.assert_called_once()
            mock_redis.publish.assert_called_once_with(
                'agent_learning_updates',
                json.dumps(learning_update)
            )
    
    @pytest.mark.asyncio
    async def test_get_active_agents_redis(self, mock_redis):
        """Test getting active agents from Redis"""
        mock_redis.keys.return_value = [
            "agent_001:agent_state",
            "agent_002:agent_state",
            "agent_003:agent_state"
        ]
        
        with patch('redis.asyncio.from_url', return_value=mock_redis):
            manager = AgentStateManager(StatePersistenceBackend.REDIS)
            await manager.initialize()
            
            active_agents = await manager.get_active_agents()
            
            assert len(active_agents) == 3
            assert "agent_001" in active_agents
            assert "agent_002" in active_agents
            assert "agent_003" in active_agents
    
    @pytest.mark.asyncio
    async def test_cleanup_expired_states(self, mock_redis):
        """Test cleanup of expired states"""
        # Mock expired state data
        expired_timestamp = datetime.utcnow() - timedelta(hours=25)
        expired_snapshot = StateSnapshot(
            agent_id="expired_agent",
            state_type=StateType.AGENT_STATE,
            timestamp=expired_timestamp,
            data={"status": "expired"}
        )
        
        mock_redis.keys.return_value = ["expired_agent:agent_state"]
        mock_redis.get.return_value = json.dumps(expired_snapshot.to_dict())
        mock_redis.delete.return_value = 1
        
        with patch('redis.asyncio.from_url', return_value=mock_redis):
            manager = AgentStateManager(StatePersistenceBackend.REDIS)
            await manager.initialize()
            
            cleaned_count = await manager.cleanup_expired_states(max_age_hours=24)
            
            assert cleaned_count == 1
            mock_redis.delete.assert_called_once_with("expired_agent:agent_state")
    
    @pytest.mark.asyncio
    async def test_health_status_healthy(self, mock_redis):
        """Test health status when system is healthy"""
        with patch('redis.asyncio.from_url', return_value=mock_redis):
            manager = AgentStateManager(StatePersistenceBackend.REDIS)
            await manager.initialize()
            
            health = await manager.get_health_status()
            
            assert health['healthy'] is True
            assert health['backend'] == 'redis'
            assert health['redis_connected'] is True
            assert 'timestamp' in health
    
    @pytest.mark.asyncio
    async def test_health_status_unhealthy(self, mock_redis):
        """Test health status when Redis connection fails"""
        mock_redis.ping.side_effect = Exception("Redis connection failed")
        
        with patch('redis.asyncio.from_url', return_value=mock_redis):
            manager = AgentStateManager(StatePersistenceBackend.REDIS)
            await manager.initialize()
            
            health = await manager.get_health_status()
            
            assert health['healthy'] is False
            assert 'error' in health
    
    @pytest.mark.asyncio
    async def test_memory_backend_operations(self, sample_agent_state):
        """Test memory backend operations"""
        manager = AgentStateManager(StatePersistenceBackend.MEMORY)
        await manager.initialize()
        
        # Test save
        result = await manager.save_agent_state("memory_agent", sample_agent_state)
        assert result is True
        
        # Test restore
        restored = await manager.restore_agent_state("memory_agent")
        assert restored is not None
        
        # Test cache content
        assert len(manager._state_cache) == 1
        assert "memory_agent:agent_state" in manager._state_cache


class TestPerformanceAndReliability:
    """Test performance and reliability aspects"""
    
    @pytest.mark.asyncio
    async def test_concurrent_state_operations(self, mock_redis, sample_agent_state):
        """Test concurrent state save/restore operations"""
        with patch('redis.asyncio.from_url', return_value=mock_redis):
            manager = AgentStateManager(StatePersistenceBackend.REDIS)
            await manager.initialize()
            
            # Create multiple concurrent operations
            tasks = []
            for i in range(10):
                agent_id = f"concurrent_agent_{i}"
                task = manager.save_agent_state(agent_id, sample_agent_state)
                tasks.append(task)
            
            # Wait for all operations to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # All operations should succeed
            assert all(result is True for result in results if not isinstance(result, Exception))
    
    @pytest.mark.asyncio
    async def test_error_resilience(self, mock_redis, sample_agent_state):
        """Test error handling and resilience"""
        # Mock Redis failure
        mock_redis.setex.side_effect = Exception("Redis write failed")
        
        with patch('redis.asyncio.from_url', return_value=mock_redis):
            manager = AgentStateManager(StatePersistenceBackend.REDIS)
            await manager.initialize()
            
            # Should handle Redis failure gracefully
            result = await manager.save_agent_state("failing_agent", sample_agent_state)
            assert result is False  # Should return False on failure
    
    @pytest.mark.asyncio
    async def test_state_snapshot_size_limits(self):
        """Test handling of large state snapshots"""
        # Create large state data
        large_data = {
            "massive_array": list(range(10000)),
            "complex_object": {
                f"key_{i}": f"value_{i}" * 100 
                for i in range(1000)
            }
        }
        
        snapshot = StateSnapshot(
            agent_id="large_state_agent",
            state_type=StateType.AGENT_STATE,
            timestamp=datetime.utcnow(),
            data=large_data
        )
        
        # Should handle serialization of large data
        serialized = snapshot.to_dict()
        assert 'data' in serialized
        
        # Should be able to deserialize
        restored = StateSnapshot.from_dict(serialized)
        assert restored.data == large_data


if __name__ == "__main__":
    pytest.main([__file__])
EOF

# Create operations test init
echo "📄 Creating tests/unit/operations/__init__.py..."
cat > tests/unit/operations/__init__.py << 'EOF'
"""
Tests for operations module
"""
EOF

# Update requirements.txt with new dependencies
echo "📄 Updating requirements.txt..."
cat >> requirements.txt << 'EOF'

# Production Container Dependencies (Sprint 5.1)
docker==6.1.3
kubernetes==28.1.0
pyyaml==6.0.1
aiofiles==23.2.1

# Container Optimization
uvloop==0.19.0
orjson==3.9.10
asyncpg==0.29.0
redis[hiredis]==5.0.1
EOF

# Run tests to verify implementation
echo "🧪 Running tests to verify implementation..."
python3 -m pytest tests/unit/operations/test_agent_state_manager.py -v

# Test basic functionality
echo "🔍 Testing basic functionality..."
python3 -c "
import asyncio
from src.operations.agent_state_manager import AgentStateManager, StatePersistenceBackend

async def test_basic():
    # Test memory backend
    manager = AgentStateManager(StatePersistenceBackend.MEMORY)
    await manager.initialize()
    
    health = await manager.get_health_status()
    print(f'✅ State manager health: {health}')
    
    active_agents = await manager.get_active_agents()
    print(f'✅ Active agents: {active_agents}')
    
    print('✅ Agent state manager basic functionality verified!')

asyncio.run(test_basic())
"

# Build Docker images (optional - can be run separately)
echo "🐳 Docker images ready to build..."
echo "Run: ./docker/scripts/build_agent_images.sh"

echo "✅ Sprint 5.1 setup complete!"

echo "
🎉 Sprint 5.1: Agent-Optimized Container Architecture - COMPLETE!

📦 What was implemented:
  ✅ Agent State Management System - Production-ready state persistence
  ✅ Multi-stage Docker Architecture - Optimized for agent workloads
  ✅ Production Docker Compose - Complete multi-service orchestration
  ✅ Kubernetes Deployments - Enterprise-grade container orchestration
  ✅ Intelligent Auto-scaling - Agent workload-aware scaling policies
  ✅ Performance Optimization - Memory and CPU tuning for reasoning tasks
  ✅ Build and Deploy Scripts - Automated image building and deployment

🚀 Key Features:
  • Agent-optimized containers with reasoning performance tuning
  • Redis-based state persistence across container restarts
  • Kubernetes auto-scaling based on conversation load
  • Multi-service architecture (orchestrator, specialists, conversation, learning)
  • Production monitoring and health checks
  • Enterprise-grade security and resource management

📋 Next Steps:
  1. Build images: ./docker/scripts/build_agent_images.sh
  2. Deploy locally: docker-compose -f docker-compose.agent-production.yml up
  3. Deploy to K8s: ./docker/scripts/deploy_k8s.sh
  4. Ready for Sprint 5.2: Agent Intelligence Monitoring & Observability

💡 This Sprint establishes production-ready containerization optimized for AI agent workloads!
"