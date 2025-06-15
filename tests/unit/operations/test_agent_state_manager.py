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
