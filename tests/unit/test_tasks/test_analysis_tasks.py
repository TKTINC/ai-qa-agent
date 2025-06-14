"""
Tests for Analysis Task Management
AI QA Agent - Enhanced Sprint 1.4
"""
import pytest
import asyncio
from datetime import datetime
from unittest.mock import patch, MagicMock

from src.tasks.analysis_tasks import AnalysisTaskManager
from src.api.routes.analysis import TaskStatus

class TestAnalysisTaskManager:
    """Test analysis task management functionality"""
    
    @pytest.fixture
    def task_manager(self):
        return AnalysisTaskManager()
    
    @pytest.fixture
    def sample_task_status(self):
        return TaskStatus(
            task_id="test-task-123",
            status="running",
            progress=0.5,
            message="Analysis in progress",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
    
    @pytest.mark.asyncio
    async def test_store_task_status_memory(self, task_manager, sample_task_status):
        """Test storing task status in memory (no Redis)"""
        # Ensure we're using memory storage
        task_manager.redis_client = None
        
        success = await task_manager.store_task_status(sample_task_status)
        assert success
        
        # Verify it's stored in memory cache
        assert sample_task_status.task_id in task_manager.task_cache
        stored_data = task_manager.task_cache[sample_task_status.task_id]
        assert stored_data["status"] == "running"
        assert stored_data["progress"] == 0.5
    
    @pytest.mark.asyncio
    async def test_get_task_status_memory(self, task_manager, sample_task_status):
        """Test retrieving task status from memory"""
        task_manager.redis_client = None
        
        # Store first
        await task_manager.store_task_status(sample_task_status)
        
        # Retrieve
        retrieved = await task_manager.get_task_status(sample_task_status.task_id)
        assert retrieved is not None
        assert retrieved.task_id == sample_task_status.task_id
        assert retrieved.status == sample_task_status.status
        assert retrieved.progress == sample_task_status.progress
    
    @pytest.mark.asyncio
    async def test_get_task_status_not_found(self, task_manager):
        """Test retrieving non-existent task"""
        task_manager.redis_client = None
        
        retrieved = await task_manager.get_task_status("nonexistent-task")
        assert retrieved is None
    
    @pytest.mark.asyncio
    async def test_get_all_tasks_memory(self, task_manager):
        """Test getting all tasks from memory"""
        task_manager.redis_client = None
        
        # Create multiple tasks
        tasks = []
        for i in range(3):
            task = TaskStatus(
                task_id=f"task-{i}",
                status="completed" if i < 2 else "running",
                progress=1.0 if i < 2 else 0.7,
                message=f"Task {i}",
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            tasks.append(task)
            await task_manager.store_task_status(task)
        
        # Get all tasks
        all_tasks = await task_manager.get_all_tasks()
        assert len(all_tasks) == 3
        
        # Get filtered tasks
        completed_tasks = await task_manager.get_all_tasks(status="completed")
        assert len(completed_tasks) == 2
        
        # Get limited tasks
        limited_tasks = await task_manager.get_all_tasks(limit=2)
        assert len(limited_tasks) == 2
    
    @pytest.mark.asyncio
    async def test_cancel_task(self, task_manager):
        """Test cancelling a task"""
        task_manager.redis_client = None
        
        # Create running task
        task = TaskStatus(
            task_id="cancel-test",
            status="running",
            progress=0.3,
            message="Running task",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        await task_manager.store_task_status(task)
        
        # Cancel task
        success = await task_manager.cancel_task("cancel-test")
        assert success
        
        # Verify cancellation
        cancelled_task = await task_manager.get_task_status("cancel-test")
        assert cancelled_task.status == "cancelled"
        assert cancelled_task.message == "Task cancelled by user"
    
    @pytest.mark.asyncio
    async def test_cancel_task_not_found(self, task_manager):
        """Test cancelling non-existent task"""
        task_manager.redis_client = None
        
        success = await task_manager.cancel_task("nonexistent")
        assert not success
    
    @pytest.mark.asyncio
    async def test_cancel_completed_task(self, task_manager):
        """Test cancelling already completed task"""
        task_manager.redis_client = None
        
        # Create completed task
        task = TaskStatus(
            task_id="completed-task",
            status="completed",
            progress=1.0,
            message="Task completed",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        await task_manager.store_task_status(task)
        
        # Try to cancel
        success = await task_manager.cancel_task("completed-task")
        assert not success  # Cannot cancel completed task
    
    @pytest.mark.asyncio
    async def test_cleanup_old_tasks(self, task_manager):
        """Test cleaning up old tasks"""
        task_manager.redis_client = None
        
        # Create old task
        old_task = TaskStatus(
            task_id="old-task",
            status="completed", 
            progress=1.0,
            message="Old task",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        # Manually set old timestamp in cache
        await task_manager.store_task_status(old_task)
        task_data = task_manager.task_cache["old-task"]
        task_data["updated_at"] = "2020-01-01T00:00:00"  # Very old date
        
        # Create recent task
        recent_task = TaskStatus(
            task_id="recent-task",
            status="completed",
            progress=1.0,
            message="Recent task",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        await task_manager.store_task_status(recent_task)
        
        # Cleanup old tasks
        await task_manager.cleanup_old_tasks(days=1)
        
        # Verify old task removed, recent task remains
        assert "old-task" not in task_manager.task_cache
        assert "recent-task" in task_manager.task_cache
    
    @patch('redis.Redis')
    def test_redis_initialization_success(self, mock_redis):
        """Test successful Redis initialization"""
        # Mock successful Redis connection
        mock_redis_instance = MagicMock()
        mock_redis.return_value = mock_redis_instance
        mock_redis_instance.ping.return_value = True
        
        task_manager = AnalysisTaskManager()
        assert task_manager.redis_client is not None
    
    @patch('redis.Redis')
    def test_redis_initialization_failure(self, mock_redis):
        """Test Redis initialization failure fallback"""
        # Mock Redis connection failure
        mock_redis.side_effect = Exception("Connection failed")
        
        task_manager = AnalysisTaskManager()
        assert task_manager.redis_client is None
    
    @pytest.mark.asyncio
    @patch('redis.Redis')
    async def test_store_with_redis_error(self, mock_redis, sample_task_status):
        """Test storing task when Redis fails"""
        # Setup mock Redis that fails on setex
        mock_redis_instance = MagicMock()
        mock_redis.return_value = mock_redis_instance
        mock_redis_instance.ping.return_value = True
        mock_redis_instance.setex.side_effect = Exception("Redis error")
        
        task_manager = AnalysisTaskManager()
        
        # Should handle Redis error gracefully
        success = await task_manager.store_task_status(sample_task_status)
        assert not success  # Should return False on error

if __name__ == "__main__":
    pytest.main([__file__])
