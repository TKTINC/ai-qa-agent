"""
Analysis Task Management System
AI QA Agent - Enhanced Sprint 1.4
"""
import asyncio
import json
import redis
from typing import Dict, List, Optional
from datetime import datetime
import os

from src.core.logging import get_logger

logger = get_logger(__name__)

class AnalysisTaskManager:
    """Manage analysis tasks with Redis backend"""
    
    def __init__(self):
        self.redis_client = None
        self.task_cache: Dict[str, any] = {}  # Fallback in-memory cache
        self._initialize_redis()
    
    def _initialize_redis(self):
        """Initialize Redis connection with fallback"""
        try:
            redis_host = os.getenv('REDIS_HOST', 'localhost')
            redis_port = int(os.getenv('REDIS_PORT', '6379'))
            self.redis_client = redis.Redis(
                host=redis_host,
                port=redis_port,
                decode_responses=True,
                socket_timeout=5
            )
            # Test connection
            self.redis_client.ping()
            logger.info("Connected to Redis for task management")
        except Exception as e:
            logger.warning(f"Redis not available, using in-memory cache: {e}")
            self.redis_client = None
    
    async def store_task_status(self, task_status) -> bool:
        """Store task status with Redis or fallback to memory"""
        try:
            task_data = {
                "task_id": task_status.task_id,
                "status": task_status.status,
                "progress": task_status.progress,
                "message": task_status.message,
                "result": task_status.result.dict() if task_status.result else None,
                "error": task_status.error,
                "created_at": task_status.created_at.isoformat(),
                "updated_at": task_status.updated_at.isoformat()
            }
            
            if self.redis_client:
                # Store in Redis with 24 hour expiration
                self.redis_client.setex(
                    f"task:{task_status.task_id}",
                    86400,  # 24 hours
                    json.dumps(task_data, default=str)
                )
            else:
                # Fallback to memory
                self.task_cache[task_status.task_id] = task_data
            
            return True
        except Exception as e:
            logger.error(f"Failed to store task status: {e}")
            return False
    
    async def get_task_status(self, task_id: str):
        """Get task status from Redis or memory"""
        try:
            task_data = None
            
            if self.redis_client:
                data = self.redis_client.get(f"task:{task_id}")
                if data:
                    task_data = json.loads(data)
            else:
                task_data = self.task_cache.get(task_id)
            
            if not task_data:
                return None
            
            # Convert back to TaskStatus object (simplified)
            from src.api.routes.analysis import TaskStatus, AnalysisResult
            
            result = None
            if task_data.get("result"):
                result = AnalysisResult(**task_data["result"])
            
            return TaskStatus(
                task_id=task_data["task_id"],
                status=task_data["status"],
                progress=task_data["progress"],
                message=task_data["message"],
                result=result,
                error=task_data.get("error"),
                created_at=datetime.fromisoformat(task_data["created_at"]),
                updated_at=datetime.fromisoformat(task_data["updated_at"])
            )
        
        except Exception as e:
            logger.error(f"Failed to get task status: {e}")
            return None
    
    async def get_all_tasks(self, status: Optional[str] = None, limit: int = 50) -> List:
        """Get all tasks with optional filtering"""
        try:
            tasks = []
            
            if self.redis_client:
                # Get all task keys
                keys = self.redis_client.keys("task:*")
                for key in keys[:limit]:
                    data = self.redis_client.get(key)
                    if data:
                        task_data = json.loads(data)
                        if not status or task_data.get("status") == status:
                            tasks.append(task_data)
            else:
                # Use memory cache
                for task_id, task_data in self.task_cache.items():
                    if not status or task_data.get("status") == status:
                        tasks.append(task_data)
                    if len(tasks) >= limit:
                        break
            
            # Convert to TaskStatus objects
            from src.api.routes.analysis import TaskStatus, AnalysisResult
            
            result_tasks = []
            for task_data in tasks:
                result = None
                if task_data.get("result"):
                    result = AnalysisResult(**task_data["result"])
                
                result_tasks.append(TaskStatus(
                    task_id=task_data["task_id"],
                    status=task_data["status"],
                    progress=task_data["progress"],
                    message=task_data["message"],
                    result=result,
                    error=task_data.get("error"),
                    created_at=datetime.fromisoformat(task_data["created_at"]),
                    updated_at=datetime.fromisoformat(task_data["updated_at"])
                ))
            
            return result_tasks
        
        except Exception as e:
            logger.error(f"Failed to get all tasks: {e}")
            return []
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task"""
        try:
            task_status = await self.get_task_status(task_id)
            if not task_status or task_status.status not in ["pending", "running"]:
                return False
            
            # Update status to cancelled
            task_status.status = "cancelled"
            task_status.message = "Task cancelled by user"
            task_status.updated_at = datetime.utcnow()
            
            await self.store_task_status(task_status)
            return True
        
        except Exception as e:
            logger.error(f"Failed to cancel task: {e}")
            return False
    
    async def cleanup_old_tasks(self, days: int = 7):
        """Clean up old completed tasks"""
        try:
            if self.redis_client:
                # Redis handles expiration automatically
                return
            
            # Clean memory cache
            cutoff = datetime.utcnow().timestamp() - (days * 86400)
            to_remove = []
            
            for task_id, task_data in self.task_cache.items():
                updated_at = datetime.fromisoformat(task_data["updated_at"])
                if updated_at.timestamp() < cutoff:
                    to_remove.append(task_id)
            
            for task_id in to_remove:
                del self.task_cache[task_id]
            
            logger.info(f"Cleaned up {len(to_remove)} old tasks")
        
        except Exception as e:
            logger.error(f"Failed to cleanup tasks: {e}")
