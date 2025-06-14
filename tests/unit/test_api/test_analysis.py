"""
Tests for Analysis API Routes
AI QA Agent - Enhanced Sprint 1.4
"""
import pytest
import asyncio
import json
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime

from src.api.main import app
from src.api.routes.analysis import (
    AnalysisRequest, AnalysisResult, TaskStatus, ComponentInfo
)

client = TestClient(app)

class TestAnalysisAPI:
    """Test analysis API endpoints"""
    
    @pytest.fixture
    def sample_analysis_request(self):
        return {
            "file_path": "test_file.py",
            "analysis_type": "file",
            "options": {"include_ml_analysis": True}
        }
    
    @pytest.fixture
    def sample_component(self):
        return ComponentInfo(
            name="test_function",
            type="function",
            start_line=1,
            end_line=10,
            complexity={"cyclomatic": 3, "cognitive": 2},
            quality_metrics={"testability_score": 0.85, "test_priority": 3},
            dependencies=["os", "sys"]
        )
    
    @pytest.fixture
    def sample_analysis_result(self, sample_component):
        return AnalysisResult(
            analysis_id="test-analysis-123",
            analysis_type="file",
            status="completed",
            components=[sample_component],
            quality_summary={"total_components": 1, "average_complexity": 3.0},
            patterns_detected=[{"pattern": "singleton", "confidence": 0.8}],
            recommendations=["Add unit tests for test_function"],
            execution_time=2.5,
            timestamp=datetime.utcnow()
        )
    
    def test_health_endpoint(self):
        """Test that health endpoint works"""
        response = client.get("/health/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
    
    @patch('src.api.routes.analysis.task_manager')
    @patch('src.api.routes.analysis.perform_analysis_task')
    def test_start_analysis_endpoint(self, mock_perform_task, mock_task_manager, sample_analysis_request):
        """Test starting analysis task"""
        # Mock task manager
        mock_task_manager.store_task_status = AsyncMock(return_value=True)
        
        response = client.post("/api/v1/analysis/analyze", json=sample_analysis_request)
        assert response.status_code == 200
        
        data = response.json()
        assert "task_id" in data
        assert data["status"] == "pending"
        assert data["progress"] == 0.0
    
    def test_start_analysis_invalid_request(self):
        """Test starting analysis with invalid request"""
        invalid_request = {
            "analysis_type": "file"
            # Missing required paths
        }
        
        response = client.post("/api/v1/analysis/analyze", json=invalid_request)
        assert response.status_code == 400
        assert "Must provide either file_path, repository_path, or code_content" in response.json()["detail"]
    
    @patch('src.api.routes.analysis.task_manager')
    def test_get_task_status_existing(self, mock_task_manager):
        """Test getting status of existing task"""
        task_status = TaskStatus(
            task_id="test-task-123",
            status="running",
            progress=0.5,
            message="Analysis in progress",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        mock_task_manager.get_task_status = AsyncMock(return_value=task_status)
        
        response = client.get("/api/v1/analysis/tasks/test-task-123")
        assert response.status_code == 200
        
        data = response.json()
        assert data["task_id"] == "test-task-123"
        assert data["status"] == "running"
        assert data["progress"] == 0.5
    
    @patch('src.api.routes.analysis.task_manager')
    def test_get_task_status_not_found(self, mock_task_manager):
        """Test getting status of non-existent task"""
        mock_task_manager.get_task_status = AsyncMock(return_value=None)
        
        response = client.get("/api/v1/analysis/tasks/nonexistent")
        assert response.status_code == 404
        assert response.json()["detail"] == "Task not found"
    
    @patch('src.api.routes.analysis.task_manager')
    def test_get_all_tasks(self, mock_task_manager):
        """Test getting all tasks"""
        tasks = [
            TaskStatus(
                task_id="task-1",
                status="completed",
                progress=1.0,
                message="Analysis completed",
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            ),
            TaskStatus(
                task_id="task-2", 
                status="running",
                progress=0.7,
                message="Analysis in progress",
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
        ]
        mock_task_manager.get_all_tasks = AsyncMock(return_value=tasks)
        
        response = client.get("/api/v1/analysis/tasks")
        assert response.status_code == 200
        
        data = response.json()
        assert len(data) == 2
        assert data[0]["task_id"] == "task-1"
        assert data[1]["task_id"] == "task-2"
    
    @patch('src.api.routes.analysis.task_manager')
    def test_cancel_task(self, mock_task_manager):
        """Test cancelling a task"""
        mock_task_manager.cancel_task = AsyncMock(return_value=True)
        
        response = client.delete("/api/v1/analysis/tasks/test-task-123")
        assert response.status_code == 200
        
        data = response.json()
        assert "cancelled successfully" in data["message"]
    
    @patch('src.api.routes.analysis.task_manager')
    def test_cancel_task_not_found(self, mock_task_manager):
        """Test cancelling non-existent task"""
        mock_task_manager.cancel_task = AsyncMock(return_value=False)
        
        response = client.delete("/api/v1/analysis/tasks/nonexistent")
        assert response.status_code == 404
        assert "Task not found or cannot be cancelled" in response.json()["detail"]
    
    @patch('src.analysis.ast_parser.PythonASTParser')
    @patch('os.path.exists')
    def test_analyze_file_sync(self, mock_exists, mock_parser_class):
        """Test synchronous file analysis"""
        mock_exists.return_value = True
        
        # Mock parser
        mock_parser = AsyncMock()
        mock_parser_class.return_value = mock_parser
        
        # Mock component
        mock_component = MagicMock()
        mock_component.name = "test_function"
        mock_component.type = "function"
        mock_component.start_line = 1
        mock_component.end_line = 10
        mock_component.complexity = {"cyclomatic": 3}
        mock_component.quality_metrics = {"testability_score": 0.85}
        mock_component.dependencies = ["os"]
        
        mock_parser.parse_file = AsyncMock(return_value=[mock_component])
        
        response = client.post("/api/v1/analysis/analyze/file?file_path=test.py")
        assert response.status_code == 200
        
        data = response.json()
        assert data["analysis_type"] == "file"
        assert data["status"] == "completed"
        assert len(data["components"]) == 1
        assert data["components"][0]["name"] == "test_function"
    
    def test_analyze_file_sync_not_found(self):
        """Test synchronous file analysis with non-existent file"""
        response = client.post("/api/v1/analysis/analyze/file?file_path=nonexistent.py")
        assert response.status_code == 404
        assert response.json()["detail"] == "File not found"
    
    @patch('src.analysis.ast_parser.PythonASTParser')
    def test_analyze_content_sync(self, mock_parser_class):
        """Test synchronous content analysis"""
        # Mock parser
        mock_parser = AsyncMock()
        mock_parser_class.return_value = mock_parser
        
        mock_component = MagicMock()
        mock_component.name = "test_function"
        mock_component.type = "function"
        mock_component.start_line = 1
        mock_component.end_line = 5
        mock_component.complexity = {"cyclomatic": 1}
        mock_component.quality_metrics = {"testability_score": 0.9}
        mock_component.dependencies = []
        
        mock_parser.parse_code_string = AsyncMock(return_value=[mock_component])
        
        request_data = {
            "code_content": "def test_function():\n    return True",
            "language": "python"
        }
        
        response = client.post("/api/v1/analysis/analyze/content", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["analysis_type"] == "content"
        assert data["status"] == "completed"
        assert len(data["components"]) == 1
    
    def test_analyze_content_unsupported_language(self):
        """Test content analysis with unsupported language"""
        request_data = {
            "code_content": "console.log('hello');",
            "language": "javascript"
        }
        
        response = client.post("/api/v1/analysis/analyze/content", json=request_data)
        assert response.status_code == 400
        assert "Only Python analysis supported" in response.json()["detail"]

class TestWebSocketEndpoints:
    """Test WebSocket endpoints for real-time updates"""
    
    def test_websocket_connection_placeholder(self):
        """Placeholder test for WebSocket functionality"""
        # WebSocket testing with TestClient is complex
        # In a real implementation, you'd use websocket test client
        # For now, we'll test the connection manager logic
        
        from src.api.routes.analysis import ConnectionManager
        
        manager = ConnectionManager()
        assert len(manager.active_connections) == 0
        assert len(manager.task_connections) == 0

class TestTaskManagement:
    """Test task management functionality"""
    
    @pytest.fixture
    def task_manager(self):
        from src.tasks.analysis_tasks import AnalysisTaskManager
        return AnalysisTaskManager()
    
    @pytest.mark.asyncio
    async def test_store_and_retrieve_task_status(self, task_manager):
        """Test storing and retrieving task status"""
        task_status = TaskStatus(
            task_id="test-task-456",
            status="running",
            progress=0.3,
            message="Test task",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        # Store task
        success = await task_manager.store_task_status(task_status)
        assert success
        
        # Retrieve task
        retrieved = await task_manager.get_task_status("test-task-456")
        assert retrieved is not None
        assert retrieved.task_id == "test-task-456"
        assert retrieved.status == "running"
        assert retrieved.progress == 0.3
    
    @pytest.mark.asyncio
    async def test_get_nonexistent_task(self, task_manager):
        """Test retrieving non-existent task"""
        task = await task_manager.get_task_status("nonexistent-task")
        assert task is None
    
    @pytest.mark.asyncio
    async def test_cancel_task(self, task_manager):
        """Test cancelling a task"""
        # First create a task
        task_status = TaskStatus(
            task_id="cancel-test",
            status="running",
            progress=0.5,
            message="Running task",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        await task_manager.store_task_status(task_status)
        
        # Cancel the task
        success = await task_manager.cancel_task("cancel-test")
        assert success
        
        # Verify it's cancelled
        cancelled_task = await task_manager.get_task_status("cancel-test")
        assert cancelled_task.status == "cancelled"

if __name__ == "__main__":
    pytest.main([__file__])
