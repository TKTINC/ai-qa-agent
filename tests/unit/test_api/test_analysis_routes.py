"""
Comprehensive tests for Analysis API Routes
Tests all analysis endpoints with realistic scenarios
"""

import pytest
import asyncio
import tempfile
import zipfile
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient

from src.api.main import app
from src.analysis.analysis_service import AnalysisProgress, ComprehensiveAnalysisResult
from src.analysis.ast_parser import Language

class TestAnalysisAPIBasic:
    """Test basic Analysis API functionality"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    @pytest.fixture
    def sample_repository_path(self):
        """Create temporary repository for testing"""
        temp_dir = tempfile.mkdtemp()
        repo_path = Path(temp_dir) / "test_repo"
        repo_path.mkdir()
        
        # Create sample Python file
        sample_file = repo_path / "test.py"
        sample_file.write_text('''
def test_function():
    return "hello"
        ''')
        
        return str(repo_path)
    
    def test_analysis_service_status(self, client):
        """Test analysis service status endpoint"""
        response = client.get("/api/v1/analysis/status")
        
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "analysis"
        assert "capabilities" in data
        assert "languages" in data["capabilities"]
    
    def test_analysis_service_test(self, client):
        """Test analysis service test endpoint"""
        response = client.get("/api/v1/analysis/test")
        
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "analysis"
        assert "test_status" in data

class TestRepositoryAnalysisEndpoint:
    """Test repository analysis endpoint"""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    @patch('src.api.routes.analysis.analysis_service')
    def test_start_repository_analysis_success(self, mock_service, client, tmp_path):
        """Test successful repository analysis start"""
        # Create temporary directory
        repo_path = tmp_path / "test_repo"
        repo_path.mkdir()
        (repo_path / "test.py").write_text("def test(): pass")
        
        # Mock service response
        mock_progress = AnalysisProgress(session_id="test-123")
        mock_service.start_analysis.return_value = mock_progress
        
        request_data = {
            "repository_path": str(repo_path),
            "language": "python"
        }
        
        response = client.post("/api/v1/analysis/repository", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "test-123"
        assert data["status"] == "analyzing"
    
    def test_start_repository_analysis_nonexistent_path(self, client):
        """Test repository analysis with non-existent path"""
        request_data = {
            "repository_path": "/nonexistent/path",
            "language": "python"
        }
        
        response = client.post("/api/v1/analysis/repository", json=request_data)
        
        assert response.status_code == 400
        assert "does not exist" in response.json()["detail"]
    
    def test_start_repository_analysis_file_not_directory(self, client, tmp_path):
        """Test repository analysis with file instead of directory"""
        # Create a file instead of directory
        file_path = tmp_path / "test.py"
        file_path.write_text("test content")
        
        request_data = {
            "repository_path": str(file_path),
            "language": "python"
        }
        
        response = client.post("/api/v1/analysis/repository", json=request_data)
        
        assert response.status_code == 400
        assert "not a directory" in response.json()["detail"]

class TestUploadAnalysisEndpoint:
    """Test upload and analysis endpoint"""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    @pytest.fixture
    def sample_zip_file(self, tmp_path):
        """Create sample ZIP file for testing"""
        # Create temporary repository
        repo_dir = tmp_path / "test_repo"
        repo_dir.mkdir()
        (repo_dir / "test.py").write_text("def test(): pass")
        
        # Create ZIP file
        zip_path = tmp_path / "test_repo.zip"
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            zipf.write(repo_dir / "test.py", "test_repo/test.py")
        
        return zip_path
    
    @patch('src.api.routes.analysis.analysis_service')
    def test_upload_and_analyze_success(self, mock_service, client, sample_zip_file):
        """Test successful upload and analysis"""
        # Mock service response
        mock_progress = AnalysisProgress(session_id="upload-123")
        mock_service.analyze_uploaded_archive.return_value = mock_progress
        
        with open(sample_zip_file, 'rb') as f:
            files = {"file": ("test_repo.zip", f, "application/zip")}
            response = client.post(
                "/api/v1/analysis/upload?language=python",
                files=files
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "upload-123"
    
    def test_upload_analysis_no_filename(self, client):
        """Test upload without filename"""
        files = {"file": ("", b"content", "application/zip")}
        response = client.post("/api/v1/analysis/upload", files=files)
        
        assert response.status_code == 400
        assert "No filename provided" in response.json()["detail"]
    
    def test_upload_analysis_unsupported_format(self, client, tmp_path):
        """Test upload with unsupported file format"""
        # Create unsupported file
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("unsupported content")
        
        with open(txt_file, 'rb') as f:
            files = {"file": ("test.txt", f, "text/plain")}
            response = client.post("/api/v1/analysis/upload", files=files)
        
        assert response.status_code == 400
        assert "Unsupported file type" in response.json()["detail"]
    
    def test_upload_analysis_large_file(self, client, tmp_path):
        """Test upload with file exceeding size limit"""
        # Create large file (simulate 60MB)
        large_content = b"x" * (60 * 1024 * 1024)  # 60MB
        
        files = {"file": ("large.zip", large_content, "application/zip")}
        response = client.post("/api/v1/analysis/upload", files=files)
        
        assert response.status_code == 413
        assert "too large" in response.json()["detail"]

class TestSessionManagementEndpoints:
    """Test session management endpoints"""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    @patch('src.api.routes.analysis.analysis_service')
    def test_list_analysis_sessions(self, mock_service, client):
        """Test listing analysis sessions"""
        # Mock service response
        mock_service.list_sessions.return_value = ["session-1", "session-2"]
        mock_progress_1 = AnalysisProgress(session_id="session-1")
        mock_progress_2 = AnalysisProgress(session_id="session-2")
        mock_service.get_progress.side_effect = [mock_progress_1, mock_progress_2]
        mock_service.get_result.return_value = None
        
        response = client.get("/api/v1/analysis/sessions")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["sessions"]) == 2
        assert data["total_count"] == 2
        assert data["page"] == 1
    
    @patch('src.api.routes.analysis.analysis_service')
    def test_list_sessions_pagination(self, mock_service, client):
        """Test session listing with pagination"""
        # Mock many sessions
        session_ids = [f"session-{i}" for i in range(25)]
        mock_service.list_sessions.return_value = session_ids
        
        def mock_get_progress(session_id):
            return AnalysisProgress(session_id=session_id)
        
        mock_service.get_progress.side_effect = mock_get_progress
        mock_service.get_result.return_value = None
        
        response = client.get("/api/v1/analysis/sessions?page=2&page_size=10")
        
        assert response.status_code == 200
        data = response.json()
        assert data["page"] == 2
        assert data["page_size"] == 10
        assert data["total_count"] == 25
        assert data["has_previous"] is True
        assert data["has_next"] is True
    
    @patch('src.api.routes.analysis.analysis_service')
    def test_get_analysis_session(self, mock_service, client):
        """Test getting specific analysis session"""
        session_id = "test-session"
        mock_progress = AnalysisProgress(session_id=session_id)
        mock_progress.update_step("Processing")
        mock_service.get_progress.return_value = mock_progress
        
        response = client.get(f"/api/v1/analysis/sessions/{session_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == session_id
        assert data["current_step"] == "Processing"
    
    @patch('src.api.routes.analysis.analysis_service')
    def test_get_analysis_session_not_found(self, mock_service, client):
        """Test getting non-existent analysis session"""
        mock_service.get_progress.return_value = None
        
        response = client.get("/api/v1/analysis/sessions/nonexistent")
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]
    
    @patch('src.api.routes.analysis.analysis_service')
    def test_delete_analysis_session(self, mock_service, client):
        """Test deleting analysis session"""
        session_id = "test-session"
        mock_service.cleanup_session.return_value = True
        
        response = client.delete(f"/api/v1/analysis/sessions/{session_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert "deleted successfully" in data["message"]
    
    @patch('src.api.routes.analysis.analysis_service')
    def test_delete_analysis_session_not_found(self, mock_service, client):
        """Test deleting non-existent analysis session"""
        mock_service.cleanup_session.return_value = False
        
        response = client.delete("/api/v1/analysis/sessions/nonexistent")
        
        assert response.status_code == 404

class TestResultEndpoints:
    """Test analysis result endpoints"""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    @pytest.fixture
    def mock_analysis_result(self):
        """Create mock analysis result"""
        result = Mock()
        result.session_id = "test-session"
        result.components = []
        result.ml_analysis = None
        result.graph_analysis = None
        result.quality_metrics = {
            "average_complexity": 2.5,
            "high_complexity_components": 1,
            "testability_rate": 0.8,
            "documentation_rate": 0.6,
            "total_lines_of_code": 100,
            "maintainability_score": 85.0
        }
        return result
    
    @patch('src.api.routes.analysis.analysis_service')
    def test_get_session_components(self, mock_service, client, mock_analysis_result):
        """Test getting session components"""
        mock_service.get_result.return_value = mock_analysis_result
        
        response = client.get("/api/v1/analysis/sessions/test-session/components")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    @patch('src.api.routes.analysis.analysis_service')
    def test_get_session_components_with_filters(self, mock_service, client, mock_analysis_result):
        """Test getting session components with filters"""
        mock_service.get_result.return_value = mock_analysis_result
        
        response = client.get(
            "/api/v1/analysis/sessions/test-session/components"
            "?component_type=function&min_complexity=2&max_complexity=10&limit=50"
        )
        
        assert response.status_code == 200
    
    @patch('src.api.routes.analysis.analysis_service')
    def test_get_session_patterns(self, mock_service, client, mock_analysis_result):
        """Test getting session patterns"""
        mock_service.get_result.return_value = mock_analysis_result
        
        response = client.get("/api/v1/analysis/sessions/test-session/patterns")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    @patch('src.api.routes.analysis.analysis_service')
    def test_get_session_quality_metrics(self, mock_service, client, mock_analysis_result):
        """Test getting session quality metrics"""
        mock_service.get_result.return_value = mock_analysis_result
        
        response = client.get("/api/v1/analysis/sessions/test-session/quality")
        
        assert response.status_code == 200
        data = response.json()
        assert data["average_complexity"] == 2.5
        assert data["testability_rate"] == 0.8
    
    @patch('src.api.routes.analysis.analysis_service')
    def test_get_session_ml_analysis(self, mock_service, client, mock_analysis_result):
        """Test getting session ML analysis"""
        mock_service.get_result.return_value = mock_analysis_result
        
        response = client.get("/api/v1/analysis/sessions/test-session/ml-analysis")
        
        assert response.status_code == 200
        data = response.json()
        assert "clusters" in data
        assert "anomalies" in data
        assert "detected_patterns" in data
    
    @patch('src.api.routes.analysis.analysis_service')
    def test_get_session_graph_analysis(self, mock_service, client, mock_analysis_result):
        """Test getting session graph analysis"""
        mock_service.get_result.return_value = mock_analysis_result
        
        response = client.get("/api/v1/analysis/sessions/test-session/graph-analysis")
        
        assert response.status_code == 200
        data = response.json()
        assert "centrality_analysis" in data
        assert "detected_cycles" in data
        assert "architectural_layers" in data
    
    @patch('src.api.routes.analysis.analysis_service')
    def test_get_session_summary(self, mock_service, client, mock_analysis_result):
        """Test getting session summary"""
        mock_analysis_result.repository_path = "/test/repo"
        mock_analysis_result.total_files = 10
        mock_analysis_result.analyzed_files = 8
        mock_analysis_result.total_components = 25
        mock_analysis_result.analysis_duration = 45.2
        mock_analysis_result.complexity_stats = {"mean": 2.5}
        mock_analysis_result.testability_stats = {"average_score": 0.8}
        mock_analysis_result.pattern_summary = {"design_patterns": 3}
        mock_analysis_result.repository_analysis = None
        
        mock_service.get_result.return_value = mock_analysis_result
        
        response = client.get("/api/v1/analysis/sessions/test-session/summary")
        
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "test-session"
        assert data["total_files"] == 10
        assert data["analysis_duration"] == 45.2
    
    @patch('src.api.routes.analysis.analysis_service')
    def test_result_endpoints_session_not_found(self, mock_service, client):
        """Test result endpoints with non-existent session"""
        mock_service.get_result.return_value = None
        
        endpoints = [
            "/components",
            "/patterns", 
            "/quality",
            "/ml-analysis",
            "/graph-analysis",
            "/summary"
        ]
        
        for endpoint in endpoints:
            response = client.get(f"/api/v1/analysis/sessions/nonexistent{endpoint}")
            assert response.status_code == 404

class TestAPIErrorHandling:
    """Test API error handling"""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    def test_invalid_session_id_format(self, client):
        """Test various invalid session ID formats"""
        invalid_ids = ["", "   ", "invalid/id", "id with spaces"]
        
        for invalid_id in invalid_ids:
            response = client.get(f"/api/v1/analysis/sessions/{invalid_id}")
            # Should not cause server error, might be 404 or 422
            assert response.status_code in [404, 422]
    
    @patch('src.api.routes.analysis.analysis_service')
    def test_service_exception_handling(self, mock_service, client):
        """Test handling of service exceptions"""
        mock_service.get_progress.side_effect = Exception("Service error")
        
        response = client.get("/api/v1/analysis/sessions/test-session")
        
        assert response.status_code == 500
    
    def test_request_validation_errors(self, client):
        """Test request validation error handling"""
        # Invalid request body
        response = client.post("/api/v1/analysis/repository", json={})
        
        assert response.status_code == 422  # Validation error

class TestAPIPerformance:
    """Test API performance characteristics"""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    @patch('src.api.routes.analysis.analysis_service')
    def test_status_endpoint_performance(self, mock_service, client):
        """Test status endpoint response time"""
        mock_service.list_sessions.return_value = []
        
        import time
        start_time = time.time()
        response = client.get("/api/v1/analysis/status")
        end_time = time.time()
        
        assert response.status_code == 200
        assert (end_time - start_time) < 1.0  # Should respond within 1 second
    
    @patch('src.api.routes.analysis.analysis_service')
    def test_large_session_list_performance(self, mock_service, client):
        """Test performance with large number of sessions"""
        # Mock 1000 sessions
        session_ids = [f"session-{i}" for i in range(1000)]
        mock_service.list_sessions.return_value = session_ids
        
        def mock_get_progress(session_id):
            return AnalysisProgress(session_id=session_id)
        
        mock_service.get_progress.side_effect = mock_get_progress
        mock_service.get_result.return_value = None
        
        import time
        start_time = time.time()
        response = client.get("/api/v1/analysis/sessions?page_size=50")
        end_time = time.time()
        
        assert response.status_code == 200
        assert (end_time - start_time) < 5.0  # Should handle pagination efficiently

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
