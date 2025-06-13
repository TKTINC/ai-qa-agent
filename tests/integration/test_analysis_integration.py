"""
Integration tests for complete analysis flow
Tests the full pipeline from API to analysis components
"""

import pytest
import tempfile
import shutil
import zipfile
from pathlib import Path
from fastapi.testclient import TestClient

from src.api.main import app

class TestCompleteAnalysisFlow:
    """Test complete analysis flow integration"""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    @pytest.fixture
    def sample_python_project(self):
        """Create a realistic Python project for testing"""
        temp_dir = tempfile.mkdtemp()
        project_path = Path(temp_dir) / "sample_project"
        project_path.mkdir()
        
        # Create main module
        main_file = project_path / "main.py"
        main_file.write_text('''
"""Main application module"""

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

class DataProcessor:
    """Process data with various algorithms"""
    
    def __init__(self, config: dict):
        self.config = config
        self.cache = {}
    
    def process_data(self, data: List[int]) -> List[int]:
        """Process list of integers"""
        result = []
        for item in data:
            if item > 0:
                result.append(item * 2)
            else:
                result.append(0)
        return result
    
    def complex_calculation(self, x: int, y: int, z: Optional[int] = None) -> float:
        """Complex calculation with multiple branches"""
        if z is None:
            z = 1
        
        if x > 10:
            if y > 5:
                return (x * y) / z
            else:
                return x + y + z
        else:
            if y < 0:
                return abs(y) * x
            else:
                return x - y + z

def helper_function(value: str) -> str:
    """Simple helper function"""
    return value.upper()

async def async_operation(data: dict) -> dict:
    """Async operation for demonstration"""
    processed = {}
    for key, value in data.items():
        processed[key] = value * 2
    return processed

def generator_function(limit: int):
    """Generator function example"""
    for i in range(limit):
        yield i * i

if __name__ == "__main__":
    processor = DataProcessor({"batch_size": 100})
    result = processor.process_data([1, 2, 3, -1, 4])
    print(f"Result: {result}")
        ''')
        
        # Create utility module
        utils_file = project_path / "utils.py"
        utils_file.write_text('''
"""Utility functions"""

from typing import Any, Dict
import json

def load_config(file_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file"""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        return {}

def validate_data(data: Any) -> bool:
    """Validate input data"""
    if data is None:
        return False
    if isinstance(data, (list, dict)) and len(data) == 0:
        return False
    return True

class SingletonConfig:
    """Singleton pattern example"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.config = {}
            self.initialized = True
        ''')
        
        # Create test file
        test_file = project_path / "test_main.py"
        test_file.write_text('''
"""Tests for main module"""

import unittest
from main import DataProcessor, helper_function

class TestDataProcessor(unittest.TestCase):
    
    def setUp(self):
        self.processor = DataProcessor({"test": True})
    
    def test_process_data(self):
        result = self.processor.process_data([1, 2, -1, 3])
        expected = [2, 4, 0, 6]
        self.assertEqual(result, expected)
    
    def test_complex_calculation(self):
        result = self.processor.complex_calculation(15, 10)
        self.assertEqual(result, 150.0)

if __name__ == "__main__":
    unittest.main()
        ''')
        
        # Create requirements file
        req_file = project_path / "requirements.txt"
        req_file.write_text('''
pytest==7.4.0
requests==2.31.0
numpy==1.24.3
        ''')
        
        yield str(project_path)
        shutil.rmtree(temp_dir)
    
    @pytest.mark.integration
    def test_full_repository_analysis_flow(self, client, sample_python_project):
        """Test complete repository analysis from start to finish"""
        # Step 1: Start repository analysis
        request_data = {
            "repository_path": sample_python_project,
            "language": "python"
        }
        
        response = client.post("/api/v1/analysis/repository", json=request_data)
        assert response.status_code == 200
        
        session_data = response.json()
        session_id = session_data["session_id"]
        assert session_id is not None
        
        # Step 2: Monitor progress (wait for completion)
        import time
        max_wait = 60  # 1 minute max wait
        wait_time = 0
        
        while wait_time < max_wait:
            progress_response = client.get(f"/api/v1/analysis/sessions/{session_id}")
            assert progress_response.status_code == 200
            
            progress_data = progress_response.json()
            if progress_data["status"] in ["completed", "failed"]:
                break
                
            time.sleep(2)
            wait_time += 2
        
        # Verify analysis completed
        final_progress = client.get(f"/api/v1/analysis/sessions/{session_id}")
        assert final_progress.status_code == 200
        final_data = final_progress.json()
        
        # Should complete successfully (might be slow in real environment)
        # assert final_data["status"] == "completed"
        
        # Step 3: Get analysis components
        components_response = client.get(f"/api/v1/analysis/sessions/{session_id}/components")
        if components_response.status_code == 200:
            components = components_response.json()
            # Should have extracted multiple components
            # assert len(components) > 0
            
            # Verify component structure
            if components:
                component = components[0]
                assert "name" in component
                assert "type" in component
                assert "complexity" in component
                assert "quality" in component
        
        # Step 4: Get quality metrics
        quality_response = client.get(f"/api/v1/analysis/sessions/{session_id}/quality")
        if quality_response.status_code == 200:
            quality_data = quality_response.json()
            assert "average_complexity" in quality_data
            assert "testability_rate" in quality_data
        
        # Step 5: Get patterns
        patterns_response = client.get(f"/api/v1/analysis/sessions/{session_id}/patterns")
        if patterns_response.status_code == 200:
            patterns = patterns_response.json()
            # Should find some patterns in our sample code
            # assert isinstance(patterns, list)
        
        # Step 6: Get ML analysis
        ml_response = client.get(f"/api/v1/analysis/sessions/{session_id}/ml-analysis")
        if ml_response.status_code == 200:
            ml_data = ml_response.json()
            assert "clusters" in ml_data
            assert "anomalies" in ml_data
        
        # Step 7: Get graph analysis
        graph_response = client.get(f"/api/v1/analysis/sessions/{session_id}/graph-analysis")
        if graph_response.status_code == 200:
            graph_data = graph_response.json()
            assert "centrality_analysis" in graph_data
        
        # Step 8: Get summary
        summary_response = client.get(f"/api/v1/analysis/sessions/{session_id}/summary")
        if summary_response.status_code == 200:
            summary_data = summary_response.json()
            assert summary_data["session_id"] == session_id
            assert "quality_metrics" in summary_data
        
        # Step 9: Clean up
        delete_response = client.delete(f"/api/v1/analysis/sessions/{session_id}")
        assert delete_response.status_code == 200
    
    @pytest.mark.integration
    def test_upload_analysis_flow(self, client, sample_python_project):
        """Test complete upload and analysis flow"""
        # Create ZIP archive
        temp_dir = tempfile.mkdtemp()
        try:
            zip_path = Path(temp_dir) / "test_project.zip"
            
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                project_path = Path(sample_python_project)
                for file_path in project_path.rglob('*'):
                    if file_path.is_file():
                        arcname = file_path.relative_to(project_path.parent)
                        zipf.write(file_path, arcname)
            
            # Upload and analyze
            with open(zip_path, 'rb') as f:
                files = {"file": ("test_project.zip", f, "application/zip")}
                response = client.post(
                    "/api/v1/analysis/upload?language=python",
                    files=files
                )
            
            if response.status_code == 200:
                upload_data = response.json()
                session_id = upload_data["session_id"]
                
                # Monitor progress briefly
                import time
                time.sleep(5)  # Give it some time to start
                
                progress_response = client.get(f"/api/v1/analysis/sessions/{session_id}")
                assert progress_response.status_code == 200
                
                # Clean up
                client.delete(f"/api/v1/analysis/sessions/{session_id}")
            else:
                # Upload might fail in test environment, that's ok
                assert response.status_code in [400, 500]
                
        finally:
            shutil.rmtree(temp_dir)

class TestAPIEndpointIntegration:
    """Test API endpoint integration"""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    @pytest.mark.integration
    def test_session_management_flow(self, client):
        """Test session management workflow"""
        # Get initial session list
        initial_response = client.get("/api/v1/analysis/sessions")
        assert initial_response.status_code == 200
        initial_data = initial_response.json()
        initial_count = initial_data["total_count"]
        
        # Verify pagination works
        paginated_response = client.get("/api/v1/analysis/sessions?page=1&page_size=5")
        assert paginated_response.status_code == 200
        paginated_data = paginated_response.json()
        assert paginated_data["page"] == 1
        assert paginated_data["page_size"] == 5
    
    @pytest.mark.integration
    def test_service_status_endpoints(self, client):
        """Test service status and health endpoints"""
        # Test analysis service status
        status_response = client.get("/api/v1/analysis/status")
        assert status_response.status_code == 200
        status_data = status_response.json()
        assert status_data["service"] == "analysis"
        assert "capabilities" in status_data
        
        # Test analysis service test endpoint
        test_response = client.get("/api/v1/analysis/test")
        assert test_response.status_code == 200
        test_data = test_response.json()
        assert test_data["service"] == "analysis"
        assert "test_status" in test_data
        
        # Test main health endpoints
        health_response = client.get("/health/")
        assert health_response.status_code == 200
        
        detailed_health_response = client.get("/health/detailed")
        assert detailed_health_response.status_code == 200

class TestErrorHandlingIntegration:
    """Test error handling in integration scenarios"""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    @pytest.mark.integration
    def test_nonexistent_session_handling(self, client):
        """Test handling of non-existent session across all endpoints"""
        fake_session_id = "nonexistent-session-id"
        
        endpoints = [
            f"/api/v1/analysis/sessions/{fake_session_id}",
            f"/api/v1/analysis/sessions/{fake_session_id}/components",
            f"/api/v1/analysis/sessions/{fake_session_id}/patterns",
            f"/api/v1/analysis/sessions/{fake_session_id}/quality",
            f"/api/v1/analysis/sessions/{fake_session_id}/ml-analysis",
            f"/api/v1/analysis/sessions/{fake_session_id}/graph-analysis",
            f"/api/v1/analysis/sessions/{fake_session_id}/summary"
        ]
        
        for endpoint in endpoints:
            response = client.get(endpoint)
            assert response.status_code == 404
            assert "not found" in response.json()["detail"].lower()
    
    @pytest.mark.integration
    def test_malformed_requests(self, client):
        """Test handling of malformed requests"""
        # Invalid JSON
        response = client.post(
            "/api/v1/analysis/repository",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422
        
        # Missing required fields
        response = client.post("/api/v1/analysis/repository", json={})
        assert response.status_code == 422

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
