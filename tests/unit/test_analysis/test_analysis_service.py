"""
Comprehensive tests for Analysis Service
Tests integration of all analysis components with async operations
"""

import pytest
import asyncio
import tempfile
import shutil
import zipfile
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

from src.analysis.analysis_service import (
    AnalysisService, 
    AnalysisProgress,
    ComprehensiveAnalysisResult,
    analysis_service
)
from src.analysis.ast_parser import Language

class TestAnalysisService:
    """Test Analysis Service functionality"""
    
    @pytest.fixture
    def analysis_service_instance(self):
        """Create fresh analysis service for testing"""
        return AnalysisService()
    
    @pytest.fixture
    def sample_repository(self):
        """Create temporary repository for testing"""
        temp_dir = tempfile.mkdtemp()
        repo_path = Path(temp_dir) / "test_repo"
        repo_path.mkdir()
        
        # Create sample Python file
        sample_file = repo_path / "sample.py"
        sample_file.write_text('''
def simple_function(x: int) -> int:
    """Simple test function"""
    return x * 2

class TestClass:
    """Test class"""
    
    def method(self, y: str) -> str:
        return f"Hello {y}"
        ''')
        
        yield str(repo_path)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_archive(self, sample_repository):
        """Create sample archive for upload testing"""
        temp_dir = tempfile.mkdtemp()
        archive_path = Path(temp_dir) / "test_repo.zip"
        
        with zipfile.ZipFile(archive_path, 'w') as zipf:
            repo_path = Path(sample_repository)
            for file_path in repo_path.rglob('*'):
                if file_path.is_file():
                    arcname = file_path.relative_to(repo_path.parent)
                    zipf.write(file_path, arcname)
        
        with open(archive_path, 'rb') as f:
            content = f.read()
        
        shutil.rmtree(temp_dir)
        return content, "test_repo.zip"

class TestAnalysisProgress:
    """Test AnalysisProgress functionality"""
    
    def test_progress_initialization(self):
        """Test progress tracker initialization"""
        progress = AnalysisProgress(session_id="test-123")
        
        assert progress.session_id == "test-123"
        assert progress.total_steps == 7
        assert progress.completed_steps == 0
        assert progress.current_step == "Initializing"
        assert progress.progress_percentage == 0.0
        assert progress.error_message is None
    
    def test_progress_update_step(self):
        """Test progress step updates"""
        progress = AnalysisProgress(session_id="test-123")
        
        progress.update_step("Processing files")
        
        assert progress.completed_steps == 1
        assert progress.current_step == "Processing files"
        assert progress.progress_percentage > 0
    
    def test_progress_error_handling(self):
        """Test progress error marking"""
        progress = AnalysisProgress(session_id="test-123")
        
        progress.mark_error("Test error")
        
        assert progress.error_message == "Test error"
        assert "Failed:" in progress.current_step

class TestAnalysisServiceCore:
    """Test core Analysis Service functionality"""
    
    @pytest.mark.asyncio
    async def test_start_analysis_basic(self, analysis_service_instance, sample_repository):
        """Test basic analysis startup"""
        session_id = "test-session-1"
        
        # Mock the comprehensive analysis to avoid long execution
        with patch.object(analysis_service_instance, '_run_comprehensive_analysis') as mock_analysis:
            mock_analysis.return_value = None
            
            progress = await analysis_service_instance.start_analysis(
                session_id=session_id,
                repository_path=sample_repository,
                language=Language.PYTHON
            )
            
            assert progress.session_id == session_id
            assert progress.start_time > 0
            assert session_id in analysis_service_instance._progress_cache
    
    @pytest.mark.asyncio
    async def test_start_analysis_nonexistent_path(self, analysis_service_instance):
        """Test analysis with non-existent repository path"""
        with pytest.raises(Exception):  # Should raise AnalysisError
            await analysis_service_instance.start_analysis(
                session_id="test-fail",
                repository_path="/nonexistent/path",
                language=Language.PYTHON
            )
    
    def test_progress_tracking(self, analysis_service_instance):
        """Test progress tracking functionality"""
        session_id = "test-progress"
        progress = AnalysisProgress(session_id=session_id)
        
        analysis_service_instance._progress_cache[session_id] = progress
        
        retrieved_progress = analysis_service_instance.get_progress(session_id)
        assert retrieved_progress is not None
        assert retrieved_progress.session_id == session_id
    
    def test_result_storage_retrieval(self, analysis_service_instance):
        """Test result storage and retrieval"""
        session_id = "test-result"
        mock_result = Mock()
        mock_result.session_id = session_id
        
        analysis_service_instance._result_cache[session_id] = mock_result
        
        retrieved_result = analysis_service_instance.get_result(session_id)
        assert retrieved_result is not None
        assert retrieved_result.session_id == session_id
    
    def test_session_cleanup(self, analysis_service_instance):
        """Test session cleanup functionality"""
        session_id = "test-cleanup"
        
        # Add session data
        analysis_service_instance._progress_cache[session_id] = Mock()
        analysis_service_instance._result_cache[session_id] = Mock()
        
        # Cleanup
        success = analysis_service_instance.cleanup_session(session_id)
        
        assert success is True
        assert session_id not in analysis_service_instance._progress_cache
        assert session_id not in analysis_service_instance._result_cache
    
    def test_list_sessions(self, analysis_service_instance):
        """Test session listing"""
        session_ids = ["session-1", "session-2", "session-3"]
        
        for session_id in session_ids:
            analysis_service_instance._progress_cache[session_id] = Mock()
        
        listed_sessions = analysis_service_instance.list_sessions()
        
        assert len(listed_sessions) == 3
        assert all(sid in listed_sessions for sid in session_ids)

class TestAnalysisServiceIntegration:
    """Test Analysis Service integration with components"""
    
    @pytest.mark.asyncio
    @patch('src.analysis.analysis_service.RepositoryAnalyzer')
    @patch('src.analysis.analysis_service.MLPatternDetector')
    @patch('src.analysis.analysis_service.GraphPatternAnalyzer')
    async def test_comprehensive_analysis_flow(
        self, 
        mock_graph_analyzer, 
        mock_ml_detector, 
        mock_repo_analyzer,
        analysis_service_instance,
        sample_repository
    ):
        """Test complete analysis flow with mocked components"""
        session_id = "test-integration"
        
        # Mock repository analyzer
        mock_repo_result = Mock()
        mock_repo_result.file_analyses = []
        mock_repo_result.structure.total_size = 1000
        mock_repo_result.structure.max_depth = 3
        mock_repo_result.structure.file_types = {"py": 1}
        mock_repo_analyzer.return_value.analyze_repository.return_value = mock_repo_result
        
        # Mock ML detector
        mock_ml_result = Mock()
        mock_ml_result.detected_patterns = []
        mock_ml_result.anomalies = []
        mock_ml_result.clusters = []
        mock_ml_detector.return_value.analyze_components.return_value = mock_ml_result
        
        # Mock graph analyzer
        mock_graph_result = Mock()
        mock_graph_result.anti_patterns = []
        mock_graph_result.architectural_patterns = []
        mock_graph_result.cycles = []
        mock_graph_result.layers = []
        mock_graph_result.centrality_analysis = []
        mock_graph_result.modularity_score = 0.5
        mock_graph_analyzer.return_value.analyze_dependencies.return_value = mock_graph_result
        
        # Start analysis
        progress = await analysis_service_instance.start_analysis(
            session_id=session_id,
            repository_path=sample_repository,
            language=Language.PYTHON
        )
        
        # Wait for completion (mocked, should be fast)
        await asyncio.sleep(0.1)
        
        assert progress.session_id == session_id
        assert session_id in analysis_service_instance._progress_cache
    
    @pytest.mark.asyncio
    async def test_upload_analysis_zip(self, analysis_service_instance, sample_archive):
        """Test upload and analysis of ZIP archive"""
        archive_content, filename = sample_archive
        session_id = "test-upload"
        
        # Mock the analysis to avoid long execution
        with patch.object(analysis_service_instance, 'start_analysis') as mock_start:
            mock_progress = AnalysisProgress(session_id=session_id)
            mock_start.return_value = mock_progress
            
            progress = await analysis_service_instance.analyze_uploaded_archive(
                session_id=session_id,
                archive_content=archive_content,
                filename=filename,
                language=Language.PYTHON
            )
            
            assert progress.session_id == session_id
            mock_start.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_upload_analysis_unsupported_format(self, analysis_service_instance):
        """Test upload with unsupported file format"""
        with pytest.raises(Exception):  # Should raise AnalysisError
            await analysis_service_instance.analyze_uploaded_archive(
                session_id="test-fail",
                archive_content=b"dummy content",
                filename="unsupported.txt",
                language=Language.PYTHON
            )

class TestAnalysisServiceQualityMetrics:
    """Test quality metrics calculation"""
    
    def test_calculate_quality_metrics_empty(self, analysis_service_instance):
        """Test quality metrics with empty components"""
        metrics = analysis_service_instance._calculate_quality_metrics(
            components=[], 
            repo_result=Mock(), 
            ml_result=None, 
            graph_result=None
        )
        
        assert metrics == {}
    
    def test_calculate_complexity_stats_empty(self, analysis_service_instance):
        """Test complexity stats with empty components"""
        stats = analysis_service_instance._calculate_complexity_stats([])
        
        assert stats == {}
    
    def test_calculate_testability_stats_empty(self, analysis_service_instance):
        """Test testability stats with empty components"""
        stats = analysis_service_instance._calculate_testability_stats([])
        
        assert stats == {}
    
    def test_calculate_pattern_summary_empty(self, analysis_service_instance):
        """Test pattern summary with no results"""
        summary = analysis_service_instance._calculate_pattern_summary(None, None)
        
        expected = {
            "design_patterns": 0,
            "anti_patterns": 0,
            "anomalies": 0,
            "architectural_patterns": 0
        }
        assert summary == expected

class TestAnalysisServiceArchiveHandling:
    """Test archive handling functionality"""
    
    def test_find_repository_root_with_git(self, analysis_service_instance):
        """Test finding repository root with .git directory"""
        temp_dir = tempfile.mkdtemp()
        try:
            extraction_dir = Path(temp_dir)
            git_dir = extraction_dir / ".git"
            git_dir.mkdir()
            
            root = analysis_service_instance._find_repository_root(extraction_dir)
            
            assert root == extraction_dir
        finally:
            shutil.rmtree(temp_dir)
    
    def test_find_repository_root_with_setup_py(self, analysis_service_instance):
        """Test finding repository root with setup.py"""
        temp_dir = tempfile.mkdtemp()
        try:
            extraction_dir = Path(temp_dir)
            setup_file = extraction_dir / "setup.py"
            setup_file.touch()
            
            root = analysis_service_instance._find_repository_root(extraction_dir)
            
            assert root == extraction_dir
        finally:
            shutil.rmtree(temp_dir)
    
    def test_find_repository_root_nested(self, analysis_service_instance):
        """Test finding repository root in nested structure"""
        temp_dir = tempfile.mkdtemp()
        try:
            extraction_dir = Path(temp_dir)
            nested_dir = extraction_dir / "project-main"
            nested_dir.mkdir()
            setup_file = nested_dir / "setup.py"
            setup_file.touch()
            
            root = analysis_service_instance._find_repository_root(extraction_dir)
            
            assert root == nested_dir
        finally:
            shutil.rmtree(temp_dir)

class TestGlobalAnalysisService:
    """Test global analysis service instance"""
    
    def test_global_service_exists(self):
        """Test that global analysis service is available"""
        assert analysis_service is not None
        assert isinstance(analysis_service, AnalysisService)
    
    def test_global_service_functionality(self):
        """Test basic functionality of global service"""
        sessions = analysis_service.list_sessions()
        assert isinstance(sessions, list)

# Performance and stress tests
class TestAnalysisServicePerformance:
    """Test Analysis Service performance characteristics"""
    
    def test_memory_usage_session_tracking(self, analysis_service_instance):
        """Test memory usage doesn't grow unbounded with sessions"""
        initial_sessions = len(analysis_service_instance.list_sessions())
        
        # Add many sessions
        for i in range(100):
            session_id = f"perf-test-{i}"
            progress = AnalysisProgress(session_id=session_id)
            analysis_service_instance._progress_cache[session_id] = progress
        
        assert len(analysis_service_instance.list_sessions()) == initial_sessions + 100
        
        # Cleanup all test sessions
        for i in range(100):
            analysis_service_instance.cleanup_session(f"perf-test-{i}")
        
        assert len(analysis_service_instance.list_sessions()) == initial_sessions
    
    @pytest.mark.asyncio
    async def test_concurrent_session_handling(self, analysis_service_instance):
        """Test handling multiple concurrent sessions"""
        session_ids = [f"concurrent-{i}" for i in range(5)]
        
        # Create concurrent sessions
        for session_id in session_ids:
            progress = AnalysisProgress(session_id=session_id)
            analysis_service_instance._progress_cache[session_id] = progress
        
        # Verify all sessions exist
        listed_sessions = analysis_service_instance.list_sessions()
        for session_id in session_ids:
            assert session_id in listed_sessions
        
        # Cleanup
        for session_id in session_ids:
            analysis_service_instance.cleanup_session(session_id)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
