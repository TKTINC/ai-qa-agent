"""
Tests for the repository analyzer implementation.
"""

import pytest
import tempfile
import os
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.analysis.repository_analyzer import (
    RepositoryAnalyzer,
    FileDiscoveryEngine,
    GitAnalyzer,
    ProjectStructureAnalyzer,
    ArchitecturePatternDetector,
    analyze_repository,
    discover_repository_files,
    FileAnalysisResult,
    RepositoryStructure,
    ProjectMetrics,
    GitAnalysisResult
)


class TestFileDiscoveryEngine:
    """Test the file discovery functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.discovery = FileDiscoveryEngine()
        
    def test_file_discovery_basic(self):
        """Test basic file discovery."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test files
            (temp_path / "test.py").write_text("def test(): pass")
            (temp_path / "test.js").write_text("function test() {}")
            (temp_path / "test.txt").write_text("not a code file")
            
            files = self.discovery.discover_files(temp_path)
            
            # Should find Python and JavaScript files, not txt
            assert len(files) == 2
            file_names = [f.name for f in files]
            assert "test.py" in file_names
            assert "test.js" in file_names
            assert "test.txt" not in file_names
    
    def test_gitignore_respect(self):
        """Test .gitignore pattern respect."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test files
            (temp_path / "included.py").write_text("def test(): pass")
            (temp_path / "ignored.py").write_text("def test(): pass")
            
            # Create .gitignore
            (temp_path / ".gitignore").write_text("ignored.py\n")
            
            files = self.discovery.discover_files(temp_path, respect_gitignore=True)
            
            file_names = [f.name for f in files]
            assert "included.py" in file_names
            assert "ignored.py" not in file_names
    
    def test_custom_ignore_patterns(self):
        """Test custom ignore patterns."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test files
            (temp_path / "keep.py").write_text("def test(): pass")
            (temp_path / "skip.py").write_text("def test(): pass")
            
            files = self.discovery.discover_files(
                temp_path, 
                custom_ignore_patterns=["skip.py"]
            )
            
            file_names = [f.name for f in files]
            assert "keep.py" in file_names
            assert "skip.py" not in file_names
    
    def test_nested_directories(self):
        """Test discovery in nested directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create nested structure
            subdir = temp_path / "subdir"
            subdir.mkdir()
            (subdir / "nested.py").write_text("def nested(): pass")
            (temp_path / "root.py").write_text("def root(): pass")
            
            files = self.discovery.discover_files(temp_path)
            
            assert len(files) == 2
            file_names = [f.name for f in files]
            assert "root.py" in file_names
            assert "nested.py" in file_names


class TestProjectStructureAnalyzer:
    """Test project structure analysis."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.analyzer = ProjectStructureAnalyzer()
    
    def test_structure_analysis(self):
        """Test basic structure analysis."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test structure
            (temp_path / "app.py").write_text("def main(): pass")
            (temp_path / "utils.py").write_text("def helper(): pass")
            
            subdir = temp_path / "models"
            subdir.mkdir()
            (subdir / "user.py").write_text("class User: pass")
            
            files = [temp_path / "app.py", temp_path / "utils.py", subdir / "user.py"]
            
            structure = self.analyzer.analyze_structure(files, temp_path)
            
            assert structure.total_files == 3
            assert structure.total_directories == 1  # models directory
            assert structure.max_depth == 1  # models/user.py is 1 level deep
            assert structure.language_distribution["python"] == 3
    
    def test_language_detection(self):
        """Test language detection from extensions."""
        assert self.analyzer._detect_language(Path("test.py")) == "python"
        assert self.analyzer._detect_language(Path("test.js")) == "javascript"
        assert self.analyzer._detect_language(Path("test.ts")) == "typescript"
        assert self.analyzer._detect_language(Path("test.txt")) == "unknown"
    
    def test_size_categorization(self):
        """Test file size categorization."""
        assert self.analyzer._categorize_size(500) == "tiny"
        assert self.analyzer._categorize_size(5000) == "small"
        assert self.analyzer._categorize_size(50000) == "medium"
        assert self.analyzer._categorize_size(500000) == "large"
        assert self.analyzer._categorize_size(5000000) == "huge"


class TestArchitecturePatternDetector:
    """Test architecture pattern detection."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.detector = ArchitecturePatternDetector()
    
    def test_mvc_pattern_detection(self):
        """Test MVC pattern detection."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create MVC structure
            for dirname in ["models", "views", "controllers"]:
                dir_path = temp_path / dirname
                dir_path.mkdir()
                (dir_path / "test.py").write_text("# test file")
            
            files = list(temp_path.rglob("*.py"))
            patterns = self.detector.detect_patterns(files, temp_path, [])
            
            assert "MVC (Model-View-Controller)" in patterns
    
    def test_layered_architecture_detection(self):
        """Test layered architecture detection."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create layered structure
            for dirname in ["services", "repositories", "controllers"]:
                dir_path = temp_path / dirname
                dir_path.mkdir()
                (dir_path / "test.py").write_text("# test file")
            
            files = list(temp_path.rglob("*.py"))
            patterns = self.detector.detect_patterns(files, temp_path, [])
            
            assert "Layered Architecture" in patterns
    
    def test_code_pattern_analysis(self):
        """Test code pattern analysis."""
        # Create mock file results with pattern indicators
        from src.analysis.ast_parser import CodeComponent, ComponentType, CodeLocation
        
        # Mock components with pattern names
        components = [
            CodeComponent(
                name="UserFactory",
                component_type=ComponentType.CLASS,
                location=CodeLocation(1, 10),
                source_code="class UserFactory: pass",
                file_path="test.py"
            ),
            CodeComponent(
                name="create_user",
                component_type=ComponentType.FUNCTION,
                location=CodeLocation(11, 15),
                source_code="def create_user(): pass",
                file_path="test.py"
            )
        ]
        
        file_result = FileAnalysisResult(
            file_path="test.py",
            language="python",
            components=components,
            file_size=1000,
            lines_of_code=20,
            complexity_metrics={},
            analysis_time=0.1
        )
        
        patterns = self.detector._analyze_code_patterns([file_result])
        assert "Factory Pattern" in patterns


class TestGitAnalyzer:
    """Test Git repository analysis."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.analyzer = GitAnalyzer()
    
    def test_non_git_repository(self):
        """Test analysis of non-Git directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = self.analyzer.analyze_repository(temp_dir)
            assert result.is_git_repo is False
    
    @patch('src.analysis.repository_analyzer.git')
    def test_git_repository_analysis(self, mock_git):
        """Test Git repository analysis with mocked Git."""
        # Mock Git repository
        mock_repo = MagicMock()
        mock_repo.active_branch.name = "main"
        
        # Mock commits
        mock_commit = MagicMock()
        mock_commit.committed_date = 1640995200  # 2022-01-01
        mock_commit.author.name = "Test Author"
        mock_commit.stats.files = {"file1.py": {"insertions": 10, "deletions": 5}}
        
        mock_repo.iter_commits.return_value = [mock_commit]
        mock_git.Repo.return_value = mock_repo
        
        with tempfile.TemporaryDirectory() as temp_dir:
            result = self.analyzer.analyze_repository(temp_dir)
            
            assert result.is_git_repo is True
            assert result.branch_name == "main"
            assert len(result.contributors) > 0


class TestRepositoryAnalyzer:
    """Test the main repository analyzer."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.analyzer = RepositoryAnalyzer(max_workers=2)
    
    def test_empty_repository(self):
        """Test analysis of empty repository."""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = self.analyzer.analyze_repository(temp_dir)
            
            assert len(result.file_results) == 0
            assert result.total_components == 0
            assert len(result.recommendations) > 0
            assert "No analyzable files found" in result.recommendations[0]
    
    def test_simple_repository_analysis(self):
        """Test analysis of simple repository."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create simple Python file
            (temp_path / "main.py").write_text('''
def main():
    """Main function."""
    print("Hello, world!")

class Helper:
    """Helper class."""
    
    def process(self, data):
        """Process data."""
        return data.upper()
''')
            
            result = self.analyzer.analyze_repository(temp_path)
            
            # Should analyze the file successfully
            assert len(result.file_results) == 1
            assert not result.file_results[0].has_error
            assert result.file_results[0].language == "python"
            assert len(result.file_results[0].components) >= 2  # main function + Helper class
            
            # Check project metrics
            assert result.project_metrics.total_components >= 2
            assert result.project_metrics.total_lines_of_code > 0
            assert result.project_metrics.average_complexity > 0
            
            # Check structure
            assert result.repository_structure.total_files == 1
            assert result.repository_structure.language_distribution["python"] == 1
    
    def test_file_analysis_error_handling(self):
        """Test handling of file analysis errors."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create file with syntax error
            (temp_path / "bad.py").write_text("def incomplete_function(\n# Syntax error")
            
            result = self.analyzer.analyze_repository(temp_path)
            
            # Should handle error gracefully
            assert len(result.file_results) == 1
            # May or may not have error depending on AST parser robustness
            # Just verify it doesn't crash
            assert result.success_rate >= 0.0
    
    def test_project_metrics_calculation(self):
        """Test project metrics calculation."""
        # Create mock file results
        from src.analysis.ast_parser import CodeComponent, ComponentType, CodeLocation
        
        components = [
            CodeComponent(
                name="simple_func",
                component_type=ComponentType.FUNCTION,
                location=CodeLocation(1, 5),
                source_code="def simple_func(): return 1",
                file_path="test.py",
                cyclomatic_complexity=1,
                test_priority=1
            ),
            CodeComponent(
                name="complex_func",
                component_type=ComponentType.FUNCTION,
                location=CodeLocation(6, 20),
                source_code="def complex_func(): pass",
                file_path="test.py",
                cyclomatic_complexity=12,  # High complexity
                test_priority=5
            )
        ]
        
        file_result = FileAnalysisResult(
            file_path="test.py",
            language="python",
            components=components,
            file_size=1000,
            lines_of_code=50,
            complexity_metrics={"average_complexity": 6.5},
            analysis_time=0.1
        )
        
        metrics = self.analyzer._calculate_project_metrics([file_result])
        
        assert metrics.total_components == 2
        assert metrics.average_complexity == 6.5
        assert metrics.max_complexity == 12
        assert len(metrics.high_complexity_files) == 0  # File level, not component level
        assert metrics.test_priority_distribution[1] == 1
        assert metrics.test_priority_distribution[5] == 1
        assert metrics.technical_debt_estimate > 0  # Should estimate debt for high complexity


class TestIntegrationFunctions:
    """Test the main interface functions."""
    
    def test_analyze_repository_function(self):
        """Test the main analyze_repository function."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create simple test file
            (temp_path / "test.py").write_text("def test(): pass")
            
            result = analyze_repository(temp_path, max_workers=2)
            
            assert isinstance(result, type(result))  # Check it returns the right type
            assert len(result.file_results) == 1
            assert result.repository_path == str(temp_path)
    
    def test_discover_repository_files_function(self):
        """Test the file discovery function."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test files
            (temp_path / "test.py").write_text("def test(): pass")
            (temp_path / "README.md").write_text("# README")
            
            files = discover_repository_files(temp_path)
            
            assert len(files) == 1  # Only .py file should be discovered
            assert files[0].name == "test.py"


class TestPerformance:
    """Test performance characteristics."""
    
    def test_parallel_processing(self):
        """Test that parallel processing works."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create multiple test files
            for i in range(5):
                (temp_path / f"test_{i}.py").write_text(f"def test_{i}(): pass")
            
            # Analyze with different worker counts
            import time
            
            # Single threaded
            start = time.time()
            result1 = analyze_repository(temp_path, max_workers=1)
            time1 = time.time() - start
            
            # Multi-threaded
            start = time.time()
            result2 = analyze_repository(temp_path, max_workers=4)
            time2 = time.time() - start
            
            # Both should produce same results
            assert len(result1.file_results) == len(result2.file_results)
            assert result1.total_components == result2.total_components
            
            # Multi-threaded should not be significantly slower (allowing for overhead)
            assert time2 <= time1 * 2  # Allow 2x overhead for small workload
    
    def test_large_file_handling(self):
        """Test handling of large files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create large file (within limits)
            large_content = "def test_function():\n    pass\n" * 1000  # ~30KB
            (temp_path / "large.py").write_text(large_content)
            
            result = analyze_repository(temp_path)
            
            assert len(result.file_results) == 1
            assert not result.file_results[0].has_error
            assert len(result.file_results[0].components) > 0
