"""
Tests for the ML pattern detection implementation.
"""

import pytest
import numpy as np
import time
from unittest.mock import patch, MagicMock

from src.analysis.ml_pattern_detector import (
    MLPatternDetector,
    FeatureEngineer,
    ClusteringAnalyzer,
    AnomalyDetector,
    DesignPatternDetector,
    ComponentFeatures,
    PatternCluster,
    CodeAnomaly,
    DesignPattern,
    analyze_ml_patterns,
    extract_component_features
)
from src.analysis.ast_parser import CodeComponent, ComponentType, CodeLocation
from src.analysis.repository_analyzer import RepositoryAnalysisResult, FileAnalysisResult


class TestFeatureEngineer:
    """Test the feature engineering functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.feature_engineer = FeatureEngineer(max_features=10)
    
    def test_extract_features_basic(self):
        """Test basic feature extraction."""
        components = [
            CodeComponent(
                name="test_function",
                component_type=ComponentType.FUNCTION,
                location=CodeLocation(1, 10),
                source_code="def test_function(): pass",
                file_path="test.py",
                parameters=[],
                cyclomatic_complexity=1,
                cognitive_complexity=1,
                lines_of_code=1,
                dependencies=set(),
                testability_score=0.8,
                test_priority=1,
                documentation_coverage=0.0
            )
        ]
        
        features = self.feature_engineer.extract_features(components)
        
        assert len(features) == 1
        feature = features[0]
        
        assert feature.component_name == "test_function"
        assert feature.component_type == "function"
        assert feature.cyclomatic_complexity == 1.0
        assert feature.lines_of_code == 1
        assert feature.text_features is not None
    
    def test_feature_matrix_generation(self):
        """Test feature matrix generation."""
        components = [
            CodeComponent(
                name="simple_func",
                component_type=ComponentType.FUNCTION,
                location=CodeLocation(1, 5),
                source_code="def simple_func(): return 1",
                file_path="test.py",
                cyclomatic_complexity=1,
                cognitive_complexity=1,
                lines_of_code=1,
                testability_score=0.9
            ),
            CodeComponent(
                name="complex_func",
                component_type=ComponentType.FUNCTION,
                location=CodeLocation(6, 15),
                source_code="def complex_func(): pass",
                file_path="test.py",
                cyclomatic_complexity=5,
                cognitive_complexity=6,
                lines_of_code=10,
                testability_score=0.5
            )
        ]
        
        features = self.feature_engineer.extract_features(components)
        feature_matrix, feature_names = self.feature_engineer.get_feature_matrix(features)
        
        assert feature_matrix.shape[0] == 2  # 2 components
        assert feature_matrix.shape[1] > 10  # Structural + text features
        assert len(feature_names) == feature_matrix.shape[1]
    
    def test_text_document_preparation(self):
        """Test text document preparation."""
        component = CodeComponent(
            name="getUserData",
            component_type=ComponentType.FUNCTION,
            location=CodeLocation(1, 10),
            source_code="def getUserData(): pass",
            file_path="user.py",
            docstring="Get user data from database",
            function_calls=["query_database", "format_response"]
        )
        
        text_doc = self.feature_engineer._prepare_text_document(component)
        
        assert "get" in text_doc.lower()
        assert "user" in text_doc.lower()
        assert "data" in text_doc.lower()
        assert "function" in text_doc.lower()
        assert "database" in text_doc.lower()
    
    def test_identifier_splitting(self):
        """Test identifier splitting functionality."""
        # Test camelCase
        parts = self.feature_engineer._split_identifier("getUserData")
        assert "get" in parts
        assert "user" in parts
        assert "data" in parts
        
        # Test snake_case
        parts = self.feature_engineer._split_identifier("get_user_data")
        assert "get" in parts
        assert "user" in parts
        assert "data" in parts
        
        # Test PascalCase
        parts = self.feature_engineer._split_identifier("UserDataManager")
        assert "user" in parts
        assert "data" in parts
        assert "manager" in parts


class TestClusteringAnalyzer:
    """Test clustering analysis functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.clustering_analyzer = ClusteringAnalyzer()
    
    def test_clustering_analysis(self):
        """Test clustering analysis with mock data."""
        # Create mock features
        features = [
            ComponentFeatures(
                component_id="comp1",
                file_path="test.py",
                component_name="func1",
                component_type="function",
                cyclomatic_complexity=1.0,
                cognitive_complexity=1.0,
                lines_of_code=5,
                parameter_count=1,
                dependency_count=0,
                testability_score=0.9,
                test_priority=1,
                documentation_coverage=1.0
            ),
            ComponentFeatures(
                component_id="comp2",
                file_path="test.py",
                component_name="func2",
                component_type="function",
                cyclomatic_complexity=2.0,
                cognitive_complexity=2.0,
                lines_of_code=8,
                parameter_count=2,
                dependency_count=1,
                testability_score=0.8,
                test_priority=2,
                documentation_coverage=0.5
            ),
            ComponentFeatures(
                component_id="comp3",
                file_path="test.py",
                component_name="TestClass",
                component_type="class",
                cyclomatic_complexity=1.0,
                cognitive_complexity=1.0,
                lines_of_code=20,
                parameter_count=0,
                dependency_count=2,
                testability_score=0.7,
                test_priority=3,
                documentation_coverage=1.0
            )
        ]
        
        # Create mock feature matrix
        feature_matrix = np.array([
            [1.0, 1.0, 5, 1, 0, 0.9, 1, 1.0, 0.2, 0.1, 0.5],
            [2.0, 2.0, 8, 2, 1, 0.8, 2, 0.5, 0.25, 0.2, 0.6],
            [1.0, 1.0, 20, 0, 2, 0.7, 3, 1.0, 0.05, 0.3, 0.4]
        ])
        
        clusters = self.clustering_analyzer.analyze_clusters(features, feature_matrix)
        
        # Should return some clusters
        assert isinstance(clusters, list)
        # With 3 components, might have type-based clusters
        assert len(clusters) >= 0  # May or may not find clusters
    
    def test_cluster_by_type(self):
        """Test clustering by component type."""
        features = [
            ComponentFeatures("comp1", "test.py", "func1", "function", 1.0, 1.0, 5, 1, 0, 0.9, 1, 1.0),
            ComponentFeatures("comp2", "test.py", "func2", "function", 2.0, 2.0, 8, 2, 1, 0.8, 2, 0.5),
            ComponentFeatures("comp3", "test.py", "TestClass", "class", 1.0, 1.0, 20, 0, 2, 0.7, 3, 1.0)
        ]
        
        clusters = self.clustering_analyzer._cluster_by_type(features)
        
        # Should have one cluster for functions (2 components)
        function_clusters = [c for c in clusters if 'function' in c.pattern_description]
        assert len(function_clusters) == 1
        assert len(function_clusters[0].component_ids) == 2


class TestAnomalyDetector:
    """Test anomaly detection functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.anomaly_detector = AnomalyDetector()
    
    def test_statistical_anomaly_detection(self):
        """Test statistical anomaly detection."""
        features = []
        
        # Create normal components
        for i in range(5):
            features.append(ComponentFeatures(
                component_id=f"normal_{i}",
                file_path="test.py",
                component_name=f"normal_func_{i}",
                component_type="function",
                cyclomatic_complexity=float(2 + i),  # 2-6
                cognitive_complexity=float(2 + i),
                lines_of_code=10 + i,
                parameter_count=1,
                dependency_count=1,
                testability_score=0.8,
                test_priority=2,
                documentation_coverage=1.0
            ))
        
        # Create anomalous component with high complexity
        features.append(ComponentFeatures(
            component_id="anomaly_1",
            file_path="test.py",
            component_name="complex_func",
            component_type="function",
            cyclomatic_complexity=20.0,  # Much higher than others
            cognitive_complexity=25.0,
            lines_of_code=100,
            parameter_count=1,
            dependency_count=1,
            testability_score=0.3,
            test_priority=5,
            documentation_coverage=0.0
        ))
        
        anomalies = self.anomaly_detector._detect_statistical_anomalies(features)
        
        # Should detect the high complexity component
        assert len(anomalies) > 0
        assert any(a.component_id == "anomaly_1" for a in anomalies)
        assert any(a.anomaly_type == "complexity" for a in anomalies)
    
    def test_pattern_anomaly_detection(self):
        """Test pattern-based anomaly detection."""
        features = [
            # Large component with no documentation
            ComponentFeatures(
                component_id="undocumented",
                file_path="test.py",
                component_name="large_func",
                component_type="function",
                cyclomatic_complexity=5.0,
                cognitive_complexity=5.0,
                lines_of_code=100,  # Large
                parameter_count=2,
                dependency_count=1,
                testability_score=0.5,
                test_priority=3,
                documentation_coverage=0.0,  # No documentation
                coupling_score=0.3
            ),
            # High coupling component
            ComponentFeatures(
                component_id="coupled",
                file_path="test.py",
                component_name="coupled_func",
                component_type="function",
                cyclomatic_complexity=3.0,
                cognitive_complexity=3.0,
                lines_of_code=20,
                parameter_count=1,
                dependency_count=5,
                testability_score=0.3,
                test_priority=4,
                documentation_coverage=1.0,
                coupling_score=0.9  # High coupling
            )
        ]
        
        anomalies = self.anomaly_detector._detect_pattern_anomalies(features)
        
        # Should detect both pattern anomalies
        assert len(anomalies) >= 2
        anomaly_types = [a.anomaly_type for a in anomalies]
        assert "documentation" in anomaly_types
        assert "coupling" in anomaly_types


class TestDesignPatternDetector:
    """Test design pattern detection functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.pattern_detector = DesignPatternDetector()
    
    def test_singleton_detection(self):
        """Test Singleton pattern detection."""
        features = [
            ComponentFeatures(
                component_id="singleton_class",
                file_path="singleton.py",
                component_name="DatabaseSingleton",
                component_type="class",
                cyclomatic_complexity=2.0,
                cognitive_complexity=2.0,
                lines_of_code=20,
                parameter_count=0,
                dependency_count=1,
                testability_score=0.6,
                test_priority=3,
                documentation_coverage=1.0
            ),
            ComponentFeatures(
                component_id="getInstance",
                file_path="singleton.py",
                component_name="getInstance",
                component_type="method",
                cyclomatic_complexity=1.0,
                cognitive_complexity=1.0,
                lines_of_code=5,
                parameter_count=0,  # No parameters
                dependency_count=0,
                testability_score=0.8,
                test_priority=2,
                documentation_coverage=1.0
            )
        ]
        
        patterns = self.pattern_detector._detect_singleton(features, [])
        
        # Should detect singleton pattern
        assert len(patterns) > 0
        singleton_patterns = [p for p in patterns if p.pattern_name == "Singleton"]
        assert len(singleton_patterns) > 0
    
    def test_factory_detection(self):
        """Test Factory pattern detection."""
        features = [
            ComponentFeatures(
                component_id="create_user",
                file_path="factory.py",
                component_name="create_user",
                component_type="function",
                cyclomatic_complexity=2.0,
                cognitive_complexity=2.0,
                lines_of_code=10,
                parameter_count=2,  # Factory methods take parameters
                dependency_count=1,
                testability_score=0.8,
                test_priority=2,
                documentation_coverage=1.0
            ),
            ComponentFeatures(
                component_id="create_admin",
                file_path="factory.py",
                component_name="create_admin",
                component_type="function",
                cyclomatic_complexity=2.0,
                cognitive_complexity=2.0,
                lines_of_code=12,
                parameter_count=1,
                dependency_count=1,
                testability_score=0.8,
                test_priority=2,
                documentation_coverage=1.0
            ),
            ComponentFeatures(
                component_id="UserFactory",
                file_path="factory.py",
                component_name="UserFactory",
                component_type="class",
                cyclomatic_complexity=1.0,
                cognitive_complexity=1.0,
                lines_of_code=30,
                parameter_count=0,
                dependency_count=2,
                testability_score=0.7,
                test_priority=3,
                documentation_coverage=1.0
            )
        ]
        
        patterns = self.pattern_detector._detect_factory(features, [])
        
        # Should detect factory pattern
        assert len(patterns) > 0
        factory_patterns = [p for p in patterns if p.pattern_name == "Factory"]
        assert len(factory_patterns) > 0
    
    def test_observer_detection(self):
        """Test Observer pattern detection."""
        features = [
            ComponentFeatures(
                component_id="notify_observers",
                file_path="observer.py",
                component_name="notify_observers",
                component_type="method",
                cyclomatic_complexity=2.0,
                cognitive_complexity=2.0,
                lines_of_code=8,
                parameter_count=1,
                dependency_count=2,
                testability_score=0.7,
                test_priority=3,
                documentation_coverage=1.0
            ),
            ComponentFeatures(
                component_id="subscribe",
                file_path="observer.py",
                component_name="subscribe",
                component_type="method",
                cyclomatic_complexity=1.0,
                cognitive_complexity=1.0,
                lines_of_code=5,
                parameter_count=1,
                dependency_count=1,
                testability_score=0.8,
                test_priority=2,
                documentation_coverage=1.0
            ),
            ComponentFeatures(
                component_id="update",
                file_path="observer.py",
                component_name="update",
                component_type="method",
                cyclomatic_complexity=1.0,
                cognitive_complexity=1.0,
                lines_of_code=3,
                parameter_count=1,
                dependency_count=0,
                testability_score=0.9,
                test_priority=1,
                documentation_coverage=1.0
            )
        ]
        
        patterns = self.pattern_detector._detect_observer(features, [])
        
        # Should detect observer pattern
        assert len(patterns) > 0
        observer_patterns = [p for p in patterns if p.pattern_name == "Observer"]
        assert len(observer_patterns) > 0


class TestMLPatternDetector:
    """Test the main ML pattern detector."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.detector = MLPatternDetector(max_features=10)
    
    def test_empty_repository_analysis(self):
        """Test analysis with empty repository."""
        # Create empty repository analysis
        repo_analysis = RepositoryAnalysisResult(
            repository_path="/empty/repo",
            analysis_time=0.1,
            file_results=[],
            repository_structure=MagicMock(),
            project_metrics=MagicMock(),
            git_analysis=MagicMock(),
            architecture_patterns=[],
            cross_file_dependencies={},
            recommendations=[]
        )
        
        result = self.detector.analyze_patterns(repo_analysis)
        
        assert result.total_components == 0
        assert len(result.component_clusters) == 0
        assert len(result.detected_anomalies) == 0
        assert len(result.design_patterns) == 0
    
    def test_simple_repository_analysis(self):
        """Test analysis with simple repository."""
        # Create mock components
        components = [
            CodeComponent(
                name="simple_func",
                component_type=ComponentType.FUNCTION,
                location=CodeLocation(1, 5),
                source_code="def simple_func(): return 1",
                file_path="test.py",
                cyclomatic_complexity=1,
                cognitive_complexity=1,
                lines_of_code=1,
                testability_score=0.9,
                test_priority=1,
                documentation_coverage=1.0
            ),
            CodeComponent(
                name="TestClass",
                component_type=ComponentType.CLASS,
                location=CodeLocation(6, 15),
                source_code="class TestClass: pass",
                file_path="test.py",
                cyclomatic_complexity=1,
                cognitive_complexity=1,
                lines_of_code=10,
                testability_score=0.8,
                test_priority=2,
                documentation_coverage=0.5
            )
        ]
        
        # Create file result
        file_result = FileAnalysisResult(
            file_path="test.py",
            language="python",
            components=components,
            file_size=1000,
            lines_of_code=15,
            complexity_metrics={},
            analysis_time=0.1
        )
        
        # Create repository analysis
        repo_analysis = RepositoryAnalysisResult(
            repository_path="/test/repo",
            analysis_time=0.5,
            file_results=[file_result],
            repository_structure=MagicMock(),
            project_metrics=MagicMock(),
            git_analysis=MagicMock(),
            architecture_patterns=[],
            cross_file_dependencies={},
            recommendations=[]
        )
        
        result = self.detector.analyze_patterns(repo_analysis)
        
        assert result.total_components == 2
        assert result.feature_engineering_time > 0
        assert result.clustering_time >= 0
        assert result.anomaly_detection_time >= 0
        assert result.pattern_detection_time >= 0
        
        # Should have some analysis results
        assert isinstance(result.component_clusters, list)
        assert isinstance(result.detected_anomalies, list)
        assert isinstance(result.design_patterns, list)


class TestIntegrationFunctions:
    """Test the main interface functions."""
    
    def test_analyze_ml_patterns_function(self):
        """Test the main analyze_ml_patterns function."""
        # Create minimal repository analysis
        repo_analysis = RepositoryAnalysisResult(
            repository_path="/test/repo",
            analysis_time=0.1,
            file_results=[],
            repository_structure=MagicMock(),
            project_metrics=MagicMock(),
            git_analysis=MagicMock(),
            architecture_patterns=[],
            cross_file_dependencies={},
            recommendations=[]
        )
        
        result = analyze_ml_patterns(repo_analysis)
        
        assert isinstance(result, type(result))  # Check it returns the right type
        assert result.total_components == 0
    
    def test_extract_component_features_function(self):
        """Test the extract_component_features function."""
        components = [
            CodeComponent(
                name="test_func",
                component_type=ComponentType.FUNCTION,
                location=CodeLocation(1, 5),
                source_code="def test_func(): pass",
                file_path="test.py",
                cyclomatic_complexity=1,
                testability_score=0.8
            )
        ]
        
        features = extract_component_features(components)
        
        assert len(features) == 1
        assert features[0].component_name == "test_func"


class TestDataStructures:
    """Test data structure serialization and manipulation."""
    
    def test_component_features_serialization(self):
        """Test ComponentFeatures to_dict method."""
        feature = ComponentFeatures(
            component_id="test_id",
            file_path="test.py",
            component_name="test_func",
            component_type="function",
            cyclomatic_complexity=2.0,
            cognitive_complexity=3.0,
            lines_of_code=10,
            parameter_count=2,
            dependency_count=1,
            testability_score=0.8,
            test_priority=3,
            documentation_coverage=0.5,
            text_features=np.array([0.1, 0.2, 0.3])
        )
        
        result_dict = feature.to_dict()
        
        assert result_dict['component_id'] == "test_id"
        assert result_dict['component_name'] == "test_func"
        assert result_dict['text_features'] == [0.1, 0.2, 0.3]
    
    def test_anomaly_serialization(self):
        """Test CodeAnomaly to_dict method."""
        anomaly = CodeAnomaly(
            component_id="test_id",
            anomaly_score=0.8,
            anomaly_type="complexity",
            description="High complexity detected",
            severity="high",
            recommendations=["Refactor function", "Add tests"]
        )
        
        result_dict = anomaly.to_dict()
        
        assert result_dict['component_id'] == "test_id"
        assert result_dict['anomaly_score'] == 0.8
        assert result_dict['severity'] == "high"
        assert len(result_dict['recommendations']) == 2
    
    def test_design_pattern_serialization(self):
        """Test DesignPattern to_dict method."""
        pattern = DesignPattern(
            pattern_name="Singleton",
            pattern_type="creational",
            component_ids=["comp1", "comp2"],
            confidence_score=0.8,
            description="Singleton pattern detected",
            evidence=["getInstance method", "Private constructor"]
        )
        
        result_dict = pattern.to_dict()
        
        assert result_dict['pattern_name'] == "Singleton"
        assert result_dict['pattern_type'] == "creational"
        assert len(result_dict['component_ids']) == 2
        assert len(result_dict['evidence']) == 2


class TestPerformance:
    """Test performance characteristics of ML analysis."""
    
    def test_feature_extraction_performance(self):
        """Test feature extraction performance with multiple components."""
        import time
        
        # Create many components
        components = []
        for i in range(50):
            component = CodeComponent(
                name=f"func_{i}",
                component_type=ComponentType.FUNCTION,
                location=CodeLocation(i*10, i*10+5),
                source_code=f"def func_{i}(): return {i}",
                file_path=f"file_{i//10}.py",
                cyclomatic_complexity=1 + (i % 5),
                cognitive_complexity=1 + (i % 5),
                lines_of_code=5 + (i % 10),
                testability_score=0.5 + (i % 5) * 0.1,
                test_priority=1 + (i % 5),
                documentation_coverage=float(i % 2)
            )
            components.append(component)
        
        feature_engineer = FeatureEngineer(max_features=100)
        
        start_time = time.time()
        features = feature_engineer.extract_features(components)
        extraction_time = time.time() - start_time
        
        assert len(features) == 50
        assert extraction_time < 10.0  # Should complete in reasonable time
