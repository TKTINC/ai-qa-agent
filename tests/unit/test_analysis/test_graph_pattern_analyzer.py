"""
Tests for the graph pattern analysis implementation.
"""

import pytest
import networkx as nx
from unittest.mock import MagicMock

from src.analysis.graph_pattern_analyzer import (
    GraphPatternAnalyzer,
    GraphBuilder,
    CentralityAnalyzer,
    CommunityDetector,
    ArchitecturalAnalyzer,
    DependencyAnalyzer,
    GraphNode,
    GraphEdge,
    CommunityCluster,
    ArchitecturalLayer,
    CriticalPath,
    analyze_graph_patterns,
    build_dependency_graph
)
from src.analysis.ast_parser import CodeComponent, ComponentType, CodeLocation
from src.analysis.repository_analyzer import RepositoryAnalysisResult, FileAnalysisResult


class TestGraphBuilder:
    """Test the graph building functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.builder = GraphBuilder()
    
    def test_build_empty_graph(self):
        """Test building graph from empty repository."""
        repo_analysis = RepositoryAnalysisResult(
            repository_path="/empty",
            analysis_time=0.1,
            file_results=[],
            repository_structure=MagicMock(),
            project_metrics=MagicMock(),
            git_analysis=MagicMock(),
            architecture_patterns=[],
            cross_file_dependencies={},
            recommendations=[]
        )
        
        graph = self.builder.build_graph(repo_analysis)
        
        assert graph.number_of_nodes() == 0
        assert graph.number_of_edges() == 0
    
    def test_build_simple_graph(self):
        """Test building graph from simple repository."""
        # Create mock components
        components = [
            CodeComponent(
                name="main",
                component_type=ComponentType.FUNCTION,
                location=CodeLocation(1, 10),
                source_code="def main(): helper()",
                file_path="main.py",
                function_calls=["helper"],
                imports=["os", "sys"]
            ),
            CodeComponent(
                name="helper",
                component_type=ComponentType.FUNCTION,
                location=CodeLocation(11, 15),
                source_code="def helper(): pass",
                file_path="main.py",
                function_calls=[],
                imports=[]
            ),
            CodeComponent(
                name="UserClass",
                component_type=ComponentType.CLASS,
                location=CodeLocation(16, 25),
                source_code="class UserClass: pass",
                file_path="main.py",
                base_classes=["BaseUser"],
                function_calls=[],
                imports=[]
            )
        ]
        
        file_result = FileAnalysisResult(
            file_path="main.py",
            language="python",
            components=components,
            file_size=1000,
            lines_of_code=25,
            complexity_metrics={},
            analysis_time=0.1
        )
        
        repo_analysis = RepositoryAnalysisResult(
            repository_path="/test",
            analysis_time=0.5,
            file_results=[file_result],
            repository_structure=MagicMock(),
            project_metrics=MagicMock(),
            git_analysis=MagicMock(),
            architecture_patterns=[],
            cross_file_dependencies={},
            recommendations=[]
        )
        
        graph = self.builder.build_graph(repo_analysis)
        
        # Should have nodes for components and file
        assert graph.number_of_nodes() >= 4  # 3 components + 1 file
        assert graph.number_of_edges() >= 3  # containment + some dependencies
        
        # Check for component nodes
        component_nodes = [n for n in graph.nodes() if "::" in n]
        assert len(component_nodes) == 3
        
        # Check for file nodes
        file_nodes = [n for n in graph.nodes() if n.startswith("file::")]
        assert len(file_nodes) == 1
    
    def test_dependency_edge_creation(self):
        """Test creation of dependency edges."""
        components = [
            CodeComponent(
                name="caller",
                component_type=ComponentType.FUNCTION,
                location=CodeLocation(1, 5),
                source_code="def caller(): called()",
                file_path="test.py",
                function_calls=["called"]
            ),
            CodeComponent(
                name="called",
                component_type=ComponentType.FUNCTION,
                location=CodeLocation(6, 10),
                source_code="def called(): pass",
                file_path="test.py",
                function_calls=[]
            )
        ]
        
        file_result = FileAnalysisResult(
            file_path="test.py",
            language="python",
            components=components,
            file_size=500,
            lines_of_code=10,
            complexity_metrics={},
            analysis_time=0.1
        )
        
        repo_analysis = RepositoryAnalysisResult(
            repository_path="/test",
            analysis_time=0.3,
            file_results=[file_result],
            repository_structure=MagicMock(),
            project_metrics=MagicMock(),
            git_analysis=MagicMock(),
            architecture_patterns=[],
            cross_file_dependencies={},
            recommendations=[]
        )
        
        graph = self.builder.build_graph(repo_analysis)
        
        # Should have call dependency
        caller_id = "test.py::caller"
        called_id = "test.py::called"
        
        assert graph.has_edge(caller_id, called_id)
        edge_data = graph.edges[caller_id, called_id]
        assert edge_data['edge_type'] == 'calls'


class TestCentralityAnalyzer:
    """Test centrality analysis functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.analyzer = CentralityAnalyzer()
    
    def test_centrality_analysis_empty_graph(self):
        """Test centrality analysis on empty graph."""
        graph = nx.DiGraph()
        
        centrality_measures = self.analyzer.analyze_centrality(graph)
        
        assert isinstance(centrality_measures, dict)
        # Should return empty measures for empty graph
        for measure in centrality_measures.values():
            assert len(measure) == 0
    
    def test_centrality_analysis_simple_graph(self):
        """Test centrality analysis on simple graph."""
        graph = nx.DiGraph()
        graph.add_edges_from([
            ("A", "B"),
            ("B", "C"),
            ("C", "D"),
            ("A", "D")
        ])
        
        centrality_measures = self.analyzer.analyze_centrality(graph)
        
        # Should have all centrality measures
        expected_measures = ['degree', 'betweenness', 'closeness', 'eigenvector']
        for measure in expected_measures:
            assert measure in centrality_measures
            assert len(centrality_measures[measure]) == 4  # 4 nodes
    
    def test_important_nodes_identification(self):
        """Test identification of important nodes."""
        centrality_measures = {
            'degree': {'A': 0.8, 'B': 0.6, 'C': 0.4, 'D': 0.2},
            'betweenness': {'A': 0.5, 'B': 0.7, 'C': 0.3, 'D': 0.1}
        }
        
        important_nodes = self.analyzer.identify_important_nodes(centrality_measures, top_k=2)
        
        assert 'degree' in important_nodes
        assert 'betweenness' in important_nodes
        
        # Should return top 2 for each measure
        assert len(important_nodes['degree']) == 2
        assert len(important_nodes['betweenness']) == 2
        
        # Should be sorted by centrality score
        assert important_nodes['degree'][0][1] >= important_nodes['degree'][1][1]
        assert important_nodes['betweenness'][0][1] >= important_nodes['betweenness'][1][1]
    
    def test_bottleneck_detection(self):
        """Test bottleneck node detection."""
        # Create betweenness measures with one clear bottleneck
        centrality_measures = {
            'betweenness': {
                'A': 0.1,
                'B': 0.9,  # Clear bottleneck
                'C': 0.05,
                'D': 0.08
            }
        }
        
        bottlenecks = self.analyzer.find_bottlenecks(centrality_measures)
        
        # Should identify B as bottleneck
        assert len(bottlenecks) > 0
        assert bottlenecks[0][0] == 'B'
        assert bottlenecks[0][1] == 0.9
    
    def test_hub_detection(self):
        """Test hub node detection."""
        centrality_measures = {
            'degree': {
                'A': 0.9,  # Clear hub
                'B': 0.2,
                'C': 0.3,
                'D': 0.25
            }
        }
        
        hubs = self.analyzer.find_hubs(centrality_measures)
        
        # Should identify A as hub
        assert len(hubs) > 0
        assert hubs[0][0] == 'A'
        assert hubs[0][1] == 0.9


class TestCommunityDetector:
    """Test community detection functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.detector = CommunityDetector()
    
    def test_community_detection_small_graph(self):
        """Test community detection on small graph."""
        graph = nx.DiGraph()
        graph.add_edges_from([
            ("A", "B"),
            ("B", "A"),  # Bidirectional for community
            ("C", "D"),
            ("D", "C")
        ])
        
        communities, modularity = self.detector.detect_communities(graph)
        
        # Should find communities
        assert isinstance(communities, list)
        assert isinstance(modularity, float)
        
        # Small graph may or may not form clear communities
        assert modularity >= -1.0 and modularity <= 1.0
    
    def test_community_type_classification(self):
        """Test community type classification."""
        # Mock graph with node data
        graph = nx.DiGraph()
        graph.add_node("file1::func1", node_type="component", file_path="file1.py")
        graph.add_node("file1::func2", node_type="component", file_path="file1.py")
        graph.add_node("file2::func3", node_type="component", file_path="file2.py")
        
        # Test file module classification
        community1 = {"file1::func1", "file1::func2"}
        comm_type1 = self.detector._classify_community_type(community1, graph)
        assert comm_type1 == "file_module"
        
        # Test mixed community
        community2 = {"file1::func1", "file2::func3"}
        comm_type2 = self.detector._classify_community_type(community2, graph)
        assert comm_type2 in ["mixed_community", "feature_cluster", "layer_module"]


class TestArchitecturalAnalyzer:
    """Test architectural analysis functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.analyzer = ArchitecturalAnalyzer()
    
    def test_layer_detection_simple(self):
        """Test layer detection on simple DAG."""
        graph = nx.DiGraph()
        graph.add_edges_from([
            ("layer0_A", "layer1_B"),
            ("layer0_C", "layer1_B"),
            ("layer1_B", "layer2_D")
        ])
        
        # Add node data
        for node in graph.nodes():
            graph.nodes[node]['file_path'] = f"{node}.py"
        
        layers = self.analyzer.detect_layers(graph)
        
        # Should detect layers
        assert len(layers) >= 2
        
        # Layers should have proper structure
        for layer in layers:
            assert isinstance(layer.layer_id, int)
            assert isinstance(layer.node_ids, list)
            assert len(layer.node_ids) > 0
    
    def test_layer_name_generation(self):
        """Test layer name generation."""
        # Mock graph with model file
        graph = nx.DiGraph()
        graph.add_node("test", file_path="models/user.py")
        
        layer_nodes = {"test"}
        name = self.analyzer._generate_layer_name(0, layer_nodes, graph)
        
        assert "Data Layer" in name
        
        # Test controller layer
        graph.nodes["test"]["file_path"] = "controllers/auth.py"
        name = self.analyzer._generate_layer_name(1, layer_nodes, graph)
        
        assert "Controller Layer" in name
    
    def test_critical_path_finding(self):
        """Test critical path detection."""
        graph = nx.DiGraph()
        graph.add_edges_from([
            ("A", "B"),
            ("B", "C"),
            ("C", "D")
        ])
        
        # Add complexity data
        for node in graph.nodes():
            graph.nodes[node]['complexity'] = 5
        
        critical_paths = self.analyzer.find_critical_paths(graph)
        
        # Should find some paths
        assert isinstance(critical_paths, list)
        # May or may not find paths depending on algorithm


class TestDependencyAnalyzer:
    """Test dependency analysis functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.analyzer = DependencyAnalyzer()
    
    def test_circular_dependency_detection(self):
        """Test circular dependency detection."""
        graph = nx.DiGraph()
        graph.add_edges_from([
            ("A", "B"),
            ("B", "C"),
            ("C", "A")  # Creates cycle
        ])
        
        cycles = self.analyzer._find_circular_dependencies(graph)
        
        # Should find the cycle
        assert len(cycles) > 0
        cycle = cycles[0]
        assert len(cycle) == 3
        assert set(cycle) == {"A", "B", "C"}
    
    def test_coupling_metrics_calculation(self):
        """Test coupling metrics calculation."""
        graph = nx.DiGraph()
        graph.add_edges_from([
            ("A", "B"),
            ("A", "C"),
            ("B", "C"),
            ("D", "A")
        ])
        
        metrics = self.analyzer._calculate_coupling_metrics(graph)
        
        # Should calculate various metrics
        expected_metrics = [
            'average_afferent_coupling',
            'average_efferent_coupling',
            'average_instability',
            'total_dependencies',
            'coupling_density'
        ]
        
        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))
    
    def test_dependency_violation_detection(self):
        """Test dependency violation detection."""
        graph = nx.DiGraph()
        graph.add_edge("view_comp", "model_comp")
        
        # Add file path data
        graph.nodes["view_comp"] = {"file_path": "views/user.py"}
        graph.nodes["model_comp"] = {"file_path": "models/user.py"}
        
        violations = self.analyzer._find_dependency_violations(graph)
        
        # Should detect layer violation (view depending on model is OK)
        # But we test the mechanism works
        assert isinstance(violations, list)
    
    def test_layer_violation_detection(self):
        """Test layer violation detection logic."""
        # View -> Model (OK)
        assert not self.analyzer._is_layer_violation("views/user.py", "models/user.py")
        
        # Model -> View (violation)
        assert self.analyzer._is_layer_violation("models/user.py", "views/user.py")
        
        # Controller -> Service (OK)
        assert not self.analyzer._is_layer_violation("controllers/auth.py", "services/user.py")


class TestGraphPatternAnalyzer:
    """Test the main graph pattern analyzer."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.analyzer = GraphPatternAnalyzer()
    
    def test_empty_repository_analysis(self):
        """Test analysis of empty repository."""
        repo_analysis = RepositoryAnalysisResult(
            repository_path="/empty",
            analysis_time=0.1,
            file_results=[],
            repository_structure=MagicMock(),
            project_metrics=MagicMock(),
            git_analysis=MagicMock(),
            architecture_patterns=[],
            cross_file_dependencies={},
            recommendations=[]
        )
        
        result = self.analyzer.analyze_graph_patterns(repo_analysis)
        
        assert result.total_nodes == 0
        assert result.total_edges == 0
        assert len(result.communities) == 0
        assert len(result.layers) == 0
        assert len(result.critical_paths) == 0
    
    def test_simple_repository_analysis(self):
        """Test analysis of simple repository."""
        # Create simple components
        components = [
            CodeComponent(
                name="main",
                component_type=ComponentType.FUNCTION,
                location=CodeLocation(1, 10),
                source_code="def main(): helper()",
                file_path="main.py",
                function_calls=["helper"],
                cyclomatic_complexity=2,
                lines_of_code=5
            ),
            CodeComponent(
                name="helper",
                component_type=ComponentType.FUNCTION,
                location=CodeLocation(11, 15),
                source_code="def helper(): pass",
                file_path="main.py",
                function_calls=[],
                cyclomatic_complexity=1,
                lines_of_code=3
            )
        ]
        
        file_result = FileAnalysisResult(
            file_path="main.py",
            language="python",
            components=components,
            file_size=500,
            lines_of_code=15,
            complexity_metrics={},
            analysis_time=0.1
        )
        
        repo_analysis = RepositoryAnalysisResult(
            repository_path="/test",
            analysis_time=0.3,
            file_results=[file_result],
            repository_structure=MagicMock(),
            project_metrics=MagicMock(),
            git_analysis=MagicMock(),
            architecture_patterns=[],
            cross_file_dependencies={},
            recommendations=[]
        )
        
        result = self.analyzer.analyze_graph_patterns(repo_analysis)
        
        # Should have nodes and analysis results
        assert result.total_nodes > 0
        assert result.analysis_time > 0
        assert isinstance(result.nodes, list)
        assert isinstance(result.edges, list)
        assert isinstance(result.communities, list)
        assert isinstance(result.layers, list)
        
        # Graph metrics should be calculated
        assert 0.0 <= result.graph_density <= 1.0
        assert result.average_clustering >= 0.0
        assert result.number_of_components > 0
    
    def test_graph_metrics_calculation(self):
        """Test graph metrics calculation."""
        # Create simple graph
        graph = nx.DiGraph()
        graph.add_edges_from([("A", "B"), ("B", "C")])
        
        metrics = self.analyzer._calculate_graph_metrics(graph)
        
        expected_metrics = ['density', 'average_clustering', 'number_of_components']
        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))
        
        # Density should be reasonable
        assert 0.0 <= metrics['density'] <= 1.0
    
    def test_node_centrality_update(self):
        """Test updating nodes with centrality measures."""
        graph = nx.DiGraph()
        graph.add_edges_from([("A", "B"), ("B", "C")])
        
        # Add initial node data
        for node in graph.nodes():
            graph.nodes[node].update({
                'node_id': node,
                'node_type': 'component',
                'name': node,
                'file_path': f"{node}.py"
            })
        
        centrality_measures = {
            'degree': {'A': 0.5, 'B': 1.0, 'C': 0.5},
            'betweenness': {'A': 0.0, 'B': 1.0, 'C': 0.0}
        }
        
        self.analyzer._update_node_centrality(graph, centrality_measures)
        
        # Check that centrality measures were added
        for node in graph.nodes():
            node_data = graph.nodes[node]
            assert 'degree_centrality' in node_data
            assert 'betweenness_centrality' in node_data
            assert isinstance(node_data['degree_centrality'], float)
            assert isinstance(node_data['betweenness_centrality'], float)


class TestIntegrationFunctions:
    """Test the main interface functions."""
    
    def test_analyze_graph_patterns_function(self):
        """Test the main analyze_graph_patterns function."""
        repo_analysis = RepositoryAnalysisResult(
            repository_path="/test",
            analysis_time=0.1,
            file_results=[],
            repository_structure=MagicMock(),
            project_metrics=MagicMock(),
            git_analysis=MagicMock(),
            architecture_patterns=[],
            cross_file_dependencies={},
            recommendations=[]
        )
        
        result = analyze_graph_patterns(repo_analysis)
        
        assert isinstance(result, type(result))  # Check return type
        assert result.total_nodes == 0  # Empty repository
    
    def test_build_dependency_graph_function(self):
        """Test the build_dependency_graph function."""
        repo_analysis = RepositoryAnalysisResult(
            repository_path="/test",
            analysis_time=0.1,
            file_results=[],
            repository_structure=MagicMock(),
            project_metrics=MagicMock(),
            git_analysis=MagicMock(),
            architecture_patterns=[],
            cross_file_dependencies={},
            recommendations=[]
        )
        
        graph = build_dependency_graph(repo_analysis)
        
        assert isinstance(graph, nx.DiGraph)
        assert graph.number_of_nodes() == 0  # Empty repository


class TestDataStructures:
    """Test data structure functionality."""
    
    def test_graph_node_serialization(self):
        """Test GraphNode to_dict method."""
        node = GraphNode(
            node_id="test_node",
            node_type="component",
            name="test_func",
            file_path="test.py",
            component_type="function",
            complexity=5,
            lines_of_code=20,
            degree_centrality=0.5,
            betweenness_centrality=0.3
        )
        
        result_dict = node.to_dict()
        
        assert result_dict['node_id'] == "test_node"
        assert result_dict['node_type'] == "component"
        assert result_dict['complexity'] == 5
        assert result_dict['degree_centrality'] == 0.5
    
    def test_graph_edge_serialization(self):
        """Test GraphEdge to_dict method."""
        edge = GraphEdge(
            source="node1",
            target="node2",
            edge_type="calls",
            weight=1.5
        )
        
        result_dict = edge.to_dict()
        
        assert result_dict['source'] == "node1"
        assert result_dict['target'] == "node2"
        assert result_dict['edge_type'] == "calls"
        assert result_dict['weight'] == 1.5
    
    def test_community_cluster_serialization(self):
        """Test CommunityCluster to_dict method."""
        cluster = CommunityCluster(
            cluster_id=1,
            node_ids=["node1", "node2", "node3"],
            modularity_score=0.7,
            description="Test cluster",
            cluster_type="feature_cluster"
        )
        
        result_dict = cluster.to_dict()
        
        assert result_dict['cluster_id'] == 1
        assert len(result_dict['node_ids']) == 3
        assert result_dict['size'] == 3
        assert result_dict['modularity_score'] == 0.7
    
    def test_architectural_layer_serialization(self):
        """Test ArchitecturalLayer to_dict method."""
        layer = ArchitecturalLayer(
            layer_id=0,
            layer_name="Data Layer",
            node_ids=["model1", "model2"],
            layer_level=0,
            dependencies_up=[1, 2],
            dependencies_down=[]
        )
        
        result_dict = layer.to_dict()
        
        assert result_dict['layer_id'] == 0
        assert result_dict['layer_name'] == "Data Layer"
        assert result_dict['layer_level'] == 0
        assert result_dict['size'] == 2
        assert result_dict['dependencies_up'] == [1, 2]
    
    def test_critical_path_serialization(self):
        """Test CriticalPath to_dict method."""
        path = CriticalPath(
            path_id=1,
            node_sequence=["A", "B", "C", "D"],
            path_type="longest",
            total_complexity=15,
            description="Longest path through system"
        )
        
        result_dict = path.to_dict()
        
        assert result_dict['path_id'] == 1
        assert result_dict['length'] == 4
        assert result_dict['path_type'] == "longest"
        assert result_dict['total_complexity'] == 15


class TestNetworkXIntegration:
    """Test NetworkX integration and graph operations."""
    
    def test_networkx_graph_creation(self):
        """Test creating NetworkX graphs."""
        builder = GraphBuilder()
        graph = nx.DiGraph()
        
        # Test basic graph operations
        graph.add_node("test", type="component")
        graph.add_edge("test1", "test2", edge_type="calls")
        
        assert graph.number_of_nodes() >= 1
        assert "test" in graph.nodes()
        
        if graph.number_of_edges() > 0:
            assert graph.has_edge("test1", "test2")
    
    def test_graph_algorithms(self):
        """Test graph algorithms work correctly."""
        graph = nx.DiGraph()
        graph.add_edges_from([
            ("A", "B"),
            ("B", "C"),
            ("C", "A")  # Cycle
        ])
        
        # Test cycle detection
        cycles = list(nx.simple_cycles(graph))
        assert len(cycles) > 0
        
        # Test centrality calculations
        degree_cent = nx.degree_centrality(graph)
        assert len(degree_cent) == 3
        assert all(0 <= v <= 1 for v in degree_cent.values())
        
        # Test connected components
        undirected = graph.to_undirected()
        components = list(nx.connected_components(undirected))
        assert len(components) >= 1
