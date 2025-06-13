"""
Graph-based pattern analysis for code architecture and dependencies.
Uses NetworkX for graph analysis and architectural pattern detection.
"""

import logging
import networkx as nx
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set, Union
from collections import defaultdict, Counter
import numpy as np

# Import project components
from .ast_parser import CodeComponent, ComponentType
from .repository_analyzer import RepositoryAnalysisResult, FileAnalysisResult

logger = logging.getLogger(__name__)


@dataclass
class GraphNode:
    """Represents a node in the code dependency graph."""
    node_id: str
    node_type: str  # 'component', 'file', 'module'
    name: str
    file_path: str
    component_type: Optional[str] = None
    complexity: int = 1
    lines_of_code: int = 0
    
    # Graph metrics (calculated)
    degree_centrality: float = 0.0
    betweenness_centrality: float = 0.0
    closeness_centrality: float = 0.0
    eigenvector_centrality: float = 0.0
    clustering_coefficient: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'node_id': self.node_id,
            'node_type': self.node_type,
            'name': self.name,
            'file_path': self.file_path,
            'component_type': self.component_type,
            'complexity': self.complexity,
            'lines_of_code': self.lines_of_code,
            'degree_centrality': self.degree_centrality,
            'betweenness_centrality': self.betweenness_centrality,
            'closeness_centrality': self.closeness_centrality,
            'eigenvector_centrality': self.eigenvector_centrality,
            'clustering_coefficient': self.clustering_coefficient
        }


@dataclass
class GraphEdge:
    """Represents an edge in the code dependency graph."""
    source: str
    target: str
    edge_type: str  # 'calls', 'imports', 'inherits', 'contains'
    weight: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'source': self.source,
            'target': self.target,
            'edge_type': self.edge_type,
            'weight': self.weight
        }


@dataclass
class CommunityCluster:
    """Represents a community/cluster detected in the graph."""
    cluster_id: int
    node_ids: List[str]
    modularity_score: float
    description: str
    cluster_type: str  # 'module', 'feature', 'layer'
    
    @property
    def size(self) -> int:
        """Number of nodes in cluster."""
        return len(self.node_ids)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'cluster_id': self.cluster_id,
            'node_ids': self.node_ids,
            'modularity_score': self.modularity_score,
            'description': self.description,
            'cluster_type': self.cluster_type,
            'size': self.size
        }


@dataclass
class ArchitecturalLayer:
    """Represents an architectural layer detected in the code."""
    layer_id: int
    layer_name: str
    node_ids: List[str]
    layer_level: int  # 0 = bottom, higher = more abstract
    dependencies_up: List[int]  # Layers this depends on
    dependencies_down: List[int]  # Layers that depend on this
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'layer_id': self.layer_id,
            'layer_name': self.layer_name,
            'node_ids': self.node_ids,
            'layer_level': self.layer_level,
            'dependencies_up': self.dependencies_up,
            'dependencies_down': self.dependencies_down,
            'size': len(self.node_ids)
        }


@dataclass
class CriticalPath:
    """Represents a critical path in the dependency graph."""
    path_id: int
    node_sequence: List[str]
    path_type: str  # 'longest', 'critical', 'bottleneck'
    total_complexity: int
    description: str
    
    @property
    def length(self) -> int:
        """Length of the path."""
        return len(self.node_sequence)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'path_id': self.path_id,
            'node_sequence': self.node_sequence,
            'path_type': self.path_type,
            'total_complexity': self.total_complexity,
            'description': self.description,
            'length': self.length
        }


@dataclass
class GraphAnalysisResult:
    """Complete graph analysis results."""
    total_nodes: int
    total_edges: int
    analysis_time: float
    
    # Graph structure
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    
    # Graph metrics
    graph_density: float
    average_clustering: float
    number_of_components: int
    diameter: Optional[int]
    radius: Optional[int]
    
    # Community detection
    communities: List[CommunityCluster]
    modularity_score: float
    
    # Architectural analysis
    layers: List[ArchitecturalLayer]
    critical_paths: List[CriticalPath]
    
    # Centrality analysis
    most_central_nodes: List[Tuple[str, float]]  # (node_id, centrality_score)
    bottleneck_nodes: List[Tuple[str, float]]
    hub_nodes: List[Tuple[str, float]]
    
    # Dependency analysis
    circular_dependencies: List[List[str]]
    dependency_violations: List[str]
    coupling_metrics: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'total_nodes': self.total_nodes,
            'total_edges': self.total_edges,
            'analysis_time': self.analysis_time,
            'nodes': [node.to_dict() for node in self.nodes],
            'edges': [edge.to_dict() for edge in self.edges],
            'graph_density': self.graph_density,
            'average_clustering': self.average_clustering,
            'number_of_components': self.number_of_components,
            'diameter': self.diameter,
            'radius': self.radius,
            'communities': [comm.to_dict() for comm in self.communities],
            'modularity_score': self.modularity_score,
            'layers': [layer.to_dict() for layer in self.layers],
            'critical_paths': [path.to_dict() for path in self.critical_paths],
            'most_central_nodes': self.most_central_nodes,
            'bottleneck_nodes': self.bottleneck_nodes,
            'hub_nodes': self.hub_nodes,
            'circular_dependencies': self.circular_dependencies,
            'dependency_violations': self.dependency_violations,
            'coupling_metrics': self.coupling_metrics
        }


class GraphBuilder:
    """Builds dependency graphs from code analysis results."""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.nodes = {}
        self.edges = []
    
    def build_graph(self, repo_analysis: RepositoryAnalysisResult) -> nx.DiGraph:
        """Build dependency graph from repository analysis."""
        logger.info("Building dependency graph from repository analysis")
        
        self.graph = nx.DiGraph()
        self.nodes = {}
        self.edges = []
        
        # Add nodes for all components
        self._add_component_nodes(repo_analysis)
        
        # Add file-level nodes
        self._add_file_nodes(repo_analysis)
        
        # Add dependency edges
        self._add_dependency_edges(repo_analysis)
        
        # Add containment edges (file contains components)
        self._add_containment_edges(repo_analysis)
        
        logger.info(f"Graph built: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
        return self.graph
    
    def _add_component_nodes(self, repo_analysis: RepositoryAnalysisResult) -> None:
        """Add nodes for all code components."""
        for file_result in repo_analysis.file_results:
            if file_result.has_error:
                continue
                
            for component in file_result.components:
                node_id = f"{file_result.file_path}::{component.name}"
                
                node = GraphNode(
                    node_id=node_id,
                    node_type='component',
                    name=component.name,
                    file_path=file_result.file_path,
                    component_type=component.component_type.value,
                    complexity=component.cyclomatic_complexity,
                    lines_of_code=component.lines_of_code
                )
                
                self.nodes[node_id] = node
                self.graph.add_node(node_id, **node.to_dict())
    
    def _add_file_nodes(self, repo_analysis: RepositoryAnalysisResult) -> None:
        """Add nodes for files."""
        for file_result in repo_analysis.file_results:
            if file_result.has_error:
                continue
                
            node_id = f"file::{file_result.file_path}"
            
            node = GraphNode(
                node_id=node_id,
                node_type='file',
                name=file_result.file_path.split('/')[-1],  # Just filename
                file_path=file_result.file_path,
                lines_of_code=file_result.lines_of_code
            )
            
            self.nodes[node_id] = node
            self.graph.add_node(node_id, **node.to_dict())
    
    def _add_dependency_edges(self, repo_analysis: RepositoryAnalysisResult) -> None:
        """Add dependency edges between components."""
        for file_result in repo_analysis.file_results:
            if file_result.has_error:
                continue
                
            for component in file_result.components:
                source_id = f"{file_result.file_path}::{component.name}"
                
                # Add function call dependencies
                for call in component.function_calls:
                    target_id = self._find_target_component(call, repo_analysis)
                    if target_id and target_id != source_id:
                        edge = GraphEdge(source_id, target_id, 'calls', 1.0)
                        self.edges.append(edge)
                        self.graph.add_edge(source_id, target_id, **edge.to_dict())
                
                # Add import dependencies
                for import_name in component.imports:
                    target_file = self._find_target_file(import_name, repo_analysis)
                    if target_file:
                        target_id = f"file::{target_file}"
                        edge = GraphEdge(source_id, target_id, 'imports', 0.5)
                        self.edges.append(edge)
                        self.graph.add_edge(source_id, target_id, **edge.to_dict())
        
        # Add inheritance edges for classes
        self._add_inheritance_edges(repo_analysis)
    
    def _add_inheritance_edges(self, repo_analysis: RepositoryAnalysisResult) -> None:
        """Add inheritance edges between classes."""
        for file_result in repo_analysis.file_results:
            if file_result.has_error:
                continue
                
            for component in file_result.components:
                if component.component_type == ComponentType.CLASS:
                    source_id = f"{file_result.file_path}::{component.name}"
                    
                    for base_class in component.base_classes:
                        target_id = self._find_target_component(base_class, repo_analysis)
                        if target_id and target_id != source_id:
                            edge = GraphEdge(source_id, target_id, 'inherits', 1.5)
                            self.edges.append(edge)
                            self.graph.add_edge(source_id, target_id, **edge.to_dict())
    
    def _add_containment_edges(self, repo_analysis: RepositoryAnalysisResult) -> None:
        """Add containment edges (file contains components)."""
        for file_result in repo_analysis.file_results:
            if file_result.has_error:
                continue
                
            file_id = f"file::{file_result.file_path}"
            
            for component in file_result.components:
                component_id = f"{file_result.file_path}::{component.name}"
                edge = GraphEdge(file_id, component_id, 'contains', 1.0)
                self.edges.append(edge)
                self.graph.add_edge(file_id, component_id, **edge.to_dict())
    
    def _find_target_component(self, target_name: str, 
                              repo_analysis: RepositoryAnalysisResult) -> Optional[str]:
        """Find target component by name across all files."""
        for file_result in repo_analysis.file_results:
            if file_result.has_error:
                continue
                
            for component in file_result.components:
                if component.name == target_name:
                    return f"{file_result.file_path}::{component.name}"
                
                # Also check if target_name is a method call on this component
                if '.' in target_name:
                    parts = target_name.split('.')
                    if len(parts) >= 2 and component.name == parts[0]:
                        # Look for the method in this component's file
                        method_name = parts[1]
                        for other_comp in file_result.components:
                            if (other_comp.name == method_name and 
                                other_comp.component_type == ComponentType.METHOD):
                                return f"{file_result.file_path}::{other_comp.name}"
        
        return None
    
    def _find_target_file(self, import_name: str, 
                         repo_analysis: RepositoryAnalysisResult) -> Optional[str]:
        """Find target file by import name."""
        # Simple heuristic: look for files that might match the import
        for file_result in repo_analysis.file_results:
            file_name = file_result.file_path.split('/')[-1].replace('.py', '')
            if import_name.endswith(file_name) or file_name.endswith(import_name):
                return file_result.file_path
        
        return None


class CentralityAnalyzer:
    """Analyzes node centrality and importance in the graph."""
    
    def __init__(self):
        pass
    
    def analyze_centrality(self, graph: nx.DiGraph) -> Dict[str, Dict[str, float]]:
        """Calculate various centrality measures for all nodes."""
        logger.info("Analyzing node centrality measures")
        
        centrality_measures = {}
        
        if graph.number_of_nodes() == 0:
            return centrality_measures
        
        try:
            # Degree centrality
            centrality_measures['degree'] = nx.degree_centrality(graph)
            
            # Betweenness centrality
            centrality_measures['betweenness'] = nx.betweenness_centrality(graph)
            
            # Closeness centrality (only for connected components)
            if nx.is_weakly_connected(graph):
                centrality_measures['closeness'] = nx.closeness_centrality(graph)
            else:
                # Calculate for largest connected component
                largest_cc = max(nx.weakly_connected_components(graph), key=len)
                subgraph = graph.subgraph(largest_cc)
                closeness = nx.closeness_centrality(subgraph)
                centrality_measures['closeness'] = closeness
            
            # Eigenvector centrality
            try:
                centrality_measures['eigenvector'] = nx.eigenvector_centrality(graph, max_iter=1000)
            except (nx.PowerIterationFailedConvergence, nx.NetworkXError):
                # Fallback to PageRank for difficult cases
                centrality_measures['eigenvector'] = nx.pagerank(graph)
            
        except Exception as e:
            logger.warning(f"Centrality analysis failed: {e}")
            # Return empty measures
            for measure in ['degree', 'betweenness', 'closeness', 'eigenvector']:
                centrality_measures[measure] = {}
        
        return centrality_measures
    
    def identify_important_nodes(self, centrality_measures: Dict[str, Dict[str, float]], 
                                top_k: int = 10) -> Dict[str, List[Tuple[str, float]]]:
        """Identify most important nodes based on centrality measures."""
        important_nodes = {}
        
        for measure_name, measures in centrality_measures.items():
            if measures:
                sorted_nodes = sorted(measures.items(), key=lambda x: x[1], reverse=True)
                important_nodes[measure_name] = sorted_nodes[:top_k]
            else:
                important_nodes[measure_name] = []
        
        return important_nodes
    
    def find_bottlenecks(self, centrality_measures: Dict[str, Dict[str, float]]) -> List[Tuple[str, float]]:
        """Find bottleneck nodes with high betweenness centrality."""
        betweenness = centrality_measures.get('betweenness', {})
        if not betweenness:
            return []
        
        # Nodes with betweenness > average + 2*std are considered bottlenecks
        values = list(betweenness.values())
        if not values:
            return []
        
        mean_betweenness = np.mean(values)
        std_betweenness = np.std(values)
        threshold = mean_betweenness + 2 * std_betweenness
        
        bottlenecks = [(node, score) for node, score in betweenness.items() 
                       if score > threshold]
        
        return sorted(bottlenecks, key=lambda x: x[1], reverse=True)
    
    def find_hubs(self, centrality_measures: Dict[str, Dict[str, float]]) -> List[Tuple[str, float]]:
        """Find hub nodes with high degree centrality."""
        degree = centrality_measures.get('degree', {})
        if not degree:
            return []
        
        # Nodes with degree > average + std are considered hubs
        values = list(degree.values())
        if not values:
            return []
        
        mean_degree = np.mean(values)
        std_degree = np.std(values)
        threshold = mean_degree + std_degree
        
        hubs = [(node, score) for node, score in degree.items() if score > threshold]
        
        return sorted(hubs, key=lambda x: x[1], reverse=True)


class CommunityDetector:
    """Detects communities and clusters in the dependency graph."""
    
    def __init__(self):
        pass
    
    def detect_communities(self, graph: nx.DiGraph) -> Tuple[List[CommunityCluster], float]:
        """Detect communities in the graph using modularity optimization."""
        logger.info("Detecting communities in dependency graph")
        
        if graph.number_of_nodes() < 3:
            return [], 0.0
        
        try:
            # Convert to undirected for community detection
            undirected_graph = graph.to_undirected()
            
            # Use Louvain method for community detection
            communities = self._louvain_communities(undirected_graph)
            
            # Calculate modularity
            modularity = self._calculate_modularity(undirected_graph, communities)
            
            # Create community clusters
            clusters = []
            for i, community in enumerate(communities):
                if len(community) >= 2:  # Only include non-trivial communities
                    cluster = CommunityCluster(
                        cluster_id=i,
                        node_ids=list(community),
                        modularity_score=modularity,
                        description=f"Community {i} with {len(community)} nodes",
                        cluster_type=self._classify_community_type(community, graph)
                    )
                    clusters.append(cluster)
            
            logger.info(f"Found {len(clusters)} communities with modularity {modularity:.3f}")
            return clusters, modularity
            
        except Exception as e:
            logger.warning(f"Community detection failed: {e}")
            return [], 0.0
    
    def _louvain_communities(self, graph: nx.Graph) -> List[Set[str]]:
        """Apply Louvain algorithm for community detection."""
        try:
            # Try to use networkx community detection if available
            if hasattr(nx, 'community'):
                communities = nx.community.greedy_modularity_communities(graph)
                return [set(community) for community in communities]
            else:
                # Fallback to simple connected components
                return [set(component) for component in nx.connected_components(graph)]
        except Exception:
            # Last resort: each node is its own community
            return [{node} for node in graph.nodes()]
    
    def _calculate_modularity(self, graph: nx.Graph, communities: List[Set[str]]) -> float:
        """Calculate modularity score for community partitioning."""
        try:
            if hasattr(nx, 'community'):
                # Create partition dictionary
                partition = {}
                for i, community in enumerate(communities):
                    for node in community:
                        partition[node] = i
                
                return nx.community.modularity(graph, communities)
            else:
                # Simple modularity calculation
                total_edges = graph.number_of_edges()
                if total_edges == 0:
                    return 0.0
                
                modularity = 0.0
                for community in communities:
                    subgraph = graph.subgraph(community)
                    internal_edges = subgraph.number_of_edges()
                    community_degree = sum(graph.degree(node) for node in community)
                    expected = (community_degree ** 2) / (4 * total_edges)
                    modularity += (internal_edges / total_edges) - (expected / total_edges)
                
                return modularity
        except Exception:
            return 0.0
    
    def _classify_community_type(self, community: Set[str], graph: nx.DiGraph) -> str:
        """Classify the type of community based on node characteristics."""
        node_types = []
        file_paths = set()
        
        for node_id in community:
            node_data = graph.nodes.get(node_id, {})
            node_type = node_data.get('node_type', 'unknown')
            node_types.append(node_type)
            
            file_path = node_data.get('file_path', '')
            if file_path:
                file_paths.add(file_path.split('/')[0] if '/' in file_path else file_path)
        
        # Classify based on patterns
        if len(file_paths) == 1:
            return 'file_module'
        elif all(nt == 'component' for nt in node_types):
            return 'feature_cluster'
        elif len(file_paths) > len(community) * 0.7:
            return 'layer_module'
        else:
            return 'mixed_community'


class ArchitecturalAnalyzer:
    """Analyzes architectural patterns and layering in the codebase."""
    
    def __init__(self):
        pass
    
    def detect_layers(self, graph: nx.DiGraph) -> List[ArchitecturalLayer]:
        """Detect architectural layers in the dependency graph."""
        logger.info("Detecting architectural layers")
        
        if graph.number_of_nodes() < 3:
            return []
        
        try:
            # Use topological sorting to identify layers
            layers = self._topological_layering(graph)
            
            # Create layer objects
            layer_objects = []
            for i, layer_nodes in enumerate(layers):
                if layer_nodes:  # Only create non-empty layers
                    layer = ArchitecturalLayer(
                        layer_id=i,
                        layer_name=self._generate_layer_name(i, layer_nodes, graph),
                        node_ids=list(layer_nodes),
                        layer_level=i,
                        dependencies_up=[],  # Will be calculated
                        dependencies_down=[]  # Will be calculated
                    )
                    layer_objects.append(layer)
            
            # Calculate inter-layer dependencies
            self._calculate_layer_dependencies(layer_objects, graph)
            
            logger.info(f"Detected {len(layer_objects)} architectural layers")
            return layer_objects
            
        except Exception as e:
            logger.warning(f"Layer detection failed: {e}")
            return []
    
    def _topological_layering(self, graph: nx.DiGraph) -> List[Set[str]]:
        """Use topological sorting to identify dependency layers."""
        layers = []
        remaining_graph = graph.copy()
        
        while remaining_graph.nodes():
            # Find nodes with no incoming edges (current layer)
            current_layer = set()
            for node in remaining_graph.nodes():
                if remaining_graph.in_degree(node) == 0:
                    current_layer.add(node)
            
            if not current_layer:
                # Handle cycles by breaking them
                current_layer = set([min(remaining_graph.nodes(), 
                                       key=lambda n: remaining_graph.in_degree(n))])
            
            layers.append(current_layer)
            remaining_graph.remove_nodes_from(current_layer)
        
        return layers
    
    def _generate_layer_name(self, layer_index: int, layer_nodes: Set[str], 
                           graph: nx.DiGraph) -> str:
        """Generate descriptive name for a layer."""
        # Analyze node characteristics to infer layer purpose
        file_paths = []
        component_types = []
        
        for node_id in layer_nodes:
            node_data = graph.nodes.get(node_id, {})
            file_path = node_data.get('file_path', '')
            component_type = node_data.get('component_type', '')
            
            if file_path:
                file_paths.append(file_path)
            if component_type:
                component_types.append(component_type)
        
        # Infer layer name from patterns
        if any('model' in fp.lower() for fp in file_paths):
            return f"Data Layer {layer_index}"
        elif any('controller' in fp.lower() for fp in file_paths):
            return f"Controller Layer {layer_index}"
        elif any('view' in fp.lower() for fp in file_paths):
            return f"Presentation Layer {layer_index}"
        elif any('service' in fp.lower() for fp in file_paths):
            return f"Service Layer {layer_index}"
        elif any('util' in fp.lower() for fp in file_paths):
            return f"Utility Layer {layer_index}"
        else:
            return f"Layer {layer_index}"
    
    def _calculate_layer_dependencies(self, layers: List[ArchitecturalLayer], 
                                     graph: nx.DiGraph) -> None:
        """Calculate dependencies between layers."""
        for i, layer in enumerate(layers):
            for j, other_layer in enumerate(layers):
                if i != j:
                    # Check if there are edges between layers
                    has_dependency = False
                    for node1 in layer.node_ids:
                        for node2 in other_layer.node_ids:
                            if graph.has_edge(node1, node2):
                                has_dependency = True
                                break
                        if has_dependency:
                            break
                    
                    if has_dependency:
                        if j > i:  # Dependency on higher layer
                            layer.dependencies_up.append(j)
                        else:  # Dependency on lower layer
                            layer.dependencies_down.append(j)
    
    def find_critical_paths(self, graph: nx.DiGraph) -> List[CriticalPath]:
        """Find critical paths in the dependency graph."""
        logger.info("Finding critical paths in dependency graph")
        
        paths = []
        
        try:
            # Find longest paths (considering complexity as weight)
            longest_paths = self._find_longest_paths(graph)
            paths.extend(longest_paths)
            
            # Find bottleneck paths (high betweenness nodes)
            bottleneck_paths = self._find_bottleneck_paths(graph)
            paths.extend(bottleneck_paths)
            
            logger.info(f"Found {len(paths)} critical paths")
            return paths
            
        except Exception as e:
            logger.warning(f"Critical path analysis failed: {e}")
            return []
    
    def _find_longest_paths(self, graph: nx.DiGraph) -> List[CriticalPath]:
        """Find longest paths in the graph considering complexity."""
        paths = []
        
        try:
            # Find strongly connected components first
            sccs = list(nx.strongly_connected_components(graph))
            
            # For each SCC, find longest path
            for i, scc in enumerate(sccs):
                if len(scc) > 1:
                    subgraph = graph.subgraph(scc)
                    
                    # Use approximation for longest path (NP-hard problem)
                    try:
                        # Simple heuristic: start from node with highest complexity
                        start_node = max(scc, key=lambda n: graph.nodes.get(n, {}).get('complexity', 0))
                        
                        # Use DFS to find a long path
                        path = self._dfs_longest_path(subgraph, start_node, max_depth=10)
                        
                        if len(path) > 2:
                            total_complexity = sum(
                                graph.nodes.get(node, {}).get('complexity', 1) for node in path
                            )
                            
                            critical_path = CriticalPath(
                                path_id=len(paths),
                                node_sequence=path,
                                path_type='longest',
                                total_complexity=total_complexity,
                                description=f"Longest path in component {i} ({len(path)} nodes)"
                            )
                            paths.append(critical_path)
                    except Exception:
                        continue
            
            return paths[:5]  # Return top 5 longest paths
            
        except Exception:
            return []
    
    def _dfs_longest_path(self, graph: nx.DiGraph, start: str, max_depth: int) -> List[str]:
        """Use DFS to find a long path (approximation)."""
        visited = set()
        longest_path = []
        
        def dfs(node: str, current_path: List[str], depth: int) -> None:
            nonlocal longest_path
            
            if depth > max_depth or node in visited:
                if len(current_path) > len(longest_path):
                    longest_path = current_path.copy()
                return
            
            visited.add(node)
            current_path.append(node)
            
            # Explore neighbors
            neighbors = list(graph.successors(node))
            if neighbors:
                for neighbor in neighbors:
                    dfs(neighbor, current_path, depth + 1)
            else:
                # End of path
                if len(current_path) > len(longest_path):
                    longest_path = current_path.copy()
            
            current_path.pop()
            visited.remove(node)
        
        dfs(start, [], 0)
        return longest_path
    
    def _find_bottleneck_paths(self, graph: nx.DiGraph) -> List[CriticalPath]:
        """Find paths through bottleneck nodes."""
        paths = []
        
        try:
            # Calculate betweenness centrality
            betweenness = nx.betweenness_centrality(graph)
            
            # Find high betweenness nodes
            high_betweenness = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:3]
            
            for i, (bottleneck_node, centrality) in enumerate(high_betweenness):
                if centrality > 0.1:  # Significant bottleneck
                    # Find paths through this bottleneck
                    predecessors = list(graph.predecessors(bottleneck_node))
                    successors = list(graph.successors(bottleneck_node))
                    
                    if predecessors and successors:
                        # Create path through bottleneck
                        path = [predecessors[0], bottleneck_node, successors[0]]
                        
                        total_complexity = sum(
                            graph.nodes.get(node, {}).get('complexity', 1) for node in path
                        )
                        
                        critical_path = CriticalPath(
                            path_id=len(paths),
                            node_sequence=path,
                            path_type='bottleneck',
                            total_complexity=total_complexity,
                            description=f"Bottleneck path through {bottleneck_node} (centrality: {centrality:.3f})"
                        )
                        paths.append(critical_path)
            
            return paths
            
        except Exception:
            return []


class DependencyAnalyzer:
    """Analyzes dependency patterns and violations."""
    
    def __init__(self):
        pass
    
    def analyze_dependencies(self, graph: nx.DiGraph) -> Dict[str, Any]:
        """Analyze dependency patterns and find violations."""
        logger.info("Analyzing dependency patterns")
        
        analysis = {
            'circular_dependencies': [],
            'dependency_violations': [],
            'coupling_metrics': {}
        }
        
        try:
            # Find circular dependencies
            analysis['circular_dependencies'] = self._find_circular_dependencies(graph)
            
            # Find dependency violations
            analysis['dependency_violations'] = self._find_dependency_violations(graph)
            
            # Calculate coupling metrics
            analysis['coupling_metrics'] = self._calculate_coupling_metrics(graph)
            
            logger.info(f"Dependency analysis complete: {len(analysis['circular_dependencies'])} cycles found")
            return analysis
            
        except Exception as e:
            logger.warning(f"Dependency analysis failed: {e}")
            return analysis
    
    def _find_circular_dependencies(self, graph: nx.DiGraph) -> List[List[str]]:
        """Find circular dependencies in the graph."""
        cycles = []
        
        try:
            # Find all simple cycles
            simple_cycles = list(nx.simple_cycles(graph))
            
            # Filter out trivial cycles and very long ones
            for cycle in simple_cycles:
                if 2 <= len(cycle) <= 10:  # Reasonable cycle length
                    cycles.append(cycle)
            
            # Sort by cycle length (shorter cycles are more problematic)
            cycles.sort(key=len)
            
            return cycles[:10]  # Return top 10 cycles
            
        except Exception:
            return []
    
    def _find_dependency_violations(self, graph: nx.DiGraph) -> List[str]:
        """Find architectural dependency violations."""
        violations = []
        
        try:
            # Look for upward dependencies in layers
            # This is a simplified heuristic
            
            for edge in graph.edges():
                source, target = edge
                source_data = graph.nodes.get(source, {})
                target_data = graph.nodes.get(target, {})
                
                source_file = source_data.get('file_path', '')
                target_file = target_data.get('file_path', '')
                
                # Check for potential layer violations
                if self._is_layer_violation(source_file, target_file):
                    violation = f"Layer violation: {source} -> {target}"
                    violations.append(violation)
            
            return violations[:20]  # Return top 20 violations
            
        except Exception:
            return []
    
    def _is_layer_violation(self, source_file: str, target_file: str) -> bool:
        """Check if dependency represents a layer violation."""
        # Simple heuristic based on directory names
        layer_hierarchy = ['view', 'controller', 'service', 'model', 'util']
        
        source_layer = None
        target_layer = None
        
        for i, layer in enumerate(layer_hierarchy):
            if layer in source_file.lower():
                source_layer = i
            if layer in target_file.lower():
                target_layer = i
        
        # Violation if higher layer depends on lower layer
        if source_layer is not None and target_layer is not None:
            return source_layer < target_layer
        
        return False
    
    def _calculate_coupling_metrics(self, graph: nx.DiGraph) -> Dict[str, float]:
        """Calculate various coupling metrics."""
        metrics = {}
        
        try:
            # Afferent coupling (Ca) - incoming dependencies
            # Efferent coupling (Ce) - outgoing dependencies
            
            total_nodes = graph.number_of_nodes()
            if total_nodes == 0:
                return metrics
            
            afferent_coupling = {}
            efferent_coupling = {}
            
            for node in graph.nodes():
                afferent_coupling[node] = graph.in_degree(node)
                efferent_coupling[node] = graph.out_degree(node)
            
            # Calculate average coupling
            avg_afferent = sum(afferent_coupling.values()) / total_nodes
            avg_efferent = sum(efferent_coupling.values()) / total_nodes
            
            # Instability metric (Ce / (Ca + Ce))
            instability_scores = []
            for node in graph.nodes():
                ca = afferent_coupling[node]
                ce = efferent_coupling[node]
                if ca + ce > 0:
                    instability = ce / (ca + ce)
                    instability_scores.append(instability)
            
            avg_instability = sum(instability_scores) / len(instability_scores) if instability_scores else 0
            
            metrics = {
                'average_afferent_coupling': avg_afferent,
                'average_efferent_coupling': avg_efferent,
                'average_instability': avg_instability,
                'total_dependencies': graph.number_of_edges(),
                'coupling_density': graph.number_of_edges() / (total_nodes * (total_nodes - 1)) if total_nodes > 1 else 0
            }
            
            return metrics
            
        except Exception:
            return {}


class GraphPatternAnalyzer:
    """Main graph-based pattern analyzer."""
    
    def __init__(self):
        self.graph_builder = GraphBuilder()
        self.centrality_analyzer = CentralityAnalyzer()
        self.community_detector = CommunityDetector()
        self.architectural_analyzer = ArchitecturalAnalyzer()
        self.dependency_analyzer = DependencyAnalyzer()
    
    def analyze_graph_patterns(self, repo_analysis: RepositoryAnalysisResult) -> GraphAnalysisResult:
        """
        Perform comprehensive graph-based pattern analysis.
        
        Args:
            repo_analysis: Repository analysis results from Prompt 1.2
            
        Returns:
            Complete graph analysis results
        """
        import time
        start_time = time.time()
        
        logger.info("Starting graph-based pattern analysis")
        
        # Build dependency graph
        graph = self.graph_builder.build_graph(repo_analysis)
        
        if graph.number_of_nodes() == 0:
            logger.warning("Empty graph - no nodes to analyze")
            return self._create_empty_result(time.time() - start_time)
        
        # Analyze centrality
        centrality_measures = self.centrality_analyzer.analyze_centrality(graph)
        important_nodes = self.centrality_analyzer.identify_important_nodes(centrality_measures)
        bottlenecks = self.centrality_analyzer.find_bottlenecks(centrality_measures)
        hubs = self.centrality_analyzer.find_hubs(centrality_measures)
        
        # Update node attributes with centrality measures
        self._update_node_centrality(graph, centrality_measures)
        
        # Detect communities
        communities, modularity = self.community_detector.detect_communities(graph)
        
        # Architectural analysis
        layers = self.architectural_analyzer.detect_layers(graph)
        critical_paths = self.architectural_analyzer.find_critical_paths(graph)
        
        # Dependency analysis
        dependency_analysis = self.dependency_analyzer.analyze_dependencies(graph)
        
        # Calculate graph metrics
        graph_metrics = self._calculate_graph_metrics(graph)
        
        # Extract nodes and edges for result
        nodes = [GraphNode(**graph.nodes[node_id]) for node_id in graph.nodes()]
        edges = [GraphEdge(u, v, graph.edges[u, v].get('edge_type', 'unknown'), 
                          graph.edges[u, v].get('weight', 1.0)) 
                for u, v in graph.edges()]
        
        analysis_time = time.time() - start_time
        
        result = GraphAnalysisResult(
            total_nodes=graph.number_of_nodes(),
            total_edges=graph.number_of_edges(),
            analysis_time=analysis_time,
            nodes=nodes,
            edges=edges,
            graph_density=graph_metrics['density'],
            average_clustering=graph_metrics['average_clustering'],
            number_of_components=graph_metrics['number_of_components'],
            diameter=graph_metrics.get('diameter'),
            radius=graph_metrics.get('radius'),
            communities=communities,
            modularity_score=modularity,
            layers=layers,
            critical_paths=critical_paths,
            most_central_nodes=important_nodes.get('degree', [])[:10],
            bottleneck_nodes=bottlenecks[:10],
            hub_nodes=hubs[:10],
            circular_dependencies=dependency_analysis['circular_dependencies'],
            dependency_violations=dependency_analysis['dependency_violations'],
            coupling_metrics=dependency_analysis['coupling_metrics']
        )
        
        logger.info(f"Graph pattern analysis complete in {analysis_time:.2f}s")
        logger.info(f"Graph: {result.total_nodes} nodes, {result.total_edges} edges, "
                   f"{len(communities)} communities, {len(layers)} layers")
        
        return result
    
    def _update_node_centrality(self, graph: nx.DiGraph, 
                               centrality_measures: Dict[str, Dict[str, float]]) -> None:
        """Update graph nodes with centrality measures."""
        for node_id in graph.nodes():
            node_data = graph.nodes[node_id]
            
            # Update with centrality measures
            node_data['degree_centrality'] = centrality_measures.get('degree', {}).get(node_id, 0.0)
            node_data['betweenness_centrality'] = centrality_measures.get('betweenness', {}).get(node_id, 0.0)
            node_data['closeness_centrality'] = centrality_measures.get('closeness', {}).get(node_id, 0.0)
            node_data['eigenvector_centrality'] = centrality_measures.get('eigenvector', {}).get(node_id, 0.0)
            
            # Calculate clustering coefficient for individual nodes
            try:
                node_data['clustering_coefficient'] = nx.clustering(graph.to_undirected(), node_id)
            except:
                node_data['clustering_coefficient'] = 0.0
    
    def _calculate_graph_metrics(self, graph: nx.DiGraph) -> Dict[str, Any]:
        """Calculate basic graph metrics."""
        metrics = {}
        
        try:
            # Graph density
            metrics['density'] = nx.density(graph)
            
            # Average clustering
            undirected = graph.to_undirected()
            metrics['average_clustering'] = nx.average_clustering(undirected)
            
            # Number of connected components
            metrics['number_of_components'] = nx.number_weakly_connected_components(graph)
            
            # Diameter and radius (for largest connected component)
            if nx.is_weakly_connected(graph):
                undirected_connected = undirected
            else:
                # Use largest connected component
                largest_cc = max(nx.weakly_connected_components(graph), key=len)
                undirected_connected = undirected.subgraph(largest_cc)
            
            try:
                if undirected_connected.number_of_nodes() > 1:
                    metrics['diameter'] = nx.diameter(undirected_connected)
                    metrics['radius'] = nx.radius(undirected_connected)
                else:
                    metrics['diameter'] = 0
                    metrics['radius'] = 0
            except:
                metrics['diameter'] = None
                metrics['radius'] = None
            
        except Exception as e:
            logger.warning(f"Graph metrics calculation failed: {e}")
            # Provide defaults
            metrics = {
                'density': 0.0,
                'average_clustering': 0.0,
                'number_of_components': 1,
                'diameter': None,
                'radius': None
            }
        
        return metrics
    
    def _create_empty_result(self, analysis_time: float) -> GraphAnalysisResult:
        """Create empty graph analysis result."""
        return GraphAnalysisResult(
            total_nodes=0,
            total_edges=0,
            analysis_time=analysis_time,
            nodes=[],
            edges=[],
            graph_density=0.0,
            average_clustering=0.0,
            number_of_components=0,
            diameter=None,
            radius=None,
            communities=[],
            modularity_score=0.0,
            layers=[],
            critical_paths=[],
            most_central_nodes=[],
            bottleneck_nodes=[],
            hub_nodes=[],
            circular_dependencies=[],
            dependency_violations=[],
            coupling_metrics={}
        )


# Main interface function
def analyze_graph_patterns(repo_analysis: RepositoryAnalysisResult) -> GraphAnalysisResult:
    """
    Analyze repository using graph-based pattern detection.
    
    Args:
        repo_analysis: Repository analysis results from repository_analyzer
        
    Returns:
        Complete graph analysis results
    """
    analyzer = GraphPatternAnalyzer()
    return analyzer.analyze_graph_patterns(repo_analysis)


def build_dependency_graph(repo_analysis: RepositoryAnalysisResult) -> nx.DiGraph:
    """
    Build dependency graph from repository analysis.
    
    Args:
        repo_analysis: Repository analysis results
        
    Returns:
        NetworkX directed graph representing dependencies
    """
    builder = GraphBuilder()
    return builder.build_graph(repo_analysis)
