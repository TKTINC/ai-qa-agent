"""
ML-powered pattern detection engine for advanced code analysis.
Uses machine learning techniques for pattern recognition and anomaly detection.
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set, Union
from collections import defaultdict, Counter
import pickle
import json

# Machine Learning imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Import project components
from .ast_parser import CodeComponent, ComponentType, analyze_complexity_metrics
from .repository_analyzer import RepositoryAnalysisResult, FileAnalysisResult

logger = logging.getLogger(__name__)


@dataclass
class ComponentFeatures:
    """Feature representation of a code component for ML analysis."""
    component_id: str
    file_path: str
    component_name: str
    component_type: str
    
    # Structural features
    cyclomatic_complexity: float
    cognitive_complexity: float
    lines_of_code: int
    parameter_count: int
    dependency_count: int
    testability_score: float
    test_priority: int
    documentation_coverage: float
    
    # Derived features with defaults
    complexity_density: float = 0.0
    coupling_score: float = 0.0
    cohesion_score: float = 0.0
    
    # Text features (will be populated by TF-IDF)
    text_features: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            'component_id': self.component_id,
            'file_path': self.file_path,
            'component_name': self.component_name,
            'component_type': self.component_type,
            'cyclomatic_complexity': self.cyclomatic_complexity,
            'cognitive_complexity': self.cognitive_complexity,
            'lines_of_code': self.lines_of_code,
            'parameter_count': self.parameter_count,
            'dependency_count': self.dependency_count,
            'testability_score': self.testability_score,
            'test_priority': self.test_priority,
            'documentation_coverage': self.documentation_coverage,
            'complexity_density': self.complexity_density,
            'coupling_score': self.coupling_score,
            'cohesion_score': self.cohesion_score
        }
        if self.text_features is not None:
            result['text_features'] = self.text_features.tolist()
        return result


@dataclass
class PatternCluster:
    """Represents a cluster of similar components."""
    cluster_id: int
    component_ids: List[str]
    centroid_features: np.ndarray
    cluster_type: str  # 'similar_functions', 'similar_classes', etc.
    pattern_description: str
    confidence_score: float
    representative_component: Optional[str] = None
    
    @property
    def size(self) -> int:
        """Number of components in cluster."""
        return len(self.component_ids)


@dataclass
class CodeAnomaly:
    """Represents an anomalous code component."""
    component_id: str
    anomaly_score: float
    anomaly_type: str  # 'complexity', 'structure', 'naming', etc.
    description: str
    severity: str  # 'low', 'medium', 'high'
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'component_id': self.component_id,
            'anomaly_score': self.anomaly_score,
            'anomaly_type': self.anomaly_type,
            'description': self.description,
            'severity': self.severity,
            'recommendations': self.recommendations
        }


@dataclass
class DesignPattern:
    """Represents a detected design pattern."""
    pattern_name: str
    pattern_type: str  # 'creational', 'structural', 'behavioral'
    component_ids: List[str]
    confidence_score: float
    description: str
    evidence: List[str]  # Evidence for pattern detection
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'pattern_name': self.pattern_name,
            'pattern_type': self.pattern_type,
            'component_ids': self.component_ids,
            'confidence_score': self.confidence_score,
            'description': self.description,
            'evidence': self.evidence
        }


@dataclass
class MLAnalysisResult:
    """Complete ML-powered analysis results."""
    total_components: int
    feature_engineering_time: float
    clustering_time: float
    anomaly_detection_time: float
    pattern_detection_time: float
    
    # ML Results
    component_clusters: List[PatternCluster]
    detected_anomalies: List[CodeAnomaly]
    design_patterns: List[DesignPattern]
    
    # Feature Analysis
    feature_importance: Dict[str, float]
    dimensionality_reduction: Optional[Dict[str, Any]] = None
    
    # Quality Metrics
    clustering_quality: float  # Silhouette score
    anomaly_detection_rate: float
    pattern_confidence_avg: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'total_components': self.total_components,
            'feature_engineering_time': self.feature_engineering_time,
            'clustering_time': self.clustering_time,
            'anomaly_detection_time': self.anomaly_detection_time,
            'pattern_detection_time': self.pattern_detection_time,
            'component_clusters': [c.__dict__ for c in self.component_clusters],
            'detected_anomalies': [a.to_dict() for a in self.detected_anomalies],
            'design_patterns': [p.to_dict() for p in self.design_patterns],
            'feature_importance': self.feature_importance,
            'dimensionality_reduction': self.dimensionality_reduction,
            'clustering_quality': self.clustering_quality,
            'anomaly_detection_rate': self.anomaly_detection_rate,
            'pattern_confidence_avg': self.pattern_confidence_avg
        }


class FeatureEngineer:
    """Extracts and engineers features from code components for ML analysis."""
    
    def __init__(self, max_features: int = 1000):
        self.max_features = max_features
        self.text_vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2),
            lowercase=True,
            token_pattern=r'(?u)\b\w\w+\b'
        )
        self.scaler = StandardScaler()
        self.fitted = False
        
    def extract_features(self, components: List[CodeComponent]) -> List[ComponentFeatures]:
        """Extract features from code components."""
        if not components:
            return []
            
        logger.info(f"Extracting features from {len(components)} components")
        
        # Create component features
        features = []
        text_documents = []
        
        for i, component in enumerate(components):
            # Create structural features
            feature = ComponentFeatures(
                component_id=f"{component.file_path}:{component.name}",
                file_path=component.file_path,
                component_name=component.name,
                component_type=component.component_type.value,
                cyclomatic_complexity=float(component.cyclomatic_complexity),
                cognitive_complexity=float(component.cognitive_complexity),
                lines_of_code=component.lines_of_code,
                parameter_count=len(component.parameters),
                dependency_count=len(component.dependencies),
                testability_score=component.testability_score,
                test_priority=component.test_priority,
                documentation_coverage=component.documentation_coverage
            )
            
            # Calculate derived features
            feature.complexity_density = component.complexity_density
            feature.coupling_score = self._calculate_coupling_score(component)
            feature.cohesion_score = self._calculate_cohesion_score(component)
            
            features.append(feature)
            
            # Prepare text for TF-IDF
            text_doc = self._prepare_text_document(component)
            text_documents.append(text_doc)
        
        # Extract text features using TF-IDF
        if text_documents:
            try:
                if not self.fitted:
                    text_features_matrix = self.text_vectorizer.fit_transform(text_documents)
                    self.fitted = True
                else:
                    text_features_matrix = self.text_vectorizer.transform(text_documents)
                
                # Assign text features to components
                for i, feature in enumerate(features):
                    feature.text_features = text_features_matrix[i].toarray().flatten()
                    
            except Exception as e:
                logger.warning(f"Text feature extraction failed: {e}")
                # Assign zero vectors if text extraction fails
                for feature in features:
                    feature.text_features = np.zeros(self.max_features)
        
        logger.info(f"Feature extraction complete: {len(features)} feature vectors")
        return features
    
    def _calculate_coupling_score(self, component: CodeComponent) -> float:
        """Calculate coupling score based on dependencies."""
        # Simple heuristic: normalize dependency count
        max_reasonable_deps = 10
        return min(1.0, len(component.dependencies) / max_reasonable_deps)
    
    def _calculate_cohesion_score(self, component: CodeComponent) -> float:
        """Calculate cohesion score based on component characteristics."""
        # Heuristic: well-documented, focused components have higher cohesion
        base_score = 0.5
        
        # Bonus for documentation
        if component.docstring:
            base_score += 0.2
        
        # Penalty for high complexity
        if component.cyclomatic_complexity > 10:
            base_score -= 0.3
        
        # Bonus for reasonable size
        if 5 <= component.lines_of_code <= 50:
            base_score += 0.2
            
        return max(0.0, min(1.0, base_score))
    
    def _prepare_text_document(self, component: CodeComponent) -> str:
        """Prepare text document for TF-IDF analysis."""
        text_parts = []
        
        # Component name (split camelCase/snake_case)
        name_parts = self._split_identifier(component.name)
        text_parts.extend(name_parts)
        
        # Component type
        text_parts.append(component.component_type.value)
        
        # Parameter names
        for param in component.parameters:
            param_parts = self._split_identifier(param.name)
            text_parts.extend(param_parts)
        
        # Docstring (if available)
        if component.docstring:
            # Clean and tokenize docstring
            cleaned_docstring = component.docstring.lower().replace('\n', ' ')
            text_parts.append(cleaned_docstring)
        
        # Function calls (for semantic similarity)
        for call in component.function_calls[:5]:  # Limit to avoid noise
            call_parts = self._split_identifier(call)
            text_parts.extend(call_parts)
        
        return ' '.join(text_parts)
    
    def _split_identifier(self, identifier: str) -> List[str]:
        """Split identifier into meaningful parts."""
        import re
        
        # Split camelCase and PascalCase
        parts = re.sub(r'(?<!^)(?=[A-Z])', ' ', identifier).split()
        
        # Split snake_case
        all_parts = []
        for part in parts:
            all_parts.extend(part.split('_'))
        
        # Filter out short/empty parts
        return [part.lower() for part in all_parts if len(part) > 1]
    
    def get_feature_matrix(self, features: List[ComponentFeatures]) -> Tuple[np.ndarray, List[str]]:
        """Get feature matrix for ML algorithms."""
        if not features:
            return np.array([]), []
        
        # Structural features
        structural_features = []
        feature_names = [
            'cyclomatic_complexity', 'cognitive_complexity', 'lines_of_code',
            'parameter_count', 'dependency_count', 'testability_score',
            'test_priority', 'documentation_coverage', 'complexity_density',
            'coupling_score', 'cohesion_score'
        ]
        
        for feature in features:
            row = [
                feature.cyclomatic_complexity, feature.cognitive_complexity,
                feature.lines_of_code, feature.parameter_count, feature.dependency_count,
                feature.testability_score, feature.test_priority, feature.documentation_coverage,
                feature.complexity_density, feature.coupling_score, feature.cohesion_score
            ]
            structural_features.append(row)
        
        structural_matrix = np.array(structural_features)
        
        # Combine with text features if available
        if features[0].text_features is not None:
            text_matrix = np.array([f.text_features for f in features])
            combined_matrix = np.hstack([structural_matrix, text_matrix])
            
            # Add text feature names
            text_feature_names = [f'text_feature_{i}' for i in range(text_matrix.shape[1])]
            all_feature_names = feature_names + text_feature_names
        else:
            combined_matrix = structural_matrix
            all_feature_names = feature_names
        
        # Scale features
        if not hasattr(self.scaler, 'scale_'):
            scaled_matrix = self.scaler.fit_transform(combined_matrix)
        else:
            scaled_matrix = self.scaler.transform(combined_matrix)
        
        return scaled_matrix, all_feature_names


class ClusteringAnalyzer:
    """Performs clustering analysis to find similar code patterns."""
    
    def __init__(self):
        self.dbscan_model = None
        self.kmeans_model = None
        
    def analyze_clusters(self, features: List[ComponentFeatures], 
                        feature_matrix: np.ndarray) -> List[PatternCluster]:
        """Perform clustering analysis on components."""
        if len(features) < 3:
            logger.warning("Too few components for clustering analysis")
            return []
        
        logger.info(f"Performing clustering analysis on {len(features)} components")
        
        clusters = []
        
        # DBSCAN clustering for density-based grouping
        dbscan_clusters = self._perform_dbscan(features, feature_matrix)
        clusters.extend(dbscan_clusters)
        
        # K-means clustering for centroid-based grouping
        kmeans_clusters = self._perform_kmeans(features, feature_matrix)
        clusters.extend(kmeans_clusters)
        
        # Component type clustering (group similar types)
        type_clusters = self._cluster_by_type(features)
        clusters.extend(type_clusters)
        
        logger.info(f"Clustering analysis complete: {len(clusters)} clusters found")
        return clusters
    
    def _perform_dbscan(self, features: List[ComponentFeatures], 
                       feature_matrix: np.ndarray) -> List[PatternCluster]:
        """Perform DBSCAN clustering."""
        try:
            # Optimize DBSCAN parameters
            n_samples = len(features)
            eps = 0.5 if n_samples > 20 else 0.3
            min_samples = max(2, min(5, n_samples // 10))
            
            self.dbscan_model = DBSCAN(eps=eps, min_samples=min_samples)
            cluster_labels = self.dbscan_model.fit_predict(feature_matrix)
            
            return self._create_clusters_from_labels(
                features, cluster_labels, "dbscan", "Density-based similar components"
            )
            
        except Exception as e:
            logger.warning(f"DBSCAN clustering failed: {e}")
            return []
    
    def _perform_kmeans(self, features: List[ComponentFeatures], 
                       feature_matrix: np.ndarray) -> List[PatternCluster]:
        """Perform K-means clustering."""
        try:
            # Determine optimal number of clusters
            n_samples = len(features)
            n_clusters = min(max(2, n_samples // 5), 8)  # 2-8 clusters
            
            self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = self.kmeans_model.fit_predict(feature_matrix)
            
            return self._create_clusters_from_labels(
                features, cluster_labels, "kmeans", "Centroid-based similar components"
            )
            
        except Exception as e:
            logger.warning(f"K-means clustering failed: {e}")
            return []
    
    def _cluster_by_type(self, features: List[ComponentFeatures]) -> List[PatternCluster]:
        """Create clusters based on component types."""
        type_groups = defaultdict(list)
        
        for i, feature in enumerate(features):
            type_groups[feature.component_type].append(i)
        
        clusters = []
        for comp_type, indices in type_groups.items():
            if len(indices) >= 2:  # Only create cluster if multiple components
                cluster = PatternCluster(
                    cluster_id=len(clusters),
                    component_ids=[features[i].component_id for i in indices],
                    centroid_features=np.array([]),  # Not applicable for type clustering
                    cluster_type="component_type",
                    pattern_description=f"All {comp_type} components",
                    confidence_score=1.0  # High confidence for type grouping
                )
                clusters.append(cluster)
        
        return clusters
    
    def _create_clusters_from_labels(self, features: List[ComponentFeatures], 
                                   labels: np.ndarray, cluster_type: str,
                                   description: str) -> List[PatternCluster]:
        """Create PatternCluster objects from clustering labels."""
        clusters = []
        unique_labels = set(labels)
        
        for label in unique_labels:
            if label == -1:  # Skip noise points in DBSCAN
                continue
                
            # Get components in this cluster
            cluster_indices = np.where(labels == label)[0]
            if len(cluster_indices) < 2:  # Skip singleton clusters
                continue
                
            component_ids = [features[i].component_id for i in cluster_indices]
            
            # Calculate cluster confidence based on size and cohesion
            cluster_size = len(cluster_indices)
            confidence = min(1.0, cluster_size / 10.0)  # Larger clusters = higher confidence
            
            cluster = PatternCluster(
                cluster_id=len(clusters),
                component_ids=component_ids,
                centroid_features=np.array([]),  # Will be calculated if needed
                cluster_type=cluster_type,
                pattern_description=f"{description} (size: {cluster_size})",
                confidence_score=confidence,
                representative_component=component_ids[0] if component_ids else None
            )
            clusters.append(cluster)
        
        return clusters


class AnomalyDetector:
    """Detects anomalous code components using ML techniques."""
    
    def __init__(self):
        self.isolation_forest = IsolationForest(
            contamination=0.1,  # Expect 10% anomalies
            random_state=42
        )
        
    def detect_anomalies(self, features: List[ComponentFeatures], 
                        feature_matrix: np.ndarray) -> List[CodeAnomaly]:
        """Detect anomalous components."""
        if len(features) < 5:
            logger.warning("Too few components for anomaly detection")
            return []
        
        logger.info(f"Performing anomaly detection on {len(features)} components")
        
        anomalies = []
        
        # Isolation Forest anomaly detection
        isolation_anomalies = self._detect_isolation_anomalies(features, feature_matrix)
        anomalies.extend(isolation_anomalies)
        
        # Statistical outlier detection
        statistical_anomalies = self._detect_statistical_anomalies(features)
        anomalies.extend(statistical_anomalies)
        
        # Pattern-based anomaly detection
        pattern_anomalies = self._detect_pattern_anomalies(features)
        anomalies.extend(pattern_anomalies)
        
        # Remove duplicates and sort by severity
        unique_anomalies = self._deduplicate_anomalies(anomalies)
        
        logger.info(f"Anomaly detection complete: {len(unique_anomalies)} anomalies found")
        return unique_anomalies
    
    def _detect_isolation_anomalies(self, features: List[ComponentFeatures], 
                                   feature_matrix: np.ndarray) -> List[CodeAnomaly]:
        """Use Isolation Forest for anomaly detection."""
        try:
            anomaly_scores = self.isolation_forest.fit_predict(feature_matrix)
            outlier_scores = self.isolation_forest.score_samples(feature_matrix)
            
            anomalies = []
            for i, (score, is_anomaly) in enumerate(zip(outlier_scores, anomaly_scores)):
                if is_anomaly == -1:  # Anomaly detected
                    severity = self._calculate_severity(-score)  # More negative = more anomalous
                    
                    anomaly = CodeAnomaly(
                        component_id=features[i].component_id,
                        anomaly_score=float(-score),
                        anomaly_type="structural",
                        description=f"Component shows unusual structural patterns",
                        severity=severity,
                        recommendations=self._generate_structural_recommendations(features[i])
                    )
                    anomalies.append(anomaly)
            
            return anomalies
            
        except Exception as e:
            logger.warning(f"Isolation Forest anomaly detection failed: {e}")
            return []
    
    def _detect_statistical_anomalies(self, features: List[ComponentFeatures]) -> List[CodeAnomaly]:
        """Detect statistical outliers in complexity metrics."""
        anomalies = []
        
        # Extract complexity values
        complexities = [f.cyclomatic_complexity for f in features]
        
        if not complexities:
            return anomalies
        
        # Calculate statistical thresholds
        mean_complexity = np.mean(complexities)
        std_complexity = np.std(complexities)
        threshold = mean_complexity + 2 * std_complexity  # 2 standard deviations
        
        for feature in features:
            if feature.cyclomatic_complexity > threshold:
                severity = "high" if feature.cyclomatic_complexity > threshold * 1.5 else "medium"
                
                anomaly = CodeAnomaly(
                    component_id=feature.component_id,
                    anomaly_score=feature.cyclomatic_complexity / threshold,
                    anomaly_type="complexity",
                    description=f"Unusually high complexity ({feature.cyclomatic_complexity:.1f})",
                    severity=severity,
                    recommendations=self._generate_complexity_recommendations(feature)
                )
                anomalies.append(anomaly)
        
        return anomalies
    
    def _detect_pattern_anomalies(self, features: List[ComponentFeatures]) -> List[CodeAnomaly]:
        """Detect pattern-based anomalies."""
        anomalies = []
        
        for feature in features:
            # Large component with no documentation
            if (feature.lines_of_code > 50 and 
                feature.documentation_coverage == 0.0):
                
                anomaly = CodeAnomaly(
                    component_id=feature.component_id,
                    anomaly_score=feature.lines_of_code / 50.0,
                    anomaly_type="documentation",
                    description=f"Large component ({feature.lines_of_code} LOC) lacks documentation",
                    severity="medium",
                    recommendations=["Add comprehensive docstring", "Consider breaking into smaller functions"]
                )
                anomalies.append(anomaly)
            
            # High coupling
            if feature.coupling_score > 0.8:
                anomaly = CodeAnomaly(
                    component_id=feature.component_id,
                    anomaly_score=feature.coupling_score,
                    anomaly_type="coupling",
                    description="High coupling detected",
                    severity="medium",
                    recommendations=["Reduce dependencies", "Apply dependency injection", "Consider interface segregation"]
                )
                anomalies.append(anomaly)
            
            # Many parameters
            if feature.parameter_count > 6:
                anomaly = CodeAnomaly(
                    component_id=feature.component_id,
                    anomaly_score=feature.parameter_count / 6.0,
                    anomaly_type="parameters",
                    description=f"Too many parameters ({feature.parameter_count})",
                    severity="low",
                    recommendations=["Use parameter objects", "Apply builder pattern", "Split function responsibility"]
                )
                anomalies.append(anomaly)
        
        return anomalies
    
    def _calculate_severity(self, score: float) -> str:
        """Calculate severity based on anomaly score."""
        if score > 0.7:
            return "high"
        elif score > 0.4:
            return "medium"
        else:
            return "low"
    
    def _generate_structural_recommendations(self, feature: ComponentFeatures) -> List[str]:
        """Generate recommendations for structural anomalies."""
        recommendations = ["Review component structure and design"]
        
        if feature.cyclomatic_complexity > 10:
            recommendations.append("Reduce cyclomatic complexity through refactoring")
        
        if feature.lines_of_code > 100:
            recommendations.append("Consider breaking into smaller functions")
        
        if feature.coupling_score > 0.7:
            recommendations.append("Reduce external dependencies")
        
        return recommendations
    
    def _generate_complexity_recommendations(self, feature: ComponentFeatures) -> List[str]:
        """Generate recommendations for complexity anomalies."""
        recommendations = ["Refactor to reduce complexity"]
        
        if feature.parameter_count > 5:
            recommendations.append("Reduce number of parameters")
        
        if feature.dependency_count > 8:
            recommendations.append("Minimize external dependencies")
        
        recommendations.extend([
            "Extract complex logic into separate functions",
            "Apply single responsibility principle",
            "Add comprehensive unit tests"
        ])
        
        return recommendations
    
    def _deduplicate_anomalies(self, anomalies: List[CodeAnomaly]) -> List[CodeAnomaly]:
        """Remove duplicate anomalies and sort by severity."""
        # Group by component_id
        anomaly_dict = {}
        for anomaly in anomalies:
            key = anomaly.component_id
            if key not in anomaly_dict or anomaly.anomaly_score > anomaly_dict[key].anomaly_score:
                anomaly_dict[key] = anomaly
        
        # Sort by severity and score
        severity_order = {"high": 3, "medium": 2, "low": 1}
        sorted_anomalies = sorted(
            anomaly_dict.values(),
            key=lambda x: (severity_order.get(x.severity, 0), x.anomaly_score),
            reverse=True
        )
        
        return sorted_anomalies


class DesignPatternDetector:
    """Detects design patterns using ML-enhanced heuristics."""
    
    def __init__(self):
        self.pattern_detectors = {
            'Singleton': self._detect_singleton,
            'Factory': self._detect_factory,
            'Observer': self._detect_observer,
            'Strategy': self._detect_strategy,
            'Builder': self._detect_builder,
            'Decorator': self._detect_decorator,
            'Adapter': self._detect_adapter
        }
    
    def detect_patterns(self, features: List[ComponentFeatures], 
                       clusters: List[PatternCluster]) -> List[DesignPattern]:
        """Detect design patterns in the codebase."""
        logger.info(f"Detecting design patterns in {len(features)} components")
        
        patterns = []
        
        # Run each pattern detector
        for pattern_name, detector_func in self.pattern_detectors.items():
            try:
                detected_patterns = detector_func(features, clusters)
                patterns.extend(detected_patterns)
            except Exception as e:
                logger.warning(f"Pattern detection failed for {pattern_name}: {e}")
        
        # Validate and filter patterns
        validated_patterns = self._validate_patterns(patterns)
        
        logger.info(f"Design pattern detection complete: {len(validated_patterns)} patterns found")
        return validated_patterns
    
    def _detect_singleton(self, features: List[ComponentFeatures], 
                         clusters: List[PatternCluster]) -> List[DesignPattern]:
        """Detect Singleton pattern."""
        patterns = []
        
        for feature in features:
            evidence = []
            confidence = 0.0
            
            # Look for singleton indicators in name
            name_lower = feature.component_name.lower()
            if any(keyword in name_lower for keyword in ['singleton', 'instance', 'getinstance']):
                evidence.append(f"Name '{feature.component_name}' suggests singleton")
                confidence += 0.4
            
            # Look for typical singleton characteristics
            if feature.component_type == 'class':
                evidence.append("Class component (potential singleton container)")
                confidence += 0.2
            
            if feature.parameter_count == 0 and feature.component_type == 'method':
                evidence.append("Parameterless method (potential getInstance)")
                confidence += 0.3
            
            if confidence > 0.5:
                pattern = DesignPattern(
                    pattern_name="Singleton",
                    pattern_type="creational",
                    component_ids=[feature.component_id],
                    confidence_score=confidence,
                    description="Singleton pattern implementation detected",
                    evidence=evidence
                )
                patterns.append(pattern)
        
        return patterns
    
    def _detect_factory(self, features: List[ComponentFeatures], 
                       clusters: List[PatternCluster]) -> List[DesignPattern]:
        """Detect Factory pattern."""
        patterns = []
        
        # Look for factory indicators
        factory_candidates = []
        
        for feature in features:
            name_lower = feature.component_name.lower()
            evidence = []
            confidence = 0.0
            
            # Factory naming patterns
            if any(keyword in name_lower for keyword in ['factory', 'create', 'build', 'make']):
                evidence.append(f"Name '{feature.component_name}' suggests factory method")
                confidence += 0.5
            
            # Factory characteristics
            if feature.component_type in ['function', 'method']:
                if feature.parameter_count > 0:  # Factory methods typically take parameters
                    evidence.append("Method with parameters (typical factory signature)")
                    confidence += 0.2
                
                if 'create' in name_lower or 'build' in name_lower:
                    evidence.append("Creation verb in method name")
                    confidence += 0.3
            
            if confidence > 0.6:
                factory_candidates.append((feature, confidence, evidence))
        
        # Group related factory methods
        if len(factory_candidates) >= 2:
            # Create pattern for factory group
            all_ids = [candidate[0].component_id for candidate in factory_candidates]
            all_evidence = []
            for _, _, evidence in factory_candidates:
                all_evidence.extend(evidence)
            
            avg_confidence = sum(candidate[1] for candidate in factory_candidates) / len(factory_candidates)
            
            pattern = DesignPattern(
                pattern_name="Factory",
                pattern_type="creational",
                component_ids=all_ids,
                confidence_score=avg_confidence,
                description="Factory pattern implementation detected",
                evidence=list(set(all_evidence))  # Remove duplicates
            )
            patterns.append(pattern)
        
        return patterns
    
    def _detect_observer(self, features: List[ComponentFeatures], 
                        clusters: List[PatternCluster]) -> List[DesignPattern]:
        """Detect Observer pattern."""
        patterns = []
        
        observer_indicators = []
        
        for feature in features:
            name_lower = feature.component_name.lower()
            evidence = []
            confidence = 0.0
            
            # Observer naming patterns
            observer_keywords = ['observer', 'notify', 'subscribe', 'listen', 'update', 'event']
            matching_keywords = [kw for kw in observer_keywords if kw in name_lower]
            
            if matching_keywords:
                evidence.append(f"Observer keywords in name: {', '.join(matching_keywords)}")
                confidence += 0.4 * len(matching_keywords)
            
            # Observer method characteristics
            if feature.component_type in ['method', 'function']:
                if 'notify' in name_lower and feature.parameter_count >= 1:
                    evidence.append("Notify method with parameters")
                    confidence += 0.3
                
                if 'update' in name_lower:
                    evidence.append("Update method (observer interface)")
                    confidence += 0.3
            
            if confidence > 0.5:
                observer_indicators.append((feature, confidence, evidence))
        
        # If multiple observer components found, create pattern
        if len(observer_indicators) >= 2:
            all_ids = [indicator[0].component_id for indicator in observer_indicators]
            all_evidence = []
            for _, _, evidence in observer_indicators:
                all_evidence.extend(evidence)
            
            avg_confidence = sum(indicator[1] for indicator in observer_indicators) / len(observer_indicators)
            
            pattern = DesignPattern(
                pattern_name="Observer",
                pattern_type="behavioral",
                component_ids=all_ids,
                confidence_score=avg_confidence,
                description="Observer pattern implementation detected",
                evidence=list(set(all_evidence))
            )
            patterns.append(pattern)
        
        return patterns
    
    def _detect_strategy(self, features: List[ComponentFeatures], 
                        clusters: List[PatternCluster]) -> List[DesignPattern]:
        """Detect Strategy pattern."""
        patterns = []
        
        # Look for strategy-like clusters (similar interfaces, different implementations)
        for cluster in clusters:
            if cluster.size >= 3:  # Need multiple strategies
                cluster_features = [f for f in features if f.component_id in cluster.component_ids]
                
                # Check if components have similar signatures (same parameter count)
                if cluster_features:
                    param_counts = [f.parameter_count for f in cluster_features]
                    if len(set(param_counts)) <= 2:  # Similar parameter counts
                        evidence = [
                            f"Cluster of {cluster.size} components with similar signatures",
                            f"Similar parameter counts: {set(param_counts)}"
                        ]
                        
                        # Check for strategy naming
                        names = [f.component_name.lower() for f in cluster_features]
                        if any('strategy' in name or 'algorithm' in name for name in names):
                            evidence.append("Strategy/algorithm keywords in names")
                        
                        confidence = min(0.9, 0.3 + (cluster.size * 0.1))
                        
                        pattern = DesignPattern(
                            pattern_name="Strategy",
                            pattern_type="behavioral",
                            component_ids=cluster.component_ids,
                            confidence_score=confidence,
                            description="Strategy pattern implementation detected",
                            evidence=evidence
                        )
                        patterns.append(pattern)
        
        return patterns
    
    def _detect_builder(self, features: List[ComponentFeatures], 
                       clusters: List[PatternCluster]) -> List[DesignPattern]:
        """Detect Builder pattern."""
        patterns = []
        
        builder_components = []
        
        for feature in features:
            name_lower = feature.component_name.lower()
            evidence = []
            confidence = 0.0
            
            # Builder naming patterns
            if 'builder' in name_lower:
                evidence.append(f"Builder keyword in name '{feature.component_name}'")
                confidence += 0.5
            
            if 'build' in name_lower and feature.component_type in ['method', 'function']:
                evidence.append("Build method detected")
                confidence += 0.3
            
            # Builder method chaining indicators
            if (feature.component_type == 'method' and 
                feature.parameter_count <= 2):  # Typical builder method signature
                evidence.append("Method signature typical of builder pattern")
                confidence += 0.2
            
            if confidence > 0.4:
                builder_components.append((feature, confidence, evidence))
        
        # Group builder components
        if builder_components:
            all_ids = [comp[0].component_id for comp in builder_components]
            all_evidence = []
            for _, _, evidence in builder_components:
                all_evidence.extend(evidence)
            
            avg_confidence = sum(comp[1] for comp in builder_components) / len(builder_components)
            
            pattern = DesignPattern(
                pattern_name="Builder",
                pattern_type="creational",
                component_ids=all_ids,
                confidence_score=avg_confidence,
                description="Builder pattern implementation detected",
                evidence=list(set(all_evidence))
            )
            patterns.append(pattern)
        
        return patterns
    
    def _detect_decorator(self, features: List[ComponentFeatures], 
                         clusters: List[PatternCluster]) -> List[DesignPattern]:
        """Detect Decorator pattern."""
        patterns = []
        
        for feature in features:
            name_lower = feature.component_name.lower()
            evidence = []
            confidence = 0.0
            
            # Decorator naming patterns
            if 'decorator' in name_lower or 'decorate' in name_lower:
                evidence.append(f"Decorator keyword in name '{feature.component_name}'")
                confidence += 0.6
            
            if 'wrapper' in name_lower or 'wrap' in name_lower:
                evidence.append(f"Wrapper keyword in name '{feature.component_name}'")
                confidence += 0.4
            
            # Decorator characteristics
            if (feature.component_type in ['class', 'function'] and 
                feature.parameter_count >= 1):
                evidence.append("Component takes parameters (potential decorator)")
                confidence += 0.2
            
            if confidence > 0.5:
                pattern = DesignPattern(
                    pattern_name="Decorator",
                    pattern_type="structural",
                    component_ids=[feature.component_id],
                    confidence_score=confidence,
                    description="Decorator pattern implementation detected",
                    evidence=evidence
                )
                patterns.append(pattern)
        
        return patterns
    
    def _detect_adapter(self, features: List[ComponentFeatures], 
                       clusters: List[PatternCluster]) -> List[DesignPattern]:
        """Detect Adapter pattern."""
        patterns = []
        
        for feature in features:
            name_lower = feature.component_name.lower()
            evidence = []
            confidence = 0.0
            
            # Adapter naming patterns
            if 'adapter' in name_lower or 'adapt' in name_lower:
                evidence.append(f"Adapter keyword in name '{feature.component_name}'")
                confidence += 0.6
            
            if 'convert' in name_lower or 'transform' in name_lower:
                evidence.append(f"Conversion keyword in name '{feature.component_name}'")
                confidence += 0.3
            
            # Adapter characteristics (typically takes input and adapts it)
            if (feature.component_type in ['class', 'function'] and 
                feature.parameter_count >= 1):
                evidence.append("Takes parameters (potential adapter interface)")
                confidence += 0.2
            
            if confidence > 0.5:
                pattern = DesignPattern(
                    pattern_name="Adapter",
                    pattern_type="structural",
                    component_ids=[feature.component_id],
                    confidence_score=confidence,
                    description="Adapter pattern implementation detected",
                    evidence=evidence
                )
                patterns.append(pattern)
        
        return patterns
    
    def _validate_patterns(self, patterns: List[DesignPattern]) -> List[DesignPattern]:
        """Validate and filter detected patterns."""
        validated = []
        
        for pattern in patterns:
            # Filter by confidence threshold
            if pattern.confidence_score >= 0.4:
                validated.append(pattern)
        
        # Remove duplicate patterns (same components)
        seen_component_sets = set()
        final_patterns = []
        
        for pattern in validated:
            component_set = frozenset(pattern.component_ids)
            if component_set not in seen_component_sets:
                seen_component_sets.add(component_set)
                final_patterns.append(pattern)
        
        return final_patterns


class MLPatternDetector:
    """Main ML-powered pattern detection engine."""
    
    def __init__(self, max_features: int = 1000):
        self.feature_engineer = FeatureEngineer(max_features)
        self.clustering_analyzer = ClusteringAnalyzer()
        self.anomaly_detector = AnomalyDetector()
        self.pattern_detector = DesignPatternDetector()
    
    def analyze_patterns(self, repo_analysis: RepositoryAnalysisResult) -> MLAnalysisResult:
        """
        Perform comprehensive ML-powered pattern analysis.
        
        Args:
            repo_analysis: Repository analysis results from Prompt 1.2
            
        Returns:
            Complete ML analysis results
        """
        import time
        start_time = time.time()
        
        logger.info("Starting ML-powered pattern analysis")
        
        # Extract all components from repository analysis
        all_components = []
        for file_result in repo_analysis.file_results:
            if not file_result.has_error:
                all_components.extend(file_result.components)
        
        if not all_components:
            logger.warning("No components found for ML analysis")
            return self._create_empty_result()
        
        logger.info(f"Analyzing {len(all_components)} components with ML techniques")
        
        # Feature Engineering
        feature_start = time.time()
        component_features = self.feature_engineer.extract_features(all_components)
        feature_matrix, feature_names = self.feature_engineer.get_feature_matrix(component_features)
        feature_time = time.time() - feature_start
        
        # Clustering Analysis
        cluster_start = time.time()
        clusters = self.clustering_analyzer.analyze_clusters(component_features, feature_matrix)
        cluster_time = time.time() - cluster_start
        
        # Anomaly Detection
        anomaly_start = time.time()
        anomalies = self.anomaly_detector.detect_anomalies(component_features, feature_matrix)
        anomaly_time = time.time() - anomaly_start
        
        # Design Pattern Detection
        pattern_start = time.time()
        design_patterns = self.pattern_detector.detect_patterns(component_features, clusters)
        pattern_time = time.time() - pattern_start
        
        # Calculate analysis metrics
        analysis_metrics = self._calculate_analysis_metrics(
            component_features, feature_matrix, clusters, anomalies, design_patterns
        )
        
        total_time = time.time() - start_time
        
        result = MLAnalysisResult(
            total_components=len(all_components),
            feature_engineering_time=feature_time,
            clustering_time=cluster_time,
            anomaly_detection_time=anomaly_time,
            pattern_detection_time=pattern_time,
            component_clusters=clusters,
            detected_anomalies=anomalies,
            design_patterns=design_patterns,
            feature_importance=analysis_metrics['feature_importance'],
            dimensionality_reduction=analysis_metrics['dimensionality_reduction'],
            clustering_quality=analysis_metrics['clustering_quality'],
            anomaly_detection_rate=analysis_metrics['anomaly_detection_rate'],
            pattern_confidence_avg=analysis_metrics['pattern_confidence_avg']
        )
        
        logger.info(f"ML pattern analysis complete in {total_time:.2f}s")
        logger.info(f"Found: {len(clusters)} clusters, {len(anomalies)} anomalies, {len(design_patterns)} patterns")
        
        return result
    
    def _calculate_analysis_metrics(self, features: List[ComponentFeatures], 
                                   feature_matrix: np.ndarray,
                                   clusters: List[PatternCluster],
                                   anomalies: List[CodeAnomaly],
                                   patterns: List[DesignPattern]) -> Dict[str, Any]:
        """Calculate quality metrics for the analysis."""
        metrics = {}
        
        # Feature importance (simplified)
        if feature_matrix.size > 0:
            feature_variance = np.var(feature_matrix, axis=0)
            feature_names = [f'feature_{i}' for i in range(len(feature_variance))]
            metrics['feature_importance'] = dict(zip(feature_names, feature_variance.tolist()))
        else:
            metrics['feature_importance'] = {}
        
        # Clustering quality
        if len(clusters) > 0 and feature_matrix.size > 0:
            try:
                # Use DBSCAN clustering for silhouette score
                from sklearn.cluster import DBSCAN
                dbscan = DBSCAN(eps=0.5, min_samples=2)
                cluster_labels = dbscan.fit_predict(feature_matrix)
                
                if len(set(cluster_labels)) > 1:  # More than one cluster
                    silhouette = silhouette_score(feature_matrix, cluster_labels)
                    metrics['clustering_quality'] = float(silhouette)
                else:
                    metrics['clustering_quality'] = 0.0
            except:
                metrics['clustering_quality'] = 0.0
        else:
            metrics['clustering_quality'] = 0.0
        
        # Anomaly detection rate
        if features:
            metrics['anomaly_detection_rate'] = len(anomalies) / len(features)
        else:
            metrics['anomaly_detection_rate'] = 0.0
        
        # Pattern confidence average
        if patterns:
            metrics['pattern_confidence_avg'] = sum(p.confidence_score for p in patterns) / len(patterns)
        else:
            metrics['pattern_confidence_avg'] = 0.0
        
        # Dimensionality reduction info (PCA)
        if feature_matrix.size > 0 and feature_matrix.shape[1] > 2:
            try:
                pca = PCA(n_components=min(2, feature_matrix.shape[1]))
                pca_result = pca.fit_transform(feature_matrix)
                metrics['dimensionality_reduction'] = {
                    'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
                    'n_components': pca.n_components_,
                    'total_variance_explained': float(sum(pca.explained_variance_ratio_))
                }
            except:
                metrics['dimensionality_reduction'] = None
        else:
            metrics['dimensionality_reduction'] = None
        
        return metrics
    
    def _create_empty_result(self) -> MLAnalysisResult:
        """Create empty ML analysis result."""
        return MLAnalysisResult(
            total_components=0,
            feature_engineering_time=0.0,
            clustering_time=0.0,
            anomaly_detection_time=0.0,
            pattern_detection_time=0.0,
            component_clusters=[],
            detected_anomalies=[],
            design_patterns=[],
            feature_importance={},
            dimensionality_reduction=None,
            clustering_quality=0.0,
            anomaly_detection_rate=0.0,
            pattern_confidence_avg=0.0
        )


# Main interface function
def analyze_ml_patterns(repo_analysis: RepositoryAnalysisResult) -> MLAnalysisResult:
    """
    Analyze repository using ML-powered pattern detection.
    
    Args:
        repo_analysis: Repository analysis results from repository_analyzer
        
    Returns:
        Complete ML analysis results
    """
    detector = MLPatternDetector()
    return detector.analyze_patterns(repo_analysis)


def extract_component_features(components: List[CodeComponent]) -> List[ComponentFeatures]:
    """
    Extract ML features from code components.
    
    Args:
        components: List of code components to analyze
        
    Returns:
        List of feature representations
    """
    feature_engineer = FeatureEngineer()
    return feature_engineer.extract_features(components)
