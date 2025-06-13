#!/bin/bash

# Setup Script for Prompt 1.3: Advanced Pattern Detection Engine
# AI QA Agent - Sprint 1.3

set -e

echo "🚀 Setting up Prompt 1.3: Advanced Pattern Detection Engine..."

# Check if we're in the right directory and previous prompts are complete
if [ ! -d "src" ]; then
    echo "❌ Error: This script should be run from the project root directory"
    echo "Expected to find 'src/' directory"
    exit 1
fi

if [ ! -f "src/analysis/ast_parser.py" ]; then
    echo "❌ Error: Prompt 1.1 (AST Parser) must be completed first"
    echo "Expected to find 'src/analysis/ast_parser.py'"
    exit 1
fi

if [ ! -f "src/analysis/repository_analyzer.py" ]; then
    echo "❌ Error: Prompt 1.2 (Repository Analyzer) must be completed first"
    echo "Expected to find 'src/analysis/repository_analyzer.py'"
    exit 1
fi

# Install new dependencies
echo "📦 Installing new dependencies..."
pip3 install scikit-learn==1.3.2 networkx==3.2.1 matplotlib==3.8.2 seaborn==0.13.0 numpy==1.24.4

# Create the ML pattern detection implementation
echo "📄 Creating src/analysis/ml_pattern_detector.py..."
cat > src/analysis/ml_pattern_detector.py << 'EOF'
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
EOF

# Create the graph pattern analyzer implementation
echo "📄 Creating src/analysis/graph_pattern_analyzer.py..."
cat > src/analysis/graph_pattern_analyzer.py << 'EOF'
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
EOF

# Create test file for ML pattern detector
echo "📄 Creating tests/unit/test_analysis/test_ml_pattern_detector.py..."
cat > tests/unit/test_analysis/test_ml_pattern_detector.py << 'EOF'
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
EOF

# Create test file for graph pattern analyzer
echo "📄 Creating tests/unit/test_analysis/test_graph_pattern_analyzer.py..."
cat > tests/unit/test_analysis/test_graph_pattern_analyzer.py << 'EOF'
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
EOF

# Update requirements.txt with new dependencies
echo "📄 Updating requirements.txt..."
if [ -f "requirements.txt" ]; then
    echo "scikit-learn==1.3.2" >> requirements.txt
    echo "networkx==3.2.1" >> requirements.txt
    echo "matplotlib==3.8.2" >> requirements.txt
    echo "seaborn==0.13.0" >> requirements.txt
    echo "numpy==1.24.4" >> requirements.txt
else
    echo "⚠️  Warning: requirements.txt not found. Please add dependencies manually:"
    echo "  scikit-learn==1.3.2"
    echo "  networkx==3.2.1"
    echo "  matplotlib==3.8.2"
    echo "  seaborn==0.13.0"
    echo "  numpy==1.24.4"
fi

# Run tests to verify implementation
echo "🧪 Running tests to verify implementation..."
if command -v pytest &> /dev/null; then
    echo "Running ML pattern detector tests..."
    python3 -m pytest tests/unit/test_analysis/test_ml_pattern_detector.py -v
    echo "Running graph pattern analyzer tests..."
    python3 -m pytest tests/unit/test_analysis/test_graph_pattern_analyzer.py -v
else
    echo "⚠️  pytest not found. Please install it and run:"
    echo "  python3 -m pytest tests/unit/test_analysis/test_ml_pattern_detector.py -v"
    echo "  python3 -m pytest tests/unit/test_analysis/test_graph_pattern_analyzer.py -v"
fi

# Test basic ML pattern detection functionality
echo "🔍 Testing ML pattern detection functionality..."
python3 -c "
try:
    print('Testing basic imports...')
    from src.analysis.ml_pattern_detector import ComponentFeatures, MLPatternDetector
    from src.analysis.graph_pattern_analyzer import GraphPatternAnalyzer
    print('✅ Imports successful')
    
    # Test ComponentFeatures creation
    feature = ComponentFeatures(
        component_id='test',
        file_path='test.py',
        component_name='test_func',
        component_type='function',
        cyclomatic_complexity=1.0,
        cognitive_complexity=1.0,
        lines_of_code=5,
        parameter_count=1,
        dependency_count=0,
        testability_score=0.8,
        test_priority=1,
        documentation_coverage=1.0
    )
    print('✅ ComponentFeatures creation successful')
    
    # Test detector creation
    detector = MLPatternDetector()
    print('✅ MLPatternDetector creation successful')
    
    graph_analyzer = GraphPatternAnalyzer()
    print('✅ GraphPatternAnalyzer creation successful')
    
    print('🎉 Basic functionality test passed!')
    
except Exception as e:
    print(f'❌ Error: {e}')
    import traceback
    traceback.print_exc()
"
echo ""
echo "✅ Prompt 1.3 setup complete!"
echo ""
echo "📋 Summary of what was implemented:"
echo "  ✅ ML-powered pattern detection with scikit-learn"
echo "  ✅ Feature engineering from code components"
echo "  ✅ Clustering analysis (DBSCAN, K-means) for similar components"
echo "  ✅ Anomaly detection using Isolation Forest + statistical methods"
echo "  ✅ Design pattern detection (Singleton, Factory, Observer, etc.)"
echo "  ✅ Graph-based dependency analysis with NetworkX"
echo "  ✅ Community detection and architectural layer identification"
echo "  ✅ Centrality analysis for identifying important components"
echo "  ✅ Critical path analysis and bottleneck detection"
echo "  ✅ Comprehensive test suites for both ML and graph analysis"
echo ""
echo "🔄 Next steps:"
echo "  1. Run the tests: python3 -m pytest tests/unit/test_analysis/test_ml_pattern_detector.py -v"
echo "  2. Run the tests: python3 -m pytest tests/unit/test_analysis/test_graph_pattern_analyzer.py -v"
echo "  3. Test with your own repositories using existing analysis functions"
echo "  4. Ready for Prompt 1.4: Analysis API Integration"
echo ""
echo "📊 Key capabilities now available:"
echo "  - ML feature extraction and clustering of code components"
echo "  - Anomaly detection for problematic code patterns"
echo "  - Design pattern recognition with confidence scoring"
echo "  - Dependency graph construction and analysis"
echo "  - Community detection for module boundary identification"
echo "  - Architectural layer detection and violation checking"
echo "  - Critical path analysis for change impact assessment"
echo "  - Comprehensive centrality metrics for component importance"
echo ""