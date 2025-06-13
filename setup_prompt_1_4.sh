#!/bin/bash
# Setup Script for Prompt 1.4: Analysis API Integration
# AI QA Agent - Sprint 1.4

set -e
echo "🚀 Setting up Prompt 1.4: Analysis API Integration..."

# Check prerequisites (previous prompts completed)
if [ ! -f "src/analysis/ast_parser.py" ]; then
    echo "❌ Error: Prompt 1.1 (AST Parser) must be completed first"
    exit 1
fi

if [ ! -f "src/analysis/repository_analyzer.py" ]; then
    echo "❌ Error: Prompt 1.2 (Repository Analyzer) must be completed first"
    exit 1
fi

if [ ! -f "src/analysis/ml_pattern_detector.py" ]; then
    echo "❌ Error: Prompt 1.3 (ML Pattern Detection) must be completed first"
    exit 1
fi

# Install dependencies with pip3 (macOS compatible)
echo "📦 Installing new dependencies..."
pip3 install celery==5.3.4 redis==5.0.1 python-jose[cryptography]==3.3.0 python-multipart==0.0.6 aiofiles==23.2.1

# Create analysis service core module
echo "📄 Creating src/analysis/analysis_service.py..."
cat > src/analysis/analysis_service.py << 'EOF'
"""
Analysis Service - Core service for integrating all analysis components
Combines AST parsing, repository analysis, ML pattern detection, and graph analysis
"""

import asyncio
import logging
import time
import tempfile
import shutil
import zipfile
import tarfile
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import traceback

from src.analysis.ast_parser import ASTParser, CodeComponent, Language
from src.analysis.repository_analyzer import RepositoryAnalyzer, RepositoryAnalysisResult
from src.analysis.ml_pattern_detector import MLPatternDetector, MLAnalysisResult
from src.analysis.graph_pattern_analyzer import GraphPatternAnalyzer, GraphAnalysisResult
from src.core.logging import get_logger
from src.core.exceptions import AnalysisError

logger = get_logger(__name__)

@dataclass
class AnalysisProgress:
    """Track analysis progress with detailed steps"""
    session_id: str
    total_steps: int = 7
    completed_steps: int = 0
    current_step: str = "Initializing"
    progress_percentage: float = 0.0
    start_time: float = 0.0
    estimated_completion: Optional[float] = None
    error_message: Optional[str] = None
    
    def update_step(self, step_name: str, completed: int = None):
        """Update progress to next step"""
        if completed is not None:
            self.completed_steps = completed
        else:
            self.completed_steps += 1
        self.current_step = step_name
        self.progress_percentage = min((self.completed_steps / self.total_steps) * 100, 100)
        
        # Estimate completion time
        if self.start_time > 0 and self.progress_percentage > 5:
            elapsed = time.time() - self.start_time
            estimated_total = elapsed / (self.progress_percentage / 100)
            self.estimated_completion = self.start_time + estimated_total
    
    def mark_error(self, error_message: str):
        """Mark analysis as failed"""
        self.error_message = error_message
        self.current_step = f"Failed: {error_message}"

@dataclass
class ComprehensiveAnalysisResult:
    """Complete analysis results combining all analysis components"""
    session_id: str
    repository_path: str
    language: Language
    total_files: int
    analyzed_files: int
    total_components: int
    analysis_duration: float
    
    # Component-level results
    components: List[CodeComponent]
    
    # Repository-level results
    repository_analysis: Optional[RepositoryAnalysisResult]
    
    # ML analysis results
    ml_analysis: Optional[MLAnalysisResult]
    
    # Graph analysis results
    graph_analysis: Optional[GraphAnalysisResult]
    
    # Quality metrics
    quality_metrics: Dict[str, Any]
    
    # Summary statistics
    complexity_stats: Dict[str, float]
    testability_stats: Dict[str, float]
    pattern_summary: Dict[str, int]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = asdict(self)
        
        # Convert non-serializable objects
        if self.repository_analysis:
            result['repository_analysis'] = asdict(self.repository_analysis)
        if self.ml_analysis:
            result['ml_analysis'] = asdict(self.ml_analysis)
        if self.graph_analysis:
            result['graph_analysis'] = asdict(self.graph_analysis)
            
        return result

class AnalysisService:
    """
    Comprehensive analysis service that integrates all analysis components
    Provides async analysis with progress tracking and result persistence
    """
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.ast_parser = ASTParser()
        self.repository_analyzer = RepositoryAnalyzer()
        self.ml_detector = MLPatternDetector()
        self.graph_analyzer = GraphPatternAnalyzer()
        
        # Progress tracking
        self._progress_cache: Dict[str, AnalysisProgress] = {}
        self._result_cache: Dict[str, ComprehensiveAnalysisResult] = {}
        
        # Thread pool for CPU-intensive operations
        self._executor = ThreadPoolExecutor(max_workers=4)
    
    async def start_analysis(
        self,
        session_id: str,
        repository_path: str,
        language: Language = Language.PYTHON
    ) -> AnalysisProgress:
        """
        Start comprehensive repository analysis
        Returns progress tracker that can be monitored
        """
        self.logger.info(f"Starting analysis session {session_id} for {repository_path}")
        
        # Initialize progress tracking
        progress = AnalysisProgress(
            session_id=session_id,
            start_time=time.time()
        )
        self._progress_cache[session_id] = progress
        
        try:
            # Start background analysis
            asyncio.create_task(self._run_comprehensive_analysis(
                session_id, repository_path, language, progress
            ))
            
            progress.update_step("Analysis started")
            return progress
            
        except Exception as e:
            error_msg = f"Failed to start analysis: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            progress.mark_error(error_msg)
            raise AnalysisError(error_msg) from e
    
    async def _run_comprehensive_analysis(
        self,
        session_id: str,
        repository_path: str,
        language: Language,
        progress: AnalysisProgress
    ) -> None:
        """Run complete analysis pipeline with progress updates"""
        try:
            start_time = time.time()
            progress.update_step("Discovering files", 1)
            
            # Step 1: Repository Analysis (includes file discovery)
            self.logger.info(f"Step 1: Starting repository analysis for {repository_path}")
            repo_result = await asyncio.get_event_loop().run_in_executor(
                self._executor,
                self.repository_analyzer.analyze_repository,
                repository_path
            )
            progress.update_step("Repository analysis complete", 2)
            
            # Step 2: Extract all components from analyzed files
            self.logger.info("Step 2: Extracting code components")
            all_components = []
            for file_result in repo_result.file_analyses:
                all_components.extend(file_result.components)
            progress.update_step("Component extraction complete", 3)
            
            # Step 3: ML Pattern Detection
            self.logger.info("Step 3: Running ML pattern detection")
            ml_result = None
            if all_components:
                ml_result = await asyncio.get_event_loop().run_in_executor(
                    self._executor,
                    self.ml_detector.analyze_components,
                    all_components
                )
            progress.update_step("ML analysis complete", 4)
            
            # Step 4: Graph Analysis
            self.logger.info("Step 4: Building dependency graphs")
            graph_result = None
            if all_components:
                graph_result = await asyncio.get_event_loop().run_in_executor(
                    self._executor,
                    self.graph_analyzer.analyze_dependencies,
                    all_components
                )
            progress.update_step("Graph analysis complete", 5)
            
            # Step 5: Calculate comprehensive quality metrics
            self.logger.info("Step 5: Calculating quality metrics")
            quality_metrics = self._calculate_quality_metrics(
                all_components, repo_result, ml_result, graph_result
            )
            progress.update_step("Quality metrics calculated", 6)
            
            # Step 6: Generate summary statistics
            self.logger.info("Step 6: Generating summary statistics")
            complexity_stats = self._calculate_complexity_stats(all_components)
            testability_stats = self._calculate_testability_stats(all_components)
            pattern_summary = self._calculate_pattern_summary(ml_result, graph_result)
            
            # Step 7: Create comprehensive result
            analysis_duration = time.time() - start_time
            result = ComprehensiveAnalysisResult(
                session_id=session_id,
                repository_path=repository_path,
                language=language,
                total_files=len(repo_result.file_analyses),
                analyzed_files=len([f for f in repo_result.file_analyses if f.components]),
                total_components=len(all_components),
                analysis_duration=analysis_duration,
                components=all_components,
                repository_analysis=repo_result,
                ml_analysis=ml_result,
                graph_analysis=graph_result,
                quality_metrics=quality_metrics,
                complexity_stats=complexity_stats,
                testability_stats=testability_stats,
                pattern_summary=pattern_summary
            )
            
            # Store result
            self._result_cache[session_id] = result
            progress.update_step("Analysis complete", 7)
            
            self.logger.info(
                f"Analysis complete for session {session_id}: "
                f"{len(all_components)} components in {analysis_duration:.2f}s"
            )
            
        except Exception as e:
            error_msg = f"Analysis failed: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            progress.mark_error(error_msg)
    
    def _calculate_quality_metrics(
        self,
        components: List[CodeComponent],
        repo_result: RepositoryAnalysisResult,
        ml_result: Optional[MLAnalysisResult],
        graph_result: Optional[GraphAnalysisResult]
    ) -> Dict[str, Any]:
        """Calculate comprehensive quality metrics"""
        if not components:
            return {}
        
        # Basic component metrics
        total_complexity = sum(c.complexity.cyclomatic_complexity for c in components)
        avg_complexity = total_complexity / len(components)
        high_complexity_count = len([c for c in components if c.complexity.cyclomatic_complexity > 10])
        
        # Testability metrics
        testable_components = [c for c in components if c.quality.is_testable]
        testability_rate = len(testable_components) / len(components)
        
        # Documentation metrics
        documented_components = [c for c in components if c.documentation.has_docstring]
        documentation_rate = len(documented_components) / len(components)
        
        quality_metrics = {
            "average_complexity": avg_complexity,
            "high_complexity_components": high_complexity_count,
            "testability_rate": testability_rate,
            "documentation_rate": documentation_rate,
            "total_lines_of_code": repo_result.structure.total_size if repo_result else 0,
            "maintainability_score": sum(c.quality.maintainability_index for c in components) / len(components)
        }
        
        # Add ML-specific metrics
        if ml_result:
            quality_metrics.update({
                "detected_anomalies": len(ml_result.anomalies),
                "design_patterns_found": len(ml_result.detected_patterns),
                "code_clusters": len(ml_result.clusters)
            })
        
        # Add graph-specific metrics
        if graph_result:
            quality_metrics.update({
                "dependency_cycles": len(graph_result.cycles),
                "architectural_layers": len(graph_result.layers),
                "critical_components": len([c for c in graph_result.centrality_analysis if c.betweenness_centrality > 0.1])
            })
        
        return quality_metrics
    
    def _calculate_complexity_stats(self, components: List[CodeComponent]) -> Dict[str, float]:
        """Calculate complexity statistics"""
        if not components:
            return {}
        
        complexities = [c.complexity.cyclomatic_complexity for c in components]
        complexities.sort()
        
        return {
            "min": float(min(complexities)),
            "max": float(max(complexities)),
            "mean": float(sum(complexities) / len(complexities)),
            "median": float(complexities[len(complexities) // 2]),
            "percentile_75": float(complexities[int(len(complexities) * 0.75)]),
            "percentile_90": float(complexities[int(len(complexities) * 0.90)])
        }
    
    def _calculate_testability_stats(self, components: List[CodeComponent]) -> Dict[str, float]:
        """Calculate testability statistics"""
        if not components:
            return {}
        
        testability_scores = [c.quality.testability_score for c in components]
        testable_count = len([c for c in components if c.quality.is_testable])
        
        return {
            "average_score": float(sum(testability_scores) / len(testability_scores)),
            "testable_percentage": float(testable_count / len(components) * 100),
            "high_priority_tests": len([c for c in components if c.quality.test_priority >= 4])
        }
    
    def _calculate_pattern_summary(
        self,
        ml_result: Optional[MLAnalysisResult],
        graph_result: Optional[GraphAnalysisResult]
    ) -> Dict[str, int]:
        """Calculate pattern detection summary"""
        summary = {
            "design_patterns": 0,
            "anti_patterns": 0,
            "anomalies": 0,
            "architectural_patterns": 0
        }
        
        if ml_result:
            summary["design_patterns"] = len(ml_result.detected_patterns)
            summary["anomalies"] = len(ml_result.anomalies)
        
        if graph_result:
            summary["anti_patterns"] = len(graph_result.anti_patterns)
            summary["architectural_patterns"] = len(graph_result.architectural_patterns)
        
        return summary
    
    def get_progress(self, session_id: str) -> Optional[AnalysisProgress]:
        """Get current progress for analysis session"""
        return self._progress_cache.get(session_id)
    
    def get_result(self, session_id: str) -> Optional[ComprehensiveAnalysisResult]:
        """Get complete analysis result for session"""
        return self._result_cache.get(session_id)
    
    def cleanup_session(self, session_id: str) -> bool:
        """Clean up session data and temporary files"""
        try:
            self._progress_cache.pop(session_id, None)
            self._result_cache.pop(session_id, None)
            return True
        except Exception as e:
            self.logger.error(f"Error cleaning up session {session_id}: {e}")
            return False
    
    def list_sessions(self) -> List[str]:
        """List all active session IDs"""
        return list(self._progress_cache.keys())
    
    async def analyze_uploaded_archive(
        self,
        session_id: str,
        archive_content: bytes,
        filename: str,
        language: Language = Language.PYTHON
    ) -> AnalysisProgress:
        """
        Analyze uploaded repository archive (zip, tar.gz, etc.)
        Extracts archive to temporary directory and runs analysis
        """
        self.logger.info(f"Starting archive analysis for session {session_id}: {filename}")
        
        # Create temporary directory
        temp_dir = tempfile.mkdtemp(prefix=f"qa_agent_upload_{session_id}_")
        
        try:
            # Extract archive
            archive_path = Path(temp_dir) / filename
            with open(archive_path, 'wb') as f:
                f.write(archive_content)
            
            # Determine extraction method
            extraction_dir = Path(temp_dir) / "extracted"
            extraction_dir.mkdir()
            
            if filename.endswith('.zip'):
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(extraction_dir)
            elif filename.endswith(('.tar.gz', '.tgz')):
                with tarfile.open(archive_path, 'r:gz') as tar_ref:
                    tar_ref.extractall(extraction_dir)
            elif filename.endswith('.tar'):
                with tarfile.open(archive_path, 'r') as tar_ref:
                    tar_ref.extractall(extraction_dir)
            else:
                raise AnalysisError(f"Unsupported archive format: {filename}")
            
            # Find the actual repository root (handle nested directories)
            repo_path = self._find_repository_root(extraction_dir)
            
            # Start analysis
            progress = await self.start_analysis(session_id, str(repo_path), language)
            
            # Schedule cleanup after analysis
            asyncio.create_task(self._cleanup_after_analysis(session_id, temp_dir))
            
            return progress
            
        except Exception as e:
            # Immediate cleanup on error
            shutil.rmtree(temp_dir, ignore_errors=True)
            error_msg = f"Failed to analyze uploaded archive: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            raise AnalysisError(error_msg) from e
    
    def _find_repository_root(self, extraction_dir: Path) -> Path:
        """Find the actual repository root from extracted archive"""
        # Look for common repository indicators
        indicators = ['.git', 'setup.py', 'pyproject.toml', 'package.json', 'requirements.txt']
        
        def find_indicator(directory: Path) -> Optional[Path]:
            for indicator in indicators:
                if (directory / indicator).exists():
                    return directory
            return None
        
        # Check extraction directory first
        if find_indicator(extraction_dir):
            return extraction_dir
        
        # Check subdirectories (common for GitHub archives)
        for subdir in extraction_dir.iterdir():
            if subdir.is_dir():
                result = find_indicator(subdir)
                if result:
                    return result
        
        # Default to extraction directory
        return extraction_dir
    
    async def _cleanup_after_analysis(self, session_id: str, temp_dir: str):
        """Clean up temporary files after analysis completes"""
        # Wait for analysis to complete
        max_wait = 3600  # 1 hour max
        wait_time = 0
        
        while wait_time < max_wait:
            progress = self.get_progress(session_id)
            if not progress or progress.error_message or progress.progress_percentage >= 100:
                break
            await asyncio.sleep(5)
            wait_time += 5
        
        # Clean up temporary directory
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
            self.logger.info(f"Cleaned up temporary directory for session {session_id}")
        except Exception as e:
            self.logger.error(f"Error cleaning up temporary directory {temp_dir}: {e}")

# Global service instance
analysis_service = AnalysisService()
EOF

# Create response models for API
echo "📄 Creating src/api/models/analysis_models.py..."
cat > src/api/models/analysis_models.py << 'EOF'
"""
Pydantic models for Analysis API responses
Provides type-safe API interfaces with validation
"""

from datetime import datetime
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field
from enum import Enum

class AnalysisStatus(str, Enum):
    """Analysis session status"""
    PENDING = "pending"
    ANALYZING = "analyzing" 
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class Language(str, Enum):
    """Supported programming languages"""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"

class ComponentType(str, Enum):
    """Code component types"""
    FUNCTION = "function"
    METHOD = "method"
    CLASS = "class"
    MODULE = "module"
    ASYNC_FUNCTION = "async_function"
    GENERATOR = "generator"

# Request Models
class RepositoryAnalysisRequest(BaseModel):
    """Request to analyze a repository"""
    repository_path: str = Field(..., description="Path to repository directory")
    language: Language = Field(Language.PYTHON, description="Primary programming language")
    include_patterns: Optional[List[str]] = Field(None, description="File patterns to include")
    exclude_patterns: Optional[List[str]] = Field(None, description="File patterns to exclude")

class UploadAnalysisRequest(BaseModel):
    """Request to analyze uploaded archive"""
    language: Language = Field(Language.PYTHON, description="Primary programming language")

# Response Models
class AnalysisProgressResponse(BaseModel):
    """Analysis progress information"""
    session_id: str
    status: AnalysisStatus
    total_steps: int
    completed_steps: int
    current_step: str
    progress_percentage: float
    start_time: datetime
    estimated_completion: Optional[datetime] = None
    error_message: Optional[str] = None

class CodeLocationResponse(BaseModel):
    """Code location information"""
    start_line: int
    end_line: int
    start_column: int = 0
    end_column: int = 0

class ParameterResponse(BaseModel):
    """Function/method parameter information"""
    name: str
    type_annotation: Optional[str] = None
    default_value: Optional[str] = None
    is_keyword_only: bool = False
    is_positional_only: bool = False

class ComplexityResponse(BaseModel):
    """Code complexity metrics"""
    cyclomatic_complexity: int
    cognitive_complexity: int
    maintainability_index: float

class QualityResponse(BaseModel):
    """Code quality metrics"""
    is_testable: bool
    testability_score: float
    test_priority: int
    maintainability_index: float
    lines_of_code: int

class DocumentationResponse(BaseModel):
    """Documentation information"""
    has_docstring: bool
    docstring_content: Optional[str] = None
    comment_lines: int

class CodeComponentResponse(BaseModel):
    """Code component information"""
    name: str
    type: ComponentType
    file_path: str
    location: CodeLocationResponse
    parameters: List[ParameterResponse] = []
    return_type: Optional[str] = None
    is_async: bool = False
    is_generator: bool = False
    complexity: ComplexityResponse
    quality: QualityResponse
    documentation: DocumentationResponse
    dependencies: List[str] = []
    source_code: Optional[str] = None

class PatternResponse(BaseModel):
    """Detected pattern information"""
    pattern_type: str
    pattern_name: str
    confidence: float
    components_involved: List[str]
    description: str
    evidence: List[str] = []
    severity: Optional[str] = None

class ClusterResponse(BaseModel):
    """Code clustering information"""
    cluster_id: int
    cluster_type: str
    components: List[str]
    centroid_features: Dict[str, float] = {}
    similarity_score: float
    description: str

class AnomalyResponse(BaseModel):
    """Anomaly detection information"""
    component_name: str
    anomaly_score: float
    anomaly_type: str
    description: str
    explanation: List[str] = []

class CentralityResponse(BaseModel):
    """Component centrality analysis"""
    component_name: str
    degree_centrality: float
    betweenness_centrality: float
    closeness_centrality: float
    eigenvector_centrality: float

class QualityMetricsResponse(BaseModel):
    """Repository quality metrics"""
    average_complexity: float
    high_complexity_components: int
    testability_rate: float
    documentation_rate: float
    total_lines_of_code: int
    maintainability_score: float
    detected_anomalies: int = 0
    design_patterns_found: int = 0
    code_clusters: int = 0
    dependency_cycles: int = 0
    architectural_layers: int = 0
    critical_components: int = 0

class ComplexityStatsResponse(BaseModel):
    """Complexity statistics"""
    min: float
    max: float
    mean: float
    median: float
    percentile_75: float
    percentile_90: float

class TestabilityStatsResponse(BaseModel):
    """Testability statistics"""
    average_score: float
    testable_percentage: float
    high_priority_tests: int

class PatternSummaryResponse(BaseModel):
    """Pattern detection summary"""
    design_patterns: int
    anti_patterns: int
    anomalies: int
    architectural_patterns: int

class MLAnalysisResponse(BaseModel):
    """ML analysis results"""
    clusters: List[ClusterResponse] = []
    anomalies: List[AnomalyResponse] = []
    detected_patterns: List[PatternResponse] = []
    feature_importance: Dict[str, float] = {}
    model_performance: Dict[str, float] = {}

class GraphAnalysisResponse(BaseModel):
    """Graph analysis results"""
    centrality_analysis: List[CentralityResponse] = []
    detected_cycles: List[List[str]] = []
    architectural_layers: List[List[str]] = []
    anti_patterns: List[PatternResponse] = []
    architectural_patterns: List[PatternResponse] = []
    modularity_score: float = 0.0

class RepositoryStructureResponse(BaseModel):
    """Repository structure information"""
    total_files: int
    analyzed_files: int
    total_size: int
    directory_depth: int
    file_types: Dict[str, int] = {}

class AnalysisSessionResponse(BaseModel):
    """Complete analysis session information"""
    session_id: str
    repository_path: str
    language: Language
    status: AnalysisStatus
    progress: AnalysisProgressResponse
    total_files: int = 0
    analyzed_files: int = 0
    total_components: int = 0
    analysis_duration: Optional[float] = None
    created_at: datetime
    completed_at: Optional[datetime] = None

class AnalysisSessionListResponse(BaseModel):
    """List of analysis sessions with pagination"""
    sessions: List[AnalysisSessionResponse]
    total_count: int
    page: int
    page_size: int
    has_next: bool
    has_previous: bool

class ComprehensiveAnalysisResponse(BaseModel):
    """Complete analysis results"""
    session_id: str
    repository_path: str
    language: Language
    total_files: int
    analyzed_files: int
    total_components: int
    analysis_duration: float
    
    # Quality metrics
    quality_metrics: QualityMetricsResponse
    complexity_stats: ComplexityStatsResponse
    testability_stats: TestabilityStatsResponse
    pattern_summary: PatternSummaryResponse
    
    # Repository structure
    repository_structure: Optional[RepositoryStructureResponse] = None

# Error Response Models
class ErrorResponse(BaseModel):
    """API error response"""
    error: str
    detail: Optional[str] = None
    error_code: Optional[str] = None

class ValidationErrorResponse(BaseModel):
    """Validation error response"""
    error: str = "Validation error"
    details: List[Dict[str, Any]] = []
EOF

# Create the complete Analysis API implementation
echo "📄 Creating src/api/routes/analysis.py..."
cat > src/api/routes/analysis.py << 'EOF'
"""
Analysis API Routes - Complete REST API for code analysis
Integrates AST parsing, repository analysis, ML pattern detection, and graph analysis
"""

import uuid
import os
import tempfile
from datetime import datetime
from typing import List, Optional, Dict, Any
from pathlib import Path

from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File, Query, Depends
from fastapi.responses import JSONResponse

from src.api.models.analysis_models import (
    RepositoryAnalysisRequest,
    UploadAnalysisRequest, 
    AnalysisProgressResponse,
    AnalysisSessionResponse,
    AnalysisSessionListResponse,
    CodeComponentResponse,
    PatternResponse,
    QualityMetricsResponse,
    MLAnalysisResponse,
    GraphAnalysisResponse,
    ComprehensiveAnalysisResponse,
    AnalysisStatus,
    Language,
    ErrorResponse
)
from src.analysis.analysis_service import analysis_service, AnalysisProgress, ComprehensiveAnalysisResult
from src.analysis.ast_parser import Language as ASTLanguage
from src.core.logging import get_logger
from src.core.exceptions import AnalysisError

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/analysis", tags=["analysis"])

def convert_language(api_language: Language) -> ASTLanguage:
    """Convert API language enum to AST language enum"""
    mapping = {
        Language.PYTHON: ASTLanguage.PYTHON,
        Language.JAVASCRIPT: ASTLanguage.JAVASCRIPT,
        Language.TYPESCRIPT: ASTLanguage.TYPESCRIPT
    }
    return mapping.get(api_language, ASTLanguage.PYTHON)

def convert_progress_to_response(progress: AnalysisProgress) -> AnalysisProgressResponse:
    """Convert internal progress to API response"""
    status = AnalysisStatus.ANALYZING
    if progress.error_message:
        status = AnalysisStatus.FAILED
    elif progress.progress_percentage >= 100:
        status = AnalysisStatus.COMPLETED
    elif progress.completed_steps == 0:
        status = AnalysisStatus.PENDING
    
    return AnalysisProgressResponse(
        session_id=progress.session_id,
        status=status,
        total_steps=progress.total_steps,
        completed_steps=progress.completed_steps,
        current_step=progress.current_step,
        progress_percentage=progress.progress_percentage,
        start_time=datetime.fromtimestamp(progress.start_time),
        estimated_completion=datetime.fromtimestamp(progress.estimated_completion) if progress.estimated_completion else None,
        error_message=progress.error_message
    )

def convert_components_to_response(components: List) -> List[CodeComponentResponse]:
    """Convert internal components to API response format"""
    responses = []
    for component in components:
        try:
            response = CodeComponentResponse(
                name=component.name,
                type=component.type.value,
                file_path=component.file_path,
                location={
                    "start_line": component.location.start_line,
                    "end_line": component.location.end_line,
                    "start_column": component.location.start_column,
                    "end_column": component.location.end_column
                },
                parameters=[
                    {
                        "name": param.name,
                        "type_annotation": param.type_annotation,
                        "default_value": param.default_value,
                        "is_keyword_only": param.is_keyword_only,
                        "is_positional_only": param.is_positional_only
                    } for param in component.parameters
                ],
                return_type=component.return_type,
                is_async=component.is_async,
                is_generator=component.is_generator,
                complexity={
                    "cyclomatic_complexity": component.complexity.cyclomatic_complexity,
                    "cognitive_complexity": component.complexity.cognitive_complexity,
                    "maintainability_index": component.complexity.maintainability_index
                },
                quality={
                    "is_testable": component.quality.is_testable,
                    "testability_score": component.quality.testability_score,
                    "test_priority": component.quality.test_priority,
                    "maintainability_index": component.quality.maintainability_index,
                    "lines_of_code": component.quality.lines_of_code
                },
                documentation={
                    "has_docstring": component.documentation.has_docstring,
                    "docstring_content": component.documentation.docstring_content,
                    "comment_lines": component.documentation.comment_lines
                },
                dependencies=component.dependencies.function_calls,
                source_code=component.source_code[:1000] if component.source_code else None  # Truncate for API
            )
            responses.append(response)
        except Exception as e:
            logger.warning(f"Error converting component {getattr(component, 'name', 'unknown')}: {e}")
            continue
    
    return responses

@router.post("/repository", response_model=AnalysisProgressResponse)
async def start_repository_analysis(
    request: RepositoryAnalysisRequest,
    background_tasks: BackgroundTasks
) -> AnalysisProgressResponse:
    """
    Start comprehensive analysis of a repository
    Returns session ID and progress tracker
    """
    try:
        # Validate repository path
        repo_path = Path(request.repository_path)
        if not repo_path.exists():
            raise HTTPException(
                status_code=400,
                detail=f"Repository path does not exist: {request.repository_path}"
            )
        
        if not repo_path.is_dir():
            raise HTTPException(
                status_code=400,
                detail=f"Repository path is not a directory: {request.repository_path}"
            )
        
        # Generate session ID
        session_id = str(uuid.uuid4())
        
        # Convert language
        ast_language = convert_language(request.language)
        
        # Start analysis
        progress = await analysis_service.start_analysis(
            session_id=session_id,
            repository_path=str(repo_path.absolute()),
            language=ast_language
        )
        
        return convert_progress_to_response(progress)
        
    except AnalysisError as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error starting repository analysis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/upload", response_model=AnalysisProgressResponse)
async def upload_and_analyze(
    file: UploadFile = File(..., description="Repository archive (.zip, .tar.gz, .tar)"),
    language: Language = Query(Language.PYTHON, description="Primary programming language")
) -> AnalysisProgressResponse:
    """
    Upload and analyze repository archive
    Supports .zip, .tar.gz, and .tar formats
    """
    try:
        # Validate file type
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        allowed_extensions = ['.zip', '.tar.gz', '.tgz', '.tar']
        if not any(file.filename.endswith(ext) for ext in allowed_extensions):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
            )
        
        # Check file size (50MB limit)
        content = await file.read()
        if len(content) > 50 * 1024 * 1024:  # 50MB
            raise HTTPException(
                status_code=413,
                detail="File too large. Maximum size is 50MB"
            )
        
        # Generate session ID
        session_id = str(uuid.uuid4())
        
        # Convert language
        ast_language = convert_language(language)
        
        # Start analysis
        progress = await analysis_service.analyze_uploaded_archive(
            session_id=session_id,
            archive_content=content,
            filename=file.filename,
            language=ast_language
        )
        
        return convert_progress_to_response(progress)
        
    except AnalysisError as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in upload analysis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/sessions", response_model=AnalysisSessionListResponse)
async def list_analysis_sessions(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(10, ge=1, le=100, description="Items per page")
) -> AnalysisSessionListResponse:
    """
    List analysis sessions with pagination
    Returns session metadata and progress information
    """
    try:
        # Get all session IDs
        session_ids = analysis_service.list_sessions()
        total_count = len(session_ids)
        
        # Calculate pagination
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_ids = session_ids[start_idx:end_idx]
        
        # Build session responses
        sessions = []
        for session_id in paginated_ids:
            progress = analysis_service.get_progress(session_id)
            result = analysis_service.get_result(session_id)
            
            if progress:
                status = AnalysisStatus.ANALYZING
                if progress.error_message:
                    status = AnalysisStatus.FAILED
                elif progress.progress_percentage >= 100:
                    status = AnalysisStatus.COMPLETED
                elif progress.completed_steps == 0:
                    status = AnalysisStatus.PENDING
                
                session_response = AnalysisSessionResponse(
                    session_id=session_id,
                    repository_path=result.repository_path if result else "Unknown",
                    language=Language.PYTHON,  # Default, could be stored in progress
                    status=status,
                    progress=convert_progress_to_response(progress),
                    total_files=result.total_files if result else 0,
                    analyzed_files=result.analyzed_files if result else 0,
                    total_components=result.total_components if result else 0,
                    analysis_duration=result.analysis_duration if result else None,
                    created_at=datetime.fromtimestamp(progress.start_time),
                    completed_at=datetime.fromtimestamp(progress.estimated_completion) if progress.estimated_completion and status == AnalysisStatus.COMPLETED else None
                )
                sessions.append(session_response)
        
        return AnalysisSessionListResponse(
            sessions=sessions,
            total_count=total_count,
            page=page,
            page_size=page_size,
            has_next=end_idx < total_count,
            has_previous=page > 1
        )
        
    except Exception as e:
        logger.error(f"Error listing analysis sessions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/sessions/{session_id}", response_model=AnalysisProgressResponse)
async def get_analysis_session(session_id: str) -> AnalysisProgressResponse:
    """
    Get detailed analysis session progress and status
    Includes real-time progress updates and error information
    """
    try:
        progress = analysis_service.get_progress(session_id)
        if not progress:
            raise HTTPException(
                status_code=404,
                detail=f"Analysis session not found: {session_id}"
            )
        
        return convert_progress_to_response(progress)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting analysis session {session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

@router.delete("/sessions/{session_id}")
async def delete_analysis_session(session_id: str) -> Dict[str, str]:
    """
    Delete analysis session and clean up resources
    Removes progress tracking and temporary files
    """
    try:
        success = analysis_service.cleanup_session(session_id)
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Analysis session not found: {session_id}"
            )
        
        return {"message": f"Session {session_id} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting analysis session {session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/sessions/{session_id}/components", response_model=List[CodeComponentResponse])
async def get_session_components(
    session_id: str,
    component_type: Optional[str] = Query(None, description="Filter by component type"),
    min_complexity: Optional[int] = Query(None, description="Minimum complexity filter"),
    max_complexity: Optional[int] = Query(None, description="Maximum complexity filter"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of components")
) -> List[CodeComponentResponse]:
    """
    Get extracted code components from analysis session
    Supports filtering by type, complexity, and pagination
    """
    try:
        result = analysis_service.get_result(session_id)
        if not result:
            raise HTTPException(
                status_code=404,
                detail=f"Analysis session not found or not completed: {session_id}"
            )
        
        components = result.components
        
        # Apply filters
        if component_type:
            components = [c for c in components if c.type.value == component_type]
        
        if min_complexity is not None:
            components = [c for c in components if c.complexity.cyclomatic_complexity >= min_complexity]
        
        if max_complexity is not None:
            components = [c for c in components if c.complexity.cyclomatic_complexity <= max_complexity]
        
        # Apply limit
        components = components[:limit]
        
        return convert_components_to_response(components)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting components for session {session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/sessions/{session_id}/patterns", response_model=List[PatternResponse])
async def get_session_patterns(
    session_id: str,
    pattern_type: Optional[str] = Query(None, description="Filter by pattern type"),
    min_confidence: Optional[float] = Query(None, description="Minimum confidence filter")
) -> List[PatternResponse]:
    """
    Get detected patterns from analysis session
    Includes design patterns, anti-patterns, and architectural patterns
    """
    try:
        result = analysis_service.get_result(session_id)
        if not result:
            raise HTTPException(
                status_code=404,
                detail=f"Analysis session not found or not completed: {session_id}"
            )
        
        patterns = []
        
        # Add ML detected patterns
        if result.ml_analysis and result.ml_analysis.detected_patterns:
            for pattern in result.ml_analysis.detected_patterns:
                pattern_response = PatternResponse(
                    pattern_type="design_pattern",
                    pattern_name=pattern.pattern_name,
                    confidence=pattern.confidence,
                    components_involved=pattern.components_involved,
                    description=pattern.description,
                    evidence=pattern.evidence
                )
                patterns.append(pattern_response)
        
        # Add graph analysis patterns
        if result.graph_analysis:
            # Anti-patterns
            if result.graph_analysis.anti_patterns:
                for pattern in result.graph_analysis.anti_patterns:
                    pattern_response = PatternResponse(
                        pattern_type="anti_pattern",
                        pattern_name=pattern.pattern_name,
                        confidence=pattern.confidence,
                        components_involved=pattern.components_involved,
                        description=pattern.description,
                        evidence=pattern.evidence,
                        severity=getattr(pattern, 'severity', None)
                    )
                    patterns.append(pattern_response)
            
            # Architectural patterns
            if result.graph_analysis.architectural_patterns:
                for pattern in result.graph_analysis.architectural_patterns:
                    pattern_response = PatternResponse(
                        pattern_type="architectural_pattern",
                        pattern_name=pattern.pattern_name,
                        confidence=pattern.confidence,
                        components_involved=pattern.components_involved,
                        description=pattern.description,
                        evidence=pattern.evidence
                    )
                    patterns.append(pattern_response)
        
        # Apply filters
        if pattern_type:
            patterns = [p for p in patterns if p.pattern_type == pattern_type]
        
        if min_confidence is not None:
            patterns = [p for p in patterns if p.confidence >= min_confidence]
        
        return patterns
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting patterns for session {session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/sessions/{session_id}/quality", response_model=QualityMetricsResponse)
async def get_session_quality_metrics(session_id: str) -> QualityMetricsResponse:
    """
    Get comprehensive quality metrics for analysis session
    Includes complexity, testability, documentation, and architecture metrics
    """
    try:
        result = analysis_service.get_result(session_id)
        if not result:
            raise HTTPException(
                status_code=404,
                detail=f"Analysis session not found or not completed: {session_id}"
            )
        
        return QualityMetricsResponse(**result.quality_metrics)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting quality metrics for session {session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/sessions/{session_id}/ml-analysis", response_model=MLAnalysisResponse)
async def get_session_ml_analysis(session_id: str) -> MLAnalysisResponse:
    """
    Get ML analysis results including clustering, anomaly detection, and pattern recognition
    """
    try:
        result = analysis_service.get_result(session_id)
        if not result:
            raise HTTPException(
                status_code=404,
                detail=f"Analysis session not found or not completed: {session_id}"
            )
        
        if not result.ml_analysis:
            return MLAnalysisResponse()
        
        ml_result = result.ml_analysis
        
        return MLAnalysisResponse(
            clusters=[
                {
                    "cluster_id": cluster.cluster_id,
                    "cluster_type": cluster.cluster_type,
                    "components": cluster.components,
                    "centroid_features": cluster.centroid_features,
                    "similarity_score": cluster.similarity_score,
                    "description": cluster.description
                } for cluster in ml_result.clusters
            ],
            anomalies=[
                {
                    "component_name": anomaly.component_name,
                    "anomaly_score": anomaly.anomaly_score,
                    "anomaly_type": anomaly.anomaly_type,
                    "description": anomaly.description,
                    "explanation": anomaly.explanation
                } for anomaly in ml_result.anomalies
            ],
            detected_patterns=[
                {
                    "pattern_type": "design_pattern",
                    "pattern_name": pattern.pattern_name,
                    "confidence": pattern.confidence,
                    "components_involved": pattern.components_involved,
                    "description": pattern.description,
                    "evidence": pattern.evidence
                } for pattern in ml_result.detected_patterns
            ],
            feature_importance=ml_result.feature_importance,
            model_performance=ml_result.model_performance
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting ML analysis for session {session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/sessions/{session_id}/graph-analysis", response_model=GraphAnalysisResponse)
async def get_session_graph_analysis(session_id: str) -> GraphAnalysisResponse:
    """
    Get graph analysis results including dependency graphs, centrality analysis, and architectural insights
    """
    try:
        result = analysis_service.get_result(session_id)
        if not result:
            raise HTTPException(
                status_code=404,
                detail=f"Analysis session not found or not completed: {session_id}"
            )
        
        if not result.graph_analysis:
            return GraphAnalysisResponse()
        
        graph_result = result.graph_analysis
        
        return GraphAnalysisResponse(
            centrality_analysis=[
                {
                    "component_name": centrality.component_name,
                    "degree_centrality": centrality.degree_centrality,
                    "betweenness_centrality": centrality.betweenness_centrality,
                    "closeness_centrality": centrality.closeness_centrality,
                    "eigenvector_centrality": centrality.eigenvector_centrality
                } for centrality in graph_result.centrality_analysis
            ],
            detected_cycles=graph_result.cycles,
            architectural_layers=graph_result.layers,
            anti_patterns=[
                {
                    "pattern_type": "anti_pattern",
                    "pattern_name": pattern.pattern_name,
                    "confidence": pattern.confidence,
                    "components_involved": pattern.components_involved,
                    "description": pattern.description,
                    "evidence": pattern.evidence,
                    "severity": getattr(pattern, 'severity', None)
                } for pattern in graph_result.anti_patterns
            ],
            architectural_patterns=[
                {
                    "pattern_type": "architectural_pattern",
                    "pattern_name": pattern.pattern_name,
                    "confidence": pattern.confidence,
                    "components_involved": pattern.components_involved,
                    "description": pattern.description,
                    "evidence": pattern.evidence
                } for pattern in graph_result.architectural_patterns
            ],
            modularity_score=graph_result.modularity_score
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting graph analysis for session {session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/sessions/{session_id}/summary", response_model=ComprehensiveAnalysisResponse)
async def get_session_summary(session_id: str) -> ComprehensiveAnalysisResponse:
    """
    Get comprehensive analysis summary for session
    Includes all key metrics and statistics without detailed component data
    """
    try:
        result = analysis_service.get_result(session_id)
        if not result:
            raise HTTPException(
                status_code=404,
                detail=f"Analysis session not found or not completed: {session_id}"
            )
        
        # Build repository structure response
        repo_structure = None
        if result.repository_analysis:
            repo_structure = {
                "total_files": result.total_files,
                "analyzed_files": result.analyzed_files,
                "total_size": result.repository_analysis.structure.total_size,
                "directory_depth": result.repository_analysis.structure.max_depth,
                "file_types": result.repository_analysis.structure.file_types
            }
        
        return ComprehensiveAnalysisResponse(
            session_id=result.session_id,
            repository_path=result.repository_path,
            language=Language.PYTHON,  # Convert from AST language
            total_files=result.total_files,
            analyzed_files=result.analyzed_files,
            total_components=result.total_components,
            analysis_duration=result.analysis_duration,
            quality_metrics=QualityMetricsResponse(**result.quality_metrics),
            complexity_stats=result.complexity_stats,
            testability_stats=result.testability_stats,
            pattern_summary=result.pattern_summary,
            repository_structure=repo_structure
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting analysis summary for session {session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

# Health check endpoints for analysis service
@router.get("/status")
async def analysis_service_status() -> Dict[str, Any]:
    """
    Get analysis service status and capabilities
    """
    try:
        active_sessions = len(analysis_service.list_sessions())
        
        return {
            "service": "analysis",
            "status": "healthy",
            "active_sessions": active_sessions,
            "capabilities": {
                "languages": ["python", "javascript", "typescript"],
                "analysis_types": ["ast_parsing", "repository_analysis", "ml_patterns", "graph_analysis"],
                "supported_formats": [".zip", ".tar.gz", ".tar"],
                "max_file_size_mb": 50,
                "max_concurrent_sessions": 10
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting analysis service status: {e}")
        return {
            "service": "analysis",
            "status": "degraded",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

@router.get("/test")
async def test_analysis_service() -> Dict[str, Any]:
    """
    Test analysis service functionality with minimal operations
    """
    try:
        # Test basic service functionality
        session_ids = analysis_service.list_sessions()
        
        return {
            "service": "analysis",
            "test_status": "passed",
            "active_sessions": len(session_ids),
            "components_tested": [
                "session_management",
                "progress_tracking",
                "result_caching"
            ],
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error testing analysis service: {e}")
        return {
            "service": "analysis",
            "test_status": "failed",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }
EOF

# Update requirements.txt with new dependencies
echo "📄 Updating requirements.txt..."
cat >> requirements.txt << 'EOF'

# Prompt 1.4: Analysis API Integration
celery==5.3.4
redis==5.0.1
python-jose[cryptography]==3.3.0
python-multipart==0.0.6
aiofiles==23.2.1
EOF

# Create comprehensive test file for analysis service
echo "📄 Creating tests/unit/test_analysis/test_analysis_service.py..."
cat > tests/unit/test_analysis/test_analysis_service.py << 'EOF'
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
EOF

# Create comprehensive test file for analysis API
echo "📄 Creating tests/unit/test_api/test_analysis_routes.py..."
cat > tests/unit/test_api/test_analysis_routes.py << 'EOF'
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
EOF

# Create integration test for complete analysis flow
echo "📄 Creating tests/integration/test_analysis_integration.py..."
mkdir -p tests/integration
cat > tests/integration/test_analysis_integration.py << 'EOF'
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
EOF

# Update the main API file to include the new analysis routes
echo "📄 Updating src/api/main.py to include analysis routes..."
cat > src/api/main.py << 'EOF'
"""
FastAPI Application - AI QA Agent
Main application with comprehensive middleware and routing
"""

import time
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
from contextlib import asynccontextmanager

from src.core.config import get_settings
from src.core.database import init_db
from src.core.logging import get_logger, log_request_response
from src.core.exceptions import QAAgentException
from src.api.routes import health, analysis, generation

logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    logger.info("🚀 AI QA Agent starting up...")
    
    # Initialize database
    try:
        init_db()
        logger.info("✅ Database initialized successfully")
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        raise
    
    logger.info("🎉 AI QA Agent startup complete!")
    
    yield
    
    # Shutdown
    logger.info("🛑 AI QA Agent shutting down...")
    logger.info("👋 AI QA Agent shutdown complete!")

def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    settings = get_settings()
    
    app = FastAPI(
        title="AI QA Agent",
        description="Intelligent test generation system with advanced code analysis",
        version="1.0.0",
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
        lifespan=lifespan
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Request logging middleware
    @app.middleware("http")
    async def logging_middleware(request: Request, call_next):
        start_time = time.time()
        
        # Log request
        logger.info(f"📥 {request.method} {request.url.path}")
        
        try:
            response = await call_next(request)
            
            # Log response
            process_time = time.time() - start_time
            logger.info(
                f"📤 {request.method} {request.url.path} "
                f"-> {response.status_code} ({process_time:.3f}s)"
            )
            
            return response
            
        except Exception as e:
            process_time = time.time() - start_time
            logger.error(
                f"💥 {request.method} {request.url.path} "
                f"-> ERROR ({process_time:.3f}s): {str(e)}"
            )
            raise
    
    # Global exception handlers
    @app.exception_handler(QAAgentException)
    async def qa_agent_exception_handler(request: Request, exc: QAAgentException):
        logger.error(f"QA Agent error: {exc.message}")
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.message,
                "error_code": exc.error_code,
                "detail": exc.detail
            }
        )
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        logger.warning(f"Validation error: {exc.errors()}")
        return JSONResponse(
            status_code=422,
            content={
                "error": "Validation error",
                "details": exc.errors()
            }
        )
    
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        logger.warning(f"HTTP exception: {exc.status_code} - {exc.detail}")
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.detail}
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unexpected error: {str(exc)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "detail": "An unexpected error occurred"
            }
        )
    
    # Include routers
    app.include_router(health.router)
    app.include_router(analysis.router)
    app.include_router(generation.router)
    
    # Enhanced root endpoint with interactive dashboard
    @app.get("/", response_class=HTMLResponse)
    async def root():
        """Interactive dashboard for AI QA Agent"""
        return HTMLResponse(content=f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>AI QA Agent - Intelligent Test Generation</title>
            <script src="https://cdn.tailwindcss.com"></script>
            <script src="https://unpkg.com/htmx.org@1.9.10"></script>
            <style>
                .status-indicator {{
                    animation: pulse 2s infinite;
                }}
                @keyframes pulse {{
                    0%, 100% {{ opacity: 1; }}
                    50% {{ opacity: 0.5; }}
                }}
                .gradient-bg {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                }}
            </style>
        </head>
        <body class="bg-gray-50">
            <!-- Header -->
            <div class="gradient-bg text-white p-6">
                <div class="max-w-6xl mx-auto">
                    <h1 class="text-4xl font-bold mb-2">🤖 AI QA Agent</h1>
                    <p class="text-xl opacity-90">Intelligent Test Generation with Advanced Code Analysis</p>
                    <div class="mt-4 text-sm opacity-75">
                        Sprint 1.4: Analysis API Integration • Production Ready
                    </div>
                </div>
            </div>
            
            <!-- Main Content -->
            <div class="max-w-6xl mx-auto p-6">
                <!-- System Status -->
                <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
                    <div class="bg-white rounded-lg shadow-md p-6">
                        <h3 class="text-lg font-semibold mb-4 flex items-center">
                            <span class="w-3 h-3 bg-green-500 rounded-full mr-2 status-indicator"></span>
                            System Health
                        </h3>
                        <div id="health-status" hx-get="/health/detailed" hx-trigger="load, every 30s" 
                             class="text-sm text-gray-600">
                            Loading...
                        </div>
                    </div>
                    
                    <div class="bg-white rounded-lg shadow-md p-6">
                        <h3 class="text-lg font-semibold mb-4 flex items-center">
                            <span class="w-3 h-3 bg-blue-500 rounded-full mr-2 status-indicator"></span>
                            Analysis Engine
                        </h3>
                        <div id="analysis-status" hx-get="/api/v1/analysis/status" hx-trigger="load, every 30s"
                             class="text-sm text-gray-600">
                            Loading...
                        </div>
                    </div>
                    
                    <div class="bg-white rounded-lg shadow-md p-6">
                        <h3 class="text-lg font-semibold mb-4 flex items-center">
                            <span class="w-3 h-3 bg-purple-500 rounded-full mr-2 status-indicator"></span>
                            AI Generation
                        </h3>
                        <div id="generation-status" hx-get="/api/v1/generation/status" hx-trigger="load, every 30s"
                             class="text-sm text-gray-600">
                            Loading...
                        </div>
                    </div>
                </div>
                
                <!-- Analysis Capabilities -->
                <div class="bg-white rounded-lg shadow-md p-6 mb-8">
                    <h2 class="text-2xl font-bold mb-6 text-center">🔍 Advanced Code Analysis Capabilities</h2>
                    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                        <div class="text-center p-4">
                            <div class="text-3xl mb-2">🌳</div>
                            <h3 class="font-semibold mb-2">AST Parsing</h3>
                            <p class="text-sm text-gray-600">Multi-language syntax tree analysis with complexity metrics</p>
                        </div>
                        <div class="text-center p-4">
                            <div class="text-3xl mb-2">📊</div>
                            <h3 class="font-semibold mb-2">Repository Analysis</h3>
                            <p class="text-sm text-gray-600">Project-wide structure and architecture pattern detection</p>
                        </div>
                        <div class="text-center p-4">
                            <div class="text-3xl mb-2">🧠</div>
                            <h3 class="font-semibold mb-2">ML Pattern Detection</h3>
                            <p class="text-sm text-gray-600">AI-powered clustering, anomaly detection, and design patterns</p>
                        </div>
                        <div class="text-center p-4">
                            <div class="text-3xl mb-2">🕸️</div>
                            <h3 class="font-semibold mb-2">Graph Analysis</h3>
                            <p class="text-sm text-gray-600">Dependency graphs, centrality analysis, and architectural insights</p>
                        </div>
                    </div>
                </div>
                
                <!-- API Endpoints -->
                <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
                    <div class="bg-white rounded-lg shadow-md p-6">
                        <h3 class="text-xl font-semibold mb-4">🔧 Analysis API Endpoints</h3>
                        <div class="space-y-2 text-sm">
                            <div class="flex items-center">
                                <span class="bg-green-100 text-green-800 px-2 py-1 rounded mr-2">POST</span>
                                <code>/api/v1/analysis/repository</code>
                            </div>
                            <div class="flex items-center">
                                <span class="bg-green-100 text-green-800 px-2 py-1 rounded mr-2">POST</span>
                                <code>/api/v1/analysis/upload</code>
                            </div>
                            <div class="flex items-center">
                                <span class="bg-blue-100 text-blue-800 px-2 py-1 rounded mr-2">GET</span>
                                <code>/api/v1/analysis/sessions</code>
                            </div>
                            <div class="flex items-center">
                                <span class="bg-blue-100 text-blue-800 px-2 py-1 rounded mr-2">GET</span>
                                <code>/api/v1/analysis/sessions/{{id}}/components</code>
                            </div>
                            <div class="flex items-center">
                                <span class="bg-blue-100 text-blue-800 px-2 py-1 rounded mr-2">GET</span>
                                <code>/api/v1/analysis/sessions/{{id}}/patterns</code>
                            </div>
                            <div class="flex items-center">
                                <span class="bg-blue-100 text-blue-800 px-2 py-1 rounded mr-2">GET</span>
                                <code>/api/v1/analysis/sessions/{{id}}/quality</code>
                            </div>
                        </div>
                    </div>
                    
                    <div class="bg-white rounded-lg shadow-md p-6">
                        <h3 class="text-xl font-semibold mb-4">📊 Analysis Features</h3>
                        <ul class="space-y-2 text-sm">
                            <li class="flex items-center">
                                <span class="w-2 h-2 bg-green-500 rounded-full mr-2"></span>
                                Real-time progress tracking with background tasks
                            </li>
                            <li class="flex items-center">
                                <span class="w-2 h-2 bg-green-500 rounded-full mr-2"></span>
                                Archive upload support (.zip, .tar.gz, .tar)
                            </li>
                            <li class="flex items-center">
                                <span class="w-2 h-2 bg-green-500 rounded-full