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
