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
