"""
Analysis API Routes - Complete implementation with background tasks
AI QA Agent - Enhanced Sprint 1.4
"""
from fastapi import APIRouter, BackgroundTasks, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
import asyncio
import uuid
import os
from datetime import datetime

from src.core.logging import get_logger
from src.core.exceptions import QAAgentError
from src.analysis.ast_parser import PythonASTParser
from src.analysis.repository_analyzer import RepositoryAnalyzer
from src.analysis.ml_pattern_detector import MLPatternDetector
from src.analysis.graph_pattern_analyzer import GraphPatternAnalyzer
from src.tasks.analysis_tasks import AnalysisTaskManager

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1/analysis", tags=["analysis"])

# Analysis task manager for background processing
task_manager = AnalysisTaskManager()

# Request/Response Models
class AnalysisRequest(BaseModel):
    """Request model for code analysis"""
    file_path: Optional[str] = None
    repository_path: Optional[str] = None
    code_content: Optional[str] = None
    analysis_type: str = Field(..., description="Type of analysis: file, repository, or content")
    options: Dict[str, Any] = Field(default_factory=dict)

class ComponentInfo(BaseModel):
    """Component information model"""
    name: str
    type: str
    start_line: int
    end_line: int
    complexity: Dict[str, float]
    quality_metrics: Dict[str, float]
    dependencies: List[str]

class AnalysisResult(BaseModel):
    """Analysis result model"""
    analysis_id: str
    analysis_type: str
    status: str
    components: List[ComponentInfo]
    quality_summary: Dict[str, Any]
    patterns_detected: List[Dict[str, Any]]
    recommendations: List[str]
    execution_time: float
    timestamp: datetime

class TaskStatus(BaseModel):
    """Background task status model"""
    task_id: str
    status: str  # pending, running, completed, failed
    progress: float  # 0.0 to 1.0
    message: str
    result: Optional[AnalysisResult] = None
    error: Optional[str] = None
    created_at: datetime
    updated_at: datetime

class ProgressUpdate(BaseModel):
    """Progress update model for WebSocket"""
    task_id: str
    progress: float
    message: str
    details: Optional[Dict[str, Any]] = None

# WebSocket connection manager for real-time progress
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.task_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, task_id: str = None):
        await websocket.accept()
        self.active_connections.append(websocket)
        if task_id:
            if task_id not in self.task_connections:
                self.task_connections[task_id] = []
            self.task_connections[task_id].append(websocket)
        logger.info(f"WebSocket connected for task: {task_id}")

    def disconnect(self, websocket: WebSocket, task_id: str = None):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        if task_id and task_id in self.task_connections:
            if websocket in self.task_connections[task_id]:
                self.task_connections[task_id].remove(websocket)
            if not self.task_connections[task_id]:
                del self.task_connections[task_id]
        logger.info(f"WebSocket disconnected for task: {task_id}")

    async def send_progress_update(self, task_id: str, progress: ProgressUpdate):
        if task_id in self.task_connections:
            disconnected = []
            for connection in self.task_connections[task_id]:
                try:
                    await connection.send_json(progress.dict())
                except Exception as e:
                    logger.warning(f"Failed to send progress update: {e}")
                    disconnected.append(connection)
            
            # Clean up disconnected connections
            for conn in disconnected:
                self.disconnect(conn, task_id)

manager = ConnectionManager()

# Analysis Endpoints

@router.post("/analyze", response_model=TaskStatus)
async def start_analysis(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks
) -> TaskStatus:
    """Start analysis task with background processing"""
    try:
        # Validate request
        if not any([request.file_path, request.repository_path, request.code_content]):
            raise HTTPException(
                status_code=400,
                detail="Must provide either file_path, repository_path, or code_content"
            )

        # Create task
        task_id = str(uuid.uuid4())
        task_status = TaskStatus(
            task_id=task_id,
            status="pending",
            progress=0.0,
            message="Analysis task created",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )

        # Store task status
        await task_manager.store_task_status(task_status)

        # Start background analysis
        background_tasks.add_task(
            perform_analysis_task,
            task_id,
            request
        )

        logger.info(f"Started analysis task: {task_id}")
        return task_status

    except Exception as e:
        logger.error(f"Failed to start analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/tasks/{task_id}", response_model=TaskStatus)
async def get_task_status(task_id: str) -> TaskStatus:
    """Get status of analysis task"""
    try:
        task_status = await task_manager.get_task_status(task_id)
        if not task_status:
            raise HTTPException(status_code=404, detail="Task not found")
        return task_status
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get task status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/tasks", response_model=List[TaskStatus])
async def get_all_tasks(
    status: Optional[str] = None,
    limit: int = 50
) -> List[TaskStatus]:
    """Get all analysis tasks with optional filtering"""
    try:
        tasks = await task_manager.get_all_tasks(status=status, limit=limit)
        return tasks
    except Exception as e:
        logger.error(f"Failed to get tasks: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/tasks/{task_id}")
async def cancel_task(task_id: str) -> Dict[str, str]:
    """Cancel running analysis task"""
    try:
        success = await task_manager.cancel_task(task_id)
        if not success:
            raise HTTPException(status_code=404, detail="Task not found or cannot be cancelled")
        
        return {"message": f"Task {task_id} cancelled successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel task: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.websocket("/tasks/{task_id}/progress")
async def websocket_progress(websocket: WebSocket, task_id: str):
    """WebSocket endpoint for real-time progress updates"""
    await manager.connect(websocket, task_id)
    try:
        # Send current status if task exists
        task_status = await task_manager.get_task_status(task_id)
        if task_status:
            progress = ProgressUpdate(
                task_id=task_id,
                progress=task_status.progress,
                message=task_status.message
            )
            await websocket.send_json(progress.dict())

        # Keep connection alive and handle any messages
        while True:
            try:
                data = await websocket.receive_text()
                # Echo back for connection health check
                await websocket.send_text(f"Echo: {data}")
            except WebSocketDisconnect:
                break
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        manager.disconnect(websocket, task_id)

# Quick Analysis Endpoints (synchronous for simple cases)

@router.post("/analyze/file", response_model=AnalysisResult)
async def analyze_file_sync(
    file_path: str,
    include_ml_analysis: bool = False
) -> AnalysisResult:
    """Synchronous file analysis for quick results"""
    try:
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")

        start_time = datetime.utcnow()
        
        # Parse the file
        parser = PythonASTParser()
        components = await parser.parse_file(file_path)
        
        # Convert components to response format
        component_infos = []
        for comp in components:
            component_infos.append(ComponentInfo(
                name=comp.name,
                type=comp.type,
                start_line=comp.start_line,
                end_line=comp.end_line,
                complexity=comp.complexity,
                quality_metrics=comp.quality_metrics,
                dependencies=comp.dependencies
            ))

        # Basic quality summary
        quality_summary = {
            "total_components": len(components),
            "average_complexity": sum(c.complexity.get("cyclomatic", 0) for c in components) / len(components) if components else 0,
            "average_testability": sum(c.quality_metrics.get("testability_score", 0) for c in components) / len(components) if components else 0,
            "high_priority_components": len([c for c in components if c.quality_metrics.get("test_priority", 0) >= 4])
        }

        patterns_detected = []
        recommendations = []

        # Optional ML analysis
        if include_ml_analysis:
            try:
                ml_detector = MLPatternDetector()
                ml_results = await ml_detector.analyze_components(components)
                patterns_detected = ml_results.get("patterns", [])
                recommendations.extend(ml_results.get("recommendations", []))
            except Exception as e:
                logger.warning(f"ML analysis failed: {e}")

        execution_time = (datetime.utcnow() - start_time).total_seconds()

        result = AnalysisResult(
            analysis_id=str(uuid.uuid4()),
            analysis_type="file",
            status="completed",
            components=component_infos,
            quality_summary=quality_summary,
            patterns_detected=patterns_detected,
            recommendations=recommendations,
            execution_time=execution_time,
            timestamp=start_time
        )

        logger.info(f"Completed file analysis: {file_path} in {execution_time:.2f}s")
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze/content", response_model=AnalysisResult)
async def analyze_content_sync(
    code_content: str,
    language: str = "python",
    include_complexity: bool = True
) -> AnalysisResult:
    """Synchronous content analysis for code snippets"""
    try:
        start_time = datetime.utcnow()

        if language.lower() != "python":
            raise HTTPException(status_code=400, detail="Only Python analysis supported currently")

        # Parse the content
        parser = PythonASTParser()
        components = await parser.parse_code_string(code_content)

        # Convert to response format
        component_infos = []
        for comp in components:
            component_infos.append(ComponentInfo(
                name=comp.name,
                type=comp.type,
                start_line=comp.start_line,
                end_line=comp.end_line,
                complexity=comp.complexity if include_complexity else {},
                quality_metrics=comp.quality_metrics,
                dependencies=comp.dependencies
            ))

        quality_summary = {
            "total_components": len(components),
            "languages_detected": [language],
            "analysis_scope": "content"
        }

        execution_time = (datetime.utcnow() - start_time).total_seconds()

        result = AnalysisResult(
            analysis_id=str(uuid.uuid4()),
            analysis_type="content",
            status="completed",
            components=component_infos,
            quality_summary=quality_summary,
            patterns_detected=[],
            recommendations=[],
            execution_time=execution_time,
            timestamp=start_time
        )

        logger.info(f"Completed content analysis in {execution_time:.2f}s")
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Content analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Background task implementation
async def perform_analysis_task(task_id: str, request: AnalysisRequest):
    """Perform analysis task in background with progress updates"""
    try:
        # Update task status to running
        await update_task_progress(task_id, 0.1, "Starting analysis...")

        if request.analysis_type == "file" and request.file_path:
            result = await analyze_file_background(task_id, request.file_path, request.options)
        elif request.analysis_type == "repository" and request.repository_path:
            result = await analyze_repository_background(task_id, request.repository_path, request.options)
        elif request.analysis_type == "content" and request.code_content:
            result = await analyze_content_background(task_id, request.code_content, request.options)
        else:
            raise ValueError("Invalid analysis request")

        # Mark task as completed
        await complete_task(task_id, result)

    except Exception as e:
        logger.error(f"Analysis task {task_id} failed: {e}")
        await fail_task(task_id, str(e))

async def analyze_file_background(task_id: str, file_path: str, options: Dict[str, Any]) -> AnalysisResult:
    """Background file analysis with full capabilities"""
    await update_task_progress(task_id, 0.2, f"Parsing file: {file_path}")
    
    # Parse file
    parser = PythonASTParser()
    components = await parser.parse_file(file_path)
    
    await update_task_progress(task_id, 0.5, "Running ML analysis...")
    
    # ML pattern detection
    ml_detector = MLPatternDetector()
    ml_results = await ml_detector.analyze_components(components)
    
    await update_task_progress(task_id, 0.8, "Generating recommendations...")
    
    # Create result
    result = AnalysisResult(
        analysis_id=str(uuid.uuid4()),
        analysis_type="file",
        status="completed",
        components=[ComponentInfo(
            name=c.name,
            type=c.type,
            start_line=c.start_line,
            end_line=c.end_line,
            complexity=c.complexity,
            quality_metrics=c.quality_metrics,
            dependencies=c.dependencies
        ) for c in components],
        quality_summary=ml_results.get("quality_summary", {}),
        patterns_detected=ml_results.get("patterns", []),
        recommendations=ml_results.get("recommendations", []),
        execution_time=0.0,  # Will be calculated
        timestamp=datetime.utcnow()
    )
    
    return result

async def analyze_repository_background(task_id: str, repo_path: str, options: Dict[str, Any]) -> AnalysisResult:
    """Background repository analysis with full capabilities"""
    await update_task_progress(task_id, 0.2, f"Analyzing repository: {repo_path}")
    
    # Repository analysis
    repo_analyzer = RepositoryAnalyzer()
    repo_results = await repo_analyzer.analyze_repository(repo_path)
    
    await update_task_progress(task_id, 0.6, "Running ML pattern detection...")
    
    # ML analysis on all components
    all_components = []
    for file_result in repo_results.files:
        all_components.extend(file_result.components)
    
    ml_detector = MLPatternDetector()
    ml_results = await ml_detector.analyze_components(all_components)
    
    await update_task_progress(task_id, 0.9, "Generating final report...")
    
    # Create comprehensive result
    result = AnalysisResult(
        analysis_id=str(uuid.uuid4()),
        analysis_type="repository",
        status="completed",
        components=[ComponentInfo(
            name=c.name,
            type=c.type,
            start_line=c.start_line,
            end_line=c.end_line,
            complexity=c.complexity,
            quality_metrics=c.quality_metrics,
            dependencies=c.dependencies
        ) for c in all_components],
        quality_summary={
            **repo_results.summary,
            **ml_results.get("quality_summary", {})
        },
        patterns_detected=ml_results.get("patterns", []),
        recommendations=ml_results.get("recommendations", []),
        execution_time=0.0,
        timestamp=datetime.utcnow()
    )
    
    return result

async def analyze_content_background(task_id: str, content: str, options: Dict[str, Any]) -> AnalysisResult:
    """Background content analysis"""
    await update_task_progress(task_id, 0.3, "Parsing code content...")
    
    parser = PythonASTParser()
    components = await parser.parse_code_string(content)
    
    await update_task_progress(task_id, 0.8, "Analyzing patterns...")
    
    result = AnalysisResult(
        analysis_id=str(uuid.uuid4()),
        analysis_type="content",
        status="completed",
        components=[ComponentInfo(
            name=c.name,
            type=c.type,
            start_line=c.start_line,
            end_line=c.end_line,
            complexity=c.complexity,
            quality_metrics=c.quality_metrics,
            dependencies=c.dependencies
        ) for c in components],
        quality_summary={
            "total_components": len(components),
            "analysis_scope": "content"
        },
        patterns_detected=[],
        recommendations=[],
        execution_time=0.0,
        timestamp=datetime.utcnow()
    )
    
    return result

# Helper functions for task management
async def update_task_progress(task_id: str, progress: float, message: str):
    """Update task progress and notify WebSocket connections"""
    task_status = await task_manager.get_task_status(task_id)
    if task_status:
        task_status.progress = progress
        task_status.message = message
        task_status.status = "running"
        task_status.updated_at = datetime.utcnow()
        
        await task_manager.store_task_status(task_status)
        
        # Send WebSocket update
        progress_update = ProgressUpdate(
            task_id=task_id,
            progress=progress,
            message=message
        )
        await manager.send_progress_update(task_id, progress_update)

async def complete_task(task_id: str, result: AnalysisResult):
    """Mark task as completed with result"""
    task_status = await task_manager.get_task_status(task_id)
    if task_status:
        task_status.status = "completed"
        task_status.progress = 1.0
        task_status.message = "Analysis completed successfully"
        task_status.result = result
        task_status.updated_at = datetime.utcnow()
        
        await task_manager.store_task_status(task_status)
        
        progress_update = ProgressUpdate(
            task_id=task_id,
            progress=1.0,
            message="Analysis completed"
        )
        await manager.send_progress_update(task_id, progress_update)

async def fail_task(task_id: str, error: str):
    """Mark task as failed with error"""
    task_status = await task_manager.get_task_status(task_id)
    if task_status:
        task_status.status = "failed"
        task_status.message = "Analysis failed"
        task_status.error = error
        task_status.updated_at = datetime.utcnow()
        
        await task_manager.store_task_status(task_status)
        
        progress_update = ProgressUpdate(
            task_id=task_id,
            progress=task_status.progress,
            message="Analysis failed"
        )
        await manager.send_progress_update(task_id, progress_update)
