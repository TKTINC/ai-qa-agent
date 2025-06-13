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
