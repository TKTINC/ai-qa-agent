"""
Repository Analysis System for multi-file project analysis.
Builds on AST parser to provide repository-wide insights and metrics.
"""

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Union, Tuple
import chardet
import pathspec
from collections import defaultdict, Counter

# Git integration
try:
    import git
    GIT_AVAILABLE = True
except ImportError:
    GIT_AVAILABLE = False
    logging.warning("GitPython not available - Git analysis will be disabled")

from .ast_parser import parse_code_file, CodeComponent, analyze_complexity_metrics

logger = logging.getLogger(__name__)


@dataclass
class FileAnalysisResult:
    """Results from analyzing a single file."""
    file_path: str
    language: str
    components: List[CodeComponent]
    file_size: int
    lines_of_code: int
    complexity_metrics: Dict[str, Any]
    analysis_time: float
    error: Optional[str] = None
    
    @property
    def component_count(self) -> int:
        """Get total number of components in file."""
        return len(self.components)
    
    @property
    def has_error(self) -> bool:
        """Check if analysis had errors."""
        return self.error is not None


@dataclass
class RepositoryStructure:
    """Information about repository structure and organization."""
    total_files: int
    total_directories: int
    max_depth: int
    language_distribution: Dict[str, int]
    file_size_distribution: Dict[str, int]
    directory_structure: Dict[str, Any]
    naming_patterns: Dict[str, List[str]]
    
    def calculate_organization_score(self) -> float:
        """Calculate how well-organized the repository is (0-1.0)."""
        # Factors: reasonable depth, balanced distribution, consistent naming
        depth_score = max(0, 1.0 - (self.max_depth - 3) / 10.0)  # Penalize very deep nesting
        
        # Language concentration (better to have fewer languages well-organized)
        lang_count = len(self.language_distribution)
        lang_score = max(0.5, 1.0 - (lang_count - 1) / 5.0)
        
        return (depth_score + lang_score) / 2.0


@dataclass
class ProjectMetrics:
    """Aggregate metrics for the entire project."""
    total_components: int
    total_lines_of_code: int
    average_complexity: float
    max_complexity: int
    high_complexity_files: List[str]
    component_type_distribution: Dict[str, int]
    test_priority_distribution: Dict[int, int]
    technical_debt_estimate: float  # In hours
    maintainability_score: float  # 0-1.0
    hotspot_files: List[Tuple[str, float]]  # (file_path, hotspot_score)
    
    def calculate_quality_score(self) -> float:
        """Calculate overall project quality score (0-1.0)."""
        # Combine multiple factors
        complexity_score = max(0, 1.0 - (self.average_complexity - 5) / 15.0)
        maintainability_score = self.maintainability_score
        debt_score = max(0, 1.0 - self.technical_debt_estimate / 100.0)
        
        return (complexity_score + maintainability_score + debt_score) / 3.0


@dataclass
class GitAnalysisResult:
    """Results from Git repository analysis."""
    is_git_repo: bool
    branch_name: Optional[str] = None
    total_commits: int = 0
    recent_commits: int = 0  # Last 30 days
    contributors: List[str] = field(default_factory=list)
    file_change_frequency: Dict[str, int] = field(default_factory=dict)
    hot_files: List[Tuple[str, int]] = field(default_factory=list)  # (file_path, change_count)
    repository_age_days: int = 0
    
    def calculate_activity_score(self) -> float:
        """Calculate repository activity score (0-1.0)."""
        if not self.is_git_repo:
            return 0.5  # Neutral for non-git repos
            
        # Recent activity factor
        if self.repository_age_days > 0:
            activity_rate = self.recent_commits / max(1, self.repository_age_days / 30)
            activity_score = min(1.0, activity_rate / 10.0)  # Normalize to 10 commits/month
        else:
            activity_score = 0.0
            
        return activity_score


@dataclass
class RepositoryAnalysisResult:
    """Complete analysis results for a repository."""
    repository_path: str
    analysis_time: float
    file_results: List[FileAnalysisResult]
    repository_structure: RepositoryStructure
    project_metrics: ProjectMetrics
    git_analysis: GitAnalysisResult
    architecture_patterns: List[str]
    cross_file_dependencies: Dict[str, Set[str]]
    recommendations: List[str]
    
    @property
    def success_rate(self) -> float:
        """Calculate percentage of files successfully analyzed."""
        if not self.file_results:
            return 0.0
        successful = len([r for r in self.file_results if not r.has_error])
        return successful / len(self.file_results)
    
    @property
    def total_components(self) -> int:
        """Get total components across all files."""
        return sum(r.component_count for r in self.file_results if not r.has_error)


class FileDiscoveryEngine:
    """Discovers and filters files for analysis."""
    
    def __init__(self, max_file_size: int = 50 * 1024 * 1024):  # 50MB default
        self.max_file_size = max_file_size
        self.supported_extensions = {
            '.py', '.pyw',  # Python
            '.js', '.mjs', '.jsx',  # JavaScript
            '.ts', '.tsx',  # TypeScript
        }
        
    def discover_files(self, directory: Union[str, Path], 
                      respect_gitignore: bool = True,
                      custom_ignore_patterns: Optional[List[str]] = None) -> List[Path]:
        """
        Discover analyzable files in directory.
        
        Args:
            directory: Directory to analyze
            respect_gitignore: Whether to respect .gitignore patterns
            custom_ignore_patterns: Additional patterns to ignore
            
        Returns:
            List of file paths to analyze
        """
        directory = Path(directory)
        if not directory.exists():
            raise ValueError(f"Directory does not exist: {directory}")
            
        logger.info(f"Discovering files in {directory}")
        
        # Load ignore patterns
        ignore_spec = self._load_ignore_patterns(directory, respect_gitignore, custom_ignore_patterns)
        
        # Discover files
        files = []
        for root, dirs, filenames in os.walk(directory):
            root_path = Path(root)
            
            # Filter directories
            dirs[:] = [d for d in dirs if not self._should_ignore_directory(root_path / d, ignore_spec)]
            
            # Process files
            for filename in filenames:
                file_path = root_path / filename
                
                if self._should_analyze_file(file_path, ignore_spec):
                    files.append(file_path)
        
        # Sort by size (smaller files first for faster initial results)
        files.sort(key=lambda f: f.stat().st_size if f.exists() else 0)
        
        logger.info(f"Discovered {len(files)} files for analysis")
        return files
    
    def _load_ignore_patterns(self, directory: Path, respect_gitignore: bool, 
                             custom_patterns: Optional[List[str]]) -> Optional[pathspec.PathSpec]:
        """Load ignore patterns from .gitignore and custom patterns."""
        patterns = []
        
        # Default ignore patterns
        default_patterns = [
            '__pycache__/',
            '*.pyc',
            '*.pyo',
            '*.pyd',
            '.git/',
            '.svn/',
            '.hg/',
            '.tox/',
            '.pytest_cache/',
            'node_modules/',
            '.venv/',
            'venv/',
            'env/',
            '*.egg-info/',
            'dist/',
            'build/',
            '.DS_Store',
            'Thumbs.db',
            '*.log',
            '*.tmp',
            '*.temp',
        ]
        patterns.extend(default_patterns)
        
        # Load .gitignore if requested
        if respect_gitignore:
            gitignore_path = directory / '.gitignore'
            if gitignore_path.exists():
                try:
                    with open(gitignore_path, 'r', encoding='utf-8') as f:
                        gitignore_patterns = f.read().splitlines()
                    patterns.extend(gitignore_patterns)
                    logger.debug(f"Loaded {len(gitignore_patterns)} patterns from .gitignore")
                except Exception as e:
                    logger.warning(f"Failed to read .gitignore: {e}")
        
        # Add custom patterns
        if custom_patterns:
            patterns.extend(custom_patterns)
        
        # Create pathspec
        if patterns:
            return pathspec.PathSpec.from_lines('gitwildmatch', patterns)
        return None
    
    def _should_ignore_directory(self, dir_path: Path, ignore_spec: Optional[pathspec.PathSpec]) -> bool:
        """Check if directory should be ignored."""
        if ignore_spec:
            # Convert to relative path for pattern matching
            try:
                return ignore_spec.match_file(str(dir_path.name) + '/')
            except:
                return False
        return False
    
    def _should_analyze_file(self, file_path: Path, ignore_spec: Optional[pathspec.PathSpec]) -> bool:
        """Check if file should be analyzed."""
        # Check extension
        if file_path.suffix.lower() not in self.supported_extensions:
            return False
        
        # Check if ignored
        if ignore_spec and ignore_spec.match_file(str(file_path)):
            return False
        
        # Check file size
        try:
            if file_path.stat().st_size > self.max_file_size:
                logger.warning(f"Skipping large file: {file_path} ({file_path.stat().st_size} bytes)")
                return False
        except:
            return False
            
        # Check encoding (must be text)
        if not self._is_text_file(file_path):
            return False
            
        return True
    
    def _is_text_file(self, file_path: Path) -> bool:
        """Check if file is a text file using encoding detection."""
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read(1024)  # Read first 1KB
                
            if not raw_data:
                return True  # Empty file is text
                
            # Use chardet to detect encoding
            result = chardet.detect(raw_data)
            confidence = result.get('confidence', 0)
            
            # Consider it text if confidence is reasonable
            return confidence > 0.7
            
        except Exception:
            return False


class GitAnalyzer:
    """Analyzes Git repository metadata and history."""
    
    def __init__(self):
        self.git_available = GIT_AVAILABLE
    
    def analyze_repository(self, repo_path: Union[str, Path]) -> GitAnalysisResult:
        """Analyze Git repository."""
        repo_path = Path(repo_path)
        
        if not self.git_available:
            logger.warning("Git analysis unavailable - GitPython not installed")
            return GitAnalysisResult(is_git_repo=False)
        
        try:
            repo = git.Repo(repo_path, search_parent_directories=True)
            return self._analyze_git_repo(repo)
        except git.InvalidGitRepositoryError:
            logger.info(f"Not a Git repository: {repo_path}")
            return GitAnalysisResult(is_git_repo=False)
        except Exception as e:
            logger.error(f"Error analyzing Git repository: {e}")
            return GitAnalysisResult(is_git_repo=False)
    
    def _analyze_git_repo(self, repo: 'git.Repo') -> GitAnalysisResult:
        """Analyze Git repository details."""
        try:
            # Basic repository info
            branch_name = repo.active_branch.name if repo.active_branch else None
            
            # Count commits
            commits = list(repo.iter_commits())
            total_commits = len(commits)
            
            # Recent commits (last 30 days)
            current_time = time.time()
            thirty_days_ago = current_time - (30 * 24 * 60 * 60)
            recent_commits = len([
                c for c in commits[:100]  # Limit to avoid performance issues
                if c.committed_date > thirty_days_ago
            ])
            
            # Contributors
            contributors = list(set(c.author.name for c in commits[:100]))
            
            # File change frequency
            file_changes = defaultdict(int)
            for commit in commits[:100]:  # Limit for performance
                try:
                    for item in commit.stats.files:
                        file_changes[item] += 1
                except:
                    continue
            
            # Hot files (most frequently changed)
            hot_files = sorted(file_changes.items(), key=lambda x: x[1], reverse=True)[:10]
            
            # Repository age
            if commits:
                first_commit_date = min(c.committed_date for c in commits[-10:])  # Last 10 for performance
                repo_age_days = int((current_time - first_commit_date) / (24 * 60 * 60))
            else:
                repo_age_days = 0
            
            return GitAnalysisResult(
                is_git_repo=True,
                branch_name=branch_name,
                total_commits=total_commits,
                recent_commits=recent_commits,
                contributors=contributors,
                file_change_frequency=dict(file_changes),
                hot_files=hot_files,
                repository_age_days=repo_age_days
            )
            
        except Exception as e:
            logger.error(f"Error in Git analysis: {e}")
            return GitAnalysisResult(is_git_repo=True)  # Mark as git repo but with limited info


class ProjectStructureAnalyzer:
    """Analyzes project structure and organization."""
    
    def analyze_structure(self, files: List[Path], base_path: Path) -> RepositoryStructure:
        """Analyze project structure from file list."""
        directories = set()
        max_depth = 0
        language_dist = Counter()
        size_dist = Counter()
        
        # Analyze each file
        for file_path in files:
            # Track directories
            directories.update(file_path.relative_to(base_path).parents)
            
            # Calculate depth
            depth = len(file_path.relative_to(base_path).parts) - 1
            max_depth = max(max_depth, depth)
            
            # Language distribution
            language = self._detect_language(file_path)
            language_dist[language] += 1
            
            # Size distribution
            try:
                size = file_path.stat().st_size
                size_category = self._categorize_size(size)
                size_dist[size_category] += 1
            except:
                size_dist['unknown'] += 1
        
        # Build directory structure
        dir_structure = self._build_directory_tree(files, base_path)
        
        # Analyze naming patterns
        naming_patterns = self._analyze_naming_patterns(files)
        
        return RepositoryStructure(
            total_files=len(files),
            total_directories=len(directories),
            max_depth=max_depth,
            language_distribution=dict(language_dist),
            file_size_distribution=dict(size_dist),
            directory_structure=dir_structure,
            naming_patterns=naming_patterns
        )
    
    def _detect_language(self, file_path: Path) -> str:
        """Detect language from file extension."""
        ext = file_path.suffix.lower()
        if ext in ['.py', '.pyw']:
            return 'python'
        elif ext in ['.js', '.mjs', '.jsx']:
            return 'javascript'
        elif ext in ['.ts', '.tsx']:
            return 'typescript'
        else:
            return 'unknown'
    
    def _categorize_size(self, size: int) -> str:
        """Categorize file size."""
        if size < 1024:  # < 1KB
            return 'tiny'
        elif size < 10 * 1024:  # < 10KB
            return 'small'
        elif size < 100 * 1024:  # < 100KB
            return 'medium'
        elif size < 1024 * 1024:  # < 1MB
            return 'large'
        else:
            return 'huge'
    
    def _build_directory_tree(self, files: List[Path], base_path: Path) -> Dict[str, Any]:
        """Build a tree structure of directories."""
        tree = {}
        
        for file_path in files:
            relative_path = file_path.relative_to(base_path)
            parts = relative_path.parts
            
            current = tree
            for part in parts[:-1]:  # All but the file name
                if part not in current:
                    current[part] = {}
                current = current[part]
            
            # Add file to current directory
            if '__files__' not in current:
                current['__files__'] = []
            current['__files__'].append(parts[-1])
        
        return tree
    
    def _analyze_naming_patterns(self, files: List[Path]) -> Dict[str, List[str]]:
        """Analyze naming patterns in the codebase."""
        patterns = {
            'snake_case': [],
            'camelCase': [],
            'PascalCase': [],
            'kebab-case': [],
            'other': []
        }
        
        for file_path in files:
            stem = file_path.stem
            
            if '_' in stem and stem.islower():
                patterns['snake_case'].append(stem)
            elif '-' in stem:
                patterns['kebab-case'].append(stem)
            elif stem[0].isupper() and any(c.isupper() for c in stem[1:]):
                patterns['PascalCase'].append(stem)
            elif stem[0].islower() and any(c.isupper() for c in stem):
                patterns['camelCase'].append(stem)
            else:
                patterns['other'].append(stem)
        
        return patterns


class ArchitecturePatternDetector:
    """Detects common architectural patterns in the codebase."""
    
    def detect_patterns(self, files: List[Path], base_path: Path, 
                       file_results: List[FileAnalysisResult]) -> List[str]:
        """Detect architectural patterns."""
        patterns = []
        
        # Analyze directory structure for patterns
        dir_names = set()
        for file_path in files:
            relative_path = file_path.relative_to(base_path)
            dir_names.update(part.lower() for part in relative_path.parts[:-1])
        
        # MVC Pattern
        if {'models', 'views', 'controllers'}.issubset(dir_names) or \
           {'model', 'view', 'controller'}.issubset(dir_names):
            patterns.append('MVC (Model-View-Controller)')
        
        # Layered Architecture
        if {'services', 'repositories', 'controllers'}.issubset(dir_names) or \
           {'service', 'repository', 'controller'}.issubset(dir_names):
            patterns.append('Layered Architecture')
        
        # Clean Architecture
        if {'entities', 'usecases', 'interfaces', 'frameworks'}.intersection(dir_names) or \
           {'domain', 'application', 'infrastructure'}.issubset(dir_names):
            patterns.append('Clean Architecture')
        
        # Microservices indicators
        if {'services', 'api', 'gateway'}.issubset(dir_names) or \
           len([d for d in dir_names if 'service' in d]) > 2:
            patterns.append('Microservices Architecture')
        
        # Component-based
        if {'components', 'modules'}.intersection(dir_names):
            patterns.append('Component-Based Architecture')
        
        # Plugin/Extension pattern
        if {'plugins', 'extensions', 'addons'}.intersection(dir_names):
            patterns.append('Plugin Architecture')
        
        # Analyze code patterns
        code_patterns = self._analyze_code_patterns(file_results)
        patterns.extend(code_patterns)
        
        return patterns
    
    def _analyze_code_patterns(self, file_results: List[FileAnalysisResult]) -> List[str]:
        """Analyze code for design patterns."""
        patterns = []
        
        # Count pattern indicators
        pattern_indicators = defaultdict(int)
        
        for result in file_results:
            if result.has_error:
                continue
                
            for component in result.components:
                # Singleton pattern
                if 'singleton' in component.name.lower() or \
                   'instance' in component.name.lower():
                    pattern_indicators['Singleton Pattern'] += 1
                
                # Factory pattern
                if 'factory' in component.name.lower() or \
                   'create' in component.name.lower():
                    pattern_indicators['Factory Pattern'] += 1
                
                # Observer pattern
                if 'observer' in component.name.lower() or \
                   'notify' in component.name.lower() or \
                   'subscribe' in component.name.lower():
                    pattern_indicators['Observer Pattern'] += 1
                
                # Strategy pattern
                if 'strategy' in component.name.lower() or \
                   'algorithm' in component.name.lower():
                    pattern_indicators['Strategy Pattern'] += 1
                
                # Builder pattern
                if 'builder' in component.name.lower() or \
                   'build' in component.name.lower():
                    pattern_indicators['Builder Pattern'] += 1
        
        # Add patterns with sufficient evidence
        for pattern, count in pattern_indicators.items():
            if count >= 2:  # Require at least 2 instances
                patterns.append(pattern)
        
        return patterns


class RepositoryAnalyzer:
    """Main repository analysis engine."""
    
    def __init__(self, max_workers: int = 4, max_file_size: int = 50 * 1024 * 1024):
        self.max_workers = max_workers
        self.file_discovery = FileDiscoveryEngine(max_file_size)
        self.git_analyzer = GitAnalyzer()
        self.structure_analyzer = ProjectStructureAnalyzer()
        self.pattern_detector = ArchitecturePatternDetector()
    
    def analyze_repository(self, repo_path: Union[str, Path],
                          respect_gitignore: bool = True,
                          custom_ignore_patterns: Optional[List[str]] = None) -> RepositoryAnalysisResult:
        """
        Analyze entire repository.
        
        Args:
            repo_path: Path to repository root
            respect_gitignore: Whether to respect .gitignore patterns
            custom_ignore_patterns: Additional patterns to ignore
            
        Returns:
            Complete repository analysis results
        """
        repo_path = Path(repo_path)
        start_time = time.time()
        
        logger.info(f"Starting repository analysis: {repo_path}")
        
        # Discover files
        files = self.file_discovery.discover_files(
            repo_path, respect_gitignore, custom_ignore_patterns
        )
        
        if not files:
            logger.warning("No analyzable files found")
            return self._create_empty_result(repo_path, time.time() - start_time)
        
        # Analyze files in parallel
        file_results = self._analyze_files_parallel(files)
        
        # Analyze repository structure
        repo_structure = self.structure_analyzer.analyze_structure(files, repo_path)
        
        # Git analysis
        git_analysis = self.git_analyzer.analyze_repository(repo_path)
        
        # Detect architecture patterns
        architecture_patterns = self.pattern_detector.detect_patterns(
            files, repo_path, file_results
        )
        
        # Calculate project metrics
        project_metrics = self._calculate_project_metrics(file_results)
        
        # Analyze cross-file dependencies
        cross_deps = self._analyze_cross_file_dependencies(file_results)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            file_results, project_metrics, repo_structure, git_analysis
        )
        
        analysis_time = time.time() - start_time
        
        result = RepositoryAnalysisResult(
            repository_path=str(repo_path),
            analysis_time=analysis_time,
            file_results=file_results,
            repository_structure=repo_structure,
            project_metrics=project_metrics,
            git_analysis=git_analysis,
            architecture_patterns=architecture_patterns,
            cross_file_dependencies=cross_deps,
            recommendations=recommendations
        )
        
        logger.info(f"Repository analysis complete: {len(file_results)} files, "
                   f"{result.total_components} components in {analysis_time:.2f}s")
        
        return result
    
    def _analyze_files_parallel(self, files: List[Path]) -> List[FileAnalysisResult]:
        """Analyze multiple files in parallel."""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all file analysis tasks
            future_to_file = {
                executor.submit(self._analyze_single_file, file_path): file_path
                for file_path in files
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error analyzing {file_path}: {e}")
                    # Create error result
                    results.append(FileAnalysisResult(
                        file_path=str(file_path),
                        language="unknown",
                        components=[],
                        file_size=0,
                        lines_of_code=0,
                        complexity_metrics={},
                        analysis_time=0.0,
                        error=str(e)
                    ))
        
        return results
    
    def _analyze_single_file(self, file_path: Path) -> FileAnalysisResult:
        """Analyze a single file."""
        start_time = time.time()
        
        try:
            # Parse file with AST parser
            components = parse_code_file(file_path)
            
            # Calculate file metrics
            file_size = file_path.stat().st_size
            
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                lines_of_code = len([line for line in lines if line.strip()])
            
            # Calculate complexity metrics for this file
            complexity_metrics = analyze_complexity_metrics(components) if components else {}
            
            # Detect language
            language = self._detect_file_language(file_path)
            
            analysis_time = time.time() - start_time
            
            return FileAnalysisResult(
                file_path=str(file_path),
                language=language,
                components=components,
                file_size=file_size,
                lines_of_code=lines_of_code,
                complexity_metrics=complexity_metrics,
                analysis_time=analysis_time
            )
            
        except Exception as e:
            analysis_time = time.time() - start_time
            return FileAnalysisResult(
                file_path=str(file_path),
                language="unknown",
                components=[],
                file_size=0,
                lines_of_code=0,
                complexity_metrics={},
                analysis_time=analysis_time,
                error=str(e)
            )
    
    def _detect_file_language(self, file_path: Path) -> str:
        """Detect programming language from file."""
        ext = file_path.suffix.lower()
        if ext in ['.py', '.pyw']:
            return 'python'
        elif ext in ['.js', '.mjs', '.jsx']:
            return 'javascript'
        elif ext in ['.ts', '.tsx']:
            return 'typescript'
        else:
            return 'unknown'
    
    def _calculate_project_metrics(self, file_results: List[FileAnalysisResult]) -> ProjectMetrics:
        """Calculate aggregate project metrics."""
        all_components = []
        total_loc = 0
        high_complexity_files = []
        
        for result in file_results:
            if result.has_error:
                continue
                
            all_components.extend(result.components)
            total_loc += result.lines_of_code
            
            # Check for high complexity files
            if result.complexity_metrics.get('max_complexity', 0) >= 10:
                high_complexity_files.append(result.file_path)
        
        if not all_components:
            # Return empty metrics
            return ProjectMetrics(
                total_components=0,
                total_lines_of_code=total_loc,
                average_complexity=0.0,
                max_complexity=0,
                high_complexity_files=[],
                component_type_distribution={},
                test_priority_distribution={},
                technical_debt_estimate=0.0,
                maintainability_score=1.0,
                hotspot_files=[]
            )
        
        # Calculate metrics
        complexities = [c.cyclomatic_complexity for c in all_components]
        avg_complexity = sum(complexities) / len(complexities)
        max_complexity = max(complexities)
        
        # Component type distribution
        type_dist = Counter(c.component_type.value for c in all_components)
        
        # Test priority distribution
        priority_dist = Counter(c.test_priority for c in all_components)
        
        # Technical debt estimate (rough heuristic)
        high_complexity_components = [c for c in all_components if c.cyclomatic_complexity >= 10]
        tech_debt = len(high_complexity_components) * 2.0  # 2 hours per high complexity component
        
        # Maintainability score
        maintainability = max(0.0, 1.0 - (avg_complexity - 5) / 20.0)
        
        # Hotspot files (files with high complexity and many components)
        hotspots = []
        for result in file_results:
            if result.has_error:
                continue
            
            if result.components:
                avg_file_complexity = sum(c.cyclomatic_complexity for c in result.components) / len(result.components)
                component_density = len(result.components) / max(1, result.lines_of_code / 100)
                hotspot_score = avg_file_complexity * component_density
                
                if hotspot_score > 5.0:  # Threshold for hotspot
                    hotspots.append((result.file_path, hotspot_score))
        
        hotspots.sort(key=lambda x: x[1], reverse=True)
        
        return ProjectMetrics(
            total_components=len(all_components),
            total_lines_of_code=total_loc,
            average_complexity=avg_complexity,
            max_complexity=max_complexity,
            high_complexity_files=high_complexity_files,
            component_type_distribution=dict(type_dist),
            test_priority_distribution=dict(priority_dist),
            technical_debt_estimate=tech_debt,
            maintainability_score=maintainability,
            hotspot_files=hotspots[:10]  # Top 10 hotspots
        )
    
    def _analyze_cross_file_dependencies(self, file_results: List[FileAnalysisResult]) -> Dict[str, Set[str]]:
        """Analyze dependencies between files."""
        dependencies = defaultdict(set)
        
        # Simple implementation - look for import patterns
        for result in file_results:
            if result.has_error:
                continue
                
            file_path = result.file_path
            
            for component in result.components:
                for dependency in component.dependencies:
                    # Simple heuristic: if dependency looks like a local module
                    if '.' in dependency and not dependency.startswith('__'):
                        dependencies[file_path].add(dependency)
        
        return dict(dependencies)
    
    def _generate_recommendations(self, file_results: List[FileAnalysisResult],
                                 project_metrics: ProjectMetrics,
                                 repo_structure: RepositoryStructure,
                                 git_analysis: GitAnalysisResult) -> List[str]:
        """Generate improvement recommendations."""
        recommendations = []
        
        # High complexity recommendations
        if project_metrics.average_complexity > 8:
            recommendations.append(
                f"High average complexity ({project_metrics.average_complexity:.1f}). "
                "Consider refactoring complex functions."
            )
        
        if project_metrics.high_complexity_files:
            recommendations.append(
                f"{len(project_metrics.high_complexity_files)} files have high complexity. "
                "Prioritize these for refactoring."
            )
        
        # Technical debt
        if project_metrics.technical_debt_estimate > 20:
            recommendations.append(
                f"Estimated technical debt: {project_metrics.technical_debt_estimate:.1f} hours. "
                "Consider addressing high-priority components."
            )
        
        # Test coverage recommendations
        high_priority_count = project_metrics.test_priority_distribution.get(5, 0) + \
                             project_metrics.test_priority_distribution.get(4, 0)
        if high_priority_count > 0:
            recommendations.append(
                f"{high_priority_count} components need high-priority testing. "
                "Focus test generation on these components."
            )
        
        # Structure recommendations
        if repo_structure.max_depth > 8:
            recommendations.append(
                f"Deep directory nesting ({repo_structure.max_depth} levels). "
                "Consider flattening the structure."
            )
        
        # Git activity
        if git_analysis.is_git_repo and git_analysis.recent_commits == 0:
            recommendations.append(
                "No recent commits detected. Ensure the codebase is actively maintained."
            )
        
        # Language diversity
        if len(repo_structure.language_distribution) > 3:
            recommendations.append(
                "Multiple programming languages detected. "
                "Ensure consistent coding standards across languages."
            )
        
        return recommendations
    
    def _create_empty_result(self, repo_path: Path, analysis_time: float) -> RepositoryAnalysisResult:
        """Create empty result for repositories with no analyzable files."""
        return RepositoryAnalysisResult(
            repository_path=str(repo_path),
            analysis_time=analysis_time,
            file_results=[],
            repository_structure=RepositoryStructure(0, 0, 0, {}, {}, {}, {}),
            project_metrics=ProjectMetrics(0, 0, 0.0, 0, [], {}, {}, 0.0, 1.0, []),
            git_analysis=GitAnalysisResult(False),
            architecture_patterns=[],
            cross_file_dependencies={},
            recommendations=["No analyzable files found in repository."]
        )


# Main interface functions
def analyze_repository(repo_path: Union[str, Path], 
                      max_workers: int = 4,
                      respect_gitignore: bool = True,
                      custom_ignore_patterns: Optional[List[str]] = None) -> RepositoryAnalysisResult:
    """
    Analyze a complete repository.
    
    Args:
        repo_path: Path to repository root
        max_workers: Number of parallel workers for file analysis
        respect_gitignore: Whether to respect .gitignore patterns
        custom_ignore_patterns: Additional patterns to ignore
        
    Returns:
        Complete repository analysis results
    """
    analyzer = RepositoryAnalyzer(max_workers=max_workers)
    return analyzer.analyze_repository(repo_path, respect_gitignore, custom_ignore_patterns)


def discover_repository_files(repo_path: Union[str, Path],
                             respect_gitignore: bool = True,
                             custom_ignore_patterns: Optional[List[str]] = None) -> List[Path]:
    """
    Discover analyzable files in a repository.
    
    Args:
        repo_path: Path to repository root
        respect_gitignore: Whether to respect .gitignore patterns
        custom_ignore_patterns: Additional patterns to ignore
        
    Returns:
        List of discoverable file paths
    """
    discovery = FileDiscoveryEngine()
    return discovery.discover_files(repo_path, respect_gitignore, custom_ignore_patterns)
