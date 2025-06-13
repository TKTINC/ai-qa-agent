"""
Advanced AST parsing system for multi-language code analysis.
Supports Python (full implementation) and JavaScript/TypeScript (placeholder structure).
"""

import ast
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Union
import astroid
import radon.complexity as cc
import radon.metrics as rm
from radon.raw import analyze

logger = logging.getLogger(__name__)


class Language(Enum):
    """Supported programming languages."""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"


class ComponentType(Enum):
    """Types of code components that can be analyzed."""
    FUNCTION = "function"
    METHOD = "method"
    CLASS = "class"
    MODULE = "module"
    ASYNC_FUNCTION = "async_function"
    GENERATOR = "generator"
    PROPERTY = "property"


@dataclass
class CodeLocation:
    """Represents location information for code components."""
    line_start: int
    line_end: int
    column_start: int = 0
    column_end: int = 0
    
    @property
    def line_count(self) -> int:
        """Calculate number of lines in the component."""
        return max(1, self.line_end - self.line_start + 1)


@dataclass
class Parameter:
    """Represents a function/method parameter."""
    name: str
    type_annotation: Optional[str] = None
    default_value: Optional[str] = None
    is_vararg: bool = False
    is_kwarg: bool = False
    is_keyword_only: bool = False


@dataclass
class CodeComponent:
    """Comprehensive representation of a code component."""
    # Basic information
    name: str
    component_type: ComponentType
    location: CodeLocation
    source_code: str
    file_path: str
    
    # Function/Method specific
    parameters: List[Parameter] = field(default_factory=list)
    return_type: Optional[str] = None
    is_async: bool = False
    is_generator: bool = False
    is_property: bool = False
    is_static: bool = False
    is_class_method: bool = False
    
    # Class specific
    base_classes: List[str] = field(default_factory=list)
    methods: List[str] = field(default_factory=list)
    properties: List[str] = field(default_factory=list)
    
    # Complexity metrics
    cyclomatic_complexity: int = 1
    cognitive_complexity: int = 1
    complexity_density: float = 0.0
    maintainability_index: float = 100.0
    
    # Dependencies
    imports: List[str] = field(default_factory=list)
    function_calls: List[str] = field(default_factory=list)
    dependencies: Set[str] = field(default_factory=set)
    
    # Quality metrics
    lines_of_code: int = 0
    comment_lines: int = 0
    documentation_coverage: float = 0.0
    testability_score: float = 0.0
    test_priority: int = 1  # 1-5 scale
    
    # Documentation
    docstring: Optional[str] = None
    comments: List[str] = field(default_factory=list)
    
    def calculate_complexity_density(self) -> float:
        """Calculate complexity per line of code."""
        if self.lines_of_code > 0:
            self.complexity_density = self.cyclomatic_complexity / self.lines_of_code
        return self.complexity_density
    
    def calculate_testability_score(self) -> float:
        """Calculate testability based on complexity and dependencies."""
        # Lower complexity and fewer dependencies = higher testability
        complexity_factor = max(0, 1.0 - (self.cyclomatic_complexity - 1) / 20.0)
        dependency_factor = max(0, 1.0 - len(self.dependencies) / 10.0)
        parameter_factor = max(0, 1.0 - len(self.parameters) / 8.0)
        
        self.testability_score = (complexity_factor + dependency_factor + parameter_factor) / 3.0
        return self.testability_score
    
    def calculate_test_priority(self) -> int:
        """Calculate test priority (1-5) based on complexity and usage."""
        # Higher complexity = higher priority
        if self.cyclomatic_complexity >= 10:
            self.test_priority = 5
        elif self.cyclomatic_complexity >= 7:
            self.test_priority = 4
        elif self.cyclomatic_complexity >= 4:
            self.test_priority = 3
        elif self.cyclomatic_complexity >= 2:
            self.test_priority = 2
        else:
            self.test_priority = 1
            
        return self.test_priority


class PythonASTParser:
    """Advanced Python AST parser with comprehensive analysis."""
    
    def __init__(self):
        self.components: List[CodeComponent] = []
        self.file_path: str = ""
        self.source_lines: List[str] = []
        
    def parse_file(self, file_path: Union[str, Path]) -> List[CodeComponent]:
        """Parse a Python file and extract all components."""
        file_path = Path(file_path)
        self.file_path = str(file_path)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
                
            self.source_lines = source_code.splitlines()
            return self.parse_source(source_code)
            
        except Exception as e:
            logger.error(f"Error parsing file {file_path}: {e}")
            return []
    
    def parse_source(self, source_code: str) -> List[CodeComponent]:
        """Parse Python source code and extract components."""
        self.components = []
        
        try:
            # Parse with both ast and astroid for comprehensive analysis
            tree = ast.parse(source_code)
            
            # Use astroid for additional analysis
            try:
                astroid_module = astroid.parse(source_code)
                self._analyze_with_astroid(astroid_module)
            except Exception as e:
                logger.warning(f"Astroid analysis failed: {e}")
            
            # Extract components using AST visitor
            visitor = CodeComponentVisitor(self.file_path, self.source_lines)
            visitor.visit(tree)
            self.components = visitor.components
            
            # Calculate complexity metrics
            self._calculate_complexity_metrics(source_code)
            
            # Analyze dependencies
            self._analyze_dependencies(tree)
            
            # Calculate quality scores
            self._calculate_quality_metrics()
            
            logger.info(f"Parsed {len(self.components)} components from source")
            return self.components
            
        except SyntaxError as e:
            logger.error(f"Syntax error in source code: {e}")
            return []
        except Exception as e:
            logger.error(f"Error parsing source code: {e}")
            return []
    
    def _analyze_with_astroid(self, module: astroid.Module) -> None:
        """Use astroid for additional analysis capabilities."""
        try:
            # Extract imports and dependencies
            for node in module.nodes_of_class(astroid.Import):
                for name, alias in node.names:
                    # Will be used to enhance dependency analysis
                    pass
                    
            for node in module.nodes_of_class(astroid.ImportFrom):
                if node.modname:
                    # Will be used to enhance dependency analysis
                    pass
                    
        except Exception as e:
            logger.warning(f"Astroid analysis error: {e}")
    
    def _calculate_complexity_metrics(self, source_code: str) -> None:
        """Calculate complexity metrics for all components."""
        try:
            # Calculate radon complexity
            complexity_results = cc.cc_visit(source_code)
            
            # Create mapping of component names to complexity
            complexity_map = {}
            for result in complexity_results:
                complexity_map[result.name] = result.complexity
            
            # Apply complexity to components
            for component in self.components:
                if component.name in complexity_map:
                    component.cyclomatic_complexity = complexity_map[component.name]
                
                # Calculate cognitive complexity (simplified heuristic)
                component.cognitive_complexity = self._estimate_cognitive_complexity(component)
                
                # Calculate complexity density
                component.calculate_complexity_density()
                
        except Exception as e:
            logger.warning(f"Complexity calculation failed: {e}")
    
    def _estimate_cognitive_complexity(self, component: CodeComponent) -> int:
        """Estimate cognitive complexity using heuristics."""
        # Simplified cognitive complexity estimation
        cognitive = component.cyclomatic_complexity
        
        # Add complexity for nested structures
        nesting_indicators = ['if', 'for', 'while', 'try', 'with']
        source_lines = component.source_code.lower().split('\n')
        
        for line in source_lines:
            for indicator in nesting_indicators:
                if indicator in line:
                    cognitive += line.count('    ') // 4  # Indentation level
                    
        return max(1, cognitive)
    
    def _analyze_dependencies(self, tree: ast.AST) -> None:
        """Analyze dependencies and function calls."""
        try:
            dependency_visitor = DependencyVisitor()
            dependency_visitor.visit(tree)
            
            # Map dependencies to components by line numbers
            for component in self.components:
                component.dependencies = dependency_visitor.get_dependencies_for_range(
                    component.location.line_start, component.location.line_end
                )
                component.function_calls = dependency_visitor.get_calls_for_range(
                    component.location.line_start, component.location.line_end
                )
                component.imports = dependency_visitor.imports
                
        except Exception as e:
            logger.warning(f"Dependency analysis failed: {e}")
    
    def _calculate_quality_metrics(self) -> None:
        """Calculate quality metrics for all components."""
        for component in self.components:
            # Calculate lines of code
            component.lines_of_code = component.location.line_count
            
            # Calculate documentation coverage
            if component.docstring:
                component.documentation_coverage = 1.0
            else:
                component.documentation_coverage = 0.0
            
            # Calculate testability score
            component.calculate_testability_score()
            
            # Calculate test priority
            component.calculate_test_priority()


class CodeComponentVisitor(ast.NodeVisitor):
    """AST visitor to extract code components."""
    
    def __init__(self, file_path: str, source_lines: List[str]):
        self.file_path = file_path
        self.source_lines = source_lines
        self.components: List[CodeComponent] = []
        self.current_class: Optional[str] = None
        
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit function definitions."""
        self._create_function_component(node, is_async=False)
        self.generic_visit(node)
    
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Visit async function definitions."""
        self._create_function_component(node, is_async=True)
        self.generic_visit(node)
    
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit class definitions."""
        old_class = self.current_class
        self.current_class = node.name
        
        # Create class component
        location = CodeLocation(
            line_start=node.lineno,
            line_end=node.end_lineno or node.lineno,
            column_start=node.col_offset,
            column_end=node.end_col_offset or 0
        )
        
        source_code = self._extract_source_code(location)
        
        # Extract base classes
        base_classes = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                base_classes.append(base.id)
            elif isinstance(base, ast.Attribute):
                base_classes.append(ast.unparse(base))
        
        # Extract docstring
        docstring = ast.get_docstring(node)
        
        component = CodeComponent(
            name=node.name,
            component_type=ComponentType.CLASS,
            location=location,
            source_code=source_code,
            file_path=self.file_path,
            base_classes=base_classes,
            docstring=docstring
        )
        
        self.components.append(component)
        self.generic_visit(node)
        self.current_class = old_class
    
    def _create_function_component(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef], is_async: bool = False) -> None:
        """Create a function component from AST node."""
        location = CodeLocation(
            line_start=node.lineno,
            line_end=node.end_lineno or node.lineno,
            column_start=node.col_offset,
            column_end=node.end_col_offset or 0
        )
        
        source_code = self._extract_source_code(location)
        
        # Determine component type
        if self.current_class:
            component_type = ComponentType.METHOD
        elif is_async:
            component_type = ComponentType.ASYNC_FUNCTION
        elif self._is_generator(node):
            component_type = ComponentType.GENERATOR
        else:
            component_type = ComponentType.FUNCTION
        
        # Extract parameters
        parameters = self._extract_parameters(node.args)
        
        # Extract return type
        return_type = None
        if node.returns:
            try:
                return_type = ast.unparse(node.returns)
            except:
                return_type = str(node.returns)
        
        # Extract docstring
        docstring = ast.get_docstring(node)
        
        # Check for decorators
        is_property = any(
            (isinstance(d, ast.Name) and d.id == 'property') or
            (isinstance(d, ast.Attribute) and d.attr == 'property')
            for d in node.decorator_list
        )
        
        is_static = any(
            isinstance(d, ast.Name) and d.id == 'staticmethod'
            for d in node.decorator_list
        )
        
        is_class_method = any(
            isinstance(d, ast.Name) and d.id == 'classmethod'
            for d in node.decorator_list
        )
        
        component = CodeComponent(
            name=node.name,
            component_type=component_type,
            location=location,
            source_code=source_code,
            file_path=self.file_path,
            parameters=parameters,
            return_type=return_type,
            is_async=is_async,
            is_generator=self._is_generator(node),
            is_property=is_property,
            is_static=is_static,
            is_class_method=is_class_method,
            docstring=docstring
        )
        
        self.components.append(component)
    
    def _extract_parameters(self, args: ast.arguments) -> List[Parameter]:
        """Extract function parameters from AST arguments."""
        parameters = []
        
        # Regular arguments
        for i, arg in enumerate(args.args):
            param = Parameter(
                name=arg.arg,
                type_annotation=ast.unparse(arg.annotation) if arg.annotation else None
            )
            
            # Check for default values
            default_offset = len(args.args) - len(args.defaults)
            if i >= default_offset:
                default_idx = i - default_offset
                try:
                    param.default_value = ast.unparse(args.defaults[default_idx])
                except:
                    param.default_value = str(args.defaults[default_idx])
            
            parameters.append(param)
        
        # Keyword-only arguments
        for i, arg in enumerate(args.kwonlyargs):
            param = Parameter(
                name=arg.arg,
                type_annotation=ast.unparse(arg.annotation) if arg.annotation else None,
                is_keyword_only=True
            )
            
            if args.kw_defaults and i < len(args.kw_defaults) and args.kw_defaults[i]:
                try:
                    param.default_value = ast.unparse(args.kw_defaults[i])
                except:
                    param.default_value = str(args.kw_defaults[i])
            
            parameters.append(param)
        
        # *args parameter
        if args.vararg:
            param = Parameter(
                name=args.vararg.arg,
                type_annotation=ast.unparse(args.vararg.annotation) if args.vararg.annotation else None,
                is_vararg=True
            )
            parameters.append(param)
        
        # **kwargs parameter
        if args.kwarg:
            param = Parameter(
                name=args.kwarg.arg,
                type_annotation=ast.unparse(args.kwarg.annotation) if args.kwarg.annotation else None,
                is_kwarg=True
            )
            parameters.append(param)
        
        return parameters
    
    def _is_generator(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> bool:
        """Check if function is a generator (contains yield)."""
        for child in ast.walk(node):
            if isinstance(child, (ast.Yield, ast.YieldFrom)):
                return True
        return False
    
    def _extract_source_code(self, location: CodeLocation) -> str:
        """Extract source code for the given location."""
        try:
            start_idx = max(0, location.line_start - 1)
            end_idx = min(len(self.source_lines), location.line_end)
            return '\n'.join(self.source_lines[start_idx:end_idx])
        except:
            return ""


class DependencyVisitor(ast.NodeVisitor):
    """AST visitor to analyze dependencies and function calls."""
    
    def __init__(self):
        self.imports: List[str] = []
        self.function_calls: List[tuple] = []  # (name, line_number)
        self.dependencies: List[tuple] = []    # (name, line_number)
    
    def visit_Import(self, node: ast.Import) -> None:
        """Visit import statements."""
        for alias in node.names:
            self.imports.append(alias.name)
            self.dependencies.append((alias.name, node.lineno))
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Visit from-import statements."""
        if node.module:
            self.imports.append(node.module)
            for alias in node.names:
                imported_name = f"{node.module}.{alias.name}"
                self.dependencies.append((imported_name, node.lineno))
        self.generic_visit(node)
    
    def visit_Call(self, node: ast.Call) -> None:
        """Visit function calls."""
        try:
            if isinstance(node.func, ast.Name):
                self.function_calls.append((node.func.id, node.lineno))
            elif isinstance(node.func, ast.Attribute):
                call_name = ast.unparse(node.func)
                self.function_calls.append((call_name, node.lineno))
        except:
            pass
        self.generic_visit(node)
    
    def get_dependencies_for_range(self, start_line: int, end_line: int) -> Set[str]:
        """Get dependencies within a specific line range."""
        return {
            name for name, line in self.dependencies
            if start_line <= line <= end_line
        }
    
    def get_calls_for_range(self, start_line: int, end_line: int) -> List[str]:
        """Get function calls within a specific line range."""
        return [
            name for name, line in self.function_calls
            if start_line <= line <= end_line
        ]


class MultiLanguageParser:
    """Main parser interface supporting multiple languages."""
    
    def __init__(self):
        self.python_parser = PythonASTParser()
        # Placeholder for future JS/TS parsers
        self.js_parser = None
        self.ts_parser = None
    
    def parse_file(self, file_path: Union[str, Path]) -> List[CodeComponent]:
        """Parse a file based on its language."""
        file_path = Path(file_path)
        language = self._detect_language(file_path)
        
        if language == Language.PYTHON:
            return self.python_parser.parse_file(file_path)
        elif language == Language.JAVASCRIPT:
            logger.warning("JavaScript parsing not yet implemented")
            return self._create_placeholder_component(file_path, Language.JAVASCRIPT)
        elif language == Language.TYPESCRIPT:
            logger.warning("TypeScript parsing not yet implemented")
            return self._create_placeholder_component(file_path, Language.TYPESCRIPT)
        else:
            logger.warning(f"Unsupported language for file: {file_path}")
            return []
    
    def _detect_language(self, file_path: Path) -> Language:
        """Detect programming language from file extension."""
        extension = file_path.suffix.lower()
        
        if extension in ['.py', '.pyw']:
            return Language.PYTHON
        elif extension in ['.js', '.mjs']:
            return Language.JAVASCRIPT
        elif extension in ['.ts', '.tsx']:
            return Language.TYPESCRIPT
        else:
            # Default to Python for unknown extensions
            return Language.PYTHON
    
    def _create_placeholder_component(self, file_path: Path, language: Language) -> List[CodeComponent]:
        """Create placeholder component for unsupported languages."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
                
            location = CodeLocation(
                line_start=1,
                line_end=len(source_code.splitlines())
            )
            
            component = CodeComponent(
                name=file_path.stem,
                component_type=ComponentType.MODULE,
                location=location,
                source_code=source_code,
                file_path=str(file_path),
                lines_of_code=location.line_count
            )
            
            return [component]
            
        except Exception as e:
            logger.error(f"Error creating placeholder for {file_path}: {e}")
            return []


# Main parser interface
def parse_code_file(file_path: Union[str, Path]) -> List[CodeComponent]:
    """
    Parse a code file and return extracted components.
    
    Args:
        file_path: Path to the code file to parse
        
    Returns:
        List of CodeComponent objects extracted from the file
    """
    parser = MultiLanguageParser()
    return parser.parse_file(file_path)


def analyze_complexity_metrics(components: List[CodeComponent]) -> Dict[str, Any]:
    """
    Analyze complexity metrics across multiple components.
    
    Args:
        components: List of CodeComponent objects
        
    Returns:
        Dictionary containing aggregate complexity metrics
    """
    if not components:
        return {}
    
    complexities = [c.cyclomatic_complexity for c in components]
    testability_scores = [c.testability_score for c in components]
    
    return {
        "total_components": len(components),
        "average_complexity": sum(complexities) / len(complexities),
        "max_complexity": max(complexities),
        "min_complexity": min(complexities),
        "high_complexity_count": len([c for c in complexities if c >= 10]),
        "average_testability": sum(testability_scores) / len(testability_scores),
        "components_by_type": {
            comp_type.value: len([c for c in components if c.component_type == comp_type])
            for comp_type in ComponentType
        },
        "test_priority_distribution": {
            f"priority_{i}": len([c for c in components if c.test_priority == i])
            for i in range(1, 6)
        }
    }
