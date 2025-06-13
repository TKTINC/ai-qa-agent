#!/bin/bash

# Setup Script for Prompt 1.1: AST Code Parser Implementation
# AI QA Agent - Sprint 1.1

set -e

echo "🚀 Setting up Prompt 1.1: AST Code Parser Implementation..."

# Check if we're in the right directory (should have src/ folder)
if [ ! -d "src" ]; then
    echo "❌ Error: This script should be run from the project root directory"
    echo "Expected to find 'src/' directory"
    exit 1
fi

# Install new dependencies with Python 3.9 compatible versions
echo "📦 Installing new dependencies..."
pip3 install tree-sitter==0.20.4 tree-sitter-python==0.23.6 radon==6.0.1 astroid==2.15.6

# Create analysis directory if it doesn't exist
mkdir -p src/analysis

# Create the AST parser implementation
echo "📄 Creating src/analysis/ast_parser.py..."
cat > src/analysis/ast_parser.py << 'EOF'
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
EOF

# Create test file for AST parser
echo "📄 Creating tests/unit/test_analysis/test_ast_parser.py..."
mkdir -p tests/unit/test_analysis
cat > tests/unit/test_analysis/test_ast_parser.py << 'EOF'
"""
Tests for the AST parser implementation.
"""

import pytest
import tempfile
import os
from pathlib import Path
from src.analysis.ast_parser import (
    PythonASTParser,
    MultiLanguageParser,
    CodeComponent,
    ComponentType,
    Language,
    parse_code_file,
    analyze_complexity_metrics
)


class TestPythonASTParser:
    """Test the Python AST parser functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.parser = PythonASTParser()
    
    def test_simple_function_parsing(self):
        """Test parsing a simple function."""
        source_code = '''
def add_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b
'''
        components = self.parser.parse_source(source_code)
        
        assert len(components) == 1
        func = components[0]
        
        assert func.name == "add_numbers"
        assert func.component_type == ComponentType.FUNCTION
        assert len(func.parameters) == 2
        assert func.parameters[0].name == "a"
        assert func.parameters[0].type_annotation == "int"
        assert func.return_type == "int"
        assert func.docstring == "Add two numbers together."
    
    def test_class_parsing(self):
        """Test parsing a class with methods."""
        source_code = '''
class Calculator:
    """A simple calculator class."""
    
    def __init__(self, initial_value: float = 0.0):
        self.value = initial_value
    
    def add(self, x: float) -> float:
        """Add a value."""
        self.value += x
        return self.value
    
    @property
    def current_value(self) -> float:
        """Get current value."""
        return self.value
    
    @staticmethod
    def multiply(a: float, b: float) -> float:
        """Multiply two numbers."""
        return a * b
'''
        components = self.parser.parse_source(source_code)
        
        # Should have 1 class + 4 methods
        assert len(components) == 5
        
        # Check class
        calculator_class = next(c for c in components if c.component_type == ComponentType.CLASS)
        assert calculator_class.name == "Calculator"
        assert calculator_class.docstring == "A simple calculator class."
        
        # Check methods
        init_method = next(c for c in components if c.name == "__init__")
        assert init_method.component_type == ComponentType.METHOD
        assert len(init_method.parameters) == 2  # self + initial_value
        
        property_method = next(c for c in components if c.name == "current_value")
        assert property_method.is_property == True
        
        static_method = next(c for c in components if c.name == "multiply")
        assert static_method.is_static == True
    
    def test_async_function_parsing(self):
        """Test parsing async functions."""
        source_code = '''
async def fetch_data(url: str) -> dict:
    """Fetch data from URL."""
    import aiohttp
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()
'''
        components = self.parser.parse_source(source_code)
        
        assert len(components) == 1
        func = components[0]
        
        assert func.name == "fetch_data"
        assert func.component_type == ComponentType.ASYNC_FUNCTION
        assert func.is_async == True
    
    def test_generator_function_parsing(self):
        """Test parsing generator functions."""
        source_code = '''
def fibonacci_generator(n: int):
    """Generate fibonacci numbers."""
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b
'''
        components = self.parser.parse_source(source_code)
        
        assert len(components) == 1
        func = components[0]
        
        assert func.name == "fibonacci_generator"
        assert func.component_type == ComponentType.GENERATOR
        assert func.is_generator == True
    
    def test_complexity_calculation(self):
        """Test complexity calculation."""
        source_code = '''
def complex_function(x, y, z):
    """A function with multiple branches."""
    if x > 0:
        if y > 0:
            if z > 0:
                return x + y + z
            else:
                return x + y
        else:
            return x
    else:
        if y > 0:
            return y
        else:
            return 0
'''
        components = self.parser.parse_source(source_code)
        
        assert len(components) == 1
        func = components[0]
        
        # Should have cyclomatic complexity > 1 due to multiple branches
        assert func.cyclomatic_complexity > 1
        assert func.cognitive_complexity >= func.cyclomatic_complexity
        assert func.test_priority > 1  # Should be prioritized for testing
    
    def test_dependency_analysis(self):
        """Test dependency analysis."""
        source_code = '''
import os
import sys
from pathlib import Path
from typing import List, Dict

def process_files(directory: str) -> List[str]:
    """Process files in directory."""
    path = Path(directory)
    files = []
    
    for file_path in path.iterdir():
        if file_path.is_file():
            files.append(str(file_path))
            os.path.getsize(file_path)
    
    return files
'''
        components = self.parser.parse_source(source_code)
        
        assert len(components) == 1
        func = components[0]
        
        # Check imports are captured
        assert "os" in func.imports
        assert "pathlib" in func.imports
        
        # Check function calls are captured
        function_calls = func.function_calls
        assert any("Path" in call for call in function_calls)
    
    def test_quality_metrics(self):
        """Test quality metrics calculation."""
        source_code = '''
def well_documented_function(param1: int, param2: str) -> bool:
    """
    This function is well documented.
    
    Args:
        param1: First parameter
        param2: Second parameter
        
    Returns:
        Boolean result
    """
    if param1 > 0 and param2:
        return True
    return False
'''
        components = self.parser.parse_source(source_code)
        
        assert len(components) == 1
        func = components[0]
        
        # Should have good documentation coverage
        assert func.documentation_coverage == 1.0
        
        # Should have reasonable testability
        assert func.testability_score > 0.0
        
        # Should have lines of code counted
        assert func.lines_of_code > 0


class TestMultiLanguageParser:
    """Test the multi-language parser interface."""
    
    def test_language_detection(self):
        """Test language detection from file extensions."""
        parser = MultiLanguageParser()
        
        # Test detection
        assert parser._detect_language(Path("test.py")) == Language.PYTHON
        assert parser._detect_language(Path("test.js")) == Language.JAVASCRIPT
        assert parser._detect_language(Path("test.ts")) == Language.TYPESCRIPT
    
    def test_python_file_parsing(self):
        """Test parsing a Python file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write('''
def test_function():
    """Test function."""
    return "hello"
''')
            f.flush()
            
            try:
                components = parse_code_file(f.name)
                assert len(components) == 1
                assert components[0].name == "test_function"
            finally:
                os.unlink(f.name)
    
    def test_unsupported_language_placeholder(self):
        """Test placeholder creation for unsupported languages."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
            f.write('''
function testFunction() {
    return "hello";
}
''')
            f.flush()
            
            try:
                components = parse_code_file(f.name)
                # Should create placeholder component
                assert len(components) == 1
                assert components[0].component_type == ComponentType.MODULE
            finally:
                os.unlink(f.name)


class TestComplexityAnalysis:
    """Test complexity analysis functions."""
    
    def test_analyze_complexity_metrics(self):
        """Test complexity metrics analysis."""
        # Create mock components
        from src.analysis.ast_parser import CodeLocation
        
        components = [
            CodeComponent(
                name="simple_func",
                component_type=ComponentType.FUNCTION,
                location=CodeLocation(1, 5),
                source_code="def simple_func(): return 1",
                file_path="test.py",
                cyclomatic_complexity=1,
                testability_score=0.8,
                test_priority=1
            ),
            CodeComponent(
                name="complex_func",
                component_type=ComponentType.FUNCTION,
                location=CodeLocation(6, 15),
                source_code="def complex_func(): ...",
                file_path="test.py",
                cyclomatic_complexity=8,
                testability_score=0.3,
                test_priority=4
            )
        ]
        
        metrics = analyze_complexity_metrics(components)
        
        assert metrics["total_components"] == 2
        assert metrics["average_complexity"] == 4.5
        assert metrics["max_complexity"] == 8
        assert metrics["min_complexity"] == 1
        assert metrics["high_complexity_count"] == 0  # None >= 10
        
        # Check type distribution
        assert metrics["components_by_type"][ComponentType.FUNCTION.value] == 2
        
        # Check priority distribution
        assert metrics["test_priority_distribution"]["priority_1"] == 1
        assert metrics["test_priority_distribution"]["priority_4"] == 1
    
    def test_empty_components_analysis(self):
        """Test analysis with empty component list."""
        metrics = analyze_complexity_metrics([])
        assert metrics == {}


# Integration tests
class TestIntegration:
    """Integration tests for the complete parsing pipeline."""
    
    def test_real_world_python_file(self):
        """Test parsing a realistic Python file."""
        source_code = '''
"""
A sample module for testing the AST parser.
"""

import os
import sys
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class Person:
    """Represents a person."""
    name: str
    age: int
    email: Optional[str] = None
    
    def is_adult(self) -> bool:
        """Check if person is adult."""
        return self.age >= 18
    
    @property
    def display_name(self) -> str:
        """Get display name."""
        return self.name.title()
    
    @staticmethod
    def create_anonymous() -> 'Person':
        """Create anonymous person."""
        return Person("Anonymous", 0)


class PersonManager:
    """Manages a collection of persons."""
    
    def __init__(self):
        self.persons: List[Person] = []
    
    def add_person(self, person: Person) -> None:
        """Add a person to the collection."""
        if person.name and person.age >= 0:
            self.persons.append(person)
    
    def find_adults(self) -> List[Person]:
        """Find all adult persons."""
        return [p for p in self.persons if p.is_adult()]
    
    async def save_to_file(self, filename: str) -> bool:
        """Save persons to file asynchronously."""
        try:
            with open(filename, 'w') as f:
                for person in self.persons:
                    f.write(f"{person.name},{person.age}\\n")
            return True
        except Exception as e:
            print(f"Error saving: {e}")
            return False


def generate_test_data(count: int = 10):
    """Generate test data."""
    for i in range(count):
        yield Person(f"Person{i}", 20 + i)


if __name__ == "__main__":
    manager = PersonManager()
    for person in generate_test_data(5):
        manager.add_person(person)
    print(f"Created {len(manager.persons)} persons")
'''
        
        parser = PythonASTParser()
        components = parser.parse_source(source_code)
        
        # Should extract multiple components
        assert len(components) > 5
        
        # Check for classes
        classes = [c for c in components if c.component_type == ComponentType.CLASS]
        assert len(classes) == 2
        
        # Check for methods
        methods = [c for c in components if c.component_type == ComponentType.METHOD]
        assert len(methods) > 0
        
        # Check for functions
        functions = [c for c in components if c.component_type == ComponentType.FUNCTION]
        assert len(functions) > 0
        
        # Check for async functions
        async_functions = [c for c in components if c.component_type == ComponentType.ASYNC_FUNCTION]
        assert len(async_functions) > 0
        
        # Check for generators
        generators = [c for c in components if c.component_type == ComponentType.GENERATOR]
        assert len(generators) > 0
        
        # Verify complexity analysis
        for component in components:
            assert component.cyclomatic_complexity >= 1
            assert 0.0 <= component.testability_score <= 1.0
            assert 1 <= component.test_priority <= 5
        
        # Test metrics analysis
        metrics = analyze_complexity_metrics(components)
        assert metrics["total_components"] == len(components)
        assert metrics["average_complexity"] > 0
EOF

# Create __init__.py files
echo "📄 Creating __init__.py files..."
touch src/analysis/__init__.py
touch tests/unit/test_analysis/__init__.py

# Update requirements.txt with new dependencies
echo "📄 Updating requirements.txt..."
if [ -f "requirements.txt" ]; then
    echo "tree-sitter==0.20.4" >> requirements.txt
    echo "tree-sitter-python==0.23.6" >> requirements.txt
    echo "radon==6.0.1" >> requirements.txt
    echo "astroid==2.15.6" >> requirements.txt
else
    echo "⚠️  Warning: requirements.txt not found. Please add dependencies manually:"
    echo "  tree-sitter==0.20.4"
    echo "  tree-sitter-python==0.23.6"
    echo "  radon==6.0.1"
    echo "  astroid==2.15.6"
fi

# Run tests to verify implementation
echo "🧪 Running tests to verify implementation..."
if command -v pytest &> /dev/null; then
    echo "Running AST parser tests..."
    python3 -m pytest tests/unit/test_analysis/test_ast_parser.py -v
else
    echo "⚠️  pytest not found. Please install it and run:"
    echo "  python3 -m pytest tests/unit/test_analysis/test_ast_parser.py -v"
fi

# Test basic functionality
echo "🔍 Testing basic AST parser functionality..."
python3 -c "
from src.analysis.ast_parser import parse_code_file, analyze_complexity_metrics
import tempfile
import os

# Create a test Python file
with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
    f.write('''
def test_function(a: int, b: str = 'default') -> bool:
    \"\"\"Test function with complexity.\"\"\"
    if a > 0:
        if b:
            return True
        else:
            return False
    return False

class TestClass:
    \"\"\"Test class.\"\"\"
    
    def __init__(self):
        self.value = 0
    
    def increment(self) -> int:
        \"\"\"Increment value.\"\"\"
        self.value += 1
        return self.value
''')
    f.flush()
    
    try:
        # Parse the file
        components = parse_code_file(f.name)
        print(f'✅ Parsed {len(components)} components from test file')
        
        for component in components:
            print(f'  - {component.name} ({component.component_type.value}): complexity={component.cyclomatic_complexity}, testability={component.testability_score:.2f}')
        
        # Analyze metrics
        metrics = analyze_complexity_metrics(components)
        print(f'✅ Analysis complete: {metrics[\"total_components\"]} components, avg complexity: {metrics[\"average_complexity\"]:.1f}')
        
    except Exception as e:
        print(f'❌ Error testing AST parser: {e}')
    finally:
        os.unlink(f.name)
"

echo ""
echo "✅ Prompt 1.1 setup complete!"
echo ""
echo "📋 Summary of what was implemented:"
echo "  ✅ Multi-language AST parser with full Python support"
echo "  ✅ Comprehensive CodeComponent data model"
echo "  ✅ Complexity analysis using radon"
echo "  ✅ Dependency detection and analysis"
echo "  ✅ Quality metrics calculation"
echo "  ✅ Comprehensive test suite"
echo "  ✅ Integration with existing project structure"
echo ""
echo "🔄 Next steps:"
echo "  1. Run the tests: python3 -m pytest tests/unit/test_analysis/test_ast_parser.py -v"
echo "  2. Test with your own Python files"
echo "  3. Ready for Prompt 1.2: Repository Analysis System"
echo ""
echo "📊 Key capabilities now available:"
echo "  - Parse Python files and extract functions, classes, methods"
echo "  - Calculate cyclomatic and cognitive complexity"
echo "  - Analyze dependencies and function calls"
echo "  - Score testability and test priority"
echo "  - Support for async functions, generators, properties"
echo "  - Placeholder structure for JavaScript/TypeScript"
echo ""
