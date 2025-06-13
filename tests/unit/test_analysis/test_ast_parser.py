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
