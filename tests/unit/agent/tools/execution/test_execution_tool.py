"""
Tests for Agent Execution Tool

Comprehensive tests for the intelligent execution tool including safe execution,
monitoring, result interpretation, and learning capabilities.
"""

import pytest
import asyncio
import tempfile
import os
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime

from src.agent.tools.execution_tool import (
    ExecutionTool,
    SafeExecutionEnvironment,
    ResourceMonitor,
    TestResultParser,
    IntelligentResultAnalyzer,
    ExecutionResult,
    TestResult,
    MonitoredExecutionResult,
    TestResultInterpretation
)
from src.execution.models import ExecutionContext


class TestResourceMonitor:
    """Test resource monitoring capabilities"""
    
    def setup_method(self):
        self.monitor = ResourceMonitor()
    
    def test_monitoring_initialization(self):
        """Test monitor initialization"""
        assert self.monitor.start_time is None
        assert self.monitor.process is None
        assert self.monitor.peak_memory == 0
    
    def test_monitoring_data_collection(self):
        """Test monitoring data structure"""
        data = self.monitor.get_monitoring_data()
        
        # Should return dict with expected keys
        expected_keys = [
            "duration", "initial_memory_mb", "peak_memory_mb", 
            "memory_increase_mb", "average_cpu_percent", "cpu_samples"
        ]
        
        for key in expected_keys:
            assert key in data


class TestTestResultParser:
    """Test test result parsing capabilities"""
    
    def setup_method(self):
        self.parser = TestResultParser()
    
    def test_pytest_output_parsing(self):
        """Test parsing of pytest output"""
        pytest_output = """
test_example.py::test_function PASSED                          [100%]
test_example.py::test_another FAILED                           [100%]

====================== 1 failed, 1 passed in 0.12s ======================
        """
        
        result = self.parser.parse_pytest_output(pytest_output, "", 1)
        
        assert result.total_tests == 2
        assert result.passed_tests == 1
        assert result.failed_tests == 1
        assert result.success == False  # Exit code 1
    
    def test_unittest_output_parsing(self):
        """Test parsing of unittest output"""
        unittest_output = """
test_function (test_example.TestCase) ... ok
test_another (test_example.TestCase) ... FAIL

======================================================================
FAIL: test_another (test_example.TestCase)
----------------------------------------------------------------------
AssertionError: False != True

----------------------------------------------------------------------
Ran 2 tests in 0.001s

FAILED (failures=1)
        """
        
        result = self.parser.parse_unittest_output(unittest_output, "", 1)
        
        assert result.total_tests == 2
        assert result.passed_tests == 1
        assert result.failed_tests == 1


class TestIntelligentResultAnalyzer:
    """Test intelligent result analysis"""
    
    def setup_method(self):
        self.analyzer = IntelligentResultAnalyzer()
    
    def test_failure_categorization(self):
        """Test categorization of test failures"""
        assertion_failure = TestResult(
            test_name="test_assertion",
            status="failed",
            duration=0.1,
            traceback="AssertionError: Expected True but got False"
        )
        
        type_error_failure = TestResult(
            test_name="test_type",
            status="failed", 
            duration=0.1,
            traceback="TypeError: unsupported operand type(s)"
        )
        
        failures = [assertion_failure, type_error_failure]
        analysis = self.analyzer.analyze_test_failures(failures)
        
        assert "assertion_failure" in analysis["failure_categories"]
        assert "type_error" in analysis["failure_categories"]
        assert len(analysis["improvement_suggestions"]) > 0


class TestExecutionTool:
    """Test the main execution tool"""
    
    def setup_method(self):
        self.tool = ExecutionTool()
    
    @pytest.mark.asyncio
    async def test_tool_can_handle_execution_tasks(self):
        """Test tool capability assessment for execution tasks"""
        assert self.tool.can_handle("execute tests") > 0.8
        assert self.tool.can_handle("run pytest") > 0.8
        assert self.tool.can_handle("validate code") > 0.3
        assert self.tool.can_handle("make coffee") < 0.2
    
    @pytest.mark.asyncio
    async def test_simple_code_execution(self):
        """Test execution of simple test code"""
        simple_test = """
def test_simple():
    assert 1 + 1 == 2

def test_another():
    assert "hello" == "hello"
        """
        
        parameters = {
            "tests": simple_test,
            "language": "python",
            "framework": "pytest"
        }
        
        # Mock the execution environment to avoid actual subprocess
        with patch.object(self.tool, 'execute_with_monitoring') as mock_exec:
            mock_result = MonitoredExecutionResult(
                execution_result=ExecutionResult(
                    success=True,
                    total_tests=2,
                    passed_tests=2,
                    failed_tests=0,
                    error_tests=0,
                    skipped_tests=0,
                    total_duration=0.1,
                    output="2 tests passed",
                    error_output="",
                    exit_code=0
                ),
                monitoring_data={"duration": 0.1, "memory_mb": 50},
                interpretation="All tests passed successfully",
                insights=["Fast execution"],
                recommendations=["Good test coverage"],
                confidence=0.9
            )
            mock_exec.return_value = mock_result
            
            result = await self.tool.execute(parameters)
            
            assert result.success == True
            assert "execution_result" in result.data
    
    def test_execution_command_building(self):
        """Test building of execution commands"""
        # Test Python pytest command
        python_context = ExecutionContext(
            language="python",
            framework="pytest",
            tests="",
            context={}
        )
        
        command = self.tool._build_execution_command(python_context, "/tmp/test.py", {})
        assert any("python" in str(cmd) for cmd in command)
        assert "pytest" in command


if __name__ == "__main__":
    pytest.main([__file__])
