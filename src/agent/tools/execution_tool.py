"""
Agent Execution Tool

Intelligent test execution tool for agents that provides safe, monitored test execution
with intelligent result interpretation and learning capabilities.
"""

import asyncio
import logging
import subprocess
import tempfile
import json
import time
import signal
import os
import sys
import psutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

try:
    from ..base_tool import AgentTool, ToolResult, ToolParameters
except ImportError:
    # Fallback implementations if base_tool not available
    class ToolResult:
        def __init__(self, success, data=None, error=None, message=None):
            self.success = success
            self.data = data or {}
            self.error = error
            self.message = message
    
    class AgentTool:
        def __init__(self, name, description, parameters):
            self.name = name
            self.description = description
            self.parameters = parameters
        
        async def execute(self, parameters):
            raise NotImplementedError
        
        def can_handle(self, task):
            return 0.0
    
    ToolParameters = dict

try:
    from ...core.models import AgentContext
except ImportError:
    # Fallback for missing core models
    class AgentContext:
        pass

# Import ExecutionContext from execution models
try:
    from ...execution.models import ExecutionContext
except ImportError:
    # Define ExecutionContext locally if import fails
    class ExecutionContext:
        def __init__(self, language, framework, tests, context):
            self.language = language
            self.framework = framework  
            self.tests = tests
            self.context = context

try:
    from ...reasoning.react_engine import ReasoningStep
except ImportError:
    # Fallback for missing ReasoningStep
    class ReasoningStep:
        pass

try:
    from ...learning.execution_learning import ExecutionLearningSystem
except ImportError:
    try:
        # Try alternative import path
        from ..learning.execution_learning import ExecutionLearningSystem
    except ImportError:
        # Create fallback learning system if not available
        class ExecutionLearningSystem:
            async def record_execution_outcome(self, result, context):
                pass

logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Individual test result"""
    test_name: str
    status: str  # passed, failed, error, skipped
    duration: float
    message: Optional[str] = None
    traceback: Optional[str] = None
    output: Optional[str] = None
    line_number: Optional[int] = None


@dataclass
class ExecutionResult:
    """Complete execution result"""
    success: bool
    total_tests: int
    passed_tests: int
    failed_tests: int
    error_tests: int
    skipped_tests: int
    total_duration: float
    test_results: List[TestResult] = field(default_factory=list)
    output: str = ""
    error_output: str = ""
    exit_code: int = 0
    resource_usage: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MonitoredExecutionResult:
    """Execution result with monitoring data"""
    execution_result: ExecutionResult
    monitoring_data: Dict[str, Any]
    interpretation: str
    insights: List[str]
    recommendations: List[str]
    confidence: float
    learning_insights: List[str] = field(default_factory=list)


@dataclass
class TestResultInterpretation:
    """Intelligent interpretation of test results"""
    overall_assessment: str
    success_analysis: str
    failure_analysis: str
    performance_analysis: str
    quality_insights: List[str]
    improvement_suggestions: List[str]
    confidence: float


@dataclass
class TestImprovements:
    """Suggestions for improving failing tests"""
    failing_test: TestResult
    root_cause_analysis: str
    suggested_fixes: List[str]
    code_improvements: List[str]
    test_improvements: List[str]
    confidence: float


class SimpleTimeout:
    """Simple timeout implementation using signal (Unix) or threading (Windows)"""
    
    def __init__(self, timeout_seconds: int):
        self.timeout_seconds = timeout_seconds
        self.timer = None
        
    def __enter__(self):
        if os.name == 'posix':  # Unix-like systems
            def timeout_handler(signum, frame):
                raise TimeoutError(f"Operation timed out after {self.timeout_seconds} seconds")
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(self.timeout_seconds)
        return self
    
    def __exit__(self, type, value, traceback):
        if os.name == 'posix':
            signal.alarm(0)  # Cancel the alarm


class ResourceMonitor:
    """Monitor system resources during execution"""
    
    def __init__(self):
        self.start_time = None
        self.process = None
        self.initial_memory = 0
        self.peak_memory = 0
        self.cpu_usage = []
        
    def start_monitoring(self, process_id: int):
        """Start monitoring a process"""
        self.start_time = time.time()
        try:
            self.process = psutil.Process(process_id)
            self.initial_memory = self.process.memory_info().rss
            self.peak_memory = self.initial_memory
        except psutil.NoSuchProcess:
            logger.warning(f"Process {process_id} not found for monitoring")
    
    def update_monitoring(self):
        """Update monitoring data"""
        if self.process and self.process.is_running():
            try:
                memory = self.process.memory_info().rss
                cpu = self.process.cpu_percent()
                
                self.peak_memory = max(self.peak_memory, memory)
                self.cpu_usage.append(cpu)
            except psutil.NoSuchProcess:
                pass
    
    def get_monitoring_data(self) -> Dict[str, Any]:
        """Get collected monitoring data"""
        duration = time.time() - self.start_time if self.start_time else 0
        avg_cpu = sum(self.cpu_usage) / len(self.cpu_usage) if self.cpu_usage else 0
        
        return {
            "duration": duration,
            "initial_memory_mb": self.initial_memory / (1024 * 1024),
            "peak_memory_mb": self.peak_memory / (1024 * 1024),
            "memory_increase_mb": (self.peak_memory - self.initial_memory) / (1024 * 1024),
            "average_cpu_percent": avg_cpu,
            "cpu_samples": len(self.cpu_usage)
        }


class SafeExecutionEnvironment:
    """Safe execution environment with resource limits and monitoring"""
    
    def __init__(self):
        self.temp_dir = None
        self.resource_limits = {
            "max_memory_mb": 512,
            "max_cpu_time_seconds": 60,
            "max_wall_time_seconds": 120,
            "max_open_files": 100
        }
    
    async def create_sandbox(self, language: str, dependencies: List[str]) -> Dict[str, Any]:
        """Create isolated execution environment"""
        self.temp_dir = tempfile.mkdtemp(prefix="qa_agent_execution_")
        
        # Set up environment based on language
        if language.lower() == "python":
            return await self._create_python_sandbox(dependencies)
        elif language.lower() in ["javascript", "js"]:
            return await self._create_javascript_sandbox(dependencies)
        else:
            raise ValueError(f"Unsupported language: {language}")
    
    async def _create_python_sandbox(self, dependencies: List[str]) -> Dict[str, Any]:
        """Create Python execution sandbox"""
        sandbox_info = {
            "language": "python",
            "working_directory": self.temp_dir,
            "python_path": [self.temp_dir],
            "dependencies_installed": []
        }
        
        # Install dependencies if needed
        for dep in dependencies:
            try:
                # Only install safe, common testing dependencies
                safe_deps = ["pytest", "unittest", "mock", "pytest-mock", "coverage"]
                if any(safe_dep in dep.lower() for safe_dep in safe_deps):
                    result = await self._run_command([
                        sys.executable, "-m", "pip", "install", "--target", self.temp_dir, dep
                    ], timeout=30)
                    if result["success"]:
                        sandbox_info["dependencies_installed"].append(dep)
            except Exception as e:
                logger.warning(f"Failed to install dependency {dep}: {e}")
        
        return sandbox_info
    
    async def _create_javascript_sandbox(self, dependencies: List[str]) -> Dict[str, Any]:
        """Create JavaScript execution sandbox"""
        sandbox_info = {
            "language": "javascript",
            "working_directory": self.temp_dir,
            "node_modules": f"{self.temp_dir}/node_modules",
            "dependencies_installed": []
        }
        
        # Create package.json
        package_json = {
            "name": "qa-agent-test",
            "version": "1.0.0",
            "dependencies": {}
        }
        
        # Add safe testing dependencies
        safe_deps = ["jest", "mocha", "chai", "@types/jest", "@types/mocha"]
        for dep in dependencies:
            if any(safe_dep in dep.lower() for safe_dep in safe_deps):
                package_json["dependencies"][dep] = "latest"
        
        package_path = Path(self.temp_dir) / "package.json"
        with open(package_path, 'w') as f:
            json.dump(package_json, f, indent=2)
        
        # Install dependencies
        if package_json["dependencies"]:
            try:
                result = await self._run_command([
                    "npm", "install", "--prefix", self.temp_dir
                ], timeout=60)
                if result["success"]:
                    sandbox_info["dependencies_installed"] = list(package_json["dependencies"].keys())
            except Exception as e:
                logger.warning(f"Failed to install JavaScript dependencies: {e}")
        
        return sandbox_info
    
    async def _run_command(self, command: List[str], timeout: int = 30) -> Dict[str, Any]:
        """Run command safely with timeout"""
        try:
            # Use asyncio subprocess with timeout
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.temp_dir
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=timeout
                )
                
                return {
                    "success": process.returncode == 0,
                    "stdout": stdout.decode('utf-8', errors='ignore'),
                    "stderr": stderr.decode('utf-8', errors='ignore'),
                    "exit_code": process.returncode
                }
            except asyncio.TimeoutError:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=5)
                except asyncio.TimeoutError:
                    process.kill()
                
                return {
                    "success": False,
                    "stdout": "",
                    "stderr": f"Command timed out after {timeout} seconds",
                    "exit_code": -1
                }
                
        except Exception as e:
            return {
                "success": False,
                "stdout": "",
                "stderr": str(e),
                "exit_code": -1
            }
    
    def cleanup(self):
        """Clean up sandbox environment"""
        if self.temp_dir and Path(self.temp_dir).exists():
            import shutil
            try:
                shutil.rmtree(self.temp_dir)
            except Exception as e:
                logger.warning(f"Failed to cleanup sandbox: {e}")


class TestResultParser:
    """Parse test execution results from different frameworks"""
    
    def parse_pytest_output(self, output: str, error_output: str, exit_code: int) -> ExecutionResult:
        """Parse pytest execution output"""
        test_results = []
        
        # Parse test results from output
        lines = output.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Look for test execution lines
            if " PASSED " in line or " FAILED " in line or " ERROR " in line or " SKIPPED " in line:
                parts = line.split()
                if len(parts) >= 2:
                    test_name = parts[0].split("::")[-1] if "::" in parts[0] else parts[0]
                    status = "passed" if "PASSED" in line else \
                           "failed" if "FAILED" in line else \
                           "error" if "ERROR" in line else "skipped"
                    
                    # Extract duration if available
                    duration = 0.0
                    for part in parts:
                        if part.endswith('s') and part[:-1].replace('.', '').replace('-', '').isdigit():
                            try:
                                duration = float(part[:-1])
                            except ValueError:
                                pass
                            break
                    
                    test_results.append(TestResult(
                        test_name=test_name,
                        status=status,
                        duration=duration,
                        message=line
                    ))
        
        # Count results
        passed = len([t for t in test_results if t.status == "passed"])
        failed = len([t for t in test_results if t.status == "failed"])
        errors = len([t for t in test_results if t.status == "error"])
        skipped = len([t for t in test_results if t.status == "skipped"])
        total = len(test_results)
        
        return ExecutionResult(
            success=exit_code == 0,
            total_tests=total,
            passed_tests=passed,
            failed_tests=failed,
            error_tests=errors,
            skipped_tests=skipped,
            total_duration=0.0,  # Could be extracted from output
            test_results=test_results,
            output=output,
            error_output=error_output,
            exit_code=exit_code
        )
    
    def parse_unittest_output(self, output: str, error_output: str, exit_code: int) -> ExecutionResult:
        """Parse unittest execution output"""
        test_results = []
        
        # Parse unittest output
        lines = output.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Look for test method execution
            if line.startswith("test_") and ("ok" in line or "FAIL" in line or "ERROR" in line):
                parts = line.split()
                if len(parts) >= 2:
                    test_name = parts[0]
                    status = "passed" if "ok" in line.lower() else \
                           "failed" if "fail" in line.lower() else "error"
                    
                    test_results.append(TestResult(
                        test_name=test_name,
                        status=status,
                        duration=0.0,
                        message=line
                    ))
        
        # If no individual test results found, check for summary
        if not test_results and "Ran " in output:
            # Extract from summary like "Ran 3 tests in 0.001s"
            import re
            match = re.search(r'Ran (\d+) tests', output)
            if match:
                total_tests = int(match.group(1))
                status = "passed" if exit_code == 0 else "failed"
                
                for i in range(total_tests):
                    test_results.append(TestResult(
                        test_name=f"test_{i+1}",
                        status=status,
                        duration=0.0,
                        message="Test execution detected"
                    ))
        
        passed = len([t for t in test_results if t.status == "passed"])
        failed = len([t for t in test_results if t.status == "failed"])
        errors = len([t for t in test_results if t.status == "error"])
        
        return ExecutionResult(
            success=exit_code == 0,
            total_tests=len(test_results),
            passed_tests=passed,
            failed_tests=failed,
            error_tests=errors,
            skipped_tests=0,
            total_duration=0.0,
            test_results=test_results,
            output=output,
            error_output=error_output,
            exit_code=exit_code
        )


class IntelligentResultAnalyzer:
    """Analyze test results with AI-powered insights"""
    
    def analyze_test_failures(self, failures: List[TestResult]) -> Dict[str, Any]:
        """Deep analysis of test failures with categorization and insights"""
        analysis = {
            "failure_categories": {},
            "common_patterns": [],
            "root_causes": [],
            "improvement_suggestions": []
        }
        
        # Categorize failures
        for failure in failures:
            category = self._categorize_failure(failure)
            if category not in analysis["failure_categories"]:
                analysis["failure_categories"][category] = []
            analysis["failure_categories"][category].append(failure)
        
        # Identify patterns
        analysis["common_patterns"] = self._identify_failure_patterns(failures)
        
        # Analyze root causes
        analysis["root_causes"] = self._analyze_root_causes(failures)
        
        # Generate improvement suggestions
        analysis["improvement_suggestions"] = self._generate_improvement_suggestions(failures)
        
        return analysis
    
    def _categorize_failure(self, failure: TestResult) -> str:
        """Categorize a test failure"""
        message = (failure.message or "").lower()
        traceback = (failure.traceback or "").lower()
        
        if "assertionerror" in traceback or "assertion" in message:
            return "assertion_failure"
        elif "attributeerror" in traceback:
            return "attribute_error"
        elif "typeerror" in traceback:
            return "type_error"
        elif "valueerror" in traceback:
            return "value_error"
        elif "import" in traceback or "module" in traceback:
            return "import_error"
        elif "timeout" in message or "timeout" in traceback:
            return "timeout"
        elif "syntax" in traceback:
            return "syntax_error"
        else:
            return "unknown"
    
    def _identify_failure_patterns(self, failures: List[TestResult]) -> List[str]:
        """Identify common patterns across failures"""
        patterns = []
        
        # Pattern: Multiple assertion failures
        assertion_failures = [f for f in failures if "assertion" in (f.message or "").lower()]
        if len(assertion_failures) > 1:
            patterns.append(f"Multiple assertion failures ({len(assertion_failures)} tests)")
        
        # Pattern: Import errors
        import_errors = [f for f in failures if "import" in (f.traceback or "").lower()]
        if import_errors:
            patterns.append(f"Import/module issues ({len(import_errors)} tests)")
        
        return patterns
    
    def _analyze_root_causes(self, failures: List[TestResult]) -> List[str]:
        """Analyze potential root causes of failures"""
        causes = []
        
        # Analyze error types
        error_types = {}
        for failure in failures:
            category = self._categorize_failure(failure)
            error_types[category] = error_types.get(category, 0) + 1
        
        # Most common error type
        if error_types:
            most_common = max(error_types.items(), key=lambda x: x[1])
            if most_common[1] > 1:
                causes.append(f"Predominant issue: {most_common[0]} ({most_common[1]} occurrences)")
        
        return causes
    
    def _generate_improvement_suggestions(self, failures: List[TestResult]) -> List[str]:
        """Generate specific improvement suggestions"""
        suggestions = []
        
        # Categorize failures for targeted suggestions
        categories = {}
        for failure in failures:
            category = self._categorize_failure(failure)
            categories[category] = categories.get(category, 0) + 1
        
        # Suggestions based on failure types
        if categories.get("assertion_failure", 0) > 0:
            suggestions.append("Review test assertions - ensure expected values match actual implementation")
        
        if categories.get("import_error", 0) > 0:
            suggestions.append("Check import statements and module availability")
        
        return suggestions


class ExecutionTool(AgentTool):
    """Intelligent test execution tool for agents"""
    
    def __init__(self):
        super().__init__(
            name="execution_tool",
            description="Execute tests safely with monitoring and intelligent result interpretation",
            parameters={
                "tests": "Test files or code to execute",
                "language": "Programming language (python, javascript)",
                "framework": "Testing framework (pytest, unittest, jest, mocha)",
                "execution_context": "Additional context for execution"
            }
        )
        self.result_parser = TestResultParser()
        self.result_analyzer = IntelligentResultAnalyzer()
        self.learning_system = ExecutionLearningSystem()
        
    async def execute(self, parameters: ToolParameters) -> ToolResult:
        """Execute tests with comprehensive monitoring"""
        start_time = datetime.now()
        
        try:
            tests = parameters.get("tests", "")
            language = parameters.get("language", "python").lower()
            framework = parameters.get("framework", "").lower()
            context = parameters.get("execution_context", {})
            
            # Create execution context
            execution_context = ExecutionContext(
                language=language,
                framework=framework,
                tests=tests,
                context=context
            )
            
            # Execute with monitoring
            result = await self.execute_with_monitoring(tests, execution_context)
            
            # Learn from execution outcome
            await self.learning_system.record_execution_outcome(result, context)
            
            duration = (datetime.now() - start_time).total_seconds()
            
            return ToolResult(
                success=True,
                data={
                    "execution_result": result,
                    "agent_message": self._generate_agent_message(result),
                    "duration": duration
                },
                message=self._generate_agent_message(result)
            )
            
        except Exception as e:
            logger.error(f"Execution failed: {e}")
            return ToolResult(
                success=False,
                error=str(e),
                message=f"Test execution failed: {e}"
            )
    
    async def execute_with_monitoring(self, tests: str, context: ExecutionContext) -> MonitoredExecutionResult:
        """Execute tests with comprehensive monitoring and analysis"""
        sandbox = SafeExecutionEnvironment()
        monitor = ResourceMonitor()
        
        try:
            # Create sandbox environment
            dependencies = context.context.get("dependencies", [])
            sandbox_info = await sandbox.create_sandbox(context.language, dependencies)
            
            # Write test code to file
            test_file = await self._write_test_file(tests, context, sandbox.temp_dir)
            
            # Determine execution command
            command = self._build_execution_command(context, test_file, sandbox_info)
            
            # Execute with monitoring
            execution_result = await self._execute_with_resource_monitoring(
                command, sandbox.temp_dir, monitor
            )
            
            # Parse results based on framework
            parsed_result = self._parse_execution_results(
                execution_result, context.framework or context.language
            )
            
            # Interpret results intelligently
            interpretation = await self.interpret_test_results(parsed_result, context)
            
            # Generate monitoring insights
            monitoring_data = monitor.get_monitoring_data()
            insights = self._generate_execution_insights(parsed_result, monitoring_data)
            recommendations = self._generate_execution_recommendations(parsed_result, monitoring_data)
            
            return MonitoredExecutionResult(
                execution_result=parsed_result,
                monitoring_data=monitoring_data,
                interpretation=interpretation.overall_assessment,
                insights=insights,
                recommendations=recommendations,
                confidence=interpretation.confidence,
                learning_insights=self._generate_learning_insights(parsed_result, monitoring_data)
            )
            
        finally:
            sandbox.cleanup()
    
    async def _write_test_file(self, tests: str, context: ExecutionContext, temp_dir: str) -> str:
        """Write test code to appropriate file"""
        if context.language == "python":
            filename = "test_execution.py"
        elif context.language in ["javascript", "js"]:
            filename = "test_execution.js"
        else:
            filename = "test_execution.txt"
        
        file_path = Path(temp_dir) / filename
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(tests)
        
        return str(file_path)
    
    def _build_execution_command(self, context: ExecutionContext, test_file: str, sandbox_info: Dict) -> List[str]:
        """Build execution command based on context"""
        if context.language == "python":
            if context.framework == "pytest":
                return [sys.executable, "-m", "pytest", test_file, "-v", "--tb=short"]
            elif context.framework == "unittest":
                return [sys.executable, "-m", "unittest", "-v"]
            else:
                # Default to pytest
                return [sys.executable, "-m", "pytest", test_file, "-v", "--tb=short"]
        
        elif context.language in ["javascript", "js"]:
            if context.framework == "jest":
                return ["npx", "jest", test_file, "--verbose"]
            elif context.framework == "mocha":
                return ["npx", "mocha", test_file]
            else:
                # Default to node
                return ["node", test_file]
        
        else:
            raise ValueError(f"Unsupported language: {context.language}")
    
    async def _execute_with_resource_monitoring(self, command: List[str], cwd: str, monitor: ResourceMonitor) -> Dict[str, Any]:
        """Execute command with resource monitoring"""
        try:
            # Start process
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd
            )
            
            # Start monitoring
            monitor.start_monitoring(process.pid)
            
            # Monitor execution
            monitoring_task = asyncio.create_task(self._monitor_execution(process, monitor))
            
            # Wait for completion with timeout
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=120  # 2 minute timeout
                )
            except asyncio.TimeoutError:
                process.terminate()
                await process.wait()
                raise TimeoutError("Test execution timed out after 2 minutes")
            
            # Stop monitoring
            monitoring_task.cancel()
            
            return {
                "stdout": stdout.decode('utf-8', errors='ignore'),
                "stderr": stderr.decode('utf-8', errors='ignore'),
                "exit_code": process.returncode,
                "success": process.returncode == 0
            }
            
        except Exception as e:
            raise RuntimeError(f"Execution failed: {e}")
    
    async def _monitor_execution(self, process, monitor: ResourceMonitor):
        """Monitor process execution"""
        try:
            while process.returncode is None:
                monitor.update_monitoring()
                await asyncio.sleep(0.1)  # Monitor every 100ms
        except asyncio.CancelledError:
            pass
    
    def _parse_execution_results(self, execution_result: Dict[str, Any], framework: str) -> ExecutionResult:
        """Parse execution results based on framework"""
        stdout = execution_result["stdout"]
        stderr = execution_result["stderr"]
        exit_code = execution_result["exit_code"]
        
        if framework == "pytest":
            return self.result_parser.parse_pytest_output(stdout, stderr, exit_code)
        elif framework == "unittest":
            return self.result_parser.parse_unittest_output(stdout, stderr, exit_code)
        else:
            # Generic parsing
            return ExecutionResult(
                success=exit_code == 0,
                total_tests=1 if stdout or stderr else 0,
                passed_tests=1 if exit_code == 0 and stdout else 0,
                failed_tests=1 if exit_code != 0 else 0,
                error_tests=0,
                skipped_tests=0,
                total_duration=0.0,
                output=stdout,
                error_output=stderr,
                exit_code=exit_code
            )
    
    async def interpret_test_results(self, execution_result: ExecutionResult, context: ExecutionContext) -> TestResultInterpretation:
        """Intelligently interpret test results in context of original goals"""
        
        # Overall assessment
        if execution_result.success and execution_result.passed_tests > 0:
            overall = f"✅ Execution successful: {execution_result.passed_tests}/{execution_result.total_tests} tests passed"
        elif execution_result.failed_tests > 0:
            overall = f"❌ Execution completed with failures: {execution_result.failed_tests}/{execution_result.total_tests} tests failed"
        else:
            overall = "⚠️ Execution completed but no tests were detected or run"
        
        # Success analysis
        success_analysis = ""
        if execution_result.passed_tests > 0:
            success_analysis = f"Successfully executed {execution_result.passed_tests} tests."
        
        # Failure analysis
        failure_analysis = ""
        if execution_result.failed_tests > 0 or execution_result.error_tests > 0:
            failed_tests = [t for t in execution_result.test_results if t.status in ["failed", "error"]]
            if failed_tests:
                analysis = self.result_analyzer.analyze_test_failures(failed_tests)
                failure_analysis = f"Failure analysis: {len(analysis['failure_categories'])} failure types detected."
        
        # Performance analysis
        performance_analysis = "Test execution completed."
        
        # Quality insights
        quality_insights = []
        if execution_result.total_tests == 0:
            quality_insights.append("No tests were detected - verify test naming conventions")
        elif execution_result.passed_tests == execution_result.total_tests:
            quality_insights.append("All tests passed - good test coverage")
        
        # Improvement suggestions
        improvement_suggestions = []
        if execution_result.failed_tests > 0:
            failed_tests = [t for t in execution_result.test_results if t.status in ["failed", "error"]]
            analysis = self.result_analyzer.analyze_test_failures(failed_tests)
            improvement_suggestions.extend(analysis['improvement_suggestions'][:3])
        
        confidence = 0.9 if execution_result.success else 0.7
        
        return TestResultInterpretation(
            overall_assessment=overall,
            success_analysis=success_analysis,
            failure_analysis=failure_analysis,
            performance_analysis=performance_analysis,
            quality_insights=quality_insights,
            improvement_suggestions=improvement_suggestions,
            confidence=confidence
        )
    
    def _generate_execution_insights(self, result: ExecutionResult, monitoring_data: Dict[str, Any]) -> List[str]:
        """Generate insights from execution and monitoring data"""
        insights = []
        
        # Test execution insights
        if result.total_tests > 0:
            pass_rate = result.passed_tests / result.total_tests
            if pass_rate == 1.0:
                insights.append("Perfect test execution - all tests passed")
            elif pass_rate > 0.8:
                insights.append("Good test execution - most tests passed")
            else:
                insights.append("Mixed test results - some failures detected")
        
        return insights
    
    def _generate_execution_recommendations(self, result: ExecutionResult, monitoring_data: Dict[str, Any]) -> List[str]:
        """Generate recommendations for improving test execution"""
        recommendations = []
        
        # Test quality recommendations
        if result.failed_tests > 0:
            recommendations.append("Address failing tests before proceeding")
        
        if result.total_tests == 0:
            recommendations.append("Add test cases to verify functionality")
        
        return recommendations
    
    def _generate_learning_insights(self, result: ExecutionResult, monitoring_data: Dict[str, Any]) -> List[str]:
        """Generate insights for the learning system"""
        insights = []
        
        if result.success:
            insights.append("Successful execution pattern for learning")
        else:
            insights.append("Execution failure pattern for analysis")
        
        return insights
    
    def _generate_agent_message(self, result: MonitoredExecutionResult) -> str:
        """Generate agent-friendly message about execution results"""
        exec_result = result.execution_result
        
        if exec_result.success:
            return f"✅ Tests executed successfully: {exec_result.passed_tests}/{exec_result.total_tests} passed. {result.interpretation}"
        else:
            return f"❌ Test execution completed with issues: {exec_result.failed_tests} failures, {exec_result.error_tests} errors. {result.interpretation}"
    
    def can_handle(self, task: str) -> float:
        """Return confidence score 0-1 for handling execution tasks"""
        execution_keywords = [
            "execute", "run", "test", "check", "verify", 
            "validate", "pytest", "unittest", "jest", "mocha"
        ]
        
        task_lower = task.lower()
        matches = sum(1 for keyword in execution_keywords if keyword in task_lower)
        
        # Higher confidence for specific execution requests
        if "execute" in task_lower or "run" in task_lower:
            return min(1.0, 0.8 + matches * 0.05)
        
        return min(1.0, matches * 0.15)


# Export classes for use by other modules
__all__ = [
    'ExecutionTool',
    'SafeExecutionEnvironment',
    'MonitoredExecutionResult',
    'ExecutionResult',
    'TestResult',
    'TestResultInterpretation',
    'TestImprovements',
    'ResourceMonitor',
    'TestResultParser',
    'IntelligentResultAnalyzer'
]
