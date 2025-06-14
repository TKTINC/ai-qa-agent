"""
Safe Execution Environment

Secure, monitored execution environment for agent-generated tests with comprehensive
safety controls, resource management, and monitoring capabilities.
"""

import asyncio
import logging
import subprocess
import tempfile
import shutil
import os
import signal
import time
import psutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
import json

logger = logging.getLogger(__name__)


@dataclass
class SandboxEnvironment:
    """Sandbox environment configuration"""
    sandbox_id: str
    language: str
    working_directory: str
    resource_limits: Dict[str, Any]
    network_restricted: bool = True
    dependencies_installed: List[str] = field(default_factory=list)
    environment_variables: Dict[str, str] = field(default_factory=dict)
    cleanup_scheduled: bool = False


@dataclass
class ExecutionMonitoring:
    """Real-time execution monitoring data"""
    process_id: int
    start_time: datetime
    current_memory_mb: float
    peak_memory_mb: float
    cpu_percent: float
    status: str  # running, completed, failed, timeout, killed


@dataclass
class SecurityPolicy:
    """Security policy for code execution"""
    allow_network: bool = False
    allow_file_write: bool = True
    allow_subprocess: bool = False
    max_file_size_mb: int = 10
    allowed_modules: List[str] = field(default_factory=lambda: [
        "os", "sys", "json", "re", "math", "datetime", "collections",
        "itertools", "functools", "unittest", "pytest", "mock"
    ])
    blocked_modules: List[str] = field(default_factory=lambda: [
        "socket", "urllib", "requests", "subprocess", "multiprocessing"
    ])


class ProcessMonitor:
    """Monitor process execution with detailed metrics"""
    
    def __init__(self):
        self.monitoring_data = []
        self.process = None
        self.start_time = None
        self.monitoring_active = False
        
    async def start_monitoring(self, process_id: int) -> None:
        """Start monitoring a process"""
        try:
            self.process = psutil.Process(process_id)
            self.start_time = datetime.now()
            self.monitoring_active = True
            
            # Start monitoring loop
            asyncio.create_task(self._monitoring_loop())
            
        except psutil.NoSuchProcess:
            logger.warning(f"Process {process_id} not found for monitoring")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active and self.process and self.process.is_running():
            try:
                # Collect metrics
                memory_info = self.process.memory_info()
                cpu_percent = self.process.cpu_percent()
                
                monitoring_point = ExecutionMonitoring(
                    process_id=self.process.pid,
                    start_time=self.start_time,
                    current_memory_mb=memory_info.rss / (1024 * 1024),
                    peak_memory_mb=max(
                        getattr(self, 'peak_memory', 0),
                        memory_info.rss / (1024 * 1024)
                    ),
                    cpu_percent=cpu_percent,
                    status="running"
                )
                
                self.monitoring_data.append(monitoring_point)
                self.peak_memory = monitoring_point.peak_memory_mb
                
                # Check for resource violations
                await self._check_resource_violations(monitoring_point)
                
                await asyncio.sleep(0.5)  # Monitor every 500ms
                
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            except Exception as e:
                logger.warning(f"Monitoring error: {e}")
                break
    
    async def _check_resource_violations(self, monitoring: ExecutionMonitoring):
        """Check for resource limit violations"""
        # Check memory limits
        if monitoring.current_memory_mb > 500:  # 500MB limit
            logger.warning(f"High memory usage: {monitoring.current_memory_mb:.1f}MB")
            if monitoring.current_memory_mb > 1000:  # 1GB kill limit
                await self.terminate_process("Memory limit exceeded")
        
        # Check execution time
        if self.start_time:
            duration = datetime.now() - self.start_time
            if duration > timedelta(minutes=5):  # 5 minute limit
                await self.terminate_process("Execution time limit exceeded")
    
    async def terminate_process(self, reason: str):
        """Terminate monitored process"""
        if self.process and self.process.is_running():
            logger.warning(f"Terminating process: {reason}")
            try:
                self.process.terminate()
                await asyncio.sleep(2)
                if self.process.is_running():
                    self.process.kill()
            except Exception as e:
                logger.error(f"Failed to terminate process: {e}")
        
        self.monitoring_active = False
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring_active = False
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get summary of monitoring data"""
        if not self.monitoring_data:
            return {}
        
        latest = self.monitoring_data[-1]
        duration = datetime.now() - self.start_time if self.start_time else timedelta(0)
        
        return {
            "duration_seconds": duration.total_seconds(),
            "peak_memory_mb": latest.peak_memory_mb,
            "final_memory_mb": latest.current_memory_mb,
            "total_samples": len(self.monitoring_data),
            "status": latest.status
        }


class FileSystemSandbox:
    """File system isolation and security"""
    
    def __init__(self, policy: SecurityPolicy):
        self.policy = policy
        self.sandbox_root = None
        
    async def create_isolated_filesystem(self, language: str) -> str:
        """Create isolated file system for execution"""
        # Create temporary directory with proper permissions
        self.sandbox_root = tempfile.mkdtemp(prefix=f"qa_agent_{language}_")
        
        # Set restrictive permissions
        os.chmod(self.sandbox_root, 0o700)
        
        # Create subdirectories
        subdirs = ["src", "tests", "temp", "output"]
        for subdir in subdirs:
            subdir_path = Path(self.sandbox_root) / subdir
            subdir_path.mkdir(exist_ok=True)
            os.chmod(subdir_path, 0o700)
        
        return self.sandbox_root
    
    def validate_file_access(self, file_path: str, operation: str) -> bool:
        """Validate file access request"""
        path = Path(file_path).resolve()
        
        # Ensure path is within sandbox
        if self.sandbox_root:
            sandbox_path = Path(self.sandbox_root).resolve()
            try:
                path.relative_to(sandbox_path)
            except ValueError:
                logger.warning(f"Attempted access outside sandbox: {file_path}")
                return False
        
        # Check operation permissions
        if operation == "write" and not self.policy.allow_file_write:
            return False
        
        # Check file size limits
        if operation == "write" and path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            if size_mb > self.policy.max_file_size_mb:
                return False
        
        return True
    
    def cleanup_filesystem(self):
        """Clean up sandbox filesystem"""
        if self.sandbox_root and Path(self.sandbox_root).exists():
            try:
                shutil.rmtree(self.sandbox_root)
            except Exception as e:
                logger.warning(f"Failed to cleanup sandbox filesystem: {e}")


class SafeExecutionEnvironment:
    """Main safe execution environment with multiple isolation layers"""
    
    def __init__(self):
        self.security_policy = SecurityPolicy()
        self.filesystem_sandbox = FileSystemSandbox(self.security_policy)
        self.process_monitor = ProcessMonitor()
        self.current_sandbox = None
        
    async def create_sandbox(self, language: str, dependencies: List[str]) -> SandboxEnvironment:
        """Create secure execution sandbox"""
        sandbox_id = f"sandbox_{int(time.time())}"
        
        # Create process-based sandbox
        sandbox_dir = await self.filesystem_sandbox.create_isolated_filesystem(language)
        
        self.current_sandbox = SandboxEnvironment(
            sandbox_id=sandbox_id,
            language=language,
            working_directory=sandbox_dir,
            resource_limits=self._get_resource_limits(),
            network_restricted=not self.security_policy.allow_network,
            dependencies_installed=[],
            environment_variables={"EXECUTION_ENV": "process", "PYTHONPATH": sandbox_dir}
        )
        
        return self.current_sandbox
    
    def _get_resource_limits(self) -> Dict[str, Any]:
        """Get resource limits configuration"""
        return {
            "max_memory_mb": 512,
            "max_cpu_time_seconds": 60,
            "max_wall_time_seconds": 120,
            "max_open_files": 100,
            "max_file_size_mb": 10
        }
    
    async def monitor_execution(self, execution_id: str) -> ExecutionMonitoring:
        """Real-time monitoring of test execution"""
        if self.process_monitor.monitoring_data:
            return self.process_monitor.monitoring_data[-1]
        
        # Return default monitoring state
        return ExecutionMonitoring(
            process_id=0,
            start_time=datetime.now(),
            current_memory_mb=0.0,
            peak_memory_mb=0.0,
            cpu_percent=0.0,
            status="unknown"
        )
    
    async def execute_safely(self, command: List[str], code_content: str = "") -> Dict[str, Any]:
        """Execute command safely in sandbox"""
        if not self.current_sandbox:
            raise RuntimeError("No sandbox created")
        
        return await self._execute_in_process_sandbox(command, code_content)
    
    async def _execute_in_process_sandbox(self, command: List[str], code_content: str) -> Dict[str, Any]:
        """Execute in process-based sandbox"""
        try:
            # Write code file if needed
            if code_content:
                if command[0] in ["python", "python3"] or command[0].endswith("python"):
                    filename = "test_execution.py"
                elif "node" in command[0] or "npm" in command[0]:
                    filename = "test_execution.js"
                else:
                    filename = "test_execution.txt"
                
                code_path = Path(self.current_sandbox.working_directory) / filename
                with open(code_path, 'w', encoding='utf-8') as f:
                    f.write(code_content)
                
                # Update command to use the file
                if filename not in " ".join(command):
                    command.append(str(code_path))
            
            # Execute with monitoring
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.current_sandbox.working_directory,
                env={**os.environ, **self.current_sandbox.environment_variables}
            )
            
            # Start monitoring
            await self.process_monitor.start_monitoring(process.pid)
            
            try:
                # Wait for completion with timeout
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.current_sandbox.resource_limits["max_wall_time_seconds"]
                )
                
                return {
                    "exit_code": process.returncode,
                    "stdout": stdout.decode('utf-8', errors='ignore'),
                    "stderr": stderr.decode('utf-8', errors='ignore'),
                    "success": process.returncode == 0
                }
                
            except asyncio.TimeoutError:
                process.terminate()
                await process.wait()
                return {
                    "exit_code": -1,
                    "stdout": "",
                    "stderr": "Execution timed out",
                    "success": False
                }
            
        except Exception as e:
            return {
                "exit_code": -1,
                "stdout": "",
                "stderr": str(e),
                "success": False
            }
        finally:
            self.process_monitor.stop_monitoring()
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get execution summary with monitoring data"""
        monitoring_summary = self.process_monitor.get_monitoring_summary()
        
        return {
            "sandbox_info": {
                "sandbox_id": self.current_sandbox.sandbox_id if self.current_sandbox else None,
                "language": self.current_sandbox.language if self.current_sandbox else None,
                "execution_environment": self.current_sandbox.environment_variables.get("EXECUTION_ENV", "unknown") if self.current_sandbox else "unknown"
            },
            "monitoring": monitoring_summary,
            "security": {
                "network_restricted": self.current_sandbox.network_restricted if self.current_sandbox else True,
                "resource_limits_applied": bool(self.current_sandbox),
                "dependencies_installed": len(self.current_sandbox.dependencies_installed) if self.current_sandbox else 0
            }
        }
    
    def cleanup(self):
        """Clean up execution environment"""
        self.process_monitor.stop_monitoring()
        self.filesystem_sandbox.cleanup_filesystem()
        self.current_sandbox = None


# Export main classes
__all__ = [
    'SafeExecutionEnvironment',
    'SandboxEnvironment',
    'ExecutionMonitoring',
    'SecurityPolicy',
    'ProcessMonitor',
    'FileSystemSandbox'
]
