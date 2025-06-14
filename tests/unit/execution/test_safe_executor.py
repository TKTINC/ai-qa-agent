"""
Tests for Safe Execution Environment

Tests for security, resource management, and monitoring capabilities.
"""

import pytest
import asyncio
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.execution.safe_executor import (
    SafeExecutionEnvironment,
    SecurityPolicy,
    FileSystemSandbox,
    ProcessMonitor,
    SandboxEnvironment
)


class TestSecurityPolicy:
    """Test security policy configuration"""
    
    def test_default_security_policy(self):
        """Test default security policy settings"""
        policy = SecurityPolicy()
        
        assert policy.allow_network == False
        assert policy.allow_file_write == True
        assert policy.allow_subprocess == False
        assert policy.max_file_size_mb == 10
        assert "unittest" in policy.allowed_modules
        assert "socket" in policy.blocked_modules


class TestFileSystemSandbox:
    """Test file system sandbox functionality"""
    
    def setup_method(self):
        self.policy = SecurityPolicy()
        self.sandbox = FileSystemSandbox(self.policy)
    
    @pytest.mark.asyncio
    async def test_create_isolated_filesystem(self):
        """Test creation of isolated file system"""
        try:
            sandbox_path = await self.sandbox.create_isolated_filesystem("python")
            
            assert Path(sandbox_path).exists()
            assert Path(sandbox_path).is_dir()
            
            # Check subdirectories
            subdirs = ["src", "tests", "temp", "output"]
            for subdir in subdirs:
                subdir_path = Path(sandbox_path) / subdir
                assert subdir_path.exists()
                assert subdir_path.is_dir()
                
        finally:
            self.sandbox.cleanup_filesystem()


class TestSafeExecutionEnvironment:
    """Test main safe execution environment"""
    
    def setup_method(self):
        self.env = SafeExecutionEnvironment()
    
    @pytest.mark.asyncio
    async def test_create_sandbox(self):
        """Test sandbox creation"""
        try:
            sandbox = await self.env.create_sandbox("python", [])
            
            assert isinstance(sandbox, SandboxEnvironment)
            assert sandbox.language == "python"
            assert sandbox.sandbox_id is not None
            assert len(sandbox.resource_limits) > 0
            
        finally:
            self.env.cleanup()
    
    def test_resource_limits_configuration(self):
        """Test resource limits configuration"""
        limits = self.env._get_resource_limits()
        
        required_keys = [
            "max_memory_mb", "max_cpu_time_seconds", 
            "max_wall_time_seconds", "max_open_files", "max_file_size_mb"
        ]
        
        for key in required_keys:
            assert key in limits
            assert isinstance(limits[key], (int, float))
            assert limits[key] > 0


if __name__ == "__main__":
    pytest.main([__file__])
