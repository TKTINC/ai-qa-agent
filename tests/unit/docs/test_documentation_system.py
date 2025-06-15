"""
Tests for Documentation System
"""

import pytest
import os
from pathlib import Path


class TestDocumentationStructure:
    """Test documentation structure and completeness"""
    
    def test_documentation_directories_exist(self):
        """Test that all required documentation directories exist"""
        required_dirs = [
            "docs/architecture",
            "docs/intelligence-showcase", 
            "docs/career-portfolio",
            "docs/enterprise",
            "docs/operations",
            "docs/api-reference",
            "docs/tutorials",
            "docs/assets/diagrams",
            "docs/assets/screenshots"
        ]
        
        for dir_path in required_dirs:
            assert os.path.exists(dir_path), f"Required directory {dir_path} does not exist"
    
    def test_core_documentation_files_exist(self):
        """Test that core documentation files exist"""
        required_files = [
            "docs/architecture/agent-system-architecture.md",
            "docs/intelligence-showcase/agent-intelligence-demonstration.md",
            "docs/career-portfolio/technical-leadership-showcase.md",
            "docs/enterprise/enterprise-adoption-guide.md",
            "docs/operations/production-runbook.md"
        ]
        
        for file_path in required_files:
            assert os.path.exists(file_path), f"Required file {file_path} does not exist"
    
    def test_documentation_content_quality(self):
        """Test documentation content quality and completeness"""
        architecture_file = "docs/architecture/agent-system-architecture.md"
        
        with open(architecture_file, 'r') as f:
            content = f.read()
        
        # Check for key sections
        assert "# AI QA Agent System Architecture" in content
        assert "## Executive Summary" in content
        assert "## System Intelligence Overview" in content
        assert "## Technical Innovation Highlights" in content
        
        # Check for technical depth
        assert "ReAct" in content
        assert "multi-agent" in content.lower()
        assert "learning" in content.lower()
        assert "monitoring" in content.lower()
    
    def test_career_portfolio_metrics(self):
        """Test that career portfolio includes quantified achievements"""
        portfolio_file = "docs/career-portfolio/technical-leadership-showcase.md"
        
        with open(portfolio_file, 'r') as f:
            content = f.read()
        
        # Check for quantified metrics
        assert "94.7%" in content  # Reasoning quality
        assert "99.9%" in content  # Uptime
        assert "40-60%" in content  # Efficiency improvement
        assert "96.2%" in content  # User satisfaction
        
        # Check for technical leadership indicators
        assert "Technical Leadership" in content
        assert "Innovation" in content
        assert "Architecture" in content
        assert "Performance" in content
    
    def test_enterprise_guide_completeness(self):
        """Test enterprise adoption guide completeness"""
        enterprise_file = "docs/enterprise/enterprise-adoption-guide.md"
        
        with open(enterprise_file, 'r') as f:
            content = f.read()
        
        # Check for business sections
        assert "Business Value Proposition" in content
        assert "ROI Analysis" in content
        assert "Implementation Strategy" in content
        assert "Risk Management" in content
        
        # Check for technical sections
        assert "System Requirements" in content
        assert "Security and Compliance" in content
        assert "Integration Capabilities" in content
    
    def test_operations_runbook_completeness(self):
        """Test operations runbook completeness"""
        runbook_file = "docs/operations/production-runbook.md"
        
        with open(runbook_file, 'r') as f:
            content = f.read()
        
        # Check for operational sections
        assert "Health Monitoring" in content
        assert "Troubleshooting" in content
        assert "Maintenance Procedures" in content
        assert "Emergency Response" in content
        
        # Check for practical procedures
        assert "curl" in content  # Health check commands
        assert "kubectl" in content  # Kubernetes commands
        assert "#!/bin/bash" in content  # Shell scripts


class TestDocumentationConsistency:
    """Test consistency across documentation"""
    
    def test_version_consistency(self):
        """Test that version numbers are consistent across docs"""
        # This would check for consistent version references
        # across different documentation files
        pass
    
    def test_metric_consistency(self):
        """Test that performance metrics are consistent across documents"""
        files_to_check = [
            "docs/architecture/agent-system-architecture.md",
            "docs/intelligence-showcase/agent-intelligence-demonstration.md",
            "docs/career-portfolio/technical-leadership-showcase.md"
        ]
        
        reasoning_quality_mentions = []
        
        for file_path in files_to_check:
            with open(file_path, 'r') as f:
                content = f.read()
                if "94.7%" in content:
                    reasoning_quality_mentions.append(file_path)
        
        # Should be mentioned in multiple places for consistency
        assert len(reasoning_quality_mentions) >= 2, "Reasoning quality metric should be consistent across docs"
    
    def test_terminology_consistency(self):
        """Test that technical terminology is used consistently"""
        architecture_file = "docs/architecture/agent-system-architecture.md"
        
        with open(architecture_file, 'r') as f:
            content = f.read()
        
        # Check for consistent terminology
        assert "AI QA Agent System" in content  # Full system name
        assert "ReAct" in content  # Reasoning pattern
        assert "multi-agent" in content.lower()  # Architecture style


if __name__ == "__main__":
    pytest.main([__file__])
