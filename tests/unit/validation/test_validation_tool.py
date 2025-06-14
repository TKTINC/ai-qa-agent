"""
Tests for Standalone Validation Tool
"""

import pytest
import asyncio
import sys
import os

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, '..', '..', '..', 'src')
sys.path.insert(0, src_path)

def test_validation_tool_import():
    """Test that validation tool can be imported"""
    try:
        from validation.validation_tool import StandaloneValidationTool
        tool = StandaloneValidationTool()
        assert tool.name == "validation_tool"
        print("✅ StandaloneValidationTool import and creation successful")
    except Exception as e:
        pytest.fail(f"Failed to import or create StandaloneValidationTool: {e}")

@pytest.mark.asyncio
async def test_basic_validation():
    """Test basic validation functionality"""
    from validation.validation_tool import StandaloneValidationTool
    
    tool = StandaloneValidationTool()
    
    # Test simple valid code
    result = await tool.execute({
        "code": "def test_example():\n    assert True",
        "language": "python",
        "validation_type": "all"
    })
    
    assert result.success == True
    assert "validation_result" in result.data
    print("✅ Basic validation test passed")

@pytest.mark.asyncio
async def test_validation_with_issues():
    """Test validation of code with issues"""
    from validation.validation_tool import StandaloneValidationTool
    
    tool = StandaloneValidationTool()
    
    # Test code with syntax error
    result = await tool.execute({
        "code": "def test_bad(:\n    assert True",  # Missing closing parenthesis
        "language": "python",
        "validation_type": "all"
    })
    
    assert result.success == True  # Tool executed successfully
    validation_result = result.data["validation_result"]
    assert not validation_result.validation_passed  # But validation found issues
    assert len(validation_result.issues_found) > 0
    print("✅ Validation with issues test passed")

if __name__ == "__main__":
    pytest.main([__file__])
