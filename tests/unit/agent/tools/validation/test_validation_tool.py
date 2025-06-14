"""
Working tests for Agent Validation Tool
"""

import pytest
import asyncio
import sys
import os

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, '..', '..', '..', '..', 'src')
sys.path.insert(0, src_path)

def test_validation_tool_import():
    """Test that validation tool can be imported"""
    try:
        from agent.tools.validation_tool import ValidationTool
        tool = ValidationTool()
        assert tool.name == "validation_tool"
        print("✅ ValidationTool import and creation successful")
    except Exception as e:
        pytest.fail(f"Failed to import or create ValidationTool: {e}")

@pytest.mark.asyncio
async def test_basic_validation():
    """Test basic validation functionality"""
    from agent.tools.validation_tool import ValidationTool
    
    tool = ValidationTool()
    
    # Test simple valid code
    result = await tool.execute({
        "code": "def test_example():\n    assert True",
        "language": "python",
        "validation_type": "all"
    })
    
    assert result.success == True
    assert "validation_result" in result.data
    print("✅ Basic validation test passed")

if __name__ == "__main__":
    pytest.main([__file__])
