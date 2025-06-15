#!/usr/bin/env python3
"""
Verification script for Sprint 4.1: Conversational Agent Interface
"""

import asyncio
import sys
import importlib
from pathlib import Path

def check_file_exists(file_path: str) -> bool:
    """Check if file exists"""
    return Path(file_path).exists()

def check_import(module_name: str) -> bool:
    """Check if module can be imported"""
    try:
        importlib.import_module(module_name)
        return True
    except ImportError as e:
        print(f"Import error for {module_name}: {e}")
        return False

async def verify_agent_chat_interface():
    """Verify agent chat interface functionality"""
    try:
        from src.web.components.agent_chat import AgentChatInterface, ConversationManager
        
        # Test initialization
        chat_interface = AgentChatInterface()
        assert chat_interface is not None
        
        # Test conversation manager
        assert len(chat_interface.conversation_manager.active_agents) == 5
        
        # Test agent selection
        test_message = "Help me with testing and security"
        requires_collab = await chat_interface._requires_collaboration(test_message)
        assert isinstance(requires_collab, bool)
        
        if requires_collab:
            agents = await chat_interface._select_agents_for_collaboration(test_message)
            assert isinstance(agents, list)
            assert len(agents) > 0
        
        print("✅ Agent chat interface verification passed")
        return True
        
    except Exception as e:
        print(f"❌ Agent chat interface verification failed: {e}")
        return False

async def verify_visualization_service():
    """Verify agent visualization service"""
    try:
        from src.web.services.agent_visualization import AgentVisualizationService
        
        service = AgentVisualizationService()
        
        # Test activity recording
        activity_id = await service.record_agent_activity(
            "test_agent", "test_activity", "Test description", 0.9
        )
        assert activity_id is not None
        
        # Test reasoning session
        await service.start_reasoning_session(
            "test_session", "test_agent", "Initial thought"
        )
        assert "test_session" in service.reasoning_sessions
        
        # Test performance summary
        summary = await service.get_agent_performance_summary("test_agent")
        assert isinstance(summary, dict)
        assert "performance_score" in summary
        
        print("✅ Visualization service verification passed")
        return True
        
    except Exception as e:
        print(f"❌ Visualization service verification failed: {e}")
        return False

async def verify_web_routes():
    """Verify web routes functionality"""
    try:
        from src.web.routes.agent_interface import router
        from fastapi.testclient import TestClient
        from src.api.main import app
        
        # Test that routes are properly configured
        client = TestClient(app)
        
        # Test main page
        response = client.get("/web/")
        assert response.status_code == 200
        
        print("✅ Web routes verification passed")
        return True
        
    except Exception as e:
        print(f"❌ Web routes verification failed: {e}")
        return False

def main():
    """Main verification function"""
    print("🚀 Verifying Sprint 4.1: Conversational Agent Interface")
    print("=" * 60)
    
    # Check file existence
    required_files = [
        "src/web/components/agent_chat.py",
        "src/web/services/agent_visualization.py",
        "src/web/templates/agent_chat.html",
        "src/web/routes/agent_interface.py",
        "tests/unit/web/components/test_agent_chat.py",
        "tests/integration/web/test_agent_interface_routes.py"
    ]
    
    print("📁 Checking file existence...")
    files_ok = True
    for file_path in required_files:
        if check_file_exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - NOT FOUND")
            files_ok = False
    
    if not files_ok:
        print("\n❌ Some required files are missing!")
        return False
    
    # Check imports
    print("\n📦 Checking imports...")
    required_imports = [
        "src.web.components.agent_chat",
        "src.web.services.agent_visualization",
        "src.web.routes.agent_interface"
    ]
    
    imports_ok = True
    for module in required_imports:
        if check_import(module):
            print(f"✅ {module}")
        else:
            print(f"❌ {module} - IMPORT FAILED")
            imports_ok = False
    
    if not imports_ok:
        print("\n❌ Some imports failed!")
        return False
    
    # Run async verifications
    print("\n🧪 Running functionality tests...")
    async def run_verifications():
        results = await asyncio.gather(
            verify_agent_chat_interface(),
            verify_visualization_service(),
            verify_web_routes(),
            return_exceptions=True
        )
        return all(result is True for result in results)
    
    verification_passed = asyncio.run(run_verifications())
    
    if verification_passed:
        print("\n🎉 Sprint 4.1 verification completed successfully!")
        print("\nNext steps:")
        print("1. Run the application: uvicorn src.api.main:app --reload")
        print("2. Visit http://localhost:8000/web/ to see the agent interface")
        print("3. Test the conversational interface with agent collaboration")
        print("4. Proceed to Sprint 4.2: Agent Intelligence Analytics & Visualization")
        return True
    else:
        print("\n❌ Sprint 4.1 verification failed!")
        print("Please check the errors above and fix them before proceeding.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
