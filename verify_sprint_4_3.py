#!/usr/bin/env python3
"""
Verification script for Sprint 4.3: Compelling Demos & Agent Showcase
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

async def verify_scenario_engine():
    """Verify demo scenario engine functionality"""
    try:
        from src.web.demos.scenario_engine import DemoOrchestrator, ScenarioLibrary, DemoType
        
        # Test scenario library
        library = ScenarioLibrary()
        scenarios = library.list_scenarios()
        assert len(scenarios) >= 4
        
        # Test demo orchestrator
        orchestrator = DemoOrchestrator()
        session_id = "test_verification"
        
        demo_execution = await orchestrator.start_demo(
            DemoType.LEGACY_RESCUE.value,
            "technical", 
            session_id
        )
        assert demo_execution is not None
        
        print("✅ Demo scenario engine verification passed")
        return True
        
    except Exception as e:
        print(f"❌ Demo scenario engine verification failed: {e}")
        return False

async def verify_interactive_platform():
    """Verify interactive demo platform functionality"""
    try:
        from src.web.demos.interactive.demo_platform import InteractiveDemoManager
        
        # Test platform initialization
        platform = InteractiveDemoManager()
        assert platform is not None
        
        # Test session creation
        session_id = await platform.create_demo_session({
            "demo_type": "legacy_rescue",
            "audience_type": "technical"
        })
        assert session_id is not None
        
        print("✅ Interactive demo platform verification passed")
        return True
        
    except Exception as e:
        print(f"❌ Interactive demo platform verification failed: {e}")
        return False

async def verify_demo_routes():
    """Verify demo routes functionality"""
    try:
        from src.web.routes.demo_routes import router
        
        # Test that routes are properly configured
        assert router is not None
        
        print("✅ Demo routes verification passed")
        return True
        
    except Exception as e:
        print(f"❌ Demo routes verification failed: {e}")
        return False

def main():
    """Main verification function"""
    print("🚀 Verifying Sprint 4.3: Compelling Demos & Agent Showcase")
    print("=" * 65)
    
    # Check file existence
    required_files = [
        "src/web/demos/scenario_engine.py",
        "src/web/demos/interactive/demo_platform.py",
        "src/web/routes/demo_routes.py",
        "tests/unit/web/demos/test_scenario_engine.py"
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
        "src.web.demos.scenario_engine",
        "src.web.demos.interactive.demo_platform",
        "src.web.routes.demo_routes"
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
            verify_scenario_engine(),
            verify_interactive_platform(),
            verify_demo_routes(),
            return_exceptions=True
        )
        return all(result is True for result in results)
    
    verification_passed = asyncio.run(run_verifications())
    
    if verification_passed:
        print("\n🎉 Sprint 4.3 verification completed successfully!")
        print("\nNext steps:")
        print("1. Run the application: uvicorn src.api.main:app --reload")
        print("2. Visit http://localhost:8000/web/demos/ to see the demo center")
        print("3. Try the Legacy Code Rescue Mission demo")
        print("4. Explore the Interactive Demo Platform")
        print("5. Experience real-time agent collaboration")
        print("6. Ready for Sprint 5: Production Deployment!")
        return True
    else:
        print("\n❌ Sprint 4.3 verification failed!")
        print("Please check the errors above and fix them before proceeding.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
