"""
Demo Routes
Handles demo presentation, interactive experiences, and showcase functionality
for the AI agent system demonstrations.
"""

import asyncio
from datetime import datetime
from typing import Dict, Optional, List
import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Request, Query, HTTPException, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from src.web.demos.scenario_engine import demo_scenario_engine, narrative_generator, DemoType, AudienceType
from src.web.demos.interactive.demo_platform import interactive_demo_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/web/demos", tags=["Demos"])

# Templates setup - handle missing templates gracefully
try:
    templates = Jinja2Templates(directory="src/web/templates")
except Exception:
    templates = None

class DemoSessionRequest(BaseModel):
    demo_type: str = "legacy_rescue"
    audience_type: str = "technical"
    audience_count: int = 1
    presentation_mode: bool = False
    customization: Dict[str, Any] = {}

class DemoStepRequest(BaseModel):
    step_number: Optional[int] = None
    audience_input: Optional[str] = None
    speed_multiplier: float = 1.0

@router.get("/", response_class=HTMLResponse)
async def demos_home(request: Request):
    """Render the main demos home page"""
    if templates is None:
        return HTMLResponse("""
        <!DOCTYPE html>
        <html>
        <head><title>AI Agent Demos</title></head>
        <body>
        <h1>🎭 AI Agent Demonstration Center</h1>
        <p>Experience the power of AI agents through compelling demonstrations!</p>
        <h2>Available Demos:</h2>
        <ul>
            <li><a href="/web/demos/legacy-rescue">Legacy Code Rescue Mission</a></li>
            <li><a href="/web/demos/debugging-session">Real-Time Debugging Session</a></li>
            <li><a href="/web/demos/ai-teaching">AI Teaching Assistant</a></li>
            <li><a href="/web/demos/interactive">Interactive Demo Platform</a></li>
        </ul>
        <h2>API Endpoints:</h2>
        <ul>
            <li><a href="/web/demos/api/scenarios">/api/scenarios</a> - List available scenarios</li>
            <li><a href="/web/demos/api/interactive/create">/api/interactive/create</a> - Create demo session</li>
        </ul>
        </body>
        </html>
        """)
    
    return templates.TemplateResponse(
        "demos/home.html",
        {
            "request": request,
            "title": "AI Agent Demonstrations",
            "timestamp": datetime.now().isoformat()
        }
    )

@router.get("/legacy-rescue", response_class=HTMLResponse)
async def legacy_rescue_demo(request: Request):
    """Legacy Code Rescue Mission demo page"""
    return HTMLResponse("""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Legacy Code Rescue Mission - AI Agent Demo</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <style>
            .demo-step { 
                border-left: 4px solid #3b82f6; 
                background: #f8fafc; 
                transition: all 0.3s ease;
            }
            .demo-step:hover { background: #f1f5f9; }
            .agent-message { 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border-radius: 12px;
            }
            .user-message { 
                background: #f3f4f6; 
                border-radius: 12px;
            }
        </style>
    </head>
    <body class="bg-gray-50">
        <div class="container mx-auto px-4 py-8">
            <div class="bg-white rounded-lg shadow-lg p-8">
                <h1 class="text-4xl font-bold text-gray-800 mb-4">🏗️ Legacy Code Rescue Mission</h1>
                <p class="text-xl text-gray-600 mb-8">Watch AI agents tackle a 10,000-line untested legacy codebase with strategic analysis and automated solutions</p>
                
                <div class="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-8">
                    <div class="bg-blue-50 p-6 rounded-lg">
                        <h3 class="text-lg font-semibold mb-2">🎯 Scenario</h3>
                        <p>Inherited legacy codebase with zero tests, no documentation, critical business logic</p>
                    </div>
                    <div class="bg-green-50 p-6 rounded-lg">
                        <h3 class="text-lg font-semibold mb-2">🤖 Agents</h3>
                        <p>Test Architect, Code Reviewer, Security Specialist collaborate on solution</p>
                    </div>
                    <div class="bg-purple-50 p-6 rounded-lg">
                        <h3 class="text-lg font-semibold mb-2">⏱️ Duration</h3>
                        <p>8 minutes of intelligent analysis and strategic planning</p>
                    </div>
                </div>
                
                <div id="demo-content" class="space-y-6">
                    <div class="text-center">
                        <button id="start-demo" class="px-8 py-3 bg-blue-600 text-white rounded-lg text-lg font-semibold hover:bg-blue-700 transition">
                            🚀 Start Demo
                        </button>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            let currentStep = 0;
            const demoSteps = [
                {
                    user: "I inherited this 10,000-line Python codebase with zero tests. The previous developer left no documentation. I need to add comprehensive testing before we can make any changes. Where do I even start?",
                    agent: "I understand this is a challenging situation! Let me help you create a strategic approach to testing this legacy codebase. I'll analyze the code structure, identify critical components, and prioritize testing efforts for maximum impact.",
                    reasoning: "User has a complex legacy code problem requiring strategic planning, risk assessment, and systematic approach. Need to engage Test Architect for strategy and Code Reviewer for quality assessment.",
                    tools: ["repository_analyzer", "complexity_analyzer", "dependency_mapper"],
                    agents: ["Test Architect", "Code Reviewer"]
                },
                {
                    user: "Here's the repository structure. Please analyze it and tell me what you find.",
                    agent: "I've analyzed your codebase and found several key insights:\\n\\n🔍 **Architecture Analysis:**\\n- 3 core modules with high interconnectivity\\n- User authentication system (critical - needs testing first)\\n- Payment processing module (high risk)\\n- 15 utility functions used across multiple modules\\n\\n📊 **Complexity Hotspots:**\\n- `UserManager.authenticate()` - Complexity: 12 (very high)\\n- `PaymentProcessor.process_payment()` - Complexity: 8 (high)\\n- `DataValidator.validate_input()` - Complexity: 6 (medium-high)\\n\\n🎯 **Recommended Testing Priority:**\\n1. Authentication system (security critical)\\n2. Payment processing (business critical)\\n3. Core utility functions (widespread impact)",
                    reasoning: "Test Architect identifies critical paths, Code Reviewer assesses quality issues, analysis tools provide complexity metrics. Prioritizing by risk and business impact.",
                    tools: ["ast_parser", "graph_analyzer", "pattern_detector"],
                    agents: ["Test Architect", "Code Reviewer", "Security Specialist"]
                }
            ];
            
            document.getElementById('start-demo').addEventListener('click', startDemo);
            
            function startDemo() {
                document.getElementById('demo-content').innerHTML = '<div class="space-y-6" id="demo-steps"></div>';
                showNextStep();
            }
            
            function showNextStep() {
                if (currentStep >= demoSteps.length) {
                    showDemoComplete();
                    return;
                }
                
                const step = demoSteps[currentStep];
                const stepHtml = `
                    <div class="demo-step p-6 rounded-lg mb-6">
                        <div class="user-message p-4 mb-4">
                            <div class="font-semibold text-gray-700 mb-2">👤 User:</div>
                            <div class="text-gray-800">${step.user}</div>
                        </div>
                        <div class="agent-message p-4">
                            <div class="font-semibold mb-2">🤖 AI Agents:</div>
                            <div class="whitespace-pre-line">${step.agent}</div>
                            <div class="mt-4 flex flex-wrap gap-2">
                                <span class="text-xs bg-white bg-opacity-20 px-2 py-1 rounded">🧠 Reasoning</span>
                                ${step.tools.map(tool => `<span class="text-xs bg-white bg-opacity-20 px-2 py-1 rounded">🔧 ${tool}</span>`).join('')}
                                ${step.agents.map(agent => `<span class="text-xs bg-white bg-opacity-20 px-2 py-1 rounded">👥 ${agent}</span>`).join('')}
                            </div>
                        </div>
                        <div class="mt-4 text-center">
                            <button onclick="showNextStep()" class="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700">
                                Next Step →
                            </button>
                        </div>
                    </div>
                `;
                
                document.getElementById('demo-steps').innerHTML += stepHtml;
                currentStep++;
                
                // Scroll to new step
                setTimeout(() => {
                    document.querySelector('.demo-step:last-child').scrollIntoView({ behavior: 'smooth' });
                }, 100);
            }
            
            function showDemoComplete() {
                const completeHtml = `
                    <div class="bg-green-50 border border-green-200 p-6 rounded-lg text-center">
                        <h2 class="text-2xl font-bold text-green-800 mb-4">🎉 Demo Complete!</h2>
                        <p class="text-green-700 mb-4">You've seen how AI agents can strategically approach legacy code testing challenges.</p>
                        <div class="flex justify-center space-x-4">
                            <button onclick="location.reload()" class="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700">
                                🔄 Replay Demo
                            </button>
                            <a href="/web/demos/" class="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700">
                                🎭 Try Other Demos
                            </a>
                        </div>
                    </div>
                `;
                document.getElementById('demo-steps').innerHTML += completeHtml;
            }
        </script>
    </body>
    </html>
    """)

@router.get("/interactive", response_class=HTMLResponse)
async def interactive_demo_platform(request: Request):
    """Interactive demo platform page"""
    return HTMLResponse("""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Interactive AI Agent Demo Platform</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <style>
            .demo-card { 
                transition: all 0.3s ease; 
                cursor: pointer;
            }
            .demo-card:hover { 
                transform: translateY(-4px); 
                box-shadow: 0 12px 24px rgba(0,0,0,0.15);
            }
            .audience-selector { 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            }
        </style>
    </head>
    <body class="bg-gray-50">
        <div class="container mx-auto px-4 py-8">
            <div class="bg-white rounded-lg shadow-lg p-8 mb-8">
                <h1 class="text-4xl font-bold text-gray-800 mb-4">🎮 Interactive Demo Platform</h1>
                <p class="text-xl text-gray-600 mb-8">Experience AI agents through hands-on, interactive demonstrations</p>
                
                <!-- Audience Selection -->
                <div class="audience-selector text-white p-6 rounded-lg mb-8">
                    <h2 class="text-2xl font-bold mb-4">👥 Select Your Audience Type</h2>
                    <div class="grid grid-cols-1 md:grid-cols-4 gap-4">
                        <button class="audience-btn p-4 bg-white bg-opacity-20 rounded-lg hover:bg-opacity-30" data-audience="technical">
                            <div class="text-2xl mb-2">🔧</div>
                            <div class="font-semibold">Technical</div>
                            <div class="text-sm">Deep technical details</div>
                        </button>
                        <button class="audience-btn p-4 bg-white bg-opacity-20 rounded-lg hover:bg-opacity-30" data-audience="business">
                            <div class="text-2xl mb-2">💼</div>
                            <div class="font-semibold">Business</div>
                            <div class="text-sm">ROI and efficiency focus</div>
                        </button>
                        <button class="audience-btn p-4 bg-white bg-opacity-20 rounded-lg hover:bg-opacity-30" data-audience="educational">
                            <div class="text-2xl mb-2">🎓</div>
                            <div class="font-semibold">Educational</div>
                            <div class="text-sm">Learning and teaching</div>
                        </button>
                        <button class="audience-btn p-4 bg-white bg-opacity-20 rounded-lg hover:bg-opacity-30" data-audience="executive">
                            <div class="text-2xl mb-2">👔</div>
                            <div class="font-semibold">Executive</div>
                            <div class="text-sm">Strategic overview</div>
                        </button>
                    </div>
                </div>
                
                <!-- Demo Selection -->
                <div id="demo-selection" class="grid grid-cols-1 lg:grid-cols-2 gap-8">
                    <div class="demo-card bg-gradient-to-br from-blue-50 to-blue-100 p-6 rounded-lg border border-blue-200" data-demo="legacy_rescue">
                        <div class="text-4xl mb-4">🏗️</div>
                        <h3 class="text-xl font-bold mb-2">Legacy Code Rescue Mission</h3>
                        <p class="text-gray-600 mb-4">Watch agents tackle a 10,000-line untested legacy codebase</p>
                        <div class="flex justify-between text-sm text-gray-500">
                            <span>⏱️ 8 minutes</span>
                            <span>📊 Intermediate</span>
                        </div>
                    </div>
                    
                    <div class="demo-card bg-gradient-to-br from-red-50 to-red-100 p-6 rounded-lg border border-red-200" data-demo="debugging_session">
                        <div class="text-4xl mb-4">🔍</div>
                        <h3 class="text-xl font-bold mb-2">Real-Time Debugging Session</h3>
                        <p class="text-gray-600 mb-4">See agents collaborate to solve production issues</p>
                        <div class="flex justify-between text-sm text-gray-500">
                            <span>⏱️ 6 minutes</span>
                            <span>📊 Advanced</span>
                        </div>
                    </div>
                    
                    <div class="demo-card bg-gradient-to-br from-green-50 to-green-100 p-6 rounded-lg border border-green-200" data-demo="ai_teaching">
                        <div class="text-4xl mb-4">🎓</div>
                        <h3 class="text-xl font-bold mb-2">AI Teaching Assistant</h3>
                        <p class="text-gray-600 mb-4">Experience personalized learning with agent tutors</p>
                        <div class="flex justify-between text-sm text-gray-500">
                            <span>⏱️ 10 minutes</span>
                            <span>📊 Beginner</span>
                        </div>
                    </div>
                    
                    <div class="demo-card bg-gradient-to-br from-purple-50 to-purple-100 p-6 rounded-lg border border-purple-200" data-demo="custom_exploration">
                        <div class="text-4xl mb-4">⚡</div>
                        <h3 class="text-xl font-bold mb-2">Custom Exploration</h3>
                        <p class="text-gray-600 mb-4">Explore agent capabilities with your own scenarios</p>
                        <div class="flex justify-between text-sm text-gray-500">
                            <span>⏱️ Unlimited</span>
                            <span>📊 Any Level</span>
                        </div>
                    </div>
                </div>
                
                <div id="demo-interface" class="hidden mt-8">
                    <div class="bg-gray-800 text-white p-6 rounded-lg">
                        <div class="flex justify-between items-center mb-4">
                            <h2 id="demo-title" class="text-xl font-bold">Demo Starting...</h2>
                            <button id="end-demo" class="px-4 py-2 bg-red-600 rounded hover:bg-red-700">End Demo</button>
                        </div>
                        <div id="demo-content" class="space-y-4 max-h-96 overflow-y-auto">
                            <!-- Demo content will be inserted here -->
                        </div>
                        <div class="mt-4 flex space-x-4">
                            <button id="next-step" class="px-4 py-2 bg-blue-600 rounded hover:bg-blue-700">Next Step</button>
                            <input id="audience-input" type="text" placeholder="Ask a question or provide input..." class="flex-1 px-3 py-2 bg-gray-700 rounded text-white">
                            <button id="send-input" class="px-4 py-2 bg-green-600 rounded hover:bg-green-700">Send</button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            let selectedAudience = 'technical';
            let selectedDemo = null;
            let currentSession = null;
            
            // Audience selection
            document.querySelectorAll('.audience-btn').forEach(btn => {
                btn.addEventListener('click', () => {
                    document.querySelectorAll('.audience-btn').forEach(b => b.classList.remove('bg-opacity-40'));
                    btn.classList.add('bg-opacity-40');
                    selectedAudience = btn.dataset.audience;
                });
            });
            
            // Demo selection
            document.querySelectorAll('.demo-card').forEach(card => {
                card.addEventListener('click', () => {
                    selectedDemo = card.dataset.demo;
                    startInteractiveDemo();
                });
            });
            
            async function startInteractiveDemo() {
                try {
                    const response = await fetch('/web/demos/api/interactive/create', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            demo_type: selectedDemo,
                            audience_type: selectedAudience,
                            audience_count: 1,
                            presentation_mode: false
                        })
                    });
                    
                    const data = await response.json();
                    if (data.session_id) {
                        currentSession = data.session_id;
                        showDemoInterface(data);
                    }
                } catch (error) {
                    console.error('Error starting demo:', error);
                }
            }
            
            function showDemoInterface(demoData) {
                document.getElementById('demo-selection').classList.add('hidden');
                document.getElementById('demo-interface').classList.remove('hidden');
                document.getElementById('demo-title').textContent = demoData.scenario?.title || 'Interactive Demo';
                
                const introHtml = `
                    <div class="bg-blue-900 p-4 rounded mb-4">
                        <h3 class="font-bold mb-2">📋 Demo Overview</h3>
                        <p>${demoData.scenario?.description || 'Interactive demo experience'}</p>
                        <div class="mt-2 flex flex-wrap gap-2">
                            ${(demoData.scenario?.learning_objectives || []).map(obj => 
                                `<span class="text-xs bg-blue-700 px-2 py-1 rounded">${obj}</span>`
                            ).join('')}
                        </div>
                    </div>
                `;
                
                document.getElementById('demo-content').innerHTML = introHtml;
            }
            
            // Demo controls
            document.getElementById('next-step').addEventListener('click', async () => {
                if (!currentSession) return;
                
                try {
                    const response = await fetch(`/web/demos/api/interactive/${currentSession}/step`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({})
                    });
                    
                    const data = await response.json();
                    if (data.status === 'success') {
                        addDemoStep(data);
                    }
                } catch (error) {
                    console.error('Error executing step:', error);
                }
            });
            
            document.getElementById('send-input').addEventListener('click', async () => {
                const input = document.getElementById('audience-input').value;
                if (!input || !currentSession) return;
                
                try {
                    const response = await fetch(`/web/demos/api/interactive/${currentSession}/step`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ audience_input: input })
                    });
                    
                    const data = await response.json();
                    if (data.status === 'success') {
                        addDemoStep(data);
                        document.getElementById('audience-input').value = '';
                    }
                } catch (error) {
                    console.error('Error sending input:', error);
                }
            });
            
            document.getElementById('end-demo').addEventListener('click', () => {
                document.getElementById('demo-selection').classList.remove('hidden');
                document.getElementById('demo-interface').classList.add('hidden');
                currentSession = null;
            });
            
            function addDemoStep(stepData) {
                const stepHtml = `
                    <div class="bg-gray-700 p-4 rounded mb-4">
                        <div class="mb-2">
                            <span class="text-blue-300 font-semibold">👤 User:</span>
                            <div class="mt-1">${stepData.step_info?.user_input || 'Continuing demo...'}</div>
                        </div>
                        <div class="mb-2">
                            <span class="text-green-300 font-semibold">🤖 Agents:</span>
                            <div class="mt-1 whitespace-pre-line">${stepData.response || 'Processing...'}</div>
                        </div>
                        ${stepData.step_info?.learning_points ? `
                            <div class="mt-2 flex flex-wrap gap-1">
                                ${stepData.step_info.learning_points.map(point => 
                                    `<span class="text-xs bg-yellow-700 px-2 py-1 rounded">💡 ${point}</span>`
                                ).join('')}
                            </div>
                        ` : ''}
                    </div>
                `;
                
                document.getElementById('demo-content').innerHTML += stepHtml;
                document.getElementById('demo-content').scrollTop = document.getElementById('demo-content').scrollHeight;
            }
        </script>
    </body>
    </html>
    """)

# API Endpoints
@router.get("/api/scenarios")
async def list_demo_scenarios():
    """List all available demo scenarios"""
    try:
        scenarios = demo_scenario_engine.scenario_library.list_scenarios()
        
        return {
            "success": True,
            "scenarios": [
                {
                    "id": scenario.scenario_id,
                    "title": scenario.title,
                    "description": scenario.description,
                    "demo_type": scenario.demo_type.value,
                    "target_audience": scenario.target_audience.value,
                    "duration_minutes": scenario.duration_minutes,
                    "complexity_level": scenario.complexity_level,
                    "learning_objectives": scenario.learning_objectives
                }
                for scenario in scenarios
            ]
        }
    except Exception as e:
        logger.error(f"Error listing scenarios: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing scenarios: {str(e)}")

@router.post("/api/interactive/create")
async def create_demo_session(session_request: DemoSessionRequest):
    """Create new interactive demo session"""
    try:
        session_id = await interactive_demo_manager.create_demo_session({
            "demo_type": session_request.demo_type,
            "audience_type": session_request.audience_type,
            "audience_count": session_request.audience_count,
            "presentation_mode": session_request.presentation_mode,
            "customization": session_request.customization
        })
        
        # Get demo introduction
        intro_data = await interactive_demo_manager.get_demo_introduction(session_id)
        
        return {
            "success": True,
            "session_id": session_id,
            **intro_data
        }
        
    except Exception as e:
        logger.error(f"Error creating demo session: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating demo session: {str(e)}")

@router.post("/api/interactive/{session_id}/step")
async def execute_demo_step(session_id: str, step_request: DemoStepRequest):
    """Execute demo step with audience interaction"""
    try:
        result = await interactive_demo_manager.execute_demo_step_interactive(
            session_id=session_id,
            step_number=step_request.step_number,
            audience_input=step_request.audience_input
        )
        
        return {
            "success": True,
            **result
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error executing demo step: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error executing demo step: {str(e)}")

@router.get("/api/interactive/{session_id}/status")
async def get_demo_status(session_id: str):
    """Get demo session status"""
    try:
        status = await demo_scenario_engine.get_demo_status(session_id)
        
        return {
            "success": True,
            **status
        }
        
    except Exception as e:
        logger.error(f"Error getting demo status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting demo status: {str(e)}")

@router.get("/api/interactive/{session_id}/engagement")
async def get_engagement_metrics(session_id: str):
    """Get audience engagement metrics"""
    try:
        metrics = await interactive_demo_manager.get_audience_engagement_metrics(session_id)
        
        return {
            "success": True,
            "engagement_metrics": metrics
        }
        
    except Exception as e:
        logger.error(f"Error getting engagement metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting engagement metrics: {str(e)}")

@router.delete("/api/interactive/{session_id}")
async def end_demo_session(session_id: str):
    """End demo session and get summary"""
    try:
        summary = await interactive_demo_manager.end_demo_session(session_id)
        
        return {
            "success": True,
            **summary
        }
        
    except Exception as e:
        logger.error(f"Error ending demo session: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error ending demo session: {str(e)}")

@router.get("/api/narratives/{demo_type}/{audience_type}")
async def get_demo_narrative(demo_type: str, audience_type: str):
    """Get demo narrative for specific demo and audience type"""
    try:
        # Get scenario
        scenario = demo_scenario_engine.scenario_library.get_scenario(demo_type)
        if not scenario:
            raise HTTPException(status_code=404, detail=f"Demo type {demo_type} not found")
        
        # Generate narrative
        narrative = await narrative_generator.create_demo_narrative(
            scenario, AudienceType(audience_type)
        )
        
        return {
            "success": True,
            "narrative": narrative,
            "scenario_info": {
                "title": scenario.title,
                "description": scenario.description,
                "learning_objectives": scenario.learning_objectives
            }
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting demo narrative: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting demo narrative: {str(e)}")

@router.websocket("/ws/demo/{session_id}")
async def demo_websocket(websocket: WebSocket, session_id: str):
    """WebSocket for real-time demo interaction"""
    try:
        await websocket.accept()
        logger.info(f"Demo WebSocket connected for session {session_id}")
        
        # Send initial demo status
        status = await demo_scenario_engine.get_demo_status(session_id)
        await websocket.send_json({
            "type": "demo_status",
            "data": status
        })
        
        while True:
            try:
                # Wait for client messages
                data = await websocket.receive_json()
                
                if data.get("type") == "execute_step":
                    # Execute demo step
                    result = await interactive_demo_manager.execute_demo_step_interactive(
                        session_id=session_id,
                        audience_input=data.get("audience_input")
                    )
                    
                    await websocket.send_json({
                        "type": "step_result",
                        "data": result
                    })
                
                elif data.get("type") == "get_status":
                    # Get current status
                    status = await demo_scenario_engine.get_demo_status(session_id)
                    await websocket.send_json({
                        "type": "demo_status", 
                        "data": status
                    })
                
                elif data.get("type") == "end_demo":
                    # End demo
                    summary = await interactive_demo_manager.end_demo_session(session_id)
                    await websocket.send_json({
                        "type": "demo_ended",
                        "data": summary
                    })
                    break
                    
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"Error in demo WebSocket: {str(e)}")
                await websocket.send_json({
                    "type": "error",
                    "message": f"Demo error: {str(e)}"
                })
                
    except WebSocketDisconnect:
        logger.info(f"Demo WebSocket disconnected for session {session_id}")
    except Exception as e:
        logger.error(f"Demo WebSocket error: {str(e)}")
