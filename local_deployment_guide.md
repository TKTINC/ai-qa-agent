# AI QA Agent - Complete Local Deployment Guide

## Overview
This guide provides step-by-step instructions to build and deploy the complete AI QA Agent system on your local machine, including all Sprint 1-5 components: analysis engine, agent intelligence, learning systems, web interface, and intelligent operations.

## Prerequisites

### System Requirements
- **Operating System**: macOS, Linux, or Windows with WSL2
- **Python**: 3.9 or higher
- **Memory**: Minimum 8GB RAM (16GB recommended)
- **Storage**: Minimum 10GB free space
- **Network**: Stable internet connection for AI provider APIs

### Required Software
- **Git**: For repository management
- **Docker**: For containerized services (Redis, PostgreSQL)
- **Node.js**: For frontend dependencies (optional)

## Step 1: Environment Setup

### 1.1 Clone the Repository
```bash
# Clone the repository
git clone <repository-url>
cd ai-qa-agent

# Verify all Sprint files are present
ls -la src/
# Should see: agent/, analysis/, api/, chat/, core/, monitoring/, operations/, validation/, web/
```

### 1.2 Create Python Virtual Environment
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Verify Python version
python --version
# Should show Python 3.9+ 
```

### 1.3 Install Python Dependencies
```bash
# Install all dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Verify key packages
python -c "import fastapi, sqlalchemy, openai, anthropic, redis, numpy, sklearn; print('All packages installed successfully')"
```

## Step 2: Infrastructure Services Setup

### 2.1 Start Docker Services
```bash
# Start Redis and PostgreSQL using Docker Compose
docker-compose up -d redis postgres

# Verify services are running
docker ps
# Should show redis and postgres containers running

# Test Redis connection
docker exec -it ai-qa-agent_redis_1 redis-cli ping
# Should return: PONG

# Test PostgreSQL connection
docker exec -it ai-qa-agent_postgres_1 psql -U postgres -c "SELECT version();"
```

### 2.2 Initialize Database
```bash
# Run database initialization
python -c "
from src.core.database import engine, Base
Base.metadata.create_all(bind=engine)
print('Database initialized successfully')
"

# Verify database tables
python -c "
from src.core.database import engine
from sqlalchemy import text
with engine.connect() as conn:
    result = conn.execute(text('SELECT tablename FROM pg_tables WHERE schemaname = \\'public\\';'))
    tables = [row[0] for row in result]
    print(f'Created tables: {tables}')
"
```

## Step 3: Configuration Setup

### 3.1 Environment Variables
```bash
# Create .env file
cat > .env << 'EOF'
# Environment
ENVIRONMENT=development
DEBUG=true

# Database
DATABASE_URL=postgresql://postgres:password@localhost:5432/qaagent
REDIS_URL=redis://localhost:6379/0

# AI Providers (REQUIRED - Add your API keys)
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Optional: Mock mode for testing without real API keys
MOCK_LLM_PROVIDERS=false

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json

# Security
SECRET_KEY=your-secret-key-for-development

# Monitoring
PROMETHEUS_ENABLED=true
JAEGER_ENABLED=true
EOF

# Make sure to replace API keys with your actual keys!
echo "⚠️  IMPORTANT: Update .env file with your actual OpenAI and Anthropic API keys"
```

### 3.2 Verify Configuration
```bash
# Test configuration loading
python -c "
from src.core.config import get_settings
settings = get_settings()
print(f'Environment: {settings.environment}')
print(f'Database URL configured: {bool(settings.database_url)}')
print(f'Redis URL configured: {bool(settings.redis_url)}')
print(f'OpenAI API Key configured: {bool(settings.openai_api_key)}')
"
```

## Step 4: Core Services Deployment

### 4.1 Start the Main API Server
```bash
# Start FastAPI server
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# In a new terminal, verify API is running
curl http://localhost:8000/health/
# Should return: {"status": "healthy", "timestamp": "..."}
```

### 4.2 Test Core API Endpoints
```bash
# Test health endpoint
curl http://localhost:8000/health/detailed

# Test analysis status
curl http://localhost:8000/api/v1/analysis/status

# Test agent status
curl http://localhost:8000/api/v1/agent/status

# All should return successful responses
```

## Step 5: Agent System Deployment

### 5.1 Initialize Agent Components
```bash
# In a new terminal, test agent system
python -c "
import asyncio
from src.agent.orchestrator import get_agent_orchestrator

async def test_agent():
    orchestrator = await get_agent_orchestrator()
    print('Agent orchestrator initialized successfully')
    
asyncio.run(test_agent())
"
```

### 5.2 Test Multi-Agent Collaboration
```bash
# Test specialist agents
python -c "
import asyncio
from src.agent.specialists.test_architect import TestArchitect

async def test_specialists():
    architect = TestArchitect()
    await architect.initialize()
    print('Test Architect initialized successfully')
    
asyncio.run(test_specialists())
"
```

## Step 6: Learning System Deployment

### 6.1 Initialize Learning Engine
```bash
# Test learning system
python -c "
import asyncio
from src.agent.learning.learning_engine import get_learning_engine

async def test_learning():
    engine = await get_learning_engine()
    print('Learning engine initialized successfully')
    
asyncio.run(test_learning())
"
```

### 6.2 Test Analytics System
```bash
# Test analytics APIs
curl -X GET http://localhost:8000/api/v1/learning/analytics/agent-intelligence
```

## Step 7: Web Interface Deployment

### 7.1 Start Web Interface
```bash
# The web interface is integrated with the main API
# Test web routes
curl http://localhost:8000/
curl http://localhost:8000/agent-chat
curl http://localhost:8000/analytics
curl http://localhost:8000/demos

# All should return HTML content
```

### 7.2 Access Web Interface
```bash
# Open web browser to:
echo "🌐 Web Interface URLs:"
echo "Main Dashboard: http://localhost:8000/"
echo "Agent Chat: http://localhost:8000/agent-chat"
echo "Analytics: http://localhost:8000/analytics"
echo "Demos: http://localhost:8000/demos"
echo "API Documentation: http://localhost:8000/docs"
```

## Step 8: Monitoring & Operations Deployment

### 8.1 Start Monitoring Services
```bash
# Start Prometheus and Grafana
docker-compose up -d prometheus grafana

# Verify monitoring services
curl http://localhost:9090/  # Prometheus
curl http://localhost:3000/  # Grafana (admin/admin)
```

### 8.2 Initialize Intelligent Operations
```bash
# Test cost optimization
python -c "
import asyncio
from src.operations.optimization.cost_optimization import get_cost_optimizer

async def test_cost_opt():
    optimizer = await get_cost_optimizer()
    result = await optimizer.optimize_ai_provider_usage()
    print(f'Cost optimization initialized: {result.savings_percentage:.1f}% potential savings')
    
asyncio.run(test_cost_opt())
"

# Test intelligent alerting
python -c "
import asyncio
from src.operations.alerting.intelligent_alerting import get_intelligent_alerting

async def test_alerting():
    alerting = await get_intelligent_alerting()
    print('Intelligent alerting system initialized successfully')
    
asyncio.run(test_alerting())
"
```

## Step 9: Complete System Validation

### 9.1 Run System Health Check
```bash
# Comprehensive health check
curl http://localhost:8000/health/detailed | python -m json.tool

# Check all service components
python -c "
import asyncio
import aiohttp

async def health_check():
    endpoints = [
        'http://localhost:8000/health/',
        'http://localhost:8000/api/v1/analysis/status',
        'http://localhost:8000/api/v1/agent/status',
        'http://localhost:8000/health/monitoring'
    ]
    
    async with aiohttp.ClientSession() as session:
        for endpoint in endpoints:
            try:
                async with session.get(endpoint) as response:
                    status = response.status
                    print(f'{endpoint}: {status} ({'✅' if status == 200 else '❌'})')
            except Exception as e:
                print(f'{endpoint}: Error - {e}')

asyncio.run(health_check())
"
```

### 9.2 Test Complete Agent Workflow
```bash
# Run comprehensive demo
python demo_sprint_5_4_complete.py

# Should show successful initialization of all components
```

## Step 10: Background Services (Optional)

### 10.1 Start Background Task Worker
```bash
# In a new terminal, start Celery worker for background tasks
celery -A src.tasks.worker worker --loglevel=info

# In another terminal, start Celery beat for scheduled tasks
celery -A src.tasks.worker beat --loglevel=info
```

### 10.2 Start Monitoring Collection
```bash
# Start metrics collection (if not using Docker)
python -c "
import asyncio
from src.monitoring.metrics.agent_intelligence_metrics import AgentIntelligenceMetrics

async def start_metrics():
    metrics = AgentIntelligenceMetrics()
    print('Metrics collection started')
    
asyncio.run(start_metrics())
"
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Database Connection Error
```bash
# Check PostgreSQL is running
docker ps | grep postgres

# Reset database if needed
docker-compose down postgres
docker-compose up -d postgres
```

#### 2. Redis Connection Error
```bash
# Check Redis is running
docker ps | grep redis

# Test Redis connectivity
python -c "import redis; r = redis.Redis(host='localhost', port=6379); print(r.ping())"
```

#### 3. API Key Issues
```bash
# Verify API keys are set
python -c "
from src.core.config import get_settings
settings = get_settings()
print(f'OpenAI key configured: {bool(settings.openai_api_key and len(settings.openai_api_key) > 10)}')
print(f'Anthropic key configured: {bool(settings.anthropic_api_key and len(settings.anthropic_api_key) > 10)}')
"

# Enable mock mode for testing without API keys
export MOCK_LLM_PROVIDERS=true
```

#### 4. Port Conflicts
```bash
# Check if ports are in use
lsof -i :8000  # FastAPI
lsof -i :6379  # Redis
lsof -i :5432  # PostgreSQL

# Kill processes if needed
kill -9 <PID>
```

#### 5. Module Import Errors
```bash
# Ensure you're in the project root and virtual environment is activated
pwd  # Should show ai-qa-agent directory
which python  # Should point to venv/bin/python

# Add project to Python path if needed
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

## Performance Optimization

### 10.1 Production-like Settings
```bash
# For better performance, update .env:
cat >> .env << 'EOF'
# Performance settings
UVICORN_WORKERS=4
REDIS_MAX_CONNECTIONS=100
DATABASE_POOL_SIZE=20
LOG_LEVEL=WARNING
EOF
```

### 10.2 Resource Monitoring
```bash
# Monitor system resources
htop  # CPU and memory usage
docker stats  # Container resource usage
```

## Verification Checklist

- [ ] ✅ Python 3.9+ environment activated
- [ ] ✅ All dependencies installed successfully
- [ ] ✅ Docker services (Redis, PostgreSQL) running
- [ ] ✅ Database initialized with all tables
- [ ] ✅ Environment variables configured
- [ ] ✅ FastAPI server running on port 8000
- [ ] ✅ Health endpoints returning 200 status
- [ ] ✅ Agent system initialized successfully
- [ ] ✅ Learning engine operational
- [ ] ✅ Web interface accessible
- [ ] ✅ Monitoring services running
- [ ] ✅ Intelligent operations initialized
- [ ] ✅ Complete system demo runs successfully

## Next Steps

Once local deployment is complete:

1. **Run Sanity Tests**: Follow the Sanity Testing Guide
2. **Test E2E Workflows**: Follow the E2E Functional Testing Guide
3. **Cloud Deployment**: Follow the AWS Cloud Deployment Guide
4. **Production Configuration**: Update settings for production use

## Support

For issues during deployment:

1. Check the troubleshooting section above
2. Review application logs: `tail -f logs/app.log`
3. Check Docker service logs: `docker-compose logs <service>`
4. Verify all prerequisites are met
5. Ensure API keys are correctly configured

---

**🎉 Congratulations! Your AI QA Agent system is now running locally with all Sprint 1-5 capabilities operational.**