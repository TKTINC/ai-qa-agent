# 🤖 AI QA Agent

**Intelligent test generation for any codebase using advanced AI and code analysis**

[![CI](https://github.com/yourusername/ai-qa-agent/workflows/CI/badge.svg)](https://github.com/yourusername/ai-qa-agent/actions)
[![Coverage](https://codecov.io/gh/yourusername/ai-qa-agent/branch/main/graph/badge.svg)](https://codecov.io/gh/yourusername/ai-qa-agent)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)

## 🎯 Project Overview

AI QA Agent is a production-ready system that automatically generates comprehensive test suites for any codebase using:

- **🔍 Intelligent Code Analysis** - AST parsing and pattern recognition
- **🤖 AI Test Generation** - Multi-provider LLM integration (OpenAI, Anthropic)
- **⚡ Real-time Processing** - FastAPI backend with async processing
- **🎨 Modern Web Interface** - Interactive dashboard for monitoring and control
- **🐳 Production Ready** - Docker deployment with monitoring and logging

## ✨ Key Features

- **Multi-Language Support** - Python, JavaScript, TypeScript, and more
- **Advanced Code Analysis** - Function complexity, dependency mapping, pattern detection
- **AI-Powered Generation** - Context-aware test creation with multiple strategies
- **Quality Validation** - Syntax checking, execution testing, and quality scoring
- **Real-time Monitoring** - Progress tracking and comprehensive health checks
- **Scalable Architecture** - Microservices design with Redis task queuing

## 🏗️ Architecture

```
┌─────────────────┬─────────────────┬─────────────────┐
│   Web Interface │   FastAPI API   │  Background     │
│   (HTMX/Tailwind│   (Async Endpoints) │  Tasks (Celery) │
└─────────────────┼─────────────────┼─────────────────┘
                  │                 │
┌─────────────────┼─────────────────┼─────────────────┐
│  Code Analysis  │  AI Generation  │  Test Validation│
│  • AST Parser   │  • OpenAI       │  • Syntax Check │
│  • Complexity   │  • Anthropic    │  • Execution    │
│  • Patterns     │  • Local Models │  • Quality Score│
└─────────────────┼─────────────────┼─────────────────┘
                  │                 │
            ┌─────────────────────────────────────┐
            │     Data Layer (SQLAlchemy)         │
            │  • Analysis Sessions • Generated Tests │
            │  • Code Components  • Task Status     │
            └─────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Git
- Docker (optional, for containerized deployment)

### Installation

1. **Clone and setup the project:**
```bash
git clone https://github.com/yourusername/ai-qa-agent.git
cd ai-qa-agent
chmod +x scripts/setup.sh
./scripts/setup.sh
```

2. **Configure environment:**
```bash
# Edit .env file with your API keys
cp .env.example .env
# Add your OpenAI and/or Anthropic API keys
```

3. **Run the application:**
```bash
source venv/bin/activate
uvicorn src.main:app --reload
```

4. **Access the application:**
- **API**: http://localhost:8000
- **Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up --build

# Access at http://localhost:8000
```

## 📁 Project Structure

```
ai-qa-agent/
├── src/
│   ├── core/              # Configuration, database, logging
│   ├── analysis/          # Code parsing and analysis
│   ├── generation/        # AI test generation
│   ├── validation/        # Test validation and scoring
│   ├── api/              # FastAPI routes and models
│   ├── web/              # Web interface templates
│   └── utils/            # Shared utilities
├── tests/                # Comprehensive test suite
├── docs/                 # Documentation
├── examples/             # Sample projects and demos
└── scripts/              # Development and deployment scripts
```

## 🧪 Development

### Running Tests
```bash
# Run all tests with coverage
pytest

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
pytest tests/e2e/

# Run with coverage report
pytest --cov=src --cov-report=html
```

### Code Quality
```bash
# Format code
black src tests

# Type checking
mypy src

# Linting
flake8 src tests

# Security scanning
bandit -r src
```

### Development Server
```bash
# Run with auto-reload
uvicorn src.main:app --reload --log-level debug

# Run background worker (if using Celery)
celery -A src.core.tasks worker --loglevel=info
```

## 📊 API Endpoints

### Health & Monitoring
- `GET /health` - Basic health check
- `GET /health/detailed` - Detailed system metrics
- `GET /health/ready` - Kubernetes readiness probe
- `GET /health/metrics` - Application metrics

### Code Analysis
- `POST /api/v1/analysis/repository` - Analyze repository
- `GET /api/v1/analysis/{session_id}` - Get analysis results
- `GET /api/v1/analysis/{session_id}/components` - List components

### Test Generation
- `POST /api/v1/generation/generate` - Generate tests
- `GET /api/v1/generation/{task_id}` - Get generation status
- `GET /api/v1/generation/results/{session_id}` - Get generated tests

### Validation
- `POST /api/v1/validation/validate` - Validate test code
- `GET /api/v1/validation/{test_id}/score` - Get quality score

## 🔧 Configuration

### Environment Variables
```bash
# AI Providers
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here

# Database
DATABASE_URL=sqlite:///./app.db  # or postgresql://...

# Redis (for background tasks)
REDIS_URL=redis://localhost:6379

# Application
DEBUG=False
LOG_LEVEL=INFO
MAX_UPLOAD_SIZE=50MB
```

### AI Provider Configuration
The system supports multiple AI providers with automatic fallback:

1. **OpenAI GPT-4** - Primary provider for complex analysis
2. **Anthropic Claude** - Fallback with different strengths
3. **Local Models** - For privacy-sensitive deployments

## 🚀 Deployment

### Production Deployment
```bash
# Build production image
docker build -t ai-qa-agent:latest .

# Deploy with docker-compose
docker-compose -f docker-compose.prod.yml up -d

# Or deploy to Kubernetes
kubectl apply -f k8s/
```

### Environment-Specific Configs
- **Development**: SQLite, debug logging, auto-reload
- **Testing**: In-memory database, mock AI providers
- **Production**: PostgreSQL, structured logging, monitoring

## 📈 Monitoring

### Health Checks
The application provides comprehensive health monitoring:
- Database connectivity
- AI provider availability  
- System resource usage
- Background task status

### Metrics
Key metrics tracked:
- Analysis success rate
- Test generation quality scores
- Response times
- Resource utilization

### Logging
Structured logging with multiple levels:
- Application logs: `logs/app.log`
- Error logs: `logs/error.log` 
- Access logs: `logs/access.log`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes and add tests
4. Run quality checks: `./scripts/check-quality.sh`
5. Commit changes: `git commit -m 'Add amazing feature'`
6. Push to branch: `git push origin feature/amazing-feature`
7. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙋‍♂️ Support

- **Documentation**: [Full documentation](docs/)
- **Issues**: [GitHub Issues](https://github.com/yourusername/ai-qa-agent/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/ai-qa-agent/discussions)

## 🎯 Roadmap

- [ ] **v1.1**: Advanced pattern detection with ML models
- [ ] **v1.2**: Visual test reporting and analytics
- [ ] **v1.3**: Integration with popular CI/CD platforms
- [ ] **v2.0**: Multi-repository analysis and comparison

---

**Built with ❤️ for the developer community**
