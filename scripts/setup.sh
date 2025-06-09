#!/bin/bash
# AI QA Agent Development Setup

set -e

echo "🚀 Setting up AI QA Agent development environment..."

# Check Python version
if ! python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)"; then
    echo "❌ Python 3.11+ required"
    exit 1
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "📋 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements-dev.txt

# Install pre-commit hooks
echo "🔧 Setting up pre-commit hooks..."
pre-commit install

# Create .env file
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "📝 Created .env file - please add your API keys"
fi

echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Activate virtual environment: source venv/bin/activate"
echo "  2. Edit .env with your API keys"
echo "  3. Run the application: python src/main.py"
echo "  4. Visit: http://localhost:8000/health"
echo ""
echo "Development commands:"
echo "  • Run tests: pytest"
echo "  • Format code: black src tests"
echo "  • Type check: mypy src"
echo "  • Run app: uvicorn src.main:app --reload"
