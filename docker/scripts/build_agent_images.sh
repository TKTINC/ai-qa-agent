#!/bin/bash
# Build script for all agent Docker images

set -e

echo "🚀 Building AI QA Agent Docker Images..."

# Build base image
echo "📦 Building agent base image..."
docker build -f docker/agent-system/Dockerfile.agent-base -t qa-agent/base:latest .

# Build orchestrator
echo "📦 Building agent orchestrator..."
docker build -f docker/agent-system/Dockerfile.orchestrator -t qa-agent/orchestrator:latest .

# Build specialists
echo "📦 Building specialist agents..."
docker build -f docker/agent-system/Dockerfile.specialists -t qa-agent/specialists:latest .

# Build conversation manager
echo "📦 Building conversation manager..."
docker build -f docker/agent-system/Dockerfile.conversation -t qa-agent/conversation:latest .

# Build learning engine
echo "📦 Building learning engine..."
docker build -f docker/agent-system/Dockerfile.learning -t qa-agent/learning:latest .

echo "✅ All agent images built successfully!"

# List built images
echo "📋 Built images:"
docker images qa-agent/*
