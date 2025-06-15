#!/bin/bash
# Kubernetes deployment script for AI QA Agent

set -e

echo "🚀 Deploying AI QA Agent to Kubernetes..."

# Apply namespace and configs
echo "📝 Creating namespace and configurations..."
kubectl apply -f k8s/configs/namespace.yaml
kubectl apply -f k8s/configs/agent-config.yaml

# Deploy core components
echo "📦 Deploying agent components..."
kubectl apply -f k8s/agent-system/orchestrator-deployment.yaml
kubectl apply -f k8s/agent-system/specialists-deployment.yaml
kubectl apply -f k8s/agent-system/conversation-deployment.yaml

# Wait for deployments
echo "⏳ Waiting for deployments to be ready..."
kubectl wait --for=condition=available --timeout=300s deployment/agent-orchestrator -n qa-agent
kubectl wait --for=condition=available --timeout=300s deployment/specialist-agents -n qa-agent
kubectl wait --for=condition=available --timeout=300s deployment/conversation-manager -n qa-agent

# Show status
echo "📊 Deployment status:"
kubectl get pods -n qa-agent
kubectl get services -n qa-agent

echo "✅ AI QA Agent deployed successfully to Kubernetes!"
