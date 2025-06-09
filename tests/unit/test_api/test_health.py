"""Tests for health monitoring endpoints."""

import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_health_check():
    """Test basic health check."""
    response = client.get("/health/")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "timestamp" in data


def test_detailed_health_check():
    """Test detailed health check."""
    response = client.get("/health/detailed")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "system" in data
    assert "database" in data


def test_readiness_check():
    """Test readiness probe."""
    response = client.get("/health/ready")
    assert response.status_code == 200
    data = response.json()
    assert "ready" in data
    assert "checks" in data


def test_metrics_endpoint():
    """Test metrics endpoint."""
    response = client.get("/health/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "business_metrics" in data
    assert "system_metrics" in data


def test_liveness_check():
    """Test liveness probe."""
    response = client.get("/health/live")
    assert response.status_code == 200
    data = response.json()
    assert data["alive"] == True
