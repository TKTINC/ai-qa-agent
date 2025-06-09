"""Tests for FastAPI application main module."""

import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_root_endpoint():
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_analysis_status():
    """Test analysis service status."""
    response = client.get("/api/v1/analysis/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"


def test_generation_status():
    """Test generation service status."""
    response = client.get("/api/v1/generation/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
