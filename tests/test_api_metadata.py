"""Confirms /health and /metadata return real values, not placeholders."""

from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_health_returns_200():
    response = client.get("/health")
    assert response.status_code == 200


def test_metadata_returns_real_values():
    response = client.get("/metadata")
    assert response.status_code == 200
    data = response.json()
    assert data["model_version"] is not None
    assert "GBM" in data["model_version"]
    assert data["training_range"]["start"] is not None
    assert data["training_range"]["end"] is not None