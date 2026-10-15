"""Smoke test: confirms the FastAPI app boots and /health responds.

Run with: pytest
"""

from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_health_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
