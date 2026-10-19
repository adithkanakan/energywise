"""Confirms /health and /metadata return real values, not placeholders."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)

DATA_FILE = Path("data/processed/hourly_energy_weather.parquet")


def test_health_returns_200():
    response = client.get("/health")
    assert response.status_code == 200


@pytest.mark.skipif(
    not DATA_FILE.exists(),
    reason="processed data not present (it is gitignored); run the pipeline first",
)
def test_metadata_returns_real_values():
    response = client.get("/metadata")
    assert response.status_code == 200
    data = response.json()
    assert data["model_version"] is not None
    assert "GBM" in data["model_version"]
    assert data["training_range"]["start"] is not None
    assert data["training_range"]["end"] is not None
