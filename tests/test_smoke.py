"""Smoke tests for the Self-Healing MLOps project."""

import os
import sys
import importlib

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_api_module_imports():
    """Verify core API module can be imported."""
    import app.api  # noqa: F401


def test_model_module_imports():
    """Verify model wrapper module can be imported."""
    import app.model  # noqa: F401


def test_preprocessing_imports():
    """Verify preprocessing module can be imported."""
    import src.preprocessing  # noqa: F401


def test_feature_engineering_imports():
    """Verify feature engineering module can be imported."""
    import src.feature_engineering  # noqa: F401


def test_stats_utils_imports():
    """Verify stats utilities can be imported."""
    import utils.stats_utils  # noqa: F401


def test_fastapi_app_creation():
    """Verify FastAPI app instance is created correctly."""
    from app.api import app
    assert app.title == "Self-Healing MLOps API"


def test_health_endpoint():
    """Verify health endpoint returns expected structure."""
    from fastapi.testclient import TestClient
    from app.api import app

    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"
