"""Tests for FastAPI endpoints using the test client."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from fraud_detection_mlops.serving.api.main import app
from fraud_detection_mlops.serving.api.predictor import Predictor


def _valid_payload() -> dict:
    return {
        **{f"V{i}": 0.0 for i in range(1, 29)},
        "Amount": 142.50,
    }


@pytest.fixture
def mock_predictor() -> Predictor:
    """A pre-loaded predictor that returns mock scores."""
    p = Predictor()
    p._loaded = True
    p._model_version = "test-v1"
    return p


# @pytest.fixture
# def client(mock_predictor: Predictor) -> TestClient:
#     """Test client with model loading bypassed."""
#     with patch("fraud_detection_mlops.serving.api.main._predictor", mock_predictor):
#         with patch("fraud_detection_mlops.serving.api.main._app_start_time", 0.0):
#             yield TestClient(app, raise_server_exceptions=True)


@pytest.fixture
def client(mock_predictor: Predictor) -> TestClient:
    """Test client with model loading bypassed."""
    with (
        patch("fraud_detection_mlops.serving.api.main._predictor", mock_predictor),
        patch("fraud_detection_mlops.serving.api.main._app_start_time", 0.0),
    ):
        yield TestClient(app, raise_server_exceptions=True)


class TestRootEndpoint:
    def test_root_returns_200(self, client: TestClient) -> None:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json()["service"] == "fraud-detection-api"


class TestHealthEndpoint:
    def test_health_returns_200(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_schema(self, client: TestClient) -> None:
        data = client.get("/health").json()
        assert "status" in data
        assert "model_loaded" in data
        assert "model_version" in data
        assert "uptime_seconds" in data

    def test_health_shows_model_loaded(self, client: TestClient) -> None:
        data = client.get("/health").json()
        assert data["model_loaded"] is True
        assert data["model_version"] == "test-v1"


class TestPredictEndpoint:
    def test_valid_request_returns_200(self, client: TestClient) -> None:
        response = client.post("/predict", json=_valid_payload())
        assert response.status_code == 200

    def test_response_has_correct_fields(self, client: TestClient) -> None:
        data = client.post("/predict", json=_valid_payload()).json()
        assert "fraud_probability" in data
        assert "is_fraud" in data
        assert "model_version" in data
        assert "request_id" in data
        assert "latency_ms" in data

    def test_fraud_probability_in_valid_range(self, client: TestClient) -> None:
        data = client.post("/predict", json=_valid_payload()).json()
        assert 0.0 <= data["fraud_probability"] <= 1.0

    def test_is_fraud_is_boolean(self, client: TestClient) -> None:
        data = client.post("/predict", json=_valid_payload()).json()
        assert isinstance(data["is_fraud"], bool)

    def test_negative_amount_returns_422(self, client: TestClient) -> None:
        payload = {**_valid_payload(), "Amount": -50.0}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_missing_feature_returns_422(self, client: TestClient) -> None:
        payload = _valid_payload()
        del payload["V1"]
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_custom_request_id_echoed(self, client: TestClient) -> None:
        payload = {**_valid_payload(), "request_id": "trace-abc-123"}
        data = client.post("/predict", json=payload).json()
        assert data["request_id"] == "trace-abc-123"

    def test_model_version_in_response(self, client: TestClient) -> None:
        data = client.post("/predict", json=_valid_payload()).json()
        assert data["model_version"] == "test-v1"


class TestMetricsEndpoint:
    def test_metrics_returns_200(self, client: TestClient) -> None:
        response = client.get("/metrics")
        assert response.status_code == 200

    def test_metrics_contains_prometheus_format(self, client: TestClient) -> None:
        content = client.get("/metrics").text
        assert "fraud_predict_requests_total" in content
