"""Tests for the Predictor class."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from fraud_detection_mlops.serving.api.predictor import ModelNotLoadedError, Predictor


def _make_features() -> dict[str, float]:
    return {
        **{f"V{i}": float(i) * 0.01 for i in range(1, 29)},
        "Amount": 50.0,
    }


class TestPredictor:
    def test_not_loaded_initially(self) -> None:
        p = Predictor()
        assert not p.is_loaded
        assert p.model_version == "unloaded"

    def test_predict_before_load_raises(self) -> None:
        p = Predictor()
        with pytest.raises(ModelNotLoadedError):
            p.predict(_make_features())

    def test_mock_model_returns_low_probability(self) -> None:
        """When no model file exists, predictor uses mock — returns 0.05."""
        p = Predictor()
        # Simulate load with mock fallback (no MLflow, no local file)
        with patch.object(p, "_load_local_fallback") as mock_fallback:
            mock_fallback.side_effect = lambda: setattr(p, "_model_version", "mock")
            p._loaded = True
            p._model_version = "mock"

        score, is_fraud = p.predict(_make_features())
        assert score == pytest.approx(0.05)
        assert is_fraud is False

    def test_threshold_controls_fraud_decision(self) -> None:
        """Threshold=0.03 should flag mock score of 0.05 as fraud."""
        p = Predictor(threshold=0.03)
        p._loaded = True
        p._model_version = "mock"

        score, is_fraud = p.predict(_make_features())
        assert is_fraud is True

    def test_high_threshold_flags_nothing_as_fraud(self) -> None:
        """Threshold=0.99 means mock score 0.05 is not fraud."""
        p = Predictor(threshold=0.99)
        p._loaded = True
        p._model_version = "mock"

        score, is_fraud = p.predict(_make_features())
        assert is_fraud is False

    def test_predict_returns_probability_in_range(self) -> None:
        p = Predictor()
        p._loaded = True
        p._model_version = "mock"

        score, _ = p.predict(_make_features())
        assert 0.0 <= score <= 1.0
