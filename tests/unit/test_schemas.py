"""Tests for request/response schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from fraud_detection_mlops.serving.api.schemas import PredictRequest


def _valid_request_data() -> dict:
    return {
        **{f"V{i}": 0.0 for i in range(1, 29)},
        "Amount": 142.50,
    }


class TestPredictRequest:
    def test_valid_request_parses(self) -> None:
        req = PredictRequest(**_valid_request_data())
        assert req.Amount == 142.50

    def test_request_id_auto_generated(self) -> None:
        req = PredictRequest(**_valid_request_data())
        assert req.request_id is not None
        assert len(req.request_id) == 36  # UUID4 format

    def test_custom_request_id_accepted(self) -> None:
        data = {**_valid_request_data(), "request_id": "my-trace-123"}
        req = PredictRequest(**data)
        assert req.request_id == "my-trace-123"

    def test_negative_amount_rejected(self) -> None:
        data = {**_valid_request_data(), "Amount": -10.0}
        with pytest.raises(ValidationError, match="Amount"):
            PredictRequest(**data)

    def test_zero_amount_rejected(self) -> None:
        data = {**_valid_request_data(), "Amount": 0.0}
        with pytest.raises(ValidationError):
            PredictRequest(**data)

    def test_missing_pca_feature_rejected(self) -> None:
        data = _valid_request_data()
        del data["V1"]
        with pytest.raises(ValidationError):
            PredictRequest(**data)

    def test_to_feature_dict_excludes_request_id(self) -> None:
        req = PredictRequest(**_valid_request_data())
        feature_dict = req.to_feature_dict()
        assert "request_id" in req.model_dump()
        assert "request_id" not in feature_dict

    def test_to_feature_dict_has_all_features(self) -> None:
        req = PredictRequest(**_valid_request_data())
        feature_dict = req.to_feature_dict()
        for i in range(1, 29):
            assert f"V{i}" in feature_dict
        assert "Amount" in feature_dict
