"""Tests for the data validator — good data passes, bad data raises."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# from data.validation.validator import DataValidationError, FraudDataValidator
from fraud_detection_mlops.data.validation.expectations import FraudExpectations
from fraud_detection_mlops.data.validation.validator import (
    DataValidationError,
    FraudDataValidator,
)


def _make_valid_df(n: int = 1000) -> pd.DataFrame:
    """Create a valid fraud dataset for testing."""
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "Time": np.linspace(0, 172800, n),
            "Amount": rng.uniform(0.01, 500.0, n),
            "Class": rng.choice([0, 1], n, p=[0.998, 0.002]),
            **{f"V{i}": rng.normal(0, 1, n) for i in range(1, 29)},
        }
    )


def _test_expectations(n: int = 1000) -> FraudExpectations:
    """
    Expectations with relaxed volume/fraud-rate thresholds for unit tests.

    In production the validator uses full thresholds (100K+ rows).
    In tests we use small DataFrames — override only the volume constraints.
    """
    return FraudExpectations(
        min_row_count=10,
        max_row_count=100_000,
        min_fraud_rate=0.0,  # small df may have zero fraud
        max_fraud_rate=1.0,
    )


class TestFraudDataValidator:
    def test_valid_dataframe_passes(self, tmp_path: Path) -> None:
        df = _make_valid_df()
        validator = FraudDataValidator(
            report_dir=tmp_path, expectations=_test_expectations()
        )
        # Should not raise
        validator.validate(df, run_name="test_valid")

    def test_missing_column_fails(self, tmp_path: Path) -> None:
        df = _make_valid_df().drop(columns=["Amount"])
        validator = FraudDataValidator(
            report_dir=tmp_path, expectations=_test_expectations()
        )
        with pytest.raises(DataValidationError, match="Amount"):
            validator.validate(df, run_name="test_missing_col")

    def test_null_values_fail(self, tmp_path: Path) -> None:
        df = _make_valid_df()
        df.loc[0:10, "Amount"] = None
        validator = FraudDataValidator(
            report_dir=tmp_path, expectations=_test_expectations()
        )
        with pytest.raises(DataValidationError, match="null"):
            validator.validate(df, run_name="test_nulls")

    def test_invalid_class_labels_fail(self, tmp_path: Path) -> None:
        df = _make_valid_df()
        df.loc[0:5, "Class"] = 99
        validator = FraudDataValidator(
            report_dir=tmp_path, expectations=_test_expectations()
        )
        with pytest.raises(DataValidationError, match="invalid values"):
            validator.validate(df, run_name="test_bad_labels")

    def test_negative_amount_fails(self, tmp_path: Path) -> None:
        df = _make_valid_df()
        df.loc[0:5, "Amount"] = -100.0
        validator = FraudDataValidator(
            report_dir=tmp_path, expectations=_test_expectations()
        )
        with pytest.raises(DataValidationError, match="below"):
            validator.validate(df, run_name="test_neg_amount")

    def test_report_file_written(self, tmp_path: Path) -> None:
        df = _make_valid_df()
        validator = FraudDataValidator(
            report_dir=tmp_path, expectations=_test_expectations()
        )
        validator.validate(df, run_name="test_report")
        report = tmp_path / "test_report_report.txt"
        assert report.exists()
        content = report.read_text()
        assert "PASS" in content
        assert "Failed: 0" in content

    def test_failed_report_contains_failure_details(self, tmp_path: Path) -> None:
        df = _make_valid_df().drop(columns=["Amount"])
        validator = FraudDataValidator(
            report_dir=tmp_path, expectations=_test_expectations()
        )
        with pytest.raises(DataValidationError):
            validator.validate(df, run_name="test_fail_report")
        report = tmp_path / "test_fail_report_report.txt"
        assert report.exists()
        content = report.read_text()
        assert "FAIL" in content

    def test_all_failures_reported_at_once(self, tmp_path: Path) -> None:
        """Validator collects all failures, not just the first."""
        df = _make_valid_df()
        df.loc[0:5, "Amount"] = -1.0
        df.loc[0:5, "Class"] = 99
        validator = FraudDataValidator(
            report_dir=tmp_path, expectations=_test_expectations()
        )
        with pytest.raises(DataValidationError) as exc_info:
            validator.validate(df, run_name="test_multi_fail")
        # Both failures should appear in the error message
        assert "Amount" in str(exc_info.value)
        assert "invalid values" in str(exc_info.value)
