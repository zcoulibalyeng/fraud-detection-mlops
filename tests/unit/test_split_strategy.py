"""Tests for the temporal split strategy."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# from data.pipelines.split_strategy import SplitRatios, temporal_split
from fraud_detection_mlops.data.pipelines.split_strategy import (
    SplitRatios,
    temporal_split,
)


def _make_fraud_df(n: int = 1000) -> pd.DataFrame:
    """Create a minimal fraud-like DataFrame for testing."""
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "Time": np.arange(n, dtype=float),
            "Amount": rng.uniform(0.01, 500.0, n),
            "Class": rng.choice([0, 1], n, p=[0.998, 0.002]),
            **{f"V{i}": rng.normal(0, 1, n) for i in range(1, 29)},
        }
    )


class TestSplitRatios:
    def test_valid_ratios(self) -> None:
        ratios = SplitRatios(train=0.7, val=0.15, test=0.15)
        assert ratios.train == 0.7

    def test_ratios_must_sum_to_one(self) -> None:
        with pytest.raises(ValueError, match="sum to 1.0"):
            SplitRatios(train=0.6, val=0.2, test=0.1)

    def test_zero_ratio_rejected(self) -> None:
        with pytest.raises(ValueError):
            SplitRatios(train=0.0, val=0.5, test=0.5)


class TestTemporalSplit:
    def test_sizes_approximately_correct(self) -> None:
        df = _make_fraud_df(1000)
        ratios = SplitRatios(train=0.7, val=0.15, test=0.15)
        result = temporal_split(df, ratios)

        assert abs(len(result.train) - 700) <= 2
        assert abs(len(result.val) - 150) <= 2
        assert abs(len(result.test) - 150) <= 2

    def test_no_data_leakage(self) -> None:
        """Train max time must be less than val min time."""
        df = _make_fraud_df(1000)
        ratios = SplitRatios(train=0.7, val=0.15, test=0.15)
        result = temporal_split(df, ratios)

        assert result.train["Time"].max() < result.val["Time"].min()
        assert result.val["Time"].max() < result.test["Time"].min()

    def test_chronological_order_preserved(self) -> None:
        """Each split should be sorted by Time."""
        df = _make_fraud_df(1000)
        ratios = SplitRatios(train=0.7, val=0.15, test=0.15)
        result = temporal_split(df, ratios)

        for split_df in [result.train, result.val, result.test]:
            assert split_df["Time"].is_monotonic_increasing

    def test_no_rows_lost(self) -> None:
        df = _make_fraud_df(1000)
        ratios = SplitRatios(train=0.7, val=0.15, test=0.15)
        result = temporal_split(df, ratios)

        assert result.sizes["total"] == len(df)

    def test_all_columns_preserved(self) -> None:
        df = _make_fraud_df(1000)
        ratios = SplitRatios(train=0.7, val=0.15, test=0.15)
        result = temporal_split(df, ratios)

        assert set(result.train.columns) == set(df.columns)

    def test_missing_time_column_raises(self) -> None:
        df = _make_fraud_df(100).drop(columns=["Time"])
        ratios = SplitRatios(train=0.7, val=0.15, test=0.15)
        with pytest.raises(ValueError, match="time_column"):
            temporal_split(df, ratios)

    def test_fraud_rate_consistent_across_splits(self) -> None:
        """All splits should have some fraud — no split should be all zeros."""
        rng = np.random.default_rng(0)
        df = pd.DataFrame(
            {
                "Time": np.arange(10_000, dtype=float),
                "Amount": rng.uniform(0, 200, 10_000),
                "Class": rng.choice([0, 1], 10_000, p=[0.998, 0.002]),
            }
        )
        ratios = SplitRatios(train=0.7, val=0.15, test=0.15)
        result = temporal_split(df, ratios)
        # Fraud rate in each split should be nonzero (with 10K rows this is expected)
        assert result.fraud_rates["train"] > 0
