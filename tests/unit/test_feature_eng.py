"""Tests for feature engineering pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from fraud_detection_mlops.training.features.feature_eng import (
    ALL_FEATURES,
    AmountLogTransformer,
    FeatureSelector,
    build_feature_pipeline,
    prepare_features,
)


def _make_df(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "Time": np.arange(n, dtype=float),
            "Amount": rng.uniform(0.01, 500.0, n),
            "Class": rng.choice([0, 1], n, p=[0.99, 0.01]),
            **{f"V{i}": rng.normal(0, 1, n) for i in range(1, 29)},
        }
    )


class TestAmountLogTransformer:
    def test_log_transforms_amount(self) -> None:
        df = pd.DataFrame({"Amount": [0.0, 1.0, 99.0, 25000.0]})
        t = AmountLogTransformer()
        result = t.transform(df)
        expected = np.log1p([0.0, 1.0, 99.0, 25000.0])
        np.testing.assert_allclose(result["Amount"].values, expected)

    def test_fit_returns_self(self) -> None:
        t = AmountLogTransformer()
        result = t.fit(pd.DataFrame({"Amount": [1.0]}))
        assert result is t

    def test_no_negative_outputs_for_positive_input(self) -> None:
        df = pd.DataFrame({"Amount": np.linspace(0, 1000, 100)})
        result = AmountLogTransformer().transform(df)
        assert (result["Amount"] >= 0).all()


class TestFeatureSelector:
    def test_selects_correct_columns(self) -> None:
        df = _make_df()
        selector = FeatureSelector()
        result = selector.fit_transform(df)
        assert list(result.columns) == ALL_FEATURES
        assert "Time" not in result.columns
        assert "Class" not in result.columns

    def test_raises_on_missing_column_at_fit(self) -> None:
        df = _make_df().drop(columns=["V1"])
        selector = FeatureSelector()
        with pytest.raises(ValueError, match="missing columns"):
            selector.fit(df)

    def test_raises_on_missing_column_at_transform(self) -> None:
        df_train = _make_df()
        df_test = _make_df().drop(columns=["V1"])
        selector = FeatureSelector()
        selector.fit(df_train)
        with pytest.raises(ValueError, match="missing columns"):
            selector.transform(df_test)


class TestBuildFeaturePipeline:
    def test_returns_pipeline(self) -> None:
        pipeline = build_feature_pipeline()
        assert isinstance(pipeline, Pipeline)

    def test_pipeline_has_three_steps(self) -> None:
        pipeline = build_feature_pipeline()
        assert len(pipeline.steps) == 3

    def test_output_shape_correct(self) -> None:
        df = _make_df(100)
        pipeline = build_feature_pipeline()
        X = pipeline.fit_transform(df)
        assert X.shape == (100, len(ALL_FEATURES))

    def test_scaler_standardizes_output(self) -> None:
        df = _make_df(1000)
        pipeline = build_feature_pipeline()
        X = pipeline.fit_transform(df)
        # After StandardScaler, mean≈0 and std≈1 for each column
        np.testing.assert_allclose(X.mean(axis=0), 0.0, atol=1e-10)
        np.testing.assert_allclose(X.std(axis=0), 1.0, atol=0.05)


class TestPrepareFeatures:
    def test_returns_correct_shapes(self) -> None:
        df = _make_df(200)
        pipeline = build_feature_pipeline()
        X, y = prepare_features(df, pipeline, fit=True)
        assert X.shape == (200, len(ALL_FEATURES))
        assert y.shape == (200,)

    def test_fit_false_on_val_does_not_refit(self) -> None:
        df_train = _make_df(200)
        df_val = _make_df(50)
        pipeline = build_feature_pipeline()
        prepare_features(df_train, pipeline, fit=True)
        # Should not raise — pipeline already fitted
        X_val, _ = prepare_features(df_val, pipeline, fit=False)
        assert X_val.shape[1] == len(ALL_FEATURES)

    def test_raises_without_target_column(self) -> None:
        df = _make_df().drop(columns=["Class"])
        pipeline = build_feature_pipeline()
        with pytest.raises(ValueError, match="Class"):
            prepare_features(df, pipeline, fit=True)
