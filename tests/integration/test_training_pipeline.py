"""
Integration test — trains a real model on a small dataset subset.

This is the smoke test that catches the 80% of bugs that unit
tests cannot: pipeline shape mismatches, MLflow connection issues,
model serialization failures.

Runs on every CI push (takes ~30s on small data).
Does NOT require AWS — uses local filesystem and local MLflow.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from fraud_detection_mlops.data.pipelines.split_strategy import (
    SplitRatios,
    SplitResult,
    temporal_split,
)
from fraud_detection_mlops.training.evaluate import evaluate_model
from fraud_detection_mlops.training.evaluation_gate import (
    GateDecision,
    run_evaluation_gate,
)
from fraud_detection_mlops.training.features.feature_eng import (
    build_feature_pipeline,
    prepare_features,
)
from fraud_detection_mlops.training.models.xgb_model import XGBoostFraudModel


def _make_realistic_fraud_df(n: int = 5000) -> pd.DataFrame:
    """
    Generate a realistic fraud dataset for integration testing.

    Uses n=5000 for speed — enough to train a meaningful model
    but fast enough for CI (< 30s).
    """
    rng = np.random.default_rng(42)

    # 0.2% fraud rate — realistic for credit card data
    n_fraud = max(10, int(n * 0.002))
    n_legit = n - n_fraud

    fraud_rows = pd.DataFrame(
        {
            "Time": rng.uniform(0, 172800, n_fraud),
            "Amount": rng.uniform(1.0, 200.0, n_fraud),
            "Class": 1,
            **{f"V{i}": rng.normal(2.0, 1.5, n_fraud) for i in range(1, 29)},
        }
    )
    legit_rows = pd.DataFrame(
        {
            "Time": rng.uniform(0, 172800, n_legit),
            "Amount": rng.uniform(1.0, 500.0, n_legit),
            "Class": 0,
            **{f"V{i}": rng.normal(0.0, 1.0, n_legit) for i in range(1, 29)},
        }
    )

    df = pd.concat([fraud_rows, legit_rows], ignore_index=True)
    return df.sort_values("Time").reset_index(drop=True)


@pytest.fixture(scope="module")
def split_data() -> SplitResult:
    """Prepare train/val/test splits once for all tests in this module."""
    df = _make_realistic_fraud_df(5000)
    ratios = SplitRatios(train=0.70, val=0.15, test=0.15)
    return temporal_split(df, ratios)


@pytest.fixture(scope="module")
def trained_model_and_pipeline(
    split_data: SplitResult,
) -> tuple[XGBoostFraudModel, Pipeline, np.ndarray, pd.Series, np.ndarray, pd.Series]:
    """Train a real model once — reused across all integration tests."""
    pipeline = build_feature_pipeline()

    X_train, y_train = prepare_features(split_data.train, pipeline, fit=True)
    X_val, y_val = prepare_features(split_data.val, pipeline, fit=False)

    model = XGBoostFraudModel(
        n_estimators=50,  # small for speed
        max_depth=4,
        learning_rate=0.1,
        scale_pos_weight=float((y_train == 0).sum() / max((y_train == 1).sum(), 1)),
        early_stopping_rounds=5,
        random_seed=42,
    )
    model.fit(X_train, y_train, X_val, y_val, feature_names=None)

    return model, pipeline, X_train, y_train, X_val, y_val


class TestTrainingPipelineIntegration:
    def test_model_trains_without_error(
        self, trained_model_and_pipeline: tuple[Any, ...]
    ) -> None:
        model, _, X_train, y_train, _, _ = trained_model_and_pipeline
        assert model is not None

    def test_model_produces_valid_probabilities(
        self, trained_model_and_pipeline: tuple[Any, ...], split_data: SplitResult
    ) -> None:
        model, pipeline, _, _, _, _ = trained_model_and_pipeline
        X_test, y_test = prepare_features(split_data.test, pipeline, fit=False)
        proba = model.predict_proba(X_test)
        assert proba.shape == (len(y_test),)
        assert (proba >= 0).all()
        assert (proba <= 1).all()

    def test_model_achieves_reasonable_auc(
        self, trained_model_and_pipeline: tuple[Any, ...], split_data: SplitResult
    ) -> None:
        """AUC-PR should be well above random (0.002 base rate)."""
        model, pipeline, _, _, _, _ = trained_model_and_pipeline
        X_test, y_test = prepare_features(split_data.test, pipeline, fit=False)
        metrics = evaluate_model(model, X_test, y_test, dataset_name="test")
        # With clear signal in synthetic data, AUC-PR >> 0.1
        assert metrics.auc_pr > 0.1, (
            f"AUC-PR too low: {metrics.auc_pr:.4f}. "
            "Model may not be learning signal."
        )

    def test_model_save_and_load_roundtrip(
        self, trained_model_and_pipeline: tuple[Any, ...], split_data: SplitResult
    ) -> None:
        """Saved model must produce identical predictions after loading."""
        model, pipeline, _, _, _, _ = trained_model_and_pipeline
        X_test, _ = prepare_features(split_data.test, pipeline, fit=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.joblib"
            model.save(path)
            loaded_model = XGBoostFraudModel.load(path)

        original_proba = model.predict_proba(X_test)
        loaded_proba = loaded_model.predict_proba(X_test)

        np.testing.assert_allclose(
            original_proba,
            loaded_proba,
            rtol=1e-5,
            err_msg="Save/load roundtrip changed predictions",
        )

    def test_feature_pipeline_consistent_train_serve(
        self, trained_model_and_pipeline: tuple[Any, ...], split_data: SplitResult
    ) -> None:
        """
        Pipeline fitted on train must transform single rows consistently.

        This catches training-serving skew — the most common production bug.
        """
        model, pipeline, _, _, _, _ = trained_model_and_pipeline

        # Transform a batch
        X_batch, _ = prepare_features(split_data.test.head(10), pipeline, fit=False)
        batch_proba = model.predict_proba(X_batch)

        # Transform each row individually (simulates serving)
        single_probas = []
        for _, row in split_data.test.head(10).iterrows():
            X_single, _ = prepare_features(pd.DataFrame([row]), pipeline, fit=False)
            single_probas.append(model.predict_proba(X_single)[0])

        np.testing.assert_allclose(
            batch_proba,
            single_probas,
            rtol=1e-5,
            err_msg="Batch vs single-row predictions differ — serving skew detected",
        )

    def test_evaluation_gate_runs_on_real_metrics(
        self, trained_model_and_pipeline: tuple[Any, ...], split_data: SplitResult
    ) -> None:
        """Gate runs without error on real model metrics."""
        model, pipeline, _, _, _, _ = trained_model_and_pipeline
        X_test, y_test = prepare_features(split_data.test, pipeline, fit=False)
        metrics = evaluate_model(model, X_test, y_test, dataset_name="test")
        result = run_evaluation_gate(metrics, champion_metrics=None)
        # Gate result must be one of the valid decisions
        assert result.decision in (GateDecision.PASS, GateDecision.FAIL)
        assert len(result.checks) > 0
