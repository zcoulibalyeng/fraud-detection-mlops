"""Tests for model evaluation and evaluation gate."""

from __future__ import annotations

import numpy as np
import pytest

from fraud_detection_mlops.training.evaluate import (
    ModelMetrics,
    compute_metrics,
    find_optimal_threshold,
)
from fraud_detection_mlops.training.evaluation_gate import (
    GateDecision,
    run_evaluation_gate,
)


def _perfect_metrics() -> ModelMetrics:
    return ModelMetrics(
        auc_roc=0.999,
        auc_pr=0.990,
        f1=0.950,
        precision=0.960,
        recall=0.940,
        n_samples=50_000,
        n_positives=100,
    )


def _poor_metrics() -> ModelMetrics:
    return ModelMetrics(
        auc_roc=0.55,
        auc_pr=0.30,
        f1=0.40,
        precision=0.50,
        recall=0.33,
        n_samples=50_000,
        n_positives=100,
    )


class TestComputeMetrics:
    def test_perfect_classifier(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.01, 0.02, 0.99, 0.98])
        metrics = compute_metrics(y_true, y_proba)
        assert metrics.f1 == pytest.approx(1.0)
        assert metrics.precision == pytest.approx(1.0)
        assert metrics.recall == pytest.approx(1.0)

    def test_all_wrong_classifier(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.99, 0.98, 0.01, 0.02])
        metrics = compute_metrics(y_true, y_proba)
        assert metrics.f1 == pytest.approx(0.0)

    def test_sample_counts_correct(self) -> None:
        y_true = np.array([0] * 95 + [1] * 5)
        y_proba = np.random.default_rng(0).uniform(0, 1, 100)
        metrics = compute_metrics(y_true, y_proba)
        assert metrics.n_samples == 100
        assert metrics.n_positives == 5

    def test_to_dict_has_all_keys(self) -> None:
        y_true = np.array([0, 1, 0, 1])
        y_proba = np.array([0.1, 0.9, 0.2, 0.8])
        metrics = compute_metrics(y_true, y_proba)
        d = metrics.to_dict()
        for key in ["auc_roc", "auc_pr", "f1", "precision", "recall"]:
            assert key in d


class TestFindOptimalThreshold:
    def test_returns_float_in_valid_range(self) -> None:
        rng = np.random.default_rng(42)
        y_true = rng.choice([0, 1], 500, p=[0.98, 0.02])
        y_proba = rng.uniform(0, 1, 500)
        threshold = find_optimal_threshold(y_true, y_proba, metric="f1")
        assert 0.0 < threshold < 1.0

    def test_invalid_metric_raises(self) -> None:
        y_true = np.array([0, 1])
        y_proba = np.array([0.1, 0.9])
        with pytest.raises(ValueError, match="Unknown metric"):
            find_optimal_threshold(y_true, y_proba, metric="bad_metric")


class TestEvaluationGate:
    def test_perfect_model_passes(self) -> None:
        result = run_evaluation_gate(_perfect_metrics())
        assert result.passed
        assert result.decision == GateDecision.PASS

    def test_poor_model_fails(self) -> None:
        result = run_evaluation_gate(_poor_metrics())
        assert not result.passed
        assert result.decision == GateDecision.FAIL

    def test_challenger_beats_champion_passes(self) -> None:
        champion = _perfect_metrics()
        challenger = ModelMetrics(
            auc_roc=0.999,
            auc_pr=0.996,
            f1=0.960,
            precision=0.965,
            recall=0.955,
            n_samples=50_000,
            n_positives=100,
        )
        result = run_evaluation_gate(challenger, champion)
        assert result.passed

    def test_challenger_does_not_beat_champion_fails(self) -> None:
        champion = _perfect_metrics()
        # Same metrics as champion — delta = 0, below min_delta
        result = run_evaluation_gate(champion, champion)
        assert not result.passed

    def test_failed_checks_listed(self) -> None:
        result = run_evaluation_gate(_poor_metrics())
        assert len(result.failed_checks) > 0

    def test_no_champion_skips_delta_checks(self) -> None:
        """First deployment: no champion means no delta checks."""
        result = run_evaluation_gate(_perfect_metrics(), champion_metrics=None)
        check_names = [c.check_name for c in result.checks]
        assert not any("delta" in name for name in check_names)
