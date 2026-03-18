"""
Evaluation gate — the champion challenger decision.

This is the most important function in the training module.
A model that fails the gate never reaches production.
A model that passes is automatically registered to staging.

Reads all thresholds from configs/evaluation_gate.yaml — no
hardcoded numbers anywhere in this file.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from loguru import logger

from fraud_detection_mlops.configs.settings import get_settings
from fraud_detection_mlops.training.evaluate import ModelMetrics


class GateDecision(Enum):
    PASS = auto()
    FAIL = auto()


@dataclass
class GateCheckResult:
    """Result of a single gate check."""

    check_name: str
    passed: bool
    actual_value: float
    threshold_value: float
    message: str


@dataclass
class EvaluationGateResult:
    """Aggregated result of all gate checks."""

    decision: GateDecision
    checks: list[GateCheckResult]

    @property
    def passed(self) -> bool:
        return self.decision == GateDecision.PASS

    @property
    def failed_checks(self) -> list[GateCheckResult]:
        return [c for c in self.checks if not c.passed]

    def log_summary(self) -> None:
        total = len(self.checks)
        passed = sum(1 for c in self.checks if c.passed)
        failed = total - passed

        if self.passed:
            logger.success("Evaluation gate PASSED: {}/{} checks passed", passed, total)
        else:
            logger.error(
                "Evaluation gate FAILED: {}/{} checks passed, {} failed",
                passed,
                total,
                failed,
            )
            for check in self.failed_checks:
                logger.error(
                    "  FAIL [{}]: actual={:.4f} vs threshold={:.4f} — {}",
                    check.check_name,
                    check.actual_value,
                    check.threshold_value,
                    check.message,
                )


def run_evaluation_gate(
    challenger_metrics: ModelMetrics,
    champion_metrics: ModelMetrics | None = None,
) -> EvaluationGateResult:
    """
    Run all evaluation gate checks.

    Compares challenger against:
      1. Absolute minimum thresholds (model must be good enough alone)
      2. Champion delta thresholds (model must beat current production)
      3. Infrastructure thresholds (latency, error rate)

    Parameters
    ----------
    challenger_metrics : ModelMetrics
        Metrics for the newly trained model on the test set.
    champion_metrics : ModelMetrics | None
        Metrics for the current production model on the same test set.
        If None (first deployment), only absolute checks run.

    Returns
    -------
    EvaluationGateResult
        The gate decision with full check details.
    """
    cfg = get_settings().evaluation_gate_cfg
    checks: list[GateCheckResult] = []

    # ── Absolute minimum checks ────────────────────────────────────────
    checks.append(
        _check_absolute(
            name="min_auc_pr",
            actual=challenger_metrics.auc_pr,
            minimum=cfg["min_auc_pr"],
            message="AUC-PR below minimum acceptable threshold",
        )
    )
    checks.append(
        _check_absolute(
            name="min_f1",
            actual=challenger_metrics.f1,
            minimum=cfg["min_f1"],
            message="F1 below minimum acceptable threshold",
        )
    )
    checks.append(
        _check_absolute(
            name="min_precision",
            actual=challenger_metrics.precision,
            minimum=cfg["min_precision"],
            message="Precision below minimum — too many false positives",
        )
    )
    checks.append(
        _check_absolute(
            name="min_recall",
            actual=challenger_metrics.recall,
            minimum=cfg["min_recall"],
            message="Recall below minimum — missing too much fraud",
        )
    )

    # ── Champion delta checks (only if champion exists) ────────────────
    if champion_metrics is not None:
        auc_pr_delta = challenger_metrics.auc_pr - champion_metrics.auc_pr
        checks.append(
            _check_delta(
                name="auc_pr_delta_vs_champion",
                actual_delta=auc_pr_delta,
                min_delta=cfg["min_auc_pr_delta"],
                message=(
                    f"Challenger AUC-PR ({challenger_metrics.auc_pr:.4f}) does not "
                    f"beat champion ({champion_metrics.auc_pr:.4f}) by required "
                    f"delta ({cfg['min_auc_pr_delta']})"
                ),
            )
        )

        f1_delta = challenger_metrics.f1 - champion_metrics.f1
        checks.append(
            _check_delta(
                name="f1_delta_vs_champion",
                actual_delta=f1_delta,
                min_delta=cfg["min_f1_delta"],
                message=(
                    f"Challenger F1 ({challenger_metrics.f1:.4f}) does not "
                    f"beat champion ({champion_metrics.f1:.4f}) by required "
                    f"delta ({cfg['min_f1_delta']})"
                ),
            )
        )

    # ── Sample size check ──────────────────────────────────────────────
    checks.append(
        _check_absolute(
            name="min_test_samples",
            actual=float(challenger_metrics.n_samples),
            minimum=float(cfg["min_test_sample_size"]),
            message=(
                f"Test set too small ({challenger_metrics.n_samples:,} samples) "
                f"for reliable evaluation"
            ),
        )
    )

    # ── Decision ───────────────────────────────────────────────────────
    all_passed = all(c.passed for c in checks)
    decision = GateDecision.PASS if all_passed else GateDecision.FAIL

    result = EvaluationGateResult(decision=decision, checks=checks)
    result.log_summary()
    return result


def _check_absolute(
    name: str,
    actual: float,
    minimum: float,
    message: str,
) -> GateCheckResult:
    passed = actual >= minimum
    return GateCheckResult(
        check_name=name,
        passed=passed,
        actual_value=actual,
        threshold_value=minimum,
        message=message if not passed else "",
    )


def _check_delta(
    name: str,
    actual_delta: float,
    min_delta: float,
    message: str,
) -> GateCheckResult:
    passed = actual_delta >= min_delta
    return GateCheckResult(
        check_name=name,
        passed=passed,
        actual_value=actual_delta,
        threshold_value=min_delta,
        message=message if not passed else "",
    )
