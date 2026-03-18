"""
Model evaluation — metrics, slice evaluation, and fairness checks.

Separated from training so the evaluation gate can load any model
from the registry and evaluate it without re-running training.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from fraud_detection_mlops.training.models.base_model import BaseModel


@dataclass
class ModelMetrics:
    """All evaluation metrics for one model on one dataset."""

    auc_roc: float
    auc_pr: float
    f1: float
    precision: float
    recall: float
    threshold: float = 0.5
    n_samples: int = 0
    n_positives: int = 0
    slice_metrics: dict[str, ModelMetrics] = field(default_factory=dict)

    def to_dict(self) -> dict[str, float]:
        """Flat dict for MLflow logging — no nested slices."""
        return {
            "auc_roc": self.auc_roc,
            "auc_pr": self.auc_pr,
            "f1": self.f1,
            "precision": self.precision,
            "recall": self.recall,
            "threshold": self.threshold,
            "n_samples": float(self.n_samples),
            "n_positives": float(self.n_positives),
        }

    def log_summary(self, prefix: str = "") -> None:
        label = f"[{prefix}] " if prefix else ""
        logger.info(
            "{}AUC-ROC={:.4f} | AUC-PR={:.4f} | F1={:.4f} | "
            "Precision={:.4f} | Recall={:.4f} | n={:,} | fraud={:,}",
            label,
            self.auc_roc,
            self.auc_pr,
            self.f1,
            self.precision,
            self.recall,
            self.n_samples,
            self.n_positives,
        )


def compute_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> ModelMetrics:
    """
    Compute the full metric set for a set of predictions.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth binary labels.
    y_proba : np.ndarray
        Predicted fraud probabilities.
    threshold : float
        Decision threshold for binary predictions.

    Returns
    -------
    ModelMetrics
        Complete metrics for this prediction set.
    """
    y_pred = (y_proba >= threshold).astype(int)

    return ModelMetrics(
        auc_roc=float(roc_auc_score(y_true, y_proba)),
        auc_pr=float(average_precision_score(y_true, y_proba)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        threshold=threshold,
        n_samples=int(len(y_true)),
        n_positives=int(y_true.sum()),
    )


def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    metric: str = "f1",
) -> float:
    """
    Find the decision threshold that maximizes the given metric.

    Searches over 100 thresholds between 0.01 and 0.99.
    Use this once on the validation set — never on the test set.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels.
    y_proba : np.ndarray
        Predicted probabilities.
    metric : str
        One of "f1", "precision", "recall".

    Returns
    -------
    float
        The optimal threshold.
    """
    best_threshold = 0.5
    best_score = 0.0

    for threshold in np.linspace(0.01, 0.99, 100):
        y_pred = (y_proba >= threshold).astype(int)
        if metric == "f1":
            score = float(f1_score(y_true, y_pred, zero_division=0))
        elif metric == "precision":
            score = float(precision_score(y_true, y_pred, zero_division=0))
        elif metric == "recall":
            score = float(recall_score(y_true, y_pred, zero_division=0))
        else:
            raise ValueError(f"Unknown metric: {metric}")

        if score > best_score:
            best_score = score
            best_threshold = float(threshold)

    logger.info(
        "Optimal threshold for {}: {:.3f} (score={:.4f})",
        metric,
        best_threshold,
        best_score,
    )
    return best_threshold


def evaluate_model(
    model: BaseModel,
    X: np.ndarray,
    y: np.ndarray,
    threshold: float = 0.5,
    dataset_name: str = "eval",
) -> ModelMetrics:
    """
    Evaluate a model and log results.

    Parameters
    ----------
    model : BaseModel
        Any fitted model implementing BaseModel interface.
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Ground truth labels.
    threshold : float
        Decision threshold.
    dataset_name : str
        Label for log messages.

    Returns
    -------
    ModelMetrics
        The computed metrics.
    """
    y_proba = model.predict_proba(X)
    metrics = compute_metrics(y_true=y, y_proba=y_proba, threshold=threshold)
    metrics.log_summary(prefix=dataset_name)
    return metrics


def evaluate_slices(
    model: BaseModel,
    df: pd.DataFrame,
    X: np.ndarray,
    y: np.ndarray,
    slice_columns: list[str],
    threshold: float = 0.5,
) -> dict[str, ModelMetrics]:
    """
    Evaluate model performance on each slice of specified columns.

    Used in the fairness gate — ensures no population segment is
    significantly underserved by the new model.

    Parameters
    ----------
    model : BaseModel
        Fitted model to evaluate.
    df : pd.DataFrame
        Full DataFrame (needed for slice column values).
    X : np.ndarray
        Feature matrix (same row order as df).
    y : np.ndarray
        Labels (same row order as df).
    slice_columns : list[str]
        Columns to slice on (e.g. ["hour_of_day", "amount_bin"]).
    threshold : float
        Decision threshold.

    Returns
    -------
    dict[str, ModelMetrics]
        Mapping of "column=value" → metrics for that slice.
    """
    slice_results: dict[str, ModelMetrics] = {}
    y_proba = model.predict_proba(X)

    for col in slice_columns:
        if col not in df.columns:
            logger.warning("Slice column '{}' not found — skipping", col)
            continue

        for value in df[col].unique():
            mask = (df[col] == value).values
            if mask.sum() < 100:
                # Skip slices too small for reliable metrics
                continue

            slice_key = f"{col}={value}"
            slice_metrics = compute_metrics(
                y_true=y[mask],
                y_proba=y_proba[mask],
                threshold=threshold,
            )
            slice_results[slice_key] = slice_metrics
            logger.debug(
                "Slice {}: AUC-PR={:.4f}, n={:,}",
                slice_key,
                slice_metrics.auc_pr,
                slice_metrics.n_samples,
            )

    logger.info("Slice evaluation complete: {} slices evaluated", len(slice_results))
    return slice_results
