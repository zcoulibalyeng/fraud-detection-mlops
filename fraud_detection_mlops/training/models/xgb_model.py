"""
XGBoost implementation of BaseModel for fraud detection.

Wraps xgboost.XGBClassifier with:
  - Early stopping on validation AUC-PR
  - Scale pos weight for class imbalance
  - Feature importance extraction
  - Clean save/load via joblib
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from loguru import logger
from xgboost import XGBClassifier

from fraud_detection_mlops.training.models.base_model import BaseModel


class XGBoostFraudModel(BaseModel):
    """
    XGBoost binary classifier for fraud detection.

    Parameters
    ----------
    n_estimators : int
        Maximum number of boosting rounds.
    max_depth : int
        Maximum tree depth.
    learning_rate : float
        Step size shrinkage.
    subsample : float
        Fraction of samples used per tree.
    colsample_bytree : float
        Fraction of features used per tree.
    scale_pos_weight : float
        Weight for positive class — use neg_count/pos_count for imbalanced data.
    early_stopping_rounds : int
        Stop if validation metric doesn't improve for this many rounds.
    random_seed : int
        Reproducibility seed.
    """

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        scale_pos_weight: float = 578.0,
        early_stopping_rounds: int = 20,
        random_seed: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.scale_pos_weight = scale_pos_weight
        self.early_stopping_rounds = early_stopping_rounds
        self.random_seed = random_seed

        self._model: XGBClassifier | None = None
        self._feature_names: list[str] = []
        self._best_iteration: int = 0
        self._train_time_seconds: float = 0.0

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> XGBoostFraudModel:
        """
        Train the XGBoost model with early stopping on validation data.

        Parameters
        ----------
        X_train, y_train : np.ndarray
            Training features and labels.
        X_val, y_val : np.ndarray
            Validation features and labels for early stopping.
        feature_names : list[str], optional
            Names for features — used in importance plots.
        """
        logger.info(
            "Training XGBoost: train={:,} samples, val={:,} samples, "
            "fraud_rate_train={:.4%}, fraud_rate_val={:.4%}",
            len(X_train),
            len(X_val),
            y_train.mean(),
            y_val.mean(),
        )

        self._feature_names = feature_names or [
            f"f{i}" for i in range(X_train.shape[1])
        ]

        self._model = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            scale_pos_weight=self.scale_pos_weight,
            early_stopping_rounds=self.early_stopping_rounds,
            eval_metric="aucpr",
            random_state=self.random_seed,
            n_jobs=-1,
            verbosity=0,
        )

        start = time.time()
        self._model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        self._train_time_seconds = time.time() - start
        self._best_iteration = self._model.best_iteration

        logger.success(
            "Training complete: best_iteration={}, time={:.1f}s",
            self._best_iteration,
            self._train_time_seconds,
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return fraud probability for each sample."""
        self._assert_fitted()
        return self._model.predict_proba(X)[:, 1]  # type: ignore[union-attr]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return binary predictions using the given threshold."""
        return (self.predict_proba(X) >= threshold).astype(int)

    def get_feature_importance(self) -> dict[str, float]:
        """Return feature name → importance (gain) mapping."""
        self._assert_fitted()
        scores = self._model.get_booster().get_score(importance_type="gain")  # type: ignore[union-attr]
        # Map internal feature names (f0, f1, ...) back to real names
        importance = {}
        for i, name in enumerate(self._feature_names):
            internal = f"f{i}"
            importance[name] = float(scores.get(internal, 0.0))
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    def save(self, path: Path) -> None:
        """Save model to disk using joblib."""
        self._assert_fitted()
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("Model saved to {}", path)

    @classmethod
    def load(cls, path: Path) -> XGBoostFraudModel:
        """Load model from disk."""
        model = joblib.load(path)
        if not isinstance(model, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(model)}")
        logger.info("Model loaded from {}", path)
        return model

    @property
    def params(self) -> dict[str, Any]:
        return {
            "algorithm": "xgboost",
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "scale_pos_weight": self.scale_pos_weight,
            "early_stopping_rounds": self.early_stopping_rounds,
            "random_seed": self.random_seed,
            "best_iteration": self._best_iteration,
            "train_time_seconds": self._train_time_seconds,
        }

    def _assert_fitted(self) -> None:
        if self._model is None:
            raise RuntimeError("Model has not been trained yet. Call fit() first.")
