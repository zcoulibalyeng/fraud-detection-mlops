"""
Abstract base class for all models in this system.

Every model (XGBoost, LightGBM, neural net) must implement this interface.
The training script, evaluation module, and serving layer all depend only
on this interface — never on a specific model implementation.

This is the open/closed principle: open for extension (add new models),
closed for modification (training/serving code never changes).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np


class BaseModel(ABC):
    """Abstract interface every model must implement."""

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> BaseModel:
        """Train the model. Returns self for chaining."""
        ...

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return probability of the positive class.

        Returns
        -------
        np.ndarray
            1D array of fraud probabilities in [0, 1].
        """
        ...

    @abstractmethod
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return binary predictions using the given threshold."""
        ...

    @abstractmethod
    def get_feature_importance(self) -> dict[str, float]:
        """Return feature name → importance score mapping."""
        ...

    @abstractmethod
    def save(self, path: Path) -> None:
        """Serialize model to disk."""
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> BaseModel:
        """Deserialize model from disk."""
        ...

    @property
    @abstractmethod
    def params(self) -> dict[str, Any]:
        """Return the model hyperparameters as a flat dict."""
        ...
