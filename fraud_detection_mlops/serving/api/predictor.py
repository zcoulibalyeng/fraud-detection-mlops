"""
Predictor — loads the model from MLflow registry and serves predictions.

Loaded ONCE at application startup. All requests share the same instance.
Thread-safe for read operations (predict_proba is stateless after load).

This class is the bridge between the FastAPI layer and the model layer.
It knows about MLflow and the feature pipeline.
It does NOT know about HTTP, request validation, or response formatting.
"""

from __future__ import annotations

import time
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from loguru import logger

from fraud_detection_mlops.configs.settings import get_settings
from fraud_detection_mlops.training.features.feature_eng import build_feature_pipeline
from fraud_detection_mlops.training.models.xgb_model import XGBoostFraudModel


class ModelNotLoadedError(Exception):
    """Raised when predict() is called before the model is loaded."""


class Predictor:
    """
    Manages model loading and inference for the serving API.

    Parameters
    ----------
    model_name : str
        Name of the registered model in MLflow registry.
    model_stage : str
        Registry stage to load from: "Production", "Staging", or "None".
    threshold : float
        Decision threshold for binary fraud classification.
    """

    def __init__(
        self,
        model_name: str = "fraud-detector",
        model_stage: str = "Production",
        threshold: float = 0.5,
    ) -> None:
        self.model_name = model_name
        self.model_stage = model_stage
        self.threshold = threshold

        self._model: XGBoostFraudModel | None = None
        self._pipeline = build_feature_pipeline()
        self._model_version: str = "unloaded"
        self._loaded: bool = False
        self._load_time: float = 0.0

    def load(self) -> None:
        """
        Load model and pipeline from MLflow registry.

        Called once at application startup via FastAPI lifespan.
        Runs a warmup prediction after loading to eliminate cold-start
        latency on the first real request.
        """
        settings = get_settings()
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

        logger.info(
            "Loading model '{}' stage='{}' from MLflow at {}",
            self.model_name,
            self.model_stage,
            settings.mlflow_tracking_uri,
        )

        start = time.perf_counter()

        try:
            model_uri = f"models:/{self.model_name}/{self.model_stage}"
            # Load the full MLflow model (includes feature pipeline)
            loaded = mlflow.pyfunc.load_model(model_uri)
            self._model_version = (
                loaded.metadata.run_id[:8] if loaded.metadata else "unknown"
            )

            # Unwrap to our XGBoostFraudModel for typed access
            self._model = loaded.unwrap_python_model()  # type: ignore[assignment]

            # Load fitted pipeline artifact from the same run
            pipeline_uri = f"{model_uri}/feature_pipeline"
            try:
                self._pipeline = mlflow.sklearn.load_model(pipeline_uri)
                logger.info("Loaded feature pipeline from MLflow")
            except Exception:
                logger.warning(
                    "Feature pipeline not found in MLflow — "
                    "using default unfitted pipeline"
                )

        except Exception as exc:
            # In development, fall back to loading from local path
            logger.warning("MLflow load failed ({}), attempting local fallback", exc)
            self._load_local_fallback()

        load_duration = time.perf_counter() - start
        self._load_time = load_duration
        self._loaded = True

        logger.info(
            "Model loaded in {:.2f}s (version={})",
            load_duration,
            self._model_version,
        )

        # Warmup — eliminates first-request cold start
        self._warmup()

    def _load_local_fallback(self) -> None:
        """
        Load model from local filesystem for development.

        Looks for model.joblib in the project root.
        This path is never used in production — only local dev.
        """
        local_path = Path("model.joblib")
        if local_path.exists():
            self._model = XGBoostFraudModel.load(local_path)
            self._model_version = "local-dev"
            logger.info("Loaded model from local fallback: {}", local_path)
        else:
            logger.warning(
                "No model found locally. Predictor will return mock scores. "
                "Train a model first: make train"
            )
            self._model_version = "mock"

    def _warmup(self) -> None:
        """
        Run a dummy prediction to warm up model inference.

        XGBoost JIT-compiles prediction logic on the first call.
        This warmup ensures that cost is paid at startup, not on
        the first real user request.
        """
        try:
            dummy = pd.DataFrame(
                {
                    **{f"V{i}": [0.0] for i in range(1, 29)},
                    "Amount": [1.0],
                }
            )
            _ = self._predict_from_df(dummy)
            logger.info("Model warmup complete")
        except Exception as exc:
            logger.warning("Warmup failed (non-fatal): {}", exc)

    def predict(self, features: dict[str, float]) -> tuple[float, bool]:
        """
        Run inference on a single transaction.

        Parameters
        ----------
        features : dict[str, float]
            Feature dictionary from PredictRequest.to_feature_dict().

        Returns
        -------
        tuple[float, bool]
            (fraud_probability, is_fraud) pair.

        Raises
        ------
        ModelNotLoadedError
            If called before load().
        """
        if not self._loaded:
            raise ModelNotLoadedError(
                "Predictor.load() must be called before predict()"
            )

        df = pd.DataFrame([features])
        fraud_proba = self._predict_from_df(df)
        is_fraud = bool(fraud_proba >= self.threshold)
        return float(fraud_proba), is_fraud

    def _predict_from_df(self, df: pd.DataFrame) -> float:
        """Internal: apply pipeline + model to a single-row DataFrame."""
        if self._model is None or self._model_version == "mock":
            # Development mock — returns low fraud probability
            return 0.05

        X = self._pipeline.transform(df)
        proba = self._model.predict_proba(X)
        return float(proba[0])

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def model_version(self) -> str:
        return self._model_version
