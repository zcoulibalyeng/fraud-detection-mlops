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

import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
from loguru import logger

from fraud_detection_mlops.configs.settings import get_settings
from fraud_detection_mlops.training.features.feature_eng import (
    ALL_FEATURES,
    build_feature_pipeline,
)
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

        Falls back to local files if MLflow is unavailable.
        Called once at application startup via FastAPI lifespan.
        Runs a warmup prediction after loading to eliminate cold-start
        latency on the first real request.
        """
        settings = get_settings()
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

        logger.info(
            "Loading model '{}' from MLflow at {}",
            self.model_name,
            settings.mlflow_tracking_uri,
        )

        start = time.perf_counter()

        import os

        os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "5")

        try:
            client = mlflow.tracking.MlflowClient()

            # Get the latest version's run ID directly
            versions = client.search_model_versions(f"name='{self.model_name}'")
            if not versions:
                raise RuntimeError(f"No versions found for '{self.model_name}'")

            latest = sorted(versions, key=lambda v: int(v.version), reverse=True)[0]
            run_id = latest.run_id
            logger.info("Found model version={} run_id={}", latest.version, run_id[:8])

            # Load XGBoost model using run URI
            xgb_underlying = mlflow.xgboost.load_model(f"runs:/{run_id}/model")

            # Wrap in our XGBoostFraudModel
            self._model = XGBoostFraudModel.__new__(XGBoostFraudModel)
            self._model._model = xgb_underlying
            self._model._feature_names = ALL_FEATURES
            self._model._best_iteration = 0
            self._model._train_time_seconds = 0.0
            self._model_version = f"v{latest.version}-{run_id[:8]}"

            # Load the fitted feature pipeline from the same run
            try:
                pipeline_uri = f"runs:/{run_id}/feature_pipeline"
                self._pipeline = mlflow.sklearn.load_model(pipeline_uri)
                logger.info("Loaded fitted feature pipeline from MLflow")
            except Exception as pipe_exc:
                logger.warning("Could not load pipeline from MLflow: {}", pipe_exc)
                self._load_local_pipeline()

        except Exception as exc:
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

        Searches multiple common paths for model.joblib.
        """
        search_paths = [
            Path("models/model.joblib"),
            Path("model.joblib"),
            Path("fraud_detection_mlops/models/model.joblib"),
        ]

        for local_path in search_paths:
            if local_path.exists():
                self._model = XGBoostFraudModel.load(local_path)
                self._model_version = "local-dev"
                logger.info("Loaded model from local fallback: {}", local_path)
                self._load_local_pipeline()
                return

        logger.warning(
            "No model found locally at any of: {}. "
            "Predictor will return mock scores.",
            [str(p) for p in search_paths],
        )
        self._model_version = "mock"

    def _load_local_pipeline(self) -> None:
        """
        Load the fitted feature pipeline from local filesystem.

        Searches for the pipeline saved by MLflow or as a standalone file.
        If not found, fits a new pipeline on dummy data as last resort.
        """
        pipeline_paths = [
            Path("models/feature_pipeline"),
            Path("models/feature_pipeline/model.pkl"),
            Path("fraud_detection_mlops/models/feature_pipeline"),
        ]

        for pp in pipeline_paths:
            if pp.exists():
                try:
                    if pp.is_dir():
                        self._pipeline = mlflow.sklearn.load_model(str(pp))
                    else:
                        self._pipeline = joblib.load(pp)
                    logger.info("Loaded fitted pipeline from: {}", pp)
                    return
                except Exception as exc:
                    logger.warning("Failed to load pipeline from {}: {}", pp, exc)

        # Last resort: fit pipeline on the training data if available
        train_paths = [
            Path("fraud_detection_mlops/data/processed/train.parquet"),
            Path("data/processed/train.parquet"),
        ]

        for tp in train_paths:
            if tp.exists():
                try:
                    logger.info("Fitting pipeline from training data: {}", tp)
                    train_df = pd.read_parquet(tp)
                    self._pipeline = build_feature_pipeline()
                    self._pipeline.fit(train_df[ALL_FEATURES])
                    logger.info("Pipeline fitted on {:,} training rows", len(train_df))
                    return
                except Exception as exc:
                    logger.warning("Failed to fit pipeline from {}: {}", tp, exc)

        logger.warning(
            "No fitted pipeline found and no training data available. "
            "Using unfitted pipeline — predictions will be incorrect."
        )

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
