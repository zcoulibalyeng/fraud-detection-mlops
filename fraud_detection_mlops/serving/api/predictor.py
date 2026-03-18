# """
# Predictor — loads the model from MLflow registry and serves predictions.
#
# Loaded ONCE at application startup. All requests share the same instance.
# Thread-safe for read operations (predict_proba is stateless after load).
#
# This class is the bridge between the FastAPI layer and the model layer.
# It knows about MLflow and the feature pipeline.
# It does NOT know about HTTP, request validation, or response formatting.
# """
#
# from __future__ import annotations
#
# import os
# import time
# from pathlib import Path
#
# import mlflow
# import mlflow.sklearn
# import mlflow.xgboost
# import pandas as pd
# from loguru import logger
#
# from fraud_detection_mlops.configs.settings import get_settings
# from fraud_detection_mlops.training.features.feature_eng import (
#     ALL_FEATURES,
#     build_feature_pipeline,
# )
# from fraud_detection_mlops.training.models.xgb_model import XGBoostFraudModel
#
#
# class ModelNotLoadedError(Exception):
#     """Raised when predict() is called before the model is loaded."""
#
#
# class Predictor:
#     """
#     Manages model loading and inference for the serving API.
#
#     Parameters
#     ----------
#     model_name : str
#         Name of the registered model in MLflow registry.
#     model_stage : str
#         Registry stage to load from: "Production", "Staging", or "None".
#     threshold : float
#         Decision threshold for binary fraud classification.
#     """
#
#     def __init__(
#         self,
#         model_name: str = "fraud-detector",
#         model_stage: str = "Production",
#         threshold: float = 0.5,
#     ) -> None:
#         self.model_name = model_name
#         self.model_stage = model_stage
#         self.threshold = threshold
#
#         self._model: XGBoostFraudModel | None = None
#         self._pipeline = build_feature_pipeline()
#         self._model_version: str = "unloaded"
#         self._loaded: bool = False
#         self._load_time: float = 0.0
#
#
#     # def load(self) -> None:
#     #     """
#     #     Load model and pipeline from MLflow registry.
#     #     In Production, this fetches from S3/Artifact store via MLflow.
#     #     """
#     #     settings = get_settings()
#     #     mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
#     #
#     #     # Get target from Env Vars (set these in SageMaker/Docker)
#     #     model_name = os.getenv("MODEL_NAME", self.model_name)
#     #     model_alias = os.getenv("MODEL_ALIAS", "champion")
#     #
#     #     logger.info(f"Fetching model '{model_name}' with alias '{model_alias}'")
#     #
#     #     start = time.perf_counter()
#     #     os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "10")
#     #
#     #     try:
#     #         client = mlflow.tracking.MlflowClient()
#     #
#     #         # 1. Get version by alias (The MLflow 2.9+ way)
#     #         model_version = client.get_model_version_by_alias(model_name, model_alias)
#     #         run_id = model_version.run_id
#     #         self._model_version = f"v{model_version.version}-{run_id[:8]}"
#     #
#     #         # 2. Load XGBoost model directly from the run
#     #         xgb_underlying = mlflow.xgboost.load_model(f"runs:/{run_id}/model")
#     #
#     #         # Reconstruct our wrapper
#     #         self._model = XGBoostFraudModel.__new__(XGBoostFraudModel)
#     #         self._model._model = xgb_underlying
#     #         self._model._feature_names = ALL_FEATURES
#     #         self._model._best_iteration = 0  # Not strictly needed for inference
#     #
#     #         # 3. Load the EXACT fitted pipeline used in that run
#     #         pipeline_uri = f"runs:/{run_id}/feature_pipeline"
#     #         self._pipeline = mlflow.sklearn.load_model(pipeline_uri)
#     #
#     #         logger.success(
#     #             f"Production model {self._model_version} loaded from registry"
#     #         )
#     #
#     #     except Exception as exc:
#     #         logger.error(f"Registry load failed: {exc}. Falling back to local/mock.")
#     #         self._load_local_fallback()
#     #
#     #     self._load_time = time.perf_counter() - start
#     #     self._loaded = True
#     #     self._warmup()
#
#     def load(self) -> None:
#         """
#         Load model and pipeline.
#
#         Strategy:
#         1. If local model file exists, load it directly (fastest, no network)
#         2. Otherwise try MLflow registry
#         3. Fall back to mock if nothing works
#         """
#         start = time.perf_counter()
#
#         # Try local files first — works in Docker and local dev
#         local_model_paths = [
#             Path("models/model.joblib"),
#             Path("model.joblib"),
#             Path("fraud_detection_mlops/models/model.joblib"),
#         ]
#
#         for local_path in local_model_paths:
#             if local_path.exists():
#                 logger.info("Found local model at {}, loading directly", local_path)
#                 self._model = XGBoostFraudModel.load(local_path)
#                 self._model_version = "local-dev"
#                 self._load_local_pipeline()
#                 self._loaded = True
#                 load_duration = time.perf_counter() - start
#                 self._load_time = load_duration
#                 logger.info("Model loaded in {:.2f}s (version={})", load_duration, self._model_version)
#                 self._warmup()
#                 return
#
#         # No local model — try MLflow
#         settings = get_settings()
#         mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
#
#         logger.info(
#             "No local model found. Loading '{}' from MLflow at {}",
#             self.model_name,
#             settings.mlflow_tracking_uri,
#         )
#
#         try:
#             client = mlflow.tracking.MlflowClient()
#             versions = client.search_model_versions(f"name='{self.model_name}'")
#             if not versions:
#                 raise RuntimeError(f"No versions found for '{self.model_name}'")
#
#             latest = sorted(
#                 versions, key=lambda v: int(v.version), reverse=True
#             )[0]
#             run_id = latest.run_id
#             logger.info("Found model version={} run_id={}", latest.version, run_id[:8])
#
#             xgb_underlying = mlflow.xgboost.load_model(f"runs:/{run_id}/model")
#
#             self._model = XGBoostFraudModel.__new__(XGBoostFraudModel)
#             self._model._model = xgb_underlying
#             self._model._feature_names = ALL_FEATURES
#             self._model._best_iteration = 0
#             self._model._train_time_seconds = 0.0
#             self._model_version = f"v{latest.version}-{run_id[:8]}"
#
#             try:
#                 pipeline_uri = f"runs:/{run_id}/feature_pipeline"
#                 self._pipeline = mlflow.sklearn.load_model(pipeline_uri)
#                 logger.info("Loaded fitted feature pipeline from MLflow")
#             except Exception as pipe_exc:
#                 logger.warning("Could not load pipeline from MLflow: {}", pipe_exc)
#                 self._load_local_pipeline()
#
#         except Exception as exc:
#             logger.warning("MLflow load failed ({}), using mock", exc)
#             self._model_version = "mock"
#
#         load_duration = time.perf_counter() - start
#         self._load_time = load_duration
#         self._loaded = True
#
#         logger.info(
#             "Model loaded in {:.2f}s (version={})",
#             load_duration,
#             self._model_version,
#         )
#         self._warmup()
#
#     def _load_local_fallback(self) -> None:
#         """
#         Load model from local filesystem for development.
#
#         Searches multiple common paths for model.joblib.
#         """
#         search_paths = [
#             Path("models/model.joblib"),
#             Path("model.joblib"),
#             Path("fraud_detection_mlops/models/model.joblib"),
#         ]
#
#         for local_path in search_paths:
#             if local_path.exists():
#                 self._model = XGBoostFraudModel.load(local_path)
#                 self._model_version = "local-dev"
#                 logger.info("Loaded model from local fallback: {}", local_path)
#                 self._load_local_pipeline()
#                 return
#
#         logger.warning(
#             "No model found locally at any of: {}. "
#             "Predictor will return mock scores.",
#             [str(p) for p in search_paths],
#         )
#         self._model_version = "mock"
#
#
#     # def _load_local_pipeline(self) -> None:
#     #     """
#     #     Refactored: Removed 'fit on train.parquet' logic to prevent Docker
#     #     build failures. Only loads pre-fitted pipelines.
#     #     """
#     #     pipeline_paths = [
#     #         Path("models/feature_pipeline"),
#     #         Path("fraud_detection_mlops/models/feature_pipeline"),
#     #     ]
#     #
#     #     for pp in pipeline_paths:
#     #         if pp.exists():
#     #             try:
#     #                 self._pipeline = mlflow.sklearn.load_model(str(pp))
#     #                 logger.info("Loaded local fitted pipeline from: {}", pp)
#     #                 return
#     #             except Exception as exc:
#     #                 logger.warning("Failed to load local pipeline: {}", exc)
#     #
#     #     logger.error("No fitted pipeline found. Using unfitted default (INACCURATE).")
#
#     def _load_local_pipeline(self) -> None:
#         """
#         Load a pre-fitted pipeline, or fit one from training data.
#         """
#         # First try pre-saved pipeline directories
#         pipeline_paths = [
#             Path("models/feature_pipeline"),
#             Path("fraud_detection_mlops/models/feature_pipeline"),
#         ]
#
#         for pp in pipeline_paths:
#             if pp.exists():
#                 try:
#                     self._pipeline = mlflow.sklearn.load_model(str(pp))
#                     logger.info("Loaded local fitted pipeline from: {}", pp)
#                     return
#                 except Exception as exc:
#                     logger.warning("Failed to load local pipeline: {}", exc)
#
#         # No pre-saved pipeline — fit from training data
#         train_paths = [
#             Path("fraud_detection_mlops/data/processed/train.parquet"),
#             Path("data/processed/train.parquet"),
#         ]
#
#         for tp in train_paths:
#             if tp.exists():
#                 try:
#                     logger.info("Fitting pipeline from training data: {}", tp)
#                     train_df = pd.read_parquet(tp)
#                     self._pipeline = build_feature_pipeline()
#                     self._pipeline.fit(train_df[ALL_FEATURES])
#                     logger.info("Pipeline fitted on {:,} training rows", len(train_df))
#                     return
#                 except Exception as exc:
#                     logger.warning("Failed to fit pipeline from {}: {}", tp, exc)
#
#         logger.error("No fitted pipeline found and no training data. Predictions will be inaccurate.")
#
#     def _warmup(self) -> None:
#         """
#         Run a dummy prediction to warm up model inference.
#
#         XGBoost JIT-compiles prediction logic on the first call.
#         This warmup ensures that cost is paid at startup, not on
#         the first real user request.
#         """
#         try:
#             dummy = pd.DataFrame(
#                 {
#                     **{f"V{i}": [0.0] for i in range(1, 29)},
#                     "Amount": [1.0],
#                 }
#             )
#             _ = self._predict_from_df(dummy)
#             logger.info("Model warmup complete")
#         except Exception as exc:
#             logger.warning("Warmup failed (non-fatal): {}", exc)
#
#     def predict(self, features: dict[str, float]) -> tuple[float, bool]:
#         """
#         Run inference on a single transaction.
#
#         Parameters
#         ----------
#         features : dict[str, float]
#             Feature dictionary from PredictRequest.to_feature_dict().
#
#         Returns
#         -------
#         tuple[float, bool]
#             (fraud_probability, is_fraud) pair.
#
#         Raises
#         ------
#         ModelNotLoadedError
#             If called before load().
#         """
#         if not self._loaded:
#             raise ModelNotLoadedError(
#                 "Predictor.load() must be called before predict()"
#             )
#
#         df = pd.DataFrame([features])
#         fraud_proba = self._predict_from_df(df)
#         is_fraud = bool(fraud_proba >= self.threshold)
#         return float(fraud_proba), is_fraud
#
#     def _predict_from_df(self, df: pd.DataFrame) -> float:
#         """Internal: apply pipeline + model to a single-row DataFrame."""
#         if self._model is None or self._model_version == "mock":
#             return 0.05
#
#         X = self._pipeline.transform(df)
#         proba = self._model.predict_proba(X)
#         return float(proba[0])
#
#     @property
#     def is_loaded(self) -> bool:
#         return self._loaded
#
#     @property
#     def model_version(self) -> str:
#         return self._model_version


"""
Predictor — loads the model and serves predictions.

Loading strategy (in order):
  1. S3 — if ARTIFACT_BUCKET env var is set (production/SageMaker)
  2. Local files — if model.joblib exists on disk (Docker/dev)
  3. MLflow registry — if MLflow is reachable (dev with MLflow)
  4. Mock — returns 0.05 for all requests (fallback)

Loaded ONCE at application startup. All requests share the same instance.
Thread-safe for read operations (predict_proba is stateless after load).
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import joblib as _joblib
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
        Registry stage to load from.
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
        Load model and pipeline.

        Strategy:
        1. Download from S3 if ARTIFACT_BUCKET is set (production)
        2. Load from local files if they exist (Docker/dev)
        3. Try MLflow registry (dev with MLflow running)
        4. Fall back to mock
        """
        start = time.perf_counter()

        # ── Strategy 1: S3 (production) ────────────────────────────────
        artifact_bucket = os.getenv("ARTIFACT_BUCKET")
        artifact_prefix = os.getenv("ARTIFACT_PREFIX", "artifacts/latest")

        if artifact_bucket:
            logger.info(
                "Production mode: downloading from s3://{}/{}",
                artifact_bucket,
                artifact_prefix,
            )
            try:
                self._download_from_s3(artifact_bucket, artifact_prefix)
                self._finish_load(start)
                return
            except Exception as exc:
                logger.error("S3 download failed: {}. Trying next strategy.", exc)

        # ── Strategy 2: Local files (Docker/dev) ──────────────────────
        local_model_paths = [
            Path("models/model.joblib"),
            Path("model.joblib"),
            Path("fraud_detection_mlops/models/model.joblib"),
        ]

        for local_path in local_model_paths:
            if local_path.exists():
                logger.info("Found local model at {}", local_path)
                self._model = XGBoostFraudModel.load(local_path)
                self._model_version = "local-dev"
                self._load_local_pipeline()
                self._finish_load(start)
                return

        # ── Strategy 3: MLflow registry ────────────────────────────────
        settings = get_settings()
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "10")

        logger.info(
            "No local model. Trying MLflow at {}",
            settings.mlflow_tracking_uri,
        )

        try:
            client = mlflow.tracking.MlflowClient()
            versions = client.search_model_versions(f"name='{self.model_name}'")
            if not versions:
                raise RuntimeError(f"No versions found for '{self.model_name}'")

            latest = sorted(versions, key=lambda v: int(v.version), reverse=True)[0]
            run_id = latest.run_id
            logger.info(
                "Found model version={} run_id={}",
                latest.version,
                run_id[:8],
            )

            xgb_underlying = mlflow.xgboost.load_model(f"runs:/{run_id}/model")

            self._model = XGBoostFraudModel.__new__(XGBoostFraudModel)
            self._model._model = xgb_underlying
            self._model._feature_names = ALL_FEATURES
            self._model._best_iteration = 0
            self._model._train_time_seconds = 0.0
            self._model_version = f"v{latest.version}-{run_id[:8]}"

            try:
                pipeline_uri = f"runs:/{run_id}/feature_pipeline"
                self._pipeline = mlflow.sklearn.load_model(pipeline_uri)
                logger.info("Loaded fitted pipeline from MLflow")
            except Exception as pipe_exc:
                logger.warning("Could not load pipeline from MLflow: {}", pipe_exc)
                self._load_local_pipeline()

            self._finish_load(start)
            return

        except Exception as exc:
            logger.warning("MLflow load failed: {}", exc)

        # ── Strategy 4: Mock ──────────────────────────────────────────
        logger.warning("All loading strategies failed. Using mock predictor.")
        self._model_version = "mock"
        self._finish_load(start)

    def _finish_load(self, start_time: float) -> None:
        """Common cleanup after any loading strategy succeeds."""
        self._load_time = time.perf_counter() - start_time
        self._loaded = True
        logger.info(
            "Model loaded in {:.2f}s (version={})",
            self._load_time,
            self._model_version,
        )
        self._warmup()

    def _download_from_s3(self, bucket: str, prefix: str) -> None:
        """Download model and pipeline from S3."""
        import tempfile

        import boto3

        s3 = boto3.client("s3")
        tmp_dir = Path(tempfile.mkdtemp())

        # Download model
        model_key = f"{prefix}/model.joblib"
        model_path = tmp_dir / "model.joblib"
        logger.info("Downloading s3://{}/{}", bucket, model_key)
        s3.download_file(bucket, model_key, str(model_path))
        self._model = XGBoostFraudModel.load(model_path)

        # Download pipeline
        pipeline_key = f"{prefix}/feature_pipeline.joblib"
        pipeline_path = tmp_dir / "feature_pipeline.joblib"
        logger.info("Downloading s3://{}/{}", bucket, pipeline_key)
        s3.download_file(bucket, pipeline_key, str(pipeline_path))
        self._pipeline = _joblib.load(pipeline_path)

        self._model_version = f"s3-{prefix.split('/')[-1]}"
        logger.success("Model and pipeline downloaded from S3")

    def _load_local_pipeline(self) -> None:
        """
        Load a pre-fitted pipeline from disk, or fit one from training data.
        """
        # First try pre-saved pipeline files
        pipeline_paths = [
            Path("models/feature_pipeline.joblib"),
            Path("models/feature_pipeline"),
            Path("fraud_detection_mlops/models/feature_pipeline"),
        ]

        for pp in pipeline_paths:
            if pp.exists():
                try:
                    if pp.suffix == ".joblib":
                        self._pipeline = _joblib.load(pp)
                    else:
                        self._pipeline = mlflow.sklearn.load_model(str(pp))
                    logger.info("Loaded fitted pipeline from: {}", pp)
                    return
                except Exception as exc:
                    logger.warning("Failed to load pipeline from {}: {}", pp, exc)

        # Fit from training data as last resort
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

        logger.error(
            "No fitted pipeline found and no training data. "
            "Predictions will be inaccurate."
        )

    def _warmup(self) -> None:
        """Run a dummy prediction to warm up model inference."""
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
