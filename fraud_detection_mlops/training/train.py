"""
Training entry point — the script SageMaker and Airflow call.

Orchestrates the full training pipeline:
  1. Load data from S3 (or local)
  2. Build and fit feature pipeline
  3. Train XGBoost model with MLflow tracking
  4. Evaluate on val + test sets
  5. Run evaluation gate vs champion
  6. Register to MLflow registry if gate passes

Single responsibility: orchestration only.
All logic lives in the modules this imports.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, cast

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
import typer
from loguru import logger

from fraud_detection_mlops.configs.settings import get_settings
from fraud_detection_mlops.training.evaluate import (
    evaluate_model,
    find_optimal_threshold,
)
from fraud_detection_mlops.training.evaluation_gate import (
    run_evaluation_gate,
)
from fraud_detection_mlops.training.features.feature_eng import (
    ALL_FEATURES,
    build_feature_pipeline,
    prepare_features,
)
from fraud_detection_mlops.training.models.xgb_model import XGBoostFraudModel

app = typer.Typer(help="Train the fraud detection XGBoost model.")


def load_parquet(path: str) -> pd.DataFrame:
    """Load a Parquet file from local path or S3."""
    logger.info("Loading data from {}", path)
    df = pd.read_parquet(path)
    logger.info("Loaded {:,} rows × {} cols", len(df), len(df.columns))
    return df


def get_champion_metrics(
    model_name: str,
    mlflow_client: mlflow.tracking.MlflowClient,
) -> dict | None:
    """
    Fetch the current Production model's test metrics from MLflow.

    Uses the alias-based API (MLflow 2.9+) instead of deprecated stages.
    Returns None if no production model exists (first deployment).
    """
    try:
        # Try alias-based lookup first (MLflow 2.9+)
        try:
            version = mlflow_client.get_model_version_by_alias(model_name, "champion")
            run = mlflow_client.get_run(version.run_id)
            metrics = run.data.metrics
            logger.info(
                "Champion found via alias: run_id={} auc_pr={:.4f}",
                version.run_id[:8],
                metrics.get("test_auc_pr", 0.0),
            )
            return cast(dict[Any, Any], metrics)
        except Exception:
            pass

        # Fallback: search for the most recent registered version
        # that has been manually marked as production
        results = mlflow_client.search_model_versions(
            f"name='{model_name}'",
            order_by=["version_number DESC"],
            max_results=1,
        )
        if not results:
            logger.info("No champion found — this is the first deployment")
            return None

        # Check if any version has champion tag
        for mv in results:
            tags = mv.tags or {}
            if tags.get("stage") == "production":
                run = mlflow_client.get_run(mv.run_id)
                return cast(dict[Any, Any], run.data.metrics)

        logger.info("No production-tagged version found — first deployment")
        return None

    except Exception as exc:
        logger.warning("Could not fetch champion metrics: {}", exc)
        return None


def run_training_pipeline(
    train_path: str,
    val_path: str,
    test_path: str,
    output_dir: str = ".",
    register_model: bool = True,
) -> dict:
    """
    Full training pipeline — called by CLI, SageMaker, and Airflow.

    Parameters
    ----------
    train_path : str
        Path to train.parquet (local or S3).
    val_path : str
        Path to val.parquet (local or S3).
    test_path : str
        Path to test.parquet (local or S3).
    output_dir : str
        Directory to save model artifact locally.
    register_model : bool
        Whether to register the model to MLflow registry.

    Returns
    -------
    dict
        Summary of training results including gate decision.
    """
    settings = get_settings()
    model_cfg = settings.model_cfg
    training_cfg = settings.training_cfg
    mlflow_cfg = training_cfg["mlflow"]

    # ── MLflow setup ───────────────────────────────────────────────────
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(mlflow_cfg["experiment_name"])
    client = mlflow.tracking.MlflowClient()

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logger.info("MLflow run started: {}", run_id[:8])

        # ── Log run metadata ───────────────────────────────────────────
        mlflow.set_tags(
            {
                "git_commit": os.getenv("GIT_COMMIT", "local"),
                "env": settings.env,
                "model_type": "xgboost",
                "dataset": "creditcard_kaggle",
            }
        )

        # ── Step 1: Load data ──────────────────────────────────────────
        logger.info("=== Step 1: Loading data ===")
        train_df = load_parquet(train_path)
        val_df = load_parquet(val_path)
        test_df = load_parquet(test_path)

        mlflow.log_params(
            {
                "train_size": len(train_df),
                "val_size": len(val_df),
                "test_size": len(test_df),
                "train_fraud_rate": float(train_df["Class"].mean()),
                "train_path": train_path,
            }
        )

        # ── Step 2: Feature engineering ────────────────────────────────
        logger.info("=== Step 2: Feature engineering ===")
        pipeline = build_feature_pipeline()

        X_train, y_train = prepare_features(train_df, pipeline, fit=True)
        X_val, y_val = prepare_features(val_df, pipeline, fit=False)
        X_test, y_test = prepare_features(test_df, pipeline, fit=False)

        # Log pipeline as artifact — required for serving
        mlflow.sklearn.log_model(
            pipeline,
            name="feature_pipeline",
        )
        logger.info("Feature pipeline logged to MLflow")

        # ── Step 3: Build and train model ──────────────────────────────
        logger.info("=== Step 3: Training XGBoost ===")
        xgb_cfg = model_cfg["xgboost"]

        model = XGBoostFraudModel(
            n_estimators=xgb_cfg["n_estimators"],
            max_depth=xgb_cfg["max_depth"],
            learning_rate=xgb_cfg["learning_rate"],
            subsample=xgb_cfg["subsample"],
            colsample_bytree=xgb_cfg["colsample_bytree"],
            scale_pos_weight=xgb_cfg["scale_pos_weight"],
            early_stopping_rounds=xgb_cfg["early_stopping_rounds"],
            random_seed=model_cfg["random_seed"],
        )

        model.fit(
            X_train,
            y_train,
            X_val,
            y_val,
            feature_names=ALL_FEATURES,
        )

        # Log all hyperparameters
        mlflow.log_params(model.params)

        # ── Step 4: Find optimal threshold on validation set ───────────
        logger.info("=== Step 4: Finding optimal threshold ===")
        y_val_proba = model.predict_proba(X_val)
        optimal_threshold = find_optimal_threshold(
            y_proba=y_val_proba,
            y_true=y_val,
            metric="f1",
        )
        mlflow.log_param("decision_threshold", optimal_threshold)

        # ── Step 5: Evaluate on all splits ────────────────────────────
        logger.info("=== Step 5: Evaluation ===")
        train_metrics = evaluate_model(
            model,
            X_train,
            y_train,
            threshold=optimal_threshold,
            dataset_name="train",
        )
        val_metrics = evaluate_model(
            model,
            X_val,
            y_val,
            threshold=optimal_threshold,
            dataset_name="val",
        )
        test_metrics = evaluate_model(
            model,
            X_test,
            y_test,
            threshold=optimal_threshold,
            dataset_name="test",
        )

        # Log metrics with split prefix
        for split, metrics in [
            ("train", train_metrics),
            ("val", val_metrics),
            ("test", test_metrics),
        ]:
            mlflow.log_metrics(
                {f"{split}_{k}": v for k, v in metrics.to_dict().items()}
            )

        # Log feature importances
        importance = model.get_feature_importance()
        mlflow.log_dict(importance, "feature_importance.json")
        top5 = list(importance.items())[:5]
        logger.info("Top 5 features: {}", top5)

        # ── Step 6: Save model artifact ────────────────────────────────
        logger.info("=== Step 6: Saving model ===")
        model_path = Path(output_dir) / "model.joblib"
        model.save(model_path)
        # Log as MLflow model (required for register_model to work)
        mlflow.log_artifact(str(model_path), "model_artifact")

        # Also log the XGBoost model natively — this creates the
        # logged_model entry that register_model requires
        mlflow.xgboost.log_model(
            xgb_model=model._model,
            name="model",
            input_example=X_test[:1],
        )

        # ── Step 7: Evaluation gate ────────────────────────────────────
        logger.info("=== Step 7: Evaluation gate ===")
        champion_raw = get_champion_metrics(mlflow_cfg["model_name"], client)

        champion_metrics = None
        if champion_raw:
            from fraud_detection_mlops.training.evaluate import ModelMetrics

            champion_metrics = ModelMetrics(
                auc_roc=champion_raw.get("test_auc_roc", 0.0),
                auc_pr=champion_raw.get("test_auc_pr", 0.0),
                f1=champion_raw.get("test_f1", 0.0),
                precision=champion_raw.get("test_precision", 0.0),
                recall=champion_raw.get("test_recall", 0.0),
                n_samples=int(champion_raw.get("test_n_samples", 0)),
                n_positives=int(champion_raw.get("test_n_positives", 0)),
            )

        gate_result = run_evaluation_gate(test_metrics, champion_metrics)
        mlflow.log_param("gate_decision", gate_result.decision.name)
        mlflow.log_param(
            "gate_passed_checks",
            sum(1 for c in gate_result.checks if c.passed),
        )
        mlflow.log_param("gate_total_checks", len(gate_result.checks))

        # ── Step 8: Register if gate passes ───────────────────────────
        if gate_result.passed and register_model:
            logger.info("=== Step 8: Registering model ===")
            model_uri = f"runs:/{run_id}/model"
            registered = mlflow.register_model(
                model_uri=model_uri,
                name=mlflow_cfg["model_name"],
            )

            # Set alias "champion" on this version (replaces deprecated stages)
            client.set_registered_model_alias(
                name=mlflow_cfg["model_name"],
                alias="champion",
                version=registered.version,
            )

            # Tag it so we can query it later
            client.set_model_version_tag(
                name=mlflow_cfg["model_name"],
                version=registered.version,
                key="stage",
                value="production",
            )

            logger.success(
                "Model registered: name={} version={} alias=champion",
                mlflow_cfg["model_name"],
                registered.version,
            )
        elif not gate_result.passed:
            logger.error(
                "Gate FAILED — model NOT registered. " "Failed checks: {}",
                [c.check_name for c in gate_result.failed_checks],
            )

    return {
        "run_id": run_id,
        "gate_passed": gate_result.passed,
        "test_auc_pr": test_metrics.auc_pr,
        "test_f1": test_metrics.f1,
        "threshold": optimal_threshold,
    }


@app.command()
def main(
    train_path: str = typer.Option(..., help="Path to train.parquet"),
    val_path: str = typer.Option(..., help="Path to val.parquet"),
    test_path: str = typer.Option(..., help="Path to test.parquet"),
    output_dir: str = typer.Option(".", help="Directory to save model"),
    no_register: bool = typer.Option(False, help="Skip MLflow registration (dry run)"),
) -> None:
    """Train the fraud detection model and run the evaluation gate."""
    result = run_training_pipeline(
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        output_dir=output_dir,
        register_model=not no_register,
    )
    if result["gate_passed"]:
        logger.success("Training pipeline complete. Gate PASSED.")
    else:
        logger.error("Training pipeline complete. Gate FAILED.")
        raise SystemExit(1)


if __name__ == "__main__":
    app()
