"""Tests for the settings and config loading module."""

from fraud_detection_mlops.configs.settings import Settings, get_settings


def test_settings_loads() -> None:
    """Settings object initializes without error."""
    settings = Settings()
    assert settings.env == "dev"
    assert settings.aws_region == "us-east-1"


def test_model_config_loads() -> None:
    """Model YAML config loads and contains required keys."""
    settings = get_settings()
    cfg = settings.model_cfg
    assert "name" in cfg
    assert "algorithm" in cfg
    assert cfg["algorithm"] == "xgboost"
    assert "target_column" in cfg


def test_evaluation_gate_config_loads() -> None:
    """Evaluation gate config loads with all required thresholds."""
    settings = get_settings()
    cfg = settings.evaluation_gate_cfg
    assert "min_auc_pr_delta" in cfg
    assert "min_f1_delta" in cfg
    assert "max_p99_latency_ms" in cfg
    assert cfg["min_auc_pr_delta"] > 0


def test_training_config_loads() -> None:
    """Training config loads with data paths and MLflow settings."""
    settings = get_settings()
    cfg = settings.training_cfg
    assert "data" in cfg
    assert "mlflow" in cfg
    assert cfg["mlflow"]["experiment_name"] == "fraud-detection"


def test_get_settings_is_cached() -> None:
    """get_settings returns the same instance on repeated calls."""
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2
