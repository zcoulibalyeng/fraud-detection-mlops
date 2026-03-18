from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings

CONFIG_DIR = Path(__file__).parent


def _load_yaml(filename: str) -> dict[Any, Any]:
    """Load a YAML config file from the configs directory."""
    with open(CONFIG_DIR / filename) as f:
        content = yaml.safe_load(f)
    env = os.getenv("ENV", "dev")
    # Replace {env} placeholders with actual environment
    return cast(dict[Any, Any], _replace_env(content, env))


def _replace_env(obj: object, env: str) -> Any:
    """Recursively replace {env} placeholders in config values."""
    if isinstance(obj, str):
        return obj.replace("{env}", env)
    if isinstance(obj, dict):
        return {k: _replace_env(v, env) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_replace_env(item, env) for item in obj]
    return obj


class Settings(BaseSettings):
    """Central settings object — reads from env vars and YAML configs."""

    # Environment
    env: str = Field(default="dev", alias="ENV")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # AWS
    aws_region: str = Field(default="us-east-1", alias="AWS_REGION")
    aws_access_key_id: str = Field(default="", alias="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: str = Field(default="", alias="AWS_SECRET_ACCESS_KEY")
    sagemaker_role_arn: str = Field(default="", alias="SAGEMAKER_ROLE_ARN")

    # MLflow
    mlflow_tracking_uri: str = Field(
        default="http://localhost:5000", alias="MLFLOW_TRACKING_URI"
    )

    # Slack / PagerDuty
    slack_webhook_url: str = Field(default="", alias="SLACK_WEBHOOK_URL")
    pagerduty_key: str = Field(default="", alias="PAGERDUTY_KEY")

    model_config = {"populate_by_name": True, "env_file": ".env"}

    @property
    def model_cfg(self) -> dict:
        return _load_yaml("model.yaml")

    @property
    def training_cfg(self) -> dict:
        return _load_yaml("training.yaml")

    @property
    def evaluation_gate_cfg(self) -> dict:
        return _load_yaml("evaluation_gate.yaml")

    @property
    def serving_cfg(self) -> dict:
        return _load_yaml("serving.yaml")

    @property
    def monitoring_cfg(self) -> dict:
        return _load_yaml("monitoring.yaml")

    @property
    def aws_cfg(self) -> dict:
        return _load_yaml("aws.yaml")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings instance — import this everywhere."""
    return Settings()
