"""
Pydantic schemas for the serving API.

Single source of truth for request/response contracts.
FastAPI uses these for automatic validation, serialization,
and OpenAPI documentation generation.

If the input shape changes here, the API docs update automatically.
If a request violates the schema, FastAPI rejects it before the
model ever sees it — zero defensive code needed in the endpoint.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    """
    Input schema for a single fraud prediction request.

    The 28 PCA features (V1-V28) are already scaled by the card network.
    Amount is raw — the feature pipeline applies log1p + StandardScaler.
    Time is excluded — it was only used for temporal splitting.
    """

    # PCA features — already normalized by the card network
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float

    # Raw transaction amount — pipeline handles scaling
    Amount: float = Field(
        ...,
        gt=0,
        description="Transaction amount in USD. Must be positive.",
        examples=[142.50],
    )

    # Optional correlation ID from caller — generated if not provided
    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Caller-supplied request ID for tracing.",
    )

    @field_validator("Amount")
    @classmethod
    def amount_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"Amount must be positive, got {v}")
        return v

    def to_feature_dict(self) -> dict[str, float]:
        """Return only the model feature columns as a dict."""
        return {k: v for k, v in self.model_dump().items() if k not in {"request_id"}}


class PredictResponse(BaseModel):
    """
    Output schema for a single fraud prediction.

    fraud_probability: raw model score in [0, 1].
    is_fraud: binary decision using the configured threshold.
    model_version: which model version produced this prediction.
    request_id: echoed back for client-side tracing.
    latency_ms: server-side inference latency.
    """

    fraud_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probability of fraud in [0, 1].",
        examples=[0.023],
    )
    is_fraud: bool = Field(
        ...,
        description="Binary fraud decision at configured threshold.",
    )
    model_version: str = Field(
        ...,
        description="MLflow model version that produced this prediction.",
    )
    request_id: str
    latency_ms: float = Field(
        ...,
        description="Server-side inference latency in milliseconds.",
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="UTC timestamp of prediction.",
    )


class HealthResponse(BaseModel):
    """Response schema for GET /health."""

    status: str
    model_loaded: bool
    model_version: str
    uptime_seconds: float


class ReadinessResponse(BaseModel):
    """Response schema for GET /ready."""

    ready: bool
    model_name: str
    model_stage: str
    checks: dict[str, bool]
