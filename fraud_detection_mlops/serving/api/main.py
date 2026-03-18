"""
FastAPI application — the serving layer entry point.

Architecture:
  - Lifespan: loads model once at startup, handles graceful shutdown
  - Middleware: request logging, correlation IDs, latency tracking
  - Endpoints: /predict, /health, /ready, /metrics
  - Error handlers: structured JSON errors for all failure modes

Never import training code here except through the Predictor.
Never import data pipeline code here.
The serving layer knows only: schemas, predictor, settings.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)

from fraud_detection_mlops.configs.settings import get_settings
from fraud_detection_mlops.serving.api.predictor import ModelNotLoadedError, Predictor
from fraud_detection_mlops.serving.api.schemas import (
    HealthResponse,
    PredictRequest,
    PredictResponse,
    ReadinessResponse,
)

# ── Prometheus metrics ─────────────────────────────────────────────────────
# Defined at module level — shared across all requests

REQUEST_COUNT = Counter(
    "fraud_predict_requests_total",
    "Total prediction requests",
    ["status"],  # "success" or "error"
)

REQUEST_LATENCY = Histogram(
    "fraud_predict_latency_ms",
    "Prediction request latency in milliseconds",
    buckets=[5, 10, 25, 50, 100, 250, 500, 1000],
)

PREDICTION_SCORE = Histogram(
    "fraud_prediction_score",
    "Distribution of fraud probability scores",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

FRAUD_DECISIONS = Counter(
    "fraud_decisions_total",
    "Total fraud/legitimate decisions",
    ["decision"],  # "fraud" or "legitimate"
)

# ── Predictor singleton ────────────────────────────────────────────────────
_predictor: Predictor | None = None
_app_start_time: float = 0.0


def get_predictor() -> Predictor:
    """Return the global predictor instance."""
    if _predictor is None:
        raise RuntimeError("Predictor not initialized — lifespan not running")
    return _predictor


# ── Application lifespan ───────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Manage application startup and shutdown.

    Startup: load model, run warmup, mark ready.
    Shutdown: log final stats.

    Using lifespan instead of @app.on_event("startup") is the
    modern FastAPI pattern — cleaner and handles exceptions better.
    """
    global _predictor, _app_start_time

    settings = get_settings()
    cfg = settings.serving_cfg

    logger.info("Starting fraud detection serving API")
    _app_start_time = time.time()

    # Initialize and load model
    _predictor = Predictor(
        model_name=cfg["model"]["name"],
        model_stage=cfg["model"]["stage"],
        threshold=0.5,
    )
    _predictor.load()

    logger.success(
        "API ready — model={} version={}",
        cfg["model"]["name"],
        _predictor.model_version,
    )

    yield  # Application runs here

    logger.info("Shutting down fraud detection serving API")


# ── FastAPI app ────────────────────────────────────────────────────────────

settings = get_settings()

app = FastAPI(
    title="Fraud Detection API",
    description="Real-time credit card fraud detection using XGBoost",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Middleware: request logging + correlation IDs ──────────────────────────


@app.middleware("http")
async def log_requests(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:  # type: ignore[type-arg]
    """Log every request with correlation ID and latency."""
    correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
    start = time.perf_counter()

    response = await call_next(request)

    latency_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "method={} path={} status={} latency_ms={:.2f} correlation_id={}",
        request.method,
        request.url.path,
        response.status_code,
        latency_ms,
        correlation_id,
    )

    response.headers["X-Correlation-ID"] = correlation_id
    response.headers["X-Latency-Ms"] = f"{latency_ms:.2f}"
    return response


# ── Endpoints ──────────────────────────────────────────────────────────────


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    """
    Score a single credit card transaction for fraud probability.

    Returns fraud_probability in [0, 1] and a binary is_fraud decision.
    All 28 PCA features and Amount are required.
    """
    start = time.perf_counter()

    try:
        predictor = get_predictor()
        features = request.to_feature_dict()
        fraud_proba, is_fraud = predictor.predict(features)

    except ModelNotLoadedError as exc:
        REQUEST_COUNT.labels(status="error").inc()
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    except Exception as exc:
        REQUEST_COUNT.labels(status="error").inc()
        logger.exception("Prediction failed for request_id={}", request.request_id)
        raise HTTPException(
            status_code=500,
            detail="Internal prediction error. Check server logs.",
        ) from exc

    latency_ms = (time.perf_counter() - start) * 1000

    # Record Prometheus metrics
    REQUEST_COUNT.labels(status="success").inc()
    REQUEST_LATENCY.observe(latency_ms)
    PREDICTION_SCORE.observe(fraud_proba)
    FRAUD_DECISIONS.labels(decision="fraud" if is_fraud else "legitimate").inc()

    logger.debug(
        "Predicted: request_id={} score={:.4f} is_fraud={} latency_ms={:.2f}",
        request.request_id,
        fraud_proba,
        is_fraud,
        latency_ms,
    )

    return PredictResponse(
        fraud_probability=fraud_proba,
        is_fraud=is_fraud,
        model_version=predictor.model_version,
        request_id=request.request_id,
        latency_ms=latency_ms,
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """
    Liveness probe — is the server running?

    Kubernetes calls this every 10s. Returns 200 as long as the
    process is alive. Does NOT check model readiness.
    """
    predictor = get_predictor()
    return HealthResponse(
        status="ok",
        model_loaded=predictor.is_loaded,
        model_version=predictor.model_version,
        uptime_seconds=time.time() - _app_start_time,
    )


@app.get("/ready", response_model=ReadinessResponse)
async def ready() -> ReadinessResponse:
    """
    Readiness probe — is the server ready to serve traffic?

    Kubernetes calls this before routing traffic to a new pod.
    Returns 503 if the model is not loaded and warmed up.
    """
    predictor = get_predictor()
    checks = {
        "model_loaded": predictor.is_loaded,
        "model_not_mock": predictor.model_version != "mock",
    }
    all_ready = all(checks.values())

    if not all_ready:
        raise HTTPException(
            status_code=503,
            detail=f"Not ready: {checks}",
        )

    cfg = get_settings().serving_cfg
    return ReadinessResponse(
        ready=True,
        model_name=cfg["model"]["name"],
        model_stage=cfg["model"]["stage"],
        checks=checks,
    )


@app.get("/metrics")
async def metrics() -> Response:
    """
    Prometheus metrics endpoint.

    Scraped every 15s by Prometheus.
    Exposes: request counts, latency histogram, score distribution,
    fraud decision counts.
    """
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint — useful for quick smoke tests."""
    return {
        "service": "fraud-detection-api",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = get_settings().serving_cfg
    uvicorn.run(
        "serving.api.main:app",
        host=cfg["host"],
        port=cfg["port"],
        workers=1,  # multiple workers via Kubernetes, not uvicorn
        log_level="info",
    )
