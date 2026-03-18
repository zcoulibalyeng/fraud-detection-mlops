# ── Stage 1: builder ────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files first — maximizes layer cache reuse
COPY pyproject.toml uv.lock* ./

# Install production dependencies only
RUN uv sync --no-dev --frozen

# ── Stage 2: runtime ────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy the installed virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application source
COPY fraud_detection_mlops/ ./fraud_detection_mlops/

# Copy model artifact and training data for pipeline fitting
COPY models/model.joblib ./models/model.joblib
COPY fraud_detection_mlops/data/processed/train.parquet ./fraud_detection_mlops/data/processed/train.parquet

# Environment
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ENV=production \
    MLFLOW_TRACKING_URI=http://127.0.0.1:5000

# Expose serving port
EXPOSE 8080

# Never run as root in production
RUN adduser --disabled-password --gecos "" appuser
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"

# Start the server
CMD ["python", "-m", "uvicorn", "fraud_detection_mlops.serving.api.main:app", \
     "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]