# ── Stage 1: builder ────────────────────────────────────────────────────
# Install dependencies in an isolated layer.
# This layer is cached as long as pyproject.toml does not change.
FROM python:3.11-slim AS builder

WORKDIR /app

# Install uv for fast dependency installation
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files first — maximizes layer cache reuse
COPY pyproject.toml uv.lock* ./

# Install production dependencies only — no dev tools in the image
RUN uv sync --no-dev --frozen

# ── Stage 2: runtime ────────────────────────────────────────────────────
# Copy only what is needed to run the server.
# Keeps the final image lean and reduces attack surface.
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy the installed virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application source
COPY fraud_detection_mlops/configs/ ./configs/
COPY fraud_detection_mlops/serving/ ./serving/
COPY fraud_detection_mlops/training/ ./training/

# Environment
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ENV=production

# Expose serving port
EXPOSE 8080

# Never run as root in production
RUN adduser --disabled-password --gecos "" appuser
USER appuser

# Health check — Docker will mark container unhealthy if this fails
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"

# Start the server
CMD ["python", "-m", "uvicorn", "serving.api.main:app", \
     "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]