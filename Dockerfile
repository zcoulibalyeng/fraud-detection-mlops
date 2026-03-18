# ── Stage 1: builder ────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
COPY pyproject.toml uv.lock* ./
RUN uv sync --no-dev --frozen

# ── Stage 2: runtime ────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY fraud_detection_mlops/ ./fraud_detection_mlops/

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ENV=production

# Create serve script for SageMaker
RUN printf '#!/bin/bash\nexec python -m uvicorn fraud_detection_mlops.serving.api.main:app --host 0.0.0.0 --port 8080 --workers 1\n' > /app/serve && \
    chmod +x /app/serve

EXPOSE 8080

RUN adduser --disabled-password --gecos "" appuser
USER appuser

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/ping')"

ENV PATH="/app:${PATH}"
CMD ["serve"]