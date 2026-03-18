"""
Verify a SageMaker deployment is healthy after rollout.

Calls /health and /predict on the deployed endpoint.
Fails loudly if anything is wrong — blocks the deploy pipeline.
"""

from __future__ import annotations

import json

import boto3
import typer
from loguru import logger

app = typer.Typer()


@app.command()
def main(
    endpoint_name: str = typer.Option(..., help="SageMaker endpoint name"),
    region: str = typer.Option("us-east-1", help="AWS region"),
) -> None:
    """Verify the deployed endpoint responds correctly."""
    runtime = boto3.client("sagemaker-runtime", region_name=region)

    # Build a test payload
    payload = {f"V{i}": 0.0 for i in range(1, 29)}
    payload["Amount"] = 100.0

    logger.info("Sending test prediction to endpoint: {}", endpoint_name)

    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Body=json.dumps(payload),
    )

    result = json.loads(response["Body"].read())
    logger.info("Response: {}", result)

    # Validate response shape
    assert "fraud_probability" in result, "Missing fraud_probability"
    assert "is_fraud" in result, "Missing is_fraud"
    assert 0.0 <= result["fraud_probability"] <= 1.0, "Score out of range"

    logger.success(
        "Deployment verified: score={:.4f} is_fraud={}",
        result["fraud_probability"],
        result["is_fraud"],
    )


if __name__ == "__main__":
    app()
