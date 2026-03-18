"""
Deploy a new model image to SageMaker endpoint.

Called by GitHub Actions deploy workflow.
Updates the endpoint with the new Docker image — zero downtime.
"""

from __future__ import annotations

import time

import boto3
import typer
from loguru import logger

app = typer.Typer()


@app.command()
def main(
    image_uri: str = typer.Option(..., help="ECR image URI to deploy"),
    endpoint_name: str = typer.Option(..., help="SageMaker endpoint name"),
    region: str = typer.Option("us-east-1", help="AWS region"),
    wait: bool = typer.Option(True, help="Wait for deployment to complete"),
) -> None:
    """Deploy a new container image to a SageMaker endpoint."""
    sm = boto3.client("sagemaker", region_name=region)

    model_name = f"fraud-detector-{int(time.time())}"
    config_name = f"fraud-config-{int(time.time())}"

    logger.info("Creating SageMaker model: {}", model_name)
    sm.create_model(
        ModelName=model_name,
        PrimaryContainer={"Image": image_uri},
        ExecutionRoleArn=_get_sagemaker_role(),
    )

    logger.info("Creating endpoint config: {}", config_name)
    sm.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[
            {
                "VariantName": "primary",
                "ModelName": model_name,
                "InstanceType": "ml.m5.large",
                "InitialInstanceCount": 1,
                "InitialVariantWeight": 1,
            }
        ],
    )

    logger.info("Updating endpoint: {}", endpoint_name)
    try:
        sm.update_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=config_name,
        )
    except sm.exceptions.ResourceNotFoundException:
        logger.info("Endpoint not found — creating new endpoint")
        sm.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=config_name,
        )

    if wait:
        logger.info("Waiting for endpoint to become InService...")
        waiter = sm.get_waiter("endpoint_in_service")
        waiter.wait(
            EndpointName=endpoint_name,
            WaiterConfig={"MaxAttempts": 40, "Delay": 30},
        )
        logger.success("Endpoint {} is InService", endpoint_name)


def _get_sagemaker_role() -> str:
    import os

    role = os.getenv("SAGEMAKER_ROLE_ARN")
    if not role:
        raise ValueError("SAGEMAKER_ROLE_ARN environment variable not set")
    return role


if __name__ == "__main__":
    app()
