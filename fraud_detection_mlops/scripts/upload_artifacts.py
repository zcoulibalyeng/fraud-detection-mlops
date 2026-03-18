"""Upload model artifacts to S3 for production serving."""

from __future__ import annotations

import boto3
import typer
from loguru import logger

app = typer.Typer()


@app.command()
def main(
    model_path: str = typer.Option("models/model.joblib", help="Path to model"),
    pipeline_path: str = typer.Option(
        "models/feature_pipeline.joblib", help="Path to fitted pipeline"
    ),
    bucket: str = typer.Option(..., help="S3 bucket name"),
    prefix: str = typer.Option("artifacts/latest", help="S3 key prefix"),
    region: str = typer.Option("us-east-1", help="AWS region"),
) -> None:
    """Upload model and pipeline to S3."""
    s3 = boto3.client("s3", region_name=region)

    for local_path, s3_key in [
        (model_path, f"{prefix}/model.joblib"),
        (pipeline_path, f"{prefix}/feature_pipeline.joblib"),
    ]:
        logger.info("Uploading {} → s3://{}/{}", local_path, bucket, s3_key)
        s3.upload_file(local_path, bucket, s3_key)
        logger.success("Uploaded {}", s3_key)


if __name__ == "__main__":
    app()
