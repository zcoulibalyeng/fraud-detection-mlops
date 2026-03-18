"""
Data preparation pipeline — the entry point for the data module.

Reads raw CSV → validates → splits → writes Parquet to S3.
Called by Airflow DAG and directly via CLI for local development.
"""

from __future__ import annotations

import io
from pathlib import Path

import boto3
import pandas as pd
import typer
from loguru import logger

from fraud_detection_mlops.configs.settings import get_settings
from fraud_detection_mlops.data.pipelines.split_strategy import (
    SplitRatios,
    SplitResult,
    temporal_split,
)
from fraud_detection_mlops.data.validation.validator import FraudDataValidator

app = typer.Typer(help="Data preparation pipeline for fraud detection.")


def read_csv_from_s3(s3_path: str) -> pd.DataFrame:
    """Read a CSV file from S3 into a DataFrame."""
    logger.info("Reading CSV from {}", s3_path)
    s3 = boto3.client("s3")
    bucket, key = _parse_s3_path(s3_path)
    response = s3.get_object(Bucket=bucket, Key=key)
    df = pd.read_csv(io.BytesIO(response["Body"].read()))
    logger.info("Loaded {:,} rows × {} columns", len(df), len(df.columns))
    return df


def read_csv_local(local_path: str) -> pd.DataFrame:
    """Read a CSV file from the local filesystem (for development)."""
    logger.info("Reading CSV from local path: {}", local_path)
    df = pd.read_csv(local_path)
    logger.info("Loaded {:,} rows × {} columns", len(df), len(df.columns))
    return df


def write_parquet_to_s3(df: pd.DataFrame, s3_path: str) -> None:
    """Write a DataFrame as Parquet to S3."""
    logger.info("Writing {:,} rows to {}", len(df), s3_path)
    s3 = boto3.client("s3")
    bucket, key = _parse_s3_path(s3_path)
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False, engine="pyarrow")
    buffer.seek(0)
    s3.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue())
    logger.success("Wrote Parquet to {}", s3_path)


def write_parquet_local(df: pd.DataFrame, local_path: str) -> None:
    """Write a DataFrame as Parquet locally (for development)."""
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(local_path, index=False, engine="pyarrow")
    logger.success("Wrote {:,} rows to {}", len(df), local_path)


def _parse_s3_path(s3_path: str) -> tuple[str, str]:
    """Parse s3://bucket/key into (bucket, key)."""
    if not s3_path.startswith("s3://"):
        raise ValueError(f"Expected s3:// path, got: {s3_path}")
    path = s3_path[5:]
    bucket, _, key = path.partition("/")
    return bucket, key


def run_preparation_pipeline(
    input_path: str,
    output_prefix: str,
    use_s3: bool = True,
) -> SplitResult:
    """
    Full data preparation pipeline.

    Parameters
    ----------
    input_path : str
        Path to the raw CSV (s3:// or local).
    output_prefix : str
        Prefix for output Parquet files (s3:// or local dir).
    use_s3 : bool
        If True, reads/writes from S3. If False, uses local filesystem.

    Returns
    -------
    SplitResult
        The split DataFrames (useful for testing and downstream steps).
    """
    settings = get_settings()
    cfg = settings.training_cfg

    # ── Step 1: Read data ─────────────────────────────────────────────
    df = read_csv_local(input_path) if not use_s3 else read_csv_from_s3(input_path)

    # ── Step 2: Validate ──────────────────────────────────────────────
    validator = FraudDataValidator()
    validator.validate(df, run_name="raw_data_validation")

    # ── Step 3: Split ─────────────────────────────────────────────────
    data_cfg = cfg["data"]
    ratios = SplitRatios(
        train=data_cfg["train_ratio"],
        val=data_cfg["val_ratio"],
        test=data_cfg["test_ratio"],
    )
    split = temporal_split(df, ratios)

    # ── Step 4: Write Parquet ─────────────────────────────────────────
    splits = {
        "train": split.train,
        "val": split.val,
        "test": split.test,
    }

    for name, split_df in splits.items():
        out_path = f"{output_prefix}/{name}.parquet"
        if use_s3:
            write_parquet_to_s3(split_df, out_path)
        else:
            write_parquet_local(split_df, out_path)

    logger.success(
        "Pipeline complete. Train: {:,} | Val: {:,} | Test: {:,}",
        len(split.train),
        len(split.val),
        len(split.test),
    )

    return split


@app.command()
def main(
    input_path: str = typer.Option(..., help="Path to raw CSV (local or s3://)"),
    output_prefix: str = typer.Option(..., help="Output prefix (local dir or s3://)"),
    use_s3: bool = typer.Option(False, help="Use S3 for I/O (default: local)"),
) -> None:
    """Prepare the fraud dataset: validate, split, write Parquet."""

    run_preparation_pipeline(input_path, output_prefix, use_s3)


if __name__ == "__main__":
    app()
