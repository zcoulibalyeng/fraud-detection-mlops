"""
Temporal split strategy for time-series ML datasets.

Uses the Time column to preserve chronological order.
Critical: random splits leak future data into training — never use them.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from loguru import logger


@dataclass(frozen=True)
class SplitRatios:
    """Validated split ratios that must sum to 1.0."""

    train: float
    val: float
    test: float

    def __post_init__(self) -> None:
        total = self.train + self.val + self.test
        if not abs(total - 1.0) < 1e-9:
            raise ValueError(
                f"Split ratios must sum to 1.0, got {total:.4f}. "
                f"Received train={self.train}, val={self.val}, test={self.test}"
            )
        for name, ratio in [
            ("train", self.train),
            ("val", self.val),
            ("test", self.test),
        ]:
            if not 0 < ratio < 1:
                raise ValueError(f"{name} ratio must be between 0 and 1, got {ratio}")


@dataclass(frozen=True)
class SplitResult:
    """Container for the three split DataFrames."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    @property
    def sizes(self) -> dict[str, int]:
        return {
            "train": len(self.train),
            "val": len(self.val),
            "test": len(self.test),
            "total": len(self.train) + len(self.val) + len(self.test),
        }

    @property
    def fraud_rates(self) -> dict[str, float]:
        return {
            "train": self.train["Class"].mean(),
            "val": self.val["Class"].mean(),
            "test": self.test["Class"].mean(),
        }


def temporal_split(
    df: pd.DataFrame,
    ratios: SplitRatios,
    time_column: str = "Time",
) -> SplitResult:
    """
    Split a DataFrame chronologically to prevent data leakage.

    Sorts by time_column then slices by position — earlier data goes to
    training, later data to validation and test. This mirrors the real
    deployment scenario where the model always predicts the future.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to split. Must contain time_column.
    ratios : SplitRatios
        The train/val/test ratios (must sum to 1.0).
    time_column : str
        Column to sort by before splitting.

    Returns
    -------
    SplitResult
        Dataclass containing train, val, and test DataFrames.
    """
    if time_column not in df.columns:
        raise ValueError(f"time_column '{time_column}' not found in DataFrame")

    # Sort chronologically — this is the core of temporal splitting
    df_sorted = df.sort_values(time_column).reset_index(drop=True)
    n = len(df_sorted)

    train_end = int(n * ratios.train)
    val_end = train_end + int(n * ratios.val)

    train_df = df_sorted.iloc[:train_end].copy()
    val_df = df_sorted.iloc[train_end:val_end].copy()
    test_df = df_sorted.iloc[val_end:].copy()

    result = SplitResult(train=train_df, val=val_df, test=test_df)

    logger.info("Temporal split complete:")
    for split_name, size in result.sizes.items():
        if split_name != "total":
            fraud_rate = result.fraud_rates.get(split_name, 0)
            logger.info(
                "  {}: {:,} rows | fraud rate: {:.4%}", split_name, size, fraud_rate
            )

    return result
