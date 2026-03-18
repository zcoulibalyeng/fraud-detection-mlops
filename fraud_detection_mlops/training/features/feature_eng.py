"""
Feature engineering pipeline for fraud detection.

The sklearn Pipeline here is the single source of truth for feature
transforms. The SAME pipeline object is fitted on training data and
saved with the model — ensuring identical transforms at serving time.

Never fit the pipeline on validation or test data.
Never compute feature statistics outside this module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Features used for training — Time is excluded (used only for splitting)
# Amount is included but needs scaling — PCA features are already scaled
AMOUNT_FEATURES: list[str] = ["Amount"]
PCA_FEATURES: list[str] = [f"V{i}" for i in range(1, 29)]
ALL_FEATURES: list[str] = [*PCA_FEATURES, *AMOUNT_FEATURES]
TARGET_COLUMN: str = "Class"


class AmountLogTransformer(BaseEstimator, TransformerMixin):
    """
    Apply log1p to the Amount column to reduce right skew.

    Amount spans 0 to 25,691 with heavy right skew.
    log1p compresses this into a near-normal distribution
    without losing the zero values (log(0+1) = 0).

    Follows sklearn transformer API so it slots into a Pipeline.
    """

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> AmountLogTransformer:
        return self  # stateless — nothing to fit

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if "Amount" in X.columns:
            X["Amount"] = np.log1p(X["Amount"])
        return X

    def get_feature_names_out(
        self, input_features: list[str] | None = None
    ) -> list[str]:
        return input_features or ALL_FEATURES


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Select only the training features from the full DataFrame.

    Drops Time, Class, and any other non-feature columns.
    Must be the first step in the pipeline — ensures downstream
    steps never see the target or the time index.
    """

    def __init__(self, feature_columns: list[str] = ALL_FEATURES) -> None:
        self.feature_columns = feature_columns

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> FeatureSelector:
        # Validate at fit time that all expected columns are present
        missing = set(self.feature_columns) - set(X.columns)
        if missing:
            raise ValueError(f"FeatureSelector: missing columns at fit time: {missing}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        missing = set(self.feature_columns) - set(X.columns)
        if missing:
            raise ValueError(
                f"FeatureSelector: missing columns at transform time: {missing}"
            )
        return X[self.feature_columns].copy()

    def get_feature_names_out(
        self, input_features: list[str] | None = None
    ) -> list[str]:
        return self.feature_columns


def build_feature_pipeline(
    feature_columns: list[str] = ALL_FEATURES,
) -> Pipeline:
    """
    Build the feature engineering pipeline.

    Steps (in order):
        1. FeatureSelector  — drop non-feature columns
        2. AmountLogTransformer — log1p on Amount
        3. StandardScaler   — standardize all features

    The pipeline is returned unfitted. Call pipeline.fit(X_train)
    before saving. Never call fit on val/test data.

    Parameters
    ----------
    feature_columns : list[str]
        The features to select and transform.

    Returns
    -------
    Pipeline
        Unfitted sklearn Pipeline ready for fit/transform.
    """
    pipeline = Pipeline(
        steps=[
            ("selector", FeatureSelector(feature_columns=feature_columns)),
            ("log_amount", AmountLogTransformer()),
            ("scaler", StandardScaler()),
        ]
    )
    logger.debug("Built feature pipeline with {} features", len(feature_columns))
    return pipeline


def prepare_features(
    df: pd.DataFrame,
    pipeline: Pipeline,
    fit: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply the feature pipeline to a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing feature columns and target column.
    pipeline : Pipeline
        The feature engineering pipeline (fitted or unfitted).
    fit : bool
        If True, fit the pipeline on this data first.
        Only set True for training data — never for val/test.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (X, y) — transformed feature matrix and target array.
    """
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in DataFrame")

    y = df[TARGET_COLUMN].values.astype(int)

    if fit:
        X = pipeline.fit_transform(df)
        logger.info(
            "Fitted and transformed: X shape={}, y shape={}, " "fraud rate={:.4%}",
            X.shape,
            y.shape,
            y.mean(),
        )
    else:
        X = pipeline.transform(df)
        logger.info(
            "Transformed: X shape={}, y shape={}, fraud rate={:.4%}",
            X.shape,
            y.shape,
            y.mean(),
        )

    return X, y
