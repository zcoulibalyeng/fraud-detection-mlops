"""
Fraud dataset validation rules using Great Expectations v0.18+ fluent API.

All rules defined as a plain dataclass — no GE context needed at definition
time. The validator instantiates the suite when it runs validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# The 28 PCA-anonymized features in the dataset
PCA_FEATURES: list[str] = [f"V{i}" for i in range(1, 29)]
REQUIRED_COLUMNS: list[str] = ["Time", *PCA_FEATURES, "Amount", "Class"]
POSITIVE_CLASS_LABEL: int = 1
NEGATIVE_CLASS_LABEL: int = 0


@dataclass(frozen=True)
class FraudExpectations:
    """
    Declarative expectation rules for the fraud dataset.

    Decoupled from GE context — the validator reads these and applies them.
    This makes the rules testable without instantiating GE infrastructure.
    """

    required_columns: list[str] = field(default_factory=lambda: REQUIRED_COLUMNS)
    numeric_columns: list[str] = field(
        default_factory=lambda: [*PCA_FEATURES, "Amount", "Time"]
    )
    target_column: str = "Class"
    valid_class_labels: list[int] = field(
        default_factory=lambda: [POSITIVE_CLASS_LABEL, NEGATIVE_CLASS_LABEL]
    )

    # Volume
    min_row_count: int = 100_000
    max_row_count: int = 500_000

    # Class balance — fraud rate between 0.1% and 5%
    min_fraud_rate: float = 0.001
    max_fraud_rate: float = 0.050

    # Value ranges
    min_time: float = 0.0
    min_amount: float = 0.0
