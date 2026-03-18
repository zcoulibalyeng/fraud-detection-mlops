"""
Validator — runs fraud data expectations against a DataFrame.

Uses GE v0.18+ fluent API. Each expectation maps directly to one rule
in FraudExpectations. Raises DataValidationError with a clear message
describing exactly which checks failed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from loguru import logger

from fraud_detection_mlops.data.validation.expectations import FraudExpectations


class DataValidationError(Exception):
    """Raised when the dataset fails one or more validation checks."""


@dataclass
class ValidationResult:
    """Result of a single validation check."""

    check_name: str
    passed: bool
    message: str


class FraudDataValidator:
    """
    Validates a DataFrame against the fraud detection expectation rules.

    Runs every check independently and collects all failures before
    raising — so you see every problem at once, not just the first one.

    Parameters
    ----------
    report_dir : Path, optional
        Directory to save the text validation report.
    expectations : FraudExpectations, optional
        Override default expectations (useful in tests).
    """

    def __init__(
        self,
        report_dir: Path = Path("validation_reports"),
        expectations: FraudExpectations | None = None,
    ) -> None:
        self.report_dir = report_dir
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.expectations = expectations or FraudExpectations()

    def validate(self, df: pd.DataFrame, run_name: str = "validation") -> None:
        """
        Validate a DataFrame against all fraud expectations.

        Parameters
        ----------
        df : pd.DataFrame
            The raw fraud dataset to validate.
        run_name : str
            Label for this run — used in the report filename.

        Raises
        ------
        DataValidationError
            If any check fails. Message lists every failed check.
        """
        logger.info(
            "Running validation on DataFrame shape={} run={}",
            df.shape,
            run_name,
        )

        results: list[ValidationResult] = []
        exp = self.expectations

        # ── Schema checks ──────────────────────────────────────────────
        for col in exp.required_columns:
            results.append(self._check_column_exists(df, col))

        # Only run value checks if schema is intact
        schema_ok = all(r.passed for r in results)
        if not schema_ok:
            self._report_and_raise(results, run_name)

        # ── Null checks ────────────────────────────────────────────────
        for col in exp.required_columns:
            results.append(self._check_no_nulls(df, col))

        # ── Type checks ────────────────────────────────────────────────
        for col in exp.numeric_columns:
            results.append(self._check_numeric_type(df, col))

        # ── Range checks ───────────────────────────────────────────────
        results.append(self._check_min_value(df, "Time", exp.min_time))
        results.append(self._check_min_value(df, "Amount", exp.min_amount))

        # ── Target label checks ────────────────────────────────────────
        results.append(
            self._check_value_set(df, exp.target_column, exp.valid_class_labels)
        )

        # ── Volume checks ──────────────────────────────────────────────
        results.append(self._check_row_count(df, exp.min_row_count, exp.max_row_count))

        # ── Class balance checks ───────────────────────────────────────
        results.append(
            self._check_fraud_rate(
                df, exp.target_column, exp.min_fraud_rate, exp.max_fraud_rate
            )
        )

        self._report_and_raise(results, run_name)

    # ── Private check methods ──────────────────────────────────────────

    def _check_column_exists(self, df: pd.DataFrame, col: str) -> ValidationResult:
        passed = col in df.columns
        return ValidationResult(
            check_name=f"column_exists:{col}",
            passed=passed,
            message="" if passed else f"Required column '{col}' is missing",
        )

    def _check_no_nulls(self, df: pd.DataFrame, col: str) -> ValidationResult:
        null_count = int(df[col].isna().sum())
        passed = null_count == 0
        return ValidationResult(
            check_name=f"no_nulls:{col}",
            passed=passed,
            message="" if passed else f"Column '{col}' has {null_count} null values",
        )

    def _check_numeric_type(self, df: pd.DataFrame, col: str) -> ValidationResult:
        passed = pd.api.types.is_numeric_dtype(df[col])
        return ValidationResult(
            check_name=f"numeric_type:{col}",
            passed=passed,
            message=(
                ""
                if passed
                else f"Column '{col}' is not numeric (dtype={df[col].dtype})"
            ),
        )

    def _check_min_value(
        self, df: pd.DataFrame, col: str, min_val: float
    ) -> ValidationResult:
        actual_min = float(df[col].min())
        passed = actual_min >= min_val
        return ValidationResult(
            check_name=f"min_value:{col}>={min_val}",
            passed=passed,
            message=(
                ""
                if passed
                else (
                    f"Column '{col}' has values below {min_val} (min={actual_min:.4f})"
                )
            ),
        )

    def _check_value_set(
        self, df: pd.DataFrame, col: str, valid_set: list[int]
    ) -> ValidationResult:
        invalid = set(df[col].unique()) - set(valid_set)
        passed = len(invalid) == 0
        return ValidationResult(
            check_name=f"value_set:{col}",
            passed=passed,
            message=(
                "" if passed else (f"Column '{col}' contains invalid values: {invalid}")
            ),
        )

    def _check_row_count(
        self, df: pd.DataFrame, min_count: int, max_count: int
    ) -> ValidationResult:
        n = len(df)
        passed = min_count <= n <= max_count
        return ValidationResult(
            check_name=f"row_count:{min_count}<={n}<={max_count}",
            passed=passed,
            message=(
                ""
                if passed
                else (
                    f"Row count {n:,} outside expected range [{min_count:,}, {max_count:,}]"
                )
            ),
        )

    def _check_fraud_rate(
        self,
        df: pd.DataFrame,
        col: str,
        min_rate: float,
        max_rate: float,
    ) -> ValidationResult:
        rate = float(df[col].mean())
        passed = min_rate <= rate <= max_rate
        return ValidationResult(
            check_name=f"fraud_rate:{min_rate:.3f}<={rate:.4f}<={max_rate:.3f}",
            passed=passed,
            message=(
                ""
                if passed
                else (
                    f"Fraud rate {rate:.4%} outside expected range "
                    f"[{min_rate:.3%}, {max_rate:.3%}]"
                )
            ),
        )

    def _report_and_raise(self, results: list[ValidationResult], run_name: str) -> None:
        """Log all results, save report, raise if any failed."""
        passed = [r for r in results if r.passed]
        failed = [r for r in results if not r.passed]

        # Save text report
        report_path = self.report_dir / f"{run_name}_report.txt"
        with open(report_path, "w") as f:
            f.write(f"Validation report: {run_name}\n")
            f.write(f"Passed: {len(passed)} | Failed: {len(failed)}\n\n")
            for r in results:
                status = "PASS" if r.passed else "FAIL"
                f.write(f"[{status}] {r.check_name}")
                if not r.passed:
                    f.write(f" — {r.message}")
                f.write("\n")

        logger.info(
            "Validation: {}/{} checks passed",
            len(passed),
            len(results),
        )

        if failed:
            failure_messages = [r.message for r in failed]
            raise DataValidationError(
                f"Validation failed: {len(failed)} check(s) failed.\n"
                + "\n".join(f"  - {m}" for m in failure_messages)
            )

        logger.success("All {} validation checks passed", len(results))
