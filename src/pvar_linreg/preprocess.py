"""Minimal preprocessing — the DGP produces clean data. Steps mirror what a real-world
pipeline would apply, are documented, and are run defensively even on synthetic data."""

from __future__ import annotations

import pandas as pd


def drop_iqr_outliers(
    df: pd.DataFrame, target_col: str = "PVar_abs", k: float = 3.0
) -> pd.DataFrame:
    """Drop rows where ``target_col`` falls outside ``[Q1 - k*IQR, Q3 + k*IQR]``.

    The injected ~1% decimal-point errors should be removed here. A robust regression
    (RLM with Huber's T) is fit on the *un-cleaned* data in the appendix as a comparison.
    """
    raise NotImplementedError("Phase 4")


def log_transform_target(
    df: pd.DataFrame, target_col: str = "PVar_abs", eps: float = 1e-4
) -> pd.DataFrame:
    """Add ``log_target`` column. Non-negative, right-skewed → log is the natural choice.

    The Box-Cox lambda is shown in the appendix for completeness; expected ≈ 0 → log.
    """
    raise NotImplementedError("Phase 4")


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode tier, channel, country with ``drop_first=True`` for OLS interpretability."""
    raise NotImplementedError("Phase 4")


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Orchestrator: outlier-drop -> log target -> encode categoricals."""
    raise NotImplementedError("Phase 4")


__all__ = ["drop_iqr_outliers", "encode_categoricals", "log_transform_target", "prepare"]
