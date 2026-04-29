"""Feature engineering for OLS."""

from __future__ import annotations

import pandas as pd


def add_log_lead_time(df: pd.DataFrame) -> pd.DataFrame:
    """``log_lead_time = log1p(lead_time_days)`` to address right skew + non-linearity."""
    raise NotImplementedError("Phase 4")


def add_cyclical_month(df: pd.DataFrame) -> pd.DataFrame:
    """``month_sin``, ``month_cos`` from the month index."""
    raise NotImplementedError("Phase 4")


def add_log_expected_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Control for absolute price level."""
    raise NotImplementedError("Phase 4")


def add_centered_star_rating(df: pd.DataFrame) -> pd.DataFrame:
    """``star_rating_centered = star_rating - 3.0`` for interpretable intercept."""
    raise NotImplementedError("Phase 4")


def build_feature_frame(prepared: pd.DataFrame | None = None) -> pd.DataFrame:
    """Apply all feature transforms; persist to ``data/processed/features.parquet``."""
    raise NotImplementedError("Phase 4")


__all__ = [
    "add_centered_star_rating",
    "add_cyclical_month",
    "add_log_expected_rate",
    "add_log_lead_time",
    "build_feature_frame",
]
