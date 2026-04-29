"""Feature engineering for OLS."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from pvar_linreg.config import PROCESSED_DIR

log = logging.getLogger(__name__)


def add_log_lead_time(df: pd.DataFrame) -> pd.DataFrame:
    """``log_lead_time = log1p(lead_time_days)`` to address right skew + non-linearity."""
    out = df.copy()
    out["log_lead_time"] = np.log1p(out["lead_time_days"])
    return out


def add_cyclical_month(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "month" not in out.columns and "date" in out.columns:
        out["month"] = out["date"].dt.month.astype(int)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    return out


def add_log_expected_rate(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["log_expected_rate"] = np.log(out["expected_rate"].clip(lower=1.0))
    return out


def add_centered_star_rating(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["star_rating_centered"] = out["star_rating"] - 3.0
    return out


def build_feature_frame(prepared: pd.DataFrame | None = None) -> pd.DataFrame:
    if prepared is None:
        prepared = pd.read_parquet(PROCESSED_DIR / "prepared.parquet")
    out = prepared
    out = add_log_lead_time(out)
    out = add_cyclical_month(out)
    out = add_log_expected_rate(out)
    out = add_centered_star_rating(out)
    out_path = PROCESSED_DIR / "features.parquet"
    out.to_parquet(out_path, index=False)
    log.info("Wrote %s; output shape=%s", out_path, out.shape)
    return out


__all__ = [
    "add_centered_star_rating",
    "add_cyclical_month",
    "add_log_expected_rate",
    "add_log_lead_time",
    "build_feature_frame",
]
