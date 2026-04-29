"""Minimal preprocessing - the DGP produces clean data. Steps mirror what a real-world
pipeline would apply, are documented, and are run defensively even on synthetic data."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from pvar_linreg.config import PROCESSED_DIR

log = logging.getLogger(__name__)


def drop_iqr_outliers(
    df: pd.DataFrame, target_col: str = "PVar_abs", k: float = 3.0
) -> pd.DataFrame:
    """Drop rows where ``target_col`` falls outside ``[Q1 - k*IQR, Q3 + k*IQR]``.

    The injected ~1% decimal-point errors should be removed here. A robust regression
    (RLM with Huber's T) is fit on the *un-cleaned* data in the appendix as a comparison.
    """
    q1 = df[target_col].quantile(0.25)
    q3 = df[target_col].quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - k * iqr, q3 + k * iqr
    n_before = len(df)
    out = df.loc[df[target_col].between(lo, hi)].copy()
    log.info(
        "drop_iqr_outliers(%s, k=%.1f): %d -> %d (lo=%.4f, hi=%.4f)",
        target_col,
        k,
        n_before,
        len(out),
        lo,
        hi,
    )
    return out


def log_transform_target(
    df: pd.DataFrame, target_col: str = "PVar_abs", eps: float = 1e-4
) -> pd.DataFrame:
    """Add ``log_pvar_abs`` column. Non-negative, right-skewed -> log is the natural choice."""
    out = df.copy()
    out["log_pvar_abs"] = np.log(out[target_col].clip(lower=eps) + eps)
    return out


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure ``property_tier``, ``channel``, ``country`` are pandas Categorical so
    statsmodels formula ``C(...)`` produces interpretable output. Returns a *new* frame."""
    out = df.copy()
    for col in ("property_tier", "channel", "country"):
        if col in out.columns and not isinstance(out[col].dtype, pd.CategoricalDtype):
            out[col] = pd.Categorical(out[col])
    return out


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Orchestrator: outlier-drop -> log target -> encode categoricals."""
    out = drop_iqr_outliers(df)
    out = log_transform_target(out)
    out = encode_categoricals(out)
    out_path = PROCESSED_DIR / "prepared.parquet"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    log.info("prepare: wrote %s (%d rows, %d cols)", out_path, len(out), out.shape[1])
    return out


__all__ = ["drop_iqr_outliers", "encode_categoricals", "log_transform_target", "prepare"]
