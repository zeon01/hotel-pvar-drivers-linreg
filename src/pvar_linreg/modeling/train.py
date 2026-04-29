"""Fit OLS (statsmodels) and a parallel sklearn LinearRegression for cross-validated R²."""

from __future__ import annotations

import pandas as pd

from pvar_linreg.config import SEED


FORMULA: str = (
    "log_pvar_abs ~ log_lead_time + is_weekend + is_holiday "
    "+ month_sin + month_cos "
    "+ C(property_tier) + C(channel) + C(country) "
    "+ C(property_tier):C(channel) "
    "+ log_expected_rate + star_rating_centered"
)


def fit_ols(df: pd.DataFrame, formula: str = FORMULA, robust: str = "HC3"):
    """Fit ``statsmodels.formula.api.ols`` and return the result with HC3 robust SE."""
    raise NotImplementedError("Phase 4")


def fit_cluster_robust(df: pd.DataFrame, formula: str = FORMULA, group_col: str = "property_id"):
    """Fit OLS with cluster-robust SE on ``property_id`` — the recommended headline model
    because residuals within a property are likely correlated over time."""
    raise NotImplementedError("Phase 4")


def fit_rlm(df: pd.DataFrame, formula: str = FORMULA, M: str = "HuberT"):
    """Robust linear model — Huber's T or Tukey's biweight — for outlier-resistance."""
    raise NotImplementedError("Phase 4")


def cross_validated_r2(df: pd.DataFrame, n_splits: int = 5, seed: int = SEED) -> dict[str, float]:
    """sklearn.LinearRegression in a 5-fold KFold CV. Reports honest R² and RMSE."""
    raise NotImplementedError("Phase 4")


def main() -> None:
    raise NotImplementedError("Phase 4")


if __name__ == "__main__":
    main()
