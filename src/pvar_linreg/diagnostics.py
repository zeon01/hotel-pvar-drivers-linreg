"""OLS regression diagnostics: linearity, normality, homoscedasticity, independence,
multicollinearity, influence."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    """Variance Inflation Factor per column. Threshold: 5 (warn), 10 (drop or merge)."""
    raise NotImplementedError("Phase 4")


def breusch_pagan(result) -> dict[str, float]:
    """Breusch-Pagan test for linear forms of heteroskedasticity.

    At N>200k, this rejects at any reasonable level. The *visual* and the *magnitude* of
    heteroskedasticity matter more than the p-value — this is documented in the methodology.
    """
    raise NotImplementedError("Phase 4")


def white_test(result) -> dict[str, float]:
    """White's test for general (including non-linear) heteroskedasticity. Same N caveat."""
    raise NotImplementedError("Phase 4")


def jarque_bera(result) -> dict[str, float]:
    """JB normality test — useful complement to a Q-Q plot at large N."""
    raise NotImplementedError("Phase 4")


def durbin_watson(result) -> float:
    """DW statistic on residuals ordered by date. Detects AR(1); discuss limits."""
    raise NotImplementedError("Phase 4")


def cooks_distance(result) -> np.ndarray:
    """Influence diagnostic — the injected decimal-point errors should appear as outliers."""
    raise NotImplementedError("Phase 4")


def partial_regression_data(result, exog_name: str) -> pd.DataFrame:
    """Data for a single partial-regression plot. Loop over features in the notebook."""
    raise NotImplementedError("Phase 4")


__all__ = [
    "breusch_pagan",
    "compute_vif",
    "cooks_distance",
    "durbin_watson",
    "jarque_bera",
    "partial_regression_data",
    "white_test",
]
