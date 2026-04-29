"""Out-of-sample evaluation. For an inferential regression this is mostly diagnostics +
held-out R²/RMSE for honesty."""

from __future__ import annotations

import numpy as np
import pandas as pd


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """R², adjusted-R², RMSE, MAE."""
    raise NotImplementedError("Phase 4")


def residual_plot_data(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """Frame for the residuals-vs-fitted plot — the most informative diagnostic."""
    raise NotImplementedError("Phase 4")


def bootstrap_coef_ci(
    df: pd.DataFrame, formula: str, n_boot: int = 1000, cluster_col: str | None = "property_id"
) -> pd.DataFrame:
    """Cluster bootstrap on ``property_id``; return 95% percentile CIs per coefficient."""
    raise NotImplementedError("Phase 4")


def main() -> None:
    raise NotImplementedError("Phase 4")


if __name__ == "__main__":
    main()
