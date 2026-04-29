"""OLS regression diagnostics: linearity, normality, homoscedasticity, independence,
multicollinearity, influence."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson, jarque_bera

log = logging.getLogger(__name__)


def compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    """Variance Inflation Factor per numeric column.

    Threshold convention: 5 (warn), 10 (drop or merge).
    """
    if X.empty:
        return pd.DataFrame(columns=["feature", "vif"])
    numeric = X.select_dtypes(include=[np.number]).copy()
    numeric = numeric.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all").fillna(0.0)
    if numeric.shape[1] < 2:
        return pd.DataFrame({"feature": numeric.columns, "vif": [np.nan] * numeric.shape[1]})
    Xc = sm.add_constant(numeric, has_constant="add")
    arr = Xc.to_numpy()
    rows = []
    for i, col in enumerate(Xc.columns):
        if col == "const":
            continue
        rows.append({"feature": col, "vif": float(variance_inflation_factor(arr, i))})
    return pd.DataFrame(rows).sort_values("vif", ascending=False).reset_index(drop=True)


def breusch_pagan(result) -> dict[str, float]:
    """Breusch-Pagan test for linear forms of heteroskedasticity."""
    stat, p_value, fvalue, fp = het_breuschpagan(result.resid, result.model.exog)
    return {
        "lm_stat": float(stat),
        "lm_p": float(p_value),
        "f_stat": float(fvalue),
        "f_p": float(fp),
    }


def white_test(result) -> dict[str, float]:
    """White's test - more general (catches non-linear forms)."""
    stat, p_value, fvalue, fp = het_white(result.resid, result.model.exog)
    return {
        "lm_stat": float(stat),
        "lm_p": float(p_value),
        "f_stat": float(fvalue),
        "f_p": float(fp),
    }


def jarque_bera_test(result) -> dict[str, float]:
    jb_stat, jb_pval, skew, kurt = jarque_bera(result.resid)
    return {
        "jb_stat": float(jb_stat),
        "jb_p": float(jb_pval),
        "skew": float(skew),
        "kurt": float(kurt),
    }


def durbin_watson_stat(result) -> float:
    return float(durbin_watson(result.resid))


def cooks_distance(result) -> np.ndarray:
    influence = result.get_influence()
    cooks_d, _ = influence.cooks_distance
    return np.asarray(cooks_d)


def partial_regression_data(result, exog_name: str) -> pd.DataFrame:
    """Frame for a single partial-regression plot.

    For each observation: residual_y, residual_X. Exclude the focal X and the constant.
    """
    if exog_name not in result.model.exog_names:
        raise ValueError(f"{exog_name} not in model.exog_names")
    other = [c for c in result.model.exog_names if c not in {exog_name, "const"}]
    X_full = pd.DataFrame(result.model.exog, columns=result.model.exog_names)
    X_other = X_full[other]
    y = pd.Series(result.model.endog)

    res_y = sm.OLS(y, sm.add_constant(X_other)).fit().resid
    res_x = sm.OLS(X_full[exog_name], sm.add_constant(X_other)).fit().resid
    return pd.DataFrame({"residual_y": res_y, "residual_x": res_x})


def assumption_summary(result) -> pd.DataFrame:
    """Single-frame digest of every assumption check, suitable for the appendix table."""
    rows: list[dict] = []
    bp = breusch_pagan(result)
    rows.append(
        {
            "test": "Breusch-Pagan",
            "stat": bp["lm_stat"],
            "p_value": bp["lm_p"],
            "interpretation": "Reject H0 of homoscedasticity if p < 0.05 (large N: read magnitude + plot)",
        }
    )
    wh = white_test(result)
    rows.append(
        {
            "test": "White",
            "stat": wh["lm_stat"],
            "p_value": wh["lm_p"],
            "interpretation": "Same flavour as BP, more general; same large-N caveat",
        }
    )
    jb = jarque_bera_test(result)
    rows.append(
        {
            "test": "Jarque-Bera",
            "stat": jb["jb_stat"],
            "p_value": jb["jb_p"],
            "interpretation": "Tests residual normality; CLT covers CI validity at large N",
        }
    )
    dw = durbin_watson_stat(result)
    rows.append(
        {
            "test": "Durbin-Watson",
            "stat": dw,
            "p_value": float("nan"),
            "interpretation": "~2.0 = no AR(1); <1.5 or >2.5 = serial correlation",
        }
    )
    return pd.DataFrame(rows)


__all__ = [
    "assumption_summary",
    "breusch_pagan",
    "compute_vif",
    "cooks_distance",
    "durbin_watson_stat",
    "jarque_bera_test",
    "partial_regression_data",
    "white_test",
]
