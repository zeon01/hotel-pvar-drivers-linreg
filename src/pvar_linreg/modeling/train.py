"""Fit OLS (statsmodels) headline + cluster-robust + RLM + quantile regressions, and a
parallel sklearn LinearRegression for cross-validated R^2."""

from __future__ import annotations

import logging

import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score

from pvar_linreg.config import MODELS_DIR, PROCESSED_DIR, SEED

log = logging.getLogger(__name__)


FORMULA: str = (
    "log_pvar_abs ~ log_lead_time + is_weekend + is_holiday "
    "+ month_sin + month_cos "
    "+ C(property_tier) + C(channel) + C(country) "
    "+ C(property_tier):C(channel) "
    "+ log_expected_rate + star_rating_centered"
)


def fit_ols(df: pd.DataFrame, formula: str = FORMULA, cov_type: str = "HC3"):
    """Fit OLS with HC3 robust SE (default)."""
    model = smf.ols(formula=formula, data=df).fit(cov_type=cov_type)
    return model


def fit_cluster_robust(df: pd.DataFrame, formula: str = FORMULA, group_col: str = "property_id"):
    """Fit OLS with cluster-robust SE on ``property_id`` - the recommended headline model."""
    return smf.ols(formula=formula, data=df).fit(
        cov_type="cluster", cov_kwds={"groups": df[group_col]}
    )


def fit_rlm(df: pd.DataFrame, formula: str = FORMULA, M: str = "HuberT"):
    """Robust linear model for outlier resilience. Run on the *uncleaned* frame in the
    appendix as a comparison."""
    norm_class = sm.robust.norms.HuberT() if M == "HuberT" else sm.robust.norms.TukeyBiweight()
    return smf.rlm(formula=formula, data=df, M=norm_class).fit()


def fit_quantile(df: pd.DataFrame, q: float = 0.5, formula: str = FORMULA):
    """Quantile regression at quantile ``q``. Run at q in {0.5, 0.75, 0.9} in the appendix."""
    return smf.quantreg(formula=formula, data=df).fit(q=q, max_iter=2000)


def cross_validated_r2(df: pd.DataFrame, n_splits: int = 5, seed: int = SEED) -> dict[str, float]:
    """sklearn LinearRegression in 5-fold KFold CV. Honest R^2 + RMSE."""
    from patsy import dmatrices

    y, X = dmatrices(FORMULA, data=df, return_type="dataframe")
    y = y.iloc[:, 0]
    X = X.drop(columns=[c for c in X.columns if c == "Intercept"], errors="ignore")
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    r2_scores = cross_val_score(LinearRegression(), X, y, cv=cv, scoring="r2", n_jobs=-1)
    rmse_scores = -cross_val_score(
        LinearRegression(), X, y, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1
    )
    return {
        "r2_mean": float(np.mean(r2_scores)),
        "r2_std": float(np.std(r2_scores)),
        "rmse_mean": float(np.mean(rmse_scores)),
        "rmse_std": float(np.std(rmse_scores)),
    }


def main() -> None:
    import logging as _log

    _log.basicConfig(level=_log.INFO)

    df = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    log.info("Fitting OLS HC3 (n=%d)...", len(df))
    ols_hc3 = fit_ols(df, cov_type="HC3")
    log.info(ols_hc3.summary().tables[1].as_text())

    log.info("Fitting OLS cluster-robust on property_id...")
    ols_cluster = fit_cluster_robust(df)

    log.info("Fitting RLM (Huber)...")
    rlm = fit_rlm(df, M="HuberT")

    log.info("Fitting quantile regression at q=0.9 (right-tail driver)...")
    qr_90 = fit_quantile(df, q=0.9)

    log.info("Computing cross-validated R^2...")
    cv = cross_validated_r2(df)
    log.info("CV R^2=%.4f (+/- %.4f), RMSE=%.4f", cv["r2_mean"], cv["r2_std"], cv["rmse_mean"])

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(ols_hc3, MODELS_DIR / "ols_hc3.joblib")
    joblib.dump(ols_cluster, MODELS_DIR / "ols_cluster.joblib")
    joblib.dump(rlm, MODELS_DIR / "rlm_huber.joblib")
    joblib.dump(qr_90, MODELS_DIR / "qreg_q90.joblib")
    pd.Series(cv).to_csv(MODELS_DIR / "cv_summary.csv")
    log.info("Wrote OLS / cluster / RLM / quantile artefacts to %s", MODELS_DIR)


if __name__ == "__main__":
    main()
