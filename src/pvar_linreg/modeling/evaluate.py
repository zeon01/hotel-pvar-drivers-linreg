"""Out-of-sample evaluation, residual diagnostics, bootstrap CIs on coefficients,
recovered-vs-true validation."""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from pvar_linreg.config import FIGURES_DIR, MODELS_DIR, PROCESSED_DIR, SEED
from pvar_linreg.dgp import true_coefficients
from pvar_linreg.interpret import coef_table, coefficient_recovery
from pvar_linreg.modeling.splits import random_split
from pvar_linreg.modeling.train import FORMULA
from pvar_linreg.plotting import set_style

log = logging.getLogger(__name__)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def residual_plot_data(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"fitted": y_pred, "residual": y_true - y_pred})


def bootstrap_coef_ci(
    df: pd.DataFrame,
    formula: str = FORMULA,
    n_boot: int = 500,
    cluster_col: str | None = "property_id",
    seed: int = SEED,
) -> pd.DataFrame:
    """Cluster bootstrap on ``property_id`` if available, else row bootstrap.

    Returns a frame with columns ``feature, mean, ci_lo, ci_hi`` summarising the
    bootstrap distribution of each coefficient.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float]] = []
    if cluster_col and cluster_col in df.columns:
        groups = df[cluster_col].unique()
        grouped = {g: df.index[df[cluster_col] == g].to_numpy() for g in groups}
        for _ in range(n_boot):
            sampled_groups = rng.choice(groups, size=len(groups), replace=True)
            idx = np.concatenate([grouped[g] for g in sampled_groups])
            sub = df.loc[idx]
            try:
                fit = smf.ols(formula=formula, data=sub).fit(cov_type="HC3")
            except Exception:
                continue
            for name, value in fit.params.items():
                rows.append({"replicate": _, "feature": name, "coef": float(value)})
    else:
        for _ in range(n_boot):
            sub = df.sample(n=len(df), replace=True, random_state=int(rng.integers(0, 2**31 - 1)))
            try:
                fit = smf.ols(formula=formula, data=sub).fit(cov_type="HC3")
            except Exception:
                continue
            for name, value in fit.params.items():
                rows.append({"replicate": _, "feature": name, "coef": float(value)})

    boots = pd.DataFrame(rows)
    summary = (
        boots.groupby("feature")["coef"]
        .agg(
            mean="mean",
            ci_lo=lambda x: float(np.percentile(x, 2.5)),
            ci_hi=lambda x: float(np.percentile(x, 97.5)),
            n="count",
        )
        .reset_index()
    )
    return summary


def _save_residual_plot(y_true, y_pred, out: Path) -> None:
    df = residual_plot_data(y_true, y_pred)
    fig, ax = plt.subplots()
    ax.scatter(df["fitted"], df["residual"], s=4, alpha=0.3)
    ax.axhline(0, ls="--", color="gray", lw=1)
    ax.set_xlabel("Fitted")
    ax.set_ylabel("Residual")
    ax.set_title("Residuals vs fitted - OLS HC3")
    fig.savefig(out)
    plt.close(fig)


def _save_qq_plot(residuals: np.ndarray, out: Path) -> None:
    import statsmodels.api as sm

    fig = sm.qqplot(residuals, line="45", fit=True)
    ax = fig.gca()
    ax.set_title("Q-Q residuals")
    fig.savefig(out)
    plt.close(fig)


def _save_coef_forest(table: pd.DataFrame, out: Path, top_k: int = 15) -> None:
    df = table.head(top_k).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.errorbar(
        df["coef"],
        range(len(df)),
        xerr=[df["coef"] - df["ci_lo"], df["ci_hi"] - df["coef"]],
        fmt="o",
        color="#205493",
        ecolor="#999",
        capsize=3,
    )
    ax.axvline(0.0, ls="--", color="gray", lw=1)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["feature"])
    ax.set_xlabel("Coefficient (log PVar_abs scale)")
    ax.set_title("Top 15 PVar drivers - OLS HC3")
    fig.savefig(out)
    plt.close(fig)


def _save_recovery_plot(rec: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(
        rec["true"],
        rec["estimate"],
        yerr=[rec["estimate"] - rec["ci_lo"], rec["ci_hi"] - rec["estimate"]],
        fmt="o",
        color="#205493",
        ecolor="#999",
        capsize=3,
    )
    lo = float(min(rec["true"].min(), rec["estimate"].min()))
    hi = float(max(rec["true"].max(), rec["estimate"].max()))
    ax.plot([lo, hi], [lo, hi], "--", color="gray", lw=1)
    for _, row in rec.iterrows():
        ax.annotate(row["true_name"], (row["true"], row["estimate"]), fontsize=8, alpha=0.7)
    ax.set_xlabel("DGP-true coefficient")
    ax.set_ylabel("OLS estimate")
    ax.set_title("Recovered vs true coefficients (95% CI)")
    fig.savefig(out)
    plt.close(fig)


def _save_interaction_plot(df: pd.DataFrame, out: Path) -> None:
    """Tier x channel interaction visualisation."""
    fig, ax = plt.subplots(figsize=(9, 6))
    means = df.pivot_table(
        index="property_tier",
        columns="channel",
        values="log_pvar_abs",
        observed=True,
    )
    means.plot(marker="o", ax=ax)
    ax.set_ylabel("Mean log(PVar_abs)")
    ax.set_xlabel("Property tier")
    ax.set_title("Interaction: property tier x channel")
    ax.legend(title="Channel", loc="best")
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    import logging as _log

    _log.basicConfig(level=_log.INFO)
    set_style()

    df = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    train, test = random_split(df, test_size=0.2, seed=SEED)

    # Held-out R^2 / RMSE via sklearn LinearRegression on the same design matrix.
    from patsy import dmatrices

    y_train, X_train = dmatrices(FORMULA, data=train, return_type="dataframe")
    y_test, X_test = dmatrices(FORMULA, data=test, return_type="dataframe")
    y_train = y_train.iloc[:, 0]
    y_test = y_test.iloc[:, 0]
    X_train = X_train.drop(
        columns=[c for c in X_train.columns if c == "Intercept"], errors="ignore"
    )
    X_test = X_test.drop(columns=[c for c in X_test.columns if c == "Intercept"], errors="ignore")

    lr = LinearRegression().fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    test_metrics = metrics(y_test.to_numpy(), y_pred)
    log.info(
        "Held-out test R^2=%.4f, RMSE=%.4f, MAE=%.4f",
        test_metrics["r2"],
        test_metrics["rmse"],
        test_metrics["mae"],
    )
    pd.Series(test_metrics).to_csv(MODELS_DIR / "test_metrics.csv")

    # OLS HC3 (headline-table source) and cluster-robust (recommended deployment SE).
    ols_hc3 = joblib.load(MODELS_DIR / "ols_hc3.joblib")
    ols_cluster = joblib.load(MODELS_DIR / "ols_cluster.joblib")

    table_hc3 = coef_table(ols_hc3)
    table_cluster = coef_table(ols_cluster)
    table_hc3.to_csv(MODELS_DIR / "ols_hc3_coefs.csv", index=False)
    table_cluster.to_csv(MODELS_DIR / "ols_cluster_coefs.csv", index=False)

    # Recovered-vs-true validation.
    recovery = coefficient_recovery(ols_hc3, true_coefficients())
    recovery.to_csv(MODELS_DIR / "coef_recovery.csv", index=False)
    log.info(
        "Recovery: %d / %d coefficients in 95%% CI", int(recovery["in_ci"].sum()), len(recovery)
    )

    # Bootstrap CIs (cluster on property_id).
    log.info("Cluster-bootstrap on property_id (500 replicates)...")
    boot = bootstrap_coef_ci(train, n_boot=500)
    boot.to_csv(MODELS_DIR / "coef_bootstrap_ci.csv", index=False)

    # Residuals + Q-Q.
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    _save_residual_plot(y_test.to_numpy(), y_pred, FIGURES_DIR / "03_residuals_vs_fitted.png")
    _save_qq_plot(y_test.to_numpy() - y_pred, FIGURES_DIR / "03_qq_residuals.png")
    _save_coef_forest(table_hc3, FIGURES_DIR / "03_coef_forest.png")
    _save_interaction_plot(df, FIGURES_DIR / "04_tier_x_channel_interaction.png")
    if recovery["estimate"].notna().any():
        _save_recovery_plot(
            recovery.dropna(subset=["estimate"]), FIGURES_DIR / "05_recovery_plot.png"
        )

    # Heatmap of residual variance by month - heteroskedasticity preview.
    if "date" in test.columns:
        resid = y_test.to_numpy() - y_pred
        per_month = pd.DataFrame(
            {
                "month": test["date"].dt.to_period("M").astype(str).reset_index(drop=True),
                "resid": resid,
            }
        )
        var_by_month = per_month.groupby("month")["resid"].var()
        fig, ax = plt.subplots(figsize=(9, 4))
        var_by_month.plot.bar(ax=ax, color="#cf6e5e")
        ax.set_ylabel("Variance(residual)")
        ax.set_title("Residual variance by month - heteroskedasticity check")
        fig.savefig(FIGURES_DIR / "06_resid_variance_by_month.png")
        plt.close(fig)

    log.info("Evaluation complete. Figures in %s", FIGURES_DIR)


if __name__ == "__main__":
    main()
