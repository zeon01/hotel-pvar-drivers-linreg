"""Coefficient extraction, robust SE selection, plain-English interpretation, and
recovered-vs-true coefficient comparison."""

from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger(__name__)


def coef_table(result, ci_level: float = 0.95) -> pd.DataFrame:
    """coef, std-err, t, p, [ci_low, ci_high], sorted by |t|."""
    params = result.params
    se = result.bse
    tvals = result.tvalues
    pvals = result.pvalues
    conf = result.conf_int(alpha=1 - ci_level).rename(columns={0: "ci_lo", 1: "ci_hi"})

    out = pd.DataFrame(
        {
            "feature": params.index,
            "coef": params.to_numpy(),
            "std_err": se.to_numpy(),
            "t_stat": tvals.to_numpy(),
            "p": pvals.to_numpy(),
            "ci_lo": conf["ci_lo"].to_numpy(),
            "ci_hi": conf["ci_hi"].to_numpy(),
        }
    )
    out["abs_t"] = out["t_stat"].abs()
    return out.sort_values("abs_t", ascending=False).reset_index(drop=True)


def robust_se_table(result, cov_type: str = "HC3") -> pd.DataFrame:
    """Re-derive the coefficient table under a different cov_type."""
    robust = result.get_robustcov_results(cov_type=cov_type)
    return coef_table(robust)


def cluster_robust_table(result, group_series: pd.Series) -> pd.DataFrame:
    """Cluster-robust SE on ``group_series`` (typically property_id)."""
    robust = result.get_robustcov_results(cov_type="cluster", groups=group_series.to_numpy())
    return coef_table(robust)


def plain_english(result, top_k: int = 5) -> list[str]:
    """One-sentence summaries for the top-k coefficients (by |t|)."""
    table = coef_table(result)
    table = table[table["feature"] != "Intercept"].head(top_k)
    sentences: list[str] = []
    for _, row in table.iterrows():
        direction = "raises" if row["coef"] > 0 else "lowers"
        sentences.append(
            f"`{row['feature']}` {direction} log(PVar_abs) by "
            f"{abs(row['coef']):.3f} (95% CI [{row['ci_lo']:.3f}, {row['ci_hi']:.3f}], "
            f"p={row['p']:.3g})."
        )
    return sentences


def coefficient_recovery(result, true_coefs: dict[str, float]) -> pd.DataFrame:
    """Compare estimated coefficients to the DGP-true values.

    For each true-coef name, find the closest match by substring in the regression's
    exog_names and report (estimate, ci_lo, ci_hi, true, in_ci).
    """
    table = coef_table(result)
    rows: list[dict] = []
    for name, true_value in true_coefs.items():
        # Fuzzy match: prefer exact, fall back to substring.
        match = table[table["feature"] == name]
        if match.empty:
            mask = table["feature"].str.contains(name, regex=False, case=False)
            match = table[mask]
        if match.empty:
            rows.append(
                {
                    "true_name": name,
                    "matched": "(no match)",
                    "estimate": float("nan"),
                    "ci_lo": float("nan"),
                    "ci_hi": float("nan"),
                    "true": float(true_value),
                    "in_ci": False,
                }
            )
            continue
        # Take the smallest-p row.
        match = match.sort_values("p").iloc[0]
        in_ci = bool(match["ci_lo"] <= true_value <= match["ci_hi"])
        rows.append(
            {
                "true_name": name,
                "matched": match["feature"],
                "estimate": float(match["coef"]),
                "ci_lo": float(match["ci_lo"]),
                "ci_hi": float(match["ci_hi"]),
                "true": float(true_value),
                "in_ci": in_ci,
            }
        )
    return pd.DataFrame(rows)


__all__ = [
    "cluster_robust_table",
    "coef_table",
    "coefficient_recovery",
    "plain_english",
    "robust_se_table",
]
