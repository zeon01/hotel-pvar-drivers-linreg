"""End-to-end orchestrator. With ``--report`` re-renders headline figures consumed by the README."""

from __future__ import annotations

import argparse
import logging
import sys

import joblib
import matplotlib.pyplot as plt

from pvar_linreg.config import FIGURES_DIR, MODELS_DIR
from pvar_linreg.interpret import coef_table
from pvar_linreg.plotting import set_style

log = logging.getLogger(__name__)


def render_headline_forest_plot() -> None:
    set_style()
    ols = joblib.load(MODELS_DIR / "ols_hc3.joblib")
    table = coef_table(ols)
    table = table[table["feature"] != "Intercept"].head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.errorbar(
        table["coef"],
        range(len(table)),
        xerr=[table["coef"] - table["ci_lo"], table["ci_hi"] - table["coef"]],
        fmt="o",
        color="#205493",
        ecolor="#999",
        capsize=3,
    )
    ax.axvline(0.0, ls="--", color="gray", lw=1)
    ax.set_yticks(range(len(table)))
    ax.set_yticklabels(table["feature"])
    ax.set_xlabel("Coefficient (log PVar_abs)")
    ax.set_title("Top 15 PVar drivers - OLS HC3")
    fig.savefig(FIGURES_DIR / "03_coef_forest.png")
    plt.close(fig)
    log.info("Saved coefficient forest plot.")


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    if args.report:
        render_headline_forest_plot()
    return 0


if __name__ == "__main__":
    sys.exit(main())
