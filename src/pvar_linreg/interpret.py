"""Coefficient extraction, robust SE selection, plain-English interpretation."""

from __future__ import annotations

import pandas as pd


def coef_table(result, ci_level: float = 0.95) -> pd.DataFrame:
    """Return a frame: coef, std-err, t-stat, p, [ci_low, ci_high], sorted by |t|."""
    raise NotImplementedError("Phase 4")


def robust_se(result, cov_type: str = "HC3") -> pd.DataFrame:
    """Recompute the coef table under HC3 (default), HC0/1/2, or cluster-on-property_id.

    HC3 is preferred at all sample sizes per MacKinnon & White (1985); we still show
    HC0..HC2 in the appendix to make the comparison visible.
    """
    raise NotImplementedError("Phase 4")


def cluster_robust(result, group_col: pd.Series) -> pd.DataFrame:
    """Cluster-robust SE on ``group_col`` (typically ``property_id``)."""
    raise NotImplementedError("Phase 4")


def plain_english(result, top_k: int = 5) -> list[str]:
    """Translate the top-k coefficients (by |t|) into prose for the README."""
    raise NotImplementedError("Phase 4")


def coefficient_recovery(result, true_coefs: dict[str, float]) -> pd.DataFrame:
    """Compare estimated vs. DGP-true coefficients side-by-side. Unfakeable evidence."""
    raise NotImplementedError("Phase 4")


__all__ = [
    "cluster_robust",
    "coef_table",
    "coefficient_recovery",
    "plain_english",
    "robust_se",
]
