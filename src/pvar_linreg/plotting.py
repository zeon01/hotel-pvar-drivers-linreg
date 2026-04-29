"""Consistent plotting style for figures used in notebooks and reports/."""

from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns


def set_style() -> None:
    """Apply the project's matplotlib/seaborn style."""
    sns.set_theme(context="talk", style="whitegrid", font_scale=0.9)
    plt.rcParams.update(
        {
            "figure.figsize": (8, 5),
            "figure.dpi": 100,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.titleweight": "semibold",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def annotate_finding(ax, text: str) -> None:
    """Add a one-sentence subtitle under the axes title that states the *finding*."""
    raise NotImplementedError("Phase 4")


__all__ = ["annotate_finding", "set_style"]
