"""Data-generating process for synthetic OTA price-variance data.

The DGP embeds OTA Supply Operations domain knowledge — supplier-tier effects,
channel-manager noise, lead-time-dependent surge, holiday & weekend multipliers, and a
mis-specified-rate-plan failure mode that produces realistic parity violations.

Because the DGP is transparent, the OLS analysis can be checked against ground truth in
the appendix. The skill being demonstrated is the *recovery*, not the answer.
"""

from __future__ import annotations

import pandas as pd


def draw_properties(
    n: int = 5_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Sample `n` properties with tier, country, star_rating, contracted_base_rate."""
    raise NotImplementedError("Phase 4")


def daily_panel(properties: pd.DataFrame, n_days: int = 365, seed: int = 42) -> pd.DataFrame:
    """Cross-join properties with days; add date, day-of-week, month, weekend, holiday flags."""
    raise NotImplementedError("Phase 4")


def add_expected_rate(panel: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """``expected_rate = contracted_base_rate * season * weekend * holiday * lead_time_factor``.

    ``season_multiplier`` is a sinusoidal annual cycle plus holiday spikes.
    ``lead_time_factor`` introduces non-linearity concentrated in the last 14 days.
    """
    raise NotImplementedError("Phase 4")


def add_channel_observations(
    panel: pd.DataFrame,
    channels: tuple[str, ...] = ("Direct", "ChannelManagerA", "ChannelManagerB"),
    seed: int = 42,
) -> pd.DataFrame:
    """For each (property, day, channel), draw a posted_rate from expected_rate * noise.

    Channel B is more error-prone than Channel A. Per-tier base parity-violation rates:
    Strategic 0.5%, Preferred 1%, Standard 3%, Long-tail 8%. Inject ~1% severe
    "decimal-point" rate-push errors (rate 10x off) to validate diagnostics.
    """
    raise NotImplementedError("Phase 4")


def compute_pvar(panel: pd.DataFrame) -> pd.DataFrame:
    """``PVar = (posted_rate - expected_rate) / expected_rate``; ``PVar_abs = abs(PVar)``."""
    raise NotImplementedError("Phase 4")


def true_coefficients() -> dict[str, float]:
    """Return the dict of true coefficients embedded in the DGP.

    Used in the appendix to validate that the OLS estimates approximately recover them.
    """
    raise NotImplementedError("Phase 4")


def generate(
    n_rows: int = 200_000,
    seed: int = 42,
) -> pd.DataFrame:
    """End-to-end DGP. Sample down to ``n_rows`` rows uniformly and return the frame."""
    raise NotImplementedError("Phase 4")


__all__ = [
    "add_channel_observations",
    "add_expected_rate",
    "compute_pvar",
    "daily_panel",
    "draw_properties",
    "generate",
    "true_coefficients",
]
