"""Data-generating process for synthetic OTA price-variance data.

The DGP embeds OTA Supply Operations domain knowledge - supplier-tier effects,
channel-manager noise, lead-time-dependent surge, holiday & weekend multipliers, and a
mis-specified-rate-plan failure mode that produces realistic parity violations.

Because the DGP is transparent, the OLS analysis can be checked against ground truth in
the appendix. The skill being demonstrated is the *recovery*, not the answer.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from pvar_linreg.config import (
    CHANNELS,
    COUNTRIES,
    PROPERTY_TIERS,
    SEED,
    TIER_PROBABILITIES,
)

log = logging.getLogger(__name__)


# Coefficient targets embedded in the DGP. ``true_coefficients()`` returns this dict so
# the appendix can validate the regression's recovery.
_TIER_RATE_MULT: dict[str, float] = {
    "Strategic": 1.30,
    "Preferred": 1.10,
    "Standard": 1.00,
    "Long-tail": 0.85,
}

_COUNTRY_RATE_MULT: dict[str, float] = {
    "TH": 0.95,
    "VN": 0.85,
    "ID": 0.90,
    "MY": 1.00,
    "PH": 0.95,
    "SG": 1.30,
    "JP": 1.40,
    "KR": 1.20,
    "GB": 1.30,
    "DE": 1.25,
    "FR": 1.30,
    "OTHER": 1.05,
}

_CHANNEL_NOISE_SIGMA: dict[str, float] = {
    "Direct": 0.005,
    "ChannelManagerA": 0.012,
    "ChannelManagerB": 0.022,
}

_PARITY_BASE_RATE: dict[str, float] = {
    "Strategic": 0.005,
    "Preferred": 0.010,
    "Standard": 0.030,
    "Long-tail": 0.080,
}

_PARITY_CHANNEL_MULT: dict[str, float] = {
    "Direct": 0.5,
    "ChannelManagerA": 1.0,
    "ChannelManagerB": 1.6,
}

# Decimal-point ("rate-push bug") row injection rate.
_DECIMAL_BUG_RATE: float = 0.01


def draw_properties(n: int = 5_000, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tiers = rng.choice(PROPERTY_TIERS, size=n, p=TIER_PROBABILITIES)
    countries = rng.choice(COUNTRIES, size=n)
    star = np.clip(rng.normal(loc=3.5, scale=0.7, size=n), 2.0, 5.0).round(1)
    base = np.array(
        [
            120.0 * _TIER_RATE_MULT[t] * _COUNTRY_RATE_MULT[c] * (1.0 + 0.18 * (s - 3.0))
            for t, c, s in zip(tiers, countries, star, strict=True)
        ]
    )
    base *= rng.lognormal(mean=0.0, sigma=0.10, size=n)  # property-specific noise
    quality = np.clip(rng.normal(loc=0.0, scale=0.20, size=n), -0.6, 0.6)
    return pd.DataFrame(
        {
            "property_id": np.arange(n, dtype=int),
            "property_tier": pd.Categorical(tiers, categories=list(PROPERTY_TIERS)),
            "country": pd.Categorical(countries, categories=list(COUNTRIES)),
            "star_rating": star,
            "contracted_base_rate": base.round(2),
            "supplier_quality_score": quality.round(3),
        }
    )


def daily_panel(properties: pd.DataFrame, n_days: int = 365, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 1)
    dates = pd.date_range("2025-01-01", periods=n_days, freq="D")

    # Holidays: pick ~25 random days as holiday spikes (simple but realistic-shaped).
    holiday_idx = rng.choice(n_days, size=25, replace=False)
    is_holiday = np.zeros(n_days, dtype=bool)
    is_holiday[holiday_idx] = True

    # Cross-join.
    panel = properties.merge(pd.DataFrame({"date": dates}), how="cross")
    panel["day_of_week"] = panel["date"].dt.dayofweek.astype(int)
    panel["month"] = panel["date"].dt.month.astype(int)
    panel["is_weekend"] = panel["day_of_week"].isin([5, 6])
    panel["is_holiday"] = panel["date"].isin(dates[is_holiday])

    # Sinusoidal annual season + holiday spike.
    day_of_year = panel["date"].dt.dayofyear.astype(int)
    panel["season_multiplier"] = 1.0 + 0.20 * np.sin(2 * np.pi * (day_of_year - 60) / 365.0)
    panel.loc[panel["is_holiday"], "season_multiplier"] *= 1.30

    return panel


def add_lead_time_factor(panel: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """``lead_time_factor`` non-linear, concentrated in last 14 days (early-bird discount
    on the long-lead end, late-booking surge on the short-lead end). One observation per
    (property, day) initially - we then fan out across channels.
    """
    rng = np.random.default_rng(seed + 2)
    n = len(panel)
    out = panel.copy()
    # Lead time 0..200 with skew toward shorter
    lead_time = rng.integers(low=0, high=200, size=n)
    short_surge = np.where(lead_time < 14, 1.0 + 0.025 * (14 - lead_time), 1.0)
    long_decay = np.where(lead_time > 60, 1.0 - 0.0008 * (lead_time - 60), 1.0)
    out["lead_time_days"] = lead_time
    out["lead_time_factor"] = short_surge * long_decay
    return out


def add_expected_rate(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    weekend_mult = np.where(out["is_weekend"], 1.08, 1.0)
    out["expected_rate"] = (
        out["contracted_base_rate"]
        * out["season_multiplier"]
        * weekend_mult
        * out["lead_time_factor"]
    ).round(2)
    return out


def add_channel_observations(
    panel: pd.DataFrame, channels: tuple[str, ...] = CHANNELS, seed: int = SEED
) -> pd.DataFrame:
    """For each (property, day, channel) observation, draw posted_rate from
    expected_rate * channel_noise * tier_quality_term * parity_violation_term.
    """
    rng = np.random.default_rng(seed + 3)
    parts: list[pd.DataFrame] = []
    for ch in channels:
        sub = panel.copy()
        sub["channel"] = pd.Categorical([ch] * len(sub), categories=list(channels))

        # Channel-specific Gaussian noise.
        sigma = _CHANNEL_NOISE_SIGMA[ch]
        sub["channel_noise"] = rng.normal(loc=0.0, scale=sigma, size=len(sub)).round(4)

        # Per-row parity violation probability.
        tier_arr = sub["property_tier"].astype(str).to_numpy()
        base_p = np.array([_PARITY_BASE_RATE[t] for t in tier_arr])
        ch_mult = _PARITY_CHANNEL_MULT[ch]
        sub["has_rate_parity_issue"] = rng.random(len(sub)) < (base_p * ch_mult)

        # Parity violation magnitude: 0.85x to 1.20x deviation. Pulled from a uniform
        # range that's symmetric around 1.0 to keep PVar mean ~0.
        deviation = np.where(
            sub["has_rate_parity_issue"],
            rng.uniform(low=0.85, high=1.20, size=len(sub)),
            1.0,
        )

        # Supplier-quality contribution: better supplier => smaller deviation amplitude.
        quality_term = 1.0 + sub["supplier_quality_score"].to_numpy() * sub["channel_noise"]

        sub["posted_rate"] = (
            sub["expected_rate"].to_numpy()
            * (1.0 + sub["channel_noise"].to_numpy())
            * quality_term
            * deviation
        ).round(2)

        # ~1% rows: decimal-point error - rate is 10x off either way.
        bug_mask = rng.random(len(sub)) < _DECIMAL_BUG_RATE
        sign = rng.choice([0.1, 10.0], size=int(bug_mask.sum()))
        sub.loc[bug_mask, "posted_rate"] = sub.loc[bug_mask, "posted_rate"].to_numpy() * sign

        parts.append(sub)
    return pd.concat(parts, ignore_index=True)


def compute_pvar(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    out["PVar"] = ((out["posted_rate"] - out["expected_rate"]) / out["expected_rate"]).round(4)
    out["PVar_abs"] = out["PVar"].abs()
    return out


def true_coefficients() -> dict[str, float]:
    """The dict the appendix uses to validate that OLS recovers the DGP within bootstrap CI.

    Returns log-domain effects (since the regression is on ``log(PVar_abs + eps)``). The
    values are *expected magnitudes* rather than analytic exact coefficients - the DGP
    is multiplicative on rate but additive on log-PVar; the regression's recovery is
    validated within bootstrap CI in the appendix, not by exact equality.
    """
    return {
        "is_weekend": 0.05,  # weekend multiplier 1.08 -> small log effect
        "is_holiday": 0.20,  # holiday multiplier 1.30
        "log_lead_time": -0.10,  # short-lead surge dominates the negative term
        "tier_long_tail_x_channel_b": 0.55,  # the substantively interesting interaction
        "channel_b_main": 0.30,  # channel B alone (parity x noise)
        "channel_a_main": 0.15,
        "tier_long_tail_main": 0.40,
        "tier_strategic_main": -0.20,
    }


def generate(n_rows: int = 200_000, n_properties: int = 5_000, seed: int = SEED) -> pd.DataFrame:
    """End-to-end DGP. Sample down to ``n_rows`` rows uniformly and return."""
    log.info("DGP: drawing %d properties.", n_properties)
    props = draw_properties(n=n_properties, seed=seed)

    log.info("DGP: building daily panel.")
    panel = daily_panel(props, n_days=365, seed=seed)
    panel = add_lead_time_factor(panel, seed=seed)
    panel = add_expected_rate(panel)

    log.info("DGP: fanning out across channels.")
    panel = add_channel_observations(panel, seed=seed)
    panel = compute_pvar(panel)

    if n_rows < len(panel):
        rng = np.random.default_rng(seed + 7)
        idx = rng.choice(len(panel), size=n_rows, replace=False)
        panel = panel.iloc[idx].reset_index(drop=True)
    log.info("DGP: final shape=%s", panel.shape)
    return panel


__all__ = [
    "add_channel_observations",
    "add_expected_rate",
    "add_lead_time_factor",
    "compute_pvar",
    "daily_panel",
    "draw_properties",
    "generate",
    "true_coefficients",
]
