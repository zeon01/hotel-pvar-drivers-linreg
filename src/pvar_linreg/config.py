"""Project paths, seeds, DGP defaults."""

from __future__ import annotations

from pathlib import Path

ROOT: Path = Path(__file__).resolve().parents[2]

DATA_DIR: Path = ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
INTERIM_DIR: Path = DATA_DIR / "interim"
PROCESSED_DIR: Path = DATA_DIR / "processed"
EXTERNAL_DIR: Path = DATA_DIR / "external"

REPORTS_DIR: Path = ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
DOCS_FIGURES_DIR: Path = ROOT / "docs" / "figures"
MODELS_DIR: Path = ROOT / "models"

SEED: int = 42

# DGP defaults — full run is ~5.5M rows, sampling default is 200k for speed
N_PROPERTIES_FULL: int = 5_000
N_PROPERTIES_SAMPLE: int = 500
N_DAYS: int = 365
N_CHANNELS: int = 3
ROWS_SAMPLE_DEFAULT: int = 200_000

PROPERTY_TIERS: tuple[str, ...] = ("Strategic", "Preferred", "Standard", "Long-tail")
TIER_PROBABILITIES: tuple[float, ...] = (0.10, 0.20, 0.40, 0.30)

CHANNELS: tuple[str, ...] = ("Direct", "ChannelManagerA", "ChannelManagerB")

COUNTRIES: tuple[str, ...] = (
    "TH",
    "VN",
    "ID",
    "MY",
    "PH",
    "SG",
    "JP",
    "KR",
    "GB",
    "DE",
    "FR",
    "OTHER",
)

# Outcome columns
TARGET_SIGNED: str = "PVar"
TARGET_ABS: str = "PVar_abs"
