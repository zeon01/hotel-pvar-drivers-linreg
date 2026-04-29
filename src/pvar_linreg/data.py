"""Persist + load the synthetic dataset. The DGP itself lives in ``dgp.py``."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from pvar_linreg.config import PROCESSED_DIR, RAW_DIR


def write_synthetic(n_rows: int = 200_000, seed: int = 42) -> Path:
    """Run the DGP and write the result to ``data/raw/synthetic_pvar.parquet``."""
    raise NotImplementedError("Phase 4: from pvar_linreg.dgp import generate; generate(...).to_parquet(...)")


def load(path: Path | None = None) -> pd.DataFrame:
    """Read the synthetic parquet."""
    raise NotImplementedError("Phase 4")


def ensure_available(n_rows: int = 200_000, seed: int = 42) -> Path:
    """Idempotent: generate the synthetic dataset if it doesn't already exist."""
    raise NotImplementedError("Phase 4")


__all__ = ["ensure_available", "load", "write_synthetic"]
