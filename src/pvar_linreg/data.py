"""Persist + load the synthetic dataset. The DGP itself lives in ``dgp.py``."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from pvar_linreg.config import RAW_DIR, ROWS_SAMPLE_DEFAULT, SEED
from pvar_linreg.dgp import generate

log = logging.getLogger(__name__)

RAW_PARQUET: str = "synthetic_pvar.parquet"


def write_synthetic(n_rows: int = ROWS_SAMPLE_DEFAULT, seed: int = SEED) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out = RAW_DIR / RAW_PARQUET
    df = generate(n_rows=n_rows, seed=seed)
    df.to_parquet(out, index=False)
    log.info("Wrote %s (%d rows).", out, len(df))
    return out


def load(path: Path | None = None) -> pd.DataFrame:
    p = path or (RAW_DIR / RAW_PARQUET)
    if not p.exists():
        ensure_available()
    return pd.read_parquet(p)


def ensure_available(n_rows: int = ROWS_SAMPLE_DEFAULT, seed: int = SEED) -> Path:
    out = RAW_DIR / RAW_PARQUET
    if out.exists():
        return out
    return write_synthetic(n_rows=n_rows, seed=seed)


__all__ = ["RAW_PARQUET", "ensure_available", "load", "write_synthetic"]
