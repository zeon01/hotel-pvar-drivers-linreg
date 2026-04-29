"""Random 80/20 split (rows are i.i.d. across properties; this is an attribution model,
not a forecast)."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from pvar_linreg.config import SEED


def random_split(
    df: pd.DataFrame, test_size: float = 0.2, seed: int = SEED
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return train_test_split(df, test_size=test_size, random_state=seed)


__all__ = ["random_split"]
