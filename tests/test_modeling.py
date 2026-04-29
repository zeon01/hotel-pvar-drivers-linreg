"""Modeling smoke tests."""

from __future__ import annotations

import pytest


def test_random_split_proportions() -> None:
    import pandas as pd

    from pvar_linreg.modeling.splits import random_split

    df = pd.DataFrame({"x": range(1000)})
    train, test = random_split(df, test_size=0.2, seed=42)
    assert len(train) == 800
    assert len(test) == 200


def test_fit_ols_implementation_exists() -> None:
    pytest.importorskip("pvar_linreg.modeling.train")
    from pvar_linreg.modeling.train import FORMULA

    assert "log_pvar_abs" in FORMULA.lower() or "PVar_abs" in FORMULA
    assert "C(property_tier)" in FORMULA
