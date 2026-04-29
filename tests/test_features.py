"""Feature transform smoke tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def test_log_lead_time_handles_zero() -> None:
    pytest.importorskip("pvar_linreg.features")
    from pvar_linreg.features import add_log_lead_time

    df = pd.DataFrame({"lead_time_days": [0, 1, 30, 365]})
    try:
        out = add_log_lead_time(df)
    except NotImplementedError:
        pytest.skip("features.add_log_lead_time not yet implemented")

    assert (out["log_lead_time"] >= 0).all()
    assert np.isfinite(out["log_lead_time"]).all()


def test_centered_star_rating_zero_at_3() -> None:
    pytest.importorskip("pvar_linreg.features")
    from pvar_linreg.features import add_centered_star_rating

    df = pd.DataFrame({"star_rating": [2.0, 3.0, 4.5]})
    try:
        out = add_centered_star_rating(df)
    except NotImplementedError:
        pytest.skip("features.add_centered_star_rating not yet implemented")

    assert out.loc[1, "star_rating_centered"] == 0.0
