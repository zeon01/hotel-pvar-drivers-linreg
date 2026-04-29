"""Diagnostics smoke tests — full validation against statsmodels output is in the
appendix notebook."""

from __future__ import annotations

import pytest


def test_compute_vif_signature() -> None:
    pytest.importorskip("pvar_linreg.diagnostics")
    from pvar_linreg.diagnostics import compute_vif

    import pandas as pd

    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [1.5, 2.5, 3.5, 4.5]})
    try:
        out = compute_vif(X)
    except NotImplementedError:
        pytest.skip("diagnostics.compute_vif not yet implemented")

    assert "vif" in {c.lower() for c in out.columns} or "VIF" in out.columns
