"""DGP tests: shape, deterministic seed, recovery property exists."""

from __future__ import annotations

import pytest


def test_generate_returns_expected_shape() -> None:
    pytest.importorskip("pvar_linreg.dgp")
    from pvar_linreg.dgp import generate

    try:
        df = generate(n_rows=1_000, seed=42)
    except NotImplementedError:
        pytest.skip("dgp.generate not yet implemented")

    assert len(df) == 1_000
    expected_cols = {"property_id", "property_tier", "channel", "PVar", "PVar_abs"}
    assert expected_cols.issubset(df.columns)


def test_generate_is_deterministic_under_seed() -> None:
    pytest.importorskip("pvar_linreg.dgp")
    from pvar_linreg.dgp import generate

    try:
        a = generate(n_rows=500, seed=42)
        b = generate(n_rows=500, seed=42)
    except NotImplementedError:
        pytest.skip("dgp.generate not yet implemented")

    assert a.equals(b)


def test_true_coefficients_returns_dict() -> None:
    pytest.importorskip("pvar_linreg.dgp")
    from pvar_linreg.dgp import true_coefficients

    try:
        d = true_coefficients()
    except NotImplementedError:
        pytest.skip("dgp.true_coefficients not yet implemented")

    assert isinstance(d, dict)
    assert len(d) > 0
