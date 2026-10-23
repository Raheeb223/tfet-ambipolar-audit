"""Calibration of Kane (A, B) on junction J(F) data (src/calibration/btbt_calibration.py)."""

import numpy as np
import pytest

from src.calibration import btbt_calibration as bc
from src.physics import kane_btbt


def test_local_junction_current_matches_direct_integral():
    """J_local = q * integral G(F(x)) dx over the linear field of an abrupt junction."""
    Fm, Vt, A, B = 1.5e6, 2.1, 4e14, 1.9e7
    x, Ec, Ev, W = bc.junction_bands(Fm, Vt, n=20001)
    F = np.abs(np.gradient(Ec, x))
    J_direct = kane_btbt.integrated_current_density(x, F, A, B)
    assert np.isclose(bc.j_local(Fm, A, B, Vt)[0], J_direct, rtol=2e-3)


def test_nonlocal_below_local_in_a_junction():
    """Nonlocal current is smaller: it removes the dead zone and uses the path-averaged field."""
    F = np.array([1.0e6, 1.5e6, 2.0e6])
    jl = bc.j_local(F, 4e14, 1.9e7, 2.12)
    jn = bc.j_nonlocal(F, 4e14, 1.9e7, 2.12)
    assert np.all(jn < jl) and np.all(jn > 0)


def test_fit_recovers_synthetic_parameters():
    F = np.linspace(0.9e6, 2.3e6, 8)
    for model in ("local", "nonlocal"):
        J = bc.MODELS[model](F, 3e17, 2.6e7, 2.1245)
        r = bc.fit(F, J, model)
        assert abs(r["log10A"] - np.log10(3e17)) < 1e-3
        assert abs(r["B"] / 2.6e7 - 1) < 1e-4
        assert r["rms_dec"] < 1e-6


def test_reduced_mass_round_trip():
    B = kane_btbt.kane_B_from_mass(bc.EG_300, 0.29)
    assert np.isclose(bc.reduced_mass_from_B(B), 0.29)


def test_fit_recovers_parameters_at_other_exponent():
    """The structural alternatives refit (A, B) at P = 3: synthetic data made with P = 3 are
    recovered, and the default P = 2.5 fit of the same data is visibly worse."""
    F = np.linspace(1.0e6, 2.2e6, 8)
    J = bc.j_local(F, 3e18, 2.6e7, bc.EG_300 + 1.0, P=3.0)
    r = bc.fit(F, J, "local", P=3.0)
    assert r["A"] == pytest.approx(3e18, rel=1e-3) and r["B"] == pytest.approx(2.6e7, rel=1e-4)
    assert bc.fit(F, J, "local")["rms_dec"] > 10 * r["rms_dec"]
