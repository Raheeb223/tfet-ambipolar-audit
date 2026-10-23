import numpy as np
import pytest

from src.constants import Q
from src.physics import kane_btbt as kb
from src.physics import nonlocal_btbt as nl

EG = 1.12
A, B, P = kb.SI_DEFAULT_A, kb.SI_DEFAULT_B, kb.P_INDIRECT


def _ramp(F, W, flat=20e-7, n=4001):
    """Ec falls linearly by F*W (eV, F in V/cm) over [0, W], flat outside."""
    x = np.linspace(-flat, W + flat, n)
    Ec = -F * np.clip(x, 0.0, W)
    return x, Ec, Ec - EG


@pytest.mark.parametrize("F,W", [(1.5e6, 20e-7), (2.5e6, 10e-7), (1.0e6, 30e-7)])
def test_uniform_field_matches_analytic(F, W):
    # analytic: J = q G(F) (W - Eg/F); local model would give q G(F) W
    x, Ec, Ev = _ramp(F, W)
    J = nl.line_current(x, Ec, Ev, A, B, EG, P, dE=2e-4)
    J_ref = Q * kb.kane_generation(F, A, B, P)[()] * (W - EG / F)
    assert J == pytest.approx(J_ref, rel=2e-3)
    assert J < kb.uniform_field_current_density(F, W, A, B, P)


def test_no_path_when_band_bending_below_gap():
    x, Ec, Ev = _ramp(0.5e6, 20e-7)   # total drop 1.0 eV < Eg
    assert nl.line_current(x, Ec, Ev, A, B, EG, P) == 0.0


def test_mirror_profile_gives_same_current():
    x, Ec, Ev = _ramp(1.5e6, 20e-7)
    J1 = nl.line_current(x, Ec, Ev, A, B, EG, P, dE=2e-4)
    J2 = nl.line_current(x, Ec[::-1], Ev[::-1], A, B, EG, P, dE=2e-4)
    assert J2 == pytest.approx(J1, rel=1e-6)


def test_zero_at_equilibrium_and_full_in_reverse_bias():
    x, Ec, Ev = _ramp(1.5e6, 20e-7)
    flat = np.zeros_like(x)
    J_eq = nl.line_current(x, Ec, Ev, A, B, EG, P, Efn=flat, Efp=flat, dE=2e-4)
    assert abs(J_eq) < 1e-12 * nl.line_current(x, Ec, Ev, A, B, EG, P, dE=2e-4)
    # strong reverse bias: hole QFL far above all valence states on the left,
    # electron QFL far below all conduction states on the right -> occupation factor 1
    J_rev = nl.line_current(x, Ec, Ev, A, B, EG, P, Efn=flat - 10.0, Efp=flat + 10.0, dE=2e-4)
    assert J_rev == pytest.approx(nl.line_current(x, Ec, Ev, A, B, EG, P, dE=2e-4), rel=1e-9)


def test_local_overestimates_more_for_short_junctions():
    # the dead-zone fraction Eg/(F W) grows as W shrinks
    F = 2e6
    ratios = []
    for W in (30e-7, 15e-7, 8e-7):
        x, Ec, Ev = _ramp(F, W)
        ratios.append(nl.line_current(x, Ec, Ev, A, B, EG, P, dE=2e-4)
                      / kb.uniform_field_current_density(F, W, A, B, P))
    assert ratios[0] > ratios[1] > ratios[2] > 0


def test_vectorized_paths_match_reference_loop():
    # a profile with two junctions (p-i-n under reverse bias with a gate dip)
    x = np.linspace(0, 100e-7, 801)
    psi = 0.6 * np.tanh((x - 30e-7) / 5e-7) + 1.2 * (1 + np.tanh((x - 70e-7) / 3e-7)) - 0.4 * np.exp(-((x - 50e-7) / 8e-7) ** 2)
    Ec, Ev = -psi + 0.56, -psi - 0.56
    E = np.arange(Ec.min() + 1e-3, Ev.max(), 7e-3)
    k, xv, xc, iv, ic = nl._all_paths(x, Ec, Ev, E)
    ref = [(kk, p) for kk, e in enumerate(E) for p in nl.paths_at_energy(x, Ec, Ev, e)]
    assert len(ref) == len(k) > 0
    for (kk, (rxv, rxc, riv, ric)), k_, xv_, xc_, iv_, ic_ in zip(ref, k, xv, xc, iv, ic):
        assert kk == k_ and (riv, ric) == (iv_, ic_)
        assert rxv == pytest.approx(xv_) and rxc == pytest.approx(xc_)


def test_path_terms_match_line_current():
    """J from (F_eff, weight) terms equals line_current for arbitrary (A, B, P)."""
    x = np.linspace(0, 40e-7, 801)
    Ec = 1.12 * np.tanh((20e-7 - x) / 6e-7)
    Ev = Ec - 1.12
    for A, B, P in ((4e14, 1.9e7, 2.5), (8e17, 2.4e7, 2.5), (1e15, 1.5e7, 2.0)):
        J = nl.line_current(x, Ec, Ev, A, B, 1.12, P, dE=1e-3)
        F, w = nl.path_terms(x, Ec, Ev, 1.12, dE=1e-3)
        J2 = np.sum(Q * A * F ** (P - 1) * np.exp(-B / F) * w)
        assert J > 0 and np.isclose(J, J2, rtol=1e-12)
