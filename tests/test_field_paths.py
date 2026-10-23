"""Field-direction nonlocal BTBT (src/physics/nonlocal_field_paths) on a synthetic junction
stripe: a linear potential ramp of width W along direction alpha, flat elsewhere."""

import numpy as np
import pytest

from src.devices import tfet_2d as t
from src.materials import silicon
from src.physics import nonlocal_btbt as nb
from src.physics import nonlocal_field_paths as fp

A, B, P = 8.1e17, 2.43e7, 2.5
L, W, V = 30e-7, 6e-7, 1.8          # box (cm), ramp width (cm), ramp height (V): F = 3 MV/cm
EG = float(silicon.bandgap(300.0))


def stripe(alpha_deg, n=151):
    x = np.linspace(0.0, L, n)
    X, Y = np.meshgrid(x, x)
    a = np.radians(alpha_deg)
    s = (X - L / 2) * np.cos(a) + (Y - L / 2) * np.sin(a)
    psi = V * np.clip(s / W + 0.5, 0.0, 1.0)
    one = np.ones_like(psi)
    return t.Solution(x=x, y=x.copy(), psi=psi, n=one, p=one.copy(), vol=one.copy(),
                      Vg=0.0, Vd=-10.0, T=300.0)


@pytest.fixture(scope="module")
def cfg():
    return t.load_config()


@pytest.fixture(scope="module")
def j1d():
    """1D current density (A/cm^2) of the same ramp: full occupation contrast."""
    x = np.linspace(-L / 2, L / 2, 6001)
    psi = V * np.clip(x / W + 0.5, 0.0, 1.0)
    return nb.line_current(x, -psi + 0.5 * EG, -psi - 0.5 * EG, A, B, EG, P,
                           Efn=np.full(x.shape, 10.0), Efp=np.zeros(x.shape), kT=0.02585, dE=2e-3)


def _fd(sol, cfg, n_bins=12):
    return fp.current(fp.field_direction_terms(sol, cfg, n_bins=n_bins, h=2.5e-8, halves=1), A, B)


def test_axis_aligned_equals_1d(cfg, j1d):
    """Field along x: every path is in the 0-degree family, current = J_1D * stripe length."""
    I = _fd(stripe(0.0), cfg)
    assert I == pytest.approx(j1d * L * 1e-4, rel=0.02)


def test_rotation_invariance(cfg):
    I0 = _fd(stripe(0.0), cfg)
    assert _fd(stripe(90.0), cfg) == pytest.approx(I0, rel=0.02)


def test_oblique_field_not_lost(cfg):
    """At 45 degrees (a bin centre for 12 bins) every path belongs to the 45-degree family, so
    the field-direction current equals that family's current; the rows/columns max sees only
    F cos(45) and collapses."""
    from src.validation.biswas_diagnostics import angled_current
    s = stripe(45.0)
    I = _fd(s, cfg)
    c = t.variant(cfg)
    c["physics"]["btbt"].update(model="nonlocal", A_cm3s=A, B_Vcm=B, nonlocal_A_cm3s=A,
                                nonlocal_B_Vcm=B, nonlocal_vertical=True)
    family = 0.5 * angled_current(s, c, 45.0, x_win=(0.0, L), h=2.5e-8)   # one half
    assert I == pytest.approx(family, rel=0.03)
    terms = t.btbt_terms(s, c)
    rows_cols = 0.5 * max(t.btbt_from_terms(terms["nonlocal_x"], A, B),
                          t.btbt_from_terms(terms["nonlocal_y"], A, B))
    assert abs(I) > 3.0 * abs(rows_cols)


def test_bins_converge(cfg):
    s = stripe(30.0)
    I12, I24 = _fd(s, cfg, 12), _fd(s, cfg, 24)
    assert I24 == pytest.approx(I12, rel=0.05)


def test_optimal_direction_bounds(cfg):
    """Angle-complete upper bound: equals the field-direction model for a straight stripe (the
    shortest path is along the field) and is >= it on a bent band diagram."""
    s = stripe(30.0)
    opt = fp.current(fp.optimal_direction_terms(s, cfg, n_angles=12, h=2.5e-8, halves=1,
                                                A_sel=A, B_sel=B), A, B)
    assert opt == pytest.approx(_fd(s, cfg), rel=0.05)
    x = np.linspace(0.0, L, 121)
    X, Y = np.meshgrid(x, x)
    r = np.hypot(X, Y)                                  # quarter-circle junction: curved field
    psi = V * np.clip((r - 12e-7) / W, 0.0, 1.0)
    one = np.ones_like(psi)
    bent = t.Solution(x=x, y=x.copy(), psi=psi, n=one, p=one.copy(), vol=one.copy(), Vg=0.0,
                      Vd=-10.0, T=300.0)
    I_fd = abs(_fd(bent, cfg, 12))
    I_opt = abs(fp.current(fp.optimal_direction_terms(bent, cfg, n_angles=12, h=2.5e-8, halves=1,
                                                      A_sel=A, B_sel=B), A, B))
    assert I_opt >= 0.99 * I_fd > 0
