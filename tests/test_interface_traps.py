"""Interface-trap generation (tfet_2d.interface_generation): Hurkx pairing, trap spectra,
capture cross-sections, tunneling mass and gate-edge damage, on a synthetic depleted interface
with a uniform field (no DEVSIM solve)."""

import copy

import numpy as np
import pytest

from src.constants import K_B_EV, Q
from src.devices import tfet_2d as t
from src.materials import silicon
from src.physics import hurkx_tat

T = 300.0
F0 = 1.0e6                       # V/cm, uniform interface field


@pytest.fixture(scope="module")
def setup():
    cfg = t.load_config()
    it = cfg["physics"]["interface_traps"]
    it.update(enabled=True, D_it_cm2eV=1e12, n_levels=41)
    x = np.linspace(0.0, 1e-6, 51)
    y = np.array([0.0, 1e-7])
    ni = float(silicon.intrinsic_density(T))
    dep = np.full((2, x.size), 1e-3 * ni)            # fully depleted: n, p << ni
    sol = t.Solution(x=x, y=y, psi=np.zeros((2, x.size)), n=dep, p=dep.copy(),
                     vol=np.ones((2, x.size)), Vg=0.0, Vd=0.0, T=T)
    consts = (ni, float(silicon.bandgap(T)), float(silicon.nc(T)), float(silicon.nv(T)), K_B_EV * T)
    return cfg, sol, consts


def _I(cfg, sol, consts, **it_changes):
    c = copy.deepcopy(cfg)
    c["physics"]["interface_traps"].update(it_changes)
    return t.interface_generation(sol, c, np.full(sol.x.size, F0), *consts)


def test_single_level_matches_hurkx_closed_form(setup):
    """A narrow Gaussian level 0.3 eV above midgap: the n1 (hole-emission) term must carry the
    hole enhancement Gamma_p(Et - Ev) and the p1 term Gamma_n(Ec - Et) [Hurkx 1992]."""
    cfg, sol, consts = setup
    ni, Eg, Nc, Nv, kT = consts
    e0, sn, sp = 0.3, 2e-15, 5e-16
    I = _I(cfg, sol, consts, profile="gaussian", gauss_center_eV=e0, gauss_width_eV=0.004,
           n_levels=2001, sigma_n_cm2=sn, sigma_p_cm2=sp)
    N = 1e12 * Eg                                       # all traps in the one level
    v = cfg["physics"]["interface_traps"]["v_th_cms"]
    n1, p1 = ni * np.exp(e0 / kT), ni * np.exp(-e0 / kT)
    gn = hurkx_tat.gamma_fast(np.array([F0]), np.array([0.5 * Eg - e0]), T)[0]
    gp = hurkx_tat.gamma_fast(np.array([F0]), np.array([0.5 * Eg + e0]), T)[0]
    n = p = 1e-3 * ni
    R = N * v * (ni**2 - n * p) / ((n + n1) / (sp * (1 + gp)) + (p + p1) / (sn * (1 + gn)))
    expected = Q * R * (sol.x[-1] - sol.x[0])
    assert I == pytest.approx(expected, rel=2e-3)
    swapped = N * v * (ni**2 - n * p) / ((n + n1) / (sp * (1 + gn)) + (p + p1) / (sn * (1 + gp)))
    assert R / swapped > 10.0                           # the pairing is not a detail here


def test_gaussian_keeps_total_trap_number(setup):
    cfg, _, consts = setup
    Eg = consts[1]
    it = dict(cfg["physics"]["interface_traps"], profile="gaussian", gauss_center_eV=-0.2,
              gauss_width_eV=0.05, n_levels=801)
    Et, dE = t.interface_levels(it, Eg)
    assert np.sum(t.dit_shape(it, Et, Eg)) * dE == pytest.approx(Eg, rel=1e-3)


def test_cross_section_scaling(setup):
    """Equal sigma: current proportional to sigma (D_x scales as 1/sigma, S_inf unchanged).
    Unequal sigma_n, sigma_p at the same geometric mean: not a function of the product only."""
    cfg, sol, consts = setup
    I1 = _I(cfg, sol, consts)
    assert _I(cfg, sol, consts, sigma_cm2=1e-14) == pytest.approx(10 * I1, rel=1e-10)
    lvl = dict(profile="gaussian", gauss_center_eV=0.3, gauss_width_eV=0.02, n_levels=401)
    Isym = _I(cfg, sol, consts, **lvl)
    Iasym = _I(cfg, sol, consts, sigma_n_cm2=1e-14, sigma_p_cm2=1e-16, **lvl)
    assert abs(np.log10(Iasym / Isym)) > 0.5


def test_damage_profile_limits(setup):
    cfg, sol, consts = setup
    I1 = _I(cfg, sol, consts)
    assert _I(cfg, sol, consts, damage_ratio=1.0) == pytest.approx(I1, rel=1e-12)
    assert _I(cfg, sol, consts, damage_ratio=10.0, damage_length_cm=1.0) == pytest.approx(10 * I1, rel=1e-6)


def test_field_enhancement_ordering(setup):
    """Heavier tunneling mass -> less enhancement; no enhancement -> plain SRH (smallest)."""
    cfg, sol, consts = setup
    I_025 = _I(cfg, sol, consts)
    I_05 = _I(cfg, sol, consts, m_tunnel=0.5)
    I_srh = _I(cfg, sol, consts, field_enhanced=False)
    assert I_025 > I_05 > I_srh > 0


def test_gate_only_removes_spacer_traps(setup):
    """gate_only zeroes D_it outside the gate span; inside it the density is unchanged."""
    cfg = setup[0]
    it = dict(cfg["physics"]["interface_traps"], gate_only=True)
    x = np.linspace(-2e-6, 8e-6, 101)
    f = t.dit_spatial(it, x, cfg)
    g0, g1 = t.gate_span(cfg)
    inside = (x >= g0) & (x <= g1)
    assert inside.any() and (~inside).any()
    assert np.all(f[inside] == 1.0) and np.all(f[~inside] == 0.0)
