"""DEVSIM verification of the 1D p+/n+ diode (HANDOFF §9 step 1)."""

import contextlib
import copy
import io

import numpy as np
import pytest

from src.constants import EPS0_F_PER_CM, K_B_EV, Q
from src.materials import silicon
from src.physics import kane_btbt

pytestmark = pytest.mark.devsim

d = pytest.importorskip("src.devices.diode_1d")


def _run(cfg):
    with contextlib.redirect_stdout(io.StringIO()):
        return d.run(cfg)


@pytest.fixture(scope="module")
def cfg():
    c = d.load_config("configs/diode_1d.yaml")
    c["sweep"].update(anode_stop_V=-1.0, step_V=0.1)
    return c


@pytest.fixture(scope="module")
def btbt_run(cfg):
    return _run(cfg)


@pytest.fixture(scope="module")
def srh_run(cfg):
    c = copy.deepcopy(cfg)
    c["btbt"]["enabled"] = False
    return _run(c)


def test_converged_everywhere(btbt_run, srh_run):
    assert btbt_run.failed_V == [] and srh_run.failed_V == []
    assert btbt_run.V[-1] == pytest.approx(-1.0)


def test_zero_current_at_equilibrium(btbt_run):
    assert abs(btbt_run.J[0]) < 1e-20  # A/cm^2; detailed-balance factor kills BTBT at 0 V


def _short_diode_saturation(cfg, V):
    """q ni^2 (Dn/(Na Wp) + Dp/(Nd Wn)): minority diffusion to ohmic contacts (short base).

    Wp, Wn are the quasi-neutral widths: contact distance minus the depletion edge at bias V.
    """
    T = cfg["temperature_K"]
    ni, Vt = silicon.intrinsic_density(T), K_B_EV * T
    g, dop, mob = cfg["geometry"], cfg["doping"], cfg["mobility"]
    Na, Nd = dop["acceptors_cm3"], dop["donors_cm3"]
    eps = silicon.EPS_R * EPS0_F_PER_CM
    W = np.sqrt(2 * eps * (Vt * np.log(Na * Nd / ni**2) - V - 2 * Vt) * (Na + Nd) / (Q * Na * Nd))
    Wp = g["length_p_cm"] - W * Nd / (Na + Nd)
    Wn = g["length_n_cm"] - W * Na / (Na + Nd)
    return Q * ni**2 * (mob["mu_n"] * Vt / (Na * Wp) + mob["mu_p"] * Vt / (Nd * Wn))


def _balance_residual(r):
    return np.array(r.J) + np.array(r.J_btbt_int) + np.array(r.J_srh_int)


def test_current_balance_srh(srh_run, cfg):
    # J + q*int(-U_srh) = -J_diffusion: what is left is exactly the short-base diffusion current
    V = np.array(srh_run.V)
    np.testing.assert_allclose(-_balance_residual(srh_run)[1:], _short_diode_saturation(cfg, V[1:]), rtol=0.03)


def test_current_balance_btbt(btbt_run):
    # Contact current equals the integrated BTBT + SRH generation (diffusion is ~1e-11 relative here)
    J = np.array(btbt_run.J)
    np.testing.assert_array_less(np.abs(_balance_residual(btbt_run)[1:]), 1e-6 * np.abs(J[1:]))


def test_equilibrium_peak_field_matches_depletion_approximation(srh_run, cfg):
    T = cfg["temperature_K"]
    Na, Nd = cfg["doping"]["acceptors_cm3"], cfg["doping"]["donors_cm3"]
    ni = silicon.intrinsic_density(T)
    Vt = K_B_EV * T
    Vbi = Vt * np.log(Na * Nd / ni**2)
    eps = silicon.EPS_R * EPS0_F_PER_CM
    F_dep = np.sqrt(2 * Q * (Vbi - 2 * Vt) * Na * Nd / (eps * (Na + Nd)))
    assert srh_run.F_max[0] == pytest.approx(F_dep, rel=0.01)


def test_devsim_btbt_matches_numpy_kane(cfg):
    # Raw symbolic expression vs numpy, detailed-balance factor off.
    c = copy.deepcopy(cfg)
    c["btbt"]["detailed_balance_factor"] = False
    r = _run(c)
    b = c["btbt"]
    x, F, G = np.array(r.x_edge), np.array(r.F_edge), np.array(r.G_edge)
    G_np = kane_btbt.kane_generation(F, b["A_cm3s"], b["B_Vcm"], b["P"])
    m = G_np > 1e-6 * G_np.max()  # F_smooth only matters where G is ~1e-60 and below
    np.testing.assert_allclose(G[m], G_np[m], rtol=1e-4)  # residual ~1e-5 from |F| smoothing
    J_np = kane_btbt.integrated_current_density(x, F, b["A_cm3s"], b["B_Vcm"], b["P"])
    assert J_np == pytest.approx(r.J_btbt_int[-1], rel=0.01)


def test_detailed_balance_factor_effect_is_bounded(btbt_run, cfg):
    # The factor (ni^2-np)/(ni^2+np) is pulled below 1 by BTBT-generated carriers in the
    # depletion region. Measured: -6 % on J at -1 V for this device. Guard against it growing.
    b = cfg["btbt"]
    x, F = np.array(btbt_run.x_edge), np.array(btbt_run.F_edge)
    J_raw = kane_btbt.integrated_current_density(x, F, b["A_cm3s"], b["B_Vcm"], b["P"])
    assert 0.85 < btbt_run.J_btbt_int[-1] / J_raw <= 1.0


def test_btbt_dominates_srh(btbt_run, srh_run):
    assert abs(btbt_run.J[-1]) > 1e6 * abs(srh_run.J[-1])


def test_mesh_convergence(cfg, btbt_run):
    c = copy.deepcopy(cfg)
    c["mesh"]["refine"] = 2
    fine = _run(c)
    assert fine.J[-1] == pytest.approx(btbt_run.J[-1], rel=0.01)
    assert fine.F_max[-1] == pytest.approx(btbt_run.F_max[-1], rel=0.01)
