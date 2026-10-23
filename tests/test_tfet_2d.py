"""Verification of the 2D TFET electrostatics and post-processing (HANDOFF §9 step 4)."""

import numpy as np
import pytest

from src.constants import K_B_EV
from src.devices import tfet_2d as t
from src.devsim_env import devsim as ds
from src.materials import silicon

pytestmark = pytest.mark.devsim


@pytest.fixture(scope="module")
def cfg():
    c = t.load_config()
    c["physics"]["btbt"]["model"] = "both"
    c["physics"]["tat_bulk"]["enabled"] = True
    c["physics"]["interface_traps"]["enabled"] = True
    return c


@pytest.fixture(scope="module")
def dev(cfg):
    d = t.build(cfg)
    yield d
    d.delete()


def test_equilibrium_has_no_generation(dev, cfg):
    t.ramp_to(dev, 0.0, 0.0)
    r, _ = t.generation(t.extract(dev), cfg)
    for k in ("I_btbt_local", "I_btbt_nonlocal", "I_tat_bulk", "I_it"):
        assert abs(r[k]) < 1e-25, k


def test_contact_potentials(dev, cfg):
    t.ramp_to(dev, -1.0, 1.0)
    s = t.extract(dev)
    T = cfg["temperature_K"]
    ni = float(silicon.intrinsic_density(T))
    Vt = K_B_EV * T
    Na, Nd = cfg["doping"]["source_acceptors_cm3"], cfg["doping"]["drain_donors_cm3"]
    assert s.psi[:, 0] == pytest.approx(-Vt * np.arcsinh(Na / (2 * ni)), abs=1e-6)
    assert s.psi[:, -1] == pytest.approx(1.0 + Vt * np.arcsinh(Nd / (2 * ni)), abs=1e-6)


def test_gradient_field_matches_devsim_edge_field(dev):
    t.ramp_to(dev, -1.0, 1.0)
    s = t.extract(dev)
    # horizontal edges on the interface row: DEVSIM edge field = -(dpsi/dx) at edge midpoint
    get = lambda nm: np.array(ds.get_edge_model_values(device=dev.name, region=t.SI, name=nm))
    E = get("ElectricField")
    ds.edge_from_node_model(device=dev.name, region=t.SI, node_model="x")
    ds.edge_from_node_model(device=dev.name, region=t.SI, node_model="y")
    x0, x1, y0, y1 = get("x@n0"), get("x@n1"), get("y@n0"), get("y@n1")
    row = (np.abs(y0 - s.y[3]) < 1e-14) & (np.abs(y1 - s.y[3]) < 1e-14)
    xm = 0.5 * (x0 + x1)[row]
    Ex_devsim = (E * np.sign(x1 - x0))[row]          # field along +x
    dpsi = np.gradient(s.psi[3], s.x)
    Ex_grid = -np.interp(xm, s.x, dpsi)
    big = np.abs(Ex_devsim) > 1e5
    np.testing.assert_allclose(Ex_grid[big], Ex_devsim[big], rtol=0.05)


def test_gauss_law_charge_balance(dev, cfg):
    # total charge in Si (both contacts' and gates' charges included) must vanish
    t.ramp_to(dev, -1.0, 1.0)
    q_contacts = sum(ds.get_contact_charge(device=dev.name, contact=c, equation="PotentialEquation")
                     for c in ds.get_contact_list(device=dev.name))
    s = t.extract(dev)
    Na, Nd = t.doping_numpy(cfg, s.x[None, :] + 0 * s.y[:, None])
    q_si = 1.602176634e-19 * np.sum((s.p - s.n + Nd - Na) * s.vol)
    assert abs(q_si + q_contacts) < 1e-3 * np.sum(1.602176634e-19 * np.abs(Nd - Na) * s.vol)


def test_mesh_convergence(cfg):
    out = []
    for refine in (1, 2):
        c = t.variant(cfg, **{"mesh.refine": refine})
        d = t.build(c)
        try:
            t.ramp_to(d, -1.0, 1.0)
            r, _ = t.generation(t.extract(d), c)
            out.append(r)
        finally:
            d.delete()
    for k in ("I_btbt_local", "I_btbt_nonlocal", "I_tat_bulk", "I_it"):
        assert out[1][k] == pytest.approx(out[0][k], rel=0.10), k


def test_btbt_terms_reproduce_generation(dev, cfg):
    """Parameter-free BTBT terms (used for UQ) give the same currents as generation()."""
    import copy
    t.ramp_to(dev, -1.0, 1.0)
    s = t.extract(dev)
    c = copy.deepcopy(cfg)
    c["physics"]["btbt"]["nonlocal_vertical"] = True
    terms = t.btbt_terms(s, c)
    for A, B in ((4e14, 1.9e7), (8.1e17, 2.43e7)):
        b = c["physics"]["btbt"]
        b.update(A_cm3s=A, B_Vcm=B, nonlocal_A_cm3s=A, nonlocal_B_Vcm=B)
        r, _ = t.generation(s, c)
        assert np.isclose(t.btbt_from_terms(terms["local"], A, B), r["I_btbt_local"], rtol=1e-10)
        assert np.isclose(t.btbt_from_terms(terms["nonlocal_x"], A, B), r["I_btbt_nonlocal_x"], rtol=1e-10)
        assert np.isclose(t.btbt_from_terms(terms["nonlocal_y"], A, B), r["I_btbt_nonlocal_y"], rtol=1e-10, atol=1e-30)


def test_u_shaped_dit_is_linear_and_exceeds_uniform(dev, cfg):
    """U-shape adds band-edge traps on top of the midgap density: I_it rises, stays linear in D."""
    import copy
    t.ramp_to(dev, -1.0, 1.0)
    s = t.extract(dev)
    c = copy.deepcopy(cfg)
    it = c["physics"]["interface_traps"]
    it["D_it_cm2eV"] = 1e11
    i_uni = t.generation(s, c)[0]["I_it"]
    it.update(profile="u_shape", u_ratio=50.0, u_width_eV=0.1)
    i_u1 = t.generation(s, c)[0]["I_it"]
    it["D_it_cm2eV"] = 3e11
    i_u3 = t.generation(s, c)[0]["I_it"]
    assert i_u1 >= i_uni
    assert np.isclose(i_u3, 3 * i_u1, rtol=1e-9)


def test_charged_traps_fixed_point(cfg):
    """D_it = 0 leaves the solution unchanged; D_it = 1e12 converges, and with holes
    accumulated under a negative gate the amphoteric traps are net positive (donor-like empty)."""
    import copy
    c = copy.deepcopy(cfg)
    d = t.build(c)
    try:
        t.ramp_to(d, -1.0, 1.0)
        psi0 = t.extract(d).psi.copy()
        c["physics"]["interface_traps"]["D_it_cm2eV"] = 0.0
        t.solve_charged_traps(d, c)
        assert np.max(np.abs(t.extract(d).psi - psi0)) < 1e-9
        c["physics"]["interface_traps"]["D_it_cm2eV"] = 1e12
        k = t.solve_charged_traps(d, c)
        s = t.extract(d)
        Qs = t.interface_trap_charge(s, c)
        xm = np.argmin(np.abs(s.x - 0.5 * c["geometry"]["L_gate_cm"]))
        assert k < 40 and Qs[xm] > 0
        t.clear_trap_charge(d)
        assert np.max(np.abs(t.extract(d).psi - psi0)) < 1e-6
    finally:
        d.delete()
