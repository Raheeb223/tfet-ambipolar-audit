"""Quasi-Fermi-level drop estimate (src/audit/qf_validity): Ohm's law on a uniform sheet."""

import numpy as np
import pytest

from src.audit import qf_validity as qv
from src.constants import Q
from src.devices import tfet_2d as t


def test_uniform_sheet_is_ohms_law():
    cfg = t.load_config()
    Lg, Ld = cfg["geometry"]["L_gate_cm"], cfg["geometry"]["L_drain_cm"]
    x = np.linspace(-2e-6, Lg + Ld, 401)
    y = np.linspace(0.0, 5e-7, 11)
    p = np.full((y.size, x.size), 1e19)
    n = np.full((y.size, x.size), 1e20)
    n[:, x <= Lg] = 1e5                                 # holes are the majority in the channel
    p[:, x > Lg] = 1e3
    sol = t.Solution(x=x, y=y, psi=np.zeros_like(p), n=n, p=p, vol=np.ones_like(p), Vg=0, Vd=1, T=300.0)
    I_um = 1e-5
    dp, dn = qv.qf_drops(sol, cfg, I_um)
    I_cm = I_um * 1e4 / 2                               # one half of the double gate
    # the boundary point belongs to one path only, so each length is short by <= one cell (2%)
    assert dp == pytest.approx(I_cm * Lg / (Q * qv.MU_P * 1e19 * 5e-7), rel=2e-2)
    assert dn == pytest.approx(I_cm * Ld / (Q * qv.MU_N * 1e20 * 5e-7), rel=2e-2)


def test_depleted_stretch_is_excluded():
    """A depleted gap (both carriers ~0) adds nothing: generated carriers are swept by the field."""
    cfg = t.load_config()
    Lg, Ld = cfg["geometry"]["L_gate_cm"], cfg["geometry"]["L_drain_cm"]
    x = np.linspace(-2e-6, Lg + Ld, 401)
    y = np.linspace(0.0, 5e-7, 11)
    p = np.where(x <= Lg, 1e19, 1e3) * np.ones((y.size, 1))
    n = np.where(x <= Lg, 1e5, 1e20) * np.ones((y.size, 1))
    gap = (x > Lg - 5e-7) & (x < Lg + 5e-7)
    p[:, gap], n[:, gap] = 1e8, 1e8
    sol = t.Solution(x=x, y=y, psi=np.zeros_like(p), n=n, p=p, vol=np.ones_like(p), Vg=0, Vd=1, T=300.0)
    dp, dn = qv.qf_drops(sol, cfg, 1e-5)
    assert np.isfinite(dp) and dp < 1e-2 and dn < 1e-2
