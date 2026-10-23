"""Angled nonlocal paths (src/validation/biswas_diagnostics.angled_current): for bands that
vary only along x, the 0-degree family must reproduce the row integral, and tilted families
must carry less current (they see the field component F cos(theta) over a longer path)."""

import numpy as np

from src.devices import tfet_2d as t
from src.materials import silicon
from src.physics import nonlocal_btbt
from src.validation import biswas_diagnostics as bd

CFG = {"geometry": {"single_gate": True},
       "bias": {"V_source": 0.0},
       "physics": {"btbt": {"nonlocal_A_cm3s": 8.1e17, "nonlocal_B_Vcm": 2.43e7, "P": 2.5,
                            "nonlocal_dE_eV": 2e-3}}}


def _solution():
    x = np.linspace(-30e-7, 30e-7, 601)
    y = np.linspace(0.0, 10e-7, 51)
    Eg = float(silicon.bandgap(300.0))
    F = 1.5e6                                        # V/cm, uniform lateral field
    psi_x = np.clip(F * x, -0.9, 0.9)                # band bending of ~1.8 eV over 12 nm
    psi = np.tile(psi_x, (len(y), 1))
    one = np.ones_like(psi)
    return t.Solution(x=x, y=y, psi=psi, n=one, p=one, vol=one, Vg=0.0, Vd=1.0, T=300.0), Eg


def test_zero_angle_matches_rows(monkeypatch):
    monkeypatch.setattr(t, "single_gate", lambda cfg: True)
    s, Eg = _solution()
    Ec, Ev = -s.psi[0] + 0.5 * Eg, -s.psi[0] - 0.5 * Eg
    J = nonlocal_btbt.line_current(s.x, Ec, Ev, 8.1e17, 2.43e7, Eg, 2.5,
                                   Efn=np.full(s.x.shape, -1.0), Efp=np.zeros(s.x.shape),
                                   kT=0.02585, dE=2e-3)
    expected = t._per_um_base(J * (s.y[-1] - s.y[0]), 1)
    got = bd.angled_current(s, CFG, 0, x_win=(-30e-7, 30e-7), h=2e-8, ds=1e-8)
    assert expected > 0
    assert abs(got / expected - 1) < 0.05


def test_tilted_paths_carry_less_for_1d_bands(monkeypatch):
    monkeypatch.setattr(t, "single_gate", lambda cfg: True)
    s, _ = _solution()
    I0 = bd.angled_current(s, CFG, 0, x_win=(-30e-7, 30e-7))
    I30 = bd.angled_current(s, CFG, 30, x_win=(-30e-7, 30e-7))
    assert 0 < I30 < I0
