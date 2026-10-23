import numpy as np
import pandas as pd
import pytest
from scipy import integrate, special

from src.analysis import math_checks as mc
from src.audit import metrics
from src.physics import kane_btbt
from src.validation.ding2014 import sig2


def test_dit_half_closed_form_matches_bisection():
    rng = np.random.default_rng(1)
    for _ in range(200):
        I_t0 = 10 ** rng.uniform(-15, -8)
        I_ref0 = I_t0 * 10 ** rng.uniform(0.5, 6)
        a_t = 10 ** rng.uniform(-24, -19)
        a_ref = a_t * 10 ** rng.uniform(-1, 4)
        cf = float(mc.dit_half_closed_form(I_ref0, a_ref, I_t0, a_t))
        bis = metrics.dit_for_retention(I_ref0, a_ref, I_t0, a_t)
        S0, Sinf = np.log10(I_ref0 / I_t0), np.log10(a_ref / a_t)
        Dx = I_t0 / a_t
        m = float(mc.dit_half_from_metrics(S0, Sinf, Dx))
        assert (np.isnan(m) and np.isnan(cf)) or m == pytest.approx(cf, rel=1e-9)
        for D in (1e9, 1e11, 1e13):
            assert mc.suppression_from_metrics(D, S0, Sinf, Dx) == pytest.approx(
                mc.suppression(D, I_ref0, a_ref, I_t0, a_t), abs=1e-12)
        if Sinf >= S0 / 2:                     # never halves: no positive root
            assert np.isnan(cf)
        elif 1e6 <= cf <= 1e16:
            assert bis == pytest.approx(cf, rel=5e-6)
            assert mc.suppression(cf, I_ref0, a_ref, I_t0, a_t) == pytest.approx(S0 / 2, abs=1e-12)


def test_suppression_monotone_with_closed_form_sign():
    rng = np.random.default_rng(2)
    D = np.logspace(6, 16, 400)
    for _ in range(100):
        p = (10 ** rng.uniform(-10, -6), 10 ** rng.uniform(-22, -18),
             10 ** rng.uniform(-14, -9), 10 ** rng.uniform(-23, -19))
        dS = np.diff(mc.suppression(D, *p))
        s = mc.dS_dD_sign(*p)
        assert np.all(np.sign(dS[np.abs(dS) > 1e-14]) == s)
        # limits: S(0) = S0, S(inf) = S_inf
        assert mc.suppression(0.0, *p) == pytest.approx(np.log10(p[0] / p[2]))
        assert mc.suppression(1e40, *p) == pytest.approx(np.log10(p[1] / p[3]), abs=1e-9)


def test_scaled_upper_gamma():
    for x in (0.3, 2.0, 15.0, 30.0):
        assert mc.scaled_upper_gamma(0.5, x) == pytest.approx(
            np.exp(x) * special.gammaincc(0.5, x) * special.gamma(0.5), rel=1e-12)
        for s in (-0.5, -1.5, -2.5, -3.5):
            ref = integrate.quad(lambda u: u ** (s - 1) * np.exp(-(u - x)), x, np.inf,
                                 epsabs=0, epsrel=1e-12)[0]
            assert mc.scaled_upper_gamma(s, x) == pytest.approx(ref, rel=1e-8)
    with pytest.raises(ValueError):
        mc.scaled_upper_gamma(-1.0, 1.0)


def test_local_junction_current_closed_form():
    A, B, Vt, P = 4e17, 2.9e7, 2.1245, 2.5
    for Fm in (9e5, 1.5e6, 2.2e6):
        W = 2 * Vt / Fm
        ref = kane_btbt.Q * (W / Fm) * integrate.quad(
            lambda F: kane_btbt.kane_generation(F, A, B, P), 0, Fm, epsabs=0, epsrel=1e-12)[0]
        assert mc.local_junction_current(Fm, A, B, Vt, P) == pytest.approx(ref, rel=1e-9)


def test_reduced_mass_inverts_kane_B():
    for m in (0.1, 0.26, 0.36, 1.0):
        assert mc.reduced_mass_from_B(kane_btbt.kane_B_from_mass(1.1245, m)) == pytest.approx(m, rel=1e-8)


def test_arrhenius_round_trip():
    T = np.array([300, 360, 390, 420.0])
    k = 8.617333262e-5
    I = 3e-13 * np.exp(-0.43 / (k * T))
    assert mc.arrhenius_ea(T, I) == pytest.approx(0.43, rel=1e-6)
    assert mc.arrhenius_gain(0.43, 300, 420) == pytest.approx(I[-1] / I[0], rel=1e-6)


def test_sig2_rounds_outward():
    assert sig2(22.66, "floor") == "22" and sig2(22.66, "ceil") == "23"
    assert sig2(2654.3, "floor") == "2600" and sig2(3010.0, "ceil") == "3100"
    assert sig2(0.6756, "floor") == "0.67" and sig2(1.4035, "ceil") == "1.5"
    assert sig2(0.000124, "floor") == "0.00012" and sig2(0.00096, "ceil") == "0.00096"
    assert sig2(99.2, "ceil") == "100" and sig2(494.9, "floor") == "490"


def test_claims_vectorized_on_synthetic_ensemble():
    rows = []
    Dx = {"heterogate": 1e9, "low_drain_doping": 1e8, "gd_underlap": 5e10, "dual_wf_gate": 2e9}
    Sinf = {"heterogate": 1.1, "low_drain_doping": 3.1, "gd_underlap": 2.2, "dual_wf_gate": 0.0}
    for Vg in (-1.0, -0.6, -0.2):
        for dev in mc.TECHS:
            rows.append(dict(path="p", params="q", it="nominal", Vg=Vg, device=dev, S0=4.0,
                             Dx=Dx[dev], S_inf=Sinf[dev], **{"S_1e+11": 3.0, "S_1e+12": 3.95 if Vg == -1.0 else 2.0}))
    c = mc.claims_vectorized(pd.DataFrame(rows)).iloc[0]
    assert c.C1 and c.C2 and c.C3 and c.C4 and c.C5 and c.C6 and c.C7
