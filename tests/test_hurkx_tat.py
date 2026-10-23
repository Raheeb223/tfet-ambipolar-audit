import numpy as np
import pytest

from src.physics import hurkx_tat as ht

T = 300.0
DE = 0.56  # eV, ~midgap trap in Si


def test_gamma_vanishes_at_low_field():
    g = ht.gamma_integral([0.0, 1e2, 1e3, 1e4], DE, T)
    assert g[0] == 0.0
    assert np.all(np.diff(g) > 0)
    assert g[1] < 1e-2


def test_gamma_monotonic_in_field():
    F = np.logspace(3, 6, 25)
    assert np.all(np.diff(ht.gamma_integral(F, DE, T)) > 0)


def test_gamma_low_field_asymptote():
    # K -> inf: integral ~ Gamma(5/3) K^(-2/3)  =>  Gamma ~ a * Gamma(5/3) * K^(-2/3)
    from scipy.special import gamma as Gfun
    F = 10.0
    a = DE / (ht.K_B_EV * T)
    K = ht._K(F, DE, ht.M_HURKX)
    assert ht.gamma_integral(F, DE, T)[0] == pytest.approx(a * Gfun(5 / 3) * K ** (-2 / 3), rel=5e-3)


def test_closed_form_matches_integral_in_validity_range():
    F = np.array([3e5, 5e5, 7e5])
    u0 = ht.saddle_point_u0(F, DE, T)
    assert np.all((u0 > 0.05) & (u0 < 0.8))
    np.testing.assert_allclose(ht.gamma_approx(F, T), ht.gamma_integral(F, DE, T), rtol=0.15)


def test_field_gamma_value():
    # sqrt(24 * 0.25 m0 * (kT)^3)/(q hbar) at 300 K ~ 3.7e5 V/cm (hand calc)
    assert ht.field_gamma(300.0) == pytest.approx(3.69e5, rel=1e-2)


def test_reduces_to_plain_srh():
    n, p, ni = 1e12, 1e5, 1e10
    plain = (n * p - ni**2) / (1e-6 * (n + ni) + 1e-6 * (p + ni))
    assert ht.hurkx_srh_rate(n, p, ni, 1e-6, 1e-6, ni, ni) == pytest.approx(plain)


def test_field_enhancement_increases_generation():
    args = (1e4, 1e4, 1e10, 1e-6, 1e-6, 1e10, 1e10)
    r0 = ht.hurkx_srh_rate(*args)
    r1 = ht.hurkx_srh_rate(*args, gamma_n=10.0, gamma_p=10.0)
    assert r0 < 0 and r1 == pytest.approx(11.0 * r0)


def test_zero_at_equilibrium():
    ni = 1e10
    assert ht.hurkx_srh_rate(1e15, ni**2 / 1e15, ni, 1e-6, 1e-6, ni, ni, 5.0, 5.0) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("dE", [0.2, 0.45, 0.56, 0.7])
@pytest.mark.parametrize("T_", [200.0, 300.0, 400.0])
def test_gamma_fast_matches_integral_at_all_fields(dE, T_):
    # includes the high-field u0 > 1 regime where the closed form fails by ~1e7
    F = np.logspace(2, 7, 60)
    np.testing.assert_allclose(ht.gamma_fast(F, dE, T_), ht.gamma_integral(F, dE, T_), rtol=1e-5)


def test_gamma_fast_zero_field_and_shape():
    F = np.array([[0.0, 1e5], [2e6, 0.0]])
    g = ht.gamma_fast(F, DE, T)
    assert g.shape == F.shape and g[0, 0] == 0.0 and g[1, 1] == 0.0 and g[1, 0] > g[0, 1] > 0


def test_gamma_fast_array_dE_and_zero_dE():
    F = np.array([1e6, 1e6, 1e6])
    g = ht.gamma_fast(F, np.array([0.0, 0.3, 0.56]), T)
    assert g[0] == 0.0
    assert g[1] == pytest.approx(ht.gamma_integral(1e6, 0.3, T)[0], rel=1e-5)


def test_tunneling_energy_range_branches():
    ni, Nc, kT = 1e10, 2.86e19, ht.K_B_EV * T
    # depleted (n << ni): full band-to-trap distance
    assert ht.tunneling_energy_range(1e3, ni, Nc, kT, 0.56) == pytest.approx(0.56)
    # degenerate-ish n: kT ln(Nc/n) < Ec - Et
    assert ht.tunneling_energy_range(1e19, ni, Nc, kT, 0.56) == pytest.approx(kT * np.log(Nc / 1e19))
    assert ht.tunneling_energy_range(1e20, ni, Nc, kT, 0.56) == 0.0
