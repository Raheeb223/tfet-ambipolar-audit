import numpy as np
import pytest

from src.constants import Q
from src.materials import silicon
from src.physics import kane_btbt as kb


def test_zero_and_negative_field():
    G = kb.kane_generation([0.0, -1e6, 1e6], A=1e15, B=2e7)
    assert G[0] == 0.0
    assert G[1] == pytest.approx(G[2])  # depends on |F| only


def test_hand_value():
    # A=4e14, B=1.9e7, P=2.5, F=1e6 V/cm: 4e14 * 1e15 * exp(-19)
    expected = 4e14 * (1e6) ** 2.5 * np.exp(-19.0)
    assert kb.kane_generation(1e6, kb.SI_DEFAULT_A, kb.SI_DEFAULT_B, 2.5) == pytest.approx(expected, rel=1e-12)


def test_exponent_P_slope():
    # d ln G / d ln F = P + B/F
    A, B, P, F = 1e15, 2e7, 2.5, 1.5e6
    h = 1e-6
    lnG = lambda f: np.log(kb.kane_generation(f, A, B, P))
    slope = (lnG(F * (1 + h)) - lnG(F * (1 - h))) / (np.log(1 + h) - np.log(1 - h))
    assert slope == pytest.approx(P + B / F, rel=1e-6)


def test_B_from_mass_si():
    # pi sqrt(0.29 m0) (1.12 eV)^1.5 / (2 sqrt2 q hbar) = 2.57e7 V/cm (hand calc)
    assert kb.kane_B_from_mass(1.12, 0.29) == pytest.approx(2.568e7, rel=2e-3)
    # B scales as sqrt(m) and Eg^1.5
    assert kb.kane_B_from_mass(1.12, 0.33) / kb.kane_B_from_mass(1.12, 0.29) == pytest.approx(np.sqrt(0.33 / 0.29))
    assert kb.kane_B_from_mass(0.66, 0.29) / kb.kane_B_from_mass(1.12, 0.29) == pytest.approx((0.66 / 1.12) ** 1.5)


def test_temperature_scaling_raises_btbt_with_T():
    # Eg shrinks with T -> B shrinks -> more tunneling (negative TC of Zener voltage)
    Eg300, Eg400 = silicon.bandgap(300.0), silicon.bandgap(400.0)
    A, B400 = kb.kane_params_at_T(4e14, 1.9e7, Eg300, Eg400)
    assert A == 4e14 and B400 < 1.9e7
    assert kb.kane_generation(1e6, A, B400) > kb.kane_generation(1e6, 4e14, 1.9e7)


def test_uniform_field_current_units():
    # J = q G W ; W = 10 nm = 1e-6 cm
    F, W = 1.2e6, 1e-6
    J = kb.uniform_field_current_density(F, W, 4e14, 1.9e7)
    assert J == pytest.approx(Q * kb.kane_generation(F, 4e14, 1.9e7) * W)


def test_integrated_equals_uniform_for_flat_profile():
    x = np.linspace(0, 1e-6, 101)
    F = np.full_like(x, 1.2e6)
    assert kb.integrated_current_density(x, F, 4e14, 1.9e7) == pytest.approx(
        kb.uniform_field_current_density(1.2e6, 1e-6, 4e14, 1.9e7), rel=1e-12)
