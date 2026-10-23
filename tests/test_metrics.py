import numpy as np
import pytest

from src.audit import metrics as m


def test_suppression_and_retention():
    S_base = m.suppression_decades(1e-6, 1e-10)       # 4 decades claimed
    S_real = m.suppression_decades(1e-8, 1e-10)       # 2 decades real
    assert S_base == pytest.approx(4.0)
    assert m.retention(S_real, S_base) == pytest.approx(0.5)


def test_retention_negative_when_technique_hurts():
    assert m.retention(m.suppression_decades(1e-9, 2e-9), 2.0) < 0


def test_retention_rejects_zero_baseline_and_bad_currents():
    with pytest.raises(ValueError):
        m.retention(1.0, 0.0)
    with pytest.raises(ValueError):
        m.suppression_decades(0.0, 1e-9)


def test_dominant_mechanism():
    assert m.dominant_mechanism({"btbt": 1e-12, "it": 3e-11, "tat": 1e-13}) == "it"


def test_crossover_log_interpolation():
    x = np.array([1e10, 1e11, 1e12])
    I_btbt = np.array([1e-10, 1e-10, 1e-10])
    I_it = np.array([1e-12, 1e-11, 1e-9])
    xc = m.crossover(np.log10(x), I_btbt, I_it)
    assert xc == pytest.approx(11.5)
    assert m.crossover(x, I_btbt, I_btbt * 1e-3) is None      # never overtakes
    assert m.crossover(x, I_it * 1e-3, I_btbt) is None        # already above at the start


def test_suppression_vs_dit_limits():
    I_ref0, a_ref, I_t0, a_t = 1e-8, 1e-20, 1e-12, 1e-21
    assert m.suppression_vs_dit(0.0, I_ref0, a_ref, I_t0, a_t) == pytest.approx(4.0)
    assert m.suppression_vs_dit(1e20, I_ref0, a_ref, I_t0, a_t) == pytest.approx(1.0, abs=1e-6)
    assert m.trap_limited_suppression(a_ref, a_t) == pytest.approx(1.0)


def test_dit_crossover_linear():
    assert m.dit_crossover(1e-10, 1e-21) == pytest.approx(1e11)
    with pytest.raises(ValueError):
        m.dit_crossover(0.0, 1e-21)


def test_dit_for_retention_matches_closed_form():
    # a_ref = 0: S(D) = log10(I_ref0/(I_t0 + a_t D)); half of S0 = 4 at I_t0 + a_t D = 1e-10
    I_ref0, I_t0, a_t = 1e-8, 1e-12, 1e-21
    D = m.dit_for_retention(I_ref0, 0.0, I_t0, a_t, frac=0.5)
    assert D == pytest.approx((1e-10 - 1e-12) / a_t, rel=1e-5)
    # trap-limited suppression above half of S0 -> never reaches half retention
    assert m.dit_for_retention(1e-8, 1e-19, 1e-12, 1e-22, frac=0.5) is None
