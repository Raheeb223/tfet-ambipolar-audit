import pytest

from src.materials import sige, silicon


def test_si_bandgap_300K():
    assert silicon.bandgap(300.0) == pytest.approx(1.1245, abs=2e-3)
    assert silicon.bandgap(398.0) < silicon.bandgap(298.0)


def test_si_ni_300K():
    # Literature 9.65e9 (BGN-free, Altermatt) to 1.45e10 (Sze); our constants give ~1.1e10
    assert 8e9 < silicon.intrinsic_density(300.0) < 1.5e10


def test_sige_reduces_to_si():
    assert sige.bandgap_unstrained_lowT(0.0) == pytest.approx(1.155)
    assert sige.bandgap_strained_on_si_300K(0.0) == pytest.approx(1.12)
    assert sige.bandgap_strained_on_si_300K(0.3) < sige.bandgap_strained_on_si_300K(0.1)
