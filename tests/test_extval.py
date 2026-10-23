"""Independent measured checks: literature-value provenance, Arrhenius extraction, the
temperature assignment of the digitized Der Agopian valley, and the marker digitizer used
for the Biswas / Der Agopian figures."""

import os

import numpy as np
import pandas as pd
import pytest

from src.analysis import extval_figs as ev
from src.dataio import plot_digitizer as pdg
from src.dataio.digitized import DIGITIZED_DIR, MetadataError, load_metadata
from src.dataio.literature_values import load_values


def test_arrhenius_recovers_activation_energy():
    T = np.array([300.0, 360.0, 390.0, 420.0])
    I = 1e-3 * np.exp(-0.45 / (ev.K_B * T))
    assert ev.arrhenius(T, I) == pytest.approx(0.45, abs=1e-9)
    assert ev.arrhenius(T, np.full(4, 2.0)) == pytest.approx(0.0, abs=1e-12)


def test_literature_values_have_provenance():
    vals = load_values()
    assert vals["luong_Ea_offstate"]["value"] == 0.45
    for e in vals.values():
        assert e["quote"] and e["where"] and e["source"]
        assert e["role"] in ("validation", "context")


def test_literature_values_reject_missing_fields(tmp_path):
    p = tmp_path / "v.yaml"
    p.write_text("values:\n- key: x\n  value: 1.0\n  unit: eV\n  role: validation\n")
    with pytest.raises(MetadataError):
        load_values(str(p))


def test_new_datasets_are_validation_only():
    meta = {e["file"]: e for e in load_metadata()}
    for f in ("biswas2012_fig3_VD1p0.csv", "agopian2012_fig7_T300.csv"):
        assert meta[f]["role"] == "validation"
        assert meta[f]["digitizer_repeats"] == 3


def test_agopian_valley_current_rises_with_temperature():
    """Temperatures were assigned by marker order; check the result is consistent: at every
    VGS the current increases with T, and the valley minimum activates by > 1.5 decades."""
    cur = [pd.read_csv(os.path.join(DIGITIZED_DIR, f"agopian2012_fig7_T{T}.csv")) for T in ev.AGO_T]
    V = cur[0].VGS_V.to_numpy()
    for c in cur[1:]:
        assert np.allclose(c.VGS_V.to_numpy(), V)
    I = np.array([c.ID_A.to_numpy() for c in cur])
    assert (np.diff(I, axis=0) > 0).all()
    assert np.log10(I[-1].min() / I[0].min()) > 1.5


def test_marker_centroids_ignore_thin_lines():
    """Biswas Fig. 3 draws same-colour fit lines through the symbols; the opening must keep
    the markers and drop the 1-px line."""
    img = np.full((200, 300, 3), 255, np.uint8)
    img[100, 10:290] = (255, 0, 0)                      # thin red line
    for cx in (50, 150, 250):
        img[96:105, cx - 4:cx + 5] = (255, 0, 0)        # 9x9 red markers on the line
    xcal, ycal = (0, 300, 0.0, 3.0), (0, 200, 0.0, -4.0)
    pts = pdg.marker_centroids(img, "red", xcal, ycal, roi=(0, 300, 0, 200), area=(20, 200))
    assert len(pts) == 3
    assert np.allclose([p[0] for p in pts], [0.5, 1.5, 2.5], atol=0.02)
