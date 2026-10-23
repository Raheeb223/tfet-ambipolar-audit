"""Raster digitizer recovers a known curve drawn into a synthetic image."""

import numpy as np

from src.dataio import plot_digitizer as pdg


def _image():
    img = np.full((300, 400, 3), 255, np.uint8)
    xcal, ycal = (50, 350, -6.0, 0.0), (20, 280, -6.0, -14.0)   # x linear, y log10
    V = np.linspace(-6, 0, 301)
    L = -8.0 - 0.8 * (V + 6)                                   # log10 I, slope 0.8 dec/V
    for v, l in zip(V, L):
        cx = int(round(xcal[0] + (v - xcal[2]) * (xcal[1] - xcal[0]) / (xcal[3] - xcal[2])))
        cy = int(round(ycal[0] + (l - ycal[2]) * (ycal[1] - ycal[0]) / (ycal[3] - ycal[2])))
        img[cy - 3:cy + 4, cx - 3:cx + 4] = (0, 0, 255)       # blue square markers
    return img, xcal, ycal


def test_column_trace_recovers_curve():
    img, xcal, ycal = _image()
    x, y, _ = pdg.column_trace(img, "blue", np.arange(-5.5, -0.4, 0.5), xcal, ycal, (0, 300))
    assert len(x) == 11
    assert np.max(np.abs(y - (-8.0 - 0.8 * (x + 6)))) < 0.05      # < 0.05 decade


def test_repeats_report_small_spread():
    img, xcal, ycal = _image()
    x, y, spread = pdg.repeated_trace(img, "blue", np.arange(-5.5, -0.4, 0.5), xcal, ycal, (0, 300))
    assert len(x) == 11 and spread.max() < 0.05
