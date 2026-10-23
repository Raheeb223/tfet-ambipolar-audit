"""Reproducible plot digitizer for the literature figures used in calibration/validation.

Two methods, both deterministic (no hand clicking):
  * vector: read marker paths directly from a vector PDF page (PyMuPDF). Exact up to the
    marker-centre convention (bbox centre).
  * raster columns: for an embedded bitmap, mask one marker colour, remove 1-px lines with a
    morphological opening, and take the centre of the mask run in the pixel column at each
    requested x. Dense, overlapping markers form a band whose centre is the curve.
  * raster markers: connected components of the colour mask -> marker centroids (sparse markers).

Axis calibration (pixel -> data) is given explicitly from the tick positions found in the
image; `axis_map` is linear in x and linear or log10 in y. "Repeats" are the same extraction
with the column shifted by -1/0/+1 px and loose/nominal/tight colour thresholds; their spread
is reported as the digitizer error (HANDOFF §7: digitize 3x).
"""

import numpy as np
from scipy import ndimage as ndi

# colour classes: (r, g, b) int arrays -> bool mask; `tol` loosens/tightens the thresholds
def colour_mask(rgb, name, tol=0):
    r, g, b = (rgb[..., k].astype(int) for k in range(3))
    hi, lo = 170 - tol, 90 + tol
    masks = {
        "blue": (b > hi) & (r < lo) & (g < lo),
        "red": (r > hi) & (g < lo) & (b < lo),
        "green": (g > hi - 20) & (r < lo) & (b < lo),
        "cyan": (g > hi) & (b > hi) & (r < lo),
        "magenta": (r > hi) & (b > hi) & (g < lo),
        "black": (r < 60 + tol) & (g < 60 + tol) & (b < 60 + tol),
    }
    return masks[name]


def axis_map(px, p0, p1, v0, v1, log=False):
    """Linear map of pixel coordinate px given two reference pixels p0, p1 with values v0, v1.
    With log=True, v0/v1 are log10 values and the result is log10 too (caller exponentiates)."""
    return v0 + (np.asarray(px, dtype=float) - p0) * (v1 - v0) / (p1 - p0)


def _runs(col):
    idx = np.nonzero(col)[0]
    if idx.size == 0:
        return []
    br = np.nonzero(np.diff(idx) > 1)[0]
    return list(zip(np.r_[idx[0], idx[br + 1]], np.r_[idx[br], idx[-1]]))


def column_trace(rgb, colour, x_values, xcal, ycal, ylim, exclude=(), shift=0, tol=0,
                 min_run=3, pick="top"):
    """Curve y (data units, log10 if ycal is log) at each x in x_values.

    xcal = (px0, px1, v0, v1); ycal = (py0, py1, l0, l1); ylim = (row_min, row_max) searched.
    Returns arrays (x, y, band_px) for columns where the colour is present.
    """
    m = colour_mask(rgb, colour, tol)
    m = ndi.binary_opening(m, structure=np.ones((2, 2)))
    for (x0, x1, y0, y1) in exclude:
        m[y0:y1, x0:x1] = False
    px0, px1, v0, v1 = xcal
    py0, py1, l0, l1 = ycal
    xs, ys, ws = [], [], []
    for V in x_values:
        c = int(round(px0 + (V - v0) * (px1 - px0) / (v1 - v0))) + shift
        col = m[ylim[0]:ylim[1], c - 1:c + 2].any(axis=1)
        rr = [(s, e) for s, e in _runs(col) if e - s >= min_run]
        if not rr:
            continue
        s, e = {"top": rr[0], "bottom": rr[-1]}.get(pick) or max(rr, key=lambda t: t[1] - t[0])
        cy = ylim[0] + 0.5 * (s + e)
        xs.append(V)
        ys.append(axis_map(cy, py0, py1, l0, l1))
        ws.append(e - s + 1)
    return np.array(xs), np.array(ys), np.array(ws)


def marker_centroids(rgb, colour, xcal, ycal, roi, exclude=(), tol=0, area=(15, 400)):
    """Centroids (data units) of isolated markers of one colour inside roi = (x0, x1, y0, y1)."""
    m = colour_mask(rgb, colour, tol)
    keep = np.zeros(m.shape, bool)
    keep[roi[2]:roi[3], roi[0]:roi[1]] = True
    for (x0, x1, y0, y1) in exclude:
        keep[y0:y1, x0:x1] = False
    m = ndi.binary_opening(m & keep, structure=np.ones((3, 3)))
    lab, _ = ndi.label(m)
    out = []
    for k, sl in enumerate(ndi.find_objects(lab), 1):
        a = int((lab[sl] == k).sum())
        if not area[0] <= a <= area[1]:
            continue
        cy, cx = ndi.center_of_mass(lab == k)
        out.append((float(axis_map(cx, *xcal)), float(axis_map(cy, *ycal))))
    return sorted(out)


def repeated_trace(rgb, colour, x_values, xcal, ycal, ylim, **kw):
    """Three-fold extraction (shift -1/0/+1 px, tol -15/0/+15): mean and max spread per x."""
    runs = [column_trace(rgb, colour, x_values, xcal, ycal, ylim, shift=s, tol=t, **kw)
            for s, t in ((-1, -15), (0, 0), (1, 15))]
    common = sorted(set(runs[0][0]).intersection(runs[1][0], runs[2][0]))
    ys = np.array([[r[1][list(r[0]).index(x)] for x in common] for r in runs])
    return np.array(common), ys.mean(axis=0), ys.max(axis=0) - ys.min(axis=0)


def vector_markers(page, drawing_indices, xcal, ycal):
    """Marker centres from PyMuPDF drawings (bbox centre of each path, or of each closed
    3-segment triangle inside a multi-marker path). Returns list of (x, y) in data units."""
    dr = page.get_drawings()
    pts = []
    for k in drawing_indices:
        items = dr[k]["items"]
        if len(items) <= 5:
            r = dr[k]["rect"]
            centres = [(0.5 * (r.x0 + r.x1), 0.5 * (r.y0 + r.y1))]
        else:   # several open triangles drawn as one path of line segments
            centres = []
            for j in range(0, len(items) - len(items) % 3, 3):
                P = [q for it in items[j:j + 3] for q in it[1:] if hasattr(q, "x")]
                xs, ys = [q.x for q in P], [q.y for q in P]
                centres.append((0.5 * (min(xs) + max(xs)), 0.5 * (min(ys) + max(ys))))
        pts += [(float(axis_map(cx, *xcal)), float(axis_map(cy, *ycal))) for cx, cy in centres]
    return sorted(pts)
