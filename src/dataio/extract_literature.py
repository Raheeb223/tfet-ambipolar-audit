"""Extract every literature curve used by the project into data/digitized/ (CSV + metadata).

The source PDFs are copyrighted and are NOT in the repository; point LIT_PDF_DIR at a folder
that holds them (default: ~/Downloads):
  zhao phd.pdf  - J. Zhao, PhD dissertation, Univ. Notre Dame, 2007 (Fig. 2.3, vector)
  wan.pdf       - J. Wan et al., Solid-State Electron. 65-66, 226 (2011) (Figs. 6-8, raster)
  TCAD_simulation_of_SOI_TFETs_and_calibra.pdf - A. Biswas et al., Microelectron. Eng. 98, 334
                  (2012) (Fig. 3, raster)
  j.sse.2012.05.053.pdf - P. G. Der Agopian et al., Solid-State Electron. 78, 141 (2012)
                  (Fig. 7, raster)

Axis calibrations below are the tick positions measured in each image (pixels), recorded
here so the extraction is reproducible. Run: python -m src.dataio.extract_literature
"""

import io
import os

import numpy as np
import pandas as pd

from src.config import load_yaml
from src.dataio import plot_digitizer as pdg
from src.dataio.digitized import DIGITIZED_DIR

PDF_DIR = os.environ.get("LIT_PDF_DIR", os.path.join(os.path.expanduser("~"), "Downloads"))
FLOOR_LOG = -12.7     # Wan 2011 measurement floor (~1e-13 A/um noise); points below are dropped

WAN = dict(source="J. Wan, C. Le Royer, A. Zaslavsky, S. Cristoloveanu, Solid-State Electron. 65-66, 226 (2011)",
           doi_or_url="https://doi.org/10.1016/j.sse.2011.06.012")
ZHAO = dict(source="J. Zhao, PhD dissertation, Univ. of Notre Dame (2007), Fig. 2.3, compiling "
                   "R. B. Fair & H. W. Wivell, IEEE TED 23, 512 (1976) and J. M. Stork & R. D. Isaac, IEEE TED 30, 1527 (1983)",
            doi_or_url="none (dissertation, Univ. of Notre Dame library, April 2007)")
BISWAS = dict(source="A. Biswas, S. S. Dan, C. Le Royer, W. Grabinski, A. M. Ionescu, Microelectron. Eng. 98, 334 (2012)",
              doi_or_url="https://doi.org/10.1016/j.mee.2012.07.077")
AGOPIAN = dict(source="P. G. Der Agopian et al., Solid-State Electron. 78, 141 (2012)",
               doi_or_url="https://doi.org/10.1016/j.sse.2012.05.053")


def _page_image(doc, page, k):
    import pymupdf
    from PIL import Image
    pix = pymupdf.Pixmap(doc, doc[page].get_images(full=True)[k][0])
    if pix.n - pix.alpha >= 4:
        pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
    return np.asarray(Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB"))


def zhao_fig23():
    import pymupdf
    page = pymupdf.open(os.path.join(PDF_DIR, "zhao phd.pdf"))[30]
    # plot frame (pt): x 171.96 -> 0 MV/cm, 468.90 -> 3 MV/cm; y 87.66 -> 1e3, 351.36 -> 1e-7 A/cm2
    xcal, ycal = (171.96, 468.90, 0.0, 3.0), (87.66, 351.36, 3.0, -7.0)
    fair = pdg.vector_markers(page, range(25, 30), xcal, ycal)      # filled triangles
    stork = pdg.vector_markers(page, [24], xcal, ycal)              # open triangles
    return {"fair": fair, "stork": stork}


def wan_figs():
    import pymupdf
    doc = pymupdf.open(os.path.join(PDF_DIR, "wan.pdf"))
    f6 = _page_image(doc, 3, 1)     # Fig. 6 (a) simulation, (b) measured; NTFET, SiO2, VDS = 1 V
    f8 = _page_image(doc, 3, 0)     # Fig. 8 (a) NTFET HfO2, (b) PTFET HfO2
    f7 = _page_image(doc, 3, 2)     # Fig. 7 (b) simulated lateral dopant profiles
    out = {}
    Vg = np.round(np.arange(-6.0, -0.59, 0.2), 3)
    # Fig. 6b: 1E-7 at row 18, 1E-14 at row 270; -6 V at col 578, +6 V at col 923
    cal6b = dict(xcal=(578, 923, -6.0, 6.0), ycal=(18, 270, -7.0, -14.0), ylim=(10, 270),
                 exclude=[(680, 800, 25, 140), (588, 600, 30, 262)])
    for lin, col in ((10, "blue"), (25, "red"), (40, "green"), (50, "cyan")):
        out[f"wan2011_fig6b_LIN{lin}"] = pdg.repeated_trace(f6, col, Vg, **cal6b)
    # Fig. 8a: 1E-7 at row 14, 1E-14 at row 263; -3 V at col 98, +3 V at col 438
    Vg8 = np.round(np.arange(-3.0, -0.29, 0.1), 3)
    cal8a = dict(xcal=(98, 438, -3.0, 3.0), ycal=(14, 263, -7.0, -14.0), ylim=(5, 263),
                 exclude=[(200, 400, 0, 110), (110, 130, 10, 245)])
    for lin, col in ((10, "blue"), (25, "red"), (50, "cyan")):
        out[f"wan2011_fig8a_LIN{lin}"] = pdg.repeated_trace(f8, col, Vg8, **cal8a)
    # Fig. 6a (published Silvaco BTBT-only simulation): 1E-7 at row 19, 1E-17 at row 269;
    # -6 V at col 95, +6 V at col 438; sparse markers every 0.5 V
    xcal6a, ycal6a = (95, 438, -6.0, 6.0), (19, 269, -7.0, -17.0)
    for lin, col in ((10, "blue"), (20, "red"), (30, "magenta"), (40, "green")):
        pts = pdg.marker_centroids(f6, col, xcal6a, ycal6a, roi=(96, 300, 10, 268),
                                   exclude=[(220, 300, 30, 160), (125, 150, 30, 245)])
        pts = [(x, y) for x, y in pts if x < -0.5 and y > -16.0]   # drop the numerical floor
        out[f"wan2011_fig6a_sim_LIN{lin}"] = (np.array([p[0] for p in pts]),
                                              np.array([p[1] for p in pts]), np.zeros(len(pts)))
    # Fig. 7b As profile: 0 nm at col 629, 100 nm at col 922; 1E15 at row 361, 1E21 at row 125
    xcal7, ycal7 = (629, 922, 0.0, 100.0), (361, 125, 15.0, 21.0)
    near = pdg.repeated_trace(f7, "red", np.arange(2.0, 30.1, 2.0), xcal7, ycal7, (120, 362),
                              exclude=[(810, 960, 290, 362)], pick="longest")
    tail = [(x, y) for x, y in pdg.marker_centroids(f7, "red", xcal7, ycal7, roi=(630, 960, 120, 362),
                                                   exclude=[(810, 960, 290, 362)]) if x > 31]
    xs = np.r_[near[0], [t[0] for t in tail]]
    ys = np.r_[near[1], [t[1] for t in tail]]
    out["wan2011_fig7b_As_profile"] = (xs, ys, np.r_[near[2], np.zeros(len(tail))])
    return out


def _repeated_markers(rgb, colour, xcal, ycal, roi, exclude, area, dx=0.03):
    """Marker centroids extracted 3x (colour tol -15/0/+15); keep markers found in every repeat
    (matched within dx in x) and return mean y and the max spread."""
    reps = [pdg.marker_centroids(rgb, colour, xcal, ycal, roi=roi, exclude=exclude, tol=t, area=area)
            for t in (-15, 0, 15)]
    xs, ys, sp = [], [], []
    for x, y in reps[1]:
        m = [min(r, key=lambda p: abs(p[0] - x) + abs(p[1] - y)) for r in (reps[0], reps[2])]
        if all(abs(p[0] - x) < dx and abs(p[1] - y) < 0.1 for p in m):
            yy = [y] + [p[1] for p in m]
            xs.append(x); ys.append(float(np.mean(yy))); sp.append(max(yy) - min(yy))
    return np.array(xs), np.array(ys), np.array(sp)


def biswas_fig3():
    """Biswas et al. 2012, Fig. 3: measured transfer curves (symbols) of the CEA-LETI SOI n-TFET,
    LG = 400 nm, LIN = 100 nm, VD = 0.5 / 1.0 / 1.5 V. Lines of the same colour are their TCAD
    fits; the 3x3 opening in marker_centroids removes them."""
    import pymupdf
    doc = pymupdf.open(os.path.join(PDF_DIR, "TCAD_simulation_of_SOI_TFETs_and_calibra.pdf"))
    img = _page_image(doc, 2, 1)
    # ticks: 0.0 V at col 103, 3.0 V at col 615; 1e-9 at row 25, 1e-13 at row 346
    xcal, ycal = (103, 615, 0.0, 3.0), (25, 346, -9.0, -13.0)
    kw = dict(roi=(104, 614, 2, 377), exclude=[(110, 330, 2, 110)], area=(20, 200))
    return {f"biswas2012_fig3_VD{v}": _repeated_markers(img, c, xcal, ycal, **kw)
            for v, c in (("0p5", "blue"), ("1p0", "red"), ("1p5", "green"))}


def agopian_fig7():
    """Der Agopian et al. 2012, Fig. 7: measured pTFET (imec, triple gate, HfO2) at 300, 360,
    390 and 420 K, |VDS| = 0.9 V. All markers are black; where the four temperatures are
    resolved (exactly four isolated markers in one VGS column, i.e. the trap-dominated valley)
    they are assigned by order, since the current rises monotonically with T there."""
    import pymupdf
    doc = pymupdf.open(os.path.join(PDF_DIR, "j.sse.2012.05.053.pdf"))
    img = _page_image(doc, 3, 3)
    # ticks: -2.0 V at col 104, 0.5 V at col 596; 1e-7 at row 9, 1e-15 at row 372
    xcal, ycal = (104, 596, -2.0, 0.5), (9, 372, -7.0, -15.0)
    kw = dict(roi=(106, 595, 11, 371), exclude=[(110, 260, 220, 371), (330, 520, 15, 150)], area=(15, 200))
    temps = (300, 360, 390, 420)
    per_rep = []
    for tol in (-15, 0, 15):
        pts = pdg.marker_centroids(img, "black", xcal, ycal, tol=tol, **kw)
        cols = {}
        for x, y in pts:
            cols.setdefault(round(x / 0.08) * 0.08, []).append(y)
        per_rep.append({round(v, 2): sorted(ys) for v, ys in cols.items() if len(ys) == 4})
    common = sorted(set(per_rep[0]).intersection(per_rep[1], per_rep[2]))
    out = {}
    for k, T in enumerate(temps):
        y = np.array([[r[v][k] for v in common] for r in per_rep])
        out[f"agopian2012_fig7_T{T}"] = (np.array(common), y.mean(axis=0), y.max(axis=0) - y.min(axis=0))
    return out


def _entry(file, figure, label, xq, xu, yq, yu, scale, T, bias, doping, reps, err, role, base):
    return dict(file=file, **base, figure=figure, curve_label=label, x_quantity=xq, x_unit=xu,
                y_quantity=yq, y_unit=yu, y_scale=scale, temperature_K=T, bias_direction=bias,
                doping_info=doping, area_um2=None, digitizer_repeats=reps, est_error=err, role=role)


def main():
    os.makedirs(DIGITIZED_DIR, exist_ok=True)
    entries = []
    z = zhao_fig23()
    for key, label in (("fair", "Fair & Wivell 1976, As-implanted n+p"),
                       ("stork", "Stork & Isaac 1983, base-emitter n+p")):
        f = f"zhao2007_fig2p3_{key}.csv"
        pd.DataFrame({"F_MVcm": [p[0] for p in z[key]], "J_Acm2": [10 ** p[1] for p in z[key]]}) \
          .to_csv(os.path.join(DIGITIZED_DIR, f), index=False, float_format="%.5g")
        entries.append(_entry(f, "Fig. 2.3", label + ", reverse bias -1 V", "electric_field", "MV/cm",
                              "current_density", "A/cm2", "log", 300, "reverse",
                              "one-sided n+p junctions; F = peak junction field from SIMS / C-V (original papers)",
                              1, "vector PDF: exact path coordinates; marker-centre convention +-0.04 decade",
                              "core_calibration", ZHAO))
    w = wan_figs()
    meta = {
        "fig6b": ("Fig. 6(b)", "measured NTFET, 6 nm SiO2, LG = 400 nm, VDS = 1 V", "validation", 300),
        "fig8a": ("Fig. 8(a)", "measured NTFET, 3 nm HfO2 (EOT 2.2 nm), LG = 400 nm, VDS = 1 V", "validation", 300),
        "fig6a": ("Fig. 6(a)", "published Silvaco BTBT-only simulation, NTFET SiO2, VDS = 1 V", "context", 300),
    }
    for name, (x, y, spread) in w.items():
        f = name + ".csv"
        if "As_profile" in name:
            pd.DataFrame({"x_nm": x, "N_cm3": 10 ** y}).to_csv(os.path.join(DIGITIZED_DIR, f),
                                                                 index=False, float_format="%.5g")
            entries.append(_entry(f, "Fig. 7(b)", "simulated As lateral profile from the spacer edge (process simulation)",
                                  "position", "nm", "doping", "cm-3", "log", 300, "n/a",
                                  "As drain HDD 2e15 cm-2 at 20 keV, RTA (Sec. 2.1)", 3,
                                  f"raster; repeat spread <= {spread.max():.2f} decade", "structure", WAN))
            continue
        key = name.split("_")[1]
        fig, desc, role, T = meta[key]
        lin = name.split("LIN")[1]
        keep = y > FLOOR_LOG if role == "validation" else np.ones_like(y, bool)
        if keep.sum() < 3:          # whole curve at the measurement floor: keep it as an upper bound
            keep = np.ones_like(y, bool)
            desc += ", AT MEASUREMENT FLOOR (upper bound only)"
        pd.DataFrame({"VGS_V": x[keep], "ID_Aum": 10 ** y[keep]}).to_csv(
            os.path.join(DIGITIZED_DIR, f), index=False, float_format="%.5g")
        err = (f"raster; 3 repeats (column +-1 px, colour tol +-15): max spread {spread[keep].max():.2f} decade; "
               "x +-0.03 V" if role == "validation" else "raster marker centroids; +-0.05 decade")
        entries.append(_entry(f, fig, f"{desc}, L_IN = {lin} nm (ambipolar branch, VGS < 0)",
                              "gate_voltage", "V", "drain_current", "A/um", "log", T, "n/a",
                              "p+ source B, n+ drain As, intrinsic L_IN next to the drain (Sec. 2)",
                              3 if role == "validation" else 1, err, role, WAN))
    for name, (x, y, spread) in biswas_fig3().items():
        f = name + ".csv"
        vd = name.split("VD")[1].replace("p", ".")
        pd.DataFrame({"VGS_V": x, "ID_Aum": 10 ** y}).to_csv(os.path.join(DIGITIZED_DIR, f),
                                                               index=False, float_format="%.5g")
        entries.append(_entry(f, "Fig. 3", f"measured SOI n-TFET (CEA-LETI), 6 nm SiO2, LG = 400 nm, LIN = 100 nm, "
                              f"VD = {vd} V (on-state branch, VGS > 0)", "gate_voltage", "V", "drain_current",
                              "A/um", "log", 300, "n/a",
                              "p+ source B 8e19, n+ drain As 5e19, undoped 20 nm film, ~7 nm/dec junctions (Sec. 2)",
                              3, f"raster marker centroids, 3 repeats (colour tol +-15): max spread "
                              f"{spread.max():.2f} decade; x +-0.02 V", "validation", BISWAS))
    for name, (x, y, spread) in agopian_fig7().items():
        f = name + ".csv"
        T = int(name.split("_T")[1])
        pd.DataFrame({"VGS_V": x, "ID_A": 10 ** y}).to_csv(os.path.join(DIGITIZED_DIR, f),
                                                             index=False, float_format="%.5g")
        entries.append(_entry(f, "Fig. 7", f"measured triple-gate Si pTFET (imec), HfO2/TiN, Lch = 150 nm, 5 fins, "
                              f"|VDS| = 0.9 V, T = {T} K (valley only: the four temperatures are resolved there)",
                              "gate_voltage", "V", "drain_current", "A", "log", T, "n/a",
                              "undoped fin (1e15), As/B extensions, self-aligned gate (Sec. 2)", 3,
                              f"raster marker centroids, 3 repeats (colour tol +-15): max spread {spread.max():.2f} "
                              "decade; temperature assigned by marker order within each VGS column",
                              "validation", AGOPIAN))
    header = open(os.path.join(DIGITIZED_DIR, "metadata.yaml"), encoding="utf-8").read().split("datasets:")[0]
    import yaml
    with open(os.path.join(DIGITIZED_DIR, "metadata.yaml"), "w", encoding="utf-8") as fh:
        fh.write(header)
        yaml.safe_dump({"datasets": entries}, fh, sort_keys=False, allow_unicode=True, width=120)
    print(f"wrote {len(entries)} datasets to {DIGITIZED_DIR}")


if __name__ == "__main__":
    main()
