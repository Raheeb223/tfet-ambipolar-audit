"""Field-direction nonlocal BTBT on a 2D band diagram (model-form check of the production
rows/columns model).

The production model (tfet_2d.generation) traces straight paths along mesh rows and along mesh
columns and keeps the larger of the two currents. Here every tunneling path instead runs along the
local band gradient at its valence-band turning point, and each path is counted once:

  * straight line families at angles theta_b (bin centres, n_bins over 180 degrees), lines
    spaced h apart, band edges interpolated bilinearly (as biswas_diagnostics.angled_current);
  * on each line, every (energy, path) of nonlocal_btbt is kept only if the band-gradient
    direction at its valence turning point, folded to [-90, 90) degrees, is nearest to
    theta_b (paths deviate from the field by at most 90/n_bins degrees). Every start point therefore belongs to exactly one family: no double counting
    (the risk of a row + column sum) and no lost oblique paths (the risk of a row/column max);
  * the result converges as n_bins grows; n_bins = 1 is not meaningful.

ASSUMPTIONS (logged): straight paths along the gradient at the start point (as in commercial
"dynamic path" nonlocal models); paths end at the simulated-domain boundary (the symmetry plane
of the double gate included); same calibrated (A, B, P) as the 1D junction fit, which is
path-direction independent because the calibration junctions are 1D.

Returns parameter-free terms (F_eff, weight) with I (A/um) = sum(w * A * F**(P-1) * exp(-B/F)),
so BTBT parameter variants cost a weighted sum (as tfet_2d.btbt_terms).
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from src.constants import K_B_EV, Q
from src.materials import silicon
from src.physics import nonlocal_btbt


def _fold(deg):
    """Angle folded to [-90, 90): a straight line has no sign."""
    return (np.asarray(deg) + 90.0) % 180.0 - 90.0


def _window(sol, x_win):
    """x range of the traced region; None (or a None end) means the device edge."""
    lo, hi = (None, None) if x_win is None else x_win
    return (sol.x.min() if lo is None else max(lo, sol.x.min()),
            sol.x.max() if hi is None else min(hi, sol.x.max()))


def _bin_index(phi_deg, n_bins):
    """Index of the nearest bin centre (-90 + k * 180/n_bins) for folded angles."""
    return np.rint((np.asarray(phi_deg) + 90.0) / (180.0 / n_bins)).astype(int) % n_bins


def field_direction_terms(sol, cfg, n_bins=24, x_win=None, h=2.0e-8, ds=1.0e-8, halves=2):
    """(F_eff, weight, power) of every path of the field-direction model on solution `sol`."""
    b = cfg["physics"]["btbt"]
    Eg = float(silicon.bandgap(sol.T))
    kT = K_B_EV * sol.T
    Ec, Ev = -sol.psi + 0.5 * Eg, -sol.psi - 0.5 * Eg
    grid = (sol.y, sol.x)
    fEc = RegularGridInterpolator(grid, Ec, bounds_error=False, fill_value=None)
    fEv = RegularGridInterpolator(grid, Ev, bounds_error=False, fill_value=None)
    gy, gx = np.gradient(sol.psi, sol.y, sol.x, edge_order=2)
    fgx = RegularGridInterpolator(grid, gx, bounds_error=False, fill_value=None)
    fgy = RegularGridInterpolator(grid, gy, bounds_error=False, fill_value=None)
    x0, x1 = _window(sol, x_win)
    y0, y1 = sol.y.min(), sol.y.max()
    corners = np.array([[x0, y0], [x1, y0], [x0, y1], [x1, y1]])
    width = 180.0 / n_bins
    centres = -90.0 + width * np.arange(n_bins)          # includes 0 and -90 (= 90) degrees
    Efn_v, Efp_v = -sol.Vd, -cfg["bias"]["V_source"]
    Fs, ws = [], []
    for ib, th_deg in enumerate(centres):
        th = np.radians(th_deg)
        u = np.array([np.cos(th), np.sin(th)])
        nrm = np.array([-u[1], u[0]])
        offs, proj = corners @ nrm, corners @ u
        s = np.arange(proj.min() - ds, proj.max() + ds, ds)
        for c in np.arange(offs.min() + 0.5 * h, offs.max(), h):
            P = c * nrm[None, :] + s[:, None] * u[None, :]
            ok = (P[:, 0] >= x0) & (P[:, 0] <= x1) & (P[:, 1] >= y0) & (P[:, 1] <= y1)
            if ok.sum() < 5:
                continue
            P, sl = P[ok], s[ok]
            pts = np.c_[P[:, 1], P[:, 0]]
            ec, ev = fEc(pts), fEv(pts)
            if ev.max() <= ec.min():
                continue
            E = np.arange(ec.min() + 0.5 * b["nonlocal_dE_eV"], ev.max(), b["nonlocal_dE_eV"])
            k, xv, xc, iv, ic = nonlocal_btbt._all_paths(sl, ec, ev, E)
            if k.size == 0:
                continue
            start = c * nrm[None, :] + xv[:, None] * u[None, :]           # valence turning point
            sp = np.c_[start[:, 1], start[:, 0]]
            phi = _fold(np.degrees(np.arctan2(fgy(sp), fgx(sp))))
            keep = _bin_index(phi, n_bins) == ib               # nearest family: exactly one
            if not keep.any():
                continue
            k, xv, xc, iv, ic = k[keep], xv[keep], xc[keep], iv[keep], ic[keep]
            F = Eg / np.abs(xc - xv)
            Ek = E[k]
            occ = nonlocal_btbt._fermi(Ek, Efp_v, kT) - nonlocal_btbt._fermi(Ek, Efn_v, kT)
            Fs.append(F)
            ws.append(occ * b["nonlocal_dE_eV"] * h * Q * halves * 1e-4)
    if not Fs:
        return np.array([]), np.array([]), b["P"] - 1.0
    return np.concatenate(Fs), np.concatenate(ws), b["P"] - 1.0


def current(terms, A, B):
    F, w, pw = terms
    ok = F > 0
    return float(np.sum(w[ok] * A * F[ok] ** pw * np.exp(-B / F[ok]))) if F.size else 0.0


def optimal_direction_terms(sol, cfg, n_angles=36, cell=5.0e-8, x_win=None, h=2.0e-8, ds=1.0e-8,
                            halves=2, A_sel=8.1e17, B_sel=2.43e7):
    """Angle-complete upper bound: every start point tunnels along its most probable straight path.

    Line families at n_angles directions over 180 degrees trace all straight paths. Paths are
    grouped by (start cell of size `cell`, energy); for each group the family with the largest
    current (at the selection parameters A_sel, B_sel; for P >= 1 this is the family with the
    shortest paths) is kept and the others are dropped, so each start state is counted once, in
    its best direction. The result is >= every single family and >= the rows/columns maximum.
    Returns (F_eff, weight, power) like field_direction_terms (the family choice is fixed at the
    selection parameters; other (A, B, P) reuse it).
    """
    import pandas as pd

    b = cfg["physics"]["btbt"]
    dE = b["nonlocal_dE_eV"]
    Eg = float(silicon.bandgap(sol.T))
    kT = K_B_EV * sol.T
    Ec, Ev = -sol.psi + 0.5 * Eg, -sol.psi - 0.5 * Eg
    grid = (sol.y, sol.x)
    fEc = RegularGridInterpolator(grid, Ec, bounds_error=False, fill_value=None)
    fEv = RegularGridInterpolator(grid, Ev, bounds_error=False, fill_value=None)
    x0, x1 = _window(sol, x_win)
    y0, y1 = sol.y.min(), sol.y.max()
    corners = np.array([[x0, y0], [x1, y0], [x0, y1], [x1, y1]])
    E0 = Ec.min()                                   # global energy grid: E_k = E0 + (k + 1/2) dE
    Efn_v, Efp_v = -sol.Vd, -cfg["bias"]["V_source"]
    nx = int(np.ceil((x1 - x0) / cell)) + 1
    ny = int(np.ceil((y1 - y0) / cell)) + 1
    frames = []
    for ia, th_deg in enumerate(-90.0 + (180.0 / n_angles) * np.arange(n_angles)):
        th = np.radians(th_deg)
        u = np.array([np.cos(th), np.sin(th)])
        nrm = np.array([-u[1], u[0]])
        offs, proj = corners @ nrm, corners @ u
        s = np.arange(proj.min() - ds, proj.max() + ds, ds)
        for c in np.arange(offs.min() + 0.5 * h, offs.max(), h):
            P = c * nrm[None, :] + s[:, None] * u[None, :]
            ok = (P[:, 0] >= x0) & (P[:, 0] <= x1) & (P[:, 1] >= y0) & (P[:, 1] <= y1)
            if ok.sum() < 5:
                continue
            P, sl = P[ok], s[ok]
            pts = np.c_[P[:, 1], P[:, 0]]
            ec, ev = fEc(pts), fEv(pts)
            if ev.max() <= ec.min():
                continue
            k_lo = int(np.floor((ec.min() - E0) / dE))
            k_hi = int(np.ceil((ev.max() - E0) / dE))
            E = E0 + (np.arange(max(k_lo, 0), k_hi) + 0.5) * dE
            k, xv, xc, iv, ic = nonlocal_btbt._all_paths(sl, ec, ev, E)
            if k.size == 0:
                continue
            start = c * nrm[None, :] + xv[:, None] * u[None, :]
            ix = np.clip(((start[:, 0] - x0) / cell).astype(int), 0, nx - 1)
            iy = np.clip(((start[:, 1] - y0) / cell).astype(int), 0, ny - 1)
            kg = k + max(k_lo, 0)
            F = Eg / np.abs(xc - xv)
            occ = nonlocal_btbt._fermi(E[k], Efp_v, kT) - nonlocal_btbt._fermi(E[k], Efn_v, kT)
            w = occ * dE * h * Q * halves * 1e-4
            frames.append(pd.DataFrame({"key": (kg.astype(np.int64) * ny + iy) * nx + ix, "fam": ia,
                                        "F": F, "w": w}))
    if not frames:
        return np.array([]), np.array([]), b["P"] - 1.0
    d = pd.concat(frames, ignore_index=True)
    pw = b["P"] - 1.0
    d["I"] = d.w * A_sel * d.F ** pw * np.exp(-B_sel / d.F)
    g = d.groupby(["key", "fam"], sort=False)["I"].sum().reset_index()
    best = g.loc[g.groupby("key")["I"].apply(lambda v: v.abs().idxmax())][["key", "fam"]]
    d = d.merge(best, on=["key", "fam"])
    return d.F.to_numpy(), d.w.to_numpy(), pw
