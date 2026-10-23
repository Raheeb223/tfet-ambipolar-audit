"""Diagnosis of the on-state shortfall of the calibrated nonlocal model on Biswas 2012 Fig. 3.

DIAGNOSTIC ONLY: nothing here is fed back into a parameter. Each test isolates one candidate
cause named in the paper (Sec. IV-B) and reports the change of the on-state current:

  window   -- recompute with the drain quasi-Fermi level forced to -1.5 V on the VD = 0.5 V
              solution: separates the tunneling energy window (occupancy) from electrostatics.
  paths    -- add straight paths at +-30/45/60 deg (band edges interpolated on the 2D mesh) to
              the row (0 deg) and column (90 deg) paths of the production model.
  where    -- current-weighted position and path length of the row paths.
  profile  -- re-solve with source straggle 1.5 and 5 nm (production: 3 nm).

Run: python -m src.validation.biswas_diagnostics [solve|post|profile]
     -> results/validation/biswas_diag/*.npz, diagnostics.csv
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

from src.devices import tfet_2d as t
from src.materials import silicon
from src.physics import nonlocal_btbt
from src.validation import biswas2012 as bw

OUT = os.path.join("results", "validation", "biswas_diag")
POINTS = [(3.0, 0.5), (3.0, 1.0), (3.0, 1.5), (2.0, 1.0)]
K_B_EV = 8.617333262e-5


def _cfg(sigma_cm=None):
    base = bw.calibrated(t.load_config(bw.CONFIG))
    base["physics"]["interface_traps"]["D_it_cm2eV"] = bw.D_REF
    cfg = bw.device_cfg(base, 10.0)
    if sigma_cm is not None:
        cfg = t.variant(cfg, **{"doping.junction_sigma_cm": sigma_cm})
    return cfg


def solve(points=POINTS, sigma_cm=None, tag="base"):
    os.makedirs(OUT, exist_ok=True)
    cfg = _cfg(sigma_cm)
    dev = t.build(cfg)
    try:
        for Vg, Vd in points:
            t.ramp_to(dev, Vg, Vd)
            s = t.extract(dev)
            np.savez(os.path.join(OUT, f"{tag}_Vg{Vg:.1f}_Vd{Vd:.1f}.npz"), x=s.x, y=s.y, psi=s.psi,
                     n=s.n, p=s.p, vol=s.vol, Vg=s.Vg, Vd=s.Vd, T=s.T)
            r, _ = t.generation(s, cfg)
            print(tag, Vg, Vd, f"nl={r['I_btbt_nonlocal']:.3e} loc={r['I_btbt_local']:.3e}", flush=True)
    finally:
        dev.delete()


def load(tag, Vg, Vd):
    z = np.load(os.path.join(OUT, f"{tag}_Vg{Vg:.1f}_Vd{Vd:.1f}.npz"))
    return t.Solution(x=z["x"], y=z["y"], psi=z["psi"], n=z["n"], p=z["p"], vol=z["vol"],
                      Vg=float(z["Vg"]), Vd=float(z["Vd"]), T=float(z["T"]))


def _bands(sol):
    Eg = float(silicon.bandgap(sol.T))
    return -sol.psi + 0.5 * Eg, -sol.psi - 0.5 * Eg, Eg


def angled_current(sol, cfg, theta_deg, x_win=(-3.0e-6, 6.0e-6), h=2.0e-8, ds=1.0e-8, Efn=None):
    """Nonlocal current (A/um) from straight paths at angle theta (0 = along x) through the Si
    film, band edges interpolated bilinearly; parallel lines spaced h apart (perpendicular)."""
    b = cfg["physics"]["btbt"]
    Ec, Ev, Eg = _bands(sol)
    kT = K_B_EV * sol.T
    A, B = b["nonlocal_A_cm3s"], b["nonlocal_B_Vcm"]
    fEc = RegularGridInterpolator((sol.y, sol.x), Ec, bounds_error=False, fill_value=None)
    fEv = RegularGridInterpolator((sol.y, sol.x), Ev, bounds_error=False, fill_value=None)
    th = np.radians(theta_deg)
    u = np.array([np.cos(th), np.sin(th)])            # along path (x, y)
    nrm = np.array([-u[1], u[0]])                      # line normal
    y0, y1 = sol.y.min(), sol.y.max()
    xs0, xs1 = max(x_win[0], sol.x.min()), min(x_win[1], sol.x.max())
    corners = np.array([[xs0, y0], [xs1, y0], [xs0, y1], [xs1, y1]])
    offs = corners @ nrm
    Efn_v = -sol.Vd if Efn is None else Efn
    Efp_v = -cfg["bias"]["V_source"]
    total = 0.0
    for c in np.arange(offs.min(), offs.max(), h):
        # points on the line inside the window: p = c*nrm + s*u
        proj = corners @ u
        s = np.arange(proj.min() - 1e-6, proj.max() + 1e-6, ds)
        P = c * nrm[None, :] + s[:, None] * u[None, :]
        ok = (P[:, 0] >= xs0) & (P[:, 0] <= xs1) & (P[:, 1] >= y0) & (P[:, 1] <= y1)
        if ok.sum() < 5:
            continue
        P, sl = P[ok], s[ok]
        pts = np.c_[P[:, 1], P[:, 0]]
        J = nonlocal_btbt.line_current(sl, fEc(pts), fEv(pts), A, B, Eg, b["P"],
                                       Efn=np.full(sl.shape, Efn_v), Efp=np.full(sl.shape, Efp_v),
                                       kT=kT, dE=b["nonlocal_dE_eV"])
        total += J * h
    return t._per_um_base(total, 1 if t.single_gate(cfg) else 2)


def angle_scan(sol, cfg, angles, x_win):
    return {a: angled_current(sol, cfg, a, x_win=x_win) for a in angles}


def audit_scan(angles=(0, -10, -20, -30, -40, -50, 10, 20, 30, 40, 50), Vgs=(-1.0, -0.2)):
    """Same angle scan on the cached audit solutions (drain-side ambipolar junction, 300 K)."""
    from src.audit import run_audit as ra
    from src.config import load_yaml
    audit = load_yaml("configs/audit_calibrated.yaml")
    ra.CACHE = audit["cache_dir"]
    cal = json.load(open(audit["calibration"]))
    rows = []
    for d in audit["devices"]:
        cfg, sols = ra.load_solutions(d, 300.0)
        c = ra.scenario_config(cfg, "nonlocal", None, None, cal)
        Lg = c["geometry"]["L_gate_cm"]
        for Vg in Vgs:
            s = [x for x in sols if np.isclose(x.Vg, Vg)][0]
            r, _ = t.generation(s, c)
            sc = angle_scan(s, c, angles, (Lg - 3e-6, Lg + 3e-6))
            best = max(sc, key=sc.get)
            rows.append(dict(device=d, Vg=Vg, I_x=r["I_btbt_nonlocal_x"], I_y=r.get("I_btbt_nonlocal_y", 0.0),
                             I_ang_max=sc[best], best_angle=best,
                             **{f"I_ang{a}": v for a, v in sc.items()}))
            print(d, Vg, f"x={r['I_btbt_nonlocal_x']:.2e} y={r.get('I_btbt_nonlocal_y', 0):.2e} "
                  f"best={best} {sc[best]:.2e}", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT, "audit_angles.csv"), index=False)
    return out


def where(sol, cfg):
    """Current-weighted x, y of the row-path current and the typical path length."""
    b = cfg["physics"]["btbt"]
    Ec, Ev, Eg = _bands(sol)
    kT = K_B_EV * sol.T
    Js, ls = [], []
    for j in range(len(sol.y)):
        F, w = nonlocal_btbt.path_terms(sol.x, Ec[j], Ev[j], Eg, np.full(sol.x.shape, -sol.Vd),
                                        np.full(sol.x.shape, 0.0), kT, b["nonlocal_dE_eV"])
        I = w * b["nonlocal_A_cm3s"] * np.where(F > 0, F, 1.0) ** (b["P"] - 1) * np.exp(-b["nonlocal_B_Vcm"] / np.where(F > 0, F, 1e-30))
        Js.append(I.sum())
        ls.append(np.sum(I * Eg / np.where(F > 0, F, np.inf)) / max(I.sum(), 1e-300))
    Js = np.array(Js)
    wy = Js / Js.sum()
    return dict(y_mean_nm=float(np.sum(wy * sol.y) * 1e7),
                frac_top_2nm=float(Js[sol.y <= sol.y.min() + 2e-7].sum() / Js.sum()),
                path_nm=float(np.sum(wy * np.array(ls)) * 1e7))


def post():
    cfg = _cfg()
    meas = {v: bw_meas for v, bw_meas in ((0.5, "0p5"), (1.0, "1p0"), (1.5, "1p5"))}
    rows = []
    for Vg, Vd in POINTS:
        s = load("base", Vg, Vd)
        r, _ = t.generation(s, cfg)
        row = dict(Vg=Vg, Vd=Vd, I_rows=r["I_btbt_nonlocal_x"], I_cols=r["I_btbt_nonlocal_y"],
                   I_local=r["I_btbt_local"])
        if Vd in meas:
            V, I = pd.read_csv(os.path.join("data", "digitized", f"biswas2012_fig3_VD{meas[Vd]}.csv")).to_numpy().T
            row["I_meas"] = float(10 ** np.interp(Vg, V, np.log10(I)))
        for th in (0, 30, 45, 60, -30, -45, -60):
            row[f"I_ang{th}"] = angled_current(s, cfg, th)
        row["I_ang_max"] = max(row[f"I_ang{th}"] for th in (0, 30, 45, 60, -30, -45, -60))
        row["I_rows_window1p5"] = angled_current(s, cfg, 0, Efn=-1.5)
        row.update(where(s, cfg))
        rows.append(row)
        print({k: (f"{v:.3e}" if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    prof = []
    for tag in ("sig1p5", "sig5"):
        p = os.path.join(OUT, f"{tag}_Vg3.0_Vd1.0.npz")
        if os.path.exists(p):
            c = _cfg({"sig1p5": 1.5e-7, "sig5": 5e-7}[tag])
            r, _ = t.generation(load(tag, 3.0, 1.0), c)
            prof.append(dict(tag=tag, I_nl=r["I_btbt_nonlocal"], I_loc=r["I_btbt_local"]))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT, "diagnostics.csv"), index=False)
    if prof:
        pd.DataFrame(prof).to_csv(os.path.join(OUT, "profile.csv"), index=False)
        print(pd.DataFrame(prof))
    return out


def wan_scan(angles=(0, -10, -20, -30, -40, -50, 10, 20, 30, 40, 50), offset_nm=10.0):
    """Does the angle-complete path set change the Wan 2011 ambipolar validation? Re-solves
    the strongly leaking L_IN = 10 / 25 nm devices (offset 10 nm) and compares rows-only and
    angle-max nonlocal BTBT (+ bulk and interface TAT at the nominal D_it) with the data."""
    from src.validation import wan2011 as wv
    from src.analysis.validation_figs import D_NOM, measured, FLOOR
    rows = []
    for dev_name, Vgs in (("sio2", (-1.5, -2.0, -3.0, -4.0, -5.0, -6.0)),
                          ("hfo2", (-0.5, -1.0, -1.5, -2.0, -2.5, -3.0))):
        base = wv.calibrated(t.load_config(wv.DEVICES[dev_name]["config"]))
        for L in (10, 25):
            cfg = wv.device_cfg(base, L, offset_nm)
            Lg = wv.LG
            dev = t.build(cfg)
            try:
                for Vg in Vgs:
                    t.ramp_to(dev, Vg, cfg["bias"]["V_drain"])
                    s = t.extract(dev)
                    r, _ = t.generation(s, cfg)
                    sc = angle_scan(s, cfg, angles, (Lg - 3e-6, Lg + L * 1e-7 + offset_nm * 1e-7 + 3e-6))
                    trap = r["I_tat_bulk"] + D_NOM[dev_name] * r["I_it"] / wv.D_REF
                    rows.append(dict(device=dev_name, L_IN_nm=L, Vg=Vg, I_rows=r["I_btbt_nonlocal"],
                                     I_ang0=sc[0], I_ang_max=max(sc.values()),
                                     best_angle=max(sc, key=sc.get), I_trap=trap))
                    print(rows[-1], flush=True)
            finally:
                dev.delete()
    out = pd.DataFrame(rows)
    res = []
    for (dn, L), g in out.groupby(["device", "L_IN_nm"]):
        Vm, Im = measured(dn, L)
        for col in ("I_rows", "I_ang_max"):
            I = g[col] + g.I_trap
            Ii = 10 ** np.interp(Vm, g.Vg[::-1], np.log10(I.to_numpy()[::-1]))
            ok = (Im > FLOOR) & (Vm >= g.Vg.min()) & (Vm <= g.Vg.max())
            rr = np.log10(Ii[ok] / Im[ok])
            res.append(dict(device=dn, L_IN_nm=L, paths=col, n=int(ok.sum()),
                            rms=float(np.sqrt(np.mean(rr ** 2))), bias=float(rr.mean())))
    out.to_csv(os.path.join(OUT, "wan_angles.csv"), index=False)
    pd.DataFrame(res).to_csv(os.path.join(OUT, "wan_angles_residuals.csv"), index=False)
    print(pd.DataFrame(res).round(2).to_string(index=False))
    return out


def biswas_scan(angles=tuple(range(-50, 11, 5)), Vgs=tuple(np.round(np.arange(1.0, 3.01, 0.25), 2))):
    """Full on-state metric of biswas2012 (VGS >= 1 V, same definition) with angle-max paths."""
    cfg = _cfg()
    rows = []
    dev = t.build(cfg)
    try:
        for Vd in bw.VD:
            t.ramp_to(dev, Vgs[0], Vd)
            for Vg in Vgs:
                t.ramp_to(dev, Vg, Vd)
                s = t.extract(dev)
                r, _ = t.generation(s, cfg)
                sc = angle_scan(s, cfg, angles, (-3e-6, 6e-6))
                rows.append(dict(Vd=Vd, Vg=Vg, I_rows=r["I_btbt_nonlocal"], I_ang_max=max(sc.values()),
                                 best_angle=max(sc, key=sc.get),
                                 I_trap=r["I_tat_bulk"] + r["I_it"]))
                print(rows[-1], flush=True)
    finally:
        dev.delete()
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT, "biswas_angles.csv"), index=False)
    res = []
    for key, Vd in (("0p5", 0.5), ("1p0", 1.0), ("1p5", 1.5)):
        V, I = pd.read_csv(os.path.join("data", "digitized", f"biswas2012_fig3_VD{key}.csv")).to_numpy().T
        g = out[np.isclose(out.Vd, Vd)].sort_values("Vg")
        ok = V >= 1.0
        for col in ("I_rows", "I_ang_max"):
            Is = 10 ** np.interp(V[ok], g.Vg, np.log10((g[col] + g.I_trap).to_numpy()))
            rr = np.log10(Is / I[ok])
            res.append(dict(Vd=Vd, paths=col, bias=float(rr.mean()), rms=float(np.sqrt(np.mean(rr ** 2))),
                            bias_at_3V=float(rr[-1])))
    pd.DataFrame(res).to_csv(os.path.join(OUT, "biswas_angles_residuals.csv"), index=False)
    print(pd.DataFrame(res).round(2).to_string(index=False))
    return out


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "post"
    if mode == "solve":
        solve()
    elif mode == "audit":
        audit_scan()
    elif mode == "wan":
        wan_scan()
    elif mode == "biswas":
        biswas_scan()
    elif mode == "fine":
        cfg = _cfg()
        s = load("base", 3.0, 1.0)
        for a, v in angle_scan(s, cfg, range(-60, 1, 5), (-3e-6, 6e-6)).items():
            print(f"angle {a:4d}  I = {v:.3e}", flush=True)
    elif mode == "profile":
        solve(points=[(3.0, 1.0)], sigma_cm=1.5e-7, tag="sig1p5")
        solve(points=[(3.0, 1.0)], sigma_cm=5e-7, tag="sig5")
    else:
        post()
