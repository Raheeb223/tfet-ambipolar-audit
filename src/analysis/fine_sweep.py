"""Fine-bias recomputation of the current components of Fig. (components).

The audit sweep uses 0.2 V steps. Near V_GS = 0 the nonlocal BTBT drops by tens of decades
between neighbouring points, and above V_GS ~ +0.3 V the source-side (on-state) branch
switches on. This script re-solves the electrostatics of the reference and LDD devices at
`step` (default 0.02 V), with the same mesh, config and solver tolerances as the audit,
and records, at every bias:

  * the nonlocal BTBT of row (x) and column (y) paths separately, and which one is used;
  * the nonlocal BTBT split into source-side (x < L_g/2) and drain-side paths;
  * the band-overlap energy window that bounds the nonlocal paths on the drain side and on
    the source side (zero window -> no path -> exactly zero current);
  * interface and bulk TAT, local BTBT, all from the same converged solution.

It also checks that the fine sweep reproduces the audit table at the shared 0.2 V points.

Run:  python -m src.analysis.fine_sweep [--step 0.02]
Out:  results/audit_cal/fine_sweep.csv, paper/figs/fig_components.{pdf,png} via figures.py
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

from src.audit import run_audit as ra
from src.config import load_yaml
from src.devices import tfet_2d as t
from src.materials import silicon

AUDIT = "configs/audit_calibrated.yaml"
OUT = os.path.join("results", "audit_cal", "fine_sweep.csv")
DEVICES = ["reference", "low_drain_doping"]
TEMPS = [300.0, 398.15]
D_IT, TAU = 1e12, 1e-7


def _window(sol, Eg, side_mask):
    """Largest band-overlap energy (eV) available to paths that end on the given side.

    Paths need Ev(somewhere) > Ec(somewhere else); the overlap max(Ev) - min(Ec) over the
    rows, restricted to the region where the conduction-band end lies, bounds the window.
    """
    Ec = -sol.psi + 0.5 * Eg
    Ev = -sol.psi - 0.5 * Eg
    return float(np.max(Ev) - np.min(Ec[:, side_mask]))


def _split_nonlocal(sol, cfg, A, B, Eg, kT, xmid):
    """Row-path nonlocal BTBT (A/um), split at x = xmid by the location of the path."""
    from src.physics import nonlocal_btbt as nl
    b = cfg["physics"]["btbt"]
    Ec = -sol.psi + 0.5 * Eg
    Ev = -sol.psi - 0.5 * Eg
    Efn = np.full(sol.x.shape, -sol.Vd)
    Efp = np.full(sol.x.shape, -cfg["bias"]["V_source"])
    dy = t._row_widths(sol.y)
    src, drn = 0.0, 0.0
    m = sol.x < xmid
    for j in range(len(sol.y)):
        # source half and drain half evaluated as separate profiles: paths do not cross
        # the channel centre at these biases (checked: halves sum to the full row current)
        src += nl.line_current(sol.x[m], Ec[j, m], Ev[j, m], A, B, Eg, b["P"], Efn=Efn[m],
                               Efp=Efp[m], kT=kT, dE=b["nonlocal_dE_eV"]) * dy[j]
        drn += nl.line_current(sol.x[~m], Ec[j, ~m], Ev[j, ~m], A, B, Eg, b["P"], Efn=Efn[~m],
                               Efp=Efp[~m], kT=kT, dE=b["nonlocal_dE_eV"]) * dy[j]
    h = 1 if t.single_gate(cfg) else 2
    return t._per_um_base(src, h), t._per_um_base(drn, h)


def run(step):
    audit = load_yaml(AUDIT)
    calib = json.load(open(audit["calibration"]))
    Vd = audit["bias"]["V_drain"]
    Vgs = np.round(np.arange(0.6, -1.4 - 1e-9, -step), 6)
    parts = []
    for name in DEVICES:
        for T in TEMPS:
            part = OUT.replace(".csv", f"_{name}_{T:g}K.csv")
            if os.path.exists(part):                       # resume: one file per device/T
                parts.append(pd.read_csv(part))
                continue
            rows = []
            cfg = ra.device_config(audit, name, T)
            dev = t.build(cfg)
            try:
                for Vg in Vgs:
                    t.ramp_to(dev, float(Vg), Vd)      # raises if not converged
                    s = t.extract(dev)
                    Eg = float(silicon.bandgap(T))
                    xmid = 0.5 * cfg["geometry"]["L_gate_cm"]
                    rec = dict(device=name, T_K=T, Vg=float(Vg), converged=True)
                    for model in ("nonlocal", "local"):
                        c = ra.scenario_config(cfg, model, D_IT if model == "nonlocal" else 0.0,
                                               TAU, calib)
                        r, _ = t.generation(s, c)
                        if model == "nonlocal":
                            rec.update(I_btbt=r["I_btbt_nonlocal"], I_nl_x=r["I_btbt_nonlocal_x"],
                                       I_nl_y=r["I_btbt_nonlocal_y"], I_it=r["I_it"],
                                       I_tat_bulk=r["I_tat_bulk"])
                            A, B = (c["physics"]["btbt"][k] for k in ("nonlocal_A_cm3s", "nonlocal_B_Vcm"))
                            from src.physics import kane_btbt
                            A, B = kane_btbt.kane_params_at_T(A, B, float(silicon.bandgap(300.0)), Eg)
                            rec["I_nl_x_src"], rec["I_nl_x_drn"] = _split_nonlocal(
                                s, c, A, B, Eg, t.K_B_EV * T, xmid)
                        else:
                            rec["I_btbt_local"] = r["I_btbt_local"]
                    rec["win_drain_eV"] = _window(s, Eg, s.x >= xmid)
                    rec["win_source_eV"] = _window(s, Eg, s.x < xmid)
                    rows.append(rec)
                print(f"fine sweep {name} T={T}: {len(Vgs)} points", flush=True)
            finally:
                dev.delete()
            os.makedirs(os.path.dirname(OUT), exist_ok=True)
            pd.DataFrame(rows).to_csv(part, index=False)
            parts.append(pd.DataFrame(rows))
    df = pd.concat(parts, ignore_index=True)
    df.to_csv(OUT, index=False)
    return df


def check_against_audit(fine):
    """Fine sweep must reproduce the audit table at the shared 0.2 V points."""
    tab = pd.read_csv(os.path.join("results", "audit_cal", "table.csv"))
    worst = 0.0
    for (dev, T), d in fine.groupby(["device", "T_K"]):
        a = tab[(tab.kind == "realistic") & (tab.device == dev) & np.isclose(tab.T_K, T)
                & (tab.btbt_model == "nonlocal") & np.isclose(tab.D_it, D_IT)
                & np.isclose(tab.tau, TAU) & (tab.dit_profile == "uniform")]
        for _, r in a.iterrows():
            f = d[np.isclose(d.Vg, r.Vg)]
            if not len(f):
                continue
            for k in ("I_btbt", "I_it", "I_tat_bulk"):
                v, w = float(f.iloc[0][k]), float(r[k])
                if w > 1e-30:
                    worst = max(worst, abs(np.log10(v / w)))
    return worst


def write_numbers(fine, worst, step):
    """In-text numbers for the components discussion -> paper/generated/numbers_fine.tex."""
    def cross(dev, T):
        g = fine[(fine.device == dev) & np.isclose(fine.T_K, T)].sort_values("Vg")
        r = np.sign(g.I_nl_x_drn.to_numpy() - g.I_it.to_numpy())
        i = np.where(np.diff(r) != 0)[0]
        return float(np.mean(g.Vg.to_numpy()[[i[0], i[0] + 1]])) if len(i) else None

    ref = fine[fine.device == "reference"]
    lines = {
        "nFineStepmV": f"{step * 1e3:.0f}",
        "nFineAuditDev": f"10^{{{int(np.ceil(np.log10(max(worst, 1e-300))))}}}",
        "nFineRefCross": f"{cross('reference', 300.0):.2f}",
        "nFineLddCross": f"{cross('low_drain_doping', 300.0):.1f}",
        "nFineRefDrnEnd": f"{ref[ref.I_nl_x_drn > 0].Vg.max():+.2f}",
        "nFineSrcOn": f"{ref[ref.I_nl_x_src > 1e-15].Vg.min():+.1f}",
    }
    out = os.path.join("paper", "generated", "numbers_fine.tex")
    with open(out, "w", encoding="utf-8") as f:
        f.write("% generated by src/analysis/fine_sweep.py -- do not edit\n")
        for k, v in lines.items():
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    print(open(out, encoding="utf-8").read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=float, default=0.02)
    ap.add_argument("--only-check", action="store_true")
    a = ap.parse_args()
    fine = pd.read_csv(OUT) if a.only_check else run(a.step)
    worst = check_against_audit(fine)
    print(f"max |log10(fine/audit)| at shared points: {worst:.2e} dec")
    step = float(np.round(np.median(np.abs(np.diff(np.sort(fine.Vg.unique())))), 6))
    write_numbers(fine, worst, step)


if __name__ == "__main__":
    main()
