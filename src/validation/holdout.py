"""Holdout test of the interface-trap model class: activation energy of the off-state current
on devices whose interface-trap density was measured on the same wafer.

The near-off-state trap limit of the audit is conditional on the Hurkx field enhancement of the
interface traps (m* <= 0.5: the limit holds; m* >= 2 or plain SRH: it does not). The two can be
told apart by the activation energy E_a of the off-state current at a KNOWN D_it, which no
published data set provides for the same devices. This module

  freeze  writes, before any holdout data exist, the model's current components versus
          temperature (BTBT, bulk TAT, and the interface-trap slope a = dI/dD_it of every
          interface variant) to data/holdout/frozen_components.csv, its SHA-256, and a
          readable table of predicted E_a on a D_it grid. These files are committed; the git
          history dates them.
  score   reads measured I_D(V_GS, T) and the measured D_it (protocol: docs/holdout_protocol.md),
          checks the frozen hash, fits the measured E_a and lists the variants it is consistent
          with, and what each surviving variant implies for the trap limit at that D_it.

Nothing is fitted to holdout data: the score only compares. Settings and assumptions:
configs/holdout.yaml.

Run:  python -m src.validation.holdout freeze
      python -m src.validation.holdout score --meas data/holdout/measured.csv \
                                             --devices data/holdout/devices.yaml
"""

import argparse
import hashlib
import os

import numpy as np
import pandas as pd

from src.config import load_yaml

CFG = "configs/holdout.yaml"
MF = os.path.join("results", "model_form")
KB = 8.617333e-5                                    # eV/K
FROZEN = "frozen_components.csv"


# ------------------------------------------------------------------ helpers
def arrhenius(T, I):
    """E_a (eV) and its standard error from a least-squares fit of ln I against 1/(kT)."""
    x, y = 1.0 / (KB * np.asarray(T, float)), np.log(np.asarray(I, float))
    if x.size < 2:
        raise ValueError("need at least two temperatures")
    slope, icpt = np.polyfit(x, y, 1)
    if x.size == 2:
        return float(-slope), 0.0
    r = y - (slope * x + icpt)
    se = np.sqrt(np.sum(r ** 2) / (x.size - 2) / np.sum((x - x.mean()) ** 2))
    return float(-slope), float(se)


def file_hash(path):
    """SHA-256 of a text file with line endings normalized (git may convert them on checkout)."""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read().replace(b"\r\n", b"\n")).hexdigest()


def variants(comp):
    return sorted(c[3:] for c in comp.component.unique() if c.startswith("it:"))


def total_current(comp, variant, device, Vg, D):
    """I(T) = I_btbt + I_bulk_tat + D_it * a_variant, sorted by temperature."""
    g = comp[(comp.device == device) & np.isclose(comp.Vg, Vg)]
    p = g.pivot(index="T_K", columns="component", values="value").sort_index()
    return p.index.values, (p["btbt"] + p["bulk_tat"] + D * p["it:" + variant]).values


def predicted_ea(comp, variant, Vg, D_lo, D_hi, n=5):
    """Range of predicted E_a over the simulated devices and the D_it interval."""
    eas = [arrhenius(*total_current(comp, variant, d, Vg, D))[0]
           for d in comp.device.unique() for D in np.geomspace(D_lo, D_hi, n)]
    return min(eas), max(eas)


def d_cross(comp, variant, device, Vg, T=300.0):
    """D_it at which the interface-trap current equals the rest (cm^-2 eV^-1)."""
    g = comp[(comp.device == device) & np.isclose(comp.Vg, Vg) & np.isclose(comp.T_K, T)]
    v = dict(zip(g.component, g.value))
    return (v["btbt"] + v["bulk_tat"]) / v["it:" + variant]


def predicted_slope(comp, variant, Vg, Ds, T=300.0):
    """Range over the simulated devices of d ln I / d ln D_it across the D_it values Ds (a
    least-squares slope): ~1 where the traps carry the current, ~0 where they do not."""
    x, out = np.log(Ds), []
    for dev in comp.device.unique():
        g = comp[(comp.device == dev) & np.isclose(comp.Vg, Vg) & np.isclose(comp.T_K, T)]
        v = dict(zip(g.component, g.value))
        out.append(np.polyfit(x, np.log(v["btbt"] + v["bulk_tat"] + np.asarray(Ds) * v["it:" + variant]), 1)[0])
    return min(out), max(out)


def consistent(ea, se, lo, hi, cfg):
    tol = cfg["n_sigma"] * se + cfg["ea_tol_eV"]
    return lo - tol <= ea <= hi + tol


# ------------------------------------------------------------------ freeze
def freeze(cfg, force=False):
    path = os.path.join(cfg["frozen_dir"], FROZEN)
    if os.path.exists(path) and not force:
        raise RuntimeError(f"{path} exists: predictions are frozen (re-freezing after holdout "
                           "data exist would void the test)")
    sl = pd.read_csv(os.path.join(MF, "it_slopes.csv"))
    bt = pd.read_csv(os.path.join(MF, "btbt_variants.csv"))
    bt = bt[(bt.path == "rows_cols_max") & (bt.params == "nominal")]
    rows = [dict(device=r.device, Vg=r.Vg, T_K=r.T_K, component="btbt", value=r.I_btbt)
            for r in bt.itertuples()]
    rows += [dict(device=r.device, Vg=r.Vg, T_K=r.T_K,
                  component="bulk_tat" if r.variant == "bulk_tat" else "it:" + r.variant, value=r.a)
             for r in sl.itertuples()]
    comp = pd.DataFrame(rows).sort_values(["component", "device", "Vg", "T_K"])
    os.makedirs(cfg["frozen_dir"], exist_ok=True)
    comp.to_csv(path, index=False, float_format="%.10e", lineterminator="\n")
    with open(path + ".sha256", "w", newline="\n") as f:
        f.write(f"{file_hash(path)}  {FROZEN}\n")
    comp = pd.read_csv(path)
    # slope: d ln I / d ln D_it across a tenfold split centred on D (D/sqrt(10) .. D*sqrt(10))
    tab = [dict(variant=v, Vg=Vg, Dit=D, Ea_min=lo, Ea_max=hi, slope_min=s0, slope_max=s1)
           for v in variants(comp) for Vg in cfg["vg_all"] for D in cfg["dit_grid"]
           for lo, hi in [predicted_ea(comp, v, Vg, D, D, n=1)]
           for s0, s1 in [predicted_slope(comp, v, Vg, D * np.array([10 ** -0.5, 10 ** 0.5]))]]
    pd.DataFrame(tab).to_csv(os.path.join(cfg["frozen_dir"], "frozen_ea_table.csv"), index=False,
                             float_format="%.4g", lineterminator="\n")
    print("frozen:", path, file_hash(path))


# ------------------------------------------------------------------ paper numbers
def numbers(cfg, out="paper/generated/numbers_holdout.tex", D=1e12):
    """Macros for the manuscript from the frozen predictions (plain reference device, decisive
    bias, tenfold split centred on D)."""
    comp = load_frozen(cfg)
    comp = comp[comp.device == "reference"]
    Vg, Ds = cfg["vg_decisive"], D * np.array([10 ** -0.5, 10 ** 0.5])
    eas = [predicted_ea(comp, v, Vg, D, D, n=1) for v in variants(comp)]
    m = {"HoldEaLo": f"{min(e[0] for e in eas):.2f}", "HoldEaHi": f"{max(e[1] for e in eas):.2f}",
         "HoldDit": f"10^{{{np.log10(D):.0f}}}"}
    for key, v in (("Nom", "nominal"), ("Mhalf", "m_0.5"), ("Mtwo", "m_2.0"), ("Srh", "no_field_enh")):
        m["HoldSlope" + key] = f"{predicted_slope(comp, v, Vg, Ds)[0]:.2f}"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("% generated by src.validation.holdout numbers -- do not edit\n")
        for k, v in m.items():
            f.write(f"\\newcommand{{\\n{k}}}{{{v}}}\n")
    print(open(out).read())


# ------------------------------------------------------------------ score
def load_frozen(cfg):
    path = os.path.join(cfg["frozen_dir"], FROZEN)
    want = open(path + ".sha256").read().split()[0]
    if file_hash(path) != want:
        raise RuntimeError(f"{path} differs from its frozen hash: predictions were changed")
    return pd.read_csv(path)


def measured_ea(meas, cfg):
    """Per device and V_GS: E_a and its standard error over the fit window. Also flags the
    off-state point (minimum current at the temperature nearest 300 K)."""
    lo, hi = cfg["fit_window_K"]
    m = meas[(meas.T_K >= lo) & (meas.T_K <= hi)].copy()
    m["V"] = m.V_GS_V.round(4)
    rows = []
    for dev, g in m.groupby("device_id"):
        T0 = g.T_K.iloc[(g.T_K - 300.0).abs().argmin()]
        g0 = g[g.T_K == T0]
        v_off = g0.V.iloc[g0.I_D_A_per_um.abs().argmin()]
        for V, h in g.groupby("V"):
            if h.T_K.nunique() < cfg["min_temperatures"]:
                continue
            h = h.groupby("T_K").I_D_A_per_um.apply(lambda s: s.abs().mean())
            ea, se = arrhenius(h.index.values, h.values)
            rows.append(dict(device_id=dev, V_GS_V=V, Ea=ea, Ea_se=se, n_T=h.size,
                             I_ref=float(h.iloc[np.abs(h.index.values - 300.0).argmin()]),
                             off_state=bool(np.isclose(V, v_off))))
    return pd.DataFrame(rows)


def score(cfg, meas_path, dev_path):
    comp = load_frozen(cfg)
    if cfg.get("sim_devices"):                    # compare with these simulated devices only
        comp = comp[comp.device.isin(cfg["sim_devices"])]
    meas = pd.read_csv(meas_path)
    devs = load_yaml(dev_path)["devices"]
    ea = measured_ea(meas, cfg)
    if not ea.off_state.any():
        raise ValueError(f"no off-state point with >= {cfg['min_temperatures']} temperatures in "
                         f"{cfg['fit_window_K']} K")
    Vg = cfg["vg_decisive"]
    rows = []
    for r in ea[ea.off_state].itertuples():
        d = devs[r.device_id]
        D, D_lo, D_hi = d["Dit_cm2eV"], d.get("Dit_lo_cm2eV", d["Dit_cm2eV"]), d.get("Dit_hi_cm2eV", d["Dit_cm2eV"])
        for v in variants(comp):
            lo, hi = predicted_ea(comp, v, Vg, D_lo, D_hi)
            dx = [d_cross(comp, v, dv, Vg) for dv in comp.device.unique()]
            rows.append(dict(device_id=r.device_id, Dit=D, Ea_meas=r.Ea, Ea_se=r.Ea_se,
                             variant=v, Ea_pred_min=lo, Ea_pred_max=hi,
                             consistent=consistent(r.Ea, r.Ea_se, lo, hi, cfg),
                             Dx_min=min(dx), Dx_max=max(dx),
                             # trap-limited at the measured D_it in every simulated device / none
                             trap_limit="all" if max(dx) < D else "none" if min(dx) > D else "some"))
    out = pd.DataFrame(rows)
    out["slope_meas"], out["slope_ok"] = np.nan, True
    # D_it split: devices of one design (same "split" key) with different measured D_it
    off = ea[ea.off_state].set_index("device_id")
    split = {k: d["split"] for k, d in devs.items() if "split" in d}
    for key in sorted(set(split.values())):
        ids = [k for k, v in split.items() if v == key and k in off.index]
        Ds = np.array([devs[i]["Dit_cm2eV"] for i in ids])
        if len(set(Ds)) < 2:
            continue
        s_meas = np.polyfit(np.log(Ds), np.log(off.loc[ids, "I_ref"].values), 1)[0]
        for v in variants(comp):
            lo, hi = predicted_slope(comp, v, Vg, Ds)
            sel = out.device_id.isin(ids) & (out.variant == v)
            out.loc[sel, "slope_meas"] = s_meas
            out.loc[sel, "slope_ok"] = lo - cfg["slope_tol"] <= s_meas <= hi + cfg["slope_tol"]
    out["Ea_ok"] = out.consistent
    out["consistent"] = out.Ea_ok & out.slope_ok
    os.makedirs(cfg["out_dir"], exist_ok=True)
    ea.to_csv(os.path.join(cfg["out_dir"], "ea_measured.csv"), index=False)
    out.to_csv(os.path.join(cfg["out_dir"], "verdict.csv"), index=False)
    return ea, out


def summary(out):
    lines = []
    for dev, g in out.groupby("device_id"):
        ok = g[g.consistent]
        r = g.iloc[0]
        lines.append(f"{dev}: D_it = {r.Dit:.2g} cm^-2 eV^-1, E_a = {r.Ea_meas:.3f} +- {r.Ea_se:.3f} eV"
                     + ("" if np.isnan(r.slope_meas) else f", d ln I / d ln D_it = {r.slope_meas:.2f}"))
        if ok.empty:
            lines.append("  no interface variant is consistent: the trap model class is rejected "
                         "for this device; the conditional trap limit is not settled by it")
            continue
        lines.append("  consistent variants: " + ", ".join(ok.variant))
        verdicts = set(ok.trap_limit)
        if verdicts == {"all"}:
            lines.append("  all of them put every technique trap-limited at this D_it: the trap limit HOLDS")
        elif verdicts == {"none"}:
            lines.append("  none of them makes any technique trap-limited at this D_it: the trap limit FAILS")
        else:
            lines.append("  the consistent variants disagree on the trap limit at this D_it: not settled"
                         + (" (a D_it split of the same design would separate them)" if np.isnan(r.slope_meas) else ""))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["freeze", "score", "numbers"])
    ap.add_argument("--meas", default="data/holdout/measured.csv")
    ap.add_argument("--devices", default="data/holdout/devices.yaml")
    ap.add_argument("--force", action="store_true", help="re-freeze (only before any holdout data)")
    a = ap.parse_args()
    cfg = load_yaml(CFG)
    if a.step == "freeze":
        freeze(cfg, a.force)
    elif a.step == "numbers":
        numbers(cfg)
    else:
        _, out = score(cfg, a.meas, a.devices)
        print(summary(out))


if __name__ == "__main__":
    main()
