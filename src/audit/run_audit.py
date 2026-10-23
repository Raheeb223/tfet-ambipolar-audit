"""Run the ambipolar-suppression audit (HANDOFF §9 step 5).

1. solve(): for every device x temperature, solve electrostatics over the Vg sweep at
   V_drain and cache the Si fields (results/audit/cache/*.npz). Traps are not charged in
   the electrostatics (ASSUMPTION, logged in the paper), so D_it, tau and the BTBT model
   only enter post-processing and can be swept without re-solving.
2. evaluate(): post-process every cached solution under every scenario -> table.csv
3. summarize(): suppression S, retention R, dominant mechanism -> summary.csv

Run:  python -m src.audit.run_audit [--only-evaluate]
With `calibration:` in the audit file, the realistic scenarios use the Kane (A, B) fitted to
measured junction data (src.calibration.btbt_calibration) separately for the local and the
nonlocal model; the baseline keeps the literature defaults, because that is what the audited
papers report.
"""

import argparse
import itertools
import json
import os

import numpy as np
import pandas as pd

from src.audit import metrics
from src.config import load_yaml
from src.devices import tfet_2d as t

OUT = os.path.join("results", "audit")      # overridden by audit["out_dir"] in main()
CACHE = os.path.join(OUT, "cache")


def device_config(audit, name, T):
    base = t.load_config(audit["base_config"])
    changes = dict(audit.get("common_changes") or {})     # e.g. sensitivity runs
    changes.update(audit["devices"][name] or {})
    changes["mesh.refine"] = audit["mesh_refine"]
    changes["temperature_K"] = T
    changes["bias.V_drain"] = audit["bias"]["V_drain"]
    return t.variant(base, **changes)


def _cache_path(name, T):
    return os.path.join(CACHE, f"{name}_T{T:.2f}.npz")


def solve(audit, names=None, temps=None, force=False):
    os.makedirs(CACHE, exist_ok=True)
    names = names or list(audit["devices"])
    temps = temps or audit["temperatures_K"]
    Vd = audit["bias"]["V_drain"]
    for name, T in itertools.product(names, temps):
        path = _cache_path(name, T)
        if os.path.exists(path) and not force:
            continue
        cfg = device_config(audit, name, T)
        dev = t.build(cfg)
        data = {}
        try:
            for k, Vg in enumerate(audit["bias"]["Vg_sweep"]):
                t.ramp_to(dev, Vg, Vd)
                s = t.extract(dev)
                data.update({f"psi_{k}": s.psi, f"n_{k}": s.n, f"p_{k}": s.p})
            np.savez_compressed(path, x=s.x, y=s.y, vol=s.vol, Vg=np.array(audit["bias"]["Vg_sweep"]),
                                Vd=Vd, T=T, cfg=json.dumps(cfg), **data)
            print(f"solved {name} T={T}")
        finally:
            dev.delete()


def load_solutions(name, T):
    z = np.load(_cache_path(name, T))
    cfg = json.loads(str(z["cfg"]))
    sols = [t.Solution(x=z["x"], y=z["y"], psi=z[f"psi_{k}"], n=z[f"n_{k}"], p=z[f"p_{k}"],
                       vol=z["vol"], Vg=float(Vg), Vd=float(z["Vd"]), T=float(z["T"]))
            for k, Vg in enumerate(z["Vg"])]
    return cfg, sols


def scenario_config(cfg, btbt_model, D_it, tau, calib=None, profile="uniform"):
    c = t.variant(cfg)
    ph = c["physics"]
    ph["btbt"]["model"] = btbt_model
    if calib:
        ph["btbt"].update(A_cm3s=calib["local"]["A"], B_Vcm=calib["local"]["B"],
                          nonlocal_A_cm3s=calib["nonlocal"]["A"], nonlocal_B_Vcm=calib["nonlocal"]["B"],
                          nonlocal_vertical=True)
    ph["interface_traps"]["profile"] = profile
    ph["tat_bulk"]["enabled"] = tau is not None
    if tau is not None:
        ph["tat_bulk"]["tau_n_s"] = ph["tat_bulk"]["tau_p_s"] = tau
    ph["interface_traps"]["enabled"] = D_it is not None and D_it > 0
    if D_it:
        ph["interface_traps"]["D_it_cm2eV"] = D_it
    return c


def evaluate(audit):
    rows = []
    sc = audit["scenarios"]
    base = audit["baseline"]
    calib = json.load(open(audit["calibration"])) if audit.get("calibration") else None
    profiles = sc.get("D_it_profile", ["uniform"])
    for name, T in itertools.product(audit["devices"], audit["temperatures_K"]):
        cfg, sols = load_solutions(name, T)
        combos = [("baseline", base["btbt_model"], None, None, "uniform")] if T == base["T_K"] else []
        combos += [("realistic", m, D, tau, pr) for m, D, tau, pr in
                   itertools.product(sc["btbt_model"], sc["D_it_cm2eV"], sc["tau_s"], profiles)
                   if not (pr != "uniform" and not D)]
        for kind, model, D_it, tau, prof in combos:
            c = scenario_config(cfg, model, D_it, tau, calib if kind == "realistic" else None, prof)
            for s in sols:
                r, _ = t.generation(s, c)
                I_btbt = r["I_btbt_nonlocal"] if model == "nonlocal" else r["I_btbt_local"]
                comp = {"btbt": I_btbt, "tat_bulk": r.get("I_tat_bulk", 0.0), "it": r.get("I_it", 0.0)}
                rows.append(dict(device=name, T_K=T, kind=kind, btbt_model=model,
                                 D_it=D_it or 0.0, tau=tau if tau is not None else np.nan,
                                 dit_profile=prof,
                                 Vg=s.Vg, I_total=sum(comp.values()),
                                 I_btbt=comp["btbt"], I_tat_bulk=comp["tat_bulk"], I_it=comp["it"],
                                 dominant=metrics.dominant_mechanism(comp)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "table.csv"), index=False)
    return df


def summarize(audit, df):
    Vg = audit["bias"]["Vg_ambipolar"]
    d = df[np.isclose(df.Vg, Vg)].copy()
    keys = ["kind", "T_K", "btbt_model", "D_it", "tau", "dit_profile"]
    d["tau"] = d["tau"].fillna(-1.0)
    ref = d[d.device == "reference"].set_index(keys)["I_total"]
    rows = []
    base = d[(d.kind == "baseline")].set_index("device")["I_total"]
    S_base = {k: float(metrics.suppression_decades(base["reference"], v)) for k, v in base.items()}
    for (dev, *k), I in d.set_index(["device"] + keys)["I_total"].items():
        if dev == "reference":
            continue
        S = float(metrics.suppression_decades(ref[tuple(k)], I))
        rows.append(dict(device=dev, **dict(zip(keys, k)), I_ref=ref[tuple(k)], I_tech=I,
                         S=S, S_baseline=S_base[dev], R=float(metrics.retention(S, S_base[dev]))))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT, "summary.csv"), index=False)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", default="configs/audit.yaml")
    ap.add_argument("--only-evaluate", action="store_true")
    a = ap.parse_args()
    audit = load_yaml(a.audit)
    global OUT, CACHE
    OUT = audit.get("out_dir", OUT)
    CACHE = audit.get("cache_dir", os.path.join(OUT, "cache"))
    os.makedirs(OUT, exist_ok=True)
    if not a.only_evaluate:
        solve(audit)
    df = evaluate(audit)
    s = summarize(audit, df)
    print(s[s.kind == "baseline"][["device", "S_baseline"]].to_string(index=False))
    print(s.groupby(["device", "btbt_model"])["R"].describe().to_string())


if __name__ == "__main__":
    main()
