"""Charged-trap check of the audit: re-solve the electrostatics with self-consistent amphoteric
interface-trap charge (tfet_2d.solve_charged_traps) and compare the ambipolar current and the
suppression S with the uncharged-trap audit (where I(D_it) is exactly linear).

Calibrated BTBT parameters (local and nonlocal), nonlocal paths in x and y, uniform D_it.
Run: python -m src.audit.charged_traps   -> results/audit_cal/charged_traps.csv
"""

import json
import os

import numpy as np
import pandas as pd

from src.audit import run_audit as ra
from src.config import load_yaml
from src.devices import tfet_2d as t

D_LIST = [1e11, 1e12, 5e12, 1e13]    # 1e13: upper end of the verified linear range
VG_LIST = [-0.2, -0.6, -1.0]     # -0.2 V: the near-off-state bias of the main claim
TEMPS = [300.0, 398.15]
TAU = 1e-7


def main(audit_file="configs/audit_calibrated.yaml"):
    audit = load_yaml(audit_file)
    cal = json.load(open(audit["calibration"]))
    rows = []
    Vd = audit["bias"]["V_drain"]
    for name in audit["devices"]:
        for T in TEMPS:
            cfg0 = ra.device_config(audit, name, T)
            dev = t.build(cfg0)
            try:
                for Vg in VG_LIST:
                    t.clear_trap_charge(dev)
                    t.ramp_to(dev, Vg, Vd)
                    for D in [0.0] + D_LIST:
                        c = ra.scenario_config(cfg0, "nonlocal", D if D else None, TAU, cal)
                        it_iter = 0
                        for charged in ((False, True) if D else (False,)):
                            if charged:
                                try:
                                    it_iter = t.solve_charged_traps(dev, c)
                                except RuntimeError:          # recorded as not converged
                                    t.clear_trap_charge(dev)
                                    rows.append(dict(device=name, T_K=T, Vg=Vg, D_it=D, charged=True,
                                                     iterations=-1, I_btbt_local=np.nan, I_btbt_nl=np.nan,
                                                     I_tat_bulk=np.nan, I_it=np.nan))
                                    continue
                            r, _ = t.generation(t.extract(dev), c)
                            rows.append(dict(device=name, T_K=T, Vg=Vg, D_it=D, charged=charged,
                                             iterations=it_iter, I_btbt_local=r["I_btbt_local"],
                                             I_btbt_nl=r["I_btbt_nonlocal"], I_tat_bulk=r["I_tat_bulk"],
                                             I_it=r.get("I_it", 0.0)))
                        t.clear_trap_charge(dev)
                        print(name, T, Vg, D, f"{rows[-1]['I_btbt_nl']:.3e} {rows[-1]['I_it']:.3e}", flush=True)
            finally:
                dev.delete()
    df = pd.DataFrame(rows)
    out = audit.get("out_dir", "results/audit_cal")
    os.makedirs(out, exist_ok=True)
    df.to_csv(os.path.join(out, "charged_traps.csv"), index=False)
    # suppression with and without trap charge
    df["I_total_nl"] = df.I_btbt_nl + df.I_tat_bulk + df.I_it
    key = ["T_K", "Vg", "D_it", "charged"]
    ref = df[df.device == "reference"].set_index(key)["I_total_nl"]
    tech = df[df.device != "reference"].copy()
    tech["S"] = [np.log10(ref[tuple(r[k] for k in key)] / r.I_total_nl) for _, r in tech.iterrows()]
    piv = tech.pivot_table(index=["device", "T_K", "Vg", "D_it"], columns="charged", values="S")
    piv.columns = ["S_uncharged", "S_charged"]
    piv["dS"] = piv.S_charged - piv.S_uncharged
    piv.to_csv(os.path.join(out, "charged_traps_S.csv"))
    print(piv.round(2).to_string())


if __name__ == "__main__":
    main()
