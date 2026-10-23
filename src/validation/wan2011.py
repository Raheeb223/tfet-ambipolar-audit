"""Blind validation on the measured SOI n-TFETs of Wan et al. 2011 (role: validation).

The BTBT parameters come from results/calibration/btbt_calibration.json (junction data only).
Device structure from Wan 2011 / Biswas 2012 (configs/wan2011_*.yaml). The implant-edge offset
(not stated in the papers) is NOT fitted: it is scanned over an a-priori range as a systematic.
Interface traps are scanned through linearity: I_it is stored per unit D_it (the device is
solved without trap charge, so I_it = D_it * i_it exactly).

Run: python -m src.validation.wan2011    -> results/validation/wan2011_sim.csv
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

from src.devices import tfet_2d as t

OUT = os.path.join("results", "validation")
CAL = os.path.join("results", "calibration", "btbt_calibration.json")
LG = 100e-7
DEVICES = {
    "sio2": dict(config="configs/wan2011_sio2.yaml", L_IN_nm=[10, 25, 40, 50],
                 Vg=list(np.round(np.arange(-0.5, -6.01, -0.5), 3))),
    "hfo2": dict(config="configs/wan2011_hfo2.yaml", L_IN_nm=[10, 25, 50],
                 Vg=list(np.round(np.arange(-0.25, -3.01, -0.25), 3))),
}
OFFSETS_NM = [0.0, 10.0, 20.0]     # a-priori systematic (spacer / mask alignment), not fitted
D_REF = 1.0e11


def calibrated(cfg):
    cal = json.load(open(CAL))
    b = cfg["physics"]["btbt"]
    b.update(A_cm3s=cal["local"]["A"], B_Vcm=cal["local"]["B"],
             nonlocal_A_cm3s=cal["nonlocal"]["A"], nonlocal_B_Vcm=cal["nonlocal"]["B"],
             nonlocal_vertical=True, model="both")
    cfg["physics"]["interface_traps"]["D_it_cm2eV"] = D_REF
    return cfg


def device_cfg(base, L_IN_nm, offset_nm):
    L = L_IN_nm * 1e-7
    off = offset_nm * 1e-7
    return t.variant(base, **{"geometry.L_gate_cm": LG + L, "geometry.gate_underlap_drain_cm": L,
                              "doping.drain_profile.offset_cm": off,
                              "mesh.fine_ranges_cm": [[LG - 1.0e-6, LG + L + off]]})


def run(which, offsets=OFFSETS_NM):
    spec = DEVICES[which]
    base = calibrated(t.load_config(spec["config"]))
    rows = []
    for off in offsets:
        for L in spec["L_IN_nm"]:
            cfg = device_cfg(base, L, off)
            dev = t.build(cfg)
            sols = []
            try:
                for Vg in spec["Vg"]:
                    t.ramp_to(dev, Vg, cfg["bias"]["V_drain"])
                    sols.append(t.extract(dev))
                    r, _ = t.generation(sols[-1], cfg)
                    rows.append(dict(device=which, L_IN_nm=L, offset_nm=off, Vg=Vg,
                                     I_btbt_local=r["I_btbt_local"], I_btbt_nl=r["I_btbt_nonlocal"],
                                     I_btbt_nl_x=r["I_btbt_nonlocal_x"], I_btbt_nl_y=r.get("I_btbt_nonlocal_y", 0.0),
                                     I_tat_bulk=r["I_tat_bulk"], i_it_per_D=r["I_it"] / D_REF))
                    print(which, off, L, Vg, f"nl={r['I_btbt_nonlocal']:.2e} loc={r['I_btbt_local']:.2e} "
                          f"it={r['I_it']:.2e}", flush=True)
            finally:
                dev.delete()
            if sols:   # cached for post-processing checks (src.audit.model_form)
                t.save_solutions(cache_path(which, L, off), sols, cfg)
    return pd.DataFrame(rows)


def cache_path(which, L_IN_nm, offset_nm):
    return os.path.join(OUT, "cache", f"wan_{which}_L{L_IN_nm:g}_off{offset_nm:g}.npz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--devices", nargs="+", default=list(DEVICES))
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    df = pd.concat([run(w) for w in a.devices], ignore_index=True)
    path = os.path.join(OUT, "wan2011_sim.csv")
    df.to_csv(path, index=False)
    print("wrote", path, len(df))


if __name__ == "__main__":
    main()
