"""Second frozen-model validation: on-state transfer curves of the CEA-LETI SOI n-TFET of Biswas et al.
2012, Fig. 3 (role: validation; measured at VD = 0.5, 1.0, 1.5 V, LG = 400 nm, LIN = 100 nm).

Same wafer and process as Wan 2011 (20 nm Si, 145 nm BOX, 6 nm SiO2, WF 4.3 eV, source B 8e19,
~7 nm/dec junctions), so configs/wan2011_sio2.yaml is reused unchanged except for the stated
LG and LIN. The BTBT parameters are the junction-calibrated ones (results/calibration); nothing
is fitted to these curves. This tests a different branch (source-side, on-state tunneling) and a
different gate length from the Wan ambipolar validation.

Note: Biswas et al. fitted their own non-local model to these curves and needed m_r = 0.033
(B = 1.47e7 V/cm); our parameters come from junction data and are not adjusted here.

Run: python -m src.validation.biswas2012   -> results/validation/biswas2012_sim.csv
"""

import os

import numpy as np
import pandas as pd

from src.devices import tfet_2d as t
from src.validation.wan2011 import OFFSETS_NM, calibrated

OUT = os.path.join("results", "validation")
CONFIG = "configs/wan2011_sio2.yaml"
LG = 400e-7                      # [Biswas 2012, Sec. 4.1] LG = 400 nm
L_IN = 100e-7                    # [Biswas 2012, Sec. 4.1] LIN = 100 nm
VD = [0.5, 1.0, 1.5]             # [Biswas 2012, Fig. 3]
VG = list(np.round(np.arange(0.0, 3.001, 0.25), 3))
D_REF = 1.0e10                   # nominal thermal-SiO2 D_it (same as the Wan SiO2 validation)


def device_cfg(base, offset_nm):
    off = offset_nm * 1e-7
    return t.variant(base, **{"geometry.L_gate_cm": LG + L_IN, "geometry.gate_underlap_drain_cm": L_IN,
                              "doping.drain_profile.offset_cm": off,
                              "mesh.fine_ranges_cm": [[-1.0e-6, 1.0e-6], [LG - 1.0e-6, LG + L_IN + off]]})


def run(offsets=OFFSETS_NM):
    base = calibrated(t.load_config(CONFIG))
    base["physics"]["interface_traps"]["D_it_cm2eV"] = D_REF
    rows = []
    for off in offsets:
        cfg = device_cfg(base, off)
        dev = t.build(cfg)
        sols = []
        try:
            for Vd in VD:
                t.ramp_to(dev, VG[0], Vd)
                for Vg in VG:
                    t.ramp_to(dev, Vg, Vd)
                    sols.append(t.extract(dev))
                    r, _ = t.generation(sols[-1], cfg)
                    rows.append(dict(offset_nm=off, Vd=Vd, Vg=Vg, I_btbt_local=r["I_btbt_local"],
                                     I_btbt_nl=r["I_btbt_nonlocal"], I_tat_bulk=r["I_tat_bulk"],
                                     i_it_per_D=r["I_it"] / D_REF))
                    print(off, Vd, Vg, f"nl={r['I_btbt_nonlocal']:.2e} loc={r['I_btbt_local']:.2e} "
                          f"tat={r['I_tat_bulk']:.2e} it={r['I_it']:.2e}", flush=True)
        finally:
            dev.delete()
        if sols:   # cached for post-processing checks (src.audit.model_form)
            t.save_solutions(cache_path(off), sols, cfg)
    return pd.DataFrame(rows)


def cache_path(offset_nm):
    return os.path.join(OUT, "cache", f"biswas_off{offset_nm:g}.npz")


def field_coverage(points=((1.5, 1.0), (2.0, 1.0), (3.0, 0.5), (3.0, 1.0), (3.0, 1.5)), offset_nm=10.0):
    """Current-weighted median tunneling field and the fraction of the BTBT current generated
    inside the calibrated field range, at selected on-state biases (same definition as
    src/analysis/coverage.py)."""
    import json
    cal = json.load(open(os.path.join("results", "calibration", "btbt_calibration.json")))
    Fmin, Fmax = min(cal["data"]["F_Vcm"]), max(cal["data"]["F_Vcm"])
    base = calibrated(t.load_config(CONFIG))
    cfg = device_cfg(base, offset_nm)
    dev = t.build(cfg)
    rows = []
    try:
        for Vg, Vd in points:
            t.ramp_to(dev, Vg, Vd)
            tm = t.btbt_terms(t.extract(dev), cfg)
            for model, names, p in (("local", ["local"], cal["local"]),
                                    ("nonlocal", ["nonlocal_x", "nonlocal_y"], cal["nonlocal"])):
                F = np.concatenate([tm[n][0] for n in names])
                Fs = np.where(F > 0, F, 1e-30)
                I = np.concatenate([tm[n][1] for n in names]) * p["A"] * Fs ** tm[names[0]][2] * np.exp(-p["B"] / Fs)
                o = np.argsort(F)
                cdf = np.cumsum(I[o]) / I.sum()
                rows.append(dict(Vg=Vg, Vd=Vd, model=model,
                                 F_median_MVcm=float(F[o][np.searchsorted(cdf, 0.5)] / 1e6),
                                 frac_in_range=float(I[(F >= Fmin) & (F <= Fmax)].sum() / I.sum()),
                                 frac_below=float(I[F < Fmin].sum() / I.sum())))
    finally:
        dev.delete()
    return pd.DataFrame(rows)


def main():
    import sys
    os.makedirs(OUT, exist_ok=True)
    if "--coverage" in sys.argv:
        cov = field_coverage()
        cov.to_csv(os.path.join(OUT, "biswas2012_coverage.csv"), index=False)
        print(cov.round(3).to_string(index=False))
        return
    df = run()
    path = os.path.join(OUT, "biswas2012_sim.csv")
    df.to_csv(path, index=False)
    print("wrote", path, len(df))


if __name__ == "__main__":
    main()
