"""Which fields carry the BTBT current in the audit devices, and are they inside the field range
of the calibration data? (A calibrated model is only supported where it was calibrated.)

For each device at 300 K and the ambipolar biases: the current-weighted median field and the
fraction of the current generated at fields inside [F_min, F_max] of the calibration data, for
the local model (point field) and the nonlocal model (path-averaged field, x and y paths).
Run: python -m src.analysis.coverage  -> results/audit_cal/calibration_coverage.csv
"""

import json
import os

import numpy as np
import pandas as pd

from src.audit import run_audit as ra
from src.config import load_yaml
from src.devices import tfet_2d as t


def coverage(audit_file="configs/audit_calibrated.yaml", T=300.0, Vgs=(-1.0, -0.6, -0.4, -0.2)):
    audit = load_yaml(audit_file)
    ra.CACHE = audit["cache_dir"]
    cal = json.load(open(audit["calibration"]))
    Fmin, Fmax = min(cal["data"]["F_Vcm"]), max(cal["data"]["F_Vcm"])
    rows = []
    for Vg in Vgs:
        for d in audit["devices"]:
            cfg, sols = ra.load_solutions(d, T)
            s = [x for x in sols if np.isclose(x.Vg, Vg)][0]
            tm = t.btbt_terms(s, ra.scenario_config(cfg, "nonlocal", None, None, cal))
            for model, names, p in (("local", ["local"], cal["local"]),
                                    ("nonlocal", ["nonlocal_x", "nonlocal_y"], cal["nonlocal"])):
                F = np.concatenate([tm[n][0] for n in names])
                I = np.concatenate([tm[n][1] * p["A"] * np.where(tm[n][0] > 0, tm[n][0], 1.0) ** tm[n][2]
                                    * np.exp(-p["B"] / np.where(tm[n][0] > 0, tm[n][0], 1e-30)) for n in names])
                o = np.argsort(F)
                cdf = np.cumsum(I[o]) / I.sum()
                rows.append(dict(device=d, Vg=Vg, model=model,
                                 F_median_MVcm=float(F[o][np.searchsorted(cdf, 0.5)] / 1e6),
                                 frac_in_range=float(I[(F >= Fmin) & (F <= Fmax)].sum() / I.sum()),
                                 frac_above=float(I[F > Fmax].sum() / I.sum())))
    out = pd.DataFrame(rows)
    os.makedirs(audit["out_dir"], exist_ok=True)
    out.to_csv(os.path.join(audit["out_dir"], "calibration_coverage.csv"), index=False)
    return out


if __name__ == "__main__":
    print(coverage().round(2).to_string(index=False))
