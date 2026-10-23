"""Propagate the BTBT calibration uncertainty to the audit metrics (HANDOFF UQ step).

For each device and temperature the cached electrostatic solution at the ambipolar bias is
reduced once to parameter-free BTBT terms (tfet_2d.btbt_terms); every Monte Carlo sample of
(log10 A, B) drawn from the calibration covariance then costs only a weighted sum. The interface
and bulk TAT terms do not depend on (A, B) and are evaluated once.

Outputs results/uq/uq_samples.csv and results/uq/uq_summary.csv:
  S(D_it) for D_it in {0, 1e11, 1e12, 5e12}, D_x = I_btbt/a, and the probability that each
  technique keeps its rank, all as median and 5-95 % interval.
Run: python -m src.uq.btbt_uq
"""

import json
import os

import numpy as np
import pandas as pd

from src.audit import run_audit as ra
from src.config import load_yaml
from src.devices import tfet_2d as t
from src.materials import silicon
from src.physics import kane_btbt

OUT = os.path.join("results", "uq")
N = 2000
SEED = 20261023
D_LIST = [0.0, 1e11, 1e12, 5e12]


def _samples(cal, model, rng, n=N):
    r = cal[model]
    sd = np.array([r["sd_log10A"], r["sd_B"]])
    cov = np.array([[1.0, r["corr"]], [r["corr"], 1.0]]) * np.outer(sd, sd)
    s = rng.multivariate_normal([r["log10A"], r["B"]], cov, size=n)
    return 10 ** s[:, 0], s[:, 1]


def device_terms(audit, name, T, Vg, tau=1e-7):
    ra.CACHE = audit.get("cache_dir", ra.CACHE)
    cfg, sols = ra.load_solutions(name, T)
    s = [x for x in sols if np.isclose(x.Vg, Vg)][0]
    c = ra.scenario_config(cfg, "local", 1.0, tau)        # D_it = 1 -> I_it is the slope a
    c["physics"]["btbt"]["nonlocal_vertical"] = True
    r, _ = t.generation(s, c)
    return t.btbt_terms(s, c), r["I_it"], r["I_tat_bulk"]


def run(audit_file="configs/audit_calibrated.yaml", Vg=-1.0, temps=(300.0, 398.15), n=N):
    audit = load_yaml(audit_file)
    cal = json.load(open(audit["calibration"]))
    rng = np.random.default_rng(SEED)
    draws = {m: _samples(cal, m, rng, n) for m in ("local", "nonlocal")}
    rows = []
    for T in temps:
        Eg_ratio_args = (float(silicon.bandgap(300.0)), float(silicon.bandgap(T)))
        terms = {d: device_terms(audit, d, T, Vg) for d in audit["devices"]}
        for model in ("local", "nonlocal"):
            A, B = draws[model]
            for k in range(n):
                At, Bt = kane_btbt.kane_params_at_T(A[k], B[k], *Eg_ratio_args)
                I0, a = {}, {}
                for d, (tm, i_it, i_tb) in terms.items():
                    if model == "local":
                        Ib = t.btbt_from_terms(tm["local"], At, Bt)
                    else:
                        Ib = max(t.btbt_from_terms(tm["nonlocal_x"], At, Bt),
                                 t.btbt_from_terms(tm["nonlocal_y"], At, Bt))
                    I0[d], a[d] = Ib + i_tb, i_it
                    rows.append(dict(T_K=T, model=model, sample=k, device=d, I_btbt=Ib,
                                     I_tat_bulk=i_tb, a_it=i_it))
    df = pd.DataFrame(rows)
    return df


def summarize(df):
    out = []
    techs = [d for d in df.device.unique() if d != "reference"]
    for (T, model), g in df.groupby(["T_K", "model"]):
        piv = g.pivot(index="sample", columns="device")
        I0 = piv["I_btbt"] + piv["I_tat_bulk"]
        a = piv["a_it"]
        S = {D: np.log10((I0["reference"] + a["reference"] * D).values[:, None] /
                         (I0[techs] + a[techs] * D).values) for D in D_LIST}
        rank_nom = {D: np.argsort(-np.median(S[D], axis=0)) for D in D_LIST}
        for j, d in enumerate(techs):
            Dx = (I0[d] / a[d]).values           # all non-interface generation, as in the paper
            row = dict(T_K=T, model=model, device=d,
                       S_inf=float(np.log10(a["reference"].iloc[0] / a[d].iloc[0])),
                       Dx_med=float(np.median(Dx)), Dx_p05=float(np.percentile(Dx, 5)),
                       Dx_p95=float(np.percentile(Dx, 95)))
            for D in D_LIST:
                v = S[D][:, j]
                row.update({f"S_{D:.0e}_med": float(np.median(v)), f"S_{D:.0e}_p05": float(np.percentile(v, 5)),
                            f"S_{D:.0e}_p95": float(np.percentile(v, 95))})
                ranks = np.argsort(np.argsort(-S[D], axis=1), axis=1)[:, j]
                nominal = int(np.where(rank_nom[D] == j)[0][0])
                row[f"P_rank_{D:.0e}"] = float(np.mean(ranks == nominal))
            out.append(row)
    return pd.DataFrame(out)


def main():
    os.makedirs(OUT, exist_ok=True)
    frames = []
    for Vg in (-1.0, -0.6, -0.2):
        df = run(Vg=Vg)
        df["Vg"] = Vg
        frames.append(df)
        s = summarize(df)
        s["Vg"] = Vg
        s.to_csv(os.path.join(OUT, f"uq_summary_Vg{Vg:+.1f}.csv"), index=False)
        print(f"Vg={Vg}")
        print(s.round(3).to_string(index=False))
    pd.concat(frames).to_csv(os.path.join(OUT, "uq_samples.csv.gz"), index=False)


if __name__ == "__main__":
    main()
