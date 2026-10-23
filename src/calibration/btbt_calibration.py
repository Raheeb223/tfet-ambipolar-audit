"""Calibrate the Kane (A, B) parameters of the local and the nonlocal BTBT model on measured
Si junction tunneling data (role core_calibration only; validation data are refused by
src.dataio.digitized.calibration_datasets).

Data: J at -1 V vs peak junction field F_max for one-sided n+p Si junctions
(Fair & Wivell 1976; Stork & Isaac 1983; compiled in Zhao 2007, Fig. 2.3).

Junction model (ASSUMPTIONS, logged):
  * abrupt one-sided junction in the depletion approximation: the field falls linearly from
    F_max to 0 over W = 2 V_t / F_max, V_t = V_bi + V_R. For any abrupt junction (one- or
    two-sided) J then depends only on (F_max, V_t), so the unknown doping split is irrelevant;
  * V_bi = Eg/q (Fermi levels at the band edges of the degenerate sides), as in Zhao 2007;
    varied 0.95-1.25 V as a systematic;
  * P = 2.5 fixed (phonon-assisted Si), only A and B are fitted; T = 300 K.
Local:    J = q * integral G(F(x)) dx = q (W / F_max) * integral_0^F_max G(F) dF.
Nonlocal: src.physics.nonlocal_btbt.line_current on the same band profile, with the
          n-side quasi-Fermi level at Ec and the p-side one at Ev (+ V_R).
"""

import numpy as np
from scipy.optimize import least_squares

from src.constants import Q
from src.physics import kane_btbt, nonlocal_btbt
from src.physics.kane_btbt import P_INDIRECT

EG_300 = 1.1245   # eV, src.materials.silicon.bandgap(300)


def junction_bands(F_max, V_t, Eg=EG_300, n=3001):
    """Ec, Ev (eV) of an abrupt reverse-biased n+p junction: n+ side (x < 0) flat, depletion
    on the p side over [0, W]. Returns x (cm), Ec, Ev, W."""
    W = 2.0 * V_t / F_max
    x = np.linspace(-0.25 * W, 1.25 * W, n)
    u = np.clip(x / W, 0.0, 1.0)
    Ec = V_t * (1.0 - (1.0 - u) ** 2)      # electron energy rises by V_t across the depletion
    return x, Ec, Ec - Eg, W


def j_local(F_max, A, B, V_t, P=P_INDIRECT, n=4001):
    F_max = np.atleast_1d(np.asarray(F_max, dtype=float))
    out = np.empty_like(F_max)
    for k, Fm in enumerate(F_max):
        F = np.linspace(0.0, Fm, n)
        out[k] = Q * (2.0 * V_t / Fm) / Fm * np.trapezoid(kane_btbt.kane_generation(F, A, B, P), F)
    return out


def j_nonlocal(F_max, A, B, V_t, V_R=1.0, Eg=EG_300, P=P_INDIRECT, dE=2e-3, kT=0.02585):
    F_max = np.atleast_1d(np.asarray(F_max, dtype=float))
    out = np.empty_like(F_max)
    for k, Fm in enumerate(F_max):
        x, Ec, Ev, _ = junction_bands(Fm, V_t, Eg)
        Efn = np.full_like(x, Ec[0])               # n+ side: E_F at the conduction-band edge
        Efp = np.full_like(x, Ec[0] + V_R)         # p side: E_F = Ev(p) = Ec(n) + V_R  (V_bi = Eg)
        out[k] = nonlocal_btbt.line_current(x, Ec, Ev, A, B, Eg, P, Efn=Efn, Efp=Efp, kT=kT, dE=dE)
    return out


MODELS = {"local": j_local, "nonlocal": j_nonlocal}


def fit(F, J, model, V_t=EG_300 + 1.0, p0=(np.log10(4e14), 1.9e7)):
    """Least squares in log10 J over (log10 A, B). Returns dict with values, 1-sigma errors,
    correlation, rms residual (decades) and residuals."""
    f = MODELS[model]
    F, J = np.asarray(F, float), np.asarray(J, float)

    def res(p):
        return np.log10(f(F, 10 ** p[0], p[1] * 1e7, V_t)) - np.log10(J)

    sol = least_squares(res, x0=[p0[0], p0[1] / 1e7], x_scale=[1.0, 0.1], method="lm")
    r = sol.fun
    dof = max(len(F) - 2, 1)
    s2 = float(r @ r) / dof
    cov = np.linalg.inv(sol.jac.T @ sol.jac) * s2
    sd = np.sqrt(np.diag(cov))
    return {
        "model": model, "V_t": V_t,
        "log10A": float(sol.x[0]), "A": float(10 ** sol.x[0]), "B": float(sol.x[1] * 1e7),
        "sd_log10A": float(sd[0]), "sd_B": float(sd[1] * 1e7),
        "corr": float(cov[0, 1] / (sd[0] * sd[1])),
        "rms_dec": float(np.sqrt(np.mean(r ** 2))), "max_dec": float(np.max(np.abs(r))),
        "residuals": r.tolist(), "n": len(F),
    }


def reduced_mass_from_B(B, Eg=EG_300):
    """Invert kane_btbt.kane_B_from_mass (B is proportional to sqrt(m))."""
    return (B / kane_btbt.kane_B_from_mass(Eg, 1.0)) ** 2


def load_calibration_data():
    """(F in V/cm, J in A/cm2, source labels) from every core_calibration curve of this kind."""
    import pandas as pd
    from src.dataio import digitized
    F, J, lab = [], [], []
    for e in digitized.calibration_datasets(digitized.load_metadata(), roles={"core_calibration"}):
        if e["x_quantity"] != "electric_field":
            continue
        d = digitized.load_curve(e)
        F += list(d["x"]); J += list(d["y"]); lab += [e["curve_label"]] * len(d)
    return np.array(F), np.array(J), lab


def main():
    import json
    import os
    F, J, lab = load_calibration_data()
    out = {"data": {"F_Vcm": F.tolist(), "J_Acm2": J.tolist(), "label": lab},
           "assumptions": "abrupt junction, V_bi = Eg (systematic 0.95-1.25 V), P = 2.5, 300 K"}
    for m in MODELS:
        r = fit(F, J, m)
        r["m_r"] = reduced_mass_from_B(r["B"])
        r["systematic_Vbi"] = {str(v): {k: fit(F, J, m, V_t=v + 1.0)[k] for k in ("A", "B", "rms_dec")}
                               for v in (0.95, 1.25)}
        # leave-one-dataset-out: fit on one source, predict the other
        loo = {}
        for src in sorted(set(lab)):
            tr = np.array([l_ != src for l_ in lab])
            rr = fit(F[tr], J[tr], m)
            pred = MODELS[m](F[~tr], rr["A"], rr["B"], EG_300 + 1.0)
            loo[src] = float(np.sqrt(np.mean((np.log10(pred) - np.log10(J[~tr])) ** 2)))
        r["leave_one_source_out_rms_dec"] = loo
        r["default_rms_dec"] = float(np.sqrt(np.mean((np.log10(MODELS[m](F, 4e14, 1.9e7, EG_300 + 1.0))
                                                      - np.log10(J)) ** 2)))
        out[m] = r
        print(f"{m:9s} A={r['A']:.3g} (+-{r['sd_log10A']:.2f} dec) B={r['B']:.4g}+-{r['sd_B']:.2g} "
              f"corr={r['corr']:.3f} rms={r['rms_dec']:.2f} dec (default {r['default_rms_dec']:.2f}) "
              f"m_r={r['m_r']:.3f} LOO={loo}")
    os.makedirs(os.path.join("results", "calibration"), exist_ok=True)
    with open(os.path.join("results", "calibration", "btbt_calibration.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
