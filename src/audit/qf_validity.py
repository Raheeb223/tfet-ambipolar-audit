"""Where is the fixed-quasi-Fermi-level approximation valid?

The audit ties holes to the source and electrons to the drain. Generated holes must flow from
the drain-side gate edge through the channel to the source, and generated electrons through the
drain extension to the drain contact. Their quasi-Fermi levels then drop by

    dPhi_p = integral_0^L_gate  I / (q mu_p P_s(x)) dx,   P_s = integral p dy  (sheet, cm^-2)
    dPhi_n = integral_L_gate^x_end  I / (q mu_n N_s(x)) dx

with I the current of the simulated half per cm of width. While dPhi << kT/q the fixed levels
are exact for the generation (which depends on them through the occupation only); above ~kT/q
the computed current is an overestimate. Mobilities are low-end ASSUMPTIONS (surface and
heavily doped Si): mu_p = 50, mu_n = 70 cm^2/Vs, so dPhi is an upper estimate.

The integrals run over the QUASI-NEUTRAL part of each path only (ASSUMPTION): where the
carrier is the majority (P_s > N_s for holes, N_s > P_s for electrons) and its sheet density
is at least QN_FRAC of its peak on that path. In the depleted drain junction the generated
carriers are swept out by the field; their drift is not limited by the sheet conductance,
and including it would make the integral diverge as P_s -> 0.

Run: python -m src.audit.qf_validity  -> results/model_form/qf_validity.csv
"""

import os

import numpy as np
import pandas as pd

from src.audit import run_audit as ra
from src.config import load_yaml
from src.constants import K_B_EV, Q
from src.devices import tfet_2d as t

MU_P, MU_N = 50.0, 70.0
QN_FRAC = 0.1
OUT = os.path.join("results", "model_form")


def qf_drops(sol, cfg, I_um):
    """(dPhi_p, dPhi_n) in V for a total current I_um (A/um, both halves)."""
    halves = 1 if t.single_gate(cfg) else 2
    I_cm = I_um * 1e4 / halves
    w = t._row_widths(sol.y)
    Ps = (sol.p * w[:, None]).sum(axis=0)
    Ns = (sol.n * w[:, None]).sum(axis=0)
    Lg = cfg["geometry"]["L_gate_cm"]
    tol = 1e-12
    ch, dr = (sol.x >= -tol) & (sol.x <= Lg + tol), sol.x >= Lg - tol
    ch &= (Ps > Ns) & (Ps >= QN_FRAC * Ps[ch].max())
    dr &= (Ns > Ps) & (Ns >= QN_FRAC * Ns[dr].max())
    dxp = np.trapezoid(1.0 / Ps[ch], sol.x[ch])
    dxn = np.trapezoid(1.0 / Ns[dr], sol.x[dr])
    return I_cm * dxp / (Q * MU_P), I_cm * dxn / (Q * MU_N)


def corrected(df, iters=60):
    """Estimated current with the quasi-Fermi drop counted as lost gate-drain bias (an upper
    estimate of the error): I_c = I(Vg + dPhi(I_c)), dPhi proportional to I at fixed sheet
    densities, I(Vg) interpolated in log along the device's own transfer curve. Adds I_corr and
    dlogI = log10(I / I_corr)."""
    df = df.copy()
    df["I_corr"] = df.I_total
    for (d, T, D), g in df.groupby(["device", "T_K", "D_it"]):
        g = g.sort_values("Vg")
        V, logI = g.Vg.to_numpy(), np.log10(g.I_total.to_numpy())
        for idx, r in g.iterrows():
            k = max(r.dPhi_p, r.dPhi_n) / r.I_total          # V per (A/um)
            Ic = r.I_total
            for _ in range(iters):
                Ic_new = 10 ** np.interp(min(r.Vg + k * Ic, V.max()), V, logI)
                Ic = 0.5 * (Ic + Ic_new)
            df.loc[idx, "I_corr"] = Ic
    df["dlogI"] = np.log10(df.I_total / df.I_corr)
    return df


def main(audit_file="configs/audit_calibrated.yaml"):
    audit = load_yaml(audit_file)
    ra.CACHE = audit["cache_dir"]
    tab = pd.read_csv(os.path.join(audit["out_dir"], "table.csv"))
    tab = tab[(tab.kind == "realistic") & (tab.btbt_model == "nonlocal") & np.isclose(tab.tau, 1e-7)
              & (tab.dit_profile == "uniform")]
    rows = []
    for d, T in [(d, T) for d in audit["devices"] for T in audit["temperatures_K"]]:
        cfg, sols = ra.load_solutions(d, T)
        for s in sols:
            for D in (0.0, 1e12, 5e12):
                r = tab[(tab.device == d) & np.isclose(tab.T_K, T) & np.isclose(tab.Vg, s.Vg) & np.isclose(tab.D_it, D)]
                if r.empty:
                    continue
                I = float(r.I_total.iloc[0])
                dp, dn = qf_drops(s, cfg, I)
                rows.append(dict(device=d, T_K=T, Vg=s.Vg, D_it=D, I_total=I, dPhi_p=dp, dPhi_n=dn,
                                 dPhi_over_kT=max(dp, dn) / (K_B_EV * T)))
    out = pd.DataFrame(rows)
    out = corrected(out)
    os.makedirs(OUT, exist_ok=True)
    out.to_csv(os.path.join(OUT, "qf_validity.csv"), index=False)
    print(out.groupby(["device", "Vg"]).dPhi_over_kT.max().unstack().round(3).to_string())
    print(out.groupby(["device", "Vg"]).dlogI.max().unstack().round(3).to_string())
    return out


if __name__ == "__main__":
    main()
