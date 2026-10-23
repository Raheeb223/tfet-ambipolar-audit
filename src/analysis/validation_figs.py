"""Calibration and validation figures, tables and numbers (paper Sec. III-E/IV-A).

Validation metrics were fixed before the comparison was made:
  * rms of log10(I_sim / I_meas) over every measured point above the instrument floor,
    per BTBT model and per a-priori implant offset (0, 10, 20 nm);
  * nominal D_it: 1e10 (thermal SiO2) and 1e11 cm^-2 eV^-1 (HfO2) [typical values of the
    reviews cited in the paper], with bands 1e10-1e11 and 1e11-1e12;
  * technique suppression vs L_IN read at VGS = -5 V (SiO2) and -3 V (HfO2).
Run: python -m src.analysis.validation_figs
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.analysis.figures import FIG, GEN, _e, _save  # noqa: E402
from src.calibration import btbt_calibration as bc  # noqa: E402

CAL = os.path.join("results", "calibration", "btbt_calibration.json")
SIM = os.path.join("results", "validation", "wan2011_sim.csv")
DIG = os.path.join("data", "digitized")
D_NOM = {"sio2": 1e10, "hfo2": 1e11}
D_BAND = {"sio2": (1e10, 1e11), "hfo2": (1e11, 1e12)}
FIGNO = {"sio2": "fig6b", "hfo2": "fig8a"}
VS = {"sio2": -5.0, "hfo2": -3.0}
FLOOR = 10 ** -12.7


def measured(device, L):
    p = os.path.join(DIG, f"wan2011_{FIGNO[device]}_LIN{L}.csv")
    d = pd.read_csv(p)
    return d.VGS_V.to_numpy(), d.ID_Aum.to_numpy()


def sim_curve(sim, device, L, off, model, D):
    d = sim[(sim.device == device) & (sim.L_IN_nm == L) & np.isclose(sim.offset_nm, off)].sort_values("Vg")
    I = (d.I_btbt_local if model == "local" else d.I_btbt_nl) + d.I_tat_bulk + D * d.i_it_per_D
    return d.Vg.to_numpy(), I.to_numpy()


def residuals(sim):
    rows = []
    for dev in ("sio2", "hfo2"):
        for L in sorted(sim[sim.device == dev].L_IN_nm.unique()):
            Vm, Im = measured(dev, L)
            ok = Im > FLOOR
            if ok.sum() < 3:
                continue
            for off in sorted(sim.offset_nm.unique()):
                for model in ("local", "nonlocal"):
                    for D in (0.0, D_NOM[dev], D_BAND[dev][1]):
                        Vs, Is = sim_curve(sim, dev, L, off, model, D)
                        Ii = 10 ** np.interp(Vm[ok], Vs, np.log10(Is))
                        r = np.log10(Ii / Im[ok])
                        rows.append(dict(device=dev, L_IN_nm=L, offset_nm=off, model=model, D_it=D,
                                         n=int(ok.sum()), rms=float(np.sqrt(np.mean(r ** 2))),
                                         bias=float(np.mean(r))))
    return pd.DataFrame(rows)


def suppression_vs_lin(sim):
    """Measured and simulated S(L_IN) = log10(I(L_IN,min) / I(L_IN)) at VS[device]."""
    rows = []
    for dev in ("sio2", "hfo2"):
        Ls = sorted(sim[sim.device == dev].L_IN_nm.unique())
        def meas_at(L):
            V, I = measured(dev, L)
            if I.max() <= FLOOR * 1.0001 or not (V.min() <= VS[dev] <= V.max()):
                return np.nan
            v = 10 ** np.interp(VS[dev], V, np.log10(I))
            return v if v > FLOOR else np.nan
        m0 = meas_at(Ls[0])
        for L in Ls:
            mL = meas_at(L)
            row = dict(device=dev, L_IN_nm=L, Vg=VS[dev],
                       S_meas=np.log10(m0 / mL) if np.isfinite(mL) else np.nan,
                       S_meas_lower=np.log10(m0 / FLOOR))
            for off in sorted(sim.offset_nm.unique()):
                for model in ("local", "nonlocal"):
                    D = D_NOM[dev]
                    V0, I0 = sim_curve(sim, dev, Ls[0], off, model, D)
                    V1, I1 = sim_curve(sim, dev, L, off, model, D)
                    row[f"S_{model}_off{off:.0f}"] = float(np.log10(np.interp(VS[dev], V0, I0) /
                                                                     np.interp(VS[dev], V1, I1)))
            rows.append(row)
    return pd.DataFrame(rows)


def fig_calibration(cal):
    F = np.array(cal["data"]["F_Vcm"]); J = np.array(cal["data"]["J_Acm2"]); lab = cal["data"]["label"]
    Fg = np.linspace(0.8e6, 2.5e6, 120)
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for src, mk in (("Fair", "^"), ("Stork", "v")):
        m = np.array([src in s for s in lab])
        ax.semilogy(F[m] / 1e6, J[m], mk, color="k", mfc="none" if src == "Stork" else "k", ms=5,
                    label=("Fair \\& Wivell 1976" if src == "Fair" else "Stork \\& Isaac 1983").replace("\\", ""))
    Vt = bc.EG_300 + 1.0
    for model, col in (("local", "C3"), ("nonlocal", "C0")):
        r = cal[model]
        ax.semilogy(Fg / 1e6, bc.MODELS[model](Fg, r["A"], r["B"], Vt), "-", color=col,
                    lw=3.0 if model == "local" else 1.0, alpha=0.45 if model == "local" else 1.0,
                    label=f"{model}, calibrated" + (" (coincide)" if model == "nonlocal" else ""))
        ax.semilogy(Fg / 1e6, bc.MODELS[model](Fg, 4e14, 1.9e7, Vt), "--", color=col, lw=0.9,
                    label=f"{model}, default")
    ax.set_xlabel("Peak junction field (MV/cm)")
    ax.set_ylabel("J at $-1$ V (A/cm$^2$)")
    ax.set_ylim(1e-8, 1e4)
    ax.legend(loc="lower right", frameon=False)
    _save(fig, "fig_calibration")


def fig_validation(sim):
    """Measured vs simulated ambipolar branches. Line: implant offset at the centre of the
    a-priori range (10 nm); band: envelope over offsets 0-20 nm (both at the nominal D_it).
    Measured points within 0.3 decade of the instrument floor are drawn open."""
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.7))
    cols = {10: "C0", 25: "C3", 40: "C2", 50: "C9"}
    offs = sorted(sim.offset_nm.unique())
    mid = offs[len(offs) // 2]
    for ax, dev in zip(axs, ("sio2", "hfo2")):
        for L in sorted(sim[sim.device == dev].L_IN_nm.unique()):
            c = cols.get(L, "k")
            Vm, Im = measured(dev, L)
            low = Im < FLOOR * 10 ** 0.3
            ax.semilogy(Vm[~low], Im[~low], "o", ms=2.2, color=c, label=f"meas. $L_{{IN}}$={L} nm")
            ax.semilogy(Vm[low], Im[low], "o", ms=2.2, color=c, mfc="none", mew=0.5)
            curves = [sim_curve(sim, dev, L, o, "nonlocal", D_NOM[dev]) for o in offs]
            V = curves[0][0]
            Is = np.array([np.interp(V, v, i) for v, i in curves])
            ax.fill_between(V, Is.min(axis=0), Is.max(axis=0), color=c, alpha=0.15, lw=0)
            V, I = sim_curve(sim, dev, L, mid, "nonlocal", D_NOM[dev])
            ax.semilogy(V, I, "-", color=c, lw=1.0)
            V, Il = sim_curve(sim, dev, L, mid, "local", D_NOM[dev])
            ax.semilogy(V, Il, ":", color=c, lw=1.0)
        ax.axhline(FLOOR, color="0.6", lw=0.6)
        ax.set_ylim(1e-15, 1e-5)
        ax.set_xlabel("$V_{GS}$ (V)")
        ax.set_title(("(a) 6 nm SiO$_2$ (Wan Fig. 6b)" if dev == "sio2" else "(b) HfO$_2$, EOT 2.2 nm (Wan Fig. 8a)"))
    axs[0].set_ylabel("$I_D$ (A/$\\mu$m), $V_{DS}$ = 1 V")
    axs[0].plot([], [], "k-", label="nonlocal (cal.) + TAT, offset 10 nm")
    axs[0].fill_between([], [], [], color="k", alpha=0.15, lw=0, label="offset 0-20 nm")
    axs[0].plot([], [], "k:", label="local Kane (cal.)")
    axs[0].plot([], [], "o", ms=2.2, color="k", mfc="none", mew=0.5, label="at floor")
    axs[0].legend(loc="upper center", bbox_to_anchor=(1.1, -0.2), ncol=4, frameon=False)
    _save(fig, "fig_validation")


def write_outputs():
    cal = json.load(open(CAL))
    fig_calibration(cal)
    N = {}
    for m, k in (("local", "Loc"), ("nonlocal", "Nl")):
        r = cal[m]
        N[f"CalA{k}"] = _e(r["A"]); N[f"CalB{k}"] = _e(r["B"])
        N[f"CalSdA{k}"] = f"{r['sd_log10A']:.2f}"; N[f"CalSdB{k}"] = f"{r['sd_B'] / 1e6:.1f}"
        N[f"CalRms{k}"] = f"{r['rms_dec']:.2f}"; N[f"CalDefRms{k}"] = f"{r['default_rms_dec']:.2f}"
        N[f"CalMr{k}"] = f"{r['m_r']:.2f}"; N[f"CalCorr{k}"] = f"{r['corr']:.2f}"
        loo = list(r["leave_one_source_out_rms_dec"].values())
        N[f"CalLooMax{k}"] = f"{max(loo):.2f}"
    N["CalN"] = str(len(cal["data"]["F_Vcm"]))
    N["CalFmin"] = f"{min(cal['data']['F_Vcm']) / 1e6:.2f}"
    N["CalFmax"] = f"{max(cal['data']['F_Vcm']) / 1e6:.2f}"
    if os.path.exists(SIM):
        sim = pd.read_csv(SIM)
        res = residuals(sim)
        res.to_csv(os.path.join("results", "validation", "residuals.csv"), index=False)
        sup = suppression_vs_lin(sim)
        sup.to_csv(os.path.join("results", "validation", "suppression_vs_LIN.csv"), index=False)
        fig_validation(sim)
        nom = res[[D_NOM[d] == x for d, x in zip(res.device, res.D_it)]]
        pooled = nom.groupby(["model", "offset_nm"]).apply(
            lambda x: float(np.sqrt((x.rms ** 2 * x.n).sum() / x.n.sum())))
        for m, k in (("local", "Loc"), ("nonlocal", "Nl")):
            N[f"ValRms{k}Min"] = f"{pooled[m].min():.1f}"
            N[f"ValRms{k}Max"] = f"{pooled[m].max():.1f}"
        q = lambda dev, L, m, D, col="rms": res[(res.device == dev) & (res.L_IN_nm == L) & (res.model == m)  # noqa: E731
                                              & np.isclose(res.D_it, D)][col]
        N["ValTenNlMin"] = f"{min(q('sio2', 10, 'nonlocal', D_NOM['sio2']).min(), q('hfo2', 10, 'nonlocal', D_NOM['hfo2']).min()):.1f}"
        N["ValTenNlMax"] = f"{max(q('sio2', 10, 'nonlocal', D_NOM['sio2']).max(), q('hfo2', 10, 'nonlocal', D_NOM['hfo2']).max()):.1f}"
        N["ValTenLocMin"] = f"{min(q('sio2', 10, 'local', D_NOM['sio2']).min(), q('hfo2', 10, 'local', D_NOM['hfo2']).min()):.1f}"
        N["ValTenLocMax"] = f"{max(q('sio2', 10, 'local', D_NOM['sio2']).max(), q('hfo2', 10, 'local', D_NOM['hfo2']).max()):.1f}"
        # HfO2 strongly leaking devices: BTBT alone vs with the trap path (offset 0)
        b0 = [q("hfo2", L, "nonlocal", 0.0, "bias")[res.offset_nm == 0].iloc[0] for L in (10, 25)]
        b1 = [q("hfo2", L, "nonlocal", D_NOM["hfo2"], "bias")[res.offset_nm == 0].iloc[0] for L in (10, 25)]
        N["ValHfBiasNoTrap"] = f"{np.mean(b0):.1f}"
        N["ValHfBiasTrap"] = f"{round(np.mean(b1), 1) + 0.0:.1f}"
        s25 = sup[(sup.device == "sio2") & (sup.L_IN_nm == 25)].iloc[0]
        s40 = sup[(sup.device == "sio2") & (sup.L_IN_nm == 40)].iloc[0]
        cols = [c for c in sup.columns if c.startswith("S_nonlocal")]
        N["ValSmeasTwentyFive"] = f"{s25.S_meas:.1f}"
        N["ValSmeasForty"] = f"{s40.S_meas:.1f}"
        N["ValSsimFortyMin"] = f"{s40[cols].min():.1f}"
        N["ValSsimFortyMax"] = f"{s40[cols].max():.1f}"
        N["ValSmeasLower"] = f"{s25.S_meas_lower:.1f}"
        print(res.round(2).to_string(index=False))
        print(sup.round(2).to_string(index=False))
    with open(os.path.join(GEN, "numbers_validation.tex"), "w") as f:
        f.write("% generated by src/analysis/validation_figs.py -- do not edit\n")
        for k, v in N.items():
            f.write(f"\\newcommand{{\\n{k}}}{{{v}}}\n")
    return N


if __name__ == "__main__":
    write_outputs()
