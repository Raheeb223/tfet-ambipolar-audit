"""Independent measured checks beyond the Wan 2011 validation (paper Sec. IV-B):

  (a) Biswas 2012 Fig. 3 -- on-state branch of the same CEA-LETI wafer, frozen model (no refit).
      Metric (same form as the Wan validation): mean and rms of log10(I_sim / I_meas) over the
      measured on-state points, VGS >= 1 V (below 1 V the data sit on a ~1e-13 A/um plateau
      set by gate leakage / the instrument, not by tunneling).
  (b) Temperature signature. Measured activation energies E_a of the drain current:
      Der Agopian 2012 Fig. 7 (digitized, 300-420 K, four-point Arrhenius per VGS column) and
      Luong 2017 (text values). Model E_a from the cached audit solutions at 300/358/398 K,
      split into BTBT-dominated and trap-dominated points (one component >= 10x the other at
      D_it = 1e11). Nothing is fitted.
  (c) A generated table of every measured dataset used in the paper.

Run: python -m src.analysis.extval_figs
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.analysis.figures import GEN, _e, _save  # noqa: E402
from src.dataio.digitized import DIGITIZED_DIR, load_metadata  # noqa: E402
from src.dataio.literature_values import load_values  # noqa: E402

K_B = 8.617333e-5
BIS_SIM = os.path.join("results", "validation", "biswas2012_sim.csv")
BIS_COV = os.path.join("results", "validation", "biswas2012_coverage.csv")
RULES = os.path.join("results", "audit_cal", "design_rules_vs_Vg.csv")
D_SIO2 = 1e10
VG_MIN_ON = 1.0
AGO_T = (300, 360, 390, 420)
BIS_VD = {"0p5": 0.5, "1p0": 1.0, "1p5": 1.5}
COL = {0.5: "C0", 1.0: "C3", 1.5: "C2"}


def arrhenius(T, I):
    """Activation energy (eV) of I(T) from a least-squares fit of ln I vs 1/kT."""
    x = 1.0 / (K_B * np.asarray(T, float))
    return float(-np.polyfit(x, np.log(np.asarray(I, float)), 1)[0])


# ------------------------------------------------------------------ Biswas 2012
def biswas_measured(vd_key):
    d = pd.read_csv(os.path.join(DIGITIZED_DIR, f"biswas2012_fig3_VD{vd_key}.csv"))
    return d.VGS_V.to_numpy(), d.ID_Aum.to_numpy()


def biswas_model(sim, Vd, model, D=D_SIO2):
    out = []
    for off in sorted(sim.offset_nm.unique()):
        d = sim[np.isclose(sim.Vd, Vd) & np.isclose(sim.offset_nm, off)].sort_values("Vg")
        I = (d.I_btbt_nl if model == "nonlocal" else d.I_btbt_local) + d.I_tat_bulk + D * d.i_it_per_D
        out.append((d.Vg.to_numpy(), I.to_numpy()))
    return out


def biswas_residuals(sim):
    rows = []
    for key, Vd in BIS_VD.items():
        Vm, Im = biswas_measured(key)
        ok = Vm >= VG_MIN_ON
        for model in ("local", "nonlocal"):
            for (Vs, Is), off in zip(biswas_model(sim, Vd, model), sorted(sim.offset_nm.unique())):
                Ii = 10 ** np.interp(Vm[ok], Vs, np.log10(Is))
                r = np.log10(Ii / Im[ok])
                rows.append(dict(Vd=Vd, model=model, offset_nm=off, n=int(ok.sum()),
                                 bias=float(r.mean()), rms=float(np.sqrt(np.mean(r ** 2))),
                                 bias_at_3V=float(r[-1])))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ temperature
def agopian_ea():
    """E_a per VGS column (valley, four temperatures resolved) and of the minimum current."""
    cur = {T: pd.read_csv(os.path.join(DIGITIZED_DIR, f"agopian2012_fig7_T{T}.csv")) for T in AGO_T}
    V = cur[300].VGS_V.to_numpy()
    rows = [dict(Vg=v, Ea=arrhenius(AGO_T, [cur[T].ID_A.to_numpy()[k] for T in AGO_T]))
            for k, v in enumerate(V)]
    Imin = [cur[T].ID_A.min() for T in AGO_T]
    return pd.DataFrame(rows), arrhenius(AGO_T, Imin), Imin


def model_ea(D=1e11):
    """E_a of each current component and of the total at D_it = D, per technique and bias,
    from the calibrated nonlocal audit at 300/358/398 K. Points are labelled BTBT- or
    trap-dominated when one path carries >= 10x the other at 300 K."""
    d = pd.read_csv(RULES)
    d = d[d.btbt_model == "nonlocal"]
    rows = []
    for (dev, vg), g in d.groupby(["device", "Vg"]):
        g = g.sort_values("T_K")
        if len(g) < 3:
            continue
        trap = g.I_tat_bulk + D * g.a_t
        tot = g.I_btbt + trap
        r300 = g.iloc[0]
        ratio = (r300.I_tat_bulk + D * r300.a_t) / r300.I_btbt if r300.I_btbt > 0 else np.inf
        regime = "trap" if ratio >= 10 else ("btbt" if ratio <= 0.1 else "mixed")
        rows.append(dict(device=dev, Vg=vg, regime=regime, Ea_total=arrhenius(g.T_K, tot),
                         Ea_btbt=arrhenius(g.T_K, g.I_btbt.clip(lower=1e-300)),
                         Ea_it=arrhenius(g.T_K, g.a_t), Ea_bulk=arrhenius(g.T_K, g.I_tat_bulk)))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ figure
def fig_extval(sim, ago, Ea_min, Imin, mod, lv):
    fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.45), gridspec_kw=dict(width_ratios=[1.15, 1.0, 0.95]))
    ax = axs[0]
    for key, Vd in BIS_VD.items():
        c = COL[Vd]
        Vm, Im = biswas_measured(key)
        ax.semilogy(Vm, Im, "o", ms=2.4, color=c, label=f"meas. $V_D$={Vd:g} V")
        nl = biswas_model(sim, Vd, "nonlocal")
        ax.semilogy(nl[len(nl) // 2][0], nl[len(nl) // 2][1], "-", color=c, lw=1.0)
        lo = biswas_model(sim, Vd, "local")
        ax.semilogy(lo[len(lo) // 2][0], lo[len(lo) // 2][1], ":", color=c, lw=1.0)
    ax.plot([], [], "k-", label="nonlocal (cal.) + TAT")
    ax.plot([], [], "k:", label="local (cal.)")
    ax.axvline(VG_MIN_ON, color="0.7", lw=0.6)
    ax.set_ylim(1e-16, 1e-8)
    ax.set_xlabel("$V_{GS}$ (V)")
    ax.set_ylabel("$I_D$ (A/$\\mu$m)")
    ax.set_title("(a) On-state, Biswas Fig. 3", fontsize=8)
    ax.legend(fontsize=5.5, loc="upper left", frameon=False)

    ax = axs[1]
    mk = {300: "o", 360: "D", 390: ">", 420: "*"}
    for T in AGO_T:
        d = pd.read_csv(os.path.join(DIGITIZED_DIR, f"agopian2012_fig7_T{T}.csv"))
        ax.semilogy(d.VGS_V, d.ID_A, mk[T], ms=3, color=plt.cm.inferno(0.15 + 0.6 * (T - 300) / 120),
                    label=f"{T} K")
    ax.set_xlabel("$V_{GS}$ (V)")
    ax.set_ylabel("$I_D$ (A), $|V_{DS}|$ = 0.9 V")
    ax.set_title("(b) Off-state valley, Der Agopian Fig. 7", fontsize=8)
    ax.legend(fontsize=5.5, frameon=False, loc="upper center", ncol=2)
    ax.set_ylim(1e-15, 1e-10)

    ax = axs[2]
    cats = ["BTBT-limited", "trap-limited"]
    bt, tr = mod[mod.regime == "btbt"].Ea_total, mod[mod.regime == "trap"].Ea_total
    for k, s in enumerate((bt, tr)):
        ax.plot([k - 0.12, k - 0.12], [s.min(), s.max()], "-", color="C0", lw=5, alpha=0.6,
                solid_capstyle="butt", label="model" if k == 0 else None)
    ax.plot([1 + 0.12] * len(ago), ago.Ea, "s", ms=3, color="k", mfc="none",
            label="Der Agopian, per $V_{GS}$")
    ax.plot([1 + 0.12], [Ea_min], "s", ms=4, color="k", label="Der Agopian, $I_{min}$")
    ax.plot([1 + 0.25], [lv["luong_Ea_offstate"]["value"]], "^", ms=4, color="C3",
            label="Luong, off state")
    ax.plot([0 + 0.12, 0 + 0.12], [0, lv["luong_Ea_btbt_max"]["value"]], "-", color="C3", lw=2,
            label="Luong, BTBT ($<$0.1 eV)")
    ax.set_xticks([0, 1]); ax.set_xticklabels(cats)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(0, 0.55)
    ax.set_ylabel("Activation energy $E_a$ (eV)")
    ax.set_title("(c) Temperature signature", fontsize=8)
    ax.legend(fontsize=5.0, frameon=False, loc="center left", bbox_to_anchor=(0.0, 0.62))
    fig.tight_layout()
    _save(fig, "fig_extval")


# ------------------------------------------------------------------ data table
def write_data_table():
    meta = load_metadata()
    groups = {}
    for e in meta:
        if e["role"] == "context":
            continue
        k = (e["source"].split(",")[0].split(" et al.")[0].replace("J. Zhao", "Zhao (compiling Fair, Stork)"),
             e["figure"], e["role"])
        n = len(pd.read_csv(os.path.join(DIGITIZED_DIR, e["file"])))
        g = groups.setdefault(k, [0, 0])
        g[0] += 1; g[1] += n
    cite = {"Zhao": "\\cite{fair1976,stork1983,zhao2007}", "J. Wan": "\\cite{wan2011}",
            "A. Biswas": "\\cite{biswas2012}", "P. G. Der Agopian": "\\cite{agopian2012}",
            "L. Ding": "\\cite{ding2014}"}
    what = {("core_calibration",): "fit of $A$, $B$", ("validation",): "frozen-model test",
            ("structure",): "drain profile"}
    lines = []
    for (src, figr, role), (nc, npts) in groups.items():
        c = next(v for kk, v in cite.items() if src.startswith(kk.split()[0]) or kk in src)
        lines.append(f"{c} & {figr} & {nc} & {npts} & {what[(role,)]}\\\\")
    lv = load_values()
    lines.append("\\cite{luong2017} & text & -- & 5 & $E_a$, $E_t$, $\\Dit$\\\\")
    lines.append("\\cite{krishnaraja2019} & text & -- & 3 & UL vs.\\ overlap\\\\")
    assert sum(1 for k in lv if k.startswith("luong")) == 5
    assert sum(1 for k in lv if k.startswith("krishnaraja")) == 3
    with open(os.path.join(GEN, "tab_data.tex"), "w") as f:
        f.write("% generated by src/analysis/extval_figs.py -- do not edit\n" + "\n".join(lines) + "\n")


DIAG = os.path.join("results", "validation", "biswas_diag")
TECHS = ("heterogate", "low_drain_doping", "gd_underlap", "dual_wf_gate")


def angle_numbers():
    """Macros from the on-state diagnosis (src/validation/biswas_diagnostics.py): energy
    window, angle-complete paths on Biswas and Wan, and the effect of angled paths on the
    audit at -0.2 V. Diagnostic only; the production model (rows/columns) is unchanged."""
    N = {}
    f = {k: os.path.join(DIAG, v) for k, v in (("diag", "diagnostics.csv"),
                                                ("bis", "biswas_angles_residuals.csv"),
                                                ("wan", "wan_angles_residuals.csv"),
                                                ("aud", "audit_angles.csv"))}
    if not all(os.path.exists(p) for p in f.values()):
        return N
    d = pd.read_csv(f["diag"])
    N["WinChangePct"] = f"{100 * (d.I_rows_window1p5 / d.I_rows - 1).abs().max():.0f}"
    r = d[np.isclose(d.Vg, 3.0) & np.isclose(d.Vd, 1.0)].iloc[0]
    N["AngGainThree"] = f"{r.I_ang_max / r.I_rows:.0f}"
    b = pd.read_csv(f["bis"])
    ba, br = b[b.paths == "I_ang_max"], b[b.paths == "I_rows"]
    assert (ba.bias < 0).all() and (br.bias < 0).all()
    N["BisAngShortLo"], N["BisAngShortHi"] = f"{-ba.bias.max():.1f}", f"{-ba.bias.min():.1f}"
    N["BisAngThreeAbsMax"] = f"{ba.bias_at_3V.abs().max():.1f}"
    w = pd.read_csv(f["wan"])
    wa, wr = w[w.paths == "I_ang_max"], w[w.paths == "I_rows"]
    assert (wa.bias > 0).all(), "text says angle-complete paths overpredict Wan"
    N["WanAngBiasLo"], N["WanAngBiasHi"] = f"{wa.bias.min():.1f}", f"{wa.bias.max():.1f}"
    N["WanRowsAbsLo"], N["WanRowsAbsHi"] = f"{wr.bias.abs().min():.1f}", f"{wr.bias.abs().max():.1f}"
    a = pd.read_csv(f["aud"])
    a["gain"] = a.I_ang_max / a[["I_x", "I_y"]].max(axis=1)
    g = {(dv, round(vg, 1)): gn for dv, vg, gn in zip(a.device, a.Vg, a.gain)}
    N["AngGainRefOne"] = f"{g[('reference', -1.0)]:.1f}"
    N["AngGainRefTwo"] = f"{g[('reference', -0.2)]:.1f}"
    gt = [g[(t_, -0.2)] for t_ in TECHS]
    N["AngGainTechLo"], N["AngGainTechHi"] = f"{min(gt):.1f}", f"{max(gt):.0f}"
    dS = [abs(np.log10(g[("reference", vg)] / g[(t_, vg)])) for t_ in TECHS for vg in (-1.0, -0.2)]
    N["AngDeltaSMax"] = f"{max(dS):.1f}"
    r02 = pd.read_csv(os.path.join("results", "audit_cal", "design_rules_Vg-0.2.csv"))
    r02 = r02[(r02.btbt_model == "nonlocal") & np.isclose(r02.T_K, 300.0)]
    Dx = [(row.I_tech0 + row.I_btbt * (g[(row.device, -0.2)] - 1)) / row.a_t
          for row in r02.itertuples()]
    N["DxAngLo"], N["DxAngHi"] = _e(min(Dx)), _e(max(Dx))
    words = {0: "none", 1: "one", 2: "two", 3: "three", 4: "all four"}
    N["DxAngBelowEleven"] = words[int(sum(x < 1e11 for x in Dx))]
    return N


def write_outputs():
    lv = load_values()
    N = {}
    ago, Ea_min, Imin = agopian_ea()
    mod = model_ea()
    mod.to_csv(os.path.join("results", "validation", "model_activation_energy.csv"), index=False)
    ago.to_csv(os.path.join("results", "validation", "agopian2012_Ea.csv"), index=False)
    N["AgoEaMin"] = f"{Ea_min:.2f}"
    N["AgoEaLo"] = f"{ago.Ea.min():.2f}"
    N["AgoEaHi"] = f"{ago.Ea.max():.2f}"
    N["AgoIminRatio"] = f"{np.log10(Imin[-1] / Imin[0]):.1f}"
    bt, tr = mod[mod.regime == "btbt"], mod[mod.regime == "trap"]
    N["ModEaBtbtLo"] = f"{bt.Ea_total.min():.2f}"
    N["ModEaBtbtHi"] = f"{bt.Ea_total.max():.2f}"
    N["ModEaTrapLo"] = f"{tr.Ea_total.min():.2f}"
    N["ModEaTrapHi"] = f"{tr.Ea_total.max():.2f}"
    # what the measured E_a implies for the 300 -> 398 K increase of a trap current
    dT = 1 / (K_B * 300.0) - 1 / (K_B * 398.15)
    N["TrapGainMeas"] = f"{np.exp(Ea_min * dT):.0f}"
    N["TrapGainModel"] = f"{np.exp(tr.Ea_total.max() * dT):.1f}"
    N["LuongEa"] = f"{lv['luong_Ea_offstate']['value']:.2f}"
    N["LuongTrapLo"], N["LuongTrapHi"] = (f"{v:.2f}" for v in lv["luong_trap_level"]["value"])
    N["LuongDitMos"] = _e(lv["luong_Dit_moscap"]["value"])
    N["LuongNit"] = _e(lv["luong_Nit_tfet"]["value"])
    N["KrishIminDec"] = f"{np.log10(lv['krishnaraja_Imin_overlap']['value'] / lv['krishnaraja_Imin_underlap']['value']):.1f}"
    N["KrishIonDec"] = f"{np.log10(lv['krishnaraja_Ion_ratio']['value']):.1f}"
    if os.path.exists(BIS_SIM):
        sim = pd.read_csv(BIS_SIM)
        res = biswas_residuals(sim)
        res.to_csv(os.path.join("results", "validation", "biswas2012_residuals.csv"), index=False)
        print(res.round(2).to_string(index=False))
        # unsigned magnitudes; the text states the direction, which is asserted here
        rl, rn = res[res.model == "local"], res[res.model == "nonlocal"]
        assert (rn.bias < 0).all(), "text says the nonlocal model underestimates the on-state"
        N["BisNlShortLo"], N["BisNlShortHi"] = f"{-rn.bias.max():.1f}", f"{-rn.bias.min():.1f}"
        N["BisLocAbsMax"] = f"{rl.bias.abs().max():.1f}"
        N["BisLocSign"] = "over" if (rl.bias > 0).all() else ("under" if (rl.bias < 0).all() else "mixed")
        N["BisRmsNlMax"] = f"{rn.rms.max():.1f}"
        a3 = rn.groupby("Vd").bias_at_3V.mean()
        N["BisThreeVdHi"], N["BisThreeVdLo"] = f"{-a3.loc[1.5]:.1f}", f"{-a3.loc[0.5]:.1f}"
        N["BisLocThreeMax"] = f"{rl.bias_at_3V.abs().max():.1f}"
        N["BisRmsLocMax"] = f"{rl.rms.max():.1f}"
        fig_extval(sim, ago, Ea_min, Imin, mod, lv)
        # Sensitivity of the off-state headline: if the true BTBT were higher than the nonlocal
        # model by the on-state shortfall (worst offset/VD), every technique's D_x (= I0 / a,
        # BTBT-dominated I0) would rise by the same factor.
        r02 = pd.read_csv(os.path.join("results", "audit_cal", "design_rules_Vg-0.2.csv"))
        r02 = r02[(r02.btbt_model == "nonlocal") & np.isclose(r02.T_K, 300.0)]
        words = {0: "none", 1: "one", 2: "two", 3: "three", 4: "all"}
        for tag, short in (("", -rn[np.isclose(rn.Vd, 1.0)].bias.min()), ("Worst", -rn.bias.min())):
            Dx_up = (r02.I_tech0 + r02.I_btbt * (10 ** short - 1)) / r02.a_t
            N[f"BisShort{tag}"] = f"{short:.1f}"
            N[f"DxShift{tag}Lo"], N[f"DxShift{tag}Hi"] = _e(Dx_up.min()), _e(Dx_up.max())
            N[f"DxShift{tag}BelowTwelve"] = words[int((Dx_up < 1e12).sum())]
            N[f"DxShift{tag}BelowEleven"] = words[int((Dx_up < 1e11).sum())]
    if os.path.exists(BIS_COV):
        cov = pd.read_csv(BIS_COV)
        c = cov[(cov.model == "nonlocal") & (cov.Vg >= 2.0)]
        N["BisFnlLowVg"] = f"{cov[(cov.model == 'nonlocal') & (cov.Vg < 2.0)].F_median_MVcm.min():.2f}"
        N["BisFnlLo"] = f"{c.F_median_MVcm.min():.2f}"
        N["BisFnlHi"] = f"{c.F_median_MVcm.max():.2f}"
        N["BisBelowMin"] = f"{100 * c.frac_below.min():.0f}"
        cl = cov[cov.model == "local"]
        N["BisFlocLo"] = f"{cl.F_median_MVcm.min():.2f}"
        N["BisFlocHi"] = f"{cl.F_median_MVcm.max():.2f}"
    N.update(angle_numbers())
    write_data_table()
    print(ago.round(3).to_string(index=False))
    print(mod.groupby("regime")[["Ea_total", "Ea_btbt", "Ea_it", "Ea_bulk"]].agg(["min", "max"]).round(3))
    with open(os.path.join(GEN, "numbers_extval.tex"), "w") as f:
        f.write("% generated by src/analysis/extval_figs.py -- do not edit\n")
        for k, v in N.items():
            f.write(f"\\newcommand{{\\n{k}}}{{{v}}}\n")
    return N


if __name__ == "__main__":
    for k, v in write_outputs().items():
        print(k, v)
