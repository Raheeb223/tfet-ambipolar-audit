"""Paper figures, tables and in-text numbers from results/audit*/table.csv.

Every number quoted in paper/main.tex comes from this script (paper/generated/numbers.tex,
paper/generated/tab_*.tex), so the manuscript cannot drift from the data.

Run:  python -m src.analysis.figures      -> paper/figs/*.pdf|png, paper/generated/*.tex
Inputs: results/audit_cal (calibrated BTBT; baseline = literature defaults). Set
AUDIT_DIR=results/audit for the uncalibrated run.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.audit import metrics  # noqa: E402

AUD = os.environ.get("AUDIT_DIR", os.path.join("results", "audit_cal"))
FIG = os.path.join("paper", "figs")
GEN = os.path.join("paper", "generated")
LABEL = {"reference": "Reference", "heterogate": "Hetero-gate dielectric",
         "low_drain_doping": "Low drain doping", "gd_underlap": "G-D underlap",
         "dual_wf_gate": "Drain-side WF"}
SHORT = {"heterogate": "HG", "low_drain_doping": "LDD", "gd_underlap": "UL", "dual_wf_gate": "DWF"}
TECH = ["heterogate", "low_drain_doping", "gd_underlap", "dual_wf_gate"]
COL = {"reference": "k", "heterogate": "C0", "low_drain_doping": "C2", "gd_underlap": "C1",
       "dual_wf_gate": "C4"}
TAU0, VAMB, T0, TH, DREF = 1e-7, -1.0, 300.0, 398.15, 1e12

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "legend.fontsize": 6.5,
                     "figure.dpi": 100, "savefig.dpi": 300})


def _save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)


def _sel(df, **kw):
    if "dit_profile" in df.columns and "dit_profile" not in kw:
        kw["dit_profile"] = "uniform"
    m = np.ones(len(df), bool)
    for k, v in kw.items():
        m &= np.isclose(df[k], v) if isinstance(v, float) else (df[k] == v).to_numpy()
    return df[m]


def _one(df, **kw):
    d = _sel(df, **kw)
    if len(d) != 1:
        raise LookupError(f"{len(d)} rows for {kw}")
    return d.iloc[0]


# ---- linear-in-D_it decomposition -------------------------------------------------------

def trap_params(df, dev, T, model, Vg=VAMB, tau=TAU0):
    """I0 (no interface traps) and slope a = dI/dD_it, checked for linearity at 5e12."""
    r0 = _one(df, kind="realistic", device=dev, T_K=T, btbt_model=model, D_it=0.0, tau=tau, Vg=Vg)
    r1 = _one(df, kind="realistic", device=dev, T_K=T, btbt_model=model, D_it=DREF, tau=tau, Vg=Vg)
    a = r1.I_it / DREF
    D5 = _sel(df, kind="realistic", device=dev, T_K=T, btbt_model=model, D_it=5e12, tau=tau, Vg=Vg)
    if len(D5):
        pred = r0.I_total + a * 5e12
        assert abs(D5.iloc[0].I_total / pred - 1) < 1e-6, "I_it not linear in D_it"
    return float(r0.I_total), float(a), float(r0.I_btbt), float(r0.I_tat_bulk)


def design_rules(df, Vg=VAMB):
    rows = []
    for dev in TECH:
        claim = metrics.suppression_decades(
            _one(df, kind="baseline", device="reference", Vg=Vg).I_total,
            _one(df, kind="baseline", device=dev, Vg=Vg).I_total)
        for model in ("local", "nonlocal"):
            for T in (T0, 358.15, TH):
                I_r0, a_r, _, _ = trap_params(df, "reference", T, model, Vg)
                I_t0, a_t, Ib, It = trap_params(df, dev, T, model, Vg)
                S = lambda D: float(metrics.suppression_vs_dit(D, I_r0, a_r, I_t0, a_t))  # noqa: E731
                rows.append(dict(
                    device=dev, btbt_model=model, T_K=T, Vg=Vg, S_claim=float(claim),
                    S_0=S(0.0), S_1e11=S(1e11), S_1e12=S(1e12), S_5e12=S(5e12),
                    S_inf=metrics.trap_limited_suppression(a_r, a_t),
                    D_cross=metrics.dit_crossover(I_t0, a_t),
                    D_half=metrics.dit_for_retention(I_r0, a_r, I_t0, a_t, 0.5),
                    I_tech0=I_t0, I_btbt=Ib, I_tat_bulk=It, a_t=a_t, a_ref=a_r))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(AUD, f"design_rules_Vg{Vg:+.1f}.csv"), index=False)
    return out


# ---- figures ---------------------------------------------------------------------------

def fig_structure(cfg_path="configs/tfet_2d.yaml"):
    """Half-structure schematic with the four technique regions (to scale in x)."""
    from src.config import load_yaml
    c = load_yaml(cfg_path)
    g = c["geometry"]
    Ls, Lg, Ld = (g[k] * 1e7 for k in ("L_source_cm", "L_gate_cm", "L_drain_cm"))
    ts, tox = g["t_si_cm"] * 1e7 / 2, g["t_ox_cm"] * 1e7
    fig, ax = plt.subplots(figsize=(3.4, 1.55))
    box = lambda x0, y0, w, h, fc, **k: ax.add_patch(  # noqa: E731
        mpatches.Rectangle((x0, y0), w, h, fc=fc, ec="k", lw=0.5, **k))
    box(-Ls, 0, Ls, ts, "#f4b6b6"); box(0, 0, Lg, ts, "#eeeeee"); box(Lg, 0, Ld, ts, "#b6c8f4")
    box(0, -tox, Lg, tox, "#fde9a9"); box(Lg - 20, -tox, 20, tox, "#fff7d9", hatch="////")
    box(0, -tox - 1.6, Lg, 1.6, "#888888"); box(Lg - 20, -tox - 1.6, 20, 1.6, "#5a5a8a")
    ax.annotate("", xy=(Lg - 10, -tox - 2.3), xytext=(Lg, -tox - 2.3),
                arrowprops=dict(arrowstyle="<->", lw=0.6))
    for x, y, s in [(-Ls / 2, ts / 2, "p$^+$ source"), (Lg / 2, ts / 2, "p$^-$ channel"),
                    (Lg + Ld / 2, ts / 2, "n$^+$ drain\n(LDD: $5{\\times}10^{18}$)"),
                    (Lg / 2 - 10, -tox / 2, "HfO$_2$"), (Lg / 2 - 10, -tox - 0.8, "gate")]:
        ax.text(x, y, s, ha="center", va="center", fontsize=6,
                color="w" if s == "gate" else "k")
    # technique labels outside the stack, with leaders, so they do not overlap
    ax.annotate("HG: SiO$_2$ (hatched)\nDWF: 4.0 eV (dark)", xy=(Lg - 12, -tox - 1.6),
                xytext=(Lg - 52, -tox - 4.6), fontsize=5.5, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", lw=0.4))
    ax.text(Lg - 5, -tox - 3.1, "UL", ha="center", va="bottom", fontsize=5.5)
    ax.axhline(ts, color="k", lw=0.4, ls=":")
    ax.text(Lg + Ld - 1, ts + 0.4, "symmetry plane", ha="right", va="top", fontsize=5)
    ax.set_xlim(-Ls - 1, Lg + Ld + 1); ax.set_ylim(-tox - 6.2, ts + 2.6)
    ax.invert_yaxis(); ax.set_xlabel("x (nm)"); ax.set_ylabel("y (nm)")
    ax.set_aspect(2.2)
    _save(fig, "fig_structure")


def fig_transfer(df):
    """Ambipolar branch of all five devices under three levels of physics (300 K)."""
    fig, axs = plt.subplots(1, 3, figsize=(7.1, 2.3), sharey=True)
    panels = [("(a) Lit. practice: local, default", dict(kind="baseline")),
              ("(b) Calibrated nonlocal + bulk TAT", dict(kind="realistic", btbt_model="nonlocal",
                                                   D_it=0.0, tau=TAU0, T_K=T0)),
              ("(c) (b) + interface TAT, $D_{it}=10^{12}$",
               dict(kind="realistic", btbt_model="nonlocal", D_it=DREF, tau=TAU0, T_K=T0))]
    for ax, (title, kw) in zip(axs, panels):
        for dev in ["reference"] + TECH:
            d = _sel(df, device=dev, **kw).sort_values("Vg")
            ax.semilogy(d.Vg, d.I_total, "-o", ms=2, lw=1, color=COL[dev], label=LABEL[dev])
        ax.set_title(title)
        ax.set_xlabel("$V_{GS}$ (V)")
        ax.axvline(VAMB, color="k", lw=0.4, ls=":")
        ax.grid(True, which="major", alpha=0.3)
    axs[0].set_ylabel("$I_D$ (A/µm), $V_{DS}$ = 1 V")
    axs[0].set_ylim(1e-14, 1e-3)
    axs[0].legend(loc="lower left")
    _save(fig, "fig_transfer")


def fig_components(df):
    """Current components vs V_GS: where the drain-side current comes from."""
    fig, axs = plt.subplots(1, 2, figsize=(7.1, 2.4), sharey=True)
    for ax, dev in zip(axs, ["reference", "low_drain_doping"]):
        for T, ls in ((T0, "-"), (TH, "--")):
            d = _sel(df, kind="realistic", device=dev, btbt_model="nonlocal", D_it=DREF,
                     tau=TAU0, T_K=T).sort_values("Vg")
            dl = _sel(df, kind="realistic", device=dev, btbt_model="local", D_it=0.0,
                      tau=TAU0, T_K=T).sort_values("Vg")
            tl = f"{T:.0f} K"
            ax.semilogy(d.Vg, d.I_btbt, ls, color="C0", label=f"nonlocal BTBT, {tl}")
            ax.semilogy(d.Vg, d.I_it, ls, color="C3", label=f"interface TAT, {tl}")
            ax.semilogy(d.Vg, d.I_tat_bulk, ls, color="C2", label=f"bulk TAT, {tl}")
            if T == T0:
                ax.semilogy(dl.Vg, dl.I_btbt, ":", color="0.4", label="local BTBT (cal.), 300 K")
        ax.set_title(f"{LABEL[dev]}: $D_{{it}}=10^{{12}}$, $\\tau=10^{{-7}}$ s")
        ax.set_xlabel("$V_{GS}$ (V)")
        ax.grid(True, which="major", alpha=0.3)
    axs[0].set_ylabel("Current component (A/µm)")
    axs[0].set_ylim(1e-15, 1e-5)
    h, lab = axs[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.3), frameon=False)
    _save(fig, "fig_components")


def fig_suppression(rules, rules02):
    """Claimed vs surviving suppression for each technique (decades), at two biases."""
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.4), sharey=True)
    bars = [("Literature practice (local, default param.)", lambda d: d[d.T_K == T0].S_claim, "0.6"),
            ("Calibrated nonlocal, no $D_{it}$, 300 K", lambda d: d[d.T_K == T0].S_0, "C0"),
            ("$D_{it}=10^{12}$, 300 K", lambda d: d[d.T_K == T0].S_1e12, "C1"),
            (r"$D_{it}=5{\times}10^{12}$, 398 K", lambda d: d[np.isclose(d.T_K, TH)].S_5e12, "C3"),
            (r"Trap-limited $S_\infty$", lambda d: d[d.T_K == T0].S_inf, "k")]
    w = 0.16
    for ax, r, title in ((axs[0], rules, "(a) $V_{GS}=-1$ V"), (axs[1], rules02, "(b) $V_{GS}=-0.2$ V")):
        nl = r[r.btbt_model == "nonlocal"]
        for i, (lab, f, c) in enumerate(bars):
            y = [float(f(nl[nl.device == dev]).iloc[0]) for dev in TECH]
            ax.bar(np.arange(4) + (i - 2) * w, y, w, color=c, label=lab)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xticks(range(4), [SHORT[d] for d in TECH])
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
    axs[0].set_ylabel("Suppression $S$ (dec)")
    axs[0].legend(loc="upper center", bbox_to_anchor=(1.05, -0.28), ncol=5, frameon=False, fontsize=6)
    _save(fig, "fig_suppression")


def fig_dit(df, rules_by_vg):
    """S vs D_it (continuous, exact by linearity): rows V_GS = -1 V and -0.2 V, columns
    calibrated local / nonlocal BTBT; solid 300 K, dashed 398 K; open circles D_1/2."""
    D = np.logspace(9, 13.5, 200)
    vgs = (VAMB, -0.2)
    fig, axs = plt.subplots(2, 2, figsize=(7.1, 4.3), sharex=True)
    for i, Vg in enumerate(vgs):
        rules = rules_by_vg[Vg]
        for j, model in enumerate(("local", "nonlocal")):
            ax = axs[i, j]
            for dev in TECH:
                for T, ls in ((T0, "-"), (TH, "--")):
                    I_r0, a_r, _, _ = trap_params(df, "reference", T, model, Vg)
                    I_t0, a_t, _, _ = trap_params(df, dev, T, model, Vg)
                    ax.semilogx(D, metrics.suppression_vs_dit(D, I_r0, a_r, I_t0, a_t), ls,
                                color=COL[dev], label=f"{LABEL[dev]}" if T == T0 else None)
                    r = rules[(rules.device == dev) & (rules.btbt_model == model) & np.isclose(rules.T_K, T)]
                    Dh = r.D_half.iloc[0]
                    if Dh is not None and np.isfinite(Dh) and D[0] < Dh < D[-1]:
                        ax.plot(Dh, r.S_0.iloc[0] / 2, "o", mfc="w", color=COL[dev], ms=3.5)
            ax.set_title(f"({'abcd'[2 * i + j]}) {model.capitalize()} BTBT (cal.), $V_{{GS}}={Vg:g}$ V")
            ax.axvspan(1e11, 1e12, color="0.9", zorder=0)
            ax.grid(True, which="major", alpha=0.3)
            if i == 1:
                ax.set_xlabel("$D_{it}$ (cm$^{-2}$eV$^{-1}$)")
            if j == 0:
                ax.set_ylabel("Suppression $S$ (dec)")
    h, lab = axs[0, 0].get_legend_handles_labels()
    h += [plt.Line2D([], [], color="k", ls="-"), plt.Line2D([], [], color="k", ls="--"),
          plt.Line2D([], [], color="k", ls="", marker="o", mfc="w", ms=3.5)]
    lab += ["300 K", "398 K", "$D_{1/2}$ (S halved)"]
    fig.legend(h, lab, loc="lower center", ncol=7, bbox_to_anchor=(0.5, -0.05), frameon=False)
    fig.subplots_adjust(hspace=0.3)
    _save(fig, "fig_dit")


# ---- straggle sensitivity -------------------------------------------------------------

def sensitivity_table(main_df):
    """S_claim, S_nl and S_nl(1e12) at 300 K for junction straggle 1 (main), 2, 4 nm."""
    runs = {1: main_df}
    for nm in (2, 4):
        p = os.path.join(AUD.replace("audit_cal", "audit_cal_sigma") + f"{nm}nm" if "audit_cal" in AUD else f"results/audit_sigma{nm}nm", "table.csv")
        if os.path.exists(p):
            runs[nm] = pd.read_csv(p)
    rows = []
    for nm, df in runs.items():
        for dev in TECH:
            claim = metrics.suppression_decades(
                _one(df, kind="baseline", device="reference", Vg=VAMB).I_total,
                _one(df, kind="baseline", device=dev, Vg=VAMB).I_total)
            I_r0, a_r, _, _ = trap_params(df, "reference", T0, "nonlocal")
            I_t0, a_t, _, _ = trap_params(df, dev, T0, "nonlocal")
            rows.append(dict(sigma_nm=nm, device=dev, S_claim=float(claim),
                             S_0=float(metrics.suppression_vs_dit(0.0, I_r0, a_r, I_t0, a_t)),
                             S_1e12=float(metrics.suppression_vs_dit(DREF, I_r0, a_r, I_t0, a_t)),
                             S_inf=metrics.trap_limited_suppression(a_r, a_t),
                             I_ref_local=float(trap_params(df, "reference", T0, "local")[2]),
                             I_ref_nl=I_r0, D_cross=metrics.dit_crossover(I_t0, a_t)))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(AUD, "sensitivity_straggle.csv"), index=False)
    return out


def fig_straggle(sens):
    nms = sorted(sens.sigma_nm.unique())
    if len(nms) < 2:
        return
    fig, axs = plt.subplots(1, 3, figsize=(7.1, 2.1), sharey=True)
    for ax, (col, title) in zip(axs, [("S_claim", "(a) Claim: local, BTBT only"),
                                      ("S_0", "(b) Nonlocal, no $D_{it}$"),
                                      ("S_1e12", "(c) Nonlocal, $D_{it}=10^{12}$")]):
        w = 0.8 / len(nms)
        for i, nm in enumerate(nms):
            d = sens[sens.sigma_nm == nm].set_index("device").loc[TECH]
            ax.bar(np.arange(4) + (i - (len(nms) - 1) / 2) * w, d[col], w,
                   color=plt.cm.viridis(i / max(len(nms) - 1, 1)), label=f"straggle {nm} nm")
        ax.set_xticks(range(4), [SHORT[d] for d in TECH])
        ax.set_title(title)
        ax.axhline(0, color="k", lw=0.5)
        ax.grid(True, axis="y", alpha=0.3)
    axs[0].set_ylabel("S at $V_{GS}=-1$ V (dec)")
    axs[0].legend()
    _save(fig, "fig_straggle")


# ---- LaTeX outputs --------------------------------------------------------------------

def _e(v):
    """1.23e-09 -> 1.2\\times10^{-9} (LaTeX math)."""
    if v is None or not np.isfinite(v):
        return r"\infty"
    m, e = f"{v:.1e}".split("e")
    return f"{m}\\times10^{{{int(e)}}}"


# Highest D_it at which the linear, uncharged decomposition was checked against self-consistent
# trap charge (src.audit.charged_traps). Larger D_x values are quoted only as a bound.
DX_CAP = 1e13


def _dx(v):
    """D_x / D_1/2 in LaTeX math, quoted as a bound above the verified range."""
    if v is None or not np.isfinite(v) or v > DX_CAP:
        return ">10^{13}"
    return _e(v)


def _dit(v):
    return f"${_dx(v)}$"


def write_tables(rules, rules06, sens):
    os.makedirs(GEN, exist_ok=True)
    nl, lo = rules[rules.btbt_model == "nonlocal"], rules[rules.btbt_model == "local"]
    lines = []
    for dev in TECH:
        a = nl[(nl.device == dev) & (nl.T_K == T0)].iloc[0]
        h = nl[(nl.device == dev) & np.isclose(nl.T_K, TH)].iloc[0]
        b = lo[(lo.device == dev) & (lo.T_K == T0)].iloc[0]
        q = rules06[(rules06.device == dev) & (rules06.btbt_model == "nonlocal") & (rules06.T_K == T0)].iloc[0]
        lines.append(
            f"{SHORT[dev]} & {a.S_claim:.2f} & {b.S_1e12:.2f} & {a.S_0:.2f} & {a.S_1e12:.2f} & "
            f"{h.S_5e12:.2f} & {a.S_inf:.2f} & {q.S_0:.2f} / {q.S_1e12:.2f}\\\\")
    with open(os.path.join(GEN, "tab_suppression.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")
    lines = []
    for dev in TECH:
        r = {(m, T): rules[(rules.device == dev) & (rules.btbt_model == m) & np.isclose(rules.T_K, T)].iloc[0]
             for m in ("local", "nonlocal") for T in (T0, TH)}
        lines.append(
            f"{SHORT[dev]} & {_dit(r['local', T0].D_cross)} & {_dit(r['nonlocal', T0].D_cross)} & "
            f"{_dit(r['nonlocal', TH].D_cross)} & {_dit(r['local', T0].D_half)} & "
            f"{_dit(r['nonlocal', T0].D_half)} & {_dit(r['nonlocal', TH].D_half)}\\\\")
    with open(os.path.join(GEN, "tab_dit.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")
    if sens is not None and sens.sigma_nm.nunique() > 1:
        lines = []
        for nm in sorted(sens.sigma_nm.unique()):
            d = sens[sens.sigma_nm == nm].set_index("device")
            cells = " & ".join(f"{d.loc[k].S_claim:.2f} / {d.loc[k].S_0:.2f} / {d.loc[k].S_1e12:.2f}" for k in TECH)
            lines.append(f"{nm} & ${_e(d.iloc[0].I_ref_local)}$ & ${_e(d.iloc[0].I_ref_nl)}$ & {cells}\\\\")
        with open(os.path.join(GEN, "tab_straggle.tex"), "w") as f:
            f.write("\n".join(lines) + "\n")


def write_numbers(df, rules, sens):
    """\\newcommand macros for every number quoted in the text."""
    nl, lo = rules[rules.btbt_model == "nonlocal"], rules[rules.btbt_model == "local"]
    g = lambda d, dev, T=T0: d[(d.device == dev) & np.isclose(d.T_K, T)].iloc[0]  # noqa: E731
    ref = lambda model, T, D, Vg=VAMB: _one(df, kind="realistic", device="reference",  # noqa: E731
                                            btbt_model=model, T_K=T, D_it=D, tau=TAU0, Vg=Vg)
    N = {}
    r_l, r_n = ref("local", T0, 0.0), ref("nonlocal", T0, 0.0)
    N["IrefLocal"] = _e(r_l.I_btbt)
    N["IrefNonlocal"] = _e(r_n.I_btbt)
    N["LocalOverNonlocal"] = f"{r_l.I_btbt / r_n.I_btbt:.0f}"
    for Vg, tag in ((-0.2, "Low"), (0.0, "Zero")):
        a, b = ref("local", T0, 0.0, Vg), ref("nonlocal", T0, 0.0, Vg)
        N[f"LocalOverNonlocal{tag}"] = f"{np.log10(a.I_btbt / b.I_btbt):.1f}"
    # interface TAT vs BTBT in each technique (nonlocal, 300 K, D_it = 1e12)
    for dev in ["reference"] + TECH:
        r = _one(df, kind="realistic", device=dev, btbt_model="nonlocal", T_K=T0, D_it=DREF, tau=TAU0, Vg=VAMB)
        rl = _one(df, kind="realistic", device=dev, btbt_model="local", T_K=T0, D_it=DREF, tau=TAU0, Vg=VAMB)
        k = dev.replace("_", "").replace("reference", "Ref").title().replace("Lowdraindoping", "LDD") \
            .replace("Gdunderlap", "UL").replace("Dualwfgate", "DWF").replace("Heterogate", "HG")
        N[f"ItOverBtbt{k}"] = f"{r.I_it / r.I_btbt:.2g}"
        N[f"ItOverBtbtLocal{k}"] = f"{rl.I_it / rl.I_btbt:.2g}"
    for dev in TECH:
        k = SHORT[dev]
        a, h, b = g(nl, dev), g(nl, dev, TH), g(lo, dev)
        N[f"Sclaim{k}"] = f"{a.S_claim:.2f}"
        N[f"Snl{k}"] = f"{a.S_0:.2f}"
        N[f"SnlD{k}"] = f"{a.S_1e12:.2f}"
        N[f"SworstD{k}"] = f"{h.S_5e12:.2f}"
        N[f"Sinf{k}"] = f"{a.S_inf:.2f}"
        N[f"SlocD{k}"] = f"{b.S_1e12:.2f}"
        N[f"Rworst{k}"] = f"{h.S_5e12 / a.S_claim:.2f}"
        N[f"RtrapWorst{k}"] = f"{h.S_5e12 / a.S_0:.2f}"
        N[f"Dcross{k}"] = _dx(a.D_cross)
        N[f"DcrossHot{k}"] = _dx(h.D_cross)
        N[f"Dhalf{k}"] = _dx(a.D_half)
        N[f"DhalfHot{k}"] = _dx(h.D_half)
    lost = [1 - g(nl, d, TH).S_5e12 / g(nl, d).S_0 for d in ("heterogate", "low_drain_doping", "gd_underlap")]
    N["LostMin"], N["LostMax"] = f"{100 * min(lost):.0f}", f"{100 * max(lost):.0f}"
    N["LostDWF"] = f"{100 * (1 - g(nl, 'dual_wf_gate', TH).S_5e12 / g(nl, 'dual_wf_gate').S_0):.0f}"
    loc_lost = [1 - g(lo, d, TH).S_5e12 / g(lo, d).S_claim for d in TECH]
    N["LocalLostMax"] = f"{100 * max(loc_lost):.0f}"
    if sens is not None and sens.sigma_nm.nunique() > 1:
        s4 = sens[sens.sigma_nm == sens.sigma_nm.max()].set_index("device")
        N["SigMax"] = f"{sens.sigma_nm.max():.0f}"
        N["IrefNlSigMax"] = _e(s4.iloc[0].I_ref_nl)
        N["IrefLocalSigMax"] = _e(s4.iloc[0].I_ref_local)
    with open(os.path.join(GEN, "numbers.tex"), "w") as f:
        f.write("% generated by src/analysis/figures.py -- do not edit\n")
        for k, v in N.items():
            f.write(f"\\newcommand{{\\n{k}}}{{{v}}}\n")
    return N


# ---- bias dependence -------------------------------------------------------------------

VG_BIAS = [0.0, -0.2, -0.4, -0.6, -0.8, -1.0, -1.2, -1.4]


def bias_rules(df):
    out = pd.concat([design_rules(df, Vg) for Vg in VG_BIAS], ignore_index=True)
    out.to_csv(os.path.join(AUD, "design_rules_vs_Vg.csv"), index=False)
    return out


def fig_bias(br):
    """(a) S vs V_GS at D_it = 0, 1e11, 1e12 (calibrated nonlocal, 300 K); (b) D_x vs V_GS."""
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.6))
    nl = br[(br.btbt_model == "nonlocal") & np.isclose(br.T_K, T0)]
    nh = br[(br.btbt_model == "nonlocal") & np.isclose(br.T_K, TH)]
    for dev in TECH:
        d = nl[nl.device == dev].sort_values("Vg")
        axs[0].plot(d.Vg, d.S_0, "-", color=COL[dev], lw=1.2, label=LABEL[dev])
        axs[0].plot(d.Vg, d.S_1e11, "--", color=COL[dev], lw=1.0)
        axs[0].plot(d.Vg, d.S_1e12, ":", color=COL[dev], lw=1.4)
        axs[1].semilogy(d.Vg, d.D_cross, "-o", ms=2.5, color=COL[dev], lw=1.0)
        h = nh[nh.device == dev].sort_values("Vg")
        axs[1].semilogy(h.Vg, h.D_cross, "--", color=COL[dev], lw=0.9)
    axs[0].plot([], [], "k-", label="$D_{it}=0$")
    axs[0].plot([], [], "k--", label="$10^{11}$")
    axs[0].plot([], [], "k:", label=r"$10^{12}\,$cm$^{-2}$eV$^{-1}$")
    axs[0].set_ylabel("Suppression $S$ (dec)")
    axs[0].set_title("(a) Calibrated nonlocal, 300 K")
    axs[1].axhspan(1e11, 1e12, color="0.85", zorder=0)
    axs[1].set_ylabel(r"$D_{\times}$ (cm$^{-2}$eV$^{-1}$)")
    axs[1].set_title("(b) Crossover density (solid 300 K, dashed 398 K)")
    for ax in axs:
        ax.set_xlabel("$V_{GS}$ (V), $V_{DS}$ = 1 V")
        ax.grid(True, alpha=0.3)
    axs[0].legend(loc="upper center", bbox_to_anchor=(1.15, -0.22), ncol=7, frameon=False)
    fig.subplots_adjust(wspace=0.38)
    _save(fig, "fig_bias")


def write_bias(br):
    nl = br[(br.btbt_model == "nonlocal") & np.isclose(br.T_K, T0)]
    lines, N = [], {}
    for dev in TECH:
        cells = []
        for Vg in (-0.2, -0.6, -1.0):
            r = nl[(nl.device == dev) & np.isclose(nl.Vg, Vg)].iloc[0]
            cells.append(f"{r.S_0:.2f} & {r.S_1e11:.2f} & {r.S_1e12:.2f} & ${_dx(r.D_cross)}$")
        lines.append(f"{SHORT[dev]} & " + " & ".join(cells) + "\\\\")
        r = nl[(nl.device == dev) & np.isclose(nl.Vg, -0.2)].iloc[0]
        k = SHORT[dev]
        N[f"SbiasZero{k}"] = f"{r.S_0:.1f}"
        N[f"SbiasEleven{k}"] = f"{r.S_1e11:.1f}"
        N[f"SbiasTwelve{k}"] = f"{r.S_1e12:.1f}"
        N[f"SinfBias{k}"] = f"{r.S_inf:.1f}"
        N[f"DxBias{k}"] = _dx(r.D_cross)
    s = nl[np.isclose(nl.Vg, -0.2)]
    lost = s.S_0 - s.S_1e12
    N["BiasLostMin"], N["BiasLostMax"] = f"{lost.min():.1f}", f"{lost.max():.1f}"
    lost11 = s.S_0 - s.S_1e11
    N["BiasLostElevenMin"], N["BiasLostElevenMax"] = f"{lost11.min():.1f}", f"{lost11.max():.1f}"
    with open(os.path.join(GEN, "tab_bias.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(GEN, "numbers_bias.tex"), "w") as f:
        f.write("% generated by src/analysis/figures.py -- do not edit\n")
        for k, v in N.items():
            f.write(f"\\newcommand{{\\n{k}}}{{{v}}}\n")
    return N


def main():
    df = pd.read_csv(os.path.join(AUD, "table.csv"))
    rules = design_rules(df, VAMB)
    rules06 = design_rules(df, -0.6)
    sens = sensitivity_table(df)
    fig_structure()
    fig_transfer(df)
    fig_components(df)
    rules02 = design_rules(df, -0.2)
    fig_suppression(rules, rules02)
    fig_dit(df, {VAMB: rules, -0.2: rules02})
    fig_straggle(sens)
    write_tables(rules, rules06, sens)
    N = write_numbers(df, rules, sens)
    br = bias_rules(df)
    fig_bias(br)
    N.update(write_bias(br))
    pd.set_option("display.width", 220)
    print(rules[["device", "btbt_model", "T_K", "S_claim", "S_0", "S_1e12", "S_5e12", "S_inf",
                 "D_cross", "D_half"]].to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print(sens.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print(N)


if __name__ == "__main__":
    main()
