"""Robustness of the central claims: parameter UQ, model-form ensemble, device matrix, transport
validity and trap charge (paper Secs. robust and scope).

Inputs (run first): src.audit.model_form (all parts), src.audit.qf_validity, src.uq.btbt_uq,
src.audit.charged_traps, run_audit on configs/geom_*.yaml and the straggle configs.
Writes paper/generated/numbers_robust.tex, tab_claims.tex, tab_geom.tex, tab_itvar.tex and
paper/figs/fig_robust.{pdf,png}.
Run: python -m src.analysis.robustness_figs
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.analysis.figures import COL, GEN, SHORT, _dx, _e, _save  # noqa: E402
from src.audit import model_form as mf  # noqa: E402

MF = mf.OUT
TECHS = list(mf.TECHS)
LET = " ABCDEFG"                     # claim i -> macro suffix (TeX names cannot hold digits)
DEVICE_SETS = {                       # label -> table.csv of a calibrated 300 K audit
    "Nominal": "results/audit_cal",
    "$\\sigma_j=2$\\,nm": "results/audit_cal_sigma2nm",
    "$\\sigma_j=4$\\,nm": "results/audit_cal_sigma4nm",
    "$t_{si}=5$\\,nm": "results/geom/tsi5",
    "$t_{si}=20$\\,nm": "results/geom/tsi20",
    "$t_{ox}=3$\\,nm": "results/geom/tox3",
    "$L_g=30$\\,nm": "results/geom/lg30",
    "$N_D=5\\times10^{19}$": "results/geom/nd5e19",
    "$V_{DS}=0.5$\\,V": "results/geom/vd05",
}
CLAIM_TEXT = {
    "C1_Dx02_below_1e11_all": "$\\Dx<10^{11}$ for all four techniques at \\SI{-0.2}{V}",
    "C2_Dx02_below_1e12_3of4": "$\\Dx<10^{12}$ for at least three techniques at \\SI{-0.2}{V}",
    "C3_loss02_1e12_ge_0p5_all": "every technique loses $\\ge0.5$ dec at $10^{12}$, \\SI{-0.2}{V}",
    "C4_m1V_HG_UL_DWF_loss_lt_0p1": "HG, UL, DWF lose $<0.1$ dec at $10^{12}$, \\SI{-1}{V}",
    "C5_m1V_LDD_lowest_Dx": "LDD has the lowest $\\Dx$ at \\SI{-1}{V}",
    "C6_Sinf_DWF_smallest": "DWF has the smallest $\\Sinf$ (\\SI{-1}{V})",
    "C7_Sinf_LDD_largest": "LDD has the largest $\\Sinf$ (\\SI{-1}{V})",
}
IT_LABEL = {
    "nominal": "nominal (uniform, $\\sigma=10^{-15}$, $m^*=0.25$)",
    "sigma_1e-14": "$\\sigma=10^{-14}$\\,cm$^2$", "sigma_1e-16": "$\\sigma=10^{-16}$\\,cm$^2$",
    "sigma_n100p": "$\\sigma_n/\\sigma_p=100$", "sigma_p100n": "$\\sigma_p/\\sigma_n=100$",
    "m_0.1": "$m^*=0.1$", "m_0.5": "$m^*=0.5$", "m_1.0": "$m^*=1$ (weak enh.)",
    "m_2.0": "$m^*=2$ (weak enh.)", "u_shape": "U-shaped $\\Dit(E)$",
    "gauss_mid": "Gaussian at midgap", "gauss_up0.3": "Gaussian, $E_i+0.3$\\,eV",
    "gauss_dn0.3": "Gaussian, $E_i-0.3$\\,eV", "damage_x10": "$10\\times$ damage, 10\\,nm",
    "damage_x10_5nm": "$10\\times$ damage, 5\\,nm", "damage_x10_20nm": "$10\\times$ damage, 20\\,nm",
    "gate_only": "no spacer-interface traps",
    "no_field_enh": "no field enhancement (plain SRH)",
}


def _rng(lo, hi, f="{:.2f}"):
    """Numeric range in math mode, so that negative ends render as minus signs."""
    return f"${f.format(lo)}$" if f.format(lo) == f.format(hi) else f"${f.format(lo)}$--${f.format(hi)}$"


def _drng(lo, hi):
    """Range of trap densities; a range whose ends are both capped collapses to one bound."""
    a, b = _dx(lo), _dx(hi)
    return f"${a}$" if a == b else f"${a}$--${b}$"


def device_metrics(path):
    """metrics() per Vg from a calibrated audit table (nonlocal, tau 1e-7, uniform, 300 K)."""
    d = pd.read_csv(os.path.join(path, "table.csv"))
    d = d[(d.kind == "realistic") & (d.btbt_model == "nonlocal") & np.isclose(d.tau, 1e-7)
          & np.isclose(d.T_K, 300.0)]
    if "dit_profile" in d:
        d = d[d.dit_profile == "uniform"]
    out = {}
    for Vg in (-1.0, -0.2):
        g = d[np.isclose(d.Vg, Vg)]
        I0 = g[g.D_it == 0].set_index("device").I_total
        I12 = g[np.isclose(g.D_it, 1e12)].set_index("device").I_total
        a = (I12 - I0) / 1e12
        out[Vg] = mf.metrics(I0, a)
    return out


def uq_claims():
    """Fraction of the BTBT-parameter Monte Carlo samples meeting each claim (nominal models)."""
    s = pd.read_csv(os.path.join("results", "uq", "uq_samples.csv.gz"))
    s = s[(s.model == "nonlocal") & np.isclose(s.T_K, 300.0)]
    res = {c: [] for c in mf.CLAIMS}
    piv = {Vg: s[np.isclose(s.Vg, Vg)].pivot(index="sample", columns="device") for Vg in (-1.0, -0.2)}
    for k in piv[-1.0].index:
        m = {}
        for Vg, p in piv.items():
            I0 = (p.I_btbt + p.I_tat_bulk).loc[k]
            m[Vg] = mf.metrics(I0, p.a_it.loc[k])
        for c, f in mf.CLAIMS.items():
            res[c].append(bool(f(m)))
    return {c: float(np.mean(v)) for c, v in res.items()}


def main():
    N = {}
    ens = pd.read_csv(os.path.join(MF, "ensemble.csv"))
    cl = pd.read_csv(os.path.join(MF, "claims.csv"))
    oat = pd.read_csv(os.path.join(MF, "one_at_a_time.csv"))
    tmp = pd.read_csv(os.path.join(MF, "it_temperature.csv"))
    bt = pd.read_csv(os.path.join(MF, "btbt_variants.csv"))
    alt = json.load(open(mf.ALT))

    # --- calibration extrapolation test
    lf = alt["nonlocal"]["low_field_half"]
    N["ExtrapHeldRms"] = f"{lf['heldout_rms_dec']:.2f}"
    N["ExtrapHeldBias"] = f"{lf['heldout_bias_dec']:+.2f}"
    N["ExtrapRatio"] = f"{alt['field_ratio_extrapolated']:.1f}"
    N["ExtrapSplit"] = f"{alt['field_split_Vcm'] / 1e6:.2f}"
    N["AltPRmsMax"] = f"{max(alt['nonlocal'][k]['rms_dec'] for k in ('P2', 'P3')):.2f}"

    # --- path bracket (nominal parameters, 300 K)
    b = bt[(bt.params == "nominal") & np.isclose(bt.T_K, 300.0)].pivot_table(index=["device", "Vg"], columns="path", values="I_btbt")
    r = b.loc[("reference", -0.2)]
    N["PathAcRefTwo"] = f"{r.angle_complete / r.rows_cols_max:.1f}"
    N["PathGsRefTwo"] = f"{np.log10(r.rows_cols_max / r.gradient_start):.1f}"
    r1 = b.loc[("reference", -1.0)]
    N["PathAcRefOne"] = f"{r1.angle_complete / r1.rows_cols_max:.1f}"
    tech02 = b.loc[[(k, -0.2) for k in TECHS]]
    gain = tech02.angle_complete / tech02.rows_cols_max
    N["PathAcTechLo"], N["PathAcTechHi"] = f"{gain.min():.1f}", f"{gain.max():.1f}"
    conv = pd.read_csv(os.path.join(MF, "path_convergence.csv"))
    spread = conv.groupby(["device", "Vg", "model"]).I.agg(lambda v: v.max() / v.min() - 1.0)
    N["PathConvPct"] = f"{100 * spread[spread.index.get_level_values('model') == 'angle_complete'].max():.0f}"
    N["PathConvGsFactor"] = f"{1 + spread[spread.index.get_level_values('model') == 'gradient_start'].max():.1f}"

    # --- ensemble
    N["EnsN"] = str(len(cl))
    N["EnsNpath"] = str(ens.path.nunique())
    N["EnsNparams"] = str(ens.params.nunique())
    N["EnsNit"] = str(ens["it"].nunique())
    frac = cl.drop(columns=["path", "params", "it"]).mean()
    fin = cl[cl["it"] != "no_field_enh"].drop(columns=["path", "params", "it"]).mean()
    uq = uq_claims()
    devs = {lab: device_metrics(p) for lab, p in DEVICE_SETS.items() if os.path.exists(os.path.join(p, "table.csv"))}
    dev_pass = {c: sum(bool(f(m)) for m in devs.values()) for c, f in mf.CLAIMS.items()}
    for i, c in enumerate(mf.CLAIMS, 1):
        N[f"ClaimEns{LET[i]}"] = f"{100 * frac[c]:.0f}"
        N[f"ClaimEnsEnh{LET[i]}"] = f"{100 * fin[c]:.0f}"
        N[f"ClaimUq{LET[i]}"] = f"{100 * uq[c]:.0f}"
        N[f"ClaimDev{LET[i]}"] = str(dev_pass[c])
    N["DevN"] = str(len(devs))
    e02 = ens[np.isclose(ens.Vg, -0.2)]
    enh = e02[e02["it"] != "no_field_enh"]
    N["EnsDxTwoMin"], N["EnsDxTwoMax"] = _dx(enh.Dx.min()), _dx(enh.Dx.max())
    N["EnsDxTwoMaxSrh"] = _dx(e02.Dx.max())
    # nominal path and BTBT parameters, one interface variant: D_x range over the techniques
    base = e02[(e02.path == "rows_cols_max") & (e02.params == "nominal")]
    for v, tag in (("nominal", "Nom"), ("m_1.0", "Mone"), ("m_2.0", "Mtwo"), ("no_field_enh", "Srh")):
        g = base[base["it"] == v].Dx
        N[f"DxTwo{tag}Lo"], N[f"DxTwo{tag}Hi"] = _dx(g.min()), _dx(g.max())
        N[f"DxTwo{tag}LogLo"], N[f"DxTwo{tag}LogHi"] = f"{np.log10(g.min()):.0f}", f"{np.log10(g.max()):.0f}"
    # claims that hold in every model whose interface variant has Hurkx-type enhancement with m* <= 0.5
    strong_set = cl[~cl["it"].isin(["no_field_enh", "m_1.0", "m_2.0"])]
    N["EnsNstrong"] = str(len(strong_set))
    strong = strong_set.drop(columns=["path", "params", "it"]).mean()
    for i, c in enumerate(mf.CLAIMS, 1):
        N[f"ClaimEnsStrong{LET[i]}"] = f"{100 * strong[c]:.0f}"
    loss = enh.S0 - enh["S_1e+12"]
    N["EnsLossTwoMin"] = f"{loss.min():.2f}"
    for k in TECHS:
        g = ens[np.isclose(ens.Vg, -1.0) & (ens.device == k) & (ens["it"] != "no_field_enh")]
        N[f"EnsSinfLo{SHORT[k]}"], N[f"EnsSinfHi{SHORT[k]}"] = f"{g.S_inf.min():.2f}", f"{g.S_inf.max():.2f}"
    # extrapolation: shift of the trap-free S at -1 V across the BTBT calibrations (nominal path)
    pe = ens[(ens.path == "rows_cols_max") & (ens["it"] == "nominal") & np.isclose(ens.Vg, -1.0)]
    pv = pe.pivot(index="params", columns="device", values="S0")
    N["ParamsSOneShift"] = f"{(pv - pv.loc['nominal']).abs().to_numpy().max():.2f}"
    pv2 = ens[(ens.path == "rows_cols_max") & (ens["it"] == "nominal") & np.isclose(ens.Vg, -0.2)].pivot(
        index="params", columns="device", values="Dx")
    N["ParamsDxTwoShift"] = f"{np.abs(np.log10(pv2 / pv2.loc['nominal'])).to_numpy().max():.1f}"
    minor = [v for v in mf.IT_VARIANTS if v not in ("nominal", "no_field_enh") and not v.startswith("m_")]
    pv3 = ens[(ens.path == "rows_cols_max") & (ens.params == "nominal") & np.isclose(ens.Vg, -0.2)].pivot(
        index="it", columns="device", values="Dx")
    N["ItMinorDxShift"] = f"{np.abs(np.log10(pv3.loc[minor] / pv3.loc['nominal'])).to_numpy().max():.1f}"
    o = oat[np.isclose(oat.Vg, -0.2) & (oat.metric == "Dx")]
    for f_ in ("path", "params", "it"):
        g = o[o.factor == f_] if f_ != "it" else o[(o.factor == f_)]
        N[f"OatDx{f_.capitalize()}"] = f"{np.log10(g.hi / g.lo).max():.1f}"
    o = oat[np.isclose(oat.Vg, -1.0) & (oat.metric == "S_inf") & (oat.factor == "it")]
    N["OatSinfIt"] = f"{(o.hi - o.lo).max():.2f}"

    # --- interface variants: temperature and S_inf
    t02 = tmp[np.isclose(tmp.Vg, -0.2) & tmp.device.isin(TECHS + ["reference"])]
    ea = t02.groupby("variant").Ea.agg(["min", "max"])
    enh_var = [v for v in mf.IT_VARIANTS if v != "no_field_enh"]
    N["ItEaMin"], N["ItEaMax"] = f"{ea.loc[enh_var, 'min'].min():.2f}", f"{ea.loc[enh_var, 'max'].max():.2f}"
    N["ItEaNomLo"], N["ItEaNomHi"] = f"{ea.loc['nominal', 'min']:.2f}", f"{ea.loc['nominal', 'max']:.2f}"
    N["ItEaSrhLo"], N["ItEaSrhHi"] = f"{ea.loc['no_field_enh', 'min']:.2f}", f"{ea.loc['no_field_enh', 'max']:.2f}"
    rat = t02[t02.variant == "nominal"].ratio_398_300
    N["ItGainNomMax"] = f"{rat.max():.1f}"
    for v, tag in (("m_1.0", "Mone"), ("m_2.0", "Mtwo")):
        N[f"ItEa{tag}Lo"], N[f"ItEa{tag}Hi"] = f"{ea.loc[v, 'min']:.2f}", f"{ea.loc[v, 'max']:.2f}"
    s1 = ens[(ens.path == "rows_cols_max") & (ens.params == "nominal") & np.isclose(ens.Vg, -1.0)]
    sinf = s1.pivot(index="it", columns="device", values="S_inf")
    for k in TECHS:
        N[f"SinfNom{SHORT[k]}"] = f"{sinf.loc['nominal', k]:.2f}"

    rows = []
    d02 = ens[(ens.path == "rows_cols_max") & (ens.params == "nominal") & np.isclose(ens.Vg, -0.2)].pivot(
        index="it", columns="device", values="Dx")
    wanp = os.path.join(MF, "wan_residuals.csv")
    wan = pd.read_csv(wanp) if os.path.exists(wanp) else None
    for v in mf.IT_VARIANTS:
        e = ea.loc[v]
        row = [IT_LABEL[v], _rng(sinf.loc[v].min(), sinf.loc[v].max()),
               _drng(d02.loc[v].min(), d02.loc[v].max()), _rng(e["min"], e["max"])]
        if wan is not None:
            w = wan[(wan.path == "rows_cols_max") & (wan["it"] == v) & (wan.device == "hfo2") & (wan.L_IN_nm <= 25)
                    & np.isclose(wan.D_it, 1e11) & (wan.offset_nm == 0)]   # same convention as validation_figs
            row.append(f"${w.bias.mean():+.2f}$")
        rows.append(" & ".join(row) + r" \\")
    open(os.path.join(GEN, "tab_itvar.tex"), "w").write("\n".join(rows) + "\n")
    if wan is not None:
        w = wan[(wan.device == "hfo2") & (wan.L_IN_nm <= 25) & np.isclose(wan.D_it, 1e11) & (wan.path == "rows_cols_max")
                & (wan.offset_nm == 0)]
        wb = w.groupby("it").bias.mean()
        N["WanHfBiasNom"] = f"{wb['nominal']:+.2f}"
        N["WanHfBiasSrh"] = f"{wb['no_field_enh']:+.2f}"
        w0 = wan[(wan.device == "hfo2") & (wan.L_IN_nm <= 25) & (wan.D_it == 0) & (wan.path == "rows_cols_max") & (wan["it"] == "nominal")
                 & (wan.offset_nm == 0)]
        N["WanHfBiasNoTrap"] = f"{w0.bias.mean():+.2f}"
        ok = wan[(wan.path == "rows_cols_max") & (wan.L_IN_nm == 10)
                 & (((wan.device == "sio2") & np.isclose(wan.D_it, 1e10)) | ((wan.device == "hfo2") & np.isclose(wan.D_it, 1e11)))]
        pr = wan[(wan.L_IN_nm == 10) & (wan["it"] == "nominal")
                 & (((wan.device == "sio2") & np.isclose(wan.D_it, 1e10)) | ((wan.device == "hfo2") & np.isclose(wan.D_it, 1e11)))]
        rr = pr.groupby("path").rms.agg(["min", "max"])
        for p, tag in (("rows_cols_max", "Rc"), ("gradient_start", "Gs"), ("angle_complete", "Ac")):
            if p in rr.index:
                N[f"WanRms{tag}Lo"], N[f"WanRms{tag}Hi"] = f"{rr.loc[p, 'min']:.1f}", f"{rr.loc[p, 'max']:.1f}"
        del ok
    bisp = os.path.join(MF, "biswas_residuals.csv")
    if os.path.exists(bisp):
        bis = pd.read_csv(bisp)
        bb = bis[np.isclose(bis.Vd, 1.0)].groupby("path").bias.agg(["min", "max"])
        for p, tag in (("rows_cols_max", "Rc"), ("gradient_start", "Gs"), ("angle_complete", "Ac")):
            if p in bb.index:
                N[f"BisBias{tag}Lo"], N[f"BisBias{tag}Hi"] = f"{bb.loc[p, 'min']:+.1f}", f"{bb.loc[p, 'max']:+.1f}"

    # --- claims table
    lines = []
    for i, c in enumerate(mf.CLAIMS, 1):
        lines.append(f"{CLAIM_TEXT[c]} & {100 * uq[c]:.0f} & {100 * frac[c]:.0f} ({100 * fin[c]:.0f}) & "
                     f"{dev_pass[c]}/{len(devs)} \\\\")
    open(os.path.join(GEN, "tab_claims.tex"), "w").write("\n".join(lines) + "\n")

    # --- device matrix table
    lines = []
    for lab, m in devs.items():
        dx = [m[-0.2][k]["Dx"] for k in TECHS]
        lo = [m[-0.2][k]["S0"] - m[-0.2][k]["S_1e+12"] for k in TECHS]
        si = " / ".join(f"${m[-1.0][k]['S_inf']:.2f}$" for k in TECHS)
        lines.append(f"{lab} & {_drng(min(dx), max(dx))} & {_rng(min(lo), max(lo), '{:.1f}')} & {si} \\\\")
    open(os.path.join(GEN, "tab_geom.tex"), "w").write("\n".join(lines) + "\n")
    alld = [m[-0.2][k]["Dx"] for m in devs.values() for k in TECHS]
    N["DevDxTwoMax"] = _dx(max(alld))
    N["DevLossTwoMin"] = f"{min(m[-0.2][k]['S0'] - m[-0.2][k]['S_1e+12'] for m in devs.values() for k in TECHS):.1f}"
    N["DevNearCeilMax"] = f"{max(abs(m[-0.2][k]['S_1e+12'] - m[-0.2][k]['S_inf']) for m in devs.values() for k in TECHS):.1f}"

    # --- transport validity, trap charge at -0.2 V
    qf = pd.read_csv(os.path.join(MF, "qf_validity.csv"))
    tq = qf[qf.device != "reference"]
    N["QfTechMax"] = f"{tq.dPhi_over_kT.max():.2f}"
    rq = qf[(qf.device == "reference")]
    N["QfRefOne"] = f"{rq[np.isclose(rq.Vg, -1.0)].dPhi_over_kT.max():.1f}"
    N["QfRefTwo"] = f"{rq[np.isclose(rq.Vg, -0.2)].dPhi_over_kT.max():.2f}"
    r1 = rq[np.isclose(rq.Vg, -1.0) & np.isclose(rq.T_K, 300.0) & (rq.D_it == 0)]
    N["QfRefOneDlog"] = f"{r1.dlogI.iloc[0]:.2f}"            # = reduction of every S at -1 V
    N["QfTechDlogMax"] = f"{tq.dlogI.max():.3f}"
    bad = rq[rq.dPhi_over_kT > 1.0]
    N["QfRefVgLimit"] = f"{bad.Vg.max():.1f}" if len(bad) else "none"
    # --- where the trap current is generated (it_spatial.csv)
    sp = pd.read_csv(os.path.join(MF, "it_spatial.csv"))
    N["ItColocRefNm"] = f"{sp[sp.device == 'reference'].r80_nm.max():.1f}"
    st1 = sp[sp.device.isin(TECHS) & np.isclose(sp.Vg, -1.0)]
    N["ItColocTechNm"] = f"{st1.r80_nm.max():.1f}"
    st2 = sp[sp.device.isin(TECHS) & np.isclose(sp.Vg, -0.2)].set_index("device")
    N["ItSrcFracMax"] = f"{100 * st2.frac_src10.max():.0f}"
    dxn = base[base["it"] == "nominal"].set_index("device").Dx
    drain_only = dxn / (1.0 - st2.frac_src10.reindex(dxn.index))
    N["DxDrainOnlyHi"] = _dx(drain_only.max())
    # --- floors: Auger generation and the gate-leakage density that would equal the 1e11 trap current
    fl = pd.read_csv(os.path.join(MF, "floors.csv"))
    N["AugerMax"] = _e(fl.I_auger.max())
    N["AugerBelowIt"] = f"{np.log10(fl.I_it_1e11 / fl.I_auger).min():.0f}"
    N["JgStarMin"], N["JgStarMax"] = _e(fl.J_gate_star.min()), _e(fl.J_gate_star.max())
    ch = pd.read_csv(os.path.join("results", "audit_cal", "charged_traps_S.csv")).dropna()
    c2 = ch[np.isclose(ch.Vg, -0.2)] if "Vg" in ch else ch.iloc[0:0]
    if len(c2):
        N["ChargedMaxdSTwo"] = f"{c2[c2.D_it <= 1e12].dS.abs().max():.2f}"
        N["ChargedMaxdSTwoAll"] = f"{c2.dS.abs().max():.2f}"

    with open(os.path.join(GEN, "numbers_robust.tex"), "w") as fh:
        fh.write("% generated by src.analysis.robustness_figs -- do not edit\n")
        for k, v in N.items():
            fh.write(f"\\newcommand{{\\n{k}}}{{{v}}}\n")

    fig_robust(ens, devs)
    print("\n".join(f"{k} = {v}" for k, v in N.items()))


def fig_robust(ens, devs):
    u02 = pd.read_csv(os.path.join("results", "uq", "uq_summary_Vg-0.2.csv"))
    u02 = u02[(u02.model == "nonlocal") & np.isclose(u02.T_K, 300.0)].set_index("device")
    fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.3))
    xs = np.arange(len(TECHS))
    nom = ens[(ens.path == "rows_cols_max") & (ens.params == "nominal") & (ens["it"] == "nominal")]
    enh = ens[ens["it"] != "no_field_enh"]
    panels = [(axs[0], -0.2, "Dx", "$D_\\times$ at $-0.2$ V (cm$^{-2}$eV$^{-1}$)", True),
              (axs[1], -0.2, "loss", "suppression lost at $10^{12}$, $-0.2$ V (dec)", False),
              (axs[2], -1.0, "S_inf", "$S_\\infty$ at $-1$ V (dec)", False)]
    for ax, Vg, met, lab, logy in panels:
        for i, k in enumerate(TECHS):
            def val(df):
                g = df[np.isclose(df.Vg, Vg) & (df.device == k)]
                return (g.S0 - g["S_1e+12"]) if met == "loss" else g[met]
            v_all, v_enh, v_nom = val(ens), val(enh), float(val(nom).iloc[0])
            ax.plot([i, i], [v_all.min(), v_all.max()], color="0.75", lw=5, solid_capstyle="butt")
            ax.plot([i, i], [v_enh.min(), v_enh.max()], color=COL[k], lw=2.2, solid_capstyle="butt")
            if met == "Dx" and k in u02.index:
                ax.plot([i - 0.18] * 2, [u02.loc[k, "Dx_p05"], u02.loc[k, "Dx_p95"]], color="k", lw=1.2)
            ax.plot(i, v_nom, "ko", ms=3.5, zorder=5)
            dv = [m[Vg][k]["Dx"] if met == "Dx" else (m[Vg][k]["S0"] - m[Vg][k]["S_1e+12"] if met == "loss"
                  else m[Vg][k]["S_inf"]) for m in devs.values()]
            ax.plot(np.full(len(dv), i + 0.22), dv, "s", mfc="none", mec="0.3", ms=2.8, mew=0.6)
        ax.set_xticks(xs, [SHORT[k] for k in TECHS])
        ax.set_title(lab, fontsize=7.5)
        if logy:
            ax.set_yscale("log")
            ax.axhspan(1e11, 1e12, color="0.9", zorder=0)
            ax.axhline(1e13, color="0.4", ls="--", lw=0.7, zorder=1)    # self-consistent charge checked up to here
        ax.set_xlim(-0.6, len(TECHS) - 0.4)
    axs[0].plot([], [], "ko", ms=3.5, label="nominal")
    axs[0].plot([], [], color="k", lw=1.2, label="parameter UQ 90%")
    axs[0].plot([], [], color="C0", lw=2.2, label="model-form envelope")
    axs[0].plot([], [], color="0.75", lw=5, label="incl. plain SRH")
    axs[0].plot([], [], "s", mfc="none", mec="0.3", ms=2.8, label="device matrix")
    axs[0].legend(loc="upper left", fontsize=5.5, frameon=False)
    for ax, t in zip(axs, "abc"):
        ax.text(0.02, 0.02, f"({t})", transform=ax.transAxes, fontsize=7)
    fig.tight_layout()
    _save(fig, "fig_robust")


if __name__ == "__main__":
    main()
