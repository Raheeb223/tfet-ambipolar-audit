"""Frozen-model check of the interface-trap variants on the irradiated CEA-LETI SOI n-TFET of
Ding et al. 2014 (role: validation, post hoc; nothing is fitted to these data).

Ding et al. raised the gate-interface trap density with 10-keV X-rays (charge pumping:
Delta N_it = 2.3e11 cm^-2 at 1.67 Mrad(SiO2), from about 4e10 cm^-2) and measured DC and
500-kHz pulsed transfer curves before and after. The pulsed curve suppresses trap-assisted
current, so its shift is the electrostatic shift (oxide and interface charge). Removing that
shift from the DC curve leaves the change of the trap/leakage current at fixed electrostatics:

    dI_meas(V) = I_DC,post(V + delta) - I_DC,pre(V),   delta = shift of the 500-kHz curve.

Each frozen interface-trap variant (src.audit.model_form.IT_VARIANTS) predicts that change as
a_v(V) * Delta D_it, with Delta D_it = Delta N_it / dE_cp and the charge-pumping energy window
dE_cp = 0.3-0.6 eV (ASSUMPTION, the window used in the paper). The prediction is a lower bound
when the traps are put on the gate interface only (radiation also creates traps at the BOX
and spacer interfaces, which only add current). Bias mapping (ASSUMPTION, logged): measured
and simulated gate voltages are aligned where the simulated BTBT current equals the lowest
measured 500-kHz current; the same offset is used down the DC tail, and the comparison is
repeated with the alignment moved by +-ALIGN_DV (the variant's least favourable case counts).

Run:  python -m src.validation.ding2014 [--run] [--analyse]
Out:  results/validation/ding2014_sim.csv, ding2014_check.csv, ding2014_summary.csv;
      paper/generated/numbers_ding.tex; paper/figs/fig_ding.(pdf|png)
"""

import argparse
import os

import numpy as np
import pandas as pd

from src.audit.model_form import IT_VARIANTS
from src.devices import tfet_2d as t
from src.validation.wan2011 import calibrated

OUT = os.path.join("results", "validation")
CONFIG = "configs/ding2014.yaml"
DIG = os.path.join("data", "digitized")
LG = 400e-7                      # [D] L = 0.4 um
L_IN = 25e-7                     # ASSUMPTION, see configs/ding2014.yaml
VD = 1.0                         # [D] V_d(N+) = 1 V
VG = list(np.round(np.arange(0.0, 2.401, 0.1), 3))
OFFSETS_NM = [0.0, 20.0]         # a-priori drain-implant systematic (as for Wan/Biswas), not fitted
D_REF = 1.0e11
DNIT = 2.3e11                    # [D] Table I, cm^-2
NIT_PRE = 3.7e10                 # [D] Fig. 8 / eq. (1): I_cp,max ~ 1.2e-10 A, f = 500 kHz, A_g = 4 um^2
DE_CP = (0.3, 0.6)               # eV, ASSUMPTION (Groeseneken 1984 window, as in the paper)
VARIANTS = ("nominal", "m_0.5", "m_1.0", "m_2.0", "no_field_enh", "sigma_1e-16", "gauss_mid")
I_ANCHOR = 1.5e-13               # A/um: lowest current on the measured 500-kHz pre-irradiation curve
ALIGN_DV = 0.1                   # V, ASSUMPTION: uncertainty of the bias alignment, scanned +-
GEN = os.path.join("paper", "generated")


def device_cfg(base, offset_nm):
    off = offset_nm * 1e-7
    return t.variant(base, **{"doping.drain_profile.offset_cm": off,
                              "mesh.fine_ranges_cm": [[-1.0e-6, 1.0e-6], [LG - 1.0e-6, LG + L_IN + off]]})


def trap_slopes(sol, cfg, F):
    """a_v (A/um per cm^-2 eV^-1) for each variant, all interfaces and gate interface only."""
    out = {}
    for v in VARIANTS:
        for tag, extra in (("", {}), ("_gate", {"gate_only": True})):
            c = t.variant(cfg)
            it = c["physics"]["interface_traps"]
            it.update(enabled=True, D_it_cm2eV=1.0)
            it.update(IT_VARIANTS[v])
            it.update(extra)
            c["physics"]["tat_bulk"]["enabled"] = False
            r, _ = t.generation(sol, c, F=F)
            out[f"a_{v}{tag}"] = r["I_it"]
    return out


def run(offsets=OFFSETS_NM):
    base = calibrated(t.load_config(CONFIG))
    rows = []
    for off in offsets:
        cfg = device_cfg(base, off)
        dev = t.build(cfg)
        sols = []
        try:
            t.ramp_to(dev, VG[0], VD)
            for Vg in VG:
                t.ramp_to(dev, Vg, VD)
                s = t.extract(dev)
                sols.append(s)
                r, _ = t.generation(s, cfg)
                row = dict(offset_nm=off, Vg=Vg, I_btbt_local=r["I_btbt_local"],
                           I_btbt_nl=r["I_btbt_nonlocal"], I_tat_bulk=r["I_tat_bulk"])
                row.update(trap_slopes(s, cfg, t.field_magnitude(s)))
                rows.append(row)
                print(off, Vg, f"nl={row['I_btbt_nl']:.2e} tat={row['I_tat_bulk']:.2e} "
                      f"a_nom={row['a_nominal']:.2e} a_srh={row['a_no_field_enh']:.2e}", flush=True)
        finally:
            dev.delete()
        if sols:
            t.save_solutions(os.path.join(OUT, "cache", f"ding_off{off:g}.npz"), sols, cfg)
    df = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, "ding2014_sim.csv"), index=False)
    return df


# ------------------------------------------------------------------ measured side
def curve(name):
    d = pd.read_csv(os.path.join(DIG, f"ding2014_fig4_{name}.csv"))
    return d.VG_V.values, d.ID_Aum.values


def v_at(V, I, Iq):
    """Gate voltage at current Iq on a monotone digitized curve (log-linear)."""
    return float(np.interp(np.log10(Iq), np.log10(I), V))


def i_at(V, I, Vq):
    return float(10 ** np.interp(Vq, V, np.log10(I), left=np.nan, right=np.nan))


CURVES = ("500kHz_pre", "500kHz_1p67M", "DC_pre", "DC_1p67M")


def measured(curves=None, I_levels=(1.5e-13, 2e-13, 3e-13, 5e-13, 1e-12, 2e-12)):
    """Electrostatic shift (500 kHz) and the trap/leakage current change at fixed electrostatics.
    curves: {name: (V, I)} for CURVES; read from data/digitized when None."""
    curves = curves or {k: curve(k) for k in CURVES}
    Pp, Pq, Dp, Dq = (curves[k] for k in CURVES)
    lv = np.logspace(np.log10(1.5e-13), np.log10(2e-12), 12)
    shifts = np.array([v_at(*Pq, I) - v_at(*Pp, I) for I in lv])
    # spread over current levels, but at least the digitizing resolution (1 px ~ 4 mV, two curves)
    delta, d_sd = float(shifts.mean()), max(float(shifts.std(ddof=1)), 0.010)
    rows = []
    for I in I_levels:
        V = v_at(*Dp, I)
        post = [i_at(*Dq, V + delta + k * d_sd) for k in (-1, 0, 1)]
        rows.append(dict(I_DC_pre=I, V_meas=V, dI=post[1] - I, dI_lo=min(post) - I, dI_hi=max(post) - I,
                         I_pulsed_pre=i_at(*Pp, V)))
    return pd.DataFrame(rows), delta, d_sd


# ------------------------------------------------------------------ comparison
def analyse():
    sim = pd.read_csv(os.path.join(OUT, "ding2014_sim.csv"))
    meas, delta, d_sd = measured()
    Pp = curve("500kHz_pre")
    V_anchor = v_at(*Pp, I_ANCHOR)
    dD = (DNIT / DE_CP[1], DNIT / DE_CP[0])
    rows = []
    for off, s in sim.groupby("offset_nm"):
        s = s.sort_values("Vg")
        ok = s.I_btbt_nl > 0
        V_sim = float(np.interp(np.log10(I_ANCHOR), np.log10(s.I_btbt_nl[ok]), s.Vg[ok]))
        shift = V_sim - V_anchor
        for m in meas.itertuples():
            Vs = m.V_meas + shift
            r = dict(offset_nm=off, I_DC_pre=m.I_DC_pre, V_meas=m.V_meas, V_sim=Vs,
                     dI_meas=m.dI, dI_meas_lo=m.dI_lo, dI_meas_hi=m.dI_hi,
                     I0_sim=float(10 ** np.interp(Vs, s.Vg, np.log10(s.I_btbt_nl + s.I_tat_bulk))))
            for v in VARIANTS:
                for tag in ("", "_gate"):
                    a_at = lambda V: float(10 ** np.interp(V, s.Vg, np.log10(np.maximum(s[f"a_{v}{tag}"], 1e-300))))
                    a = a_at(Vs)
                    r[f"dI_{v}{tag}_lo"], r[f"dI_{v}{tag}_hi"] = a * dD[0], a * dD[1]
                    # most favourable alignment within +-ALIGN_DV (a_v rises with V_G)
                    r[f"dI_{v}{tag}_min"] = min(a_at(Vs + k * ALIGN_DV) for k in (-1, 0, 1)) * dD[0]
                    r[f"Ipre_{v}{tag}"] = r["I0_sim"] + a * NIT_PRE / np.mean(DE_CP)
            rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "ding2014_check.csv"), index=False)
    summ = summary(df[df.offset_nm == df.offset_nm.min()])
    summ.to_csv(os.path.join(OUT, "ding2014_summary.csv"), index=False)
    write_numbers(meas, delta, summ, df)
    fig_ding(df[df.offset_nm == df.offset_nm.min()])
    print(f"electrostatic shift (500 kHz): {1e3 * delta:.0f} +- {1e3 * d_sd:.0f} mV; "
          f"Delta D_it = {dD[0]:.1e}-{dD[1]:.1e} cm^-2 eV^-1")
    for r in df.to_dict("records"):
        print(f"off={r['offset_nm']:g} I_DC,pre={r['I_DC_pre']:.1e} V={r['V_meas']:.3f} (sim {r['V_sim']:.3f}) "
              f"dI_meas={r['dI_meas']:.1e} [{r['dI_meas_lo']:.1e},{r['dI_meas_hi']:.1e}] I0_sim={r['I0_sim']:.1e}")
        for v in VARIANTS:
            g, h = r[f"dI_{v}_gate_lo"], r[f"dI_{v}_gate_hi"]
            print(f"     {v:14s} predicted dI (gate interface only) {g:.1e}-{h:.1e}  "
                  f"min/measured-max {g / max(r['dI_meas_hi'], 1e-30):.2g}  pre-tail {r[f'Ipre_{v}_gate']:.1e}")
    print(summ.to_string(index=False))
    return df


def summary(df):
    """Per variant: the smallest ratio of predicted (gate interface only, lower end of the
    D_it window, least favourable alignment) to measured (upper end) current change over
    the current levels. ratio > 1: the variant predicts more than was measured even as a
    lower bound, so it is excluded. Also the predicted pre-irradiation tail over the
    measured DC current at the same levels (a check at the measured initial N_it)."""
    ok = df.dI_meas_hi > 0
    rows = []
    for v in VARIANTS:
        ratio = df[f"dI_{v}_gate_min"][ok] / df.dI_meas_hi[ok]
        rows.append(dict(variant=v, ratio_min=float(ratio.min()), ratio_max=float(ratio.max()),
                         pre_tail_ratio_min=float((df[f"Ipre_{v}_gate"] / df.I_DC_pre).min()),
                         excluded=bool(ratio.min() > 1)))
    return pd.DataFrame(rows)


def response_slope(I_pre, dI):
    """d ln I / d ln N_it across the irradiation, from the measured current change."""
    return np.log1p(np.asarray(dI) / np.asarray(I_pre)) / np.log((NIT_PRE + DNIT) / NIT_PRE)


def sig2(x, mode):
    """x to two significant figures, rounded toward -inf ('floor') or +inf ('ceil'), so a
    quoted range always contains the computed one."""
    e = int(np.floor(np.log10(x)))
    scale = 10.0 ** (e - 1)
    v = (np.floor if mode == "floor" else np.ceil)(round(x / scale, 9)) * scale
    if v >= 10 ** (e + 1):                              # ceil carried into the next decade
        e += 1
    return f"{v:.0f}" if v >= 10 else f"{v:.{max(0, 1 - e)}f}"


def write_numbers(meas, delta, summ, df):
    from src.analysis.figures import _e
    pos = meas[meas.dI > 0]
    sl = response_slope(pos.I_DC_pre, pos.dI)
    gain = pos.dI / pos.I_DC_pre
    R = summ.set_index("variant")
    fmt = lambda x: sig2(x, "floor")    # quoted as "at least": round lower bounds down
    m = {"DingShift": f"{-1e3 * delta:.0f}",
         "DingNitPre": _e(NIT_PRE), "DingDnit": _e(DNIT),
         "DingNitRatio": f"{(NIT_PRE + DNIT) / NIT_PRE:.1f}",
         "DingDitPreLo": _e(NIT_PRE / DE_CP[1]), "DingDitPreHi": _e(NIT_PRE / DE_CP[0]),
         "DingGainLo": f"{100 * gain.min():.0f}", "DingGainHi": f"{100 * gain.max():.0f}",
         "DingSlopeLo": f"{sl.min():.2f}", "DingSlopeHi": f"{sl.max():.2f}",
         "DingAlignDV": f"{ALIGN_DV:g}",
         "DingNlev": str(int((df.offset_nm == df.offset_nm.min()).sum())),
         "DingPreTailNom": fmt(R.loc["nominal", "pre_tail_ratio_min"]),
         "DingPreTailMtwo": fmt(R.loc["m_2.0", "pre_tail_ratio_min"])}
    for v, k in (("nominal", "Nom"), ("m_0.5", "Mhalf"), ("m_1.0", "Mone"), ("m_2.0", "Mtwo"),
                 ("no_field_enh", "Srh"), ("sigma_1e-16", "SigLow"), ("gauss_mid", "Gauss")):
        m[f"DingRatio{k}Lo"] = sig2(R.loc[v, "ratio_min"], "floor")
        m[f"DingRatio{k}Hi"] = sig2(R.loc[v, "ratio_max"], "ceil")
    os.makedirs(GEN, exist_ok=True)
    with open(os.path.join(GEN, "numbers_ding.tex"), "w") as f:
        f.write("% generated by src/validation/ding2014.py -- do not edit\n")
        for k, v in m.items():
            f.write(f"\\newcommand{{\\n{k}}}{{{v}}}\n")


def fig_ding(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.analysis.figures import _save
    fig, axs = plt.subplots(1, 2, figsize=(7.16, 2.5))
    ax = axs[0]
    sty = {"DC_pre": ("k", "-"), "DC_1p67M": ("C3", "-"), "500kHz_pre": ("k", "--"),
           "500kHz_1p67M": ("C3", "--")}
    lab = {"DC_pre": "DC, pre", "DC_1p67M": "DC, 16.7 kGy", "500kHz_pre": "500 kHz, pre",
           "500kHz_1p67M": "500 kHz, 16.7 kGy"}
    for k in CURVES:
        V, I = curve(k)
        ax.semilogy(V, I, sty[k][1], color=sty[k][0], lw=1.0, label=lab[k])
    ax.set_xlabel("$V_{GS}$ (V)")
    ax.set_ylabel("$I_D$ (A/$\\mu$m)")
    ax.set_title("(a) Measured, Ding et al. Fig. 4 (digitized)", fontsize=8)
    ax.legend(fontsize=6, frameon=False, loc="upper left")
    ax = axs[1]
    d = df.sort_values("V_meas")
    ok = d.dI_meas_lo > 0                            # change resolved within the shift uncertainty
    ax.errorbar(d.V_meas[ok], d.dI_meas[ok], yerr=[d.dI_meas[ok] - d.dI_meas_lo[ok],
                d.dI_meas_hi[ok] - d.dI_meas[ok]], fmt="s", ms=3.5, color="k", capsize=2,
                label="measured", zorder=5)
    ax.plot(d.V_meas[~ok], d.dI_meas_hi[~ok], "kv", ms=4, mfc="none", label="measured, upper limit",
            zorder=5)
    names = {"nominal": "nominal ($m^*$=0.25)", "m_0.5": "$m^*$=0.5", "m_1.0": "$m^*$=1",
             "m_2.0": "$m^*$=2", "no_field_enh": "plain SRH"}
    for i, (v, nm) in enumerate(names.items()):
        ax.fill_between(d.V_meas, d[f"dI_{v}_gate_lo"], d[f"dI_{v}_gate_hi"], color=f"C{i}",
                        alpha=0.45, lw=0, label=nm)
    ax.set_yscale("log")
    ax.set_ylim(1e-17, 1e-7)
    ax.set_xlabel("$V_{GS}$ (V), measured axis")
    ax.set_ylabel("$\\Delta I$ at fixed electrostatics (A/$\\mu$m)")
    ax.set_title("(b) Current change: frozen trap variants vs. measured", fontsize=8)
    ax.legend(fontsize=5.5, frameon=False, loc="upper left", ncol=2)
    fig.tight_layout()
    _save(fig, "fig_ding")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--analyse", action="store_true")
    a = ap.parse_args()
    if a.run:
        run()
    if a.analyse or not a.run:
        analyse()


if __name__ == "__main__":
    main()
