"""Independent mathematical checks of the derived quantities reported in the paper.

Each check recomputes a published number by a route that shares no code with the pipeline
that produced it: closed-form results instead of numerical ones, physical constants from
scipy.constants instead of src.constants, and vectorized pandas instead of the per-model
loops. A check passes when the two routes agree to the stated tolerance; numbers quoted
in the paper are compared at the rounding the paper uses.

Checks
  A  design-rule algebra (Eq. 2 of the paper) on every row of results/audit_cal*/design_rules
  B  Kane parameters: B <-> reduced mass, and the closed-form local junction current
  C  conversions of measured data: Arrhenius, charge-pumping window, irradiation response
  D  model-form ensemble: combinatorics and every robustness percentage of Table 7

Out: results/math_checks.csv, paper/generated/numbers_math.tex, paper/generated/tab_math.tex
Run: python -m src.analysis.math_checks   (exit code 1 if any check fails)
"""

import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy import constants as sc
from scipy import special

GEN = os.path.join("paper", "generated")
TECHS = ["heterogate", "low_drain_doping", "gd_underlap", "dual_wf_gate"]
AGO_T = [300, 360, 390, 420]
CP_WINDOW = (0.3, 0.6)          # eV, charge-pumping energy window (Groeseneken 1984)
EG_300 = 1.1245                 # eV, the bandgap the calibration used


# ---- closed-form results --------------------------------------------------------------

def suppression(D, I_ref0, a_ref, I_t0, a_t):
    """Eq. (2): S(D) = log10[(I_ref0 + a_ref D) / (I_t0 + a_t D)]."""
    return np.log10((I_ref0 + a_ref * D) / (I_t0 + a_t * D))


def suppression_from_metrics(D, S0, S_inf, D_x):
    """Eq. (2) in the reported metrics only: with I_ref0 = I_t0 10^S0, a_ref = a_t 10^S_inf and
    D_x = I_t0 / a_t,  S(D) = log10[(10^S0 + 10^S_inf D/D_x) / (1 + D/D_x)]."""
    u = np.asarray(D, float) / np.asarray(D_x, float)
    return np.log10((10.0 ** np.asarray(S0, float) + 10.0 ** np.asarray(S_inf, float) * u) / (1.0 + u))


def dit_half_from_metrics(S0, S_inf, D_x, frac=0.5):
    """D_f = D_x (10^S0 - 10^(f S0)) / (10^(f S0) - 10^S_inf); NaN when S_inf >= f S0."""
    r = 10.0 ** (frac * np.asarray(S0, float))
    den = r - 10.0 ** np.asarray(S_inf, float)
    D = np.asarray(D_x, float) * (10.0 ** np.asarray(S0, float) - r) / np.where(den > 0, den, 1.0)
    return np.where((den > 0) & (D > 0), D, np.nan)


def dit_half_closed_form(I_ref0, a_ref, I_t0, a_t, frac=0.5):
    """D at which S(D) = frac * S(0), solved exactly.

    With r = 10^(frac S0), S(D) = frac S0 is linear in D:
        I_ref0 + a_ref D = r (I_t0 + a_t D)  =>  D = (I_ref0 - r I_t0) / (r a_t - a_ref).
    A positive root exists iff S_inf < frac S0 (for S0 > 0); otherwise returns NaN.
    """
    S0 = np.log10(I_ref0 / I_t0)
    r = 10.0 ** (frac * S0)
    den = r * a_t - a_ref
    D = np.where(den > 0, (I_ref0 - r * I_t0) / np.where(den > 0, den, 1.0), np.nan)
    return np.where(D > 0, D, np.nan)


def dS_dD_sign(I_ref0, a_ref, I_t0, a_t):
    """Sign of dS/dD, which is constant in D: d/dD of the ratio has numerator
    a_ref I_t0 - a_t I_ref0, i.e. sign(S_inf - S0)."""
    return np.sign(a_ref * I_t0 - a_t * I_ref0)


def reduced_mass_from_B(B_Vcm, Eg_eV=EG_300):
    """Invert the WKB triangular-barrier result B = pi sqrt(m) Eg^(3/2) / (2 sqrt(2) q hbar)
    with CODATA constants from scipy (independent of src.constants)."""
    Eg = Eg_eV * sc.e
    sqrt_m = B_Vcm * 100.0 * 2.0 * np.sqrt(2.0) * sc.e * sc.hbar / (np.pi * Eg ** 1.5)
    return sqrt_m ** 2 / sc.m_e


def scaled_upper_gamma(s, x):
    """e^x * Gamma(s, x) for s = 1/2 - k (k = 0, 1, 2, ...), x > 0, by downward recurrence
    from Gamma(1/2, x) = sqrt(pi) erfc(sqrt(x)):  Gamma(s, x) = [Gamma(s+1, x) - x^s e^-x] / s.
    Scaling by e^x (erfcx) avoids underflow at x ~ 30."""
    k = int(round(0.5 - s))
    if abs((0.5 - k) - s) > 1e-12 or k < 0:
        raise ValueError("s must be 1/2 - k")
    g = np.sqrt(np.pi) * special.erfcx(np.sqrt(x))      # e^x Gamma(1/2, x)
    a = 0.5
    for _ in range(k):
        a -= 1.0
        g = (g - x ** a) / a
    return g


def local_junction_current(F_max, A, B, V_t, P=2.5):
    """Exact J (A/cm^2) of local Kane BTBT in an abrupt junction whose field falls linearly
    from F_max to 0 over W = 2 V_t / F_max:
        J = q (W / F_max) A int_0^F_max F^P e^(-B/F) dF = q (W / F_max) A B^(P+1) Gamma(-P-1, B/F_max)
    (substitute u = B/F). Requires P = k - 1/2 so that the recurrence above applies."""
    x = B / F_max
    W = 2.0 * V_t / F_max
    return sc.e * (W / F_max) * A * B ** (P + 1) * scaled_upper_gamma(-P - 1, x) * np.exp(-x)


def arrhenius_ea(T, I):
    """E_a (eV) of I = I0 exp(-E_a / kT) by least squares of ln I on 1/kT."""
    x = 1.0 / (sc.k / sc.e * np.asarray(T, float))
    return -np.polyfit(x, np.log(np.asarray(I, float)), 1)[0]


def arrhenius_gain(Ea, T1, T2):
    """Current ratio I(T2)/I(T1) for activation energy Ea (eV)."""
    k = sc.k / sc.e
    return np.exp(Ea * (1.0 / (k * T1) - 1.0 / (k * T2)))


# ---- helpers --------------------------------------------------------------------------

def load_macros():
    """{name: value string} of every \\newcommand{\\nName} in paper/generated/*.tex, keyed
    without the leading 'n' (\\nCalMrLoc -> 'CalMrLoc')."""
    out = {}
    for f in glob.glob(os.path.join(GEN, "*.tex")):
        if os.path.basename(f) == "numbers_math.tex":
            continue
        for m in re.finditer(r"\\newcommand\{\\n(\w+)\}\{(.*)\}\s*$", open(f, encoding="utf-8").read(), re.M):
            out[m.group(1)] = m.group(2)
    return out


def macro_value(s):
    """Float of a macro such as '0.36', '58', '6.2\\times10^{10}', '>10^{13}'."""
    s = s.replace("$", "").replace(" ", "")
    m = re.fullmatch(r"([-+]?[\d.]+)\\times10\^\{([-+]?\d+)\}", s)
    if m:
        return float(m.group(1)) * 10 ** int(m.group(2))
    m = re.fullmatch(r"10\^\{([-+]?\d+)\}", s)
    if m:
        return 10.0 ** int(m.group(1))
    return float(s)


def rounding_tol(s):
    """Half a unit in the last digit the paper shows (relative for mantissa x 10^n)."""
    s = s.replace("$", "")
    mant = s.split("\\times")[0]
    dec = len(mant.split(".")[1]) if "." in mant else 0
    half = 0.5 * 10 ** -dec
    if "\\times" in s:
        return half / abs(float(mant)), True
    return half, False


def matches_macro(value, macro):
    """True if `value` rounds to what the paper prints (with 1e-9 slack for ties)."""
    tol, rel = rounding_tol(macro)
    ref = macro_value(macro)
    return abs(value - ref) <= (tol * abs(ref) if rel else tol) * (1 + 1e-9) + 1e-12


def record(group, check, n, dev, tol, ok, unit=""):
    return dict(group=group, check=check, n=int(n), max_dev=float(dev), tol=float(tol),
                unit=unit, passed=bool(ok))


# ---- A: design-rule algebra -----------------------------------------------------------

def check_design_rules():
    files = sorted(glob.glob(os.path.join("results", "audit_cal*", "design_rules_Vg*.csv")))
    files = [f for f in files if not f.endswith("vs_Vg.csv")]
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    I_t0, a_t, a_ref = d.I_tech0.to_numpy(), d.a_t.to_numpy(), d.a_ref.to_numpy()
    I_ref0 = I_t0 * 10.0 ** d.S_0.to_numpy()          # invert S0 = log10(I_ref0 / I_t0)
    recs = []

    dev = max(np.nanmax(np.abs(suppression(D, I_ref0, a_ref, I_t0, a_t) - d[c].to_numpy()))
              for D, c in ((1e11, "S_1e11"), (1e12, "S_1e12"), (5e12, "S_5e12")))
    recs.append(record("A", "S(D) at 1e11, 1e12, 5e12 from Eq. (2)", 3 * len(d), dev, 1e-9, dev < 1e-9, "dec"))

    # the three reported metrics are sufficient: S(D) = log10[(10^S0 + 10^Sinf D/Dx) / (1 + D/Dx)]
    dev = max(np.nanmax(np.abs(suppression_from_metrics(D, d.S_0, d.S_inf, d.D_cross) - d[c]))
              for D, c in ((1e11, "S_1e11"), (1e12, "S_1e12"), (5e12, "S_5e12")))
    recs.append(record("A", "S(D) from the metrics (S0, S_inf, D_x) alone, Eq. (A.2)", 3 * len(d), dev,
                       1e-9, dev < 1e-9, "dec"))

    dev = np.nanmax(np.abs(np.log10(a_ref / a_t) - d.S_inf))
    recs.append(record("A", "S_inf = log10(a_ref / a_t)", len(d), dev, 1e-9, dev < 1e-9, "dec"))

    dev = np.nanmax(np.abs(I_t0 / a_t / d.D_cross - 1))
    recs.append(record("A", "D_x = I_0 / a (technique device)", len(d), dev, 1e-9, dev < 1e-9, "rel"))

    dev = np.nanmax(np.abs((d.I_btbt + d.I_tat_bulk) / I_t0 - 1))
    recs.append(record("A", "I_0 = I_BTBT + I_bulk-TAT", len(d), dev, 1e-9, dev < 1e-9, "rel"))

    # at D_x the technique current is exactly 2 I_0, so removing all BTBT gains <= 2x
    two = (I_t0 + a_t * d.D_cross) / I_t0
    dev = np.nanmax(np.abs(two - 2))
    recs.append(record("A", "I(D_x) = 2 I_0 (factor-two bound)", len(d), dev, 1e-9, dev < 1e-9, "rel"))

    # at the crossover the suppression has lost at most log10(2) when S_inf <= S0
    loss_x = d.S_0 - suppression_from_metrics(d.D_cross, d.S_0, d.S_inf, d.D_cross)
    m = (d.S_inf <= d.S_0).to_numpy()
    bad = int((loss_x[m] > np.log10(2) + 1e-12).sum() + (loss_x[m] < -1e-12).sum())
    recs.append(record("A", "loss at D_x in [0, log10 2] when S_inf <= S0", m.sum(),
                       float(loss_x[m].max()), np.log10(2), bad == 0, ""))

    Dh = dit_half_closed_form(I_ref0, a_ref, I_t0, a_t)
    lo, hi = 1e6, 1e16                                 # bracket of the bisection
    has_cf = np.isfinite(Dh) & (Dh >= lo) & (Dh <= hi) & (d.S_0.to_numpy() > 0)
    has_bis = d.D_half.notna().to_numpy()
    agree = int((has_cf == has_bis).sum())
    both = has_cf & has_bis
    dev = np.nanmax(np.abs(np.log10(Dh[both] / d.D_half.to_numpy()[both]))) if both.any() else 0.0
    recs.append(record("A", "D_1/2 closed form vs bisection", both.sum(), dev, 2e-6,
                       dev < 2e-6 and agree == len(d), "dec"))
    recs.append(record("A", "D_1/2 exists iff S_inf < S0/2", len(d), len(d) - agree, 0, agree == len(d), "rows"))

    s_num = np.sign(d.S_1e12.to_numpy() - d.S_0.to_numpy())
    s_cf = dS_dD_sign(I_ref0, a_ref, I_t0, a_t)
    s_inf = np.sign(d.S_inf.to_numpy() - d.S_0.to_numpy())
    m = np.abs(d.S_1e12 - d.S_0).to_numpy() > 1e-9
    bad = int(((s_num != s_cf) | (s_cf != s_inf))[m].sum())
    recs.append(record("A", "S(D) monotone, sign(dS/dD) = sign(S_inf - S0)", m.sum(), bad, 0, bad == 0, "rows"))
    return recs, len(d), dev


# ---- B: Kane parameters ---------------------------------------------------------------

def check_kane(M):
    cal = json.load(open(os.path.join("results", "calibration", "btbt_calibration.json")))
    recs = []
    for key, mac in (("local", "CalMrLoc"), ("nonlocal", "CalMrNl")):
        m_r = reduced_mass_from_B(cal[key]["B"])
        dev = abs(m_r / cal[key]["m_r"] - 1)
        ok = dev < 1e-6 and matches_macro(m_r, M[mac])
        recs.append(record("B", f"m_r from calibrated B ({key}) = {m_r:.3f} m0", 1, dev, 1e-6, ok, "rel"))

    c = cal["local"]
    F, J = np.array(cal["data"]["F_Vcm"]), np.array(cal["data"]["J_Acm2"])
    J_cf = local_junction_current(F, c["A"], c["B"], c["V_t"], c["P"])
    from src.calibration.btbt_calibration import j_local
    jloc_dev = float(np.max(np.abs(np.log10(j_local(F, c["A"], c["B"], c["V_t"], P=c["P"]) / J_cf))))
    recs.append(record("B", "local junction J: quadrature vs closed form (incomplete gamma)",
                       len(F), jloc_dev, 1e-4, jloc_dev < 1e-4, "dec"))
    rms = np.sqrt(np.mean(np.log10(J_cf / J) ** 2))
    dev = abs(rms - c["rms_dec"])
    recs.append(record("B", f"calibration rms recomputed in closed form = {rms:.3f} dec", len(F),
                       dev, 1e-3, dev < 1e-3 and matches_macro(rms, M["CalRmsLoc"]) if "CalRmsLoc" in M
                       else dev < 1e-3, "dec"))
    return recs, jloc_dev


# ---- C: conversions of measured data --------------------------------------------------

def check_conversions(M):
    recs = []
    dig = os.path.join("data", "digitized")
    Imin = [pd.read_csv(os.path.join(dig, f"agopian2012_fig7_T{T}.csv")).ID_A.min() for T in AGO_T]
    Ea = arrhenius_ea(AGO_T, Imin)
    recs.append(record("C", f"Der Agopian E_a of I_min = {Ea:.3f} eV", 4, abs(Ea - macro_value(M["AgoEaMin"])), rounding_tol(M["AgoEaMin"])[0],
                       matches_macro(Ea, M["AgoEaMin"]), "eV"))
    dec = np.log10(Imin[-1] / Imin[0])
    recs.append(record("C", f"I_min rise 300->420 K = {dec:.2f} dec", 2, abs(dec - macro_value(M["AgoIminRatio"])),
                       rounding_tol(M["AgoIminRatio"])[0], matches_macro(dec, M["AgoIminRatio"]), "dec"))
    g = arrhenius_gain(Ea, 300.0, 398.15)
    recs.append(record("C", f"trap gain 300->398 K at measured E_a = {g:.1f}x", 1,
                       abs(g - macro_value(M["TrapGainMeas"])), rounding_tol(M["TrapGainMeas"])[0],
                       matches_macro(g, M["TrapGainMeas"]), "x"))
    mod = pd.read_csv(os.path.join("results", "validation", "model_activation_energy.csv"))
    gm = arrhenius_gain(mod[mod.regime == "trap"].Ea_total.max(), 300.0, 398.15)
    recs.append(record("C", f"model trap gain 300->398 K = {gm:.2f}x", 1,
                       abs(gm - macro_value(M["TrapGainModel"])), rounding_tol(M["TrapGainModel"])[0],
                       matches_macro(gm, M["TrapGainModel"]), "x"))

    # charge pumping: sheet density (cm^-2) over the energy window -> D_it (cm^-2 eV^-1)
    from src.validation.ding2014 import NIT_PRE, DNIT
    ratio = (NIT_PRE + DNIT) / NIT_PRE
    ok = matches_macro(ratio, M["DingNitRatio"])
    ok &= matches_macro(NIT_PRE / CP_WINDOW[1], M["DingDitPreLo"]) and matches_macro(NIT_PRE / CP_WINDOW[0], M["DingDitPreHi"])
    recs.append(record("C", f"Ding N_it ratio {ratio:.3f}; D_it,pre = N_it / (0.3-0.6 eV)", 3, 0, 0, ok, ""))
    nit_luong = macro_value(M["LuongNit"])
    lo, hi = nit_luong / CP_WINDOW[1], nit_luong / CP_WINDOW[0]
    recs.append(record("C", f"Luong D_it = {lo:.1e}-{hi:.1e} (paper: of order 1e13)", 1, 0, 0,
                       13 <= np.log10(lo) and np.log10(hi) < 13.5, ""))

    # irradiation response: slope = ln(1 + gain) / ln(N_it ratio), monotone in gain
    sl = [np.log1p(macro_value(M[k]) / 100) / np.log(ratio) for k in ("DingGainLo", "DingGainHi")]
    ok = all(abs(s - macro_value(M[k])) <= 0.006 for s, k in zip(sl, ("DingSlopeLo", "DingSlopeHi")))
    recs.append(record("C", f"response slope from gain bounds = {sl[0]:.3f}-{sl[1]:.3f}", 2,
                       max(abs(s - macro_value(M[k])) for s, k in zip(sl, ("DingSlopeLo", "DingSlopeHi"))),
                       0.006, ok, ""))

    chk = pd.read_csv(os.path.join("results", "validation", "ding2014_check.csv"))
    hi_cols = [c for c in chk.columns if c.startswith("dI_") and c.endswith("_hi") and c != "dI_meas_hi"]
    dev = max(np.nanmax(np.abs(chk[c] / chk[c[:-3] + "_lo"] - CP_WINDOW[1] / CP_WINDOW[0])) for c in hi_cols)
    recs.append(record("C", "predicted dI scales as 1/window: hi/lo = 0.6/0.3", len(hi_cols) * len(chk), dev,
                       1e-9, dev < 1e-9, "rel"))

    summ = pd.read_csv(os.path.join("results", "validation", "ding2014_summary.csv"))
    ok_rows = chk.dI_meas_hi > 0
    bad = 0
    for _, r in summ.iterrows():
        ratio_min = (chk[f"dI_{r.variant}_gate_min"][ok_rows] / chk.dI_meas_hi[ok_rows]).min()
        bad += int(abs(ratio_min / r.ratio_min - 1) > 1e-9 or (ratio_min > 1) != bool(r.excluded))
    tags = {"nominal": "Nom", "m_0.5": "Mhalf", "m_1.0": "Mone", "m_2.0": "Mtwo",
            "no_field_enh": "Srh", "sigma_1e-16": "SigLow", "gauss_mid": "Gauss"}
    # the paper quotes "at least <Lo>": Lo must not exceed the true minimum (and Hi must not
    # undercut the true maximum), each within two significant figures
    for _, r in summ.iterrows():
        lo = macro_value(M[f"DingRatio{tags[r.variant]}Lo"])
        hi = macro_value(M[f"DingRatio{tags[r.variant]}Hi"])
        bad += int(not (0.9 * r.ratio_min <= lo <= r.ratio_min and r.ratio_max <= hi <= 1.12 * r.ratio_max))
    recs.append(record("C", "irradiation exclusion: ratios recomputed, rule ratio_min > 1, "
                       "quoted bounds conservative", 2 * len(summ), bad, 0, bad == 0, "rows"))
    return recs


# ---- D: model-form ensemble -----------------------------------------------------------

def claims_vectorized(e):
    """The seven claims of Table 7, recomputed with pivots instead of per-model loops."""
    p = e.pivot_table(index=["path", "params", "it"], columns=["Vg", "device"],
                      values=["Dx", "S0", "S_1e+12", "S_inf"])
    Dx2, Dx1 = p["Dx"][-0.2], p["Dx"][-1.0]
    loss2 = p["S0"][-0.2] - p["S_1e+12"][-0.2]
    loss1 = p["S0"][-1.0] - p["S_1e+12"][-1.0]
    Si = p["S_inf"][-1.0]
    return pd.DataFrame({
        "C1": (Dx2[TECHS] < 1e11).all(axis=1),
        "C2": (Dx2[TECHS] < 1e12).sum(axis=1) >= 3,
        "C3": (loss2[TECHS] >= 0.5).all(axis=1),
        "C4": (loss1[["heterogate", "gd_underlap", "dual_wf_gate"]] < 0.1).all(axis=1),
        "C5": Dx1[TECHS].idxmin(axis=1) == "low_drain_doping",
        "C6": Si[TECHS].idxmin(axis=1) == "dual_wf_gate",
        "C7": Si[TECHS].idxmax(axis=1) == "low_drain_doping",
    })


def check_ensemble(M):
    e = pd.read_csv(os.path.join("results", "model_form", "ensemble.csv"))
    recs = []
    n_path, n_par, n_it = e.path.nunique(), e.params.nunique(), e.it.nunique()
    n_mod = len(e.groupby(["path", "params", "it"]))
    ok = (n_mod == n_path * n_par * n_it == int(M["EnsN"]) and len(e) == n_mod * e.device.nunique() * e.Vg.nunique())
    recs.append(record("D", f"{n_path} x {n_par} x {n_it} = {n_mod} models; {len(e)} rows = models x 4 devices x 3 biases",
                       len(e), 0, 0, ok, ""))
    weak = {"m_1.0", "m_2.0", "no_field_enh"}
    n_strong = n_path * n_par * (n_it - len(weak))
    recs.append(record("D", f"Hurkx-type subset: {n_path} x {n_par} x {n_it - len(weak)} = {n_strong}", 1, 0, 0,
                       n_strong == int(M["EnsNstrong"]), ""))

    cv = claims_vectorized(e)
    cl = pd.read_csv(os.path.join("results", "model_form", "claims.csv")).set_index(["path", "params", "it"])
    cl.columns = [c.split("_")[0] for c in cl.columns]
    mism = int((cv.sort_index() != cl.sort_index()[cv.columns]).to_numpy().sum())
    recs.append(record("D", "claims per model: vectorized vs pipeline", cv.size, mism, 0, mism == 0, "cells"))

    it = cv.index.get_level_values("it")
    sets = {"Ens": cv, "EnsEnh": cv[it != "no_field_enh"], "EnsStrong": cv[~it.isin(weak)]}
    bad, n = 0, 0
    for tag, s in sets.items():
        for i, c in enumerate(cv.columns):
            key = f"Claim{tag}{'ABCDEFG'[i]}"
            if key in M:
                n += 1
                bad += int(abs(100 * s[c].mean() - macro_value(M[key])) > 0.5 + 1e-9)
    recs.append(record("D", "robustness percentages of Table 7 and Sec. 4.9", n, bad, 0, bad == 0, "values"))
    return recs


# ---- output ---------------------------------------------------------------------------

TEX = [  # plain-text check names -> LaTeX, applied in order
    ("S(D) at 1e11, 1e12, 5e12", "$S(D)$ at $D=10^{11}$, $10^{12}$, $5\\times10^{12}$"),
    ("S(D) from the metrics", "$S(D)$ from the metrics"), ("(S0, S_inf, D_x)", "($S_0$, $\\Sinf$, $\\Dx$)"),
    ("S(D) monotone, sign(dS/dD) = sign(S_inf - S0)", "$S(D)$ monotone, $\\mathrm{sgn}\\,dS/dD=\\mathrm{sgn}(\\Sinf-S_0)$"),
    ("D_1/2 exists iff S_inf < S0/2", "$D_{1/2}$ exists iff $\\Sinf<S_0/2$"),
    ("loss at D_x in [0, log10 2] when S_inf <= S0", "Loss at $\\Dx$ in $[0,\\log_{10}2]$ if $\\Sinf\\le S_0$"),
    ("S_inf = log10(a_ref / a_t)", "$\\Sinf=\\log_{10}(a^{\\mathrm{ref}}/a^k)$"),
    ("D_x = I_0 / a (technique device)", "$\\Dx=I_0^k/a^k$"),
    ("I_0 = I_BTBT + I_bulk-TAT", "$I_0=I_{\\mathrm{BTBT}}+I_{\\mathrm{bulk\\,TAT}}$"),
    ("I(D_x) = 2 I_0 (factor-two bound)", "$I^k(\\Dx)=2I_0^k$ (factor-two bound)"),
    ("D_1/2 closed form vs bisection", "$D_{1/2}$: closed form, Eq.~(\\ref{eq:Dhalf}), vs.\\ bisection"),
    ("m_r from calibrated B", "$m_r$ from calibrated $B$"), (" m0", "\\,$m_0$"),
    ("local junction J: quadrature vs closed form (incomplete gamma)",
     "Local junction $J$: quadrature vs.\\ closed form, Eq.~(\\ref{eq:Jcf})"),
    ("calibration rms recomputed in closed form", "Calibration rms via Eq.~(\\ref{eq:Jcf})"),
    ("Eq. (A.2)", "Eq.~(\\ref{eq:Smetrics})"), ("Eq. (2)", "Eq.~(\\ref{eq:S})"),
    ("Der Agopian E_a of I_min", "Der Agopian $E_a$ of $I_{\\min}$"), ("I_min rise", "$I_{\\min}$ rise"),
    ("at measured E_a", "at measured $E_a$"), ("Ding N_it ratio", "Ding $N_{it}$ ratio"),
    ("D_it,pre = N_it / (0.3-0.6 eV)", "$D_{it}=N_{it}/\\Delta E$"), ("Luong D_it", "Luong $D_{it}$"),
    ("predicted dI scales as 1/window: hi/lo = 0.6/0.3", "Predicted $\\Delta I\\propto1/\\Delta E$"),
    ("rule ratio_min > 1", "rule ratio $>1$"), ("Hurkx-type subset", "Hurkx-type subset ($m^*\\le0.5$)"),
    ("vectorized vs pipeline", "independent recount vs.\\ pipeline"),
    ("->", "$\\to$"), (" x ", " $\\times$ "), ("1e+13", "$10^{13}$"), ("e+13", "$\\times10^{13}$"),
    ("1e13", "$10^{13}$"), ("58.2x", "58.2$\\times$"), ("2.16x", "2.16$\\times$"), ("= 0.", "$=$ 0."),
]


def to_tex(s):
    for a, b in TEX:
        s = s.replace(a, b)
    s = re.sub(r"(\d|\$)-(\d)", r"\1--\2", s)          # numeric ranges get an en dash
    s = s[0].upper() + s[1:] if s[0].isalpha() else s
    return s.replace("_", "\\_") if "$" not in s else s

def write_outputs(recs, nrows, dhalf_dev, jloc_dev):
    df = pd.DataFrame(recs)
    os.makedirs("results", exist_ok=True)
    df.to_csv(os.path.join("results", "math_checks.csv"), index=False)

    def e(v):
        if v == 0:
            return "0"
        m, x = f"{v:.0e}".split("e")
        return f"{m}\\times10^{{{int(x)}}}"

    N = {"MathNchecks": str(len(df)), "MathNpass": str(int(df.passed.sum())),
         "MathNrules": str(nrows), "MathDhalfDev": e(dhalf_dev), "MathJlocDev": e(jloc_dev)}
    with open(os.path.join(GEN, "numbers_math.tex"), "w", encoding="utf-8") as f:
        f.write("% generated by src/analysis/math_checks.py -- do not edit\n")
        for k, v in N.items():
            f.write(f"\\newcommand{{\\n{k}}}{{{v}}}\n")

    group_name = {"A": "Design-rule algebra", "B": "Kane parameters",
                  "C": "Conversions of measured data", "D": "Model-form ensemble"}
    lines = ["% generated by src/analysis/math_checks.py -- do not edit",
             "\\begin{tabular}{@{}p{0.62\\columnwidth}rl@{}}", "\\toprule",
             "Check & $n$ & Result \\\\", "\\midrule"]
    for g, sub in df.groupby("group"):
        lines.append(f"\\multicolumn{{3}}{{@{{}}l}}{{\\emph{{{g}. {group_name[g]}}}}} \\\\")
        for _, r in sub.iterrows():
            txt = to_tex(r.check)
            if r.unit in ("rows", "cells", "values", ""):
                res = "pass" if r.passed else "FAIL"
            else:
                res = f"{'pass' if r.passed else 'FAIL'}, ${e(r.max_dev) if r.max_dev else '0'}$"
            lines.append(f"{txt} & {r.n} & {res} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    with open(os.path.join(GEN, "tab_math.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return df


def main():
    M = load_macros()
    a, nrows, dhalf_dev = check_design_rules()
    b, jloc_dev = check_kane(M)
    recs = a + b + check_conversions(M) + check_ensemble(M)
    df = write_outputs(recs, nrows, dhalf_dev, jloc_dev)
    with pd.option_context("display.width", 200, "display.max_colwidth", 90):
        print(df[["group", "check", "n", "max_dev", "tol", "passed"]].to_string(index=False))
    print(f"{int(df.passed.sum())}/{len(df)} checks pass")
    return 0 if df.passed.all() else 1


if __name__ == "__main__":
    sys.exit(main())
