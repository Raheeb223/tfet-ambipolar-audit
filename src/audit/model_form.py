"""Model-form (structural) uncertainty of the audit, separated from parameter uncertainty
(src.uq.btbt_uq). Every factor below is a different but admissible model, not a fitted value;
nothing here is fitted to validation data.

Factors (all on the cached calibrated-audit electrostatics, traps uncharged so I = I0 + a D_it):
  * BTBT parameters (results/calibration/btbt_alternatives.json, junction data only):
    nominal P = 2.5; P = 2 and 3 refits; a fit on the low-field half of the data (its error on
    the held-out high-field half measures extrapolation); one fit per data source.
  * Nonlocal path model: rows/columns max (production), rows + columns sum, gradient-at-start
    paths (nonlocal_field_paths.field_direction_terms, the usual commercial approximation) and
    the angle-complete upper bound (optimal_direction_terms).
  * Interface-trap model (IT_VARIANTS): capture cross-sections (equal and unequal), Hurkx
    tunneling mass, no field enhancement (plain SRH bound), trap spectra (U shape, Gaussians at
    and off midgap), implant damage at the drain-side gate edge.
The full factorial of these factors gives envelopes of S, D_x and S_inf at 300 K and the
fraction of members that satisfy each central claim (CLAIMS). The fraction is a robustness
count over admissible models, not a probability.

Run:  python -m src.audit.model_form [--parts slopes btbt ensemble validation]
Out:  results/model_form/*.csv
"""

import argparse
import itertools
import json
import os

import numpy as np
import pandas as pd

from src.audit import run_audit as ra
from src.config import load_yaml
from src.devices import tfet_2d as t
from src.materials import silicon
from src.physics import kane_btbt
from src.physics import nonlocal_field_paths as fp

OUT = os.path.join("results", "model_form")
AUDIT = "configs/audit_calibrated.yaml"
ALT = os.path.join("results", "calibration", "btbt_alternatives.json")
VGS = (-1.0, -0.6, -0.2)
TEMPS = (300.0, 358.15, 398.15)
TAU = 1e-7
TECHS = ("heterogate", "low_drain_doping", "gd_underlap", "dual_wf_gate")
FD = dict(n_bins=24, h=1e-8)                       # converged: tests/test_field_paths.py, fd_convergence.csv
OPT = dict(n_angles=36, cell=5e-8, h=2e-8)
TAIL_EV = 0.05                                     # band-tail gap reduction (eV), generous
EG300 = 1.1245
C_AUGER = (2.8e-31, 9.9e-32)                       # Cn, Cp (cm^6/s), Dziewior & Schmid 1977
WIN = 3e-6                                         # path window starts 30 nm before the drain-side gate edge

# Interface-trap model variants (ASSUMPTIONS; ranges from the Si/oxide literature cited in the paper):
#  sigma 1e-16..1e-14 cm^2 (Pb-type centres), sigma_n/sigma_p = 100 either way, Hurkx mass 0.1-0.5,
#  Gaussian spectra (width 0.05 eV) at midgap (the measured level of Luong 2017) and +-0.3 eV,
#  10x damage over 10 nm at the drain junction (Luong 2017: finished TFETs ~10x capacitors).
IT_VARIANTS = {
    "nominal": {},
    "sigma_1e-14": {"sigma_cm2": 1e-14},
    "sigma_1e-16": {"sigma_cm2": 1e-16},
    "sigma_n100p": {"sigma_n_cm2": 1e-14, "sigma_p_cm2": 1e-16},
    "sigma_p100n": {"sigma_n_cm2": 1e-16, "sigma_p_cm2": 1e-14},
    "m_0.1": {"m_tunnel": 0.1},
    "m_0.5": {"m_tunnel": 0.5},
    # weak field enhancement: heavier tunneling masses, a proxy for a less efficient (e.g.
    # multiphonon-limited) trap-to-band step; they raise the trap activation energy towards the
    # measured 0.43-0.45 eV (diagnostic: the match is read from validation data, never fitted)
    "m_1.0": {"m_tunnel": 1.0},
    "m_2.0": {"m_tunnel": 2.0},
    "u_shape": {"profile": "u_shape"},
    "gauss_mid": {"profile": "gaussian", "gauss_center_eV": 0.0, "gauss_width_eV": 0.05, "n_levels": 81},
    "gauss_up0.3": {"profile": "gaussian", "gauss_center_eV": 0.3, "gauss_width_eV": 0.05, "n_levels": 81},
    "gauss_dn0.3": {"profile": "gaussian", "gauss_center_eV": -0.3, "gauss_width_eV": 0.05, "n_levels": 81},
    "damage_x10": {"damage_ratio": 10.0, "damage_length_cm": 1e-6},
    "damage_x10_5nm": {"damage_ratio": 10.0, "damage_length_cm": 5e-7},
    "damage_x10_20nm": {"damage_ratio": 10.0, "damage_length_cm": 2e-6},
    "gate_only": {"gate_only": True},
    "no_field_enh": {"field_enhanced": False},
}
# variants that change only the scale of a (D_it axis), not its device or bias dependence
SCALE_ONLY = {"sigma_1e-14", "sigma_1e-16"}


def _audit():
    audit = load_yaml(AUDIT)
    ra.CACHE = audit["cache_dir"]
    return audit, json.load(open(audit["calibration"]))


def _sol(sols, Vg):
    return [s for s in sols if np.isclose(s.Vg, Vg)][0]


def it_config(cfg, variant):
    c = ra.scenario_config(cfg, "nonlocal", 1.0, None)        # D_it = 1 -> I_it is the slope a
    c["physics"]["interface_traps"].update(IT_VARIANTS[variant])
    return c


# ------------------------------------------------------------------ interface slopes
def slopes():
    audit, _ = _audit()
    rows = []
    for d, T in itertools.product(audit["devices"], TEMPS):
        cfg, sols = ra.load_solutions(d, T)
        cb = ra.scenario_config(cfg, "nonlocal", None, TAU)
        for Vg in VGS:
            s = _sol(sols, Vg)
            F = t.field_magnitude(s)
            r, _ = t.generation(s, cb, F=F)
            rows.append(dict(device=d, T_K=T, Vg=Vg, variant="bulk_tat", a=r["I_tat_bulk"]))
            for v in IT_VARIANTS:
                r, _ = t.generation(s, it_config(cfg, v), F=F)
                rows.append(dict(device=d, T_K=T, Vg=Vg, variant=v, a=r["I_it"]))
        print("slopes", d, T, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "it_slopes.csv"), index=False)
    return df


# ------------------------------------------------------------------ BTBT path x parameters
def param_sets(cal):
    alt = json.load(open(ALT))["nonlocal"]
    A, B = cal["nonlocal"]["A"], cal["nonlocal"]["B"]
    out = {"nominal": (A, B, 2.5)}
    for k, v in alt.items():
        out[k] = (v["A"], v["B"], v["P"])
    # band tails (ASSUMPTION, approximate): an effective gap reduced by TAIL_EV, B ~ Eg^1.5
    out["band_tail"] = (A, B * ((EG300 - TAIL_EV) / EG300) ** 1.5, 2.5)
    return out


def _I(terms, A, B, P):
    F, w, _ = terms
    ok = F > 0
    return float(np.sum(w[ok] * A * F[ok] ** (P - 1.0) * np.exp(-B / F[ok]))) if F.size else 0.0


def btbt():
    """Nonlocal BTBT current per device, Vg, path model and parameter set (300 K), and per T for
    the production path model and nominal parameters."""
    audit, cal = _audit()
    params = param_sets(cal)
    Eg300 = float(silicon.bandgap(300.0))
    rows, conv = [], []
    for d in audit["devices"]:
        for T in TEMPS:
            cfg, sols = ra.load_solutions(d, T)
            c = ra.scenario_config(cfg, "nonlocal", None, None, cal)
            x0 = t.gate_span(c)[1] - WIN
            for Vg in VGS:
                s = _sol(sols, Vg)
                tm = t.btbt_terms(s, c)
                path_terms = {"rows": tm["nonlocal_x"], "cols": tm["nonlocal_y"]}
                if T == 300.0:
                    path_terms["gradient_start"] = fp.field_direction_terms(s, c, x_win=(x0, None), **FD)
                    path_terms["angle_complete"] = fp.optimal_direction_terms(
                        s, c, x_win=(x0, s.x.max()), A_sel=cal["nonlocal"]["A"], B_sel=cal["nonlocal"]["B"], **OPT)
                for pname, (A, B, P) in params.items():
                    if T != 300.0 and pname != "nominal":
                        continue
                    At, Bt = kane_btbt.kane_params_at_T(A, B, Eg300, float(silicon.bandgap(T)))
                    I = {k: _I(v, At, Bt, P) for k, v in path_terms.items()}
                    out = {"rows_cols_max": max(I["rows"], I["cols"]), "rows_cols_sum": I["rows"] + I["cols"]}
                    for k in ("gradient_start", "angle_complete"):
                        if k in I:
                            out[k] = I[k]
                    for path, val in out.items():
                        rows.append(dict(device=d, T_K=T, Vg=Vg, path=path, params=pname, I_btbt=val))
                if T == 300.0 and d in ("reference", "low_drain_doping") and Vg in (-1.0, -0.2):
                    A, B = cal["nonlocal"]["A"], cal["nonlocal"]["B"]
                    for nb_, h in ((12, 1e-8), (24, 2e-8), (24, 1e-8), (36, 5e-9)):
                        conv.append(dict(device=d, Vg=Vg, model="gradient_start", n=nb_, h_nm=h * 1e7,
                                         I=fp.current(fp.field_direction_terms(s, c, n_bins=nb_, h=h, x_win=(x0, None)), A, B)))
                    for na, cell, h in ((18, 5e-8, 2e-8), (36, 5e-8, 2e-8), (36, 2.5e-8, 1e-8)):
                        conv.append(dict(device=d, Vg=Vg, model="angle_complete", n=na, h_nm=h * 1e7,
                                         cell_nm=cell * 1e7,
                                         I=fp.current(fp.optimal_direction_terms(s, c, n_angles=na, cell=cell, h=h,
                                                                                 x_win=(x0, s.x.max()), A_sel=A, B_sel=B), A, B)))
            print("btbt", d, T, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "btbt_variants.csv"), index=False)
    pd.DataFrame(conv).to_csv(os.path.join(OUT, "path_convergence.csv"), index=False)
    return df


# ------------------------------------------------------------------ ensemble and claims
def metrics(I0, a, D_list=(1e11, 1e12)):
    """Per-technique metrics from I0[dev] (trap-free current) and a[dev] (interface slope)."""
    out = {}
    for k in TECHS:
        m = dict(S0=np.log10(I0["reference"] / I0[k]), Dx=I0[k] / a[k],
                 S_inf=np.log10(a["reference"] / a[k]))
        for D in D_list:
            m[f"S_{D:.0e}"] = np.log10((I0["reference"] + a["reference"] * D) / (I0[k] + a[k] * D))
        out[k] = m
    return out


CLAIMS = {
    # Vg = -0.2 V (near the off state)
    "C1_Dx02_below_1e11_all": lambda m: all(m[-0.2][k]["Dx"] < 1e11 for k in TECHS),
    "C2_Dx02_below_1e12_3of4": lambda m: sum(m[-0.2][k]["Dx"] < 1e12 for k in TECHS) >= 3,
    "C3_loss02_1e12_ge_0p5_all": lambda m: all(m[-0.2][k]["S0"] - m[-0.2][k]["S_1e+12"] >= 0.5 for k in TECHS),
    # Vg = -1 V
    "C4_m1V_HG_UL_DWF_loss_lt_0p1": lambda m: all(m[-1.0][k]["S0"] - m[-1.0][k]["S_1e+12"] < 0.1
                                                  for k in ("heterogate", "gd_underlap", "dual_wf_gate")),
    "C5_m1V_LDD_lowest_Dx": lambda m: min(TECHS, key=lambda k: m[-1.0][k]["Dx"]) == "low_drain_doping",
    "C6_Sinf_DWF_smallest": lambda m: min(TECHS, key=lambda k: m[-1.0][k]["S_inf"]) == "dual_wf_gate",
    "C7_Sinf_LDD_largest": lambda m: max(TECHS, key=lambda k: m[-1.0][k]["S_inf"]) == "low_drain_doping",
}


def ensemble(T=300.0):
    bt = pd.read_csv(os.path.join(OUT, "btbt_variants.csv"))
    sl = pd.read_csv(os.path.join(OUT, "it_slopes.csv"))
    bt, sl = bt[bt.T_K == T], sl[sl.T_K == T]
    bulk = sl[sl.variant == "bulk_tat"].set_index(["device", "Vg"]).a
    rows, claims = [], []
    for (path, pname), g in bt.groupby(["path", "params"]):
        Ib = g.set_index(["device", "Vg"]).I_btbt
        for v in IT_VARIANTS:
            a = sl[sl.variant == v].set_index(["device", "Vg"]).a
            m = {}
            for Vg in VGS:
                I0 = {d: Ib[(d, Vg)] + bulk[(d, Vg)] for d in ("reference",) + TECHS}
                av = {d: a[(d, Vg)] for d in ("reference",) + TECHS}
                m[Vg] = metrics(I0, av)
                for k, mm in m[Vg].items():
                    rows.append(dict(path=path, params=pname, it=v, Vg=Vg, device=k, **mm))
            claims.append(dict(path=path, params=pname, it=v, **{c: bool(f(m)) for c, f in CLAIMS.items()}))
    ens = pd.DataFrame(rows)
    cl = pd.DataFrame(claims)
    ens.to_csv(os.path.join(OUT, "ensemble.csv"), index=False)
    cl.to_csv(os.path.join(OUT, "claims.csv"), index=False)
    return ens, cl


def one_at_a_time(ens):
    """Range of each metric when one factor moves away from nominal and the others stay nominal."""
    nom = dict(path="rows_cols_max", params="nominal", it="nominal")
    rows = []
    for factor in ("path", "params", "it"):
        others = {k: v for k, v in nom.items() if k != factor}
        sub = ens[(ens[list(others)] == pd.Series(others)).all(axis=1)]
        for (Vg, dev), g in sub.groupby(["Vg", "device"]):
            for met in ("S0", "S_1e+11", "S_1e+12", "Dx", "S_inf"):
                rows.append(dict(factor=factor, Vg=Vg, device=dev, metric=met,
                                 nominal=float(g[g[factor] == nom[factor]][met].iloc[0]),
                                 lo=float(g[met].min()), hi=float(g[met].max()),
                                 lo_at=g.loc[g[met].idxmin(), factor], hi_at=g.loc[g[met].idxmax(), factor]))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT, "one_at_a_time.csv"), index=False)
    return out


def temperature():
    """Per interface variant: activation energy of the trap slope (300-398 K) and the ratio
    a(398)/a(300), and D_x at 398 K with the production BTBT model."""
    sl = pd.read_csv(os.path.join(OUT, "it_slopes.csv"))
    bt = pd.read_csv(os.path.join(OUT, "btbt_variants.csv"))
    bt = bt[(bt.path == "rows_cols_max") & (bt.params == "nominal")]
    kB = 8.617333e-5
    rows = []
    for (v, d, Vg), g in sl.groupby(["variant", "device", "Vg"]):
        g = g.sort_values("T_K")
        Ea = float(-np.polyfit(1.0 / (kB * g.T_K), np.log(g.a), 1)[0])
        row = dict(variant=v, device=d, Vg=Vg, Ea=Ea, ratio_398_300=float(g.a.iloc[-1] / g.a.iloc[0]))
        if v != "bulk_tat":
            for T in (300.0, 398.15):
                b = bt[(bt.device == d) & np.isclose(bt.Vg, Vg) & np.isclose(bt.T_K, T)].I_btbt.iloc[0]
                bk = sl[(sl.variant == "bulk_tat") & (sl.device == d) & np.isclose(sl.Vg, Vg) & np.isclose(sl.T_K, T)].a.iloc[0]
                a = g[np.isclose(g.T_K, T)].a.iloc[0]
                row[f"Dx_{T:.0f}K"] = (b + bk) / a
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT, "it_temperature.csv"), index=False)
    return out


# ------------------------------------------------------------------ measured-device checks
def validation():
    """Path models and interface variants on the cached measured-device solutions (no refit):
    Wan 2011 ambipolar (rms and bias per offset) and Biswas 2012 on-state (mean shortfall)."""
    from src.analysis import extval_figs as ev
    from src.analysis import validation_figs as vf
    from src.validation import biswas2012 as bw
    from src.validation import wan2011 as wn
    cal = json.load(open(os.path.join("results", "calibration", "btbt_calibration.json")))
    A, B = cal["nonlocal"]["A"], cal["nonlocal"]["B"]
    rows = []
    for which, spec in wn.DEVICES.items():
        for off in wn.OFFSETS_NM:
            for L in spec["L_IN_nm"]:
                cfg, sols = t.load_saved(wn.cache_path(which, L, off))
                x0 = t.gate_span(cfg)[1] - WIN
                for s in sols:
                    r = dict(device=which, L_IN_nm=L, offset_nm=off, Vg=s.Vg)
                    r["I_gradient_start"] = fp.current(fp.field_direction_terms(s, cfg, x_win=(x0, None), halves=1, **FD), A, B)
                    r["I_angle_complete"] = fp.current(fp.optimal_direction_terms(
                        s, cfg, x_win=(x0, s.x.max()), halves=1, A_sel=A, B_sel=B, **OPT), A, B)
                    F = t.field_magnitude(s)
                    for v in IT_VARIANTS:
                        c = t.variant(cfg)
                        it = c["physics"]["interface_traps"]
                        it.update(enabled=True, D_it_cm2eV=1.0)
                        it.update(IT_VARIANTS[v])
                        c["physics"]["tat_bulk"]["enabled"] = False
                        rr, _ = t.generation(s, c, F=F)
                        r[f"a_{v}"] = rr["I_it"]
                    rows.append(r)
                print("wan", which, off, L, flush=True)
    wan = pd.DataFrame(rows)
    wan.to_csv(os.path.join(OUT, "wan_variants.csv"), index=False)
    sim = pd.read_csv(vf.SIM)
    res = []
    keys = ["device", "L_IN_nm", "offset_nm", "Vg"]
    m = sim.merge(wan, on=keys, how="left")
    for path in ("rows_cols_max", "gradient_start", "angle_complete"):
        for v in IT_VARIANTS:
            s2 = m.copy()
            if path != "rows_cols_max":
                s2["I_btbt_nl"] = s2[f"I_{path}"]
            s2["i_it_per_D"] = s2[f"a_{v}"]
            r = vf.residuals(s2)
            r = r[(r.model == "nonlocal")]
            r["path"], r["it"] = path, v
            res.append(r)
    pd.concat(res).to_csv(os.path.join(OUT, "wan_residuals.csv"), index=False)

    rows = []
    for off in bw.OFFSETS_NM:
        cfg, sols = t.load_saved(bw.cache_path(off))
        for s in sols:
            if s.Vg < ev.VG_MIN_ON - 1e-9:
                continue
            rows.append(dict(offset_nm=off, Vd=s.Vd, Vg=s.Vg,
                             I_gradient_start=fp.current(fp.field_direction_terms(s, cfg, x_win=(-WIN, 2 * WIN), halves=1, **FD), A, B),
                             I_angle_complete=fp.current(fp.optimal_direction_terms(
                                 s, cfg, x_win=(-WIN, 2 * WIN), halves=1, A_sel=A, B_sel=B, **OPT), A, B)))
        print("biswas", off, flush=True)
    bis = pd.DataFrame(rows)
    bis.to_csv(os.path.join(OUT, "biswas_variants.csv"), index=False)
    sim = pd.read_csv(ev.BIS_SIM)
    m = sim.merge(bis, on=["offset_nm", "Vd", "Vg"], how="left")
    res = []
    for path in ("rows_cols_max", "gradient_start", "angle_complete"):
        s2 = m.copy()
        if path != "rows_cols_max":
            on = s2.Vg >= ev.VG_MIN_ON - 1e-9
            s2.loc[on, "I_btbt_nl"] = s2.loc[on, f"I_{path}"]
        r = ev.biswas_residuals(s2)
        r = r[r.model == "nonlocal"]
        r["path"] = path
        res.append(r)
    pd.concat(res).to_csv(os.path.join(OUT, "biswas_residuals.csv"), index=False)


def spatial():
    """Where along the interface the (nominal) trap current is generated, 300 K, D_it = 1e12.
    Writes it_spatial.csv: fraction within 10 nm of the source-side (x = 0) and of the drain-side
    (x = L_gate) gate edge, the peak position and the radius around the peak holding 80 %.
    Generation near the SOURCE edge is partly an artefact of tying electrons to the drain
    quasi-Fermi level (the source can also supply electrons), so the paper bounds D_x with the
    drain-side part only: D_x / (1 - frac_src10)."""
    audit, _ = _audit()
    cap = {}
    orig = t.interface_generation

    def grab(sol, cfg, *args, **kw):
        g = orig(sol, cfg, *args, per_x=True)
        cap["g"] = g
        return g.sum()

    rows = []
    t.interface_generation = grab
    try:
        for d in audit["devices"]:
            cfg, sols = ra.load_solutions(d, 300.0)
            c = it_config(cfg, "nominal")
            Lg = cfg["geometry"]["L_gate_cm"]
            for Vg in VGS:
                s = _sol(sols, Vg)
                t.generation(s, c)
                g, x = cap["g"], s.x
                xp = x[np.argmax(g)]
                order = np.argsort(np.abs(x - xp))
                r80 = np.abs(x - xp)[order][np.searchsorted(np.cumsum(g[order]) / g.sum(), 0.8)]
                rows.append(dict(device=d, Vg=Vg, frac_src10=g[x < 1e-6].sum() / g.sum(),
                                 frac_drn10=g[np.abs(x - Lg) < 1e-6].sum() / g.sum(),
                                 peak_from_drain_edge_nm=1e7 * (xp - Lg), r80_nm=1e7 * r80))
    finally:
        t.interface_generation = orig
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT, "it_spatial.csv"), index=False)
    print(out.round(2).to_string(index=False))
    return out


def floors():
    """Order-of-magnitude floors of mechanisms the audit omits (300 K):
      * Auger generation (Cn n + Cp p)(ni^2 - np), integrated where np < ni^2;
      * gate leakage: the current density J* (A/cm^2) through both gates over the channel whose
        full collection at the drain would equal the interface-trap current at 1e11.
    Band tails are a BTBT parameter set (param_sets: band_tail)."""
    audit, _ = _audit()
    sl = pd.read_csv(os.path.join(OUT, "it_slopes.csv"))
    rows = []
    for d in audit["devices"]:
        cfg, sols = ra.load_solutions(d, 300.0)
        ni = float(silicon.intrinsic_density(300.0))
        for Vg in VGS:
            s = _sol(sols, Vg)
            G = (C_AUGER[0] * s.n + C_AUGER[1] * s.p) * np.clip(ni**2 - s.n * s.p, 0.0, None)
            I_aug = t._per_um_base(1.602176634e-19 * np.sum(G * s.vol))
            a = sl[(sl.device == d) & np.isclose(sl.Vg, Vg) & np.isclose(sl.T_K, 300.0) & (sl.variant == "nominal")].a.iloc[0]
            A_gate = 2 * cfg["geometry"]["L_gate_cm"] * 1e-4             # both gates, 1 um width (cm^2)
            rows.append(dict(device=d, Vg=Vg, I_auger=I_aug, I_it_1e11=a * 1e11,
                             J_gate_star=a * 1e11 / A_gate))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT, "floors.csv"), index=False)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="+", default=["slopes", "btbt", "ensemble", "floors", "spatial"])
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if "slopes" in a.parts:
        slopes()
    if "btbt" in a.parts:
        btbt()
    if "ensemble" in a.parts:
        ens, cl = ensemble()
        one_at_a_time(ens)
        temperature()
        print(cl.drop(columns=["path", "params", "it"]).mean().round(3).to_string())
    if "floors" in a.parts:
        floors()
    if "spatial" in a.parts:
        spatial()
    if "validation" in a.parts:
        validation()


if __name__ == "__main__":
    main()
