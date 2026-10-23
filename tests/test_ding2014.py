"""Measured-side logic of the Ding 2014 check: the shift of the pulsed curve is removed from
the DC curve, so a pure electrostatic shift gives no trap-current change, and a known
trap-current increase is recovered."""

import numpy as np
import pandas as pd

from src.validation import ding2014 as d

V = np.linspace(1.0, 2.0, 201)


def btbt(V, shift=0.0):
    return 1.5e-13 * 10 ** ((V - shift - 1.45) / 0.2)


def tail(V, shift=0.0):
    return 1e-13 * 10 ** ((V - shift - 1.2) / 0.35)


def curves(k_trap, shift=-0.05):
    keep = lambda I: (V[I > 1e-14], I[I > 1e-14])
    return {"500kHz_pre": keep(btbt(V)), "500kHz_1p67M": keep(btbt(V, shift)),
            "DC_pre": keep(btbt(V) + tail(V)), "DC_1p67M": keep(btbt(V, shift) + k_trap * tail(V, shift))}


def test_pure_shift_gives_no_trap_change():
    m, delta, _ = d.measured(curves(1.0))
    assert abs(delta + 0.05) < 1e-3
    assert np.all(np.abs(m.dI / m.I_DC_pre) < 0.02)


def test_trap_increase_recovered():
    m, _, _ = d.measured(curves(1.5))
    row = m.iloc[1]                                  # 2e-13: tail-dominated
    expect = 0.5 * tail(row.V_meas)
    assert abs(row.dI / expect - 1) < 0.05


def test_summary_excludes_only_overpredicting_variants():
    df = pd.DataFrame(dict(offset_nm=0.0, I_DC_pre=[1e-13, 2e-13], dI_meas_hi=[2e-14, 4e-14]))
    for v in d.VARIANTS:
        k = 100.0 if v == "nominal" else 0.5
        df[f"dI_{v}_gate_min"] = k * df.dI_meas_hi
        df[f"Ipre_{v}_gate"] = df.I_DC_pre
    s = d.summary(df).set_index("variant")
    assert s.loc["nominal", "excluded"] and abs(s.loc["nominal", "ratio_min"] - 100) < 1e-9
    assert not s.drop("nominal").excluded.any()


def test_response_slope_limits():
    r = (d.NIT_PRE + d.DNIT) / d.NIT_PRE
    assert abs(d.response_slope(1.0, 0.0)) < 1e-12               # no change: s = 0
    assert abs(d.response_slope(1.0, r - 1.0) - 1.0) < 1e-12      # I proportional to N_it: s = 1
