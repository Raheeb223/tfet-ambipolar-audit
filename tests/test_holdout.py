"""Holdout scoring (src/validation/holdout): Arrhenius fit, frozen-hash guard, verdicts on
synthetic devices whose answer is known."""

import os

import numpy as np
import pandas as pd
import pytest
import yaml

from src.validation import holdout as ho

TS = (300.0, 358.15, 398.15)


def _comp(tmp_path):
    """Two devices, two interface variants: 'fast' (trap slope E_a 0.07 eV, large) and 'slow'
    (0.6 eV, small); BTBT 0.05 eV, bulk TAT 0.08 eV."""
    rows = []
    for dev, s in (("d1", 1.0), ("d2", 2.0)):
        for T in TS:
            arr = lambda A, Ea: A * np.exp(-Ea / (ho.KB * T)) / np.exp(-Ea / (ho.KB * 300.0))
            for comp, val in (("btbt", s * arr(1e-12, 0.05)), ("bulk_tat", arr(1e-14, 0.08)),
                              ("it:fast", s * arr(1e-22, 0.07)), ("it:slow", arr(1e-27, 0.6))):
                rows.append(dict(device=dev, Vg=-0.2, T_K=T, component=comp, value=val))
    path = tmp_path / ho.FROZEN
    pd.DataFrame(rows).to_csv(path, index=False, float_format="%.10e", lineterminator="\n")
    (tmp_path / (ho.FROZEN + ".sha256")).write_text(f"{ho.file_hash(path)}  {ho.FROZEN}\n")
    return dict(frozen_dir=str(tmp_path), out_dir=str(tmp_path / "out"), vg_decisive=-0.2,
                fit_window_K=[295.0, 405.0], min_temperatures=3, n_sigma=2.0, ea_tol_eV=0.02,
                slope_tol=0.15)


def _measure(tmp_path, Ea, Dit, I300=1e-12, split=False):
    """One device per (D_it, I300) pair; with split=True they form one D_it split."""
    Dit, I300 = np.atleast_1d(Dit), np.broadcast_to(I300, np.atleast_1d(Dit).shape)
    Ts = (300.0, 325.0, 350.0, 375.0, 400.0)
    rows, devs = [], {}
    for k, (D, I0) in enumerate(zip(Dit, I300)):
        dev = f"w{k + 1}"
        devs[dev] = {"Dit_cm2eV": float(D), "method": "synthetic"} | ({"split": "A"} if split else {})
        rows += [dict(device_id=dev, T_K=T, V_GS_V=V, V_DS_V=0.5,
                      I_D_A_per_um=I0 * (1 + 5 * abs(V)) * np.exp(-Ea / (ho.KB * T)) / np.exp(-Ea / (ho.KB * 300)))
                 for T in Ts for V in (-0.4, -0.2, 0.0, 0.2)]
    m, d = tmp_path / "m.csv", tmp_path / "d.yaml"
    pd.DataFrame(rows).to_csv(m, index=False)
    d.write_text(yaml.safe_dump({"devices": devs}))
    return str(m), str(d)


def test_arrhenius_recovers_activation_energy():
    T = np.array([300.0, 325.0, 350.0, 400.0])
    ea, se = ho.arrhenius(T, 3e-9 * np.exp(-0.45 / (ho.KB * T)))
    assert ea == pytest.approx(0.45, rel=1e-9) and se < 1e-9
    noisy = 3e-9 * np.exp(-0.45 / (ho.KB * T)) * np.array([1.05, 0.95, 1.04, 0.97])
    assert ho.arrhenius(T, noisy)[1] > 0


def test_total_current_and_crossing(tmp_path):
    cfg = _comp(tmp_path)
    comp = ho.load_frozen(cfg)
    T, I = ho.total_current(comp, "fast", "d1", -0.2, 1e10)
    assert list(T) == list(TS) and I[0] == pytest.approx(1e-12 + 1e-14 + 1e-12)
    assert ho.d_cross(comp, "fast", "d1", -0.2) == pytest.approx((1e-12 + 1e-14) / 1e-22)


def test_tampered_predictions_are_refused(tmp_path):
    cfg = _comp(tmp_path)
    with open(tmp_path / ho.FROZEN, "a") as f:
        f.write("d1,-0.2,300.0,btbt,1.0\n")
    with pytest.raises(RuntimeError, match="frozen hash"):
        ho.load_frozen(cfg)


def test_freeze_refuses_to_overwrite(tmp_path):
    cfg = _comp(tmp_path)
    with pytest.raises(RuntimeError, match="frozen"):
        ho.freeze(cfg)


def test_low_activation_energy_alone_does_not_settle(tmp_path):
    # 'slow' traps carry no current at 1e12, so the total is BTBT-like (low E_a) as well
    cfg = _comp(tmp_path)
    _, out = ho.score(cfg, *_measure(tmp_path, Ea=0.07, Dit=1e12))
    assert set(out[out.consistent].variant) == {"fast", "slow"}
    assert "not settled" in ho.summary(out) and "split" in ho.summary(out)


def test_dit_split_with_proportional_current_selects_field_enhanced_traps(tmp_path):
    cfg = _comp(tmp_path)
    _, out = ho.score(cfg, *_measure(tmp_path, Ea=0.07, Dit=[1e11, 1e12], I300=[1e-11, 1e-10], split=True))
    assert out.slope_meas.iloc[0] == pytest.approx(1.0)
    assert set(out[out.consistent].variant) == {"fast"}
    assert set(out[out.variant == "fast"].trap_limit) == {"all"}
    assert "HOLDS" in ho.summary(out)


def test_dit_split_with_flat_current_selects_weak_traps(tmp_path):
    cfg = _comp(tmp_path)
    _, out = ho.score(cfg, *_measure(tmp_path, Ea=0.07, Dit=[1e11, 1e12], I300=1e-12, split=True))
    assert set(out[out.consistent].variant) == {"slow"}
    assert "FAILS" in ho.summary(out)


def test_high_activation_energy_rejects_all_and_says_so(tmp_path):
    cfg = _comp(tmp_path)
    _, out = ho.score(cfg, *_measure(tmp_path, Ea=0.45, Dit=1e11))
    assert not out.consistent.any()
    assert "rejected" in ho.summary(out)


def test_off_state_point_is_the_300K_minimum(tmp_path):
    cfg = _comp(tmp_path)
    ea, _ = ho.score(cfg, *_measure(tmp_path, Ea=0.2, Dit=1e12))
    off = ea[ea.off_state]
    assert len(off) == 1 and off.V_GS_V.iloc[0] == 0.0
    assert off.Ea.iloc[0] == pytest.approx(0.2, rel=1e-6)
    assert os.path.exists(os.path.join(cfg["out_dir"], "verdict.csv"))
