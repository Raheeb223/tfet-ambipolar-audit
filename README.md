# tfet-ambipolar-audit

**How much ambipolar suppression is real?** An audit of four TFET ambipolar-suppression
techniques (hetero-gate dielectric, low drain doping, gate–drain underlap, drain-side work
function) on one double-gate Si TFET, with local and nonlocal BTBT, Hurkx field-enhanced bulk and
interface trap-assisted tunneling (TAT) and 300–398 K, simulated with the open-source solver DEVSIM.

Metrics (`src/audit/metrics.py`): suppression S = log10(I_amb,ref / I_amb,tech); because traps are
uncharged, I(D_it) = I0 + a·D_it exactly, which gives the trap-limited suppression
S_inf = log10(a_ref/a_tech), the crossover density D_x = I0/a and the half-retention density D_1/2.

**Status (2026-10-23):** BTBT calibrated on measured Si junction tunneling (Fair & Wivell 1976,
Stork & Isaac 1983, via Zhao 2007; 0.23-decade rms) and validated, with the model frozen, on the measured SOI TFETs
of Wan et al. 2011. Further frozen-model checks: on-state branch of Biswas et al. 2012 (the nonlocal
model is ~2 decades low there, reported as a limitation), temperature signature of Der Agopian
et al. 2012 and Luong 2017 (measured E_a 0.43-0.45 eV vs model <= 0.16 eV), measured D_it /
N_it anchors (Luong 2017) and measured underlap (Krishnaraja 2019). Post-hoc check on the
irradiated SOI/HfO2 TFET of Ding et al. 2014 (measured response to a charge-pumping-measured
trap increase): the strong field-enhancement trap variants (m* <= 0.5, incl. nominal) predict
at least 20x the measured current change and are excluded; m* >= 1 and plain SRH are
consistent. Calibrated audit, UQ (2000
samples), charged-trap and U-shaped-D_it checks done. The manuscript is in preparation and is
not part of this repository; the analysis scripts write its figures and numbers to a local
`paper/` folder (git-ignored).

## Setup (Windows, Python 3.11)

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

DEVSIM needs BLAS/LAPACK. The pip `mkl` package provides it, and `src/devsim_env.py` points
`DEVSIM_MATH_LIBS` at it automatically. Always import DEVSIM as `from src.devsim_env import devsim`.
That module also turns on DEVSIM's 128-bit extended precision; without it, low currents have a
~1e-7 A/cm² noise floor.

## Run

```powershell
.venv\Scripts\python -m pytest                    # all tests (~45 s)
.venv\Scripts\python -m pytest -m "not devsim"    # fast unit tests only
.venv\Scripts\python -m src.devices.diode_1d      # 1D diode sweep -> results/diode_1d/
.venv\Scripts\python -m src.dataio.digitized      # validate data/digitized/metadata.yaml
# full pipeline, in order (source PDFs in LIT_PDF_DIR, default ~/Downloads)
.venv\Scripts\python -m src.dataio.extract_literature    # digitize Zhao 2.3, Wan 6-8, Biswas 3, Der Agopian 7
.venv\Scripts\python -m src.dataio.literature_values     # validate text-quoted values (data/literature_values.yaml)
.venv\Scripts\python -m src.calibration.btbt_calibration # fit local/nonlocal Kane -> results/calibration
.venv\Scripts\python -m src.audit.run_audit                # solve + cache electrostatics (~35 min)
.venv\Scripts\python -m src.audit.run_audit --audit configs/audit_calibrated.yaml --only-evaluate
.venv\Scripts\python -m src.audit.run_audit --audit configs/audit_sigma2nm.yaml   # straggle solves
.venv\Scripts\python -m src.audit.run_audit --audit configs/audit_sigma4nm.yaml
.venv\Scripts\python -m src.audit.run_audit --audit configs/audit_cal_sigma2nm.yaml --only-evaluate
.venv\Scripts\python -m src.audit.run_audit --audit configs/audit_cal_sigma4nm.yaml --only-evaluate
.venv\Scripts\python -m src.validation.wan2011           # frozen-model validation (~40 min)
.venv\Scripts\python -m src.validation.biswas2012        # on-state frozen-model check (~1.5 h)
.venv\Scripts\python -m src.validation.biswas2012 --coverage   # tunneling fields of that check
.venv\Scripts\python -m src.validation.ding2014 --run --analyse   # post-hoc trap-response check (~1 h)
.venv\Scripts\python -m src.uq.btbt_uq                   # Monte Carlo over (A, B)
.venv\Scripts\python -m src.audit.charged_traps          # self-consistent trap charge
.venv\Scripts\python -m src.analysis.coverage            # calibrated field range check
.venv\Scripts\python -m src.analysis.figures             # audit figures, tables, numbers
.venv\Scripts\python -m src.analysis.validation_figs     # calibration/validation figures, numbers
.venv\Scripts\python -m src.analysis.checks_numbers      # robustness-check numbers
.venv\Scripts\python -m src.analysis.extval_figs         # independent checks: fig_extval, Table data
# robustness and scope
.venv\Scripts\python -m src.audit.run_audit --audit configs/geom_tsi5.yaml   # device matrix (also tsi20,
                                                     # tox3, lg30, nd5e19, vd05), ~10 min each
.venv\Scripts\python -m src.audit.model_form --parts slopes btbt ensemble floors spatial validation  # model-form ensemble
.venv\Scripts\python -m src.audit.qf_validity           # quasi-Fermi-level validity map
.venv\Scripts\python -m src.analysis.robustness_figs    # fig_robust, claim/device/trap-model tables
.venv\Scripts\python -m src.validation.holdout score   # score holdout measurements against the frozen predictions (docs/holdout_protocol.md)
# last: independent re-derivation of every derived number (exit code 1 if any check fails)
.venv\Scripts\python -m src.analysis.math_checks
```

`src.validation.wan2011` and `src.validation.biswas2012` cache their solutions in
`results/validation/cache/`; `model_form --parts validation` re-uses them to test the alternative
path and trap models on the measured devices without re-solving.

The paper never contains hand-typed results: the scripts write `paper/generated/numbers*.tex`,
which define a macro for every quoted number, and `paper/figs/` (both git-ignored).

## Layout

```
configs/            YAML device configs; every assumption is recorded here, nothing is hard-coded
data/digitized/     digitized literature curves (CSV) + metadata.yaml (provenance, units, role)
data/literature_values.yaml  measured values quoted in source texts (location + verbatim quote)
data/raw/           source PDFs (git-ignored; never commit paywalled papers)
src/constants.py    physical constants; project units: cm, V/cm, eV, cm^-3 s^-1, A/cm^2
src/config.py       YAML loader that reads 1e19-style numbers as floats
src/devsim_env.py   DEVSIM import with MKL + extended precision
src/materials/      Si Eg(T), ni(T); SiGe bandgap relations
src/physics/        kane_btbt.py (local Kane), nonlocal_btbt.py (path-based Kane),
                    nonlocal_field_paths.py (gradient-at-start and angle-complete 2D paths),
                    hurkx_tat.py (Hurkx field-enhanced SRH, quadrature Gamma)
src/devices/        diode_1d.py (1D p+/n+ diode), tfet_2d.py (2D DG-TFET, generation post-processing)
src/dataio/         digitized-data loader; enforces "never calibrate on validation data"
src/dataio/         plot_digitizer.py (vector/raster), extract_literature.py (all curves)
src/calibration/    btbt_calibration.py (junction J(F) fit of local and nonlocal Kane)
src/validation/     wan2011.py (frozen-model validation on measured SOI TFETs), biswas2012.py (on-state),
                    holdout.py (pre-registered holdout test of the trap model),
                    ding2014.py (post-hoc check on a measured radiation-induced trap increase)
src/audit/          run_audit.py (solve + cache + scenarios), metrics.py, charged_traps.py,
                    model_form.py (structural ensemble), qf_validity.py (transport validity)
src/uq/             btbt_uq.py (Monte Carlo over the calibration covariance)
src/analysis/       figures.py, validation_figs.py, checks_numbers.py, coverage.py, extval_figs.py,
                    math_checks.py (closed-form, independent re-derivation of the derived numbers)
paper/              local only (git-ignored): generated figs/ and generated/ numbers
tests/              unit + DEVSIM verification tests
results/            generated outputs (git-ignored; regenerate from scripts)
```

## Verification status (HANDOFF §9 step 1)

| Check | Status |
|---|---|
| DEVSIM upstream 1D diode example runs | done |
| Kane G: hand values, ln-slope = P + B/F, B(m*, Eg) from WKB, T scaling | tests pass |
| Hurkx Γ: → 0 at low field, low-field asymptote a·Γ(5/3)·K^(-2/3), closed form vs integral, SRH limit | tests pass |
| 1D diode: 0 V current < 1e-20 A/cm² | pass |
| 1D diode: peak field at equilibrium vs depletion approximation (< 1 %) | pass |
| 1D diode: contact current = q∫G_BTBT + q∫(−U_SRH) + short-base diffusion (< 3 %) | pass |
| 1D diode: DEVSIM Kane expression = numpy Kane | pass |
| 1D diode: mesh convergence, 2× refinement (< 1 %) | pass |
| Hurkx Γ by Gauss–Legendre quadrature vs adaptive integral (rel. err. < 1.2e-6, 1e2–1e7 V/cm) | tests pass |
| Nonlocal BTBT: uniform-field analytic limit, no path below Eg, symmetry, equilibrium, vectorized = loop | tests pass |
| 2D TFET: equilibrium generation < 1e-25, contact potentials, field vs DEVSIM edge field (< 5 %) | pass |
| 2D TFET: Gauss's-law balance (< 1e-3), mesh refine 1→2 (< 10 %; 1→3 < 6 % per component) | pass |
| Audit metrics: S, R, crossover, S(D_it) limits, D_1/2 vs closed form | tests pass |
| Interface current linear in D_it (checked at 5e12 in every analysis run) | pass |
| Digitizer recovers a synthetic curve (< 0.05 decade); repeat spread reported | tests pass |
| Calibration: synthetic (A, B) recovered; local J = direct integral; nonlocal < local | tests pass |
| Calibration on Zhao Fig. 2.3 data: rms 0.23 decade; leave-one-source-out <= 0.53 decade | done |
| Validation data never reach a fit (code-enforced role check) | test passes |
| Parameter-free BTBT terms reproduce generation() to 1e-10 (used by UQ) | test passes |
| Charged-trap fixed point: D_it = 0 unchanged, converges, correct sign | test passes |
| Frozen-model validation, Wan 2011 (nonlocal vs local, all offsets) | done |
| Interface TAT = Hurkx closed form for a single level (pairing n1<->Gamma_p, p1<->Gamma_n), sigma scaling, spectra, damage | tests pass |
| Field-direction paths: axis = 1D, 0/90 deg invariance, 45 deg = single family, bins converge; angle-complete >= gradient | tests pass |
| Calibration refit at other P recovers synthetic (A, B) | test passes |
| Quasi-Fermi drop = Ohm's law on a uniform sheet | test passes |
| Holdout scoring: Arrhenius fit, frozen-hash guard, verdicts on synthetic devices with known answers | tests pass |
| Ding 2014 check: pure shift gives no trap change, known trap increase recovered, exclusion rule, response-slope limits | tests pass |
| Closed forms: D_1/2 and S(D) from (S0, S_inf, D_x) vs bisection, monotonicity, incomplete-gamma junction J vs quadrature, B <-> m_r, Arrhenius, outward rounding of quoted bounds | tests pass |
| Independent re-derivation of the derived numbers (design rules, calibration, conversions of measured data, ensemble recount; `math_checks`) | 27/27 pass |

## Known limitations / open TODOs

- **Open: holdout test of the interface-trap model.** The near-off-state trap limit depends on
  the trap field enhancement, which no published data set constrains on the same devices.
  The post-hoc Ding 2014 check disfavours strong enhancement, but on the source-side
  junction of a different device, for radiation-induced traps, and with a logged bias
  alignment (ASSUMPTION, +-0.1 V scanned); it does not replace the holdout test.
  Predictions are frozen in `data/holdout/` (2026-10-23); the measurement that decides it (a
  D_it split of one TFET design, with transfer curves at 300-400 K) is specified in
  `docs/holdout_protocol.md`.

- **Fixed 2026-10-23: interface-TAT pairing.** `interface_generation` paired the n1 term with
  the electron enhancement Gamma_n instead of Gamma_p (Hurkx 1992; the bulk-TAT code was
  right). Midgap traps were unaffected; off-midgap traps were under-counted, so the interface
  current was 2-11x too low. All trap-dependent results were regenerated.

- **Equilibrium factor on local BTBT.** G is multiplied by (ni² − np)/(ni² + np), so the local model
  carries no current at 0 V; without it the 1D diode gives ~1e-2 A/cm² at 0 V. The factor is not from
  a paper. BTBT-generated carriers in the depletion region pull it below 1, lowering J by ~6 % at
  −1 V in the test diode. Compare it with Hurkx's D factor before calibration.
- **Hurkx closed-form Γ** overestimates Γ by up to ~10⁷ at drain-junction fields. RESOLVED: the 2D
  workflow uses `gamma_fast` (vectorized quadrature) in post-processing.
- **Hurkx ΔE** = min(E_band − E_t, kT ln(N/n)) is a common simplification, not yet checked
  line-by-line against Hurkx 1992. Its consequences are bounded by the model-form ensemble
  (tunneling mass, spectra, cross-sections, plain SRH).
- **Traps are uncharged** in the main audit (linear in D_it); `charged_traps.py` checks the
  self-consistent case (|dS| <= 0.16 decade up to 1e12, 0.37 at 5e12).
- **Not in the 2D model:** Fermi-Dirac statistics, bandgap narrowing, Auger generation, band
  tails, gate leakage. Nonlocal paths: x and y mesh lines, larger of the two reported.
- **Calibration range:** 0.93-2.23 MV/cm at 300 K. Reference and DWF devices tunnel at higher
  fields (extrapolated); see `results/audit_cal/calibration_coverage.csv`.
- **Validation:** suppression vs L_IN is not predictable without the implant-edge position.
- **On-state branch (Biswas 2012):** calibrated nonlocal BTBT is 1.9-2.1 decades low on
  average (1.1-2.2 at VGS = 3 V), inside the calibrated field range for VGS >= 2 V, with a much
  stronger V_D dependence than measured; local is within 0.7 decade. Diagnosed in `src/validation/biswas_diagnostics.py`: the
  energy window changes it by <2%; a steeper source helps but needs a profile steeper than
  measured; the main cause is path direction (oblique source-to-inversion-layer paths;
  angle-complete paths raise the 3 V current x44 and fix those points to 0.4 decade, mean
  shortfall 1.2-1.5). The same angle-complete paths overpredict the Wan devices by
  1.4-4.0 decades, so the production rows/columns model is kept (refitting would use
  validation data). With angle-complete ambipolar BTBT, D_x at -0.2 V is 3.9e8-2.2e10, still
  below 1e11 for all four techniques.
- **Temperature:** model trap current E_a <= 0.16 eV vs 0.43-0.45 eV measured; 398 K trap
  penalties are lower bounds.
- **SiGe bandgap relations and Kao A/B values** are unverified (HANDOFF §14).

## Licence

Code: MIT (`LICENSE`). Digitized data (`data/digitized/`, `data/literature_values.yaml`):
CC BY 4.0 (`data/LICENSE`); please also cite the original sources listed in the metadata.
Source PDFs are not included.
