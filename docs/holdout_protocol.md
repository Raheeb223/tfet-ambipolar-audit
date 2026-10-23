# Holdout measurement protocol: does interface-trap tunneling limit ambipolar suppression?

The audit finds that near the off state (V_GS = -0.2 V) interface traps cap the suppression of
every technique at D_it of order 1e8-1e9 cm^-2 eV^-1, **if** the traps are field-enhanced as in
the standard Hurkx model (tunneling mass m* <= 0.5). With weak enhancement (m* >= 2) or plain
SRH they do not. Published data cannot decide which applies: no study reports the trap density
and the off-state current behaviour on the same devices. This protocol defines the measurement
that decides it.

The model predictions were frozen before any such data existed:
`data/holdout/frozen_components.csv` (current components versus temperature for every trap
model variant), its SHA-256 in `frozen_components.csv.sha256`, and the readable table
`frozen_ea_table.csv`. The git history dates them. The scoring script
(`src/validation/holdout.py`) refuses to run if they change, and fits nothing to the new data.

## What the predictions say

For a plain Si p-i-n TFET (simulated `reference` device) at the off-state point, every trap
variant predicts E_a of about 0.07 eV, so **activation energy alone cannot separate them**.
What separates them is how the off-state current responds to the trap density:

| trap model (variant)              | d ln I / d ln D_it around D_it = 1e12 | trap limit near the off state |
|-----------------------------------|---------------------------------------|-------------------------------|
| Hurkx, m* = 0.25 (`nominal`)      | 0.71                                  | holds                         |
| Hurkx, m* = 0.5 (`m_0.5`)         | 0.48                                  | holds at >= 1.1e12            |
| Hurkx, m* = 2 (`m_2.0`)           | 0.05                                  | fails                         |
| plain SRH (`no_field_enh`)        | 0.00                                  | fails                         |

(Full table for all 18 variants, three gate voltages and D_it = 1e10-1e13 in
`frozen_ea_table.csv`; slopes there are across a tenfold split centred on D_it.)

A measured activation energy well above ~0.13 eV on a plain device is predicted by none of the
variants at D_it <= 1e12. That outcome rejects the whole trap model class, which would be an
important result in its own right; the published 0.43-0.45 eV (Der Agopian 2012, Luong 2017,
different devices from their trap measurements) already points that way.

## Devices

- Si (or SOI) TFETs of **one design**, from at least **two D_it splits** that differ by about a
  factor of ten. Typical: with and without forming-gas anneal (e.g. 400-450 C, H2/N2, 30 min),
  or a deliberately damaged split. All other process steps identical.
- At least 3 devices per split, so that device-to-device spread can be seen.
- Note any suppression technique (heterogate, low drain doping, underlap, dual work function);
  set `sim_devices` in `configs/holdout.yaml` to the matching simulated device(s), or to
  `[reference]` for a plain device.

## Measurements

1. **Interface-trap density, per split** (on the TFETs themselves if possible, otherwise on
   MOS capacitors or MOSFETs with the same gate stack from the same wafers):
   - preferred: charge pumping on the TFET (gated-diode configuration), base-level sweep,
     several frequencies (e.g. 100 kHz - 1 MHz) to subtract geometric components;
     D_it = I_cp / (q f A dE), stating the energy window dE used;
   - otherwise: conductance method or high-low frequency C-V on capacitors.
   - Report the value used (`Dit_cm2eV`) and a low/high bracket (`Dit_lo_cm2eV`,
     `Dit_hi_cm2eV`) reflecting the method's uncertainty.
2. **Transfer curves versus temperature, every device**:
   - V_DS = 0.5 V (record the value), V_GS swept through the off-state minimum and at least
     0.5 V into the ambipolar branch, steps <= 25 mV;
   - chuck temperatures 300, 325, 350, 375, 400 K (at least 4 inside 295-405 K, the window
     the model was evaluated in), stabilised >= 10 min each;
   - check that the gate current stays at least ten times below the drain current at the
     off-state point; otherwise the point measures gate leakage, not the channel.
   - Repeat the 300 K curve at the end to rule out drift or annealing during the run.

## Data format

`data/holdout/measured.csv` (template: `measured_template.csv`), one row per point:

| column        | unit  | meaning                                       |
|---------------|-------|-----------------------------------------------|
| device_id     | -     | unique per device                             |
| T_K           | K     | chuck temperature                             |
| V_GS_V        | V     | gate-source voltage                           |
| V_DS_V        | V     | drain-source voltage                          |
| I_D_A_per_um  | A/um  | drain current normalised to the device width  |

`data/holdout/devices.yaml` (template: `devices_template.yaml`): per device the measured D_it,
its bracket, the method and the split name.

## Scoring

    python -m src.validation.holdout score --meas data/holdout/measured.csv --devices data/holdout/devices.yaml

For every device the script fits E_a at the off-state point (the V_GS of minimum current at the
temperature nearest 300 K) and at every other gate voltage (`results/holdout/ea_measured.csv`).
For every split it fits the log-log slope of the 300 K off-state current against D_it. It
lists which trap variants are consistent with both (`results/holdout/verdict.csv`; tolerances in
`configs/holdout.yaml`) and states whether the surviving variants put the techniques
trap-limited at the measured D_it: **holds**, **fails**, **not settled**, or **model class
rejected**.

Raw data may be kept private; the scoring output can be published without it.
