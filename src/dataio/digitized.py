"""Loader/validator for digitized literature data (data/digitized/metadata.yaml).

Enforces HANDOFF §7/§12 rules in code:
  * every curve has full provenance metadata and a known role;
  * units are from a whitelist and converted to project units (cm, V/cm, A/cm^2, A/um, K);
  * `calibration_datasets()` refuses to return anything with role validation/sanity/context.

Run `python -m src.dataio.digitized` to validate the folder.
"""

import os
import warnings

import numpy as np
import pandas as pd

from src.config import load_yaml

DIGITIZED_DIR = os.path.join("data", "digitized")

REQUIRED_FIELDS = (
    "file", "source", "doi_or_url", "figure", "curve_label",
    "x_quantity", "x_unit", "y_quantity", "y_unit", "y_scale",
    "temperature_K", "bias_direction", "doping_info", "area_um2",
    "digitizer_repeats", "est_error", "role",
)
ROLES = {"core_calibration", "temperature", "validation", "sanity", "context",
         "structure"}   # structure: device input (e.g. a published doping profile), never fit
CALIBRATION_ROLES = {"core_calibration", "temperature"}
Y_SCALES = {"log", "lin"}
BIAS_DIRECTIONS = {"reverse", "forward", "both", "n/a"}

# unit -> (canonical unit, factor): value_canonical = value * factor
UNITS = {
    "V": ("V", 1.0), "mV": ("V", 1e-3),
    "V/cm": ("V/cm", 1.0), "MV/cm": ("V/cm", 1e6), "kV/cm": ("V/cm", 1e3), "V/m": ("V/cm", 1e-2),
    "A": ("A", 1.0), "mA": ("A", 1e-3), "uA": ("A", 1e-6), "nA": ("A", 1e-9), "pA": ("A", 1e-12),
    "A/cm2": ("A/cm2", 1.0), "mA/cm2": ("A/cm2", 1e-3), "kA/cm2": ("A/cm2", 1e3),
    "A/um": ("A/um", 1.0), "uA/um": ("A/um", 1e-6), "nA/um": ("A/um", 1e-9),
    "K": ("K", 1.0),
    "cm-3": ("cm-3", 1.0),
    "nm": ("cm", 1e-7), "um": ("cm", 1e-4), "cm": ("cm", 1.0),
    "eV": ("eV", 1.0),
    "1": ("1", 1.0),
}


class MetadataError(ValueError):
    pass


def load_metadata(directory=DIGITIZED_DIR):
    meta = load_yaml(os.path.join(directory, "metadata.yaml")) or {}
    entries = meta.get("datasets") or []
    for e in entries:
        validate_entry(e, directory)
    files = [e["file"] for e in entries]
    if len(files) != len(set(files)):
        raise MetadataError("duplicate 'file' entries in metadata.yaml")
    return entries


def validate_entry(e, directory=DIGITIZED_DIR):
    name = e.get("file", "<no file>")
    missing = [f for f in REQUIRED_FIELDS if f not in e]
    if missing:
        raise MetadataError(f"{name}: missing fields {missing}")
    if e["role"] not in ROLES:
        raise MetadataError(f"{name}: role {e['role']!r} not in {sorted(ROLES)}")
    if e["y_scale"] not in Y_SCALES:
        raise MetadataError(f"{name}: y_scale must be one of {sorted(Y_SCALES)}")
    if e["bias_direction"] not in BIAS_DIRECTIONS:
        raise MetadataError(f"{name}: bias_direction must be one of {sorted(BIAS_DIRECTIONS)}")
    for axis in ("x_unit", "y_unit"):
        if e[axis] not in UNITS:
            raise MetadataError(f"{name}: {axis} {e[axis]!r} not in unit whitelist")
    if not isinstance(e["digitizer_repeats"], int) or e["digitizer_repeats"] < 1:
        raise MetadataError(f"{name}: digitizer_repeats must be a positive integer")
    if e["digitizer_repeats"] < 3 and not str(e["est_error"]).startswith(("vector", "raster marker")):
        warnings.warn(f"{name}: digitized {e['digitizer_repeats']}x; HANDOFF §7 asks for 3x")
    path = os.path.join(directory, name)
    if not os.path.isfile(path):
        raise MetadataError(f"{name}: CSV not found at {path}")
    df = pd.read_csv(path)
    if df.shape[1] != 2:
        raise MetadataError(f"{name}: expected 2 columns, found {df.shape[1]}")
    if not all(np.issubdtype(t, np.number) for t in df.dtypes) or df.isna().any().any():
        raise MetadataError(f"{name}: non-numeric or missing values")
    if e["y_scale"] == "log" and (df.iloc[:, 1] <= 0).any():
        raise MetadataError(f"{name}: log-scale y must be > 0 (store magnitudes, sign in bias_direction)")


def load_curve(entry, directory=DIGITIZED_DIR):
    """Return a DataFrame with columns x, y in canonical units, and attrs = metadata + units."""
    df = pd.read_csv(os.path.join(directory, entry["file"]))
    xu, xf = UNITS[entry["x_unit"]]
    yu, yf = UNITS[entry["y_unit"]]
    out = pd.DataFrame({"x": df.iloc[:, 0].to_numpy() * xf, "y": df.iloc[:, 1].to_numpy() * yf})
    out.attrs = dict(entry, x_unit_canonical=xu, y_unit_canonical=yu)
    return out


def calibration_datasets(entries, roles=CALIBRATION_ROLES):
    """Entries allowed in a fit. Asking for validation/sanity/context data raises (rule 2)."""
    roles = set(roles)
    forbidden = roles - CALIBRATION_ROLES
    if forbidden:
        raise MetadataError(f"roles {sorted(forbidden)} must never be used for calibration")
    return [e for e in entries if e["role"] in roles]


def datasets_by_role(entries, role):
    if role not in ROLES:
        raise MetadataError(f"unknown role {role!r}")
    return [e for e in entries if e["role"] == role]


if __name__ == "__main__":
    ents = load_metadata()
    print(f"OK: {len(ents)} dataset(s) valid")
    for r in sorted(ROLES):
        print(f"  {r:17s} {len(datasets_by_role(ents, r))}")
