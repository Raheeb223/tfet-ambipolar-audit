"""Loader/validator for measured scalar values quoted in the text of a source
(data/literature_values.yaml). Same provenance rules as the digitized curves: every value has a
source, a location in that source, a verbatim quote and a role; none is used in a fit.

Run `python -m src.dataio.literature_values` to validate the file.
"""

import os

from src.config import load_yaml
from src.dataio.digitized import MetadataError

PATH = os.path.join("data", "literature_values.yaml")
REQUIRED = ("key", "value", "unit", "quantity", "device", "source", "where", "quote", "role")
ROLES = {"validation", "context"}


def load_values(path=PATH):
    entries = (load_yaml(path) or {}).get("values") or []
    keys = set()
    for e in entries:
        miss = [f for f in REQUIRED if f not in e]
        if miss:
            raise MetadataError(f"{e.get('key', '<no key>')}: missing {miss}")
        if e["role"] not in ROLES:
            raise MetadataError(f"{e['key']}: role {e['role']!r} not in {sorted(ROLES)}")
        vals = e["value"] if isinstance(e["value"], list) else [e["value"]]
        if not all(isinstance(v, (int, float)) for v in vals):
            raise MetadataError(f"{e['key']}: value must be numeric")
        if e["key"] in keys:
            raise MetadataError(f"duplicate key {e['key']}")
        keys.add(e["key"])
    return {e["key"]: e for e in entries}


def value(key, path=PATH):
    return load_values(path)[key]["value"]


if __name__ == "__main__":
    print(f"OK: {len(load_values())} literature value(s) valid")
