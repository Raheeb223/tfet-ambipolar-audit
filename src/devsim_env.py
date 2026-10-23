"""Import DEVSIM with its math libraries configured.

DEVSIM on Windows needs a BLAS/LAPACK library at import time. The pip `mkl`
package installs `mkl_rt.N.dll` under `<prefix>/Library/bin`, which DEVSIM does
not search by default, so we point DEVSIM_MATH_LIBS at it before importing.

Always do `from src.devsim_env import devsim` instead of `import devsim`.
"""

import glob
import os
import sys


def _configure_math_libs() -> None:
    if os.environ.get("DEVSIM_MATH_LIBS"):
        return
    candidates = sorted(glob.glob(os.path.join(sys.prefix, "Library", "bin", "mkl_rt*.dll")))
    if candidates:
        os.environ["DEVSIM_MATH_LIBS"] = candidates[-1]


_configure_math_libs()

import devsim  # noqa: E402

# 128-bit (float128) model evaluation and solve. Without it the contact current of a
# 1e19-doped 1D diode has a ~1e-7 A/cm^2 noise floor (nonzero at 0 V); with it,
# ~1e-25 A/cm^2. Leakage audits need currents down to ~1e-15 A/um, so always on.
for _name in ("extended_solver", "extended_model", "extended_equation"):
    devsim.set_parameter(name=_name, value=True)

__all__ = ["devsim"]
