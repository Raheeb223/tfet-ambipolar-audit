"""Si(1-x)Ge(x) bandgap relations (HANDOFF §8). UNVERIFIED: check against the
original papers before using in calibration.
"""

import numpy as np


def bandgap_unstrained_lowT(x):
    """Unstrained SiGe gap near 4 K (Weber & Alonso, PRB 40, 5683, 1989).

    Returns min(X-valley, L-valley) gap in eV. X branch valid for x <~ 0.85.
    """
    x = np.asarray(x, dtype=float)
    eg_x = 1.155 - 0.43 * x + 0.206 * x**2
    eg_l = 2.010 - 1.270 * x
    return np.minimum(eg_x, eg_l)


def bandgap_strained_on_si_300K(x):
    """Pseudomorphic SiGe on Si(100) at 300 K (Li & Sturm, IEEE TED 60, 2479, 2013).

    Eg = 1.17 - 0.896x + 0.396x^2 - 0.05 eV. Li & Sturm data span x = 0.14-0.35;
    the x ~ 0.75 TFET source is far outside that range (and would not be
    pseudomorphic at useful thicknesses) -> flag as extrapolation.
    """
    x = np.asarray(x, dtype=float)
    return 1.17 - 0.896 * x + 0.396 * x**2 - 0.05
