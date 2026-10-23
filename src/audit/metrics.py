"""Audit metrics.

Suppression of a technique in a given physics scenario s:
    S_s = log10( I_amb,ref,s / I_amb,tech,s )        (decades)

Suppression retention:
    R_s = S_s / S_baseline
where the baseline is the scenario the literature typically reports (local Kane,
default parameters, BTBT only, 300 K). R = 1: the claim survives; R = 0: the technique
no longer helps; R < 0: it makes ambipolar leakage worse.
"""

import numpy as np


def suppression_decades(I_ref, I_tech):
    I_ref, I_tech = np.asarray(I_ref, dtype=float), np.asarray(I_tech, dtype=float)
    if np.any(I_ref <= 0) or np.any(I_tech <= 0):
        raise ValueError("currents must be positive")
    return np.log10(I_ref / I_tech)


def retention(S_scenario, S_baseline):
    S_baseline = np.asarray(S_baseline, dtype=float)
    if np.any(np.abs(S_baseline) < 1e-3):
        raise ValueError("baseline suppression ~0: retention undefined")
    return np.asarray(S_scenario, dtype=float) / S_baseline


def dominant_mechanism(components):
    """Name of the largest current component in a {name: current} dict."""
    return max(components, key=lambda k: components[k])


def crossover(x, I_a, I_b):
    """First x where I_b overtakes I_a (log-linear interpolation); None if it never does."""
    x = np.asarray(x, dtype=float)
    d = np.log10(np.asarray(I_b, dtype=float)) - np.log10(np.asarray(I_a, dtype=float))
    idx = np.nonzero((d[:-1] < 0) & (d[1:] >= 0))[0]
    if idx.size == 0:
        return None
    i = idx[0]
    return float(x[i] + (x[i + 1] - x[i]) * (-d[i]) / (d[i + 1] - d[i]))


# ---- interface-trap design rules ------------------------------------------------------
# Traps are not charged in the electrostatics (ASSUMPTION), so the interface-TAT current is
# exactly linear in D_it: I(D) = I0 + a*D with a = I_it(D_ref)/D_ref. This makes the D_it
# sweep continuous and the rules below closed-form or a 1D monotone root.

def suppression_vs_dit(D, I_ref0, a_ref, I_t0, a_t):
    """S(D) = log10[(I_ref0 + a_ref D)/(I_t0 + a_t D)] for scalar or array D (cm^-2 eV^-1)."""
    D = np.asarray(D, dtype=float)
    return np.log10((I_ref0 + a_ref * D) / (I_t0 + a_t * D))


def trap_limited_suppression(a_ref, a_t):
    """S as D_it -> infinity: the suppression a technique gives when traps carry all current."""
    return float(np.log10(a_ref / a_t))


def dit_crossover(I_other, a_it):
    """D_it at which interface TAT equals all other generation in the same device."""
    if I_other <= 0 or a_it <= 0:
        raise ValueError("currents must be positive")
    return float(I_other / a_it)


def dit_for_retention(I_ref0, a_ref, I_t0, a_t, frac=0.5, lo=1e6, hi=1e16):
    """Smallest D_it at which S(D) falls to frac * S(0); None if it never does below hi.

    S(D) is monotone in D (ratio of two linear functions), so bisection in log10(D) is exact
    to the tolerance used (1e-6 decades in D).
    """
    S0 = suppression_vs_dit(0.0, I_ref0, a_ref, I_t0, a_t)
    target = frac * S0
    f = lambda D: suppression_vs_dit(D, I_ref0, a_ref, I_t0, a_t) - target  # noqa: E731
    if f(hi) > 0:
        return None
    a, b = np.log10(lo), np.log10(hi)
    if f(10**a) <= 0:
        return float(lo)
    while b - a > 1e-6:
        m = 0.5 * (a + b)
        a, b = (m, b) if f(10**m) > 0 else (a, m)
    return float(10**b)
