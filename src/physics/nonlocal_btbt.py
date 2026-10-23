"""Nonlocal (path-based) Kane BTBT, evaluated as a post-processor on a band profile.

For each energy E on a grid, a tunneling path is a forbidden interval (Ev < E < Ec)
bounded by the valence band on one side and the conduction band on the other. With
path length l(E) the effective field is F_eff = Eg / l, and the path contributes

    dJ = q * G_Kane(F_eff) * dE / F_eff * [f_v(E) - f_c(E)]          (A/cm^2)

where f_v uses the quasi-Fermi level at the valence end and f_c the one at the
conduction end. For a uniform field F over a width W this gives
J = q G(F) (W - Eg/F): the local result minus the dead zone Eg/F that a local model
wrongly fills with generation. This is the effective-field ("dynamic path") nonlocal
form used in commercial TCAD; it keeps the calibrated (A, B, P) meaning of the local model.

ASSUMPTIONS (logged, HANDOFF §12):
  * paths are straight lines along the supplied profile (in 2D: along mesh rows,
    i.e. the source-drain direction). De Michielis et al., SSE 71, 7 (2012) show the
    path direction matters; the audit reports the row-path result alongside local.
  * Eg is the bandgap of the material on the path (single material on the drain side).
Units: x in cm, energies in eV, A in cm^-3 s^-1, B in V/cm, J in A/cm^2.
"""

import numpy as np

from src.constants import Q
from src.physics.kane_btbt import P_INDIRECT


def _fermi(E, Ef, kT):
    return 0.5 * (1.0 - np.tanh(0.5 * (E - Ef) / kT))


def _crossing(x, s, i):
    """x where s changes sign between samples i and i+1 (linear interpolation)."""
    return x[i] + (x[i + 1] - x[i]) * s[i] / (s[i] - s[i + 1])


def paths_at_energy(x, Ec, Ev, E):
    """List of (x_valence, x_conduction, i_valence, i_conduction) tunneling paths at energy E."""
    sc = Ec - E               # > 0 : E below the conduction band
    sv = E - Ev               # > 0 : E above the valence band
    gap = (sc > 0) & (sv > 0)
    out = []
    n = len(x)
    i = 0
    while i < n:
        if not gap[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and gap[j + 1]:
            j += 1
        if i > 0 and j < n - 1:   # interval closed on both sides
            left_v, right_v = sv[i - 1] <= 0, sv[j + 1] <= 0
            left_c, right_c = sc[i - 1] <= 0, sc[j + 1] <= 0
            if left_v and right_c:
                out.append((_crossing(x, sv, i - 1), _crossing(x, sc, j), i - 1, j + 1))
            elif left_c and right_v:
                out.append((_crossing(x, sv, j), _crossing(x, sc, i - 1), j + 1, i - 1))
        i = j + 1
    return out


def line_current(x, Ec, Ev, A, B, Eg, P=P_INDIRECT, Efn=None, Efp=None, kT=0.0259,
                 dE=1e-3, return_spectrum=False):
    """Nonlocal BTBT current density (A/cm^2) through a 1D band profile.

    Efn / Efp: electron / hole quasi-Fermi energies (eV) along x, same sign convention
    as Ec/Ev. If None, full occupation contrast (f_v - f_c = 1) is assumed.
    """
    x, Ec, Ev = (np.asarray(v, dtype=float) for v in (x, Ec, Ev))
    E_lo, E_hi = Ec.min(), Ev.max()
    if E_hi <= E_lo:
        return (0.0, np.array([]), np.array([])) if return_spectrum else 0.0
    E = np.arange(E_lo + 0.5 * dE, E_hi, dE)
    k, F, w = _path_terms(x, Ec, Ev, E, Eg, Efn, Efp, kT)
    dJ = np.zeros_like(E)
    if k.size:
        np.add.at(dJ, k, Q * A * F ** (P - 1.0) * np.exp(-B / F) * w)
    J = float(np.sum(dJ) * dE)
    return (J, E, dJ * dE) if return_spectrum else J


def _path_terms(x, Ec, Ev, E, Eg, Efn, Efp, kT):
    k, xv, xc, iv, ic = _all_paths(x, Ec, Ev, E)
    F = Eg / np.abs(xc - xv) if k.size else np.array([])
    occ = np.ones_like(F)
    if k.size and Efn is not None and Efp is not None:
        Efn, Efp = np.asarray(Efn, dtype=float), np.asarray(Efp, dtype=float)
        occ = _fermi(E[k], Efp[iv], kT) - _fermi(E[k], Efn[ic], kT)
    return k, F, occ


def path_terms(x, Ec, Ev, Eg, Efn=None, Efp=None, kT=0.0259, dE=1e-3):
    """(F_eff, weight) of every tunneling path, so that for ANY (A, B, P)

        J = sum(q * A * F_eff**(P-1) * exp(-B/F_eff) * weight)      (A/cm^2)

    equals line_current(...). Used to propagate parameter uncertainty without re-tracing paths.
    """
    x, Ec, Ev = (np.asarray(v, dtype=float) for v in (x, Ec, Ev))
    E_lo, E_hi = Ec.min(), Ev.max()
    if E_hi <= E_lo:
        return np.array([]), np.array([])
    E = np.arange(E_lo + 0.5 * dE, E_hi, dE)
    _, F, occ = _path_terms(x, Ec, Ev, E, Eg, Efn, Efp, kT)
    return F, occ * dE


def _all_paths(x, Ec, Ev, E):
    """Vectorized `paths_at_energy` over an energy grid.

    Returns arrays (k, x_v, x_c, i_v, i_c): energy index, turning points and the
    sample indices just outside the forbidden interval on the valence/conduction side.
    """
    n = len(x)
    gap = (Ec[None, :] > E[:, None]) & (Ev[None, :] < E[:, None])
    pad = np.zeros((len(E), 1), dtype=np.int8)
    d = np.diff(np.hstack([pad, gap.astype(np.int8), pad]), axis=1)
    ks, i = np.nonzero(d == 1)      # gap starts at i
    _, j = np.nonzero(d == -1)      # gap ends at j - 1
    j = j - 1
    inner = (i > 0) & (j < n - 1)
    ks, i, j = ks[inner], i[inner], j[inner]
    e = E[ks]
    lv = e <= Ev[i - 1]             # left neighbour inside the valence band
    lc = e >= Ec[i - 1]
    rv = e <= Ev[j + 1]
    rc = e >= Ec[j + 1]

    def cross(s0, s1, a, b):        # zero of the linear interpolant between x[a], x[b]
        return x[a] + (x[b] - x[a]) * s0 / (s0 - s1)

    vc = lv & rc                    # valence on the left, conduction on the right
    cv = lc & rv
    xv = np.where(vc, cross(e - Ev[i - 1], e - Ev[np.minimum(i, n - 1)], i - 1, i),
                  cross(e - Ev[j], e - Ev[j + 1], j, j + 1))
    xc = np.where(vc, cross(Ec[j] - e, Ec[j + 1] - e, j, j + 1),
                  cross(Ec[i - 1] - e, Ec[i] - e, i - 1, i))
    iv = np.where(vc, i - 1, j + 1)
    ic = np.where(vc, j + 1, i - 1)
    keep = vc | cv
    return ks[keep], xv[keep], xc[keep], iv[keep], ic[keep]
