"""Local Kane band-to-band tunneling (BTBT) generation.

    G_BTBT = A * (F / F0)^P * exp(-B / F),      F0 = 1 V/cm
    P = 2 (direct) or 2.5 (phonon-assisted / indirect, e.g. Si)   [Kao et al., TED 2012]

Units: F, B in V/cm; A and G in cm^-3 s^-1.

Temperature enters through the bandgap. From the WKB triangular-barrier result
B = pi * sqrt(m_r) * Eg^(3/2) / (2*sqrt(2) * q * hbar), so B(T) = B_ref * (Eg(T)/Eg_ref)^(3/2).
A is held fixed with T by default (Kane's prefactor varies only weakly, ~Eg^-1/2);
that is an assumption and can be switched on with `scale_A=True`.

Known limitation (HANDOFF §6.2): a local model overestimates point tunneling vs
non-local path models; it must be calibrated and validated at device level.
"""

import numpy as np

from src.constants import HBAR_SI, M0_SI, Q

P_DIRECT = 2.0
P_INDIRECT = 2.5

# Sentaurus default local Kane parameters for Si, used as the "uncalibrated" baseline
# (quoted in Biswas et al., Microelectron. Eng. 2012).
SI_DEFAULT_A = 4.0e14   # cm^-3 s^-1
SI_DEFAULT_B = 1.9e7    # V/cm


def kane_generation(F, A, B, P=P_INDIRECT, F0=1.0):
    """BTBT generation rate (cm^-3 s^-1) for field magnitude F (V/cm).

    Returns exactly 0 where F <= 0, so the expression is safe to evaluate on
    whole field arrays.
    """
    F = np.abs(np.asarray(F, dtype=float))
    out = np.zeros_like(F)
    pos = F > 0
    out[pos] = A * (F[pos] / F0) ** P * np.exp(-B / F[pos])
    return out


def kane_B_from_mass(Eg_eV, m_r_over_m0):
    """Kane exponent B (V/cm) from the WKB triangular-barrier result.

    B = pi * sqrt(m_r) * Eg^(3/2) / (2 sqrt(2) q hbar).
    Si, Eg = 1.12 eV, m_r = 0.29 m0 -> B ~ 2.6e7 V/cm.
    The mass convention (reduced vs tunneling mass) differs between papers
    (Solomon 2004 vs Fair & Wivell); check which one each dataset reports.
    """
    m = m_r_over_m0 * M0_SI
    Eg_J = np.asarray(Eg_eV, dtype=float) * Q
    B_SI = np.pi * np.sqrt(m) * Eg_J**1.5 / (2.0 * np.sqrt(2.0) * Q * HBAR_SI)  # V/m
    return B_SI / 100.0  # V/cm


def kane_params_at_T(A_ref, B_ref, Eg_ref, Eg_T, scale_A=False):
    """Scale Kane (A, B) from a reference bandgap Eg_ref to Eg_T."""
    ratio = Eg_T / Eg_ref
    B = B_ref * ratio**1.5
    A = A_ref * ratio**-0.5 if scale_A else A_ref
    return A, B


def uniform_field_current_density(F, W, A, B, P=P_INDIRECT):
    """J (A/cm^2) = q * G(F) * W for a junction with uniform field F over width W (cm).

    This is the analytic J-E benchmark (HANDOFF §9 step 1.3): with a non-uniform
    field the correct value is q * integral(G(F(x)) dx), see `integrated_current_density`.
    """
    return Q * kane_generation(F, A, B, P) * W


def integrated_current_density(x_cm, F_of_x, A, B, P=P_INDIRECT):
    """J (A/cm^2) = q * integral G(F(x)) dx over a 1D field profile (trapezoid rule)."""
    return Q * np.trapezoid(kane_generation(F_of_x, A, B, P), x_cm)


def devsim_expression(field="ElectricField", A="btbt_A", B="btbt_B", P="btbt_P",
                      F_smooth="btbt_Fsmooth", eq_factor=None):
    """Symbolic DEVSIM expression for G_BTBT on an edge.

    |F| is regularized as sqrt(F^2 + Fs^2) so the expression and its derivative
    are smooth at F = 0 (helps Newton). With Fs << B/700 this changes nothing physical.
    `eq_factor` optionally multiplies by an expression that vanishes at equilibrium
    (see devices/diode_1d.py) so the local model gives zero current at zero bias.
    """
    Fabs = f"pow({field}^2 + {F_smooth}^2, 0.5)"
    expr = f"{A} * pow({Fabs}, {P}) * exp(-{B} / {Fabs})"
    if eq_factor:
        expr = f"({expr}) * ({eq_factor})"
    return expr
