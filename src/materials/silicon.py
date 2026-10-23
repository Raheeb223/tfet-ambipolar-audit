"""Silicon material parameters.

Sources (all standard; verify before final paper use):
  * Eg(T): Varshni form with Thurmond (1975) constants, Eg(0)=1.170 eV,
    alpha=4.73e-4 eV/K, beta=636 K -> Eg(300 K)=1.1245 eV.
  * Nc, Nv at 300 K: Green, J. Appl. Phys. 67, 2944 (1990): 2.86e19, 3.10e19 cm^-3,
    scaled as (T/300)^1.5 (ignores the T dependence of the DOS masses).
  * eps_r = 11.7 (Sze). HANDOFF §8 quotes 11.8; difference is negligible vs other
    uncertainties but is recorded here as an assumption.

NOTE: no bandgap narrowing yet (TODO, HANDOFF §6.1). ni from this module is the
undoped intrinsic density; with BGN the effective ni rises in the p+/n+ regions.
"""

import numpy as np

from src.constants import K_B_EV, T_REF

EG0 = 1.170        # eV
VARSHNI_ALPHA = 4.73e-4  # eV/K
VARSHNI_BETA = 636.0     # K
NC_300 = 2.86e19   # cm^-3
NV_300 = 3.10e19   # cm^-3
EPS_R = 11.7
MU_N_LOWFIELD = 1400.0   # cm^2/Vs, lightly doped; placeholder until doping-dependent mobility
MU_P_LOWFIELD = 450.0


def bandgap(T):
    """Indirect bandgap Eg(T) in eV (Varshni)."""
    T = np.asarray(T, dtype=float)
    return EG0 - VARSHNI_ALPHA * T**2 / (T + VARSHNI_BETA)


def nc(T):
    return NC_300 * (np.asarray(T, dtype=float) / T_REF) ** 1.5


def nv(T):
    return NV_300 * (np.asarray(T, dtype=float) / T_REF) ** 1.5


def intrinsic_density(T):
    """ni(T) in cm^-3 = sqrt(Nc Nv) exp(-Eg / 2kT). ~1.1e10 at 300 K with these constants."""
    T = np.asarray(T, dtype=float)
    return np.sqrt(nc(T) * nv(T)) * np.exp(-bandgap(T) / (2.0 * K_B_EV * T))
