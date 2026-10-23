"""Physical constants (CODATA 2018), SI units unless the name says otherwise.

Unit convention for the whole project (see HANDOFF §12, rule 3):
  length in cm, field in V/cm, energy in eV, rates in cm^-3 s^-1,
  1D current density in A/cm^2. Only SI quantities carry a `_SI` suffix.
"""

Q = 1.602176634e-19          # C, elementary charge
K_B_SI = 1.380649e-23        # J/K
K_B_EV = 8.617333262e-5      # eV/K
HBAR_SI = 1.054571817e-34    # J s
H_SI = 6.62607015e-34        # J s
M0_SI = 9.1093837015e-31     # kg
EPS0_F_PER_CM = 8.8541878128e-14  # F/cm

T_REF = 300.0  # K


def thermal_voltage(T: float) -> float:
    """kT/q in volts."""
    return K_B_EV * T
