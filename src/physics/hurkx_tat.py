"""Hurkx trap-assisted tunneling (field-enhanced SRH).

    R = (n p - ni^2) / ( tau_p/(1+Gamma_p) * (n + n1) + tau_n/(1+Gamma_n) * (p + p1) )

    Gamma = (dE/kT) * integral_0^1 exp[ (dE/kT) u - K u^(3/2) ] du
    K     = (4/3) * sqrt(2 m* dE^3) / (q hbar |F|)
    m*    = 0.25 m0 (Hurkx default)

Saddle-point closed form (used for DEVSIM, where an integral is not a symbolic expression):

    Gamma ~ 2 sqrt(3 pi) (|F|/F_Gamma) exp[(|F|/F_Gamma)^2],   F_Gamma = sqrt(24 m* (kT)^3) / (q hbar)

valid when the integrand maximum u0 = (2 dE / (3 kT K))^2 lies well inside (0, 1).

TODO(verify vs Hurkx, Klaassen & Knuvers, IEEE TED 39(2):331, 1992):
  * the exact definition of dE (Hurkx uses a carrier-density dependent energy range,
    not simply Ec - Et) and whether it is piecewise in n vs ni;
  * the integral limits and the validity condition of the closed form.
Until then `dE` is an explicit argument; nothing here picks it silently.
"""

import numpy as np
from scipy.integrate import quad

from src.constants import HBAR_SI, K_B_EV, M0_SI, Q

M_HURKX = 0.25  # m*/m0


def _K(F, dE_eV, m_rel):
    """Dimensionless K for field F (V/cm) and energy range dE (eV)."""
    F_SI = np.abs(np.asarray(F, dtype=float)) * 100.0  # V/m
    m = m_rel * M0_SI
    dE_J = dE_eV * Q
    with np.errstate(divide="ignore"):
        return (4.0 / 3.0) * np.sqrt(2.0 * m * dE_J**3) / (Q * HBAR_SI * F_SI)


def gamma_integral(F, dE_eV, T, m_rel=M_HURKX):
    """Field-enhancement factor Gamma by numerical integration (reference implementation).

    Vectorized over F; dE and T scalars. Returns 0 at F = 0.
    """
    F = np.atleast_1d(np.abs(np.asarray(F, dtype=float)))
    a = dE_eV / (K_B_EV * T)
    out = np.zeros_like(F)
    for i, (f, k) in enumerate(zip(F, _K(F, dE_eV, m_rel))):
        if f == 0.0:
            continue
        # Integrate exp(phi(u) - phi_max) and add phi_max back in log space.
        # The integrand can be a very narrow peak (width ~K^-2/3 at low field), so split
        # the interval explicitly just past the peak; quad's `points=` (QAGP) was found
        # to return wrong values with tiny error estimates in that regime.
        u0 = min((2.0 * a / (3.0 * k)) ** 2, 1.0)
        phi_max = a * u0 - k * u0**1.5
        width = max(np.sqrt(8.0 * a) / (3.0 * k), k ** (-2.0 / 3.0))
        b = min(1.0, u0 + 40.0 * width)
        f = lambda u: np.exp(a * u - k * u**1.5 - phi_max)
        val = quad(f, 0.0, b, limit=200, epsabs=0.0, epsrel=1e-10)[0]
        if b < 1.0:
            val += quad(f, b, 1.0, limit=200, epsabs=1e-12 * val, epsrel=1e-10)[0]
        out[i] = a * val * np.exp(phi_max)
    return out


_GL_X, _GL_W = np.polynomial.legendre.leggauss(24)


def gamma_fast(F, dE_eV, T, m_rel=M_HURKX, panels=6):
    """Vectorized Gamma for post-processing whole meshes (replaces the closed form).

    Same integrand and window as `gamma_integral` (peak at u0, capped at the u = 1
    endpoint for high fields), evaluated with `panels` x 24-point Gauss-Legendre.
    Valid at all fields, unlike `gamma_approx`: max relative error vs `gamma_integral`
    1.2e-6 over F = 1e2-1e7 V/cm, dE = 0.2-0.7 eV, T = 200-400 K (tests/test_hurkx_tat.py).
    `dE_eV` may be a scalar or an array broadcastable with F.
    """
    F = np.abs(np.asarray(F, dtype=float))
    dE = np.broadcast_to(np.asarray(dE_eV, dtype=float), F.shape)
    T = np.broadcast_to(np.asarray(T, dtype=float), F.shape)
    out = np.zeros(F.shape)
    pos = (F > 0) & (dE > 0)
    if not np.any(pos):
        return out
    a = dE[pos] / (K_B_EV * T[pos])
    K = _K(F[pos], dE[pos], m_rel)
    u0 = np.minimum((2.0 * a / (3.0 * K)) ** 2, 1.0)
    phi_max = a * u0 - K * u0**1.5
    width = np.maximum(np.sqrt(8.0 * a) / (3.0 * K), K ** (-2.0 / 3.0))
    lo = np.maximum(0.0, u0 - 40.0 * width)
    hi = np.minimum(1.0, u0 + 40.0 * width)
    edges = np.linspace(0.0, 1.0, panels + 1)
    total = np.zeros_like(K)
    for i in range(panels):
        l = lo + (hi - lo) * edges[i]
        h = lo + (hi - lo) * edges[i + 1]
        u = 0.5 * (h - l)[:, None] * _GL_X + 0.5 * (h + l)[:, None]
        integrand = np.exp(a[:, None] * u - K[:, None] * u**1.5 - phi_max[:, None])
        total += 0.5 * (h - l) * (integrand @ _GL_W)
    out[pos] = a * total * np.exp(phi_max)
    return out


def tunneling_energy_range(carrier, ni, band_dos, kT_eV, E_band_to_trap_eV):
    """Hurkx energy range dE (eV) for electrons (carrier=n, band_dos=Nc, Ec-Et)
    or holes (carrier=p, band_dos=Nv, Et-Ev).

    ASSUMPTION (pending check of Hurkx 1992 Sec. II): dE = Ec - Et where the carrier
    density is below ni (depletion, where TAT matters), and kT ln(Nc/n) where it is
    above ni, i.e. dE = min(Ec - Et, kT ln(Nc/n)), clipped at 0. The two branches meet
    at n = ni for a midgap trap.
    """
    carrier = np.maximum(np.asarray(carrier, dtype=float), 1e-300)
    dE = np.minimum(E_band_to_trap_eV, kT_eV * np.log(band_dos / carrier))
    return np.maximum(dE, 0.0)


def field_gamma(T, m_rel=M_HURKX):
    """F_Gamma (V/cm) = sqrt(24 m* (kT)^3) / (q hbar)."""
    kT_J = K_B_EV * T * Q
    return np.sqrt(24.0 * m_rel * M0_SI * kT_J**3) / (Q * HBAR_SI) / 100.0


def gamma_approx(F, T, m_rel=M_HURKX):
    """Hurkx saddle-point closed form for Gamma (see module docstring for validity).

    Checked against `gamma_integral` (dE = 0.56 eV, 300 K): within ~35 % for
    5e3-7e5 V/cm, but once u0 > 1 (~2 MV/cm) it overestimates by orders of magnitude
    (x4.5e7 at 2 MV/cm). Drain junctions reach those fields, so device runs use
    `gamma_fast` in post-processing instead; this form is kept for reference only.
    """
    r = np.abs(np.asarray(F, dtype=float)) / field_gamma(T, m_rel)
    return 2.0 * np.sqrt(3.0 * np.pi) * r * np.exp(r**2)


def saddle_point_u0(F, dE_eV, T, m_rel=M_HURKX):
    """Location u0 of the integrand maximum; the closed form needs 0 < u0 < 1."""
    a = dE_eV / (K_B_EV * T)
    return (2.0 * a / (3.0 * _K(F, dE_eV, m_rel))) ** 2


def hurkx_srh_rate(n, p, ni, tau_n, tau_p, n1, p1, gamma_n=0.0, gamma_p=0.0):
    """Net recombination rate R (cm^-3 s^-1); negative = net generation.

    gamma_n = gamma_p = 0 gives plain SRH.
    """
    num = n * p - ni**2
    den = tau_p / (1.0 + gamma_p) * (n + n1) + tau_n / (1.0 + gamma_n) * (p + p1)
    return num / den


def devsim_gamma_expression(field="ElectricField", F_gamma="tat_Fgamma", F_smooth="tat_Fsmooth"):
    """Symbolic DEVSIM expression for the closed-form Gamma."""
    r = f"(pow({field}^2 + {F_smooth}^2, 0.5) / {F_gamma})"
    return f"2 * pow(3 * 3.141592653589793, 0.5) * {r} * exp({r}^2)"
