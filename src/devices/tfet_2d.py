"""2D double-gate Si n-TFET in DEVSIM: electrostatics + post-processed generation.

Electrostatics: Poisson with two fixed quasi-Fermi levels, holes tied to the source
(phi_p = V_s) and electrons to the drain (phi_n = V_d):

    n = ni exp((psi - phi_n)/Vt),   p = ni exp((phi_p - psi)/Vt)

In the off and ambipolar states the device carries < 1 uA/um, the generated carriers
do not change the potential, and every generated pair is collected, so the terminal
current is I = q * integral(G) dA. That is the same balance the 1D diode test checks
against the full drift-diffusion solution (< 3 %, tests/test_diode_1d.py). It is NOT
valid in the strong on-state, which the audit does not need.

Generation (post-processed on the converged potential):
  * BTBT, local Kane on |grad psi| or nonlocal path model along mesh rows
  * bulk TAT, Hurkx field-enhanced SRH (gamma_fast, valid at all fields)
  * interface TAT, D_it(E) uniform across the gap, Hurkx-enhanced, at the Si/oxide line

Run:  python -m src.devices.tfet_2d [config.yaml]
Units: cm, V, eV; currents reported in A/um (both halves of the double gate).
"""

import copy
import itertools
from dataclasses import dataclass, field

import numpy as np
from scipy.special import erfc

from src.config import load_yaml
from src.constants import EPS0_F_PER_CM, K_B_EV, Q
from src.devsim_env import devsim as ds
from src.materials import silicon
from src.physics import hurkx_tat, kane_btbt, nonlocal_btbt

from devsim.python_packages.model_create import (  # noqa: E402
    CreateEdgeModel,
    CreateEdgeModelDerivatives,
    CreateNodeModel,
    CreateNodeModelDerivative,
    CreateSolution,
)
from devsim.python_packages.simple_physics import CreateOxideContact, GetContactBiasName  # noqa: E402

SI, OX = "si", "ox"
SRC, DRN = "source", "drain"
GATES = ("gate_s", "gate_d")
_ids = itertools.count()


def load_config(path="configs/tfet_2d.yaml"):
    return load_yaml(path)


# ---------------------------------------------------------------- geometry ----

def gate_span(cfg):
    g = cfg["geometry"]
    return -g["gate_overlap_source_cm"], g["L_gate_cm"] - g["gate_underlap_drain_cm"]


def single_gate(cfg):
    """Single-gate SOI: the whole film is simulated, Neumann at the back (thick-BOX limit)."""
    return bool(cfg["geometry"].get("single_gate", False))


def sim_thickness(cfg):
    t = cfg["geometry"]["t_si_cm"]
    return t if single_gate(cfg) else 0.5 * t


def _mesh_lines(points, fine, coarse, refine, window=6e-7):
    """Mesh positions: `fine` spacing within `window` of each point, `coarse` elsewhere."""
    return [(p, fine / refine, coarse / refine, window) for p in points]


def _build(mesh, device, cfg):
    g, m = cfg["geometry"], cfg["mesh"]
    r = float(m.get("refine", 1))
    x0, x1 = -g["L_source_cm"], g["L_gate_cm"] + g["L_drain_cm"]
    ts, tox = sim_thickness(cfg), g["t_ox_cm"]
    gx0, gx1 = gate_span(cfg)
    d = cfg["dielectric"]
    gm = cfg["gate"]
    x_hg = gx1 - d["heterogate_length_cm"]
    x_dm = gx1 - gm["dual_gate_length_cm"]

    ds.create_2d_mesh(mesh=mesh)
    # x lines: fine near junctions and every gate/dielectric edge
    feats = sorted({0.0, g["L_gate_cm"], gx0, gx1, x_hg, x_dm})
    xs = {x0: m["dx_coarse_cm"] / r, x1: m["dx_coarse_cm"] / r}
    # optional fine-spacing ranges (e.g. a diffused drain tail away from the gate edges)
    for a, b in m.get("fine_ranges_cm", []):
        for p in (max(a, x0), min(b, x1)):
            xs[p] = m["dx_junction_cm"] / r
    for p in feats:
        xs[p] = m["dx_junction_cm"] / r
        for s in (-1, 1):  # fine window edges
            q = p + s * 6e-7
            if x0 < q < x1 and all(abs(q - f) > 6e-7 - 1e-12 for f in feats):
                xs.setdefault(q, m["dx_junction_cm"] / r)
    # DEVSIM's 2D mesher only finds contacts on edges between two regions, so the
    # device sits in a thin "air" frame (no equations are solved there).
    pad = 1e-7
    for p, ps in sorted(xs.items()):
        ds.add_2d_mesh_line(mesh=mesh, dir="x", pos=p, ps=ps, ns=ps)
    for p in (x0 - pad, x1 + pad):
        ds.add_2d_mesh_line(mesh=mesh, dir="x", pos=p, ps=pad, ns=pad)
    for p, ps in ((-tox - pad, pad), (-tox, m["dy_ox_cm"] / r), (0.0, m["dy_si_interface_cm"] / r),
                  (ts, m["dy_si_center_cm"] / r), (ts + pad, pad)):
        ds.add_2d_mesh_line(mesh=mesh, dir="y", pos=p, ps=ps, ns=ps)

    bl = 1e-12
    ds.add_2d_region(mesh=mesh, material="gas", region="air")
    ds.add_2d_region(mesh=mesh, material="Silicon", region=SI, xl=x0, xh=x1, yl=0.0, yh=ts, bloat=bl)
    ds.add_2d_region(mesh=mesh, material="Oxide", region=OX, xl=x0, xh=x1, yl=-tox, yh=0.0, bloat=bl)
    ds.add_2d_interface(mesh=mesh, name="si_ox", region0=SI, region1=OX, yl=0.0, yh=0.0,
                        xl=x0, xh=x1, bloat=bl)
    ds.add_2d_contact(mesh=mesh, name=SRC, region=SI, material="metal", xl=x0, xh=x0, yl=0.0, yh=ts, bloat=bl)
    ds.add_2d_contact(mesh=mesh, name=DRN, region=SI, material="metal", xl=x1, xh=x1, yl=0.0, yh=ts, bloat=bl)
    has_d = gm["dual_gate_length_cm"] > 0
    split = x_dm if has_d else gx1
    ds.add_2d_contact(mesh=mesh, name=GATES[0], region=OX, material="metal",
                      xl=gx0, xh=split, yl=-tox, yh=-tox, bloat=bl)
    if has_d:
        # contacts must not share nodes: start half a fine step after the split
        ds.add_2d_contact(mesh=mesh, name=GATES[1], region=OX, material="metal",
                          xl=split + 0.5 * m["dx_junction_cm"] / r, xh=gx1, yl=-tox, yh=-tox, bloat=bl)
    ds.finalize_mesh(mesh=mesh)
    ds.create_device(mesh=mesh, device=device)
    return has_d


# ----------------------------------------------------------------- physics ----

def drain_profile(cfg, x):
    """Tabulated drain donor profile (doping.drain_profile), or None.

    The table gives N(d) vs distance d from the drain implant edge, measured towards the gate;
    the implant edge sits at x = L_gate + offset. N is interpolated in log10, held at the first
    table value inside the drain and at the last value (or the channel level) beyond the tail.
    """
    prof = cfg["doping"].get("drain_profile")
    if not prof:
        return None
    import pandas as pd
    t = pd.read_csv(prof["csv"])
    d_tab = t.iloc[:, 0].to_numpy() * prof.get("x_scale_cm", 1e-7)
    logN = np.log10(t.iloc[:, 1].to_numpy())
    x_imp = cfg["geometry"]["L_gate_cm"] + prof.get("offset_cm", 0.0)
    dist = x_imp - np.asarray(x, dtype=float)
    out = 10 ** np.interp(dist, np.r_[0.0, d_tab], np.r_[logN[0], logN],
                          left=logN[0], right=-np.inf if prof.get("cut_tail", True) else logN[-1])
    return np.where(dist <= 0, 10 ** logN[0], out)


def doping_expressions(cfg):
    """(acceptor, donor) DEVSIM node expressions with erfc junction profiles."""
    g, d = cfg["geometry"], cfg["doping"]
    s = d["junction_sigma_cm"]
    Lg = g["L_gate_cm"]
    bg_a = d["channel_cm3"] if d["channel_type"] == "p" else 0.0
    bg_d = d["channel_cm3"] if d["channel_type"] == "n" else 0.0
    if s > 0:
        k = 1.0 / (np.sqrt(2.0) * s)
        acc = f"{d['source_acceptors_cm3']:.6g} * 0.5 * erfc({k:.6g} * x) + {bg_a:.6g}"
        don = f"{d['drain_donors_cm3']:.6g} * 0.5 * erfc({k:.6g} * ({Lg:.6g} - x)) + {bg_d:.6g}"
    else:
        acc = f"{d['source_acceptors_cm3']:.6g} * step(-x) + {bg_a:.6g}"
        don = f"{d['drain_donors_cm3']:.6g} * step(x - {Lg:.6g}) + {bg_d:.6g}"
    return acc, don


def doping_numpy(cfg, x):
    g, d = cfg["geometry"], cfg["doping"]
    s = d["junction_sigma_cm"]
    Lg = g["L_gate_cm"]
    bg_a = d["channel_cm3"] if d["channel_type"] == "p" else 0.0
    bg_d = d["channel_cm3"] if d["channel_type"] == "n" else 0.0
    prof = drain_profile(cfg, x)
    if s > 0:
        k = 1.0 / (np.sqrt(2.0) * s)
        acc = d["source_acceptors_cm3"] * 0.5 * erfc(k * x) + bg_a
        don = d["drain_donors_cm3"] * 0.5 * erfc(k * (Lg - x)) + bg_d
    else:
        acc = d["source_acceptors_cm3"] * (x < 0) + bg_a
        don = d["drain_donors_cm3"] * (x > Lg) + bg_d
    if prof is not None:
        don = prof + bg_d
    return acc, don


def _si_models(device, cfg):
    T = cfg["temperature_K"]
    ni = float(silicon.intrinsic_density(T))
    for k, v in dict(Permittivity=silicon.EPS_R * EPS0_F_PER_CM, ElectronCharge=Q,
                     n_i=ni, V_t=K_B_EV * T, phin=0.0, phip=0.0).items():
        ds.set_parameter(device=device, region=SI, name=k, value=float(v))
    acc, don = doping_expressions(cfg)
    CreateNodeModel(device, SI, "Acceptors", acc)
    if cfg["doping"].get("drain_profile"):
        xn = np.array(ds.get_node_model_values(device=device, region=SI, name="x"))
        ds.node_solution(device=device, region=SI, name="Donors")
        ds.set_node_values(device=device, region=SI, name="Donors", values=list(doping_numpy(cfg, xn)[1]))
    else:
        CreateNodeModel(device, SI, "Donors", don)
    CreateNodeModel(device, SI, "NetDoping", "Donors - Acceptors")
    # interface-trap charge as a volume density on the interface row (cm^-3, signed like
    # NetDoping); zero unless set by solve_charged_traps()
    ds.node_solution(device=device, region=SI, name="TrapCharge")
    ds.set_node_values(device=device, region=SI, name="TrapCharge",
                       values=[0.0] * len(ds.get_node_model_values(device=device, region=SI, name="x")))
    CreateSolution(device, SI, "Potential")
    for name, expr in (("Electrons", "n_i*exp((Potential - phin)/V_t)"),
                       ("Holes", "n_i*exp((phip - Potential)/V_t)"),
                       ("SpaceCharge", "kahan3(Holes, -Electrons, NetDoping) + TrapCharge"),
                       ("PotentialNodeCharge", "-ElectronCharge * SpaceCharge")):
        CreateNodeModel(device, SI, name, expr)
        CreateNodeModelDerivative(device, SI, name, expr, "Potential")
    for name, expr in (("ElectricField", "(Potential@n0 - Potential@n1)*EdgeInverseLength"),
                       ("PotentialEdgeFlux", "Permittivity * ElectricField")):
        CreateEdgeModel(device, SI, name, expr)
        CreateEdgeModelDerivatives(device, SI, name, expr, "Potential")
    ds.equation(device=device, region=SI, name="PotentialEquation", variable_name="Potential",
                node_model="PotentialNodeCharge", edge_model="PotentialEdgeFlux",
                variable_update="log_damp")
    # Ohmic contacts: charge neutrality with the contact's own quasi-Fermi level
    for c, qf in ((SRC, "phip"), (DRN, "phin")):
        bias = GetContactBiasName(c)
        ds.set_parameter(device=device, name=bias, value=0.0)
        expr = f"Potential - {bias} - V_t * asinh(NetDoping / (2 * n_i))"
        name = f"{c}nodemodel"
        ds.contact_node_model(device=device, contact=c, name=name, equation=expr)
        ds.contact_node_model(device=device, contact=c, name=f"{name}:Potential", equation="1")
        ds.contact_equation(device=device, contact=c, name="PotentialEquation", node_model=name,
                            edge_charge_model="PotentialEdgeFlux")


def _ox_models(device, cfg):
    gx0, gx1 = gate_span(cfg)
    d = cfg["dielectric"]
    x_hg = gx1 - d["heterogate_length_cm"]
    ds.set_parameter(device=device, region=OX, name="Permittivity", value=d["eps_gate"] * EPS0_F_PER_CM)
    ds.set_parameter(device=device, region=OX, name="ElectronCharge", value=Q)
    CreateSolution(device, OX, "Potential")
    ds.edge_from_node_model(device=device, region=OX, node_model="x")
    xm = "(0.5*(x@n0 + x@n1))"
    e0 = EPS0_F_PER_CM
    eps = (f"({e0:.10g}) * ifelse({xm} < {gx0:.10g} || {xm} > {gx1:.10g}, {d['eps_spacer']:.6g}, "
           f"ifelse({xm} > {x_hg:.10g}, {d['eps_gate_drain_side']:.6g}, {d['eps_gate']:.6g}))")
    CreateEdgeModel(device, OX, "EpsEdge", eps)
    for name, expr in (("ElectricField", "(Potential@n0 - Potential@n1)*EdgeInverseLength"),
                       ("PotentialEdgeFlux", "EpsEdge * ElectricField")):
        CreateEdgeModel(device, OX, name, expr)
        CreateEdgeModelDerivatives(device, OX, name, expr, "Potential")
    ds.equation(device=device, region=OX, name="PotentialEquation", variable_name="Potential",
                edge_model="PotentialEdgeFlux", variable_update="default")
    # CreateOxideContact reuses this model if it exists; otherwise it would report gate
    # charge with the scalar Permittivity instead of the position-dependent EpsEdge.
    CreateEdgeModel(device, OX, "contactcharge_edge", "PotentialEdgeFlux")


def _interface(device):
    name = "continuousPotential"
    ds.interface_model(device=device, interface="si_ox", name=name, equation="Potential@r0-Potential@r1")
    ds.interface_model(device=device, interface="si_ox", name=f"{name}:Potential@r0", equation="1")
    ds.interface_model(device=device, interface="si_ox", name=f"{name}:Potential@r1", equation="-1")
    ds.interface_equation(device=device, interface="si_ox", name="PotentialEquation",
                          interface_model=name, type="continuous")


def gate_offset(cfg, drain_side=False):
    """Contact potential offset: psi_gate = Vg - (Phi_m - chi - Eg/2) (psi = 0 at Ei)."""
    gm = cfg["gate"]
    wf = gm["workfunction_drain_side_eV"] if drain_side else gm["workfunction_eV"]
    Eg = float(silicon.bandgap(cfg["temperature_K"]))
    return wf - (gm["chi_si_eV"] + 0.5 * Eg)


# ----------------------------------------------------------------- solver -----

@dataclass
class Device:
    cfg: dict
    name: str
    mesh: str
    has_drain_gate: bool
    Vg: float = 0.0
    Vd: float = 0.0

    def set_bias(self, Vg, Vd):
        b = self.cfg["bias"]
        ds.set_parameter(device=self.name, name=GetContactBiasName(SRC), value=b["V_source"])
        ds.set_parameter(device=self.name, name=GetContactBiasName(DRN), value=Vd)
        ds.set_parameter(device=self.name, region=SI, name="phin", value=Vd)
        ds.set_parameter(device=self.name, region=SI, name="phip", value=b["V_source"])
        ds.set_parameter(device=self.name, name=GetContactBiasName(GATES[0]),
                         value=Vg - gate_offset(self.cfg))
        if self.has_drain_gate:
            ds.set_parameter(device=self.name, name=GetContactBiasName(GATES[1]),
                             value=Vg - gate_offset(self.cfg, drain_side=True))
        self.Vg, self.Vd = Vg, Vd

    def delete(self):
        ds.delete_device(device=self.name)
        ds.delete_mesh(mesh=self.mesh)


def _solve(iters=60):
    ds.solve(type="dc", absolute_error=1e-12, relative_error=1e-10, maximum_iterations=iters)


def build(cfg, verbose=False):
    n = next(_ids)
    dev = Device(cfg=cfg, name=f"tfet2d_{n}", mesh=f"tfet2d_mesh_{n}", has_drain_gate=False)
    if not verbose:
        ds.set_parameter(name="info", value=False)
        ds.set_parameter(name="debug_level", value="none")
    # Poisson-only with fixed quasi-Fermi levels: double precision gives the same potential
    # as 128-bit (max |dpsi| = 2e-16 V at Vg=-1, Vd=1), so skip the slow extended mode here.
    for k in ("extended_solver", "extended_model", "extended_equation"):
        ds.set_parameter(name=k, value=False)
    dev.has_drain_gate = _build(dev.mesh, dev.name, cfg)
    _si_models(dev.name, cfg)
    _ox_models(dev.name, cfg)
    for c in GATES[: 2 if dev.has_drain_gate else 1]:
        ds.set_parameter(device=dev.name, name=GetContactBiasName(c), value=0.0)
        CreateOxideContact(dev.name, OX, c)
    _interface(dev.name)
    dev.set_bias(0.0, 0.0)
    _solve()
    return dev


def ramp_to(dev, Vg, Vd, step=0.1, min_step=1e-3):
    """Continuation in (Vg, Vd) from the current bias; raises if it cannot converge."""
    V0 = np.array([dev.Vg, dev.Vd])
    V1 = np.array([Vg, Vd], dtype=float)
    t, h = 0.0, step / max(np.max(np.abs(V1 - V0)), 1e-12)
    while t < 1.0 - 1e-12:
        tn = min(1.0, t + h)
        Vt_ = V0 + tn * (V1 - V0)
        old = (dev.Vg, dev.Vd)
        dev.set_bias(*Vt_)
        try:
            _solve()
            t = tn
            h *= 1.5
        except ds.error:
            dev.set_bias(*old)
            _solve()
            h /= 2.0
            if h * np.max(np.abs(V1 - V0)) < min_step:
                raise RuntimeError(f"no convergence ramping to Vg={Vg}, Vd={Vd}")


# ---------------------------------------------------------- post-processing ---

@dataclass
class Solution:
    x: np.ndarray            # (nx,) cm
    y: np.ndarray            # (ny,) cm, 0 = Si/oxide interface
    psi: np.ndarray          # (ny, nx) V
    n: np.ndarray
    p: np.ndarray
    vol: np.ndarray          # (ny, nx) node area, cm^2
    Vg: float
    Vd: float
    T: float
    meta: dict = field(default_factory=dict)


def extract(dev):
    """Si-region fields on the tensor grid."""
    get = lambda nm: np.array(ds.get_node_model_values(device=dev.name, region=SI, name=nm))
    x, y = get("x"), get("y")
    xs = np.unique(np.round(x, 14))
    ys = np.unique(np.round(y, 14))
    ix = np.searchsorted(xs, np.round(x, 14))
    iy = np.searchsorted(ys, np.round(y, 14))
    if len(x) != len(xs) * len(ys):
        raise RuntimeError("Si mesh is not a full tensor grid")
    out = {}
    for nm in ("Potential", "Electrons", "Holes", "NodeVolume"):
        a = np.empty((len(ys), len(xs)))
        a[iy, ix] = get(nm)
        out[nm] = a
    return Solution(x=xs, y=ys, psi=out["Potential"], n=out["Electrons"], p=out["Holes"],
                    vol=out["NodeVolume"], Vg=dev.Vg, Vd=dev.Vd, T=dev.cfg["temperature_K"])


def save_solutions(path, sols, cfg):
    """Cache solutions on one mesh (any mix of Vg, Vd) for later post-processing."""
    import json
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {}
    for k, s in enumerate(sols):
        data.update({f"psi_{k}": s.psi, f"n_{k}": s.n, f"p_{k}": s.p})
    s = sols[0]
    np.savez_compressed(path, x=s.x, y=s.y, vol=s.vol, Vg=np.array([q.Vg for q in sols]),
                        Vd=np.array([q.Vd for q in sols]), T=s.T, cfg=json.dumps(cfg), **data)


def load_saved(path):
    """(cfg, [Solution]) written by save_solutions."""
    import json
    z = np.load(path)
    cfg = json.loads(str(z["cfg"]))
    sols = [Solution(x=z["x"], y=z["y"], psi=z[f"psi_{k}"], n=z[f"n_{k}"], p=z[f"p_{k}"],
                     vol=z["vol"], Vg=float(Vg), Vd=float(Vd), T=float(z["T"]))
            for k, (Vg, Vd) in enumerate(zip(z["Vg"], z["Vd"]))]
    return cfg, sols


def field_magnitude(sol):
    dpdy, dpdx = np.gradient(sol.psi, sol.y, sol.x, edge_order=2)
    return np.hypot(dpdx, dpdy)


def _per_um_base(I_per_cm_half, halves=2):
    """Simulated-structure current per cm of width -> A/um (x2 for the double-gate half)."""
    return halves * I_per_cm_half * 1e-4


def generation(sol, cfg, F=None):
    """Return dict of current components (A/um) and the generation maps (cm^-3 s^-1)."""
    ph = cfg["physics"]
    T = sol.T
    kT = K_B_EV * T
    Eg = float(silicon.bandgap(T))
    ni = float(silicon.intrinsic_density(T))
    Nc, Nv = float(silicon.nc(T)), float(silicon.nv(T))
    if F is None:
        F = field_magnitude(sol)
    b = ph["btbt"]
    A, B = kane_btbt.kane_params_at_T(b["A_cm3s"], b["B_Vcm"], float(silicon.bandgap(300.0)), Eg)
    # the nonlocal model has its own calibrated (A, B); defaults to the local ones
    A_nl, B_nl = kane_btbt.kane_params_at_T(b.get("nonlocal_A_cm3s", b["A_cm3s"]),
                                            b.get("nonlocal_B_Vcm", b["B_Vcm"]),
                                            float(silicon.bandgap(300.0)), Eg)
    h = 1 if single_gate(cfg) else 2
    _per_um = lambda I: _per_um_base(I, h)   # noqa: E731
    res, maps = {}, {}

    # local Kane, with the detailed-balance factor (-> 1 in reverse bias)
    np_ = sol.n * sol.p
    eqf = (ni**2 - np_) / (ni**2 + np_)
    G_loc = kane_btbt.kane_generation(F, A, B, b["P"]) * eqf
    maps["btbt_local"] = G_loc
    res["I_btbt_local"] = _per_um(Q * np.sum(G_loc * sol.vol))
    xmid = 0.5 * cfg["geometry"]["L_gate_cm"]
    drain_side = sol.x[None, :] >= xmid
    res["I_btbt_local_drain"] = _per_um(Q * np.sum((G_loc * sol.vol)[np.broadcast_to(drain_side, G_loc.shape)]))

    if b["model"] in ("nonlocal", "both"):
        Ec = -sol.psi + 0.5 * Eg
        Ev = -sol.psi - 0.5 * Eg
        Efn = np.full(sol.x.shape, -sol.Vd)
        Efp = np.full(sol.x.shape, -cfg["bias"]["V_source"])
        dy = _row_widths(sol.y)
        J_rows = np.array([nonlocal_btbt.line_current(sol.x, Ec[j], Ev[j], A_nl, B_nl, Eg, b["P"],
                                                      Efn=Efn, Efp=Efp, kT=kT,
                                                      dE=b["nonlocal_dE_eV"])
                           for j in range(len(sol.y))])
        res["I_btbt_nonlocal_x"] = _per_um(np.sum(J_rows * dy))
        res["I_btbt_nonlocal"] = res["I_btbt_nonlocal_x"]
        if b.get("nonlocal_vertical", False):
            # vertical paths (gate-induced, GIDL-like tunneling where the gate overlaps a doped
            # region). Rows and columns sample different paths of the same 2D band diagram; the
            # larger of the two is reported (a sum would double-count paths that run diagonally).
            Efn_y = np.full(sol.y.shape, -sol.Vd)
            Efp_y = np.full(sol.y.shape, -cfg["bias"]["V_source"])
            dx = _row_widths(sol.x)
            J_cols = np.array([nonlocal_btbt.line_current(sol.y, Ec[:, i], Ev[:, i], A_nl, B_nl, Eg,
                                                          b["P"], Efn=Efn_y, Efp=Efp_y, kT=kT,
                                                          dE=b["nonlocal_dE_eV"])
                               for i in range(len(sol.x))])
            res["I_btbt_nonlocal_y"] = _per_um(np.sum(J_cols * dx))
            res["I_btbt_nonlocal"] = max(res["I_btbt_nonlocal_x"], res["I_btbt_nonlocal_y"])

    tb = ph["tat_bulk"]
    if tb["enabled"]:
        Et = tb["trap_level_eV_from_midgap"]
        n1, p1 = ni * np.exp(Et / kT), ni * np.exp(-Et / kT)
        dEn = hurkx_tat.tunneling_energy_range(sol.n, ni, Nc, kT, 0.5 * Eg - Et)
        dEp = hurkx_tat.tunneling_energy_range(sol.p, ni, Nv, kT, 0.5 * Eg + Et)
        gn = hurkx_tat.gamma_fast(F, dEn, T, tb["m_tunnel"])
        gp = hurkx_tat.gamma_fast(F, dEp, T, tb["m_tunnel"])
        U = hurkx_tat.hurkx_srh_rate(sol.n, sol.p, ni, tb["tau_n_s"], tb["tau_p_s"], n1, p1, gn, gp)
        U0 = hurkx_tat.hurkx_srh_rate(sol.n, sol.p, ni, tb["tau_n_s"], tb["tau_p_s"], n1, p1)
        maps["tat_bulk"] = -U
        res["I_tat_bulk"] = _per_um(Q * np.sum(-U * sol.vol))
        res["I_srh_bulk"] = _per_um(Q * np.sum(-U0 * sol.vol))

    it = ph["interface_traps"]
    if it["enabled"]:
        res["I_it"] = _per_um(interface_generation(sol, cfg, F[0], ni, Eg, Nc, Nv, kT))

    res["I_total_local"] = res["I_btbt_local"] + res.get("I_tat_bulk", 0.0) + res.get("I_it", 0.0)
    if "I_btbt_nonlocal" in res:
        res["I_total_nonlocal"] = res["I_btbt_nonlocal"] + res.get("I_tat_bulk", 0.0) + res.get("I_it", 0.0)
    return res, maps


def _row_widths(y):
    w = np.zeros_like(y)
    w[1:] += 0.5 * np.diff(y)
    w[:-1] += 0.5 * np.diff(y)
    return w


def dit_shape(it, e, Eg):
    """Relative D_it(E) at trap level e (eV from midgap); D_it_cm2eV sets the scale.

    uniform: 1 (D_it_cm2eV is the density at every level).
    u_shape: 1 + r * [exp(-(Eg/2 - e)/w) + exp(-(Eg/2 + e)/w)], the usual U-shaped Si/oxide
      distribution with edge-to-midgap ratio ~r (u_ratio) and tail width w; D_it_cm2eV is the
      midgap value.
    gaussian: a peak at gauss_center_eV (from midgap) of width gauss_width_eV, normalized so the
      total number of traps equals that of the uniform case (D_it_cm2eV * Eg); D_it_cm2eV is then
      the gap-averaged density.
    """
    prof = it.get("profile", "uniform")
    if prof == "uniform":
        return 1.0
    if prof == "gaussian":
        e0, w = it.get("gauss_center_eV", 0.0), it.get("gauss_width_eV", 0.05)
        return Eg / (w * np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * ((e - e0) / w) ** 2)
    if prof == "u_shape":
        r, w = it.get("u_ratio", 50.0), it.get("u_width_eV", 0.1)
        return 1.0 + r * (np.exp(-(0.5 * Eg - e) / w) + np.exp(-(0.5 * Eg + e) / w))
    raise ValueError(f"unknown D_it profile {prof!r}")


def dit_spatial(it, x, cfg):
    """Relative D_it along the interface: 1 + (r - 1) exp(-|x - x_d| / L) around the drain
    junction x_d = L_gate (implant damage at the drain-side gate edge, ASSUMPTION; r =
    damage_ratio, L = damage_length_cm). 1 everywhere when damage_ratio is absent or 1.
    gate_only: zero outside the gate span (spacer interfaces trap-free)."""
    out = np.ones_like(x, dtype=float)
    r = it.get("damage_ratio", 1.0)
    if r != 1.0:
        L = it.get("damage_length_cm", 1.0e-6)
        out = out + (r - 1.0) * np.exp(-np.abs(x - cfg["geometry"]["L_gate_cm"]) / L)
    if it.get("gate_only", False):             # no traps under the spacers (gate-stack interface only)
        g0, g1 = gate_span(cfg)
        out = np.where((x >= g0 - 1e-12) & (x <= g1 + 1e-12), out, 0.0)
    return out


def btbt_terms(sol, cfg):
    """Parameter-free BTBT terms of one solution: {name: (F, w, power)} such that
    I (A/um) = sum(w * A * F**power * exp(-B/F)) for any Kane (A, B) at this temperature.
    Names: local, nonlocal_x, nonlocal_y. Used for uncertainty propagation."""
    b = cfg["physics"]["btbt"]
    Eg = float(silicon.bandgap(sol.T))
    kT = K_B_EV * sol.T
    ni = float(silicon.intrinsic_density(sol.T))
    h = 1 if single_gate(cfg) else 2
    F = field_magnitude(sol)
    np_ = sol.n * sol.p
    eqf = (ni**2 - np_) / (ni**2 + np_)
    out = {"local": (F.ravel(), (Q * sol.vol * eqf).ravel() * h * 1e-4, b["P"])}
    Ec, Ev = -sol.psi + 0.5 * Eg, -sol.psi - 0.5 * Eg
    for name, coord, lines, widths in (("nonlocal_x", sol.x, [Ec, Ev], _row_widths(sol.y)),
                                       ("nonlocal_y", sol.y, [Ec.T, Ev.T], _row_widths(sol.x))):
        Fs, ws = [], []
        efn = np.full(coord.shape, -sol.Vd)
        efp = np.full(coord.shape, -cfg["bias"]["V_source"])
        for j in range(lines[0].shape[0]):
            f, w = nonlocal_btbt.path_terms(coord, lines[0][j], lines[1][j], Eg, efn, efp, kT,
                                            b["nonlocal_dE_eV"])
            Fs.append(f); ws.append(w * widths[j] * Q * h * 1e-4)
        out[name] = (np.concatenate(Fs), np.concatenate(ws), b["P"] - 1.0)
    return out


def btbt_from_terms(terms, A, B):
    F, w, pw = terms
    if F.size == 0:
        return 0.0
    ok = F > 0
    return float(np.sum(w[ok] * A * F[ok] ** pw * np.exp(-B / F[ok])))


def interface_levels(it, Eg):
    """Trap levels (eV from midgap) and their spacing: n_levels across the gap, ends dropped."""
    Et = np.linspace(-0.5 * Eg, 0.5 * Eg, it["n_levels"])[1:-1]
    return Et, Et[1] - Et[0]


def interface_generation(sol, cfg, F_if, ni, Eg, Nc, Nv, kT, per_x=False):
    """Interface-trap generation integrated along the Si/oxide line (per cm width, A/cm).

    Per trap level e (sheet density N = D_it(e, x) dE), Hurkx field-enhanced SRH
    (hurkx_tat.hurkx_srh_rate, the same rate as bulk TAT):

        R_s = (np - ni^2) / [ tau_p/(1 + G_p) (n + n1) + tau_n/(1 + G_n) (p + p1) ],
        tau_n = 1/(sigma_n v_th N),  tau_p = 1/(sigma_p v_th N)

    G_n uses the electron range (Ec - Et) and G_p the hole range (Et - Ev), as in Hurkx 1992:
    the n1 term is hole emission, the p1 term electron emission. Options (interface_traps):
    sigma_n_cm2 / sigma_p_cm2 (default sigma_cm2), m_tunnel (default Hurkx 0.25), the
    energy profile (dit_shape) and gate-edge damage (dit_spatial).
    per_x=True returns the contribution of each interface node (A/cm) instead of the sum.
    """
    it = cfg["physics"]["interface_traps"]
    n, p = sol.n[0], sol.p[0]
    Et, dEt = interface_levels(it, Eg)
    sig_n = it.get("sigma_n_cm2", it["sigma_cm2"])
    sig_p = it.get("sigma_p_cm2", it["sigma_cm2"])
    m = it.get("m_tunnel", hurkx_tat.M_HURKX)
    space = dit_spatial(it, sol.x, cfg)
    Rs = np.zeros_like(n)
    for e in Et:
        N = it["D_it_cm2eV"] * dit_shape(it, e, Eg) * dEt * space
        n1, p1 = ni * np.exp(e / kT), ni * np.exp(-e / kT)
        if it["field_enhanced"]:
            gn = hurkx_tat.gamma_fast(F_if, hurkx_tat.tunneling_energy_range(n, ni, Nc, kT, 0.5 * Eg - e), sol.T, m)
            gp = hurkx_tat.gamma_fast(F_if, hurkx_tat.tunneling_energy_range(p, ni, Nv, kT, 0.5 * Eg + e), sol.T, m)
        else:
            gn = gp = 0.0
        # hurkx_srh_rate with tau_n = 1/(sigma_n v N), tau_p = 1/(sigma_p v N), multiplied out
        # so that N -> 0 (far tail of a narrow spectrum) gives 0 instead of an overflow
        Rs += N * it["v_th_cms"] * (n * p - ni**2) / ((n + n1) / (sig_p * (1.0 + gp))
                                                      + (p + p1) / (sig_n * (1.0 + gn)))
    dx = _row_widths(sol.x)
    return Q * -Rs * dx if per_x else Q * np.sum(-Rs * dx)


def transfer(cfg, Vg_list, Vd=None, dev=None):
    """Sweep Vg at fixed Vd; returns list of (Vg, result dict). Builds/deletes the device."""
    own = dev is None
    dev = dev or build(cfg)
    Vd = cfg["bias"]["V_drain"] if Vd is None else Vd
    out = []
    try:
        ramp_to(dev, Vg_list[0], Vd)
        for Vg in Vg_list:
            ramp_to(dev, Vg, Vd)
            sol = extract(dev)
            res, _ = generation(sol, cfg)
            out.append((Vg, res))
    finally:
        if own:
            dev.delete()
    return out


def variant(cfg, **changes):
    """Deep-copied config with dotted-key changes, e.g. variant(c, **{"doping.drain_donors_cm3": 5e18})."""
    c = copy.deepcopy(cfg)
    for k, v in changes.items():
        d = c
        *path, last = k.split(".")
        for p in path:
            d = d[p]
        if last not in d:
            raise KeyError(k)
        d[last] = v
    return c


if __name__ == "__main__":
    import sys
    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "configs/tfet_2d.yaml")
    cfg["physics"]["btbt"]["model"] = "both"
    for Vg, r in transfer(cfg, [0.0, -0.5, -1.0, -1.5]):
        print(f"Vg={Vg:+.2f}  " + "  ".join(f"{k}={v:.3e}" for k, v in r.items()))


# ------------------------------------------------------- charged interface traps ----

def interface_trap_charge(sol, cfg):
    """Steady-state interface-trap charge Q_it(x) / q (cm^-2) on the gate interface.

    Amphoteric traps with the charge-neutrality level at midgap (levels below: donor-like,
    positive when empty; above: acceptor-like, negative when filled), SRH occupancy with equal
    capture cross-sections, f = (n + p1) / (n + n1 + p + p1), D(E) as in interface_generation.
    ASSUMPTION: no field enhancement of the occupancy (it changes f only where the generation
    current is large, i.e. in a few nm at the drain-side gate edge).
    """
    it = cfg["physics"]["interface_traps"]
    T = sol.T
    kT = K_B_EV * T
    Eg = float(silicon.bandgap(T))
    ni = float(silicon.intrinsic_density(T))
    n, p = sol.n[0], sol.p[0]
    Et, dE = interface_levels(it, Eg)
    space = dit_spatial(it, sol.x, cfg)
    Qs = np.zeros_like(n)
    for e in Et:
        n1, p1 = ni * np.exp(e / kT), ni * np.exp(-e / kT)
        f = (n + p1) / (n + n1 + p + p1)
        D = it["D_it_cm2eV"] * dit_shape(it, e, Eg) * dE * space
        Qs += D * ((1.0 - f) if e < 0 else -f)
    return Qs


def solve_charged_traps(dev, cfg, tol=1e-4, iters=40, damp=0.7):
    """Fixed-point loop: Q_it from the current solution -> TrapCharge on the interface row ->
    Poisson; repeated until the interface potential changes by < tol (V). Returns iterations."""
    get = lambda nm: np.array(ds.get_node_model_values(device=dev.name, region=SI, name=nm))
    x, y, vol = get("x"), get("y"), get("NodeVolume")
    on_if = np.isclose(y, 0.0, atol=1e-14)
    xs = np.unique(np.round(x, 14))
    wx = _row_widths(xs)
    ix = np.searchsorted(xs, np.round(x, 14))
    ends = np.isclose(x, xs[0]) | np.isclose(x, xs[-1])      # keep the contacts neutral
    rho = np.zeros_like(x)
    psi_old = None
    for k in range(iters):
        sol = extract(dev)
        Qs = interface_trap_charge(sol, cfg)                   # on the x grid of row 0
        target = np.zeros_like(x)
        sel = on_if & ~ends
        target[sel] = Qs[ix[sel]] * wx[ix[sel]] / vol[sel]     # sheet density -> node density
        rho = rho + damp * (target - rho)
        ds.set_node_values(device=dev.name, region=SI, name="TrapCharge", values=list(rho))
        _solve()
        psi = extract(dev).psi[0]
        if psi_old is not None and np.max(np.abs(psi - psi_old)) < tol:
            return k + 1
        psi_old = psi
    raise RuntimeError("charged-trap fixed point did not converge")


def clear_trap_charge(dev):
    nn = len(ds.get_node_model_values(device=dev.name, region=SI, name="x"))
    ds.set_node_values(device=dev.name, region=SI, name="TrapCharge", values=[0.0] * nn)
    _solve()
