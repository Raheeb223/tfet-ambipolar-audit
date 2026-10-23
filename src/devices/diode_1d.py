"""1D abrupt Si p+/n+ diode in DEVSIM with SRH + local Kane BTBT.

Solver-verification device (HANDOFF §9 step 1). Reverse-bias sweep, reports
J vs V and J vs max field, plus two internal checks per bias point:
  * J_btbt_int = q * sum(G_edge * EdgeLength)  (what the edge_volume_model integrates)
  * J_srh_int  = -q * sum(U_node * NodeVolume)
In steady state the anode current satisfies J + J_btbt_int + J_srh_int = 0
(DEVSIM's anode current is negative under reverse bias; minority diffusion from
the contacts is negligible at 1e19 doping).

Run:  python -m src.devices.diode_1d [config.yaml] [--outdir results/diode_1d]
Units: cm, V, A/cm^2.

Not yet included (HANDOFF §6.1, TODO): Fermi-Dirac statistics, bandgap narrowing,
doping/field-dependent mobility, Hurkx TAT.
"""

import argparse
import itertools
import json
import os
from dataclasses import asdict, dataclass, field

import numpy as np

from src.config import load_yaml
from src.constants import EPS0_F_PER_CM, K_B_EV, Q
from src.devsim_env import devsim as ds
from src.materials import silicon
from src.physics import kane_btbt

from devsim.python_packages.model_create import (  # noqa: E402  (after devsim_env)
    CreateEdgeModel,
    CreateEdgeModelDerivatives,
    CreateNodeModel,
    CreateSolution,
)
from devsim.python_packages.simple_dd import CreateBernoulli, CreateElectronCurrent, CreateHoleCurrent
from devsim.python_packages.simple_physics import (
    CreatePE,
    CreateSiliconDriftDiffusionAtContact,
    CreateSiliconPotentialOnly,
    CreateSiliconPotentialOnlyContact,
    CreateSRH,
    GetContactBiasName,
)

REGION = "Si"
ANODE, CATHODE = "top", "bot"
ECE, HCE = "ElectronContinuityEquation", "HoleContinuityEquation"
_ids = itertools.count()

# Field written out explicitly so DEVSIM's symbolic diff sees Potential@n0/@n1 directly.
_FIELD = "((Potential@n0 - Potential@n1) * EdgeInverseLength)"
_NP_EDGE = "pow(Electrons@n0 * Holes@n0 * Electrons@n1 * Holes@n1, 0.5)"


@dataclass
class SweepResult:
    config: dict
    V: list = field(default_factory=list)
    J: list = field(default_factory=list)            # total anode current density, A/cm^2
    J_btbt_int: list = field(default_factory=list)
    J_srh_int: list = field(default_factory=list)
    F_max: list = field(default_factory=list)        # V/cm
    failed_V: list = field(default_factory=list)     # bias points that did not converge (logged, not dropped silently)
    x_edge: list = field(default_factory=list)       # field profile at the last converged bias
    F_edge: list = field(default_factory=list)
    G_edge: list = field(default_factory=list)


def load_config(path):
    return load_yaml(path)


def _build_mesh(mesh, device, cfg):
    g, m = cfg["geometry"], cfg["mesh"]
    r = float(m.get("refine", 1))
    xj = g["length_p_cm"]
    L = xj + g["length_n_cm"]
    ds.create_1d_mesh(mesh=mesh)
    ds.add_1d_mesh_line(mesh=mesh, pos=0.0, ps=m["spacing_contact_cm"] / r, tag=ANODE)
    ds.add_1d_mesh_line(mesh=mesh, pos=xj, ps=m["spacing_junction_cm"] / r, tag="junction")
    ds.add_1d_mesh_line(mesh=mesh, pos=L, ps=m["spacing_contact_cm"] / r, tag=CATHODE)
    ds.add_1d_contact(mesh=mesh, name=ANODE, tag=ANODE, material="metal")
    ds.add_1d_contact(mesh=mesh, name=CATHODE, tag=CATHODE, material="metal")
    ds.add_1d_region(mesh=mesh, material="Si", region=REGION, tag1=ANODE, tag2=CATHODE)
    ds.finalize_mesh(mesh=mesh)
    ds.create_device(mesh=mesh, device=device)
    return xj


def _set_parameters(device, cfg):
    T = cfg["temperature_K"]
    ni = float(silicon.intrinsic_density(T))
    Vt = K_B_EV * T
    Et = cfg["srh"]["trap_level_eV_from_midgap"]
    p = dict(
        Permittivity=silicon.EPS_R * EPS0_F_PER_CM,
        ElectronCharge=Q,
        n_i=ni,
        T=T,
        V_t=Vt,
        mu_n=cfg["mobility"]["mu_n"],
        mu_p=cfg["mobility"]["mu_p"],
        taun=cfg["srh"]["tau_n_s"],
        taup=cfg["srh"]["tau_p_s"],
        n1=ni * np.exp(Et / Vt),
        p1=ni * np.exp(-Et / Vt),
    )
    b = cfg["btbt"]
    A, B = kane_btbt.kane_params_at_T(
        b["A_cm3s"], b["B_Vcm"],
        float(silicon.bandgap(b["B_reference_T_K"])), float(silicon.bandgap(T)),
        scale_A=b["scale_A_with_T"],
    )
    p.update(btbt_A=A, btbt_B=B, btbt_P=b["P"], btbt_Fsmooth=b["F_smooth_Vcm"],
             btbt_on=1.0 if b["enabled"] else 0.0)
    for k, v in p.items():
        ds.set_parameter(device=device, region=REGION, name=k, value=float(v))
    return A, B


def _set_doping(device, cfg, xj):
    d = cfg["doping"]
    CreateNodeModel(device, REGION, "Acceptors", f"{d['acceptors_cm3']:g} * step({xj:g} - x)")
    CreateNodeModel(device, REGION, "Donors", f"{d['donors_cm3']:g} * step(x - {xj:g})")
    CreateNodeModel(device, REGION, "NetDoping", "Donors - Acceptors")


def _create_btbt(device, cfg):
    eq = "(n_i^2 - {np}) / (n_i^2 + {np})".format(np=_NP_EDGE) if cfg["btbt"]["detailed_balance_factor"] else None
    G = "btbt_on * " + kane_btbt.devsim_expression(field=_FIELD, eq_factor=eq)
    for name, expr in (("BTBT_G", G), ("ElectronBTBT", f"ElectronCharge * {G}"),
                       ("HoleBTBT", f"-ElectronCharge * {G}")):
        CreateEdgeModel(device, REGION, name, expr)
        for v in ("Potential", "Electrons", "Holes"):
            CreateEdgeModelDerivatives(device, REGION, name, expr, v)
    # 1D: each edge gives half its length to each end node.
    CreateEdgeModel(device, REGION, "EdgeNodeVolume", "0.5 * EdgeCouple * EdgeLength")
    ds.set_parameter(name="edge_node0_volume_model", value="EdgeNodeVolume")
    ds.set_parameter(name="edge_node1_volume_model", value="EdgeNodeVolume")


def _create_drift_diffusion(device, cfg):
    CreatePE(device, REGION)
    CreateBernoulli(device, REGION)
    CreateSRH(device, REGION)
    _create_btbt(device, cfg)
    CreateElectronCurrent(device, REGION, "mu_n")
    CreateHoleCurrent(device, REGION, "mu_p")
    for name, var, charge, edge, gen, btbt in (
        (ECE, "Electrons", "-ElectronCharge * Electrons", "ElectronCurrent", "ElectronGeneration", "ElectronBTBT"),
        (HCE, "Holes", "ElectronCharge * Holes", "HoleCurrent", "HoleGeneration", "HoleBTBT"),
    ):
        tname = f"{var}Charge"
        CreateNodeModel(device, REGION, tname, charge)
        CreateNodeModel(device, REGION, f"{tname}:{var}", f"simplify(diff({charge}, {var}))")
        ds.equation(device=device, region=REGION, name=name, variable_name=var,
                    time_node_model=tname, edge_model=edge, node_model=gen,
                    edge_volume_model=btbt, variable_update="positive")
    for c in (ANODE, CATHODE):
        CreateSiliconDriftDiffusionAtContact(device, REGION, c)


def _solve(rel=1e-10, abs_=1e10, iters=40):
    ds.solve(type="dc", absolute_error=abs_, relative_error=rel, maximum_iterations=iters)


def _record(device, res, V):
    I = sum(ds.get_contact_current(device=device, contact=ANODE, equation=e) for e in (ECE, HCE))
    F = np.array(ds.get_edge_model_values(device=device, region=REGION, name="ElectricField"))
    G = np.array(ds.get_edge_model_values(device=device, region=REGION, name="BTBT_G"))
    Le = np.array(ds.get_edge_model_values(device=device, region=REGION, name="EdgeLength"))
    U = np.array(ds.get_node_model_values(device=device, region=REGION, name="USRH"))
    Vn = np.array(ds.get_node_model_values(device=device, region=REGION, name="NodeVolume"))
    x0 = np.array(ds.get_edge_model_values(device=device, region=REGION, name="x@n0"))
    x1 = np.array(ds.get_edge_model_values(device=device, region=REGION, name="x@n1"))
    res.V.append(V)
    res.J.append(I)
    res.J_btbt_int.append(Q * float(np.sum(G * Le)))
    res.J_srh_int.append(-Q * float(np.sum(U * Vn)))
    res.F_max.append(float(np.max(np.abs(F))))
    order = np.argsort(0.5 * (x0 + x1))
    res.x_edge = (0.5 * (x0 + x1))[order].tolist()
    res.F_edge = F[order].tolist()
    res.G_edge = G[order].tolist()


def run(cfg, verbose=False):
    """Build the diode, solve equilibrium, sweep the anode bias. Returns SweepResult."""
    n = next(_ids)
    device, mesh = f"diode1d_{n}", f"diode1d_mesh_{n}"
    res = SweepResult(config=cfg)
    if not verbose:
        ds.set_parameter(name="info", value=False)
        ds.set_parameter(name="debug_level", value="none")
    try:
        xj = _build_mesh(mesh, device, cfg)
        _set_parameters(device, cfg)
        _set_doping(device, cfg, xj)
        ds.edge_from_node_model(device=device, region=REGION, node_model="x")

        CreateSolution(device, REGION, "Potential")
        CreateSiliconPotentialOnly(device, REGION)
        for c in (ANODE, CATHODE):
            ds.set_parameter(device=device, name=GetContactBiasName(c), value=0.0)
            CreateSiliconPotentialOnlyContact(device, REGION, c)
        _solve(abs_=1.0)

        CreateSolution(device, REGION, "Electrons")
        CreateSolution(device, REGION, "Holes")
        ds.set_node_values(device=device, region=REGION, name="Electrons", init_from="IntrinsicElectrons")
        ds.set_node_values(device=device, region=REGION, name="Holes", init_from="IntrinsicHoles")
        _create_drift_diffusion(device, cfg)
        _solve()

        s = cfg["sweep"]
        V, stop = s["anode_start_V"], s["anode_stop_V"]
        step = abs(s["step_V"]) * (1 if stop > V else -1)
        ds.set_parameter(device=device, name=GetContactBiasName(ANODE), value=V)
        _record(device, res, V)
        while abs(stop - V) > 1e-12:
            target = V + step if abs(step) < abs(stop - V) else stop
            ds.set_parameter(device=device, name=GetContactBiasName(ANODE), value=target)
            try:
                _solve()
            except ds.error:
                res.failed_V.append(target)
                ds.set_parameter(device=device, name=GetContactBiasName(ANODE), value=V)
                _solve()  # restore last good point
                step /= 2.0
                if abs(step) < s["min_step_V"]:
                    break
                continue
            V = target
            _record(device, res, V)
            step = np.sign(step) * min(abs(step) * 1.5, abs(s["step_V"]))
        return res
    finally:
        ds.delete_device(device=device)
        ds.delete_mesh(mesh=mesh)


def save(res, outdir, tag):
    import pandas as pd
    os.makedirs(outdir, exist_ok=True)
    pd.DataFrame(dict(V=res.V, J_A_cm2=res.J, J_btbt_int=res.J_btbt_int,
                      J_srh_int=res.J_srh_int, F_max_V_cm=res.F_max)).to_csv(
        os.path.join(outdir, f"{tag}_iv.csv"), index=False)
    pd.DataFrame(dict(x_cm=res.x_edge, F_V_cm=res.F_edge, G_btbt=res.G_edge)).to_csv(
        os.path.join(outdir, f"{tag}_profile.csv"), index=False)
    meta = {k: v for k, v in asdict(res).items() if k in ("config", "failed_V")}
    with open(os.path.join(outdir, f"{tag}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def plot(results, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    for label, r in results.items():
        V, J = np.array(r.V), np.abs(r.J)
        m = V < 0
        ax1.semilogy(-V[m], J[m], label=label)
        ax2.semilogy(1e-6 * np.array(r.F_max)[m], J[m], label=label)
    ax1.set_xlabel("Reverse bias  -V_anode (V)")
    ax2.set_xlabel("Max junction field (MV/cm)")
    for ax in (ax1, ax2):
        ax.set_ylabel("|J| (A/cm$^2$)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    fig.suptitle("1D Si p+/n+ diode (1e19/1e19), DEVSIM, 300 K: verification run")
    fig.tight_layout()
    path = os.path.join(outdir, "diode_1d_J_vs_V_and_F.png")
    fig.savefig(path, dpi=150)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?", default="configs/diode_1d.yaml")
    ap.add_argument("--outdir", default="results/diode_1d")
    a = ap.parse_args()
    cfg = load_config(a.config)
    cfg_off = json.loads(json.dumps(cfg))
    cfg_off["btbt"]["enabled"] = False
    results = {"SRH only": run(cfg_off), "SRH + Kane BTBT (default A,B)": run(cfg)}
    for label, r in results.items():
        tag = "srh_only" if label == "SRH only" else "srh_btbt"
        save(r, a.outdir, tag)
        err = np.abs(np.array(r.J) + np.array(r.J_btbt_int) + np.array(r.J_srh_int))
        print(f"{label}: V_end={r.V[-1]:.3f} V  J={r.J[-1]:.4e} A/cm2  "
              f"Fmax={r.F_max[-1]:.3e} V/cm  balance err max={err.max():.2e}  failed={r.failed_V}")
    print("plot:", plot(results, a.outdir))


if __name__ == "__main__":
    main()
