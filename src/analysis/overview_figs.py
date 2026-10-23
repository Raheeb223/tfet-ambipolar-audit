"""Overview figures for talks and the graphical abstract, in black and white only.

fig_overview_block : block diagram of the whole study (data -> models -> calibration ->
                     blind validation -> audit -> metrics -> findings).
fig_overview_concept : illustrated device, band diagram and transfer-curve sketch that
                       explain the central idea (interface TAT caps ambipolar suppression).

Numbers quoted in the boxes are read from paper/generated/*.tex, so the figures never
disagree with the paper. Panels (b) and (c) of the concept figure are schematic and are
labelled as such; no simulated data is plotted in them.

Run: python -m src.analysis.overview_figs
"""

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
GEN = ROOT / "paper" / "generated"
OUT = ROOT / "paper" / "figs"

plt.rcParams.update({"font.size": 7, "font.family": "serif",
                     "mathtext.fontset": "dejavuserif", "hatch.linewidth": 0.5})

K = "black"


def macros():
    """All \\newcommand{\\name}{value} pairs from the generated number files."""
    out = {}
    for f in GEN.glob("numbers*.tex"):
        for name, val in re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}\s*$", f.read_text(),
                                    flags=re.M):
            out[name] = val
    return out


def tex_num(s):
    """'1.9\\times10^{12}' -> '$1.9\\times10^{12}$' for mathtext."""
    return "$" + s.replace("\\times", r"\times ") + "$"


# ------------------------------------------------------------------------ block diagram
def _box(ax, x, y, w, h, text, fill="white", lw=0.8, size=5.7, weight="normal", ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.08",
                                fc=fill, ec=K, lw=lw, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", size=size,
            weight=weight, linespacing=1.25)


def _arrow(ax, p, q, style="-|>", ls="-", lw=0.9, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=8, lw=lw, color=K,
                                 ls=ls, connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=0, shrinkB=0))


def fig_block(m):
    fig, ax = plt.subplots(figsize=(7.16, 4.4))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 12)
    ax.axis("off")

    cols = [(0.2, "1  DATA"), (4.2, "2  PHYSICS"), (8.2, "3  CALIBRATE, TEST"),
            (12.2, "4  AUDIT"), (16.2, "5  RESULTS")]
    W = 3.6
    for x, title in cols:
        ax.add_patch(FancyBboxPatch((x, 0.9), W, 10.2, boxstyle="round,pad=0,rounding_size=0.12",
                                    fc="#f2f2f2", ec=K, lw=0.6))
        _box(ax, x, 10.35, W, 0.75, title, fill="black", size=6.6, weight="bold")
        ax.texts[-1].set_color("white")

    # 1 data
    x = cols[0][0] + 0.2
    w = W - 0.4
    _box(ax, x, 8.3, w, 1.7, "Measured Si junction\ntunneling current\n(9 decades)\n"
         r"$\rightarrow$ calibration only")
    _box(ax, x, 5.7, w, 2.3, "4 independent measured\nTFET data sets\n"
         "(Wan 2011: SiO$_2$, HfO$_2$;\nBiswas 2012;\nDer Agopian 2012)\n"
         r"$\rightarrow$ blind validation only")
    _box(ax, x, 3.7, w, 1.7, "Published trap values\n(activation energy,\n"
         r"$D_{it}$, trap level)" "\n" r"$\rightarrow$ cross-checks")
    _box(ax, x, 1.2, w, 2.2, "21 digitised curves,\neach with source,\npage, figure and role",
         ls="--")

    # 2 physics
    x = cols[1][0] + 0.2
    _box(ax, x, 8.6, w, 1.4, "DEVSIM 2-D Poisson\n(open source)\nfixed quasi-Fermi levels")
    _box(ax, x, 6.5, w, 1.8, "Band-to-band\ntunneling (BTBT)\nKane model:\nlocal or nonlocal")
    _box(ax, x, 4.7, w, 1.5, "Bulk trap-assisted\ntunneling (TAT)\nHurkx model")
    _box(ax, x, 2.9, w, 1.5, "Interface TAT\nlinear in trap density\n"
         r"$D_{it}$ at Si/oxide")
    _box(ax, x, 1.2, w, 1.4, r"$I_D = q\int G\,dV$" "\nevery model\nunit-tested", ls="--")

    # 3 calibration / validation
    x = cols[2][0] + 0.2
    _box(ax, x, 8.0, w, 2.0, "Fit Kane A, B to\nSi junction data\n"
         f"rms error {m.get('nCalRmsNl', '?')} decade\n(nonlocal)")
    _box(ax, x, 6.6, w, 1.0, "FREEZE parameters", fill="black", weight="bold")
    ax.texts[-1].set_color("white")
    _box(ax, x, 3.9, w, 2.3, "Blind test on measured\nTFETs, no refitting\n"
         f"nonlocal: {m.get('nValTenNlMin', '?')}–{m.get('nValTenNlMax', '?')} decade\n"
         f"local: {m.get('nValTenLocMin', '?')}–{m.get('nValTenLocMax', '?')} decades")
    _box(ax, x, 1.2, w, 2.4, "Rule: never calibrate\non validation data;\n"
         "every assumption\nlogged in configs", ls="--")

    # 4 audit
    x = cols[3][0] + 0.2
    _box(ax, x, 8.3, w, 1.7, "Reference double-gate\nSi n-TFET\n"
         r"$L_g$ = 50 nm, $t_{si}$ = 10 nm" "\nHfO$_2$ gate oxide")
    _box(ax, x, 5.5, w, 2.5, "4 suppression techniques\n(one change each)\n"
         "HG: hetero-gate oxide\nLDD: low drain doping\n"
         "UL: gate–drain underlap\nDWF: drain work function")
    _box(ax, x, 3.3, w, 1.9, "Sweeps\n" r"$D_{it}$: $10^{10}$–$10^{14}$ cm$^{-2}$eV$^{-1}$"
         "\nT: 300–398 K\n" r"$V_{GS}$: $-1$ to $-0.2$ V", size=5.3)
    _box(ax, x, 1.2, w, 1.8, "Robustness checks:\ntrap charge, trap profile,\n"
         "parameter UQ, straggle,\nangled tunnel paths", ls="--")

    # 5 metrics and findings
    x = cols[4][0] + 0.2
    _box(ax, x, 8.3, w, 1.7, "Suppression S\n(decades vs reference)\n"
         r"trap-limited ceiling $S_\infty$")
    _box(ax, x, 6.3, w, 1.7, "Crossover trap\n" r"density $D_x$, where" "\n"
         "interface TAT equals\nambipolar BTBT")
    dl = tex_num(m["nDcrossLDD"]) if "nDcrossLDD" in m else "?"
    _box(ax, x, 3.3, w, 2.7, "KEY FINDING\nInterface TAT is generated\nat the ambipolar junction\n"
         r"$V_{GS}=-1$ V: $D_x>10^{14}$" "\n" f"(LDD: {dl})\n"
         r"$V_{GS}=-0.2$ V: $D_x<10^{11}$" "\n"
         f"{m.get('nBiasLostElevenMin', '?')}–{m.get('nBiasLostMax', '?')} decades lost",
         lw=1.4)
    _box(ax, x, 1.2, w, 1.8, r"Proposal: report $D_x$" "\nand the trap-limited\nceiling "
         "with every\nsuppression claim", lw=1.4)

    # flow arrows between columns (in the gaps only, never across a box)
    for (xa, _), (xb, _) in zip(cols[:-1], cols[1:]):
        _arrow(ax, (xa + W + 0.03, 10.72), (xb - 0.03, 10.72), lw=1.2)
    _arrow(ax, (3.82, 9.15), (4.38, 9.15))                  # data -> model
    _arrow(ax, (7.82, 9.3), (8.38, 9.0))                    # model -> fit
    _arrow(ax, (10.0, 8.0), (10.0, 7.6))                    # fit -> freeze
    _arrow(ax, (10.0, 6.6), (10.0, 6.2))                    # freeze -> blind
    _arrow(ax, (11.82, 7.1), (12.38, 7.1))                  # frozen -> audit
    _arrow(ax, (15.82, 6.75), (16.38, 7.15))                # audit -> metrics
    _arrow(ax, (18.0, 6.3), (18.0, 6.0))
    _arrow(ax, (18.0, 3.3), (18.0, 3.0))

    ax.text(10, 0.4, "Arrows: flow of data and models.  Dashed boxes: supporting rules and "
            "checks.  Calibration data and validation data are never mixed.",
            ha="center", size=5.8, style="italic")
    fig.tight_layout(pad=0.2)
    return fig


# ------------------------------------------------------------------- concept picture
def _iso(x, y, z, k=0.45, ang=np.deg2rad(32)):
    """Oblique projection: x right, y up, z into the page."""
    return x + k * z * np.cos(ang), y + k * z * np.sin(ang)


def _block(ax, x0, x1, y0, y1, z0, z1, fills, hatch=None, lw=0.6, zo=1):
    """Draw a cuboid: front, top and right faces with grayscale fills (front, top, side)."""
    P = lambda x, y, z: _iso(x, y, z)  # noqa: E731
    front = [P(x0, y0, z0), P(x1, y0, z0), P(x1, y1, z0), P(x0, y1, z0)]
    top = [P(x0, y1, z0), P(x1, y1, z0), P(x1, y1, z1), P(x0, y1, z1)]
    side = [P(x1, y0, z0), P(x1, y0, z1), P(x1, y1, z1), P(x1, y1, z0)]
    for pts, fc in zip((side, top, front), (fills[2], fills[1], fills[0])):
        ax.add_patch(Polygon(pts, closed=True, fc=fc, ec=K, lw=lw, hatch=hatch, zorder=zo))


def _concept_device(ax):
    ax.set_aspect("equal")
    ax.axis("off")
    D = 40                                     # depth into page, nm (illustrative)
    tsi, tox, tg = 10, 2.2, 5
    # bottom gate stack (under the channel)
    _block(ax, 0, 50, -tox - tg, -tox, 0, D, ("#5a5a5a", "#7a7a7a", "#3c3c3c"), zo=1)
    _block(ax, 0, 50, -tox, 0, 0, D, ("#d9d9d9", "#e6e6e6", "#bfbfbf"), zo=2)
    # silicon body
    _block(ax, -22, 0, 0, tsi, 0, D, ("#bdbdbd", "#cfcfcf", "#a0a0a0"), hatch="....", zo=3)
    _block(ax, 0, 50, 0, tsi, 0, D, ("#f7f7f7", "#ffffff", "#e3e3e3"), zo=3)
    _block(ax, 50, 82, 0, tsi, 0, D, ("#bdbdbd", "#cfcfcf", "#a0a0a0"), hatch="////", zo=3)
    # top gate stack
    _block(ax, 0, 50, tsi, tsi + tox, 0, D, ("#d9d9d9", "#e6e6e6", "#bfbfbf"), zo=4)
    _block(ax, 0, 50, tsi + tox, tsi + tox + tg, 0, D, ("#5a5a5a", "#7a7a7a", "#3c3c3c"), zo=5)

    # interface traps on the front face at the drain-side gate edge, both interfaces
    for yy in (tsi, 0):
        xs = np.linspace(36, 49, 6)
        px, py = _iso(xs, np.full_like(xs, yy), 0)
        ax.plot(px, py, ls="none", marker="x", ms=3.6, mew=0.9, color=K, zorder=8)
    # tunneling arrows on the front face
    ax.add_patch(FancyArrowPatch(_iso(-3, 5, 0), _iso(5, 5, 0), arrowstyle="-|>",
                                 mutation_scale=7, lw=1.0, color=K, zorder=9))
    ax.add_patch(FancyArrowPatch(_iso(46, 7.5, 0), _iso(55, 7.5, 0), arrowstyle="-|>",
                                 mutation_scale=7, lw=1.0, color=K, ls="--", zorder=9))

    def lab(xy, txt, xyt, ha="center"):
        ax.annotate(txt, xy=xy, xytext=xyt, ha=ha, va="center", size=6.2,
                    arrowprops=dict(arrowstyle="-", lw=0.5, color=K), zorder=10)

    lab(_iso(-11, 5, 0), "p$^+$ source", (-20, -16))
    lab(_iso(25, 5, 0), "p$^-$ Si channel", (25, -16))
    lab(_iso(66, 5, 0), "n$^+$ drain", (72, -16))
    lab(_iso(25, tsi + tox + tg, D * 0.5), "metal gate (top and bottom)", (14, 36))
    lab(_iso(5, tsi + 1.1, 0), "HfO$_2$ gate oxide", (-26, 24), ha="center")
    lab(_iso(44, tsi, 0), "interface traps $D_{it}$\n(drain-side gate edge)", (66, 30))
    lab(_iso(1, 5, 0), "on-state BTBT\n(source side)", (-34, 2), ha="center")
    lab(_iso(52, 7.5, 0), "ambipolar BTBT + TAT\n(drain side)", (101, -6), ha="center")
    ax.set_xlim(-50, 120)
    ax.set_ylim(-20, 40)


def _concept_bands(ax):
    x = np.linspace(-20, 80, 600)
    # schematic ambipolar bias (V_GS < 0): channel bands pushed up, abrupt drain junction
    Ec = 0.55 + 0.45 / (1 + np.exp(-(x - 0) / 2.5)) - 1.95 / (1 + np.exp(-(x - 50) / 4.0))
    Ec = Ec + 0.0
    Ev = Ec - 1.12
    ax.plot(x, Ec, color=K, lw=1.1)
    ax.plot(x, Ev, color=K, lw=1.1)
    ax.text(-18, Ec[0] + 0.08, "$E_C$", size=6.5)
    ax.text(-18, Ev[0] - 0.2, "$E_V$", size=6.5)
    # electrons tunnel from the channel valence band into the drain conduction band at one
    # energy, which must lie between Ec(drain) and Ev(channel)
    def cross(band, E):
        return x[300 + np.argmin(np.abs(band[300:] - E))]

    E0 = -0.38
    xa, xb = cross(Ev, E0), cross(Ec, E0)
    ax.annotate("", xy=(xb, E0), xytext=(xa - 3, E0),
                arrowprops=dict(arrowstyle="-|>", lw=1.0, color=K))
    ax.text(xa - 5, E0, "direct BTBT", ha="right", va="center", size=6.0)
    # TAT: the same transition split into two shorter hops through a gap state
    Et = -0.72
    xv, xc = cross(Ev, Et), cross(Ec, Et)
    xt = 0.5 * (xv + xc)
    ax.plot([xt], [Et], marker="x", ms=5, mew=1.1, color=K)
    ax.annotate("", xy=(xt - 0.8, Et), xytext=(xv - 3, Et),
                arrowprops=dict(arrowstyle="-|>", lw=0.9, color=K, ls="--"))
    ax.annotate("", xy=(xc, Et), xytext=(xt + 0.8, Et),
                arrowprops=dict(arrowstyle="-|>", lw=0.9, color=K, ls="--"))
    ax.text(xv - 5, Et - 0.04, "TAT via trap $E_t$\n(two shorter hops)", ha="right",
            va="top", size=6.0)
    ax.axvspan(0, 50, color="#ececec", lw=0, zorder=0)
    ax.text(25, -2.35, "gated channel", ha="center", size=6)
    ax.text(-10, -2.35, "source", ha="center", size=6)
    ax.text(65, -2.35, "drain", ha="center", size=6)
    ax.set_xlim(-20, 80)
    ax.set_ylim(-2.45, 1.35)
    ax.set_xlabel("position along channel")
    ax.set_ylabel("energy")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _concept_iv(ax):
    v = np.linspace(-1.2, 1.2, 300)
    on = 10 ** (-14 + 9 / (1 + np.exp(-(v - 0.45) / 0.12)))
    amb_ref = 10 ** (-14 + 6.5 / (1 + np.exp((v + 0.35) / 0.13)))
    amb_sup = 10 ** (-14 + 3.5 / (1 + np.exp((v + 0.6) / 0.13)))
    floor = 10 ** (-13.5 + 3.0 / (1 + np.exp((v + 0.1) / 0.25)))
    I_ref = on + amb_ref + 1e-15
    I_sup = on + amb_sup + 1e-15
    I_trap = on + amb_sup + floor
    ax.semilogy(v, I_ref, color=K, lw=1.1, label="reference")
    ax.semilogy(v, I_sup, color=K, lw=1.1, ls="--", label="technique, BTBT only")
    ax.semilogy(v, I_trap, color=K, lw=1.6, ls=":", label="technique + interface TAT")
    vq = -0.75
    a, b, c = (np.interp(vq, v, I) for I in (I_ref, I_sup, I_trap))
    ax.annotate("", xy=(vq, b), xytext=(vq, a),
                arrowprops=dict(arrowstyle="<->", lw=0.7, color=K))
    ax.text(vq + 0.05, np.sqrt(a * b) * 2, "claimed\nsuppression", size=5.8, va="center")
    vq2 = -1.05
    a2, c2 = np.interp(vq2, v, I_ref), np.interp(vq2, v, I_trap)
    ax.annotate("", xy=(vq2, c2), xytext=(vq2, a2),
                arrowprops=dict(arrowstyle="<->", lw=0.7, color=K))
    ax.text(vq2 - 0.05, np.sqrt(a2 * c2), "what\nsurvives", size=5.8, va="center",
            ha="right")
    ax.text(0.95, 0.5, "on\nstate", transform=ax.transAxes, ha="right", size=6)
    ax.text(0.03, 0.97, "ambipolar\nbranch", transform=ax.transAxes, va="top", size=6)
    ax.set_xlim(-1.45, 1.2)
    ax.set_ylim(1e-15, 1e-4)
    ax.set_xlabel("gate voltage $V_{GS}$")
    ax.set_ylabel("drain current (log)")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.minorticks_off()
    ax.legend(loc="upper center", frameon=False, fontsize=5.6, handlelength=2.6,
              bbox_to_anchor=(0.55, 1.03))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def fig_concept():
    fig = plt.figure(figsize=(7.16, 4.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1], hspace=0.22, wspace=0.28)
    a = fig.add_subplot(gs[0, :])
    b = fig.add_subplot(gs[1, 0])
    c = fig.add_subplot(gs[1, 1])
    _concept_device(a)
    _concept_bands(b)
    _concept_iv(c)
    a.set_title("(a) Double-gate Si tunnel FET: where the currents are generated",
                size=7.2, loc="left")
    b.set_title("(b) Ambipolar bias, drain junction (schematic)", size=7.2, loc="left")
    c.set_title("(c) Why suppression is capped (schematic)", size=7.2, loc="left")
    fig.subplots_adjust(left=0.05, right=0.99, top=0.95, bottom=0.08)
    return fig


def main():
    m = macros()
    for name, fig in (("fig_overview_block", fig_block(m)),
                      ("fig_overview_concept", fig_concept())):
        for ext in ("pdf", "png", "svg"):
            fig.savefig(OUT / f"{name}.{ext}", dpi=400 if ext == "png" else None)
        plt.close(fig)
        print("wrote", OUT / f"{name}.pdf")


if __name__ == "__main__":
    main()
