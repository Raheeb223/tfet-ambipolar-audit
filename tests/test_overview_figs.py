"""Overview figures (src/analysis/overview_figs): numbers come from the paper's generated
macros, including values with nested braces, and both figures build without error."""

import matplotlib.pyplot as plt
import pytest

from src.analysis import overview_figs as ov

# paper/ is git-ignored: on a fresh clone the macros exist only after the analysis scripts run.
pytestmark = pytest.mark.skipif(not (ov.GEN / "numbers.tex").exists(),
                                reason="paper/generated not built yet (run src.analysis.*)")


def test_macros_keep_nested_braces():
    m = ov.macros()
    assert "nCalRmsNl" in m
    if "nDcrossLDD" in m:
        assert m["nDcrossLDD"].count("{") == m["nDcrossLDD"].count("}")


def test_figures_build():
    m = ov.macros()
    for fig in (ov.fig_block(m), ov.fig_concept()):
        fig.canvas.draw()
        plt.close(fig)
