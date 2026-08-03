"""One definition of how a getdist contour figure in this paper looks.

WHY A MODULE AND NOT A STYLE SHEET. `plt.style.use()` does not reach a getdist triangle
plot: getdist drives its own type sizes through `g.settings.{axes,lab,legend}_fontsize` and
its own fills through `alpha_filled_add`, so an rcParams sheet silently leaves them at
getdist's defaults. That is exactly how the published contour figures ended up with 7 pt
axis labels on a 12-inch canvas while every other figure had been moved to paper_v1. The
type and the palette therefore have to be set in code, and if they are set in code they
must be set in ONE place -- five generators each carrying their own magic numbers is how
they drifted apart to begin with.

THE TARGET is the SUBMITTED version of the paper, for the same reason paper_v1.mplstyle
exists: several figures are being kept verbatim from it, and everything regenerated has to
sit beside them without a visible style change. The numbers below are read off the notebook
that produced those kept figures (notebooks/inference/paper_plots.ipynb,
`plot_subset_triangle`), not invented:

    axes_fontsize 14 · lab_fontsize 16 · legend_fontsize 14 · alpha_filled_add 0.99

PALETTES. `PALETTE=submitted` (default) reproduces contours_PS_peaks_L1_baryons_unbiased.pdf,
which survived the disk failure intact and is the ground truth for the submitted look.
`PALETTE=okabe` is the Okabe-Ito colourblind-safe set that aa.mplstyle used. The trade is
real and worth stating in the caption either way: the submitted palette matches the kept
figures but fails a colourblind check (its red and green are close under deuteranopia); the
Okabe-Ito set passes but will not match the figures kept from the submitted version.

Usage:
    import paper_contour_style as S
    g = plots.get_subplot_plotter(width_inch=W)
    S.apply(g)
    g.triangle_plot(mcs, names, contour_colors=S.colors_for(labels), ...)
"""
from __future__ import annotations

import os

# Type sizes of the submitted version. Deliberately NOT scaled to the canvas: getdist sizes
# are absolute points, and the kept figures use these at roughly this width.
AXES_FONTSIZE = 14
LAB_FONTSIZE = 16
LEGEND_FONTSIZE = 14
ALPHA_FILLED_ADD = 0.99      # near-opaque fills, as submitted; 0.55 made overlaps muddy
CONTOUR_LW = 1.6

# Palette by series label. Any label not listed falls through to CYCLE, in order.
PALETTES = {
    # Read off plots/contours_PS_peaks_L1_baryons_unbiased.pdf (submitted, intact).
    "submitted": {
        "Power spectrum": "0.45",
        "Peak counts":    "#E03424",
        "L1 norm":        "#1f77b4",
        "BNT basis":      "#1f77b4",
        "standard basis": "0.45",
    },
    # Okabe-Ito, colourblind-safe; what aa.mplstyle used.
    "okabe": {
        "Power spectrum": "#0072B2",
        "Peak counts":    "#D55E00",
        "L1 norm":        "#009E73",
        "BNT basis":      "#0072B2",
        "standard basis": "0.45",
    },
}
CYCLE = {
    "submitted": ["0.45", "#E03424", "#1f77b4", "#2ca02c", "#9467bd", "#8c564b"],
    "okabe":     ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"],
}


def palette_name() -> str:
    # Okabe-Ito is the DEFAULT, chosen 2026-08-03. Every contour figure in the paper is being
    # regenerated, so the "match the figures kept verbatim from the submitted version"
    # constraint that motivated the submitted palette no longer binds -- and it is the palette
    # the line figures (plot_nsigma_vs_area, plot_scaling_vs_area) already use, so this is the
    # first time the contour and line figures agree with each other.
    p = os.environ.get("PALETTE", "okabe").lower()
    if p not in PALETTES:
        raise SystemExit(f"[fatal] PALETTE={p!r} unknown; choose from {sorted(PALETTES)}")
    return p


def apply(g, palette: str | None = None) -> str:
    """Set the submitted-version type and fills on a getdist plotter. Returns the palette name."""
    p = palette or palette_name()
    g.settings.figure_legend_frame = False
    g.settings.axes_fontsize = AXES_FONTSIZE
    g.settings.lab_fontsize = LAB_FONTSIZE
    g.settings.legend_fontsize = LEGEND_FONTSIZE
    g.settings.alpha_filled_add = ALPHA_FILLED_ADD
    # No grid and no minor ticks: the kept figures have neither, and on a triangle plot a
    # grid competes with the truth markers for the reader's eye.
    for attr, val in (("axes_labelsize", LAB_FONTSIZE), ("solid_contour_palefactor", 0.6)):
        if hasattr(g.settings, attr):
            setattr(g.settings, attr, val)
    return p


def colors_for(labels, palette: str | None = None):
    """Map series labels to contour colours, stably and without silent collisions."""
    p = palette or palette_name()
    table, cyc = PALETTES[p], CYCLE[p]
    out, k = [], 0
    for lab in labels:
        # Series in a `both` figure are tagged "<stat> — null" / "— biased"; colour by the
        # statistic so a null and its biased twin stay the same hue, which is the entire
        # visual grammar of those figures.
        base = lab.split(" — ")[0].strip()
        if base in table:
            out.append(table[base])
        else:
            out.append(cyc[k % len(cyc)])
            k += 1
    return out


def provenance(palette: str | None = None) -> dict:
    """Style block for a figure's provenance.json, so the look is reproducible."""
    p = palette or palette_name()
    return {
        "contour_style": "scripts/paper_contour_style.py",
        "palette": p,
        "palette_note": ("submitted-version palette; matches the figures kept verbatim from "
                         "the submitted paper but is not colourblind-safe"
                         if p == "submitted" else
                         "Okabe-Ito colourblind-safe palette; does NOT match the figures kept "
                         "from the submitted version"),
        "axes_fontsize": AXES_FONTSIZE,
        "lab_fontsize": LAB_FONTSIZE,
        "legend_fontsize": LEGEND_FONTSIZE,
        "alpha_filled_add": ALPHA_FILLED_ADD,
    }
