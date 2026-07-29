#!/usr/bin/env python3
"""Regenerate the 6-panel masked and 7-panel (with full sky) tension figures.

Uses the current plots.py marker style (unconnected points). Run with the aname interpreter.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tension import plots  # noqa: E402

REPO = "/lustre/fsn1/projects/rech/prk/ulx34io/bar_impact"
MASKED = f"{REPO}/outputs/baryon_tension/ps_submean_l37/tables/tension_3param_agg.csv"
FULLSKY = f"{REPO}/outputs/baryon_tension/ps_fullsky_l37/tables/tension_3param_agg.csv"

# 6-panel masked
m = pd.read_csv(MASKED)
n6 = plots.plot_nsigma_vs_cut(
    m, [2000, 5000, 10000, 14000, 28000, 35000],
    f"{REPO}/plots/nsigma_vs_upper_cut_masks.pdf",
    f"{REPO}/plots/nsigma_vs_upper_cut_masks.png",
    subtitle="6 footprints | monopole-subtracted PS, l>=37 | step-40 | 3-param Q_DM, mean+/-std/5 runs",
    dedup=True)

# 7-panel with full sky
f = pd.read_csv(FULLSKY)
both = pd.concat([m.astype({"area": object}), f], ignore_index=True)
n7 = plots.plot_nsigma_vs_cut(
    both, [2000, 5000, 10000, 14000, 28000, 35000, "fullsky"],
    f"{REPO}/plots/nsigma_vs_upper_cut_with_fullsky.pdf",
    f"{REPO}/plots/nsigma_vs_upper_cut_with_fullsky.png",
    subtitle="6 footprints + full sky | PS l>=37 | step-40 | 3-param Q_DM, mean+/-std/5 runs "
             "(full sky=healpy 10-l, masked=nlb4 40-l)",
    dedup=True)
print(f"DONE: masked={n6} panels, with_fullsky={n7} panels (unconnected markers)")
