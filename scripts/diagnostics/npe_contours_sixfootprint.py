#!/usr/bin/env python3
"""getdist triangle contours (Om, S8, w0) of the score-compression NPE posteriors, BNT-580 vs
non-BNT-460, for all six footprints. Opaque fills (alpha=1). non-BNT drawn first (larger, grey) so the
tighter BNT (red) sits on top and both stay visible. Saves one PNG per area + a 2x3 montage.
Run with jaxili python (getdist)."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from getdist import plots
from getdist.mcsamples import MCSamples

P3 = [0, 1, 2]
NAMES, LAB = ["Om", "S8", "w0"], [r"\Omega_m", "S_8", "w_0"]
AREAS = [2000, 5000, 10000, 14000, 28000, 35000]
NPE = "outputs/score_experiment/npe_score"
OUT = "outputs/score_experiment/contours"
os.makedirs(OUT, exist_ok=True)
GREY, RED = "#7f7f7f", "#c0392b"


def mc(tag, A, label):
    s = np.load(f"{NPE}/posterior_summary_{tag}_{A}_mle.npy")[:, P3]
    return MCSamples(samples=s, names=NAMES, labels=LAB, label=label)


paths = []
for A in AREAS:
    non = mc("nonbnt_460", A, "non-BNT-460")
    bnt = mc("bnt_580", A, "BNT-580")
    g = plots.get_subplot_plotter(width_inch=5.2)
    g.settings.alpha_filled_add = 1.0          # opaque fills
    g.settings.solid_contour_palefactor = 0.55  # 95% shade vs 68%
    g.settings.legend_fontsize = 11
    g.triangle_plot([non, bnt], NAMES, filled=True, contour_colors=[GREY, RED],
                    legend_labels=["non-BNT-460", "BNT-580"], legend_loc="upper right")
    g.fig.suptitle(f"Area = {A} deg$^2$  (score-NPE, calibrated)", fontsize=12, y=1.02)
    p = f"{OUT}/npe_contours_{A}.png"
    g.export(p)
    paths.append(p)
    print("saved", p)

# 2x3 montage for a single overview
fig, axes = plt.subplots(2, 3, figsize=(16.5, 11))
for ax, p, A in zip(axes.ravel(), paths, AREAS):
    ax.imshow(plt.imread(p)); ax.axis("off")
fig.suptitle("Score-compression NPE posteriors (Om, S8, w0): BNT-580 (red) vs non-BNT-460 (grey), six footprints",
             fontsize=14)
fig.tight_layout()
m = "outputs/score_experiment/npe_contours_sixfootprint_montage.png"
fig.savefig(m, dpi=110, bbox_inches="tight")
print("saved", m)
