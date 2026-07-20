#!/usr/bin/env python3
"""Overlay required-cut NULL contours (14000 deg², truth-centered): BNT-580 vs non-BNT-460,
for BOTH the VMIM deep-ensemble and the linear whitening. Filled = whitening (the clean reference),
line contours = VMIM ensemble; grey = non-BNT, blue = BNT. Run with the aname interpreter (getdist).
"""
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from getdist import MCSamples, plots  # noqa: E402

REPO = "/home/tersenov/software/bar_impact"
WT = f"{REPO}/outputs/baryon_tension/vmim/ensemble"
WH = f"{REPO}/outputs/baryon_tension/bnt_whiten_test"
NAMES = ["Om", "S8", "w0", "H0", "ns", "Ob"]
LBL = [r"\Omega_m", "S_8", "w_0", "H_0", "n_s", r"\Omega_b"]
TRUTH = [0.26, 0.84, -1.0, 67.36, 0.9649, 0.0493]
SUB = ["Om", "S8", "w0"]


def stack(patt):
    fs = sorted(glob.glob(patt))
    return np.concatenate([np.load(f) for f in fs]) if fs else None


def mc(s, label):
    if s is None:
        print(f"  MISSING {label}"); return None
    print(f"  {label:32s} σ(S8)={s[:,1].std():.4f} σ(Ωm)={s[:,0].std():.4f} "
          f"S8̄={s[:,1].mean():.4f}")
    return MCSamples(samples=s, names=NAMES, labels=LBL, label=label)


sets = [
    (stack(f"{WH}/payoff_nb460/posterior_samples_ps_auto_cross_nobaryons_vs_nobaryons_bins1234_"
           "l37-460_r10_*run*.npy"), "non-BNT ℓ≤460 (whiten)", "#888888", True),
    (stack(f"{WH}/payoff_bnt580/posterior_samples_bnt_ps_auto_cross_nobaryons_vs_nobaryons_bins1234_"
           "l37-580_1024_1024_1024_r10_*run*.npy"), "BNT bin1-580 (whiten)", "#1f77b4", True),
    (stack(f"{WT}/pooled_nonbnt_460.npy"), "non-BNT ℓ≤460 (VMIM ens.)", "#888888", False),
    (stack(f"{WT}/pooled_bnt_580.npy"), "BNT bin1-580 (VMIM ens.)", "#1f77b4", False),
]
mcs, colors, filled = [], [], []
print("Contour widths (14000, NULL, truth S8=0.84):")
for s, lab, col, fil in sets:
    m = mc(s, lab)
    if m is not None:
        mcs.append(m); colors.append(col); filled.append(fil)

g = plots.get_subplot_plotter(width_inch=9)
g.settings.figure_legend_frame = False
g.settings.alpha_filled_add = 0.5
g.settings.legend_fontsize = 12
g.triangle_plot(mcs, params=SUB, filled=filled, contour_colors=colors,
                legend_labels=[m.label for m in mcs],
                markers={k: TRUTH[NAMES.index(k)] for k in SUB})
g.fig.suptitle("14000 deg² — required-cut null contours: VMIM ensemble (lines) vs whitening (filled)",
               y=1.02, fontsize=12)
out = f"{REPO}/plots/contours_bnt_vmim_vs_whiten_14000"
g.export(out + ".pdf")
plt.savefig(out + ".png", dpi=150, bbox_inches="tight")
print(f"-> {out}.png")
