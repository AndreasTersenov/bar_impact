#!/usr/bin/env python3
"""Overlay the BNT vs non-BNT posterior contours at each case's REQUIRED (just-unbiased) cut.

The point (14000 deg²): non-BNT must cut ALL bins to ℓ≤460 to drop below 0.3σ, whereas BNT only
needs to cut BIN-1 to ℓ≤580 (bins 2-4 kept at full ℓ). If BNT's contours are tighter at equal
unbiasedness, that's the payoff of only having to sacrifice bin-1's small scales.

Stacks the 5 NPE runs per case (marginalizes estimator variance). Plots the 3-param subset
(Ωm, S8, w0). Run with the aname interpreter (getdist):
  /home/tersenov/anaconda3/envs/aname/bin/python scripts/plot_bnt_vs_nonbnt_contours.py
"""
import glob
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from getdist import MCSamples, plots  # noqa: E402

REPO = "/lustre/fsn1/projects/rech/prk/ulx34io/bar_impact"
NAMES = ["Om", "S8", "w0", "H0", "ns", "Ob"]
LABELS = [r"\Omega_m", "S_8", "w_0", "H_0", "n_s", r"\Omega_b"]
TRUTH = [0.26, 0.84, -1.0, 67.36, 0.9649, 0.0493]
SUB = [0, 1, 2]  # Ωm, S8, w0

# (label, glob of the 5-run posteriors at the REQUIRED cut, color)
CASES = [
    ("non-BNT, cut all bins ℓ≤460",
     f"{REPO}/outputs/baryon_tension/ps_submean_l37/posteriors/mask_14000/null/"
     "posterior_samples_ps_auto_cross_nobaryons_vs_nobaryons_bins1234_l37-460_r10_"
     "masked_14000sqdeg_apod2.0_master_submean_noisy_s0.26_run*.npy", "#888888"),
    ("BNT, cut bin-1 ℓ≤580 (bins 2-4 full)",
     f"{REPO}/outputs/baryon_tension/bnt_ps_bin1_submean_l37/posteriors/mask_14000/null/"
     "posterior_samples_bnt_ps_auto_cross_nobaryons_vs_nobaryons_bins1234_l37-580_1024_1024_1024_"
     "r10_masked_14000sqdeg_apod2.0_master_submean_noisy_s0.26_run*.npy", "#1f77b4"),
]


def load_stacked(patt):
    fs = sorted(glob.glob(patt))
    if not fs:
        return None, 0
    return np.concatenate([np.load(f) for f in fs], axis=0), len(fs)


def main():
    mcs, colors = [], []
    print("Stacked NULL posteriors at the required (just-unbiased) cut, 14000 deg²:")
    print(f"  {'case':40s} {'runs':>4} {'σ(Ωm)':>7} {'σ(S8)':>7} {'σ(w0)':>7}  area(Ωm-S8)")
    for label, patt, color in CASES:
        s, n = load_stacked(patt)
        if s is None:
            print(f"  {label:40s}  MISSING: {os.path.basename(patt)}")
            continue
        mc = MCSamples(samples=s, names=NAMES, labels=LABELS, label=label)
        mcs.append(mc); colors.append(color)
        sig = s.std(axis=0)
        area = sig[0] * sig[1]  # crude 1σ-box proxy for the Ωm-S8 constraint
        print(f"  {label:40s} {n:>4} {sig[0]:>7.4f} {sig[1]:>7.4f} {sig[2]:>7.4f}  {area:.2e}")

    if len(mcs) < 2:
        print("Need both cases; aborting."); return

    g = plots.get_subplot_plotter(width_inch=8.5)
    g.settings.figure_legend_frame = False
    g.settings.alpha_filled_add = 0.55
    g.settings.legend_fontsize = 13
    g.triangle_plot(mcs, params=[NAMES[i] for i in SUB], filled=True,
                    contour_colors=colors, legend_labels=[m.label for m in mcs],
                    markers={NAMES[i]: TRUTH[i] for i in SUB})
    out = f"{REPO}/plots/contours_bnt_vs_nonbnt_14000_requiredcut"
    g.export(out + ".pdf")
    plt.savefig(out + ".png", dpi=150, bbox_inches="tight")
    print(f"\nSaved -> {out}.png / .pdf")


if __name__ == "__main__":
    main()
