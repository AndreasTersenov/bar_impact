#!/usr/bin/env python3
"""Proof that the de-biasing contours are score-COMPRESSED, not raw: overlay score (filled) vs raw
(dashed) null posteriors at the de-biasing cuts for BNT@460 and non-BNT@580.

Shows (a) compression tightens BOTH configs and puts the nulls back on truth (raw nulls are off at
S8≈0.87), and (b) the reliable BNT/non-BNT FoM advantage is the COMPRESSED 1.28×, not the spurious
uncompressed 1.59× (raw under-extracts non-BNT more). Run under aname (getdist)."""
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
from getdist import MCSamples, plots  # noqa: E402

REPO = "/mnt/home/tersenov/software/bar_impact"
NAMES = ["Omega_m", "S8", "w0"]; LABELS = [r"\Omega_m", "S_8", "w_0"]; TRUTH = [0.26, 0.84, -1.0]
B = f"{REPO}/outputs/baryon_tension"
SRC = {
    "score BNT @460":  f"{B}/bnt_ps_bin1_score_l37/area14000/posteriors/cut460",
    "score nonBNT @580": f"{B}/ps_cutall_score_l37/area14000/posteriors/cut580",
    "raw BNT @460":    f"{B}/_debias_raw_vs_score/bnt/posteriors/cut460",
    "raw nonBNT @580": f"{B}/_debias_raw_vs_score/non/posteriors/cut580",
}


def pool(d):
    return np.concatenate([np.load(f)[:, :3] for f in sorted(glob.glob(f"{d}/null_run*.npy"))])


def main():
    mc = {k: MCSamples(samples=pool(d), names=NAMES, labels=LABELS, label=k) for k, d in SRC.items()}
    g = plots.get_subplot_plotter(width_inch=8)
    g.settings.legend_fontsize = 12; g.settings.axes_labelsize = 15
    order = ["score nonBNT @580", "score BNT @460", "raw nonBNT @580", "raw BNT @460"]
    g.triangle_plot([mc[k] for k in order], filled=[True, True, False, False],
                    contour_colors=["0.5", "C0", "0.5", "C0"],
                    contour_ls=["-", "-", "--", "--"], contour_lws=[1.5, 1.5, 1.6, 1.6],
                    legend_labels=order, legend_loc="upper right")
    for i in range(3):
        for j in range(i + 1):
            ax = g.subplots[i, j]
            if ax is None:
                continue
            ax.axvline(TRUTH[j], color="k", ls=":", lw=1, alpha=0.6)
            if i != j:
                ax.axhline(TRUTH[i], color="k", ls=":", lw=1, alpha=0.6)
    g.fig.suptitle("Score (filled) vs raw/uncompressed (dashed) null contours at the de-biasing cut — 14000 deg²",
                   fontsize=13, y=1.02)
    g.fig.text(0.63, 0.74, "compression tightens both\n& restores on-truth nulls\n"
               "reliable BNT gain = 1.28× FoM\n(uncompressed 1.59× is spurious)",
               ha="center", va="center", fontsize=11,
               bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
    out = f"{REPO}/plots/score_vs_raw_debiased_14000"
    for ext in ("png", "pdf"):
        g.export(f"{out}.{ext}")
    print(f"wrote {out}.png / .pdf")


if __name__ == "__main__":
    main()
