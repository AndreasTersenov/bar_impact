#!/usr/bin/env python3
"""Contour overlays for the whitening result (14000 deg², NULL posteriors, stacked runs).

Plot 1 (payoff): non-BNT cut-all ℓ≤460 vs BNT cut-bin1 ℓ≤580 (bins 2-4 full), BOTH whitened —
the redo of the earlier z-score comparison that came out ~equal; now BNT should be visibly tighter.
Plot 2 (the fix): BNT-580 z-score vs BNT-580 whiten — what whitening did (recenter + tighten).

Run with the aname interpreter (getdist).
"""
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from getdist import MCSamples, plots  # noqa: E402

REPO = "/home/tersenov/software/bar_impact"
WT = f"{REPO}/outputs/baryon_tension/bnt_whiten_test"
ZB = (f"{REPO}/outputs/baryon_tension/bnt_ps_bin1_submean_l37/posteriors/mask_14000/null/"
      "posterior_samples_bnt_ps_auto_cross_nobaryons_vs_nobaryons_bins1234_l37-580_1024_1024_1024_"
      "r10_masked_14000sqdeg_apod2.0_master_submean_noisy_s0.26_run*.npy")
NAMES = ["Om", "S8", "w0", "H0", "ns", "Ob"]
LBL = [r"\Omega_m", "S_8", "w_0", "H_0", "n_s", r"\Omega_b"]
TRUTH = [0.26, 0.84, -1.0, 67.36, 0.9649, 0.0493]
SUB = ["Om", "S8", "w0"]


def stack(patt):
    fs = sorted(glob.glob(patt))
    return (np.concatenate([np.load(f) for f in fs]), len(fs)) if fs else (None, 0)


def mc(patt, label):
    s, n = stack(patt)
    if s is None:
        print(f"  MISSING: {label}")
        return None
    print(f"  {label:34s} runs={n} σ(S8)={s[:,1].std():.4f} σ(Ωm)={s[:,0].std():.4f}")
    return MCSamples(samples=s, names=NAMES, labels=LBL, label=label)


def draw(mcs, colors, out, title):
    mcs = [m for m in mcs if m is not None]
    g = plots.get_subplot_plotter(width_inch=8.5)
    g.settings.figure_legend_frame = False
    g.settings.alpha_filled_add = 0.5
    g.settings.legend_fontsize = 12
    g.triangle_plot(mcs, params=SUB, filled=True, contour_colors=colors,
                    legend_labels=[m.label for m in mcs],
                    markers={k: TRUTH[NAMES.index(k)] for k in SUB})
    g.fig.suptitle(title, y=1.02, fontsize=12)
    g.export(out + ".pdf")
    plt.savefig(out + ".png", dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"  -> {out}.png")


print("Plot 1 — payoff (whitened), 14000 NULL:")
m_nb = mc(f"{WT}/payoff_nb460/posterior_samples_ps_auto_cross_nobaryons_vs_nobaryons_bins1234_"
          "l37-460_r10_*run*.npy", "non-BNT  cut-all ℓ≤460 (whiten)")
m_b = mc(f"{WT}/payoff_bnt580/posterior_samples_bnt_ps_auto_cross_nobaryons_vs_nobaryons_bins1234_"
         "l37-580_1024_1024_1024_r10_*run*.npy", "BNT  cut-bin1 ℓ≤580 (whiten)")
draw([m_nb, m_b], ["#888888", "#1f77b4"],
     f"{REPO}/plots/contours_bnt_vs_nonbnt_14000_whiten",
     "14000 deg² — required-cut contours, whitened (BNT keeps bins 2-4 full)")

print("Plot 2 — the whitening fix on BNT-580, 14000 NULL:")
m_bz = mc(ZB, "BNT-580  z-score (old)")
m_bw = mc(f"{WT}/payoff_bnt580/posterior_samples_bnt_ps_auto_cross_nobaryons_vs_nobaryons_bins1234_"
          "l37-580_1024_1024_1024_r10_*run*.npy", "BNT-580  whiten (new)")
draw([m_bz, m_bw], ["#d62728", "#1f77b4"],
     f"{REPO}/plots/contours_bnt580_zscore_vs_whiten_14000",
     "14000 deg² — BNT bin1-580 null: z-score vs whitening")
