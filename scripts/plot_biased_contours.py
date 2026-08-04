#!/usr/bin/env python3
"""Triangle overlay (Om, S8, w0) for the BIASED observation (baryonified fiducial fed to the
nobaryons-trained model) at 14000 / full vector. Truth markers at (0.26,0.84,-1.0) so the baryon
SHIFT is visible.
  - RAW non-BNT (jaxili, no compression) BIASED at rebin=10/20/40
  - COMPRESSED non-BNT (VMIM P2g) BIASED
  - COMPRESSED BNT (VMIM P2g) BIASED
"""
import argparse
import glob

import numpy as np

RAW = "outputs/baryon_tension"
P2G = "outputs/baryon_tension/vmim_v2/p2g"
SPECS = [
    (f"{RAW}/_diagnostic_raw_nonbnt_r10/posteriors/cut1024/biased_run*.npy",
     "raw non-BNT (jaxili) rebin=10", False, "#e8b000", "--"),
    (f"{RAW}/_diagnostic_raw_nonbnt_r20/posteriors/cut1024/biased_run*.npy",
     "raw non-BNT (jaxili) rebin=20", False, "#ff7f0e", "--"),
    (f"{RAW}/ps_submean_l37/posteriors/mask_14000/biased/posterior_samples_ps_auto_cross_nobaryons_vs_baryonified_bins1234_l37-1020_r40_masked_14000sqdeg_apod2.0_master_submean_noisy_s0.26_run*.npy",
     "raw non-BNT (jaxili) rebin=40", False, "#d62728", "--"),
    (f"{P2G}/pooled_nonbnt/null_biased_pooled_nonbnt_full.npy",
     "non-BNT + VMIM compression", True, "#1f77b4", "-"),
    (f"{P2G}/pooled_bnt/null_biased_pooled_bnt_full.npy",
     "BNT + VMIM compression", True, "#2ca02c", "-"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/baryon_tension/vmim_v2/p2g/contours_biased.png")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    from getdist import MCSamples, plots

    names, labels = ["Om", "S8", "w0"], [r"\Omega_m", r"\sigma_8", "w_0"]
    mcs, filled, colors, line_args, rows = [], [], [], [], []
    for pat, label, fill, color, ls in SPECS:
        files = sorted(glob.glob(pat))
        if not files:
            print("MISSING:", pat); continue
        s = np.concatenate([np.load(f) for f in files])[:, :3]
        mcs.append(MCSamples(samples=s, names=names, labels=labels, label=label))
        filled.append(fill); colors.append(color)
        line_args.append({"lw": 2.0, "ls": ls, "color": color})
        rows.append((label, len(files), s.mean(0), s.std(0)))

    g = plots.get_subplot_plotter(width_inch=8.5)
    g.settings.legend_fontsize = 10.5
    g.settings.axes_labelsize = 15
    g.triangle_plot(mcs, filled=filled, contour_colors=colors, line_args=line_args,
                    markers={"Om": 0.26, "S8": 0.84, "w0": -1.0})
    g.export(a.out)
    print(f"[plot] wrote {a.out}  (BIASED; markers = truth, shift off them = baryon bias)\n")
    print(f"{'config':34s} {'nrun':>4}  {'Om':>16} {'S8':>16} {'w0':>16}")
    for label, n, mu, sd in rows:
        print(f"{label:34s} {n:>4}  {mu[0]:.3f}±{sd[0]:.3f}   {mu[1]:.3f}±{sd[1]:.3f}   {mu[2]:.3f}±{sd[2]:.3f}")


if __name__ == "__main__":
    main()
