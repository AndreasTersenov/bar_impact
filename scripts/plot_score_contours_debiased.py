#!/usr/bin/env python3
"""Constraining-power control plot: null posteriors at the DE-BIASING scale cut, BNT vs non-BNT.

The tension-vs-cut curves say each analysis becomes unbiased (<0.3σ) at roughly the same ℓmax —
BNT bin-1 at ℓmax≈460 (bins 2-4 left FULL), non-BNT cut-all at ℓmax≈580. The question that matters:
once you have applied the cut needed for unbiasedness, HOW MUCH MORE INFORMATION do you keep by
cutting in the BNT basis (only bin-1) instead of cutting every bin? This overlays the two null
(nobaryons) posteriors at those de-biasing cuts and reports the 3-param FoM gain.

Both are score-compressed, calibrated, on-truth nulls (5 seeds). Run under aname (getdist):
  /home/tersenov/anaconda3/envs/aname/bin/python scripts/plot_score_contours_debiased.py
"""
import argparse
import glob
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
from getdist import MCSamples, plots  # noqa: E402

REPO = "/lustre/fsn1/projects/rech/prk/ulx34io/bar_impact"
NAMES = ["Omega_m", "S8", "w0"]
LABELS = [r"\Omega_m", "S_8", "w_0"]
TRUTH = [0.26, 0.84, -1.0]
P3 = [0, 1, 2]


def fisher_fom3(cuts, bnt):
    """3-param Fisher-floor FoM = 1/sqrt(det(inv(F)[:3,:3])) — the network-independent ceiling."""
    os.environ.setdefault("FISHER_AREA", "14000")
    os.environ.setdefault("FISHER_REBIN", "20")
    sys.path.insert(0, os.path.join(REPO, "scripts"))
    import score_cut_utils as S
    cov3 = np.linalg.inv(S.build_score(cuts, bnt)["F"])[np.ix_(P3, P3)]
    return 1.0 / np.sqrt(np.linalg.det(cov3))


def load_null(tag, cut):
    d = f"{REPO}/outputs/baryon_tension/{tag}/area14000/posteriors/cut{cut}"
    files = sorted(glob.glob(f"{d}/null_run*.npy"))
    per_seed = [np.load(f)[:, :3] for f in files]
    return per_seed, np.concatenate(per_seed)


def fom3(per_seed):
    """3-param FoM = 1/sqrt(det Cov), from the seed-averaged covariance (constraining power,
    not inflated by between-seed mean scatter). Returns (FoM, sigma3)."""
    cov = np.mean([np.cov(s, rowvar=False) for s in per_seed], axis=0)
    return 1.0 / np.sqrt(np.linalg.det(cov)), np.sqrt(np.diag(cov))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bnt-cut", type=int, default=460)
    ap.add_argument("--nonbnt-cut", type=int, default=580)
    ap.add_argument("--out", default=f"{REPO}/plots/score_contours_debiased_14000")
    args = ap.parse_args()

    bnt_seeds, bnt = load_null("bnt_ps_bin1_score_l37", args.bnt_cut)
    non_seeds, non = load_null("ps_cutall_score_l37", args.nonbnt_cut)

    fb, sb = fom3(bnt_seeds)
    fn, sn = fom3(non_seeds)
    # Fisher-floor (network-independent) 3-param FoM for the same configs
    ffb = fisher_fom3([args.bnt_cut, 1024, 1024, 1024], True)
    ffn = fisher_fom3([args.nonbnt_cut] * 4, False)
    print(f"BNT bin-1  @ ℓmax {args.bnt_cut} (bins 2-4 full): σ(Ωm,S8,w0)={np.round(sb,4)}  FoM3={fb:.3e}")
    print(f"nonBNT cut-all @ ℓmax {args.nonbnt_cut}        : σ(Ωm,S8,w0)={np.round(sn,4)}  FoM3={fn:.3e}")
    print(f"  3-param FoM ratio BNT/non — Fisher floor: {ffb/ffn:.2f}×   realized (score): {fb/fn:.2f}×")

    lb = rf"BNT bin-1 cut ($\ell_{{\max}}={args.bnt_cut}$, bins 2–4 full)"
    ln = rf"non-BNT cut-all ($\ell_{{\max}}={args.nonbnt_cut}$)"
    s_bnt = MCSamples(samples=bnt, names=NAMES, labels=LABELS, label=lb)
    s_non = MCSamples(samples=non, names=NAMES, labels=LABELS, label=ln)

    g = plots.get_subplot_plotter(width_inch=7.5)
    g.settings.legend_fontsize = 13
    g.settings.axes_labelsize = 15
    g.settings.axes_fontsize = 11
    g.triangle_plot([s_non, s_bnt], filled=True,
                    contour_colors=["0.5", "C0"],
                    legend_labels=[ln, lb], legend_loc="upper right")
    # truth markers
    for i in range(3):
        for j in range(i + 1):
            ax = g.subplots[i, j]
            if ax is None:
                continue
            ax.axvline(TRUTH[j], color="k", ls=":", lw=1, alpha=0.7)
            if i != j:
                ax.axhline(TRUTH[i], color="k", ls=":", lw=1, alpha=0.7)
    g.fig.suptitle("Null contours at the de-biasing scale cut — 14000 deg²", fontsize=15, y=1.02)
    # 3-param FoM callout in the empty upper-middle panel region
    g.fig.text(0.62, 0.78, "BNT 3-param FoM advantage:\n"
               rf"$\mathbf{{{ffb/ffn:.2f}\times}}$ (Fisher information floor)" + "\n"
               rf"${fb/fn:.2f}\times$ (realized, calibrated)",
               ha="center", va="center", fontsize=12,
               bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
    g.fig.text(0.5, -0.01, "score-compressed, calibrated null posteriors (nobaryons), 5 seeds | "
               "dotted = truth | both unbiased (<0.3σ) at these cuts", ha="center", fontsize=9,
               color="0.4")
    for ext in ("png", "pdf"):
        g.export(f"{args.out}.{ext}")
    print(f"wrote {args.out}.png / .pdf")


if __name__ == "__main__":
    main()
