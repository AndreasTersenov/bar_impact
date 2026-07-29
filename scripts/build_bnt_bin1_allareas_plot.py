#!/usr/bin/env python3
"""Multi-panel BNT bin-1 vs non-BNT-cut-all baryon-tension figure (masked footprints).

One panel per footprint, overlaying:
  - non-BNT — cut all bins   (grey squares; from ps_submean_l37 — the standard analysis)
  - BNT — cut bin-1 only      (blue circles; from bnt_ps_bin1_submean_l37 — bins 2-4 full)

Both curves are the 3-param (Ωm,S₈,w₀) Gaussian Q_DM tension, mean±std over the 5 NPE
training seeds, on the same monopole-subtracted ℓ≥37, step-40 cut grid. This is the
single-panel monitor_bnt_bin1.py figure generalized to every footprint with data.

Reads the aggregated tables written by compute_tension.py (so it does NOT need tensiometer),
but still runs fine under the aname env (matplotlib):
  /home/tersenov/anaconda3/envs/aname/bin/python scripts/build_bnt_bin1_allareas_plot.py
"""
import argparse
import os
import sys
from math import ceil

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = "/lustre/fsn1/projects/rech/prk/ulx34io/bar_impact"
BNT_AGG = f"{REPO}/outputs/baryon_tension/bnt_ps_bin1_submean_l37/tables/tension_3param_agg.csv"
NONBNT_AGG = f"{REPO}/outputs/baryon_tension/ps_submean_l37/tables/tension_3param_agg.csv"
OUT_PNG = f"{REPO}/plots/nsigma_vs_upper_cut_bnt_bin1_allareas.png"
OUT_PDF = f"{REPO}/plots/nsigma_vs_upper_cut_bnt_bin1_allareas.pdf"
ALL_AREAS = (2000, 5000, 10000, 14000, 28000, 35000)
THRESHOLD = 0.3
# Per-area "% extracted at no-cut" = BNT/non-BNT tension at ℓmax=1020, same binning (lossless
# target is 100%). Measured at rebin=40 (the optimized presentation). Annotated per panel.
EXTRACTED_R40 = {2000: 87, 5000: 85, 10000: 82, 14000: 76, 28000: 93, 35000: 93}


def _apply_rcparams():
    plt.rcParams["legend.fontsize"] = 12
    plt.rcParams["axes.labelsize"] = 15
    plt.rcParams["xtick.labelsize"] = 13
    plt.rcParams["ytick.labelsize"] = 13
    plt.rcParams["axes.titlesize"] = 14


def _crossing(sub):
    """Lowest upper_cut whose mean nσ >= THRESHOLD, or None."""
    hit = sub[sub["mean"] >= THRESHOLD]["upper_cut"]
    return int(hit.min()) if len(hit) else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--areas", type=int, nargs="*", default=list(ALL_AREAS))
    p.add_argument("--bnt-agg", default=BNT_AGG, help="BNT tension table (e.g. the tables_r40 one).")
    p.add_argument("--rebin-note", default=None, help="Annotate panels with rebin* + %% extracted.")
    p.add_argument("--out-png", default=OUT_PNG)
    p.add_argument("--out-pdf", default=OUT_PDF)
    args = p.parse_args()

    bnt = pd.read_csv(args.bnt_agg)
    nonbnt = pd.read_csv(NONBNT_AGG) if os.path.exists(NONBNT_AGG) else pd.DataFrame()

    areas = [a for a in args.areas if (bnt["area"] == a).any()]
    if not areas:
        print(f"No BNT bin-1 data in {BNT_AGG} for areas {args.areas}. Nothing to plot.")
        return
    n = len(areas)

    _apply_rcparams()
    ncols = min(n, 3)
    nrows = ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.3 * nrows), sharex=True)
    axes = list(axes.flatten()) if n > 1 else [axes]

    for ax, area in zip(axes, areas):
        if len(nonbnt):
            nb = nonbnt[nonbnt["area"] == area].sort_values("upper_cut")
            if len(nb):
                ax.errorbar(nb["upper_cut"], nb["mean"], yerr=nb["std"].fillna(0),
                            fmt="s", color="0.55", ms=5, elinewidth=1.2, capsize=3,
                            label="non-BNT — cut all bins")
        b = bnt[bnt["area"] == area].sort_values("upper_cut")
        ax.errorbar(b["upper_cut"], b["mean"], yerr=b["std"].fillna(0),
                    fmt="o", color="C0", ms=6, elinewidth=1.5, capsize=4,
                    label="BNT — cut bin-1 only (bins 2-4 full)")
        ax.axhline(THRESHOLD, color="r", linestyle="--", linewidth=1.4,
                   label=f"Threshold ({THRESHOLD})")
        ax.grid(True, alpha=0.3, linestyle=":")
        ax.set_title(rf"Area = {area} deg$^2$")
        ax.set_ylim(bottom=0)
        if args.rebin_note:
            ext = EXTRACTED_R40.get(area)
            note = f"{args.rebin_note}" + (f"\n{ext}% extracted @ no-cut" if ext else "")
            ax.text(0.04, 0.96, note, transform=ax.transAxes, ha="left", va="top",
                    fontsize=9, color="0.35",
                    bbox=dict(boxstyle="round", fc="white", ec="0.85", alpha=0.8))
    for ax in axes[n:]:               # hide any unused grid cells
        ax.set_visible(False)

    legend_ax = axes[1] if n > 1 else axes[0]   # panel 0 holds the rebin note; put legend in panel 1
    legend_ax.legend(loc="upper left", fontsize=11)
    fig.supxlabel(r"Upper Cut ($\ell_{\mathrm{max}}$)", fontsize=15, y=0.03)
    fig.supylabel(r"Significance ($n_\sigma$)", fontsize=15, x=0.05)
    title = ("OPTIMAL-BNT (rebin 40) vs standard non-BNT — baryon tension vs scale cut"
             if args.rebin_note else
             "BNT bin-1-only vs non-BNT cut-all — baryon tension vs scale cut")
    fig.suptitle(title, fontsize=16)
    sub = ("monopole-subtracted PS, ℓ≥37, step-40 | 3-param Q_DM, mean±std / 3 runs | "
           "BNT at its optimal binning (no-cut % = how close to the lossless BNT=non-BNT identity)"
           if args.rebin_note else
           "monopole-subtracted PS, ℓ≥37, step-40 | 3-param Q_DM, mean±std / 5 runs")
    fig.text(0.5, 0.945, sub, ha="center", va="top", fontsize=9.5, color="0.4")
    plt.tight_layout(rect=(0.05, 0.04, 1, 0.93))

    os.makedirs(os.path.dirname(args.out_png), exist_ok=True)
    fig.savefig(args.out_pdf, bbox_inches="tight", transparent=True)
    fig.savefig(args.out_png, bbox_inches="tight", dpi=150)
    plt.close(fig)

    print(f"rendered {n} panels -> {args.out_png}")
    for area in areas:
        cb = _crossing(bnt[bnt["area"] == area].sort_values("upper_cut"))
        cg = (_crossing(nonbnt[nonbnt["area"] == area].sort_values("upper_cut"))
              if len(nonbnt) else None)
        print(f"  {area:>6} deg²: 0.3σ crossing — BNT bin-1 {cb or '>1020'}  |  "
              f"non-BNT cut-all {cg or '>1020'}")


if __name__ == "__main__":
    main()
