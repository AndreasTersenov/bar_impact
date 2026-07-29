#!/usr/bin/env python3
"""Full-sky BNT bin-1 vs non-BNT-cut-all baryon-tension panel (healpy pipeline).

Single panel: the full-sky BNT bin-1-only curve (blue) over the full-sky non-BNT cut-all
reference (grey), both the healpy (10-ℓ) pipeline. Kept SEPARATE from the masked 6-panel
figure because (a) full sky is a different estimator (healpy 10-ℓ vs masked NaMaster nlb=4
40-ℓ) — not magnitude-comparable, and (b) the full-sky BNT contours are heavily inference-
limited (2-5× wider than non-BNT — the NPE under-extracts the high-dim healpy BNT data
vector), so the flat BNT curve is a conservative, NOT a validated, null. The caveat is
annotated on the figure.

  /home/tersenov/anaconda3/envs/aname/bin/python scripts/build_bnt_bin1_fullsky_plot.py
"""
import os
import sys

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = "/lustre/fsn1/projects/rech/prk/ulx34io/bar_impact"
BNT = f"{REPO}/outputs/baryon_tension/bnt_ps_bin1_fullsky_l37/tables_r60/tension_3param_agg.csv"
GREY = f"{REPO}/outputs/baryon_tension/ps_fullsky_l37/tables/tension_3param_agg.csv"
OUT_PNG = f"{REPO}/plots/nsigma_vs_upper_cut_bnt_bin1_fullsky_optimal.png"
OUT_PDF = f"{REPO}/plots/nsigma_vs_upper_cut_bnt_bin1_fullsky_optimal.pdf"
THRESHOLD = 0.3


def main():
    bnt = pd.read_csv(BNT).sort_values("upper_cut")
    grey = pd.read_csv(GREY)
    grey = grey[grey["area"] == "fullsky"].sort_values("upper_cut")

    plt.rcParams.update({"legend.fontsize": 12, "axes.labelsize": 15,
                         "xtick.labelsize": 13, "ytick.labelsize": 13, "axes.titlesize": 15})
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    ax.errorbar(grey["upper_cut"], grey["mean"], yerr=grey["std"].fillna(0), fmt="s",
                color="0.55", ms=5, elinewidth=1.2, capsize=3, label="non-BNT — cut all bins")
    ax.errorbar(bnt["upper_cut"], bnt["mean"], yerr=bnt["std"].fillna(0), fmt="o",
                color="C0", ms=6, elinewidth=1.5, capsize=4,
                label="BNT — cut bin-1 only (bins 2-4 full)")
    ax.axhline(THRESHOLD, color="r", linestyle="--", linewidth=1.4, label=f"Threshold ({THRESHOLD})")
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.set_title(r"Full sky (optimal BNT, rebin 60) — baryon tension vs scale cut")
    ax.set_xlabel(r"Upper Cut ($\ell_{\mathrm{max}}$)")
    ax.set_ylabel(r"Significance ($n_\sigma$)")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left")
    ax.text(0.985, 0.04,
            "rebin=60 (optimized): full-sky BNT no-cut tension 0.05→1.10 (22× better) but still only\n"
            "29% of the non-BNT/lossless value (3.83) ⇒ binning HELPS but does NOT finish the job\n"
            "for the 10-ℓ healpy vector — full extraction needs score/MOPED compression",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8.0, color="0.35",
            bbox=dict(boxstyle="round", fc="white", ec="0.8", alpha=0.85))
    fig.text(0.5, 0.975, "3-param Q_DM, mean±std / 3 runs | ℓ≥37, step-40 | healpy (not magnitude-comparable to masked)",
             ha="center", va="top", fontsize=9.0, color="0.4")
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.savefig(OUT_PDF, bbox_inches="tight", transparent=True)
    fig.savefig(OUT_PNG, bbox_inches="tight", dpi=150)
    plt.close(fig)
    cb = bnt[bnt["mean"] >= THRESHOLD]["upper_cut"].min()
    cg = grey[grey["mean"] >= THRESHOLD]["upper_cut"].min()
    print(f"rendered -> {OUT_PNG}")
    print(f"  full sky: 0.3σ crossing — BNT bin-1 {'never' if cb != cb else int(cb)}  |  "
          f"non-BNT cut-all {'never' if cg != cg else int(cg)}")


if __name__ == "__main__":
    main()
