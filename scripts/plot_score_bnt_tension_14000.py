#!/usr/bin/env python3
"""Score-compressed baryon-tension-vs-scale-cut figure at 14000 deg² (single panel).

Overlays, all 3-param (Ωm,S₈,w₀) Gaussian Q_DM, mean±std over NPE training seeds:
  - score-BNT — cut bin-1 only      (blue,  solid)   bnt_ps_bin1_score_l37
  - score-nonBNT — cut all bins      (grey,  solid)   ps_cutall_score_l37   (matched, compressed)
Optionally (faint, --show-raw) the published raw-NPE curves for the "before/after" context:
  - raw-BNT bin-1 rebin-10 (the under-extracted original blue)   bnt_ps_bin1_submean_l37/tables
  - raw-nonBNT cut-all rebin-10 (the noisy/off-truth grey)        ps_submean_l37/tables

Both score curves are calibrated (TARP/SBC), on-truth nulls, and binning-independent (see
docs/PLAN_score_bnt_tension_14000.md). Run under aname (matplotlib only):
  /home/tersenov/anaconda3/envs/aname/bin/python scripts/plot_score_bnt_tension_14000.py
"""
import argparse
import os

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = "/lustre/fsn1/projects/rech/prk/ulx34io/bar_impact"
AREA = 14000
THRESH = 0.3


def agg(tag, area=AREA):
    p = f"{REPO}/outputs/baryon_tension/{tag}/tension_3param_agg.csv"
    if not os.path.exists(p):
        return pd.DataFrame()
    df = pd.read_csv(p)
    return df[df["area"] == area].sort_values("upper_cut")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show-raw", action="store_true", help="overlay faint published raw curves")
    ap.add_argument("--out", default=f"{REPO}/plots/nsigma_vs_upper_cut_bnt_bin1_score_14000")
    args = ap.parse_args()

    blue = agg("bnt_ps_bin1_score_l37/area14000/tables")
    grey = agg("ps_cutall_score_l37/area14000/tables")

    plt.rcParams.update({"axes.labelsize": 15, "xtick.labelsize": 12, "ytick.labelsize": 12,
                         "legend.fontsize": 11, "axes.titlesize": 14})
    fig, ax = plt.subplots(figsize=(7.2, 5.4))

    if args.show_raw:
        rb = agg("bnt_ps_bin1_submean_l37/tables")   # raw BNT bin-1 (rebin-10) — under-extracted
        rg = agg("ps_submean_l37/tables")             # raw nonBNT cut-all (rebin-10) — noisy
        if len(rb):
            ax.plot(rb["upper_cut"], rb["mean"], ":", color="C0", alpha=0.5, lw=1.6,
                    label="raw-BNT bin-1 (uncompressed, under-extracted)")
        if len(rg):
            ax.plot(rg["upper_cut"], rg["mean"], ":", color="0.55", alpha=0.6, lw=1.6,
                    label="raw-nonBNT cut-all (uncompressed)")

    if len(grey):
        ax.errorbar(grey["upper_cut"], grey["mean"], yerr=grey["std"].fillna(0),
                    fmt="s-", color="0.45", ms=5, lw=1.6, elinewidth=1.1, capsize=3,
                    label="score nonBNT — cut all bins")
    if len(blue):
        ax.errorbar(blue["upper_cut"], blue["mean"], yerr=blue["std"].fillna(0),
                    fmt="o-", color="C0", ms=6, lw=1.8, elinewidth=1.4, capsize=4,
                    label="score BNT — cut bin-1 only (bins 2-4 full)")

    ax.axhline(THRESH, color="r", ls="--", lw=1.3, label=f"Threshold ({THRESH}σ)")
    ax.set_xlabel(r"Upper cut $\ell_{\max}$ (BNT bin-1)")
    ax.set_ylabel(r"Baryon tension $n_\sigma$ (3-param $Q_{\rm DM}$)")
    ax.set_title("Score-compressed baryon tension vs scale cut — 14000 deg²")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3, ls=":")
    ax.legend(loc="upper left")
    fig.text(0.5, 0.005, "monopole-subtracted PS, ℓ≥37, rebin 20 (binning-independent) | "
             "mean±std / 5 seeds | calibrated TARP/SBC, on-truth nulls", ha="center", fontsize=8.5,
             color="0.4")
    plt.tight_layout(rect=(0, 0.03, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}", dpi=150, bbox_inches="tight")
    print(f"wrote {args.out}.png / .pdf")

    def cross(df):
        h = df[df["mean"] >= THRESH]["upper_cut"]
        return int(h.min()) if len(h) else None
    if len(blue):
        print(f"  score-BNT bin-1 0.3σ crossing: {cross(blue) or '>1020'}")
    if len(grey):
        print(f"  score-nonBNT cut-all 0.3σ crossing: {cross(grey) or '>1020'}")


if __name__ == "__main__":
    main()
