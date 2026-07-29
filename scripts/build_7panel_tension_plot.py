#!/usr/bin/env python3
"""Build the 7-panel σ-vs-cut figure: the 6 masked footprints + full sky.

Merges the masked campaign's aggregated tension (ps_submean_l37) with the full-sky
campaign's (ps_fullsky_l37) into one row of panels, in the paper_plots.ipynb style.
Full sky is the healpy pipeline (10-ℓ bins) vs the masked NaMaster nlb=4 (40-ℓ) — same
scale-cut trend, not magnitude-comparable (plan option a). Sampled at the same step-40 cuts.

Run with the aname interpreter (matplotlib):
  /home/tersenov/anaconda3/envs/aname/bin/python scripts/build_7panel_tension_plot.py
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # make `tension` importable

from tension import plots  # noqa: E402

REPO = "/lustre/fsn1/projects/rech/prk/ulx34io/bar_impact"
MASKED = f"{REPO}/outputs/baryon_tension/ps_submean_l37/tables/tension_3param_agg.csv"
FULLSKY = f"{REPO}/outputs/baryon_tension/ps_fullsky_l37/tables/tension_3param_agg.csv"
OUT_PDF = f"{REPO}/plots/nsigma_vs_upper_cut_with_fullsky.pdf"
OUT_PNG = f"{REPO}/plots/nsigma_vs_upper_cut_with_fullsky.png"


def main():
    masked = pd.read_csv(MASKED)
    fullsky = pd.read_csv(FULLSKY)  # 'area' column holds the string "fullsky"
    # Keep the area column object-typed so int footprints and the "fullsky" sentinel coexist.
    merged = pd.concat([masked.astype({"area": object}), fullsky], ignore_index=True)

    areas = [2000, 5000, 10000, 14000, 28000, 35000, "fullsky"]
    n = plots.plot_nsigma_vs_cut(
        merged, areas, OUT_PDF, OUT_PNG,
        subtitle="6 footprints + full sky | monopole-subtracted PS, ℓ≥37 | step-40 | "
                 "3-param Q_DM, mean±std/5 runs  (full sky = healpy 10-ℓ; masked = nlb=4 40-ℓ)",
        dedup=True,
    )
    print(f"rendered {n} panels -> {OUT_PNG}")
    # quick 0.3σ-crossing readout per panel
    for a in areas:
        sub = merged[merged["area"] == a].sort_values("upper_cut")
        if not len(sub):
            print(f"  {a}: NO DATA"); continue
        c = sub[sub["mean"] >= 0.3]["upper_cut"].min()
        label = "full sky" if a == "fullsky" else f"{a} deg²"
        print(f"  {label:>12}: 0.3σ at ℓmax {'≈'+str(int(c)) if c == c else '>1020'}")


if __name__ == "__main__":
    main()
