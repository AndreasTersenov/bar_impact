#!/usr/bin/env python3
"""Live 6-panel BNT bin-1 vs non-BNT-cut-all monitor (all masked footprints).

The multi-area generalization of monitor_bnt_bin1.py. Every --interval seconds it recomputes
the BNT bin-1 tension over whatever posteriors exist so far (read-only — safe mid-sweep) for
every footprint, and redraws a 6-panel figure in the paper_plots style: each panel overlays
the live BNT bin-1-only curve (blue, fills in as the sweep produces points) on the final
non-BNT cut-everything reference (grey, from ps_submean_l37). Red 0.3σ threshold; error bars =
std over the 5 training-seed runs. Stops when every footprint has all cuts at 5 runs, or after
--max-hours.

Run with the aname interpreter (tensiometer + matplotlib):
  /home/tersenov/anaconda3/envs/aname/bin/python scripts/monitor_bnt_bin1_allareas.py
"""
import argparse
import os
import sys
import time
import traceback
from datetime import datetime

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tension import aggregate, compute, configs  # noqa: E402

REPO = "/mnt/home/tersenov/software/bar_impact"
NONBNT_AGG = f"{REPO}/outputs/baryon_tension/ps_submean_l37/tables/tension_3param_agg.csv"
OUT_PNG = f"{REPO}/plots/nsigma_vs_upper_cut_bnt_bin1_allareas_live.png"
OUT_PDF = f"{REPO}/plots/nsigma_vs_upper_cut_bnt_bin1_allareas_live.pdf"
CUTS = list(range(340, 1021, 40))          # step-40 grid (matches the non-BNT x-axis)
AREAS = [2000, 5000, 10000, 14000, 28000, 35000]
THRESHOLD = 0.3


def _apply_rcparams():
    plt.rcParams["legend.fontsize"] = 12
    plt.rcParams["axes.labelsize"] = 14
    plt.rcParams["xtick.labelsize"] = 12
    plt.rcParams["ytick.labelsize"] = 12
    plt.rcParams["axes.titlesize"] = 14


def bnt_snapshot(min_runs=1):
    """Live BNT bin-1 aggregated tension over all areas (mean±std over runs)."""
    camp = configs.bnt_bin1_campaign(runs=(1, 2, 3, 4, 5))
    camp.areas = tuple(AREAS)
    camp.upper_cuts = tuple(CUTS)
    _, rec, n_missing = compute.collect_records(camp)
    df = pd.DataFrame()
    if rec:
        df = aggregate.aggregate_runs(aggregate.to_long(rec), ("area", "upper_cut"),
                                      "nsigma", min_runs=min_runs)
    return df, n_missing


def draw(bnt_df, nonbnt_df, when, progress):
    _apply_rcparams()
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
    axes = axes.flatten()

    for ax, area in zip(axes, AREAS):
        if len(nonbnt_df):
            nb = nonbnt_df[nonbnt_df["area"] == area].sort_values("upper_cut")
            if len(nb):
                ax.errorbar(nb["upper_cut"], nb["mean"], yerr=nb["std"].fillna(0), fmt="s",
                            color="0.55", ms=5, elinewidth=1.2, capsize=3,
                            label="non-BNT — cut all bins")
        nb_done = 0
        if len(bnt_df):
            b = bnt_df[bnt_df["area"] == area].sort_values("upper_cut")
            nb_done = len(b)
            if len(b):
                ax.errorbar(b["upper_cut"], b["mean"], yerr=b["std"].fillna(0), fmt="o",
                            color="C0", ms=6, elinewidth=1.5, capsize=4,
                            label="BNT — cut bin-1 only (bins 2-4 full)")
        ax.axhline(THRESHOLD, color="r", linestyle="--", linewidth=1.4,
                   label=f"Threshold ({THRESHOLD})")
        ax.grid(True, alpha=0.3, linestyle=":")
        ax.set_title(rf"Area = {area} deg$^2$  ({nb_done}/{len(CUTS)} cuts)")
        ax.set_ylim(bottom=0)

    # one shared legend (from the first panel's handles) + shared axis labels
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(handles, labels, loc="upper left", fontsize=11)
    fig.supxlabel(r"Upper Cut ($\ell_{\mathrm{max}}$)", fontsize=15, y=0.04)
    fig.supylabel(r"Significance ($n_\sigma$)", fontsize=15, x=0.06)
    fig.suptitle("BNT bin-1-only vs non-BNT cut-all — baryon tension vs scale cut", fontsize=16, y=0.98)
    fig.text(0.5, 0.945, f"live {when} | {progress} | monopole-subtracted PS ℓ≥37, step-40 | "
             "3-param Q_DM, mean±std / 5 runs  (BNT bin-1 ℓmax swept, bins 2-4 full)",
             ha="center", va="top", fontsize=10, color="0.4")
    plt.tight_layout(rect=(0.06, 0.05, 1, 0.94))

    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.savefig(OUT_PDF, bbox_inches="tight", transparent=True)
    fig.savefig(OUT_PNG, bbox_inches="tight", dpi=130)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--interval", type=int, default=120)
    p.add_argument("--min-runs", type=int, default=1)
    p.add_argument("--max-hours", type=float, default=6.0)
    args = p.parse_args()
    print(f"[monitor-allareas] -> {OUT_PNG}  interval={args.interval}s", flush=True)

    nonbnt = pd.read_csv(NONBNT_AGG) if os.path.exists(NONBNT_AGG) else pd.DataFrame()
    target = len(AREAS) * len(CUTS)  # (area,cut) coords fully done at 5 runs
    start = time.time()
    while True:
        try:
            df, n_missing = bnt_snapshot(args.min_runs)
            when = f"{datetime.now():%H:%M}"
            done = int((df["n"] >= 5).sum()) if len(df) else 0
            have = len(df)
            progress = f"BNT bin-1 coords: {done}/{target} at 5 runs ({have} with ≥1)"
            draw(df, nonbnt, when, progress)
            line = (f"{datetime.now():%H:%M:%S}  {progress}  missing(coord,run)={n_missing}")
            complete = done >= target
        except Exception:
            line = f"{datetime.now():%H:%M:%S}  ERROR: {traceback.format_exc().splitlines()[-1]}"
            complete = False
        print(line, flush=True)
        if complete:
            print("[monitor-allareas] all areas complete (5 runs each). final plot done. exiting.",
                  flush=True)
            break
        if (time.time() - start) / 3600.0 > args.max_hours:
            print("[monitor-allareas] max-hours reached — exiting.", flush=True)
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
