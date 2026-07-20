#!/usr/bin/env python3
"""Live monitor for the BNT bin-1 scale-cut pilot (14000 deg²).

Every --interval seconds: recompute the BNT bin-1 tension over whatever cuts have posteriors so
far (read-only, safe mid-sweep), and redraw a single panel in the paper_plots style — the BNT
bin-1-only curve overlaid on the non-BNT cut-everything reference (ps_submean_l37, 14000). The
BNT points fill in as the sweep produces them. Unconnected markers, red 0.3σ threshold, error
bars = std over runs (estimator variance). Stops when all cuts have 5 runs or after --max-hours.

Run with the aname interpreter (tensiometer + matplotlib):
  /home/tersenov/anaconda3/envs/aname/bin/python scripts/monitor_bnt_bin1.py
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
from tension import aggregate, compute, configs, plots  # noqa: E402

REPO = "/home/tersenov/software/bar_impact"
NONBNT_AGG = f"{REPO}/outputs/baryon_tension/ps_submean_l37/tables/tension_3param_agg.csv"
OUT_PDF = f"{REPO}/plots/nsigma_vs_upper_cut_bnt_bin1.pdf"
OUT_PNG = f"{REPO}/plots/nsigma_vs_upper_cut_bnt_bin1.png"
CUTS = list(range(340, 1021, 40))   # the pilot grid (matches the non-BNT 14000 x-axis)
AREA = 14000


def bnt_snapshot(min_runs):
    camp = configs.bnt_bin1_campaign(runs=(1, 2, 3, 4, 5))
    camp.areas = (AREA,)
    camp.upper_cuts = tuple(CUTS)
    _, rec, n_missing = compute.collect_records(camp)
    df = pd.DataFrame()
    if rec:
        df = aggregate.aggregate_runs(aggregate.to_long(rec), ("area", "upper_cut"),
                                      "nsigma", min_runs=min_runs)
    return df, n_missing


def draw(bnt_df, when):
    plots._apply_rcparams()
    fig, ax = plt.subplots(figsize=(7.2, 5.0))

    # non-BNT cut-everything reference (final, all 5 runs)
    if os.path.exists(NONBNT_AGG):
        nb = pd.read_csv(NONBNT_AGG)
        nb = nb[nb["area"] == AREA].sort_values("upper_cut")
        if len(nb):
            ax.errorbar(nb["upper_cut"], nb["mean"], yerr=nb["std"].fillna(0), fmt="s",
                        color="0.55", ms=5, elinewidth=1.2, capsize=3,
                        label="non-BNT — cut all bins")

    # BNT bin-1-only (live)
    if len(bnt_df):
        b = bnt_df.sort_values("upper_cut")
        ax.errorbar(b["upper_cut"], b["mean"], yerr=b["std"].fillna(0), fmt="o",
                    color="C0", ms=6, elinewidth=1.5, capsize=4,
                    label="BNT — cut bin-1 only (bins 2-4 full)")

    ax.axhline(0.3, color="r", linestyle="--", linewidth=1.5, label="Threshold (0.3)")
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.set_title(rf"Area = {AREA} deg$^2$ — baryon tension vs scale cut")
    ax.set_xlabel(r"Upper Cut ($\ell_{\mathrm{max}}$)")
    ax.set_ylabel(r"Significance ($n_\sigma$)")
    ax.legend(loc="upper left", fontsize=11)
    fig.text(0.5, 0.965, f"live {when} | BNT bin-1 ℓmax swept, bins 2-4 at full range | "
             "3-param Q_DM, mean±std/5 runs | monopole-subtracted PS ℓ≥37",
             ha="center", va="top", fontsize=8.5, color="0.4")
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    os.makedirs(os.path.dirname(OUT_PDF), exist_ok=True)
    fig.savefig(OUT_PDF, bbox_inches="tight", transparent=True)
    fig.savefig(OUT_PNG, bbox_inches="tight", dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--interval", type=int, default=180)
    p.add_argument("--min-runs", type=int, default=1)
    p.add_argument("--max-hours", type=float, default=2.5)
    args = p.parse_args()
    print(f"[monitor-bnt] -> {OUT_PNG}  interval={args.interval}s")
    start = time.time()
    while True:
        try:
            df, n_missing = bnt_snapshot(args.min_runs)
            when = f"{datetime.now():%H:%M}"
            draw(df, when)
            done = len(df)
            complete = bool(len(df) and (df["n"] >= 5).all())
            line = (f"{datetime.now():%H:%M:%S}  bnt cuts with data={done}/{len(CUTS)}  "
                    f"complete(5runs)={complete}  missing coord-runs={n_missing}")
        except Exception:
            line = f"{datetime.now():%H:%M:%S}  ERROR: {traceback.format_exc().splitlines()[-1]}"
            done, complete = 0, False
        print(line, flush=True)
        if complete and done >= len(CUTS):
            print("[monitor-bnt] all cuts complete (5 runs). final plot done. exiting.", flush=True)
            break
        if (time.time() - start) / 3600.0 > args.max_hours:
            print("[monitor-bnt] max-hours — exiting.", flush=True)
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
