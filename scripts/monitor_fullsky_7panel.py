#!/usr/bin/env python3
"""Live 7-panel monitor while the full-sky sweep runs.

Every --interval seconds: recompute the full-sky tension over COMPLETE cuts (≥ --min-runs),
merge it with the already-final 6 masked footprints, and refresh the combined 7-panel figure
in place. The full-sky panel fills in as its cuts complete. Stops when full sky is complete
or after --max-hours.

Run with the aname interpreter (tensiometer + matplotlib).
"""
import argparse
import os
import sys
import time
import traceback
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tension import aggregate, compute, configs, plots  # noqa: E402

REPO = "/lustre/fsn1/projects/rech/prk/ulx34io/bar_impact"
MASKED_AGG = f"{REPO}/outputs/baryon_tension/ps_submean_l37/tables/tension_3param_agg.csv"
OUT_PDF = f"{REPO}/plots/nsigma_vs_upper_cut_with_fullsky.pdf"
OUT_PNG = f"{REPO}/plots/nsigma_vs_upper_cut_with_fullsky.png"
CUTS = [100, 140, 180, 220, 260, 300,
        340, 380, 420, 460, 500, 540, 580, 620, 660, 700, 740, 780, 820, 860, 900, 940, 980, 1020]
AREAS = [2000, 5000, 10000, 14000, 28000, 35000, "fullsky"]


def snapshot(min_runs):
    camp = configs.fullsky_campaign(runs=(1, 2, 3, 4, 5))
    camp.upper_cuts = tuple(CUTS)
    _, rec, _ = compute.collect_records(camp)
    fs = pd.DataFrame()
    if rec:
        fs = aggregate.aggregate_runs(aggregate.to_long(rec), ("area", "upper_cut"),
                                      "nsigma", min_runs=min_runs)
    masked = pd.read_csv(MASKED_AGG).astype({"area": object})
    merged = pd.concat([masked, fs], ignore_index=True) if len(fs) else masked
    sub = (f"6 footprints + full sky (live {datetime.now():%H:%M}) | full-sky cuts done: "
           f"{len(fs)}/{len(CUTS)} | masked nlb=4 (40-ℓ), full sky healpy (10-ℓ)")
    n = plots.plot_nsigma_vs_cut(merged, AREAS, OUT_PDF, OUT_PNG, subtitle=sub, dedup=True)
    return len(fs), n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--interval", type=int, default=300)
    p.add_argument("--min-runs", type=int, default=5)
    p.add_argument("--max-hours", type=float, default=3.0)
    args = p.parse_args()
    print(f"[monitor7] -> {OUT_PNG}  interval={args.interval}s")
    start = time.time()
    while True:
        try:
            done, npan = snapshot(args.min_runs)
            line = f"{datetime.now():%H:%M:%S}  fullsky_cuts={done}/{len(CUTS)}  panels={npan}"
        except Exception:
            line = f"{datetime.now():%H:%M:%S}  ERROR: {traceback.format_exc().splitlines()[-1]}"
            done = 0
        print(line, flush=True)
        if done >= len(CUTS):
            print("[monitor7] full sky complete — final 7-panel done. exiting.", flush=True)
            break
        if (time.time() - start) / 3600.0 > args.max_hours:
            print("[monitor7] max-hours — exiting.", flush=True)
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
