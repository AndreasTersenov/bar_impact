#!/usr/bin/env python3
"""Live monitor for the baryon-tension sweep.

Every --interval seconds: recompute Q_DM tension over COMPLETE coordinates (≥ --min-runs
runs for both roles), rewrite the 3-param aggregate table, and refresh the σ-vs-cut figure
IN PLACE (same file each time, no versioning) so you can watch the curve fill in. Stops when
every coordinate is complete or after --max-hours.

Read-only on posteriors — safe to run alongside the producing sweep. Run with the aname
interpreter (tensiometer + matplotlib):
  /home/tersenov/anaconda3/envs/aname/bin/python scripts/monitor_tension_pilot.py \
      --areas 14000 --runs 1 2 3 4 5 --interval 600
"""
import argparse
import os
import shutil
import sys
import time
import traceback
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # make `tension` importable

from tension import aggregate, compute, configs, paths, plots  # noqa: E402

PLOT_PDF = "/lustre/fsn1/projects/rech/prk/ulx34io/bar_impact/plots/nsigma_vs_upper_cut_masks.pdf"
PLOT_PNG = "/lustre/fsn1/projects/rech/prk/ulx34io/bar_impact/plots/nsigma_vs_upper_cut_masks.png"
COORDS = ("area", "upper_cut")


def snapshot(camp, min_runs):
    """One refresh: returns (n_complete_coords, n_posteriors_missing, n_panels_drawn)."""
    _, rec_sub, n_missing = compute.collect_records(camp)
    long_sub = aggregate.to_long(rec_sub)
    if len(long_sub) == 0:
        return 0, n_missing, 0

    agg = aggregate.aggregate_runs(long_sub, COORDS, "nsigma", min_runs=min_runs)
    tdir = paths.tables_dir(camp.tag)
    tdir.mkdir(parents=True, exist_ok=True)
    long_sub.to_csv(tdir / "tension_3param_long.csv", index=False, float_format="%.6f")
    if len(agg):
        agg.to_csv(tdir / "tension_3param_agg.csv", index=False, float_format="%.5f")

    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    n_complete = int(len(agg))
    subtitle = (f"live: {stamp}  |  complete cuts (n≥{min_runs} per role): {n_complete}  "
                f"|  3-param Q_DM tension")
    n_panels = plots.plot_nsigma_vs_cut(agg, list(camp.areas), PLOT_PDF, PLOT_PNG,
                                        subtitle=subtitle) if len(agg) else 0
    return n_complete, n_missing, n_panels


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--areas", type=int, nargs="*", default=[14000])
    p.add_argument("--runs", type=int, nargs="*", default=[1, 2, 3, 4, 5])
    p.add_argument("--upper-cuts", type=int, nargs="*", default=None,
                   help="Cut grid to track (must match the sweep). Default: paper step-20.")
    p.add_argument("--min-runs", type=int, default=5)
    p.add_argument("--interval", type=int, default=600, help="Seconds between refreshes.")
    p.add_argument("--max-hours", type=float, default=9.0)
    p.add_argument("--once", action="store_true", help="Single snapshot then exit (for testing).")
    args = p.parse_args()

    camp = configs.submean_l37_campaign(runs=tuple(args.runs))
    camp.areas = tuple(args.areas)
    if args.upper_cuts:
        camp.upper_cuts = tuple(args.upper_cuts)
    target = len(camp.areas) * len(camp.upper_cuts)
    status_log = paths.tables_dir(camp.tag) / "interim_status.log"
    status_log.parent.mkdir(parents=True, exist_ok=True)

    # Back up an existing plot once so we never destroy a pre-existing figure.
    if os.path.exists(PLOT_PDF) and not os.path.exists(PLOT_PDF + ".bak"):
        shutil.copy(PLOT_PDF, PLOT_PDF + ".bak")

    print(f"[monitor] campaign={camp.tag} areas={list(camp.areas)} target_coords={target} "
          f"min_runs={args.min_runs} interval={args.interval}s -> {PLOT_PNG}")
    start = time.time()
    while True:
        try:
            n_complete, n_missing, n_panels = snapshot(camp, args.min_runs)
            line = (f"{datetime.now():%H:%M:%S}  complete_cuts={n_complete}/{target}  "
                    f"missing_posteriors={n_missing}  panels={n_panels}")
        except Exception:
            line = f"{datetime.now():%H:%M:%S}  SNAPSHOT ERROR: {traceback.format_exc().splitlines()[-1]}"
            n_complete = -1
        print(line, flush=True)
        with open(status_log, "a") as fh:
            fh.write(line + "\n")

        if args.once:
            break
        if n_complete >= target:
            print("[monitor] all coordinates complete — final snapshot done. exiting.", flush=True)
            break
        if (time.time() - start) / 3600.0 > args.max_hours:
            print("[monitor] max-hours reached — exiting.", flush=True)
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
