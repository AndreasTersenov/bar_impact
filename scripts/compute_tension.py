#!/usr/bin/env python3
"""Null-vs-biased baryon tension across a scale-cut sweep, from saved posteriors.

Consolidated replacement for compute_tension_statistics{,_fullsky,_l1,_peak_counts,
_fixed_nobaryons}.py (PS case implemented first). For every (footprint, upper_cut, run) it
loads the null (nobaryons-vs-nobaryons) and biased (nobaryons-vs-baryonified) posteriors,
computes Gaussian Q_DM tension in the full 6-parameter and the 3-parameter (Ωm,S₈,w₀)
spaces, aggregates across runs to mean ± std, and writes organized tables.

Run with the tensiometer env:
  /home/tersenov/anaconda3/envs/aname/bin/python scripts/compute_tension.py --submean --lmin 37

The default is the new monopole-subtracted ℓ≥37 campaign. Pass --paper-raw to reproduce the
published raw, ℓ≥100 numbers (the Stage-2 regression gate).
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # make `tension` importable

from tension import aggregate, compute, configs, paths       # noqa: E402

COORD_COLS = ["area", "upper_cut"]


def build_campaign(args) -> configs.PSCampaign:
    if args.fullsky:
        camp = configs.fullsky_campaign(lmin=args.lmin)
    elif args.paper_raw:
        camp = configs.paper_raw_l100_campaign()
    else:
        camp = configs.submean_l37_campaign(lmin=args.lmin)
        if args.areas:
            camp.areas = tuple(args.areas)
    if args.upper_cuts:
        camp.upper_cuts = tuple(args.upper_cuts)
    if args.runs:
        camp.runs = tuple(None if r == 0 else r for r in args.runs)
    return camp


def write_tables(records, out_dir, stem, min_runs=1):
    """Write long-form, aggregated (mean±std), and pivot CSVs; return the agg DataFrame."""
    out_dir.mkdir(parents=True, exist_ok=True)
    long_df = aggregate.to_long(records)
    long_df.to_csv(out_dir / f"{stem}_long.csv", index=False, float_format="%.6f")
    agg = aggregate.aggregate_runs(long_df, COORD_COLS, value="nsigma", min_runs=min_runs)
    agg.to_csv(out_dir / f"{stem}_agg.csv", index=False, float_format="%.5f")
    aggregate.pivot(agg, "upper_cut", "area", "mean").to_csv(
        out_dir / f"{stem}_pivot_mean.csv", float_format="%.5f")
    if agg["n"].max() > 1:  # std is meaningful only with >1 run
        aggregate.pivot(agg, "upper_cut", "area", "std").to_csv(
            out_dir / f"{stem}_pivot_std.csv", float_format="%.5f")
    return agg


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--submean", action="store_true", default=True,
                   help="Use monopole-subtracted posteriors (default).")
    p.add_argument("--paper-raw", action="store_true",
                   help="Reproduce the published raw ℓ≥100 numbers (legacy flat layout).")
    p.add_argument("--fullsky", action="store_true",
                   help="Full-sky (healpy) campaign instead of the masked footprints.")
    p.add_argument("--lmin", type=int, default=37, help="ℓ-floor (submean campaign). Default 37.")
    p.add_argument("--areas", type=int, nargs="*", help="Footprints (sqdeg). Default: all six.")
    p.add_argument("--upper-cuts", type=int, nargs="*", help="Upper cuts. Default: paper grid.")
    p.add_argument("--runs", type=int, nargs="*",
                   help="Run indices to aggregate (0 = the unsuffixed base run). Default: base.")
    p.add_argument("--out-dir", type=str, default=None,
                   help="Override output dir (default: the campaign's tables/ dir).")
    p.add_argument("--min-runs", type=int, default=1,
                   help="Only aggregate coordinates with at least this many good runs. "
                        "Use 5 for interim snapshots so partial cuts are excluded. Default 1.")
    args = p.parse_args()

    camp = build_campaign(args)
    out_dir = (paths.Path(args.out_dir) if args.out_dir else paths.tables_dir(camp.tag))

    print(f"Campaign: {camp.tag}  layout={camp.layout}  lmin={camp.lmin}")
    print(f"  areas={list(camp.areas)}")
    print(f"  upper_cuts={camp.upper_cuts[0]}..{camp.upper_cuts[-1]} ({len(camp.upper_cuts)} cuts)")
    print(f"  runs={list(camp.runs)}")

    rec_full, rec_sub, n_missing = compute.collect_records(camp)
    if not rec_sub:
        print(f"No posteriors found for this campaign ({n_missing} coords missing). Nothing written.")
        return
    if n_missing:
        print(f"  ⚠ {n_missing} (coord,run) posterior pairs missing — skipped.")

    agg_full = write_tables(rec_full, out_dir, "tension_6param", min_runs=args.min_runs)
    agg_sub = write_tables(rec_sub, out_dir, "tension_3param", min_runs=args.min_runs)

    print(f"\nTables → {out_dir}")
    print("\n3-param tension nσ (mean; rows=upper_cut, cols=area):")
    print(aggregate.pivot(agg_sub, "upper_cut", "area", "mean").to_string(
        float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
