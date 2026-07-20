#!/usr/bin/env python3
"""Launch the NPE scale-cut sweep for the baryon-tension campaign.

Thin entrypoint over scripts/tension/sweep.py. Runs the patched worker for every
(footprint, upper_cut, role, run) with per-run seeds, GPU pooling, NaN-retry, and QA.

Run with the jaxili interpreter:
  /home/tersenov/anaconda3/envs/jaxili/bin/python scripts/run_tension_sweep.py \
      --areas 14000 --gpus 2 3 --runs 1 2 3 4 5

Use --dry-run first to inspect the job plan (counts + an example worker command) without
training anything.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # make `tension` importable

from tension import configs, sweep  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lmin", type=int, default=37, help="ℓ-floor (default 37).")
    p.add_argument("--areas", type=int, nargs="*", help="Footprints. Default: all six.")
    p.add_argument("--upper-cuts", type=int, nargs="*", help="Upper cuts. Default: paper grid.")
    p.add_argument("--rebin", type=int, default=None,
                   help="ℓ-rebin factor (default: campaign's, 10). Raise (e.g. 40) to coarsen the "
                        "full-sky healpy data vector toward the masked 40-ℓ binning.")
    p.add_argument("--runs", type=int, nargs="*", default=[1, 2, 3, 4, 5],
                   help="Run indices (each a distinct seed). Default: 1..5.")
    p.add_argument("--gpus", type=int, nargs="*", default=[0, 1],
                   help="GPU indices to pool over.")
    p.add_argument("--seed-base", type=int, default=100,
                   help="Per-run seed = seed_base + run. Default 100.")
    p.add_argument("--max-retries", type=int, default=3,
                   help="Retries (with bumped seed) on NaN-loss / crash. Default 3.")
    p.add_argument("--jobs-per-gpu", type=int, default=1,
                   help="Concurrent jobs per GPU. NPE jobs are light (~30%% util) with a long "
                        "CPU init phase, so 3-4 packs well. Default 1.")
    p.add_argument("--mem-fraction", type=float, default=None,
                   help="Cap JAX per-process GPU preallocation (XLA_PYTHON_CLIENT_MEM_FRACTION). "
                        "Needed when packing — e.g. 0.15 (~6 GB). Default: jaxili default (~0.75).")
    p.add_argument("--fullsky", action="store_true",
                   help="Full-sky (healpy) campaign instead of the masked footprints.")
    p.add_argument("--bnt-bin1", action="store_true",
                   help="BNT campaign: sweep ONLY BNT bin-1's ℓmax (bins 2-4 full). See "
                        "docs/BNT_on_spectra.md.")
    p.add_argument("--bnt-cutall", action="store_true",
                   help="BNT campaign: sweep a uniform ℓmax on all BNT bins (cut-everything control).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the job plan and an example command; do not train.")
    args = p.parse_args()

    if args.fullsky and args.bnt_bin1:
        camp = configs.fullsky_bnt_bin1_campaign(lmin=args.lmin, runs=tuple(args.runs))
    elif args.fullsky:
        camp = configs.fullsky_campaign(lmin=args.lmin, runs=tuple(args.runs))
    elif args.bnt_bin1:
        camp = configs.bnt_bin1_campaign(lmin=args.lmin, runs=tuple(args.runs))
    elif args.bnt_cutall:
        camp = configs.bnt_cutall_campaign(lmin=args.lmin, runs=tuple(args.runs))
    else:
        camp = configs.submean_l37_campaign(lmin=args.lmin, runs=tuple(args.runs))
    if not args.fullsky and args.areas:
        camp.areas = tuple(args.areas)
    if args.upper_cuts:
        camp.upper_cuts = tuple(args.upper_cuts)
    if args.rebin:
        camp.rebin = args.rebin

    if args.dry_run:
        jobs = sweep.plan_jobs(camp, args.seed_base)
        print(f"Campaign: {camp.tag}")
        print(f"  areas={list(camp.areas)}  upper_cuts={camp.upper_cuts[0]}..{camp.upper_cuts[-1]}"
              f" ({len(camp.upper_cuts)} cuts)  runs={list(camp.runs)}")
        total = len(camp.areas) * len(camp.upper_cuts) * 2 * len(camp.runs)
        print(f"  total grid jobs = {len(camp.areas)}×{len(camp.upper_cuts)}×2×{len(camp.runs)} = {total}")
        print(f"  to run now (existing skipped) = {len(jobs)}")
        by_role = {"null": 0, "biased": 0}
        for j in jobs:
            by_role[j.role] += 1
        print(f"  by role: {by_role}")
        if jobs:
            print("\nExample worker command (first job):")
            print("  " + " ".join(sweep.build_worker_cmd(camp, jobs[0], gpu=args.gpus[0])))
            print(f"\nExample output -> {jobs[0].out_path}")
        return

    sweep.run_sweep(camp, gpus=args.gpus, seed_base=args.seed_base,
                    max_retries=args.max_retries, jobs_per_gpu=args.jobs_per_gpu,
                    mem_fraction=args.mem_fraction)


if __name__ == "__main__":
    main()
