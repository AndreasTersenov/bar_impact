#!/usr/bin/env python3
"""Orchestrator for the score-compressed BNT bin-1 tension-vs-scale-cut sweep (14000 deg²).

STAGE A (CPU, numpy): for each BNT bin-1 cut c (cuts=[c,1024,1024,1024]) build the MLE-form score
weights from the full-vector dump (slice columns to the cut; rebuild J,C,Wmle at the cut via the
validated fisher modules) and write a tiny summaries/cut<c>.npz = (theta, That[16965x6], t_null[6],
t_biased[6], sigma3_fisher). The grid is dumped ONCE at full and sliced here — see
docs/PLAN_score_bnt_tension_14000.md.

STAGE B (GPU): fan the (cut, seed) jobs across `jobs_per_gpu` packed slots on the requested GPU(s);
each slot is a long-lived score_bnt_npe_worker.py process looping its partition (jax import paid
once per slot). TARP/SBC runs on the lowest seed of each cut (the calibration gate at every cut).

Run with the jaxili interpreter (Stage A is numpy; Stage B subprocesses need jaxili+jax).
  python scripts/score_bnt_tension_sweep.py --area 14000 --gpus 2 --jobs-per-gpu 4 \
      --cuts 460 700 1020 --seeds 41 42 43           # pilot
  python scripts/score_bnt_tension_sweep.py --area 14000 --gpus 2 --jobs-per-gpu 4   # full grid
"""
import argparse
import os
import subprocess
import sys
import time

import numpy as np

REPO = "/mnt/home/tersenov/software/bar_impact"
PY_JAXILI = "/home/tersenov/anaconda3/envs/jaxili/bin/python"
FULL_GRID = list(range(340, 1021, 40))   # 18 step-40 cuts: 340,380,...,1020
DEFAULT_SEEDS = [41, 42, 43, 44, 45]
CACHE = f"{REPO}/outputs/score_experiment/cache"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--area", type=int, default=14000)
    p.add_argument("--rebin", type=int, default=20, help="ell-rebin for the score INPUT vector. "
                   "20 -> hybrid cov OK (nfeat<200). Finer (e.g. 10) needs --covk analytic.")
    p.add_argument("--cuts", type=int, nargs="*", default=FULL_GRID)
    p.add_argument("--seeds", type=int, nargs="*", default=DEFAULT_SEEDS)
    p.add_argument("--gpus", type=int, nargs="*", default=[2])
    p.add_argument("--jobs-per-gpu", type=int, default=4)
    p.add_argument("--mem-fraction", type=float, default=0.2)
    p.add_argument("--covk", default="hybrid", choices=["hybrid", "analytic"])
    p.add_argument("--mode", default="bnt_bin1", choices=["bnt_bin1", "nonbnt_cutall"],
                   help="bnt_bin1: cut ONLY BNT bin-1 (bins 2-4 full). nonbnt_cutall: cut ALL bins "
                        "uniformly, no BNT (the matched, compressed grey reference).")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--num-samples", type=int, default=3000)
    p.add_argument("--tag", default=None, help="output tag; default depends on --mode")
    p.add_argument("--skip-build", action="store_true", help="reuse existing summaries/cut*.npz")
    p.add_argument("--build-only", action="store_true", help="Stage A only; no NPE")
    return p.parse_args()


def _cache_tag(args, sim):
    rt = "" if args.rebin == 20 else f"_r{args.rebin}"
    pre = "bnt" if args.mode == "bnt_bin1" else "nonbnt"
    return f"{pre}_full{rt}_{args.area}_{sim}"


def _cuts_for(args, c):
    """Per-bin cut vector for a swept ℓmax c."""
    return [c, 1024, 1024, 1024] if args.mode == "bnt_bin1" else [c, c, c, c]


def _is_bnt(args):
    return args.mode == "bnt_bin1"


def _default_tag(args):
    return "bnt_ps_bin1_score_l37" if args.mode == "bnt_bin1" else "ps_cutall_score_l37"


def stage_a_build(args, out_root):
    """Build per-cut score summaries by slicing the full dump + rebuilding J,C,Wmle at each cut."""
    os.environ["FISHER_AREA"] = str(args.area)
    os.environ["FISHER_REBIN"] = str(args.rebin)   # fisher modules read this at import
    sys.path.insert(0, os.path.join(REPO, "scripts"))
    import score_cut_utils as S

    sdir = os.path.join(out_root, "summaries")
    os.makedirs(sdir, exist_ok=True)
    znull = np.load(f"{CACHE}/{_cache_tag(args, 'nobary')}/cache.npz")
    zbias = np.load(f"{CACHE}/{_cache_tag(args, 'bary')}/cache.npz")
    theta, x_full, xfid_null = znull["theta"], znull["x"], znull["x_fid"]
    xfid_bias = zbias["x_fid"]
    assert np.array_equal(theta, zbias["theta"]), "null/biased dumps disagree on theta"

    print(f"[stageA] {len(args.cuts)} cuts, covk={args.covk}, full nfeat={x_full.shape[1]}")
    for c in args.cuts:
        out = os.path.join(sdir, f"cut{c}.npz")
        if args.skip_build and os.path.exists(out):
            print(f"[stageA] cut{c} exists -> skip"); continue
        cuts = _cuts_for(args, c)
        keep = S.keep_indices(cuts)
        sc = S.build_score(cuts, bnt=_is_bnt(args), covk=args.covk)
        W = sc["Wmle"]
        That = S.FID + (x_full[:, keep] - xfid_null[keep]) @ W
        t_null = S.FID + (xfid_null[keep] - xfid_null[keep]) @ W      # == FID by construction
        t_biased = S.FID + (xfid_bias[keep] - xfid_null[keep]) @ W
        sig3 = np.sqrt(np.diag(np.linalg.inv(sc["F"])))[[0, 1, 2]]
        np.savez(out, theta=theta.astype(np.float32), That=That.astype(np.float32),
                 t_null=t_null.astype(np.float32), t_biased=t_biased.astype(np.float32),
                 sigma3_fisher=sig3.astype(np.float64), nfeat=sc["nfeat"])
        bias_sig = np.abs(t_biased[:3] - S.FID[:3]) / sig3
        print(f"[stageA] cut{c:4d} nfeat={sc['nfeat']:3d} sig3={np.round(sig3,4)} "
              f"bias/sig={np.round(bias_sig,2)} (Om,S8,w0)")
    return sdir


def plan_jobs(args, out_root):
    """(cut, seed) jobs whose posteriors don't both exist yet (resume)."""
    pdir = os.path.join(out_root, "posteriors")
    jobs = []
    for c in args.cuts:
        for s in args.seeds:
            fn = os.path.join(pdir, f"cut{c}", f"null_run{s}.npy")
            fb = os.path.join(pdir, f"cut{c}", f"biased_run{s}.npy")
            if os.path.exists(fn) and os.path.exists(fb):
                continue
            jobs.append((c, s))
    return jobs


def stage_b_run(args, out_root, sdir):
    pdir = os.path.join(out_root, "posteriors")
    os.makedirs(pdir, exist_ok=True)
    logdir = os.path.join(out_root, "logs")
    os.makedirs(logdir, exist_ok=True)
    jobs = plan_jobs(args, out_root)
    if not jobs:
        print("[stageB] nothing to do (all posteriors present)"); return
    # TARP/SBC on the lowest seed of each cut.
    tarp_set = {(c, min(args.seeds)) for c in args.cuts}

    slots = [g for g in args.gpus for _ in range(args.jobs_per_gpu)]
    nslots = len(slots)
    parts = [[] for _ in range(nslots)]
    for i, job in enumerate(jobs):          # round-robin across slots
        parts[i % nslots].append(job)
    print(f"[stageB] {len(jobs)} jobs over {nslots} slots on gpus {args.gpus} "
          f"(jobs_per_gpu={args.jobs_per_gpu}, mem_fraction={args.mem_fraction})")

    procs = []
    for si, part in enumerate(parts):
        if not part:
            continue
        gpu = slots[si]
        jobs_arg = ",".join(f"{c}:{s}" for c, s in part)
        tarp_arg = ",".join(f"{c}:{s}" for c, s in part if (c, s) in tarp_set)
        log = open(os.path.join(logdir, f"worker_slot{si}_gpu{gpu}.log"), "a")
        cmd = [PY_JAXILI, os.path.join(REPO, "scripts", "score_bnt_npe_worker.py"),
               "--summaries-dir", sdir, "--out-dir", pdir, "--jobs", jobs_arg,
               "--tarp-seeds", tarp_arg, "--gpu", str(gpu),
               "--mem-fraction", str(args.mem_fraction), "--epochs", str(args.epochs),
               "--num-samples", str(args.num_samples)]
        procs.append((subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=REPO), log, si))
        print(f"[stageB] slot{si} gpu{gpu}: {len(part)} jobs -> logs/worker_slot{si}_gpu{gpu}.log")
        time.sleep(12)   # stagger CUDA init: simultaneous JAX starts race -> "no supported devices"

    failed = 0
    while procs:
        for tup in list(procs):
            proc, log, si = tup
            rc = proc.poll()
            if rc is None:
                continue
            log.close(); procs.remove(tup)
            print(f"[stageB] slot{si} exited rc={rc}")
            if rc != 0:
                failed += 1
        time.sleep(5)
    done = len(plan_jobs(args, out_root))
    print(f"[stageB] all slots finished ({failed} slot errors); {done} (cut,seed) jobs still missing")


def main():
    args = parse_args()
    if args.tag is None:
        args.tag = _default_tag(args)
    rt = "" if args.rebin == 20 else f"_r{args.rebin}"
    out_root = f"{REPO}/outputs/baryon_tension/{args.tag}/area{args.area}{rt}"
    os.makedirs(out_root, exist_ok=True)
    sdir = stage_a_build(args, out_root)
    if args.build_only:
        print("[main] build-only -> done"); return
    stage_b_run(args, out_root, sdir)
    print(f"[main] sweep done -> {out_root}")


if __name__ == "__main__":
    main()
