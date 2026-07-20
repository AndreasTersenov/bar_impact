#!/usr/bin/env python3
"""Overnight: gated VMIM hyperparameter sweep + 4-config comparison + morning summary.

The smoke showed the VMIM summary is prone to OVER-CONFIDENCE (null off-truth, σ spuriously tight)
— the recipe's lessons 5-7. So rather than one run, this sweeps the bottleneck knobs
(summary_noise, summary_dim) on bnt_full, scores each by a calibration oracle (null S8 near truth AND
σ(S8) not collapsed below the whitening value), picks the winner, then runs the 4-config NULL
comparison (bnt_full/nonbnt_full oracle; bnt_580/nonbnt_460 payoff) with 3 NPE seeds, and writes
outputs/baryon_tension/vmim/MORNING_SUMMARY.md. Honest PASS/PARTIAL/FAIL with all numbers.

Run with the jaxili interpreter (nohup). Each stage is time-boxed; partial summary is written as it goes.
"""
import json
import os
import subprocess
import sys
import time
from datetime import datetime

import numpy as np

REPO = "/home/tersenov/software/bar_impact"
PY = "/home/tersenov/anaconda3/envs/jaxili/bin/python"
WORKER = f"{REPO}/scripts/run_npe_inference_auto_cross_ps_master.py"
ROOT = f"{REPO}/outputs/baryon_tension/vmim"
GPU = os.environ.get("VMIM_GPU", "0")
TRUTH_S8, TRUTH_OM = 0.84, 0.26
WHITEN_SIG_S8 = 0.024          # the whitening reference σ(S8); below ~0.018 => suspect over-confidence

# config tag -> extra worker flags (NULL posteriors: sim & fid both nobaryons)
CONFIGS = {
    "bnt_full":   ["--bnt", "--bnt-bins", "0,1,2,3", "--upper-cut", "1024"],
    "nonbnt_full": ["--upper-cut", "1024"],
    "bnt_580":    ["--bnt", "--bnt-bins", "0,1,2,3", "--upper-cuts", "580,1024,1024,1024"],
    "nonbnt_460": ["--upper-cut", "460"],
}
BASE = ["--simulation-type", "nobaryons", "--fiducial-type", "nobaryons",
        "--masked", "--mask-area-sqdeg", "14000.0", "--apodization-scale-deg", "2.0",
        "--noisy", "--noise-level", "0.26", "--subtract-mean", "--lmax", "1535",
        "--lower-cut", "37", "--rebin", "10"]

log_lines = []


def log(m):
    line = f"{datetime.now():%H:%M:%S} {m}"
    print(line, flush=True)
    log_lines.append(line)


def run(cmd, timeout):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode == 0, p.stdout + p.stderr
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"


def dump_cache(tag):
    d = f"{ROOT}/cache/{tag}"
    if os.path.exists(f"{d}/cache.npz"):
        return d
    ok, out = run([PY, WORKER] + BASE + CONFIGS[tag] + ["--samples-dir", f"{ROOT}/junk",
                  "--dump-cache", d], timeout=1200)
    log(f"dump {tag}: {'OK' if ok else 'FAIL'}")
    return d if ok else None


def compress(cache_dir, out, noise, dim, steps, mins):
    ok, _ = run([PY, f"{REPO}/scripts/vmim_compress.py", "--cache", f"{cache_dir}/cache.npz",
                 "--out", out, "--summary-dim", str(dim), "--summary-noise", str(noise),
                 "--steps", str(steps), "--val-every", "500", "--max-minutes", str(mins),
                 "--gpu", GPU], timeout=int(mins * 60 + 120))
    return ok and os.path.exists(f"{out}/compressed.npz")


def npe(comp_dir, out, tag, seeds, epochs):
    ok, _ = run([PY, f"{REPO}/scripts/npe_on_summary.py", "--compressed", f"{comp_dir}/compressed.npz",
                 "--out", out, "--tag", tag, "--seeds", seeds, "--epochs", str(epochs),
                 "--gpu", GPU], timeout=1800)
    j = f"{out}/summary_{tag}.json"
    if ok and os.path.exists(j):
        return json.load(open(j))
    return None


def oracle_score(summ):
    """Lower is better: distance of null S8 from truth + penalty for collapsed (over-confident) σ."""
    s8, sig = summ["mean"][1], summ["std"][1]
    overconf = max(0.0, (WHITEN_SIG_S8 - sig)) * 30.0   # penalize σ << whitening (spurious tightness)
    return abs(s8 - TRUTH_S8) + overconf, s8, sig


def main():
    t0 = time.time()
    os.makedirs(ROOT, exist_ok=True)
    log("=== VMIM overnight: dump caches ===")
    caches = {t: dump_cache(t) for t in CONFIGS}
    if not caches["bnt_full"]:
        write_summary("FAIL", "could not dump bnt_full cache", {}, {})
        return

    # --- Stage A: sweep bottleneck knobs on bnt_full ---
    log("=== sweep on bnt_full (oracle: null on truth, σ not collapsed) ===")
    grid = [(n, d) for n in (0.3, 0.6, 1.0) for d in (6, 8)]
    sweep = []
    for i, (noise, dim) in enumerate(grid):
        cdir = f"{ROOT}/sweep/n{noise}_d{dim}"
        if not compress(caches["bnt_full"], cdir, noise, dim, steps=12000, mins=6):
            log(f"  sweep n{noise} d{dim}: compress FAIL"); continue
        summ = npe(cdir, cdir, "bnt_full", "41", 200)
        if not summ:
            log(f"  sweep n{noise} d{dim}: npe FAIL"); continue
        sc, s8, sig = oracle_score(summ)
        sweep.append({"noise": noise, "dim": dim, "score": sc, "S8": s8, "sig": sig,
                      "tarp": summ.get("tarp", {})})
        log(f"  n={noise} d={dim}: S8={s8:.4f} σ={sig:.4f} score={sc:.4f} "
            f"tarp={summ.get('tarp',{}).get('verdict','?')}")
        write_summary("RUNNING", "sweep in progress", {"sweep": sweep}, {})

    if not sweep:
        write_summary("FAIL", "no sweep config produced a posterior", {"sweep": sweep}, {})
        return
    best = min(sweep, key=lambda r: r["score"])
    log(f"BEST sweep: noise={best['noise']} dim={best['dim']} (S8={best['S8']:.4f} σ={best['sig']:.4f})")

    # --- Stage B: 4-config NULL comparison with the winning config, 3 NPE seeds ---
    log("=== 4-config comparison with best knobs (3 seeds) ===")
    results = {}
    for tag in CONFIGS:
        if not caches[tag]:
            continue
        cdir = f"{ROOT}/final/{tag}"
        if not compress(caches[tag], cdir, best["noise"], best["dim"], steps=16000, mins=8):
            log(f"  {tag}: compress FAIL"); continue
        summ = npe(cdir, cdir, tag, "41,42,43", 250)
        if summ:
            results[tag] = summ
            log(f"  {tag}: S8={summ['mean'][1]:.4f}±{summ['std'][1]:.4f} "
                f"Om={summ['mean'][0]:.4f}±{summ['std'][0]:.4f} tarp={summ.get('tarp',{}).get('verdict','?')}")
            write_summary("RUNNING", "final comparison in progress", {"sweep": sweep, "best": best}, results)

    verdict = assess(results)
    write_summary(verdict, f"done in {(time.time()-t0)/60:.0f} min", {"sweep": sweep, "best": best}, results)
    log(f"=== DONE: {verdict} ===")


def assess(R):
    if "bnt_full" not in R or "nonbnt_full" not in R:
        return "FAIL"
    bf, nf = R["bnt_full"], R["nonbnt_full"]
    on_truth = abs(bf["mean"][1] - TRUTH_S8) < 0.015
    oracle_ok = abs(bf["mean"][1] - nf["mean"][1]) < 0.01 and on_truth
    if oracle_ok and "bnt_580" in R and "nonbnt_460" in R:
        return "PASS"
    return "PARTIAL"


def write_summary(verdict, note, meta, R):
    os.makedirs(ROOT, exist_ok=True)
    L = [f"# VMIM neural compression — overnight result ({datetime.now():%Y-%m-%d %H:%M})",
         "", f"**Verdict: {verdict}** — {note}", "",
         "Reference for what 'good' is: whitening gave null S8=0.837±0.024 (on truth), and the payoff "
         "BNT-580/non-BNT-460 area ratio 0.79; Fisher says 0.37. Goal: a calibrated summary that beats 0.79.",
         ""]
    if meta.get("best"):
        b = meta["best"]
        L += [f"**Best sweep knobs:** summary_noise={b['noise']}, summary_dim={b['dim']} "
              f"(bnt_full null S8={b['S8']:.4f}, σ={b['sig']:.4f}).", ""]
    if meta.get("sweep"):
        L += ["## Sweep (bnt_full)", "", "| noise | dim | null S8 | σ(S8) | score | TARP |",
              "|---|---|---|---|---|---|"]
        for r in meta["sweep"]:
            L.append(f"| {r['noise']} | {r['dim']} | {r['S8']:.4f} | {r['sig']:.4f} | "
                     f"{r['score']:.4f} | {r.get('tarp',{}).get('verdict','?')} |")
        L.append("")
    if R:
        L += ["## 4-config NULL comparison (truth S8=0.84, Ωm=0.26)", "",
              "| config | S8 | σ(S8) | Ωm | σ(Ωm) | TARP |", "|---|---|---|---|---|---|"]
        for t in ("nonbnt_full", "bnt_full", "nonbnt_460", "bnt_580"):
            if t in R:
                s = R[t]
                L.append(f"| {t} | {s['mean'][1]:.4f} | {s['std'][1]:.4f} | {s['mean'][0]:.4f} | "
                         f"{s['std'][0]:.4f} | {s.get('tarp',{}).get('verdict','?')} |")
        L.append("")
        if "bnt_580" in R and "nonbnt_460" in R:
            ab = R["bnt_580"]["std"][0] * R["bnt_580"]["std"][1]
            an = R["nonbnt_460"]["std"][0] * R["nonbnt_460"]["std"][1]
            L += [f"**Payoff:** BNT-580/non-BNT-460 Ωm-S8 area ratio = **{ab/an:.3f}** "
                  f"(whitening 0.79; Fisher 0.37).", ""]
    L += ["## Caveat (recipe lessons 5-7)", "",
          "Tightness ≠ correctness. A tighter BNT contour that is off-truth or fails TARP is NOT a win. "
          "If the sweep could not find a calibrated (on-truth, non-collapsed, TARP-OK) summary, the next "
          "lever is the compressor deep-ensemble (recipe lesson 7) — train 2-3 compressor seeds and pool. "
          "VMIM artifacts under outputs/baryon_tension/vmim/.", "",
          "## Run log", "```"] + log_lines[-60:] + ["```"]
    open(f"{ROOT}/MORNING_SUMMARY.md", "w").write("\n".join(L))


if __name__ == "__main__":
    main()
