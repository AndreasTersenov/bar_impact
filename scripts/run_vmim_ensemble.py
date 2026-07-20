#!/usr/bin/env python3
"""Laser-focused VMIM shot: compressor deep-ensemble (recipe lesson 7) at the on-truth noise level.

Last night's sweep showed the VMIM null climbs 0.76->0.80->0.855 as summary_noise goes 0.3->0.6->1.0
(=> calibrated mean near noise~0.9, dim=6) but stays OVER-CONFIDENT. The registered fix is the
compressor deep-ensemble: train K compressors with different seeds and POOL their posteriors per
observation, which diversifies the summary and washes out single-compressor over-confidence.

This trains K=3 compressors (noise=0.9, dim=6) per config, runs Stage-2 NPE on each, pools the
posteriors, and gates on SBC (rank-std ~0.289) + the oracle (bnt_full ensemble null on truth, ~=
nonbnt_full, sigma ~ whitening 0.024). Reuses the caches already dumped under outputs/.../vmim/cache.

Run with the jaxili interpreter (nohup). Writes outputs/baryon_tension/vmim/ENSEMBLE_RESULT.md.
"""
import json
import os
import subprocess
import time
from datetime import datetime

import numpy as np

REPO = "/home/tersenov/software/bar_impact"
PY = "/home/tersenov/anaconda3/envs/jaxili/bin/python"
ROOT = f"{REPO}/outputs/baryon_tension/vmim"
GPU = os.environ.get("VMIM_GPU", "0")
TRUTH = [0.26, 0.84, -1.0, 67.36, 0.9649, 0.0493]
WHITEN_SIG_S8 = 0.024
NOISE, DIM, KSEEDS = 0.9, 6, [41, 42, 43]
CONFIGS = ["nonbnt_full", "bnt_full", "nonbnt_460", "bnt_580"]
log_lines = []


def log(m):
    line = f"{datetime.now():%H:%M:%S} {m}"; print(line, flush=True); log_lines.append(line)


def run(cmd, timeout):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode == 0, p.stdout + p.stderr
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"


def compress(cache, out, seed):
    ok, _ = run([PY, f"{REPO}/scripts/vmim_compress.py", "--cache", cache, "--out", out,
                 "--summary-dim", str(DIM), "--summary-noise", str(NOISE), "--steps", "14000",
                 "--val-every", "500", "--max-minutes", "6", "--seed", str(seed), "--gpu", GPU],
                timeout=600)
    return ok and os.path.exists(f"{out}/compressed.npz")


def npe(comp_dir, out, tag):
    ok, _ = run([PY, f"{REPO}/scripts/npe_on_summary.py", "--compressed", f"{comp_dir}/compressed.npz",
                 "--out", out, "--tag", tag, "--seeds", "41", "--epochs", "250",
                 "--tarp-points", "150", "--gpu", GPU], timeout=1200)
    f = f"{out}/posterior_summary_{tag}.npy"
    j = f"{out}/summary_{tag}.json"
    if ok and os.path.exists(f):
        return np.load(f), (json.load(open(j)) if os.path.exists(j) else {})
    return None, {}


def main():
    t0 = time.time()
    results = {}
    for cfg in CONFIGS:
        cache = f"{ROOT}/cache/{cfg}/cache.npz"
        if not os.path.exists(cache):
            log(f"{cfg}: cache MISSING — skip"); continue
        pooled, sbcs = [], []
        for sd in KSEEDS:
            cdir = f"{ROOT}/ensemble/{cfg}/c{sd}"
            if not compress(cache, cdir, sd):
                log(f"  {cfg} c{sd}: compress FAIL"); continue
            samp, summ = npe(cdir, cdir, f"{cfg}_c{sd}")
            if samp is None:
                log(f"  {cfg} c{sd}: npe FAIL"); continue
            pooled.append(samp)
            sbcs.append(summ.get("sbc", {}).get("verdict", "?"))
            log(f"  {cfg} c{sd}: S8={samp[:,1].mean():.4f}±{samp[:,1].std():.4f} sbc={sbcs[-1]}")
        if not pooled:
            log(f"{cfg}: no compressors succeeded"); continue
        ens = np.concatenate(pooled)
        np.save(f"{ROOT}/ensemble/pooled_{cfg}.npy", ens)
        results[cfg] = {"mean": ens.mean(0).tolist(), "std": ens.std(0).tolist(),
                        "k": len(pooled), "sbc": sbcs}
        log(f"{cfg} ENSEMBLE (k={len(pooled)}): S8={ens[:,1].mean():.4f}±{ens[:,1].std():.4f} "
            f"Om={ens[:,0].mean():.4f}±{ens[:,0].std():.4f}")
        write(results, t0)
    write(results, t0, final=True)
    log("DONE")


def write(R, t0, final=False):
    L = [f"# VMIM deep-ensemble — laser shot ({datetime.now():%Y-%m-%d %H:%M})", "",
         f"K={len(KSEEDS)} compressors, noise={NOISE}, dim={DIM}, pooled posteriors. "
         f"{'FINAL' if final else 'running'} ({(time.time()-t0)/60:.0f} min).", "",
         "Bar to clear: whitening null S8=0.837±0.024 (on truth), payoff BNT-580/non-BNT-460 area 0.79; Fisher 0.37.",
         "", "## Ensemble NULL posteriors (truth S8=0.84, Ωm=0.26)", "",
         "| config | S8 | σ(S8) | Ωm | σ(Ωm) | k | SBC |", "|---|---|---|---|---|---|---|"]
    for t in CONFIGS:
        if t in R:
            r = R[t]
            L.append(f"| {t} | {r['mean'][1]:.4f} | {r['std'][1]:.4f} | {r['mean'][0]:.4f} | "
                     f"{r['std'][0]:.4f} | {r['k']} | {','.join(r['sbc'])} |")
    L.append("")
    if "bnt_full" in R and "nonbnt_full" in R:
        bf, nf = R["bnt_full"], R["nonbnt_full"]
        on_truth = abs(bf["mean"][1] - 0.84) < 0.015
        agree = abs(bf["mean"][1] - nf["mean"][1]) < 0.01
        notcollapsed = bf["std"][1] > 0.018
        L += [f"**Oracle:** bnt_full null S8={bf['mean'][1]:.4f} (on-truth<0.015: {on_truth}), "
              f"agrees with nonbnt_full ({agree}), σ not collapsed (>0.018: {notcollapsed}) -> "
              f"{'PASS' if (on_truth and agree and notcollapsed) else 'FAIL'}", ""]
    if "bnt_580" in R and "nonbnt_460" in R:
        b, c = R["bnt_580"], R["nonbnt_460"]
        ratio = (b["std"][0] * b["std"][1]) / (c["std"][0] * c["std"][1])
        L += [f"**Payoff:** BNT-580/non-BNT-460 area ratio = **{ratio:.3f}** (whitening 0.79; Fisher 0.37).", ""]
    L += ["## Log", "```"] + log_lines[-50:] + ["```"]
    open(f"{ROOT}/ENSEMBLE_RESULT.md", "w").write("\n".join(L))


if __name__ == "__main__":
    main()
