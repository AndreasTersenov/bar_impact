#!/usr/bin/env python3
"""
OVERNIGHT: redo the masked-PS SUBMEAN production at the CORRECT binning nlb=4 (processing lmax=1535),
for all masks, both sim types, grid + fiducial. This replaces the broken June 16-17 submean production
that was run at nlb=1 (lmax=1024, 1023-col files) -- nlb=1 decoupling is numerically singular at low
f_sky (cond ~1e9-1e17 for masks <=14000; see outputs/diagnostics/ps_old_vs_new/cond_vs_nlb.png), which
broke the 2000-14000 masks. nlb=4 regularizes every mask (worst 2000 -> cond 23).

Output files carry the `_submean ... _lmax1535` tags (383 bandpowers), distinct from both the raw nlb=4
files and the broken nlb=1 submean files (1023 cols, no lmax tag) -- nothing good is overwritten.
Per-map outputs are skipped if present (no --force-overwrite), so 5000 (already done by the gate) and
the dry-run's 2000 fiducial are reused.

Concurrency is capped (shared 128-core machine). Fiducials first (fast, warm the per-mask MCM cache),
then grids in priority order (14000 paper footprint first, then the broken small masks, large last).
Progress -> outputs/diagnostics/reprocess_nlb4/STATUS.log ; per-job logs in .../logs/.
"""
import subprocess, time, os
from datetime import datetime

CST = "/home/tersenov/software/cosmostat_new/cosmostat/cosmostat_new/bin/python"
SCRIPT = "scripts/cross_power_spectrum_processing_master.py"
BASE = "/home/tersenov/CosmoGridV1/stage3_forecast"
OUTROOT = "outputs/diagnostics/reprocess_nlb4"
LOGDIR = f"{OUTROOT}/logs"
STATUS = f"{OUTROOT}/STATUS.log"
os.makedirs(LOGDIR, exist_ok=True)

MASKS = [14000, 5000, 2000, 10000, 28000, 35000]   # priority: paper footprint, then small (broken), large
SIMTYPES = ["nobaryons", "baryonified"]   # fiducial needs BOTH (baryon fid = the bias-test observation)
WORKERS = 50            # per job, parallel over maps (AMD EPYC 128c, shared)
MAX_CONCURRENT = 1      # one job at a time => ~50 cores total, polite on the shared node

def build_cmd(mask, sim, dataset):
    out = f"{BASE}/fiducial" if dataset == "fiducial" else f"{BASE}/new_grid"
    cmd = [CST, SCRIPT, "--apply-mask", "--mask-area-sqdeg", str(mask),
           "--subtract-mean", "--lmax", "1535", "--num-workers", str(WORKERS),
           "--aggregate-for-inference", "--inference-output-dir", out]
    if dataset == "fiducial":
        cmd.append("--fiducial")
    if sim == "baryonified":
        cmd.append("--baryonified")
    return cmd

# fiducials first (fast + warm MCM cache per mask), then grids.
# GRID is nobaryons-only -- the analysis never used a baryonified grid; baryonified is only the
# fiducial (the contaminated "observation" for the bias test).
JOBS = ([("fiducial", m, s) for m in MASKS for s in SIMTYPES] +
        [("grid", m, "nobaryons") for m in MASKS])

def log(msg):
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
    print(line, flush=True)
    with open(STATUS, "a") as f:
        f.write(line + "\n")

log(f"===== START nlb=4 submean reprocess: {len(JOBS)} jobs, {MAX_CONCURRENT} concurrent, "
    f"{WORKERS} workers/job =====")
running, queue = {}, list(JOBS)

def launch(job):
    dataset, mask, sim = job
    lf = open(f"{LOGDIR}/{dataset}_{mask}_{sim}.log", "w")
    p = subprocess.Popen(build_cmd(mask, sim, dataset), stdout=lf, stderr=subprocess.STDOUT)
    running[p] = (job, lf)
    log(f"LAUNCH {dataset:8s} {mask:6d} {sim:11s} pid={p.pid}")

while queue or running:
    while queue and len(running) < MAX_CONCURRENT:
        launch(queue.pop(0))
    time.sleep(20)
    for p in list(running):
        if p.poll() is not None:
            job, lf = running.pop(p)
            lf.close()
            log(f"DONE   {job[0]:8s} {job[1]:6d} {job[2]:11s} exit={p.returncode}")

log("===== ALL DONE =====")
