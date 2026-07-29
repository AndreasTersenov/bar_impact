"""Config-driven NPE sweep runner for the baryon-tension campaign.

For every (footprint, upper_cut, role, run) it invokes the patched worker
(run_npe_inference_auto_cross_ps_master.py) with a per-run seed, distributes jobs across
GPUs, retries on worker failure (NaN-loss exit 42, or any crash) with a BUMPED seed, runs
the per-posterior QA gate, records everything to qa_report.csv, and writes a manifest.
Resumable: a job whose output already exists is skipped.

Run with the jaxili interpreter (the worker needs jaxili; this orchestrator + QA need only
numpy). Per-run seed = seed_base + run; a retry adds RETRY_SEED_BUMP so it can't repeat a
deterministic NaN.
"""
import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np

from . import paths, qa
from .configs import PSCampaign

WORKER = paths.REPO / "scripts" / "run_npe_inference_auto_cross_ps_master.py"
HEALPY_WORKER = paths.REPO / "scripts" / "run_npe_inference_auto_cross_ps.py"  # full-sky pipeline
PY_JAXILI = "/home/tersenov/anaconda3/envs/jaxili/bin/python"
GRID_PARAMS = Path("/lustre/fsn1/projects/rech/prk/ulx34io/cosmogrid_products/stage3_forecast/grid/cosmo_params.npy")
TRUTH = [0.26, 0.84, -1.0, 67.36, 0.9649, 0.0493]

NAN_EXIT = 42          # worker exit code for a non-finite-loss abort
RETRY_SEED_BUMP = 1000  # added to the seed on each retry
POLL_SECONDS = 5


@dataclass
class Job:
    area: int
    upper_cut: int
    role: str          # "null" | "biased"
    run: int           # 1..n_runs (labels the output slot)
    seed: int          # current seed (base + run, + RETRY_SEED_BUMP per retry)
    attempt: int = 0   # 0 on first try
    out_path: str = ""
    samples_dir: str = ""
    log_path: str = ""


def plan_jobs(camp: PSCampaign, seed_base: int) -> List[Job]:
    """One Job per (area, upper_cut, role, run); existing outputs are skipped (resume)."""
    jobs = []
    for area, upper_cut in camp.coords():
        for role in paths.ROLES:
            for run in camp.runs:
                out = camp.posterior_path(role, area, upper_cut, run)
                if out.exists():
                    continue
                jobs.append(Job(
                    area=area, upper_cut=upper_cut, role=role, run=run,
                    seed=seed_base + run,
                    out_path=str(out),
                    samples_dir=str(paths.posteriors_dir(camp.tag, area, role)),
                    log_path=str(paths.logs_dir(camp.tag) /
                               f"{paths.area_dirname(area)}_l{camp.lmin}-{upper_cut}_{role}_run{run}.log"),
                ))
    return jobs


def job_checkpoint_dir(camp: PSCampaign, job: Job) -> Path:
    """Unique checkpoint dir per (area, cut, role, run).

    The worker's checkpoint NAME omits the run, so without this two runs of the same config
    running concurrently on different GPUs would clobber each other's checkpoint. Cleaned up
    after the job (the posterior is the only artifact we keep).
    """
    return (paths.REPO / "checkpoints" / "tension_sweep" / camp.tag /
            f"{paths.area_dirname(job.area)}_l{camp.lmin}-{job.upper_cut}_{job.role}_run{job.run}")


def build_worker_cmd(camp: PSCampaign, job: Job, gpu: int) -> List[str]:
    ckpt = str(job_checkpoint_dir(camp, job))
    # BNT campaigns drive a per-bin cut (only cut_bins get job.upper_cut); else a single cut.
    if getattr(camp, "bnt", False):
        cut_args = ["--upper-cuts", ",".join(str(c) for c in camp.per_bin_cuts(job.upper_cut))]
    else:
        cut_args = ["--upper-cut", str(job.upper_cut)]
    common_tail = [
        "--lower-cut", str(camp.lmin), *cut_args,
        "--rebin", str(camp.rebin),
        "--train", "--gpu", str(gpu), "--random-seed", str(job.seed),
        "--run", str(job.run),
        "--samples-dir", job.samples_dir,
        "--checkpoint-dir", ckpt,
    ]
    if camp.pipeline == "healpy":
        # Full-sky healpy pipeline: no mask, no submean, no NaMaster lmax (lmax defaults to 1024,
        # matching the full-sky grids). BNT is supported by the healpy worker; the per-bin
        # --upper-cuts already ride in common_tail via cut_args, so we only add the BNT flags.
        bnt_args = ["--bnt", "--bnt-bins", "0,1,2,3"] if getattr(camp, "bnt", False) else []
        return [
            PY_JAXILI, str(HEALPY_WORKER),
            "--simulation-type", "nobaryons",
            "--fiducial-type", paths.FID_BY_ROLE[job.role],
            "--noisy", "--noise-level", "0.26",
        ] + bnt_args + common_tail
    bnt_args = ["--bnt", "--bnt-bins", "0,1,2,3"] if getattr(camp, "bnt", False) else []
    return [
        PY_JAXILI, str(WORKER),
        "--simulation-type", "nobaryons",
        "--fiducial-type", paths.FID_BY_ROLE[job.role],
        "--masked", "--mask-area-sqdeg", str(float(job.area)),
        "--apodization-scale-deg", "2.0",
        "--noisy", "--noise-level", "0.26",
        "--subtract-mean", "--lmax", "1535",
    ] + bnt_args + common_tail


def _qa_record(job: Job, prior_lo, prior_hi) -> dict:
    """Run the per-posterior QA gate on a just-produced output."""
    samples = np.load(job.out_path)
    rec = qa.assess_posterior(samples, role=job.role, truth=TRUTH,
                              prior_lo=prior_lo, prior_hi=prior_hi)
    return {"area": job.area, "upper_cut": job.upper_cut, "role": job.role,
            "run": job.run, "seed": job.seed, "attempt": job.attempt,
            "status": rec["status"], "reasons": rec["reasons"],
            "mean_S8": rec["mean"][1], "std_S8": rec["std"][1]}


def run_sweep(
    camp: PSCampaign,
    gpus: List[int],
    seed_base: int = 100,
    max_retries: int = 3,
    jobs_per_gpu: int = 1,
    mem_fraction: Optional[float] = None,
) -> List[dict]:
    """Execute the sweep across `gpus`, with `jobs_per_gpu` concurrent jobs on each card.

    Each NPE job is light on the GPU (~30% util) with a long CPU-bound init/data-load phase,
    so packing several per card overlaps those phases for a big throughput gain. `mem_fraction`
    caps JAX's per-process GPU preallocation (XLA_PYTHON_CLIENT_MEM_FRACTION) so the packed
    jobs fit — without it JAX grabs ~75% of the card each.
    """
    paths.ensure_campaign_tree(camp.tag, camp.areas)
    grid = np.load(GRID_PARAMS)
    prior_lo, prior_hi = grid.min(0), grid.max(0)

    queue = plan_jobs(camp, seed_base)
    n_planned = len(queue)
    # One scheduling slot per concurrent job; slot -> physical GPU.
    slot_gpu = [g for g in gpus for _ in range(jobs_per_gpu)]
    env = dict(os.environ)
    if mem_fraction is not None:
        env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(mem_fraction)
    print(f"[sweep] campaign={camp.tag}  planned={n_planned} jobs "
          f"(after skipping existing)  gpus={gpus}  jobs_per_gpu={jobs_per_gpu}  "
          f"slots={len(slot_gpu)}  mem_fraction={mem_fraction}")

    running = {}   # slot_idx -> (proc, job, logfile_handle)
    records, failures = [], []

    def launch(job: Job, slot_idx: int):
        gpu = slot_gpu[slot_idx]
        Path(job.samples_dir).mkdir(parents=True, exist_ok=True)
        Path(job.log_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(job_checkpoint_dir(camp, job), ignore_errors=True)  # fresh per attempt
        fh = open(job.log_path, "a")
        fh.write(f"\n===== attempt {job.attempt} seed {job.seed} gpu {gpu} slot {slot_idx} "
                 f"{datetime.now(timezone.utc).isoformat()} =====\n")
        fh.flush()
        proc = subprocess.Popen(build_worker_cmd(camp, job, gpu),
                                stdout=fh, stderr=subprocess.STDOUT, cwd=str(paths.REPO), env=env)
        running[slot_idx] = (proc, job, fh)

    done = 0
    while queue or running:
        # fill free slots (jobs_per_gpu per card)
        for slot_idx in range(len(slot_gpu)):
            if slot_idx not in running and queue:
                launch(queue.pop(0), slot_idx)
        # poll
        for slot_idx, (proc, job, fh) in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            fh.close()
            del running[slot_idx]
            done += 1
            ok = (rc == 0 and Path(job.out_path).exists())
            if ok:
                rec = _qa_record(job, prior_lo, prior_hi)
                records.append(rec)
                shutil.rmtree(job_checkpoint_dir(camp, job), ignore_errors=True)
                tag = rec["status"] + (f"({rec['reasons']})" if rec["reasons"] else "")
                print(f"[{done}/{n_planned}] OK   mask{job.area} l{camp.lmin}-{job.upper_cut} "
                      f"{job.role} run{job.run} -> QA {tag}")
            elif job.attempt < max_retries:
                job.attempt += 1
                job.seed += RETRY_SEED_BUMP
                reason = "NaN-loss" if rc == NAN_EXIT else f"exit {rc}"
                print(f"[{done}/{n_planned}] RETRY mask{job.area} l{camp.lmin}-{job.upper_cut} "
                      f"{job.role} run{job.run} ({reason} -> seed {job.seed})")
                queue.append(job)
            else:
                shutil.rmtree(job_checkpoint_dir(camp, job), ignore_errors=True)
                print(f"[{done}/{n_planned}] FAIL  mask{job.area} l{camp.lmin}-{job.upper_cut} "
                      f"{job.role} run{job.run} (exhausted {max_retries} retries)")
                failures.append({"area": job.area, "upper_cut": job.upper_cut,
                                 "role": job.role, "run": job.run, "last_exit": rc})
        time.sleep(POLL_SECONDS)

    _write_outputs(camp, records, failures, gpus, seed_base, max_retries, n_planned)
    return records


def _write_outputs(camp, records, failures, gpus, seed_base, max_retries, n_planned):
    import pandas as pd

    # Remove the bulky, regenerable NPE-input byproducts (~33 MB each) the worker drops next
    # to each posterior; keep posteriors + the tiny example_* records. Safe: runs after all
    # jobs finish.
    removed = 0
    for p in (paths.campaign_dir(camp.tag) / "posteriors").rglob("datavectors_npe_input_*.npy"):
        try:
            p.unlink()
            removed += 1
        except OSError:
            pass
    if removed:
        print(f"[sweep] cleaned {removed} datavectors_npe_input byproducts")

    qa_dir = paths.qa_dir(camp.tag)
    qa_dir.mkdir(parents=True, exist_ok=True)
    if records:
        pd.DataFrame(records).to_csv(qa_dir / "qa_report.csv", index=False, float_format="%.5f")
    if failures:
        pd.DataFrame(failures).to_csv(qa_dir / "failures.csv", index=False)
    flagged = sum(1 for r in records if r["status"] != "OK")
    manifest = {
        "campaign": camp.tag, "lmin": camp.lmin, "submean": camp.submean,
        "areas": list(camp.areas), "upper_cuts": list(camp.upper_cuts),
        "runs": list(camp.runs), "seed_base": seed_base, "max_retries": max_retries,
        "gpus": gpus, "n_planned_this_run": n_planned,
        "n_completed": len(records), "n_qa_flagged": flagged, "n_failed": len(failures),
        "worker": str(WORKER), "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
    }
    paths.manifest_path(camp.tag).write_text(json.dumps(manifest, indent=2))
    print(f"\n[sweep] done: {len(records)} completed ({flagged} QA-flagged), "
          f"{len(failures)} failed. QA -> {qa_dir}/qa_report.csv  manifest -> {paths.manifest_path(camp.tag)}")


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=str(paths.REPO), text=True).strip()
    except Exception:
        return "unknown"
