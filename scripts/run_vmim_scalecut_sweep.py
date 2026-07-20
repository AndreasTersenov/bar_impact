#!/usr/bin/env python3
"""Orchestrate the VMIM compressed-PS scale-cut sweep across GPUs 0,1,2,3 (packed).

Fans out (cut, config) jobs to a pool of GPU slots via subprocess.Popen, each running
scripts/vmim_cut_worker.sh (retrain compressor on the cut + NDE -> per-seed null+biased).
No scheduler — local concurrent processes, the repo's submit_*_parallel pattern.
"""
import argparse
import subprocess
import time
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cuts", default=",".join(str(c) for c in range(340, 1021, 40)))
    p.add_argument("--configs", default="nonbnt,bnt")
    p.add_argument("--comp-seeds", default="41,42,43", help="compressor seeds (ensemble members)")
    p.add_argument("--seeds", default="41,42,43", help="NDE seeds per compressor")
    p.add_argument("--gpus", default="0,2")
    p.add_argument("--pack", type=int, default=8, help="jobs per GPU")
    p.add_argument("--stagger", type=float, default=2.0, help="seconds between job launches")
    p.add_argument("--out-root", default="outputs/baryon_tension/vmim_v2/scalecuts")
    return p.parse_args()


def main():
    a = parse_args()
    cuts = [int(c) for c in a.cuts.split(",")]
    configs = a.configs.split(",")
    cseeds = [s.strip() for s in a.comp_seeds.split(",")]
    gpus = [g.strip() for g in a.gpus.split(",")]
    Path(a.out_root).mkdir(parents=True, exist_ok=True)

    jobs = [(c, cfg, cs) for c in cuts for cfg in configs for cs in cseeds]
    slots = [g for g in gpus for _ in range(a.pack)]          # one entry per concurrent slot
    print(f"[sweep] {len(jobs)} jobs over {len(slots)} slots ({len(gpus)} GPUs x {a.pack})", flush=True)

    running = {}                                              # slot_idx -> (Popen, job)
    todo = list(jobs)
    done = 0
    while todo or running:
        # fill free slots
        for si in range(len(slots)):
            if si not in running and todo:
                cut, cfg, cs = todo.pop(0)
                cmd = ["bash", "scripts/vmim_cut_worker.sh", str(cut), cfg, cs, slots[si],
                       a.out_root, a.seeds]
                running[si] = (subprocess.Popen(cmd), (cut, cfg, cs))
                time.sleep(a.stagger)             # stagger JAX/CUDA inits to avoid races
        # reap finished
        time.sleep(3)
        for si in list(running):
            proc, job = running[si]
            if proc.poll() is not None:
                done += 1
                ok = proc.returncode == 0
                print(f"[sweep] {'OK ' if ok else 'FAIL'} {job[1]} c{job[0]} cs{job[2]} "
                      f"({done}/{len(jobs)}) slot{si}gpu{slots[si]}", flush=True)
                del running[si]
    print("[sweep] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
