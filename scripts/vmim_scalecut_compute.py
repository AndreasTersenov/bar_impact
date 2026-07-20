#!/usr/bin/env python3
"""Compute 3-param Q_DM tension (Om,S8,w0) per (config, cut, seed) for the VMIM scale-cut sweep,
aggregate to mean+/-std over seeds, write a CSV. Run with the `aname` env (tensiometer).
"""
import argparse
import csv
import glob
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")
from tension.estimators import make_mcsamples, q_dm_tension  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="outputs/baryon_tension/vmim_v2/scalecuts")
    p.add_argument("--configs", default="nonbnt,bnt")
    p.add_argument("--cuts", default=",".join(str(c) for c in range(340, 1021, 40)))
    p.add_argument("--out", default="outputs/baryon_tension/vmim_v2/scalecuts/tension_agg.csv")
    return p.parse_args()


def main():
    a = parse_args()
    cuts = [int(c) for c in a.cuts.split(",")]
    rows = []
    for cfg in a.configs.split(","):
        for c in cuts:
            # all compressor-seed x NDE-seed pairs: {cfg}_c{c}/cs*/nde/null_s*_{cfg}.npy
            nulls = sorted(glob.glob(str(Path(a.root) / f"{cfg}_c{c}" / "cs*" / "nde" / f"null_s*_{cfg}.npy")))
            sigs = []
            for nf in nulls:
                bf = nf.replace(f"null_s", "biased_s")              # same dir, biased_s{seed}_{cfg}.npy
                if not os.path.exists(bf):
                    continue
                null = make_mcsamples(np.load(nf), indices=[0, 1, 2], label="null")
                bias = make_mcsamples(np.load(bf), indices=[0, 1, 2], label="biased")
                r = q_dm_tension(null, bias)
                if r["ok"] and np.isfinite(r["nsigma"]):
                    sigs.append(r["nsigma"])
            if sigs:
                rows.append((cfg, c, float(np.mean(sigs)), float(np.std(sigs)), len(sigs)))
                print(f"{cfg:7s} c{c}: nsigma = {np.mean(sigs):.3f} +/- {np.std(sigs):.3f} (n={len(sigs)})")
            else:
                print(f"{cfg:7s} c{c}: NO data")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["config", "upper_cut", "nsigma_mean", "nsigma_std", "n_seeds"])
        w.writerows(rows)
    print(f"[compute] wrote {a.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
