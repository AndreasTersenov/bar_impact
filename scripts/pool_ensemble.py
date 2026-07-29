#!/usr/bin/env python3
"""Pool Stage-2 outputs across compressor-ensemble members into one bundle the gate consumes.

The compressor deep-ensemble (NEURAL_SUMMARIZATION_RECIPE lesson 7): train K compressor seeds, run
Stage-2 on each, and pool their per-observation posteriors. This concatenates K driver output dirs'
null posteriors and TARP bundles into a single dir with the same filenames vmim_gate.py expects.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--in-dirs", required=True, help="comma-separated driver output dirs")
    p.add_argument("--tag", required=True)
    p.add_argument("--out", required=True)
    return p.parse_args()


def main():
    a = parse_args()
    dirs = [Path(d) for d in a.in_dirs.split(",") if d.strip()]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    nulls = [np.load(d / f"null_pooled_{a.tag}.npy") for d in dirs]
    null = np.concatenate(nulls, 0)
    np.save(out / f"null_pooled_{a.tag}.npy", null)

    # biased posteriors (baryonified fiducial), if the members produced them
    bfiles = [d / f"null_biased_pooled_{a.tag}.npy" for d in dirs]
    if all(f.exists() for f in bfiles):
        biased = np.concatenate([np.load(f) for f in bfiles], 0)
        np.save(out / f"null_biased_pooled_{a.tag}.npy", biased)
        bm = biased.mean(0)
        print(f"[pool] {a.tag} BIASED Om={bm[0]:.3f} S8={bm[1]:.3f} w0={bm[2]:.3f}")

    # TARP: tarp_samples is (n_draws, n_points, 6) per member; the val points (theta) are identical
    # across members (same ridx seed) -> concatenate draws.
    ts = [np.load(d / f"tarp_samples_{a.tag}.npy") for d in dirs]
    np.save(out / f"tarp_samples_{a.tag}.npy", np.concatenate(ts, 0))
    np.save(out / f"tarp_theta_{a.tag}.npy", np.load(dirs[0] / f"tarp_theta_{a.tag}.npy"))

    summ = {"tag": a.tag, "n_members": len(dirs), "n_null": int(null.shape[0]),
            "null_mean": [float(v) for v in null.mean(0)],
            "null_std": [float(v) for v in null.std(0)],
            "member_dirs": [str(d) for d in dirs]}
    (out / f"summary_{a.tag}.json").write_text(json.dumps(summ, indent=2))
    m, s = null.mean(0), null.std(0)
    print(f"[pool] {a.tag} {len(dirs)} members -> null Om={m[0]:.3f}±{s[0]:.3f} "
          f"S8={m[1]:.3f}±{s[1]:.3f} w0={m[2]:.3f}±{s[2]:.3f}")


if __name__ == "__main__":
    main()
