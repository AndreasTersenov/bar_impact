#!/usr/bin/env python3
"""Is the wavelet scale cut actually baryon-safe? Measure it, do not assume it.

The power-spectrum cut is measured: lmax=460 at 14000 deg^2 gives a 3-param Q_DM bias of
0.288 sigma, and the next step-40 cut (500) gives 0.413. The higher-order cut -- "drop the
finest wavelet scale" -- was carried as a recollection, never verified against the corrected
(submean) posteriors. If it does not clear 0.3 sigma, captioning the figure "baryon-safe"
would be wrong for two of its three curves.

Compares, per statistic, at one footprint:
    scales1234   all four detail scales (no cut)            -> expect FAIL
    scales234    finest dropped (the assumed safe cut)      -> the thing under test
    scales2345   finest dropped, coarse/mass-sheet ADDED    -> the variant asked for
Runs are used only as matched null/biased PAIRS, with the sigma(S8) collapse guard.

  /lustre/fswork/projects/rech/nzu/ulx34io/envs/aname/bin/python \
      scripts/diagnostics/check_hos_cut_is_safe.py [--area 14001]
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "scripts"))
from tension.estimators import tension_sigma, SUBSET_INDICES  # noqa: E402

SAMP = f"{REPO}/outputs/samples"
SIG_MAX = 0.08
THRESHOLD = 0.3
SCALESETS = ["scales1234", "scales234", "scales2345"]


def matched_pairs(prefix, scales, area):
    def index(role):
        out = {}
        pat = (f"{SAMP}/posterior_samples_{prefix}nobaryons_vs_{role}_bins1234_{scales}"
               f"_noisy_s0.26_masked_{area}sqdeg_submean_new_normalization*_npe.npy")
        for f in sorted(glob.glob(pat)):
            m = re.search(r"_run(\d+)", f)
            try:
                out[int(m.group(1)) if m else 1] = np.load(f)
            except Exception:
                pass          # damaged .npy raises ValueError, not an IOError
        return out
    n, b = index("nobaryons"), index("baryonified")
    return {r: (n[r], b[r]) for r in sorted(set(n) & set(b))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--area", type=int, default=14001, help="HOS footprint tag (AREA+1)")
    args = ap.parse_args()

    print(f"3-param Q_DM baryon bias, submean, footprint tag {args.area}")
    print(f"tolerance {THRESHOLD} sigma   (PS at lmax=460 measures 0.288 for reference)\n")
    result = {}
    for label, prefix in (("peaks", "pc_"), ("L1", "")):
        for scales in SCALESETS:
            pairs = matched_pairs(prefix, scales, args.area)
            ns = []
            for r, (a, bb) in pairs.items():
                if max(np.sqrt(np.cov(a[:, :3], rowvar=False)[1, 1]),
                       np.sqrt(np.cov(bb[:, :3], rowvar=False)[1, 1])) >= SIG_MAX:
                    continue
                ns.append(float(tension_sigma(a, bb, indices=SUBSET_INDICES)["nsigma"]))
            key = f"{label} {scales}"
            if not ns:
                print(f"  {key:22s} NO MATCHED PAIRS")
                result[key] = None
                continue
            m, s = float(np.mean(ns)), float(np.std(ns))
            verdict = "PASS" if m < THRESHOLD else "EXCEEDS TOLERANCE"
            print(f"  {key:22s} n={len(ns)}  nsigma = {m:.3f} +/- {s:.3f}   {verdict}")
            print(f"  {'':22s} per-seed: {', '.join(f'{x:.3f}' for x in sorted(ns))}")
            result[key] = {"n_pairs": len(ns), "mean": m, "std": s,
                           "per_seed": sorted(ns), "passes": bool(m < THRESHOLD)}
    out = f"{REPO}/outputs/diagnostics/hos_cut_safety_{args.area}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"footprint_tag": args.area, "threshold_sigma": THRESHOLD,
               "estimator": "tensiometer Q_DM -> chi2.cdf -> from_confidence_to_sigma",
               "param_subset": list(SUBSET_INDICES),
               "collapse_guard_sigma_S8_max": SIG_MAX,
               "ps_reference": {"lmax_460": 0.288, "lmax_500": 0.413},
               "results": result}, open(out, "w"), indent=2)
    print(f"\nwrote {os.path.relpath(out, REPO)}")


if __name__ == "__main__":
    main()
