#!/usr/bin/env python3
"""Convert a RAW score cache into the Stage-2 driver's compressed.npz format, with NO VMIM compression
(the summary y = the z-scored raw data vector). This lets nde_realnvp_from_summary.py produce the
"raw NPE / without compression" baseline contour using the SAME sbi_lens RealNVP NDE as the compressed
pipeline — so the only difference vs the VMIM arms is the compression itself.

Same preprocessing as vmim_compress.py (H0/100, by-cosmology split with --split-seed, per-feature
z-score + clip), just identity instead of the MLP. numpy only.
"""
import argparse
import os
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--clip-value", type=float, default=5.0)
    return p.parse_args()


def split_by_cosmology(theta, val_frac, seed):
    keys = np.round(theta, 6)
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(uniq))
    n_val = max(1, int(val_frac * len(uniq)))
    val_cosmo = set(perm[:n_val].tolist())
    is_val = np.fromiter((c in val_cosmo for c in inv), dtype=bool, count=len(inv))
    return np.where(~is_val)[0], np.where(is_val)[0]


def main():
    a = parse_args()
    z = np.load(a.cache)
    theta = z["theta"].astype(np.float64).copy()
    theta[:, 3] /= 100.0
    theta = theta.astype(np.float32)
    X = z["x"].astype(np.float64)
    x_fid = z["x_fid"].astype(np.float64)

    tr, va = split_by_cosmology(theta, a.val_frac, a.split_seed)
    mean, std = X[tr].mean(0), X[tr].std(0)
    std[std < 1e-12] = 1.0
    zc = lambda A: np.clip((np.atleast_2d(A) - mean) / std, -a.clip_value, a.clip_value).astype(np.float32)

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "compressed.npz",
             theta_tr=theta[tr], y_tr=zc(X[tr]), theta_va=theta[va], y_va=zc(X[va]),
             y_fid=zc(x_fid)[0], summary_dim=X.shape[1], preproc_kind="raw_zscore", in_dim=X.shape[1])
    leak = len({tuple(r) for r in np.round(theta[tr], 6)} & {tuple(r) for r in np.round(theta[va], 6)})
    print(f"[prep-raw] {a.cache} -> {out}/compressed.npz  dim={X.shape[1]} "
          f"train={tr.size} val={va.size} leakage={leak}")


if __name__ == "__main__":
    main()
