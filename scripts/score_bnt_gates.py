#!/usr/bin/env python3
"""Pre-training gates for the score-compressed BNT bin-1 tension sweep (numpy only, jaxili env).

GATE 1 — slice oracle: slicing the full BNT cache to cut 580 must reproduce the existing
  validated bnt_580 cache (96 feat) bit-for-bit. Confirms the column-selection logic.
GATE 2 — lossless identity: at the full vector the BNT data vector is an invertible linear map of
  the non-BNT one (A = T⊗I), and the MLE-form score summary theta_hat = FID + (x-x_fid)@W is
  invariant under x->Ax. So the per-sim grid summaries and the biased-fiducial summary must be
  identical whether built in the BNT or the non-BNT basis. This is the "at no cut, BNT≡non-BNT"
  identity proven at the summary level — no NPE needed.

Usage: FISHER_AREA is set internally. python scripts/score_bnt_gates.py
"""
import os
import sys

import numpy as np

os.environ.setdefault("FISHER_AREA", "14000")
AREA = int(os.environ["FISHER_AREA"])
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_cut_utils as S   # noqa: E402

CACHE = "outputs/score_experiment/cache"


def load_full(tag):
    z = np.load(f"{CACHE}/{tag}/cache.npz")
    return z["theta"], z["x"], z["x_fid"]


P3 = [0, 1, 2]


def sigma3(F):
    cov = np.linalg.inv(F)
    return np.sqrt(np.diag(cov))[P3]


def summaries(cuts, bnt, x, x_fid_null, x_fid_bias, covk):
    """theta_hat for the grid (x) and the biased fiducial, built in the given basis at `cuts`."""
    keep = S.keep_indices(cuts)
    sc = S.build_score(cuts, bnt, covk=covk)
    W = sc["Wmle"]
    That = S.FID + (x[:, keep] - x_fid_null[keep]) @ W
    t_bias = S.FID + (x_fid_bias[keep] - x_fid_null[keep]) @ W
    return That, t_bias, sc


def gate1():
    print("=== GATE 1: slice oracle (full -> 580 == bnt_580 cache) ===")
    _, x_full, xf_full = load_full(f"bnt_full_{AREA}_nobary")
    ref = np.load(f"{CACHE}/bnt_580_{AREA}_nobary/cache.npz")
    keep = S.keep_indices([580, 1024, 1024, 1024])
    dx = np.max(np.abs(x_full[:, keep] - ref["x"]))
    df = np.max(np.abs(xf_full[keep] - ref["x_fid"]))
    print(f"   kept {keep.size} cols (ref has {ref['x'].shape[1]})")
    print(f"   max|x_slice - x_ref|     = {dx:.2e}")
    print(f"   max|xfid_slice - xfid_ref| = {df:.2e}")
    ok = keep.size == ref["x"].shape[1] and dx < 1e-9 and df < 1e-9
    print(f"   -> {'PASS' if ok else 'FAIL'}\n")
    return ok


def gate2():
    print("=== GATE 2: lossless identity (BNT-full summary == non-BNT-full summary) ===")
    _, xb, xfb = load_full(f"bnt_full_{AREA}_nobary")
    _, xb_, xfb_bias = load_full(f"bnt_full_{AREA}_bary")
    _, xn, xfn = load_full(f"nonbnt_full_{AREA}_nobary")
    _, xn_, xfn_bias = load_full(f"nonbnt_full_{AREA}_bary")
    assert np.array_equal(xb, xb_) and np.array_equal(xn, xn_), "grid x differs between null/biased dumps"

    # --- 2a: analytic C (transforms EXACTLY as A C A^T) -> exact identity. The hard gate. ---
    print("  [2a] analytic C  (exact-transforming -> must be machine-precision identical)")
    Tb, tb, scb = summaries(S.FULL_CUTS, True, xb, xfb, xfb_bias, "analytic")
    Tn, tn, scn = summaries(S.FULL_CUTS, False, xn, xfn, xfn_bias, "analytic")
    dF = np.max(np.abs(sigma3(scb["F"]) - sigma3(scn["F"])) / sigma3(scn["F"]))
    dgrid = np.max(np.abs(Tb - Tn))
    dbias = np.max(np.abs(tb - tn))
    print(f"      sigma3 bnt={np.round(sigma3(scb['F']),5)}  nonbnt={np.round(sigma3(scn['F']),5)}")
    print(f"      max rel d sigma3 = {dF:.2e}   max|grid dtheta| = {dgrid:.2e} ({Tb.shape[0]} sims)"
          f"   max|biased dtheta| = {dbias:.2e}")
    ok2a = dF < 1e-6 and dgrid < 1e-5 and dbias < 1e-5
    print(f"      -> {'PASS' if ok2a else 'FAIL'}")

    # --- 2b: hybrid C (the production estimator). Eigen-truncation is basis-dependent, so a small
    #         nonzero deviation is EXPECTED, not a bug. Report and bound it (informational). ---
    print("  [2b] hybrid C   (basis-dependent SSC truncation -> small nonzero deviation expected)")
    Hb, hb, shb = summaries(S.FULL_CUTS, True, xb, xfb, xfb_bias, "hybrid")
    Hn, hn, shn = summaries(S.FULL_CUTS, False, xn, xfn, xfn_bias, "hybrid")
    s3b, s3n = sigma3(shb["F"]), sigma3(shn["F"])
    rel_s = np.max(np.abs(s3b - s3n) / s3n)
    rel_bias = np.max(np.abs(hb[:3] - hn[:3]) / s3n)   # biased-fid shift difference in units of sigma
    print(f"      sigma3 bnt={np.round(s3b,5)}  nonbnt={np.round(s3n,5)}   max rel d sigma3 = {rel_s:.1%}")
    print(f"      biased theta_hat bnt={np.round(hb[:3],5)} nonbnt={np.round(hn[:3],5)}")
    print(f"      |biased shift diff| in units of sigma3: {np.round(np.abs(hb[:3]-hn[:3])/s3n,3)}  (max {rel_bias:.1%})")
    print(f"      (informational: BNT-basis vs non-BNT-basis hybrid disagree by <~ this; both valid)\n")

    return ok2a


if __name__ == "__main__":
    r1 = gate1()
    r2 = gate2()
    print(f"GATES: gate1={'PASS' if r1 else 'FAIL'}  gate2={'PASS' if r2 else 'FAIL'}")
    sys.exit(0 if (r1 and r2) else 1)
