#!/usr/bin/env python3
"""MOPED-compress the PS data vector at an arbitrary ell-rebinning, for NPE training.

THE QUESTION THIS EXISTS TO ANSWER. The production score->NPE result compresses the rebin=20 vector.
That rebinning was chosen for a reason that no longer applies — fisher_bnt_vs_nonbnt.py records
"20 keeps n_feat<200 for the SAMPLE/hybrid cov", a covariance-ESTIMATION constraint, not a
compression one. Pre-averaging 20 native bandpowers with a flat kernel before an optimal linear
compression can only discard information. The Fisher scan
(outputs/diagnostics/fisher_fom_vs_rebin.csv) measures that cost at 2.1-2.6x in FoM3 going to
native. This script lets us check it end-to-end with a trained posterior instead of a Fisher bound.

WHY IT IS CHEAP. MOPED emits 6 numbers in parameter units no matter how many features go in, so the
NPE's job is IDENTICAL at rebin 20 and at native. Input dimension affects only the numpy step here.
That is the opposite of feeding raw vectors to a flow, where 2026 ill-conditioned features is the
regime that broke raw NPE on BNT in the first place.

WHY A WRONG COVARIANCE IS SURVIVABLE HERE. MOPED is lossless only when C is the true covariance, and
at fine rebinning only the analytic Gaussian is available (the hybrid's low-rank SSC/cNG term comes
from 200 permutations and is noise past a few hundred features). But the NPE learns p(theta | t)
from simulations, so an imperfect C makes t less than sufficient — costing information — WITHOUT
biasing the posterior. The resulting contour is honest and directly measurable, TARP/SBC included.
That is a strictly better epistemic position than the Fisher scan, which assumes C is right.

NOTE: no compressor training happens here, so the compressor/NDE data-splitting question that
applies to VMIM does not arise. MOPED is analytic; all rows can train the NDE without leakage.

Output: <out>/compressed.npz with theta_tr,y_tr,theta_va,y_va,y_fid,y_fid_biased — the format
scripts/nde_realnvp_from_summary.py consumes.

  FISHER_AREA=14000 FISHER_REBIN=5 python scripts/score_compress_at_rebin.py \
      --cuts 460,460,460,460 --out outputs/score_rebin_ladder/nonbnt_r5_c460
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

H0_IDX = 3          # theta = [Om, S8, w0, H0, ns, Ob]; the NDE wants H0/100 so all params are O(1)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cuts", required=True, help="4 comma ints, e.g. 460,460,460,460")
    p.add_argument("--bnt", action="store_true")
    p.add_argument("--covk", default="analytic", choices=["analytic", "hybrid", "sample"],
                   help="analytic is the only one valid at fine rebin; hybrid only for r20/r10")
    p.add_argument("--mode", default="moped", choices=["moped", "raw"],
                   help="moped = 6 quasi-MLE score summaries. raw = the z-scored data vector itself, "
                        "so the flow must find the JtC^-1 projection on its own. The raw arm is the "
                        "CONTROL for the whole compression argument: the claim is that a flow "
                        "under-learns that projection on the ill-conditioned BNT vector and returns "
                        "a too-wide, off-truth posterior. Both arms must be run at the SAME cut for "
                        "the comparison to mean anything.")
    p.add_argument("--out", required=True)
    p.add_argument("--val-frac", type=float, default=0.1,
                   help="fraction of COSMOLOGIES held out for TARP/SBC (unseen by the NDE)")
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--verify-against-cache", default=None,
                   help="path to an existing cache.npz; assert this rebuild reproduces its x/x_fid "
                        "(use at rebin=20 to prove the rebuild path is exact before trusting others)")
    return p.parse_args()


def split_by_cosmology(theta, val_frac, seed):
    """Hold out whole cosmologies. Only TARP/SBC uses the val side, but a by-cosmology split keeps
    those points genuinely unseen rather than sharing realizations with the training set."""
    keys = np.round(theta, 8)
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(uniq))
    n_val = max(1, int(val_frac * len(uniq)))
    val_cos = set(perm[:n_val].tolist())
    is_val = np.array([i in val_cos for i in inv])
    return ~is_val, is_val, len(uniq), n_val


def main():
    a = parse_args()
    cuts = [int(c) for c in a.cuts.split(",")]
    os.makedirs(a.out, exist_ok=True)

    import score_cut_utils as S
    import fisher_local_jacobian as L
    import fisher_hybrid_cov as H

    rebin = H.REBIN
    print(f"[compress] AREA={H.AREA} REBIN={rebin} cuts={cuts} bnt={a.bnt} covk={a.covk}", flush=True)

    # ---- data vectors at THIS rebin, straight from the intact sims (no cache dependency) ----
    ga, gc, nell = L.load_set("new_grid", "nobaryons", a.bnt)
    x = L.datavector(ga, gc, nell, cuts)                                   # (16965, nfeat)
    theta = L.PARAMS.copy()
    fa, fc, _ = L.load_set("fiducial", "nobaryons", a.bnt)
    perms_null = L.datavector(fa, fc, nell, cuts)                          # (200, nfeat)
    x_fid = perms_null.mean(0)
    ba, bc, _ = L.load_set("fiducial", "baryonified", a.bnt)
    x_fid_biased = L.datavector(ba, bc, nell, cuts).mean(0)
    nfeat = x.shape[1]
    print(f"[compress] x={x.shape} theta={theta.shape} nfeat={nfeat}", flush=True)

    ok = np.isfinite(x).all(1) & np.isfinite(theta).all(1)
    if not ok.all():
        print(f"[compress] dropping {int((~ok).sum())} non-finite grid rows", flush=True)
        x, theta = x[ok], theta[ok]

    # ---- gate: at rebin=20 this rebuild must reproduce the existing cache bit-for-bit ----
    if a.verify_against_cache:
        z = np.load(a.verify_against_cache)
        dx = np.max(np.abs(x[: z["x"].shape[0]] - z["x"])) if x.shape[1] == z["x"].shape[1] else np.inf
        df = np.max(np.abs(x_fid - z["x_fid"])) if x_fid.size == z["x_fid"].size else np.inf
        print(f"[gate] vs {a.verify_against_cache}: max|dx|={dx:.3e} max|dx_fid|={df:.3e}", flush=True)
        if not (dx < 1e-9 and df < 1e-9):
            print("[gate] FAIL — the rebuild does not reproduce the reference cache", flush=True)
            sys.exit(1)
        print("[gate] PASS", flush=True)

    # ---- MOPED weights at this cut/rebin ----
    d = S.build_score(cuts, bnt=a.bnt, covk=a.covk)
    W, F = d["Wmle"], d["F"]
    if d["nfeat"] != nfeat:
        print(f"[fatal] score nfeat {d['nfeat']} != data nfeat {nfeat}", flush=True)
        sys.exit(1)
    cond_C = float(np.linalg.cond(d["C"]))
    print(f"[compress] C cond={cond_C:.3e}  Fisher sigma3={np.round(d['sigma3'], 5)}  "
          f"FoM3={d['fom3']:.4e}", flush=True)

    theta = theta.copy(); theta[:, H0_IDX] /= 100.0

    if a.mode == "moped":
        # theta_hat = FID + (x - x_fid) @ Wmle, in PARAMETER units
        y = S.FID + (x - x_fid) @ W                                        # (n, 6)
        y_fid = S.FID.copy()                                               # exact by construction
        y_fid_biased = S.FID + (x_fid_biased - x_fid) @ W
        # the summary IS an estimate of theta, so it takes the same H0/100 rescaling
        y = y.copy(); y[:, H0_IDX] /= 100.0
        y_fid = y_fid.copy(); y_fid[H0_IDX] /= 100.0
        y_fid_biased = y_fid_biased.copy(); y_fid_biased[H0_IDX] /= 100.0
    else:
        # RAW control: hand the flow the data vector itself, z-scored on the TRAIN statistics only.
        # No projection is supplied, so the flow must learn JtC^-1 from the simulations — which is
        # precisely what it fails to do on the ill-conditioned BNT vector.
        mu, sd = x.mean(0), x.std(0)
        sd = np.where(sd > 0, sd, 1.0)
        y = (x - mu) / sd
        y_fid = (x_fid - mu) / sd
        y_fid_biased = (x_fid_biased - mu) / sd
        print(f"[compress] RAW mode: y is the z-scored {nfeat}-dim data vector "
              f"(no compression); |y| range [{np.abs(y).min():.2e}, {np.abs(y).max():.2e}]", flush=True)

    tr, va, n_cos, n_val_cos = split_by_cosmology(theta, a.val_frac, a.split_seed)
    print(f"[compress] split by cosmology: {n_cos} cosmologies -> {n_val_cos} val "
          f"(train rows {int(tr.sum())}, val rows {int(va.sum())})", flush=True)

    shift = y_fid_biased - y_fid
    print(f"[compress] biased-fiducial shift (Om,S8,w0) = {np.round(shift[:3], 5)}", flush=True)
    print(f"[compress] summary spread  (Om,S8,w0) = {np.round(y[:, :3].std(0), 5)}", flush=True)

    np.savez(os.path.join(a.out, "compressed.npz"),
             theta_tr=theta[tr].astype(np.float32), y_tr=y[tr].astype(np.float32),
             theta_va=theta[va].astype(np.float32), y_va=y[va].astype(np.float32),
             y_fid=y_fid.astype(np.float32), y_fid_biased=y_fid_biased.astype(np.float32),
             Wmle=W, F=F, x_fid=x_fid, x_fid_biased=x_fid_biased)

    qa = {"area": int(H.AREA), "rebin": int(rebin), "cuts": cuts, "bnt": bool(a.bnt),
          "mode": a.mode, "covk": a.covk, "nfeat": int(nfeat),
          "summary_dim": int(y.shape[1]), "cond_C": cond_C,
          "fisher_sigma3": [float(v) for v in d["sigma3"]], "fisher_fom3": float(d["fom3"]),
          "n_cosmologies": int(n_cos), "n_val_cosmologies": int(n_val_cos),
          "n_train_rows": int(tr.sum()), "n_val_rows": int(va.sum()),
          "biased_shift_3param": [float(v) for v in shift[:3]],
          "note": ("MOPED is analytic, so there is no compressor training and no compressor/NDE "
                   "leakage concern; the val split exists only to give TARP/SBC unseen cosmologies.")}
    with open(os.path.join(a.out, "compress_qa.json"), "w") as fh:
        json.dump(qa, fh, indent=2)
    print(f"[compress] mode={a.mode} wrote {a.out}/compressed.npz  y_tr={y[tr].shape} "
          f"y_fid={y_fid.shape}", flush=True)


if __name__ == "__main__":
    main()
