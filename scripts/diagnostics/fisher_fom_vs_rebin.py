#!/usr/bin/env python3
"""How much information does the ell-rebinning actually cost?

The production MOPED sweep compresses the REBIN=20 data vector. That choice was made for a reason
that no longer applies: fisher_bnt_vs_nonbnt.py records "20 keeps n_feat<200 for the SAMPLE/hybrid
cov" — a covariance-ESTIMATION constraint, not a compression one. With an analytic covariance the
feature count is unbounded, and MOPED is meant to be lossless, so pre-averaging 20 native bandpowers
into one with a flat kernel can only discard information (or tie).

There is evidence it is not free: the score results log reports rebin=10 giving "~10% tighter sigma"
than rebin=20 at the same cut, even though the TENSION was unchanged. Binning-independence was
established for the tension, not for constraining power — and the FoM is what the BNT comparison
turns on.

This measures the Fisher FoM3 (network-independent, no NPE training) as a function of FISHER_REBIN
at fixed cuts, for both arms. If FoM keeps climbing toward native, rebinning is costing real
information and a native production run is justified. If it plateaus by r10/r5, the coarse binning
is nearly free and is worth keeping as a hedge against covariance mis-specification.

FISHER_REBIN is read at IMPORT time by the fisher_* modules, so each rebin value needs a fresh
process. This script runs ONE rebin (from the env) and appends a CSV row; drive it with the loop in
scripts/jz/fisher_fom_vs_rebin.slurm.

  FISHER_AREA=14000 FISHER_REBIN=10 python scripts/diagnostics/fisher_fom_vs_rebin.py --out out.csv

--covk defaults to 'analytic' because the hybrid CANNOT follow to native resolution: its low-rank
term is estimated from 200 permutations, so past a few hundred features it is fitting noise. Use
--covk hybrid only for the r20/r10 legs, where it is meaningful, to check the two agree.
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))          # scripts/

CFGS = [("nonBNT", [460] * 4, False), ("nonBNT", [580] * 4, False),
        ("BNT", [460, 1024, 1024, 1024], True), ("BNT", [580, 1024, 1024, 1024], True)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--covk", default="analytic", choices=["analytic", "hybrid"])
    ap.add_argument("--ridge-rel", type=float, default=0.0,
                    help="relative ridge on the analytic C (see diag_analytic_cov_rank.py)")
    ap.add_argument("--order", default="order2")
    ap.add_argument("--bw", type=float, default=0.75)
    a = ap.parse_args()

    import score_cut_utils as S
    import fisher_hybrid_cov as H
    rebin = H.REBIN
    print(f"[rebin={rebin}] covk={a.covk} ridge_rel={a.ridge_rel} order={a.order} bw={a.bw}",
          flush=True)

    new = not os.path.exists(a.out)
    with open(a.out, "a") as fh:
        if new:
            fh.write("rebin,covk,ridge_rel,order,bw,arm,ell_max,n_features,"
                     "sigma_Om,sigma_S8,sigma_w0,fom3,cond_C,seconds\n")
        for arm, cuts, bnt in CFGS:
            t0 = time.time()
            try:
                if a.ridge_rel > 0:
                    # rebuild with a ridge: monkeypatch compression_cov's output via build_score's
                    # covk path is not enough, so do it explicitly here.
                    import fisher_local_jacobian as L
                    Rm = H.cut_rebin_R(H.per_spectrum_uppers(list(cuts)))
                    base = S._build_C_ANA_BNT() if bnt else H.C_ANA
                    C = Rm @ base @ Rm.T
                    C = C + a.ridge_rel * np.median(np.diag(C)) * np.eye(C.shape[0])
                    cfg = L.build_config(list(cuts), bnt)
                    J, _ = L.local_jacobian(cfg["grid_avg"], cfg["ucos"], cfg["fid_mean"],
                                            a.order, a.bw)
                    Cinv = np.linalg.inv(C)
                    F = J.T @ Cinv @ J
                    Finv = np.linalg.inv(F)
                    sig3 = np.sqrt(np.diag(Finv))[:3]
                    fom = 1.0 / np.sqrt(np.linalg.det(Finv[np.ix_([0, 1, 2], [0, 1, 2])]))
                    nfeat, condC = J.shape[0], np.linalg.cond(C)
                else:
                    d = S.build_score(list(cuts), bnt=bnt, covk=a.covk, order=a.order, bw=a.bw)
                    sig3, fom, nfeat = d["sigma3"], d["fom3"], d["nfeat"]
                    condC = np.linalg.cond(d["C"])
                dt = time.time() - t0
                fh.write(f"{rebin},{a.covk},{a.ridge_rel},{a.order},{a.bw},{arm},{cuts[0]},{nfeat},"
                         f"{sig3[0]:.6f},{sig3[1]:.6f},{sig3[2]:.6f},{fom:.6e},{condC:.6e},{dt:.1f}\n")
                fh.flush()
                print(f"  {arm}@{cuts[0]:4d} nfeat={nfeat:5d} FoM3={fom:.4e} "
                      f"sigma3={np.round(sig3,4)} cond={condC:.2e} ({dt:.0f}s)", flush=True)
            except Exception as e:
                fh.write(f"{rebin},{a.covk},{a.ridge_rel},{a.order},{a.bw},{arm},{cuts[0]},,,,,,"
                         f",FAIL:{type(e).__name__}\n")
                fh.flush()
                print(f"  {arm}@{cuts[0]:4d} FAIL {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
