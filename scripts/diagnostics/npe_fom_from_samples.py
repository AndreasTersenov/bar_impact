#!/usr/bin/env python3
"""Phase-B comparison: 3-parameter FoM (Om,S8,w0) from NPE posterior samples, for each compression
method, vs the Fisher targets. Pools seed files when given a glob. Reuses the FoM/area primitives
(FoM3 = 1/sqrt(det C3); area = pi*sqrt(det 2x2)) consistent with the Fisher side.

Usage: python scripts/diagnostics/npe_fom_from_samples.py
"""
import glob
import os
import numpy as np

P3 = [0, 1, 2]
NM = ["Om", "S8", "w0"]
FID = np.array([0.26, 0.84, -1.0])
SCORE = "outputs/score_experiment/npe_score"
WHIT = "outputs/score_experiment/npe_whiten"

# Fisher reference (hybrid k3, local order-2) from fisher_contours_fom.py
FISHER = {"nonbnt_460": dict(fom3=9.684e4, sig=[0.0228, 0.0406, 0.1135]),
          "bnt_580":    dict(fom3=2.410e5, sig=[0.0153, 0.0276, 0.0999])}


def pool(paths):
    arrs = [np.load(p) for p in paths]
    arrs = [a for a in arrs if a.ndim == 2 and a.shape[1] >= 3]
    return np.concatenate(arrs, 0) if arrs else None


def fom3(s):
    c = np.cov(s[:, P3], rowvar=False)
    return 1.0 / np.sqrt(np.linalg.det(c)), np.sqrt(np.diag(c)), s[:, P3].mean(0)


def report(name, files_non, files_bnt):
    sn, sb = pool(files_non), pool(files_bnt)
    if sn is None or sb is None:
        print(f"{name:22s}  (missing: non={sn is not None} bnt={sb is not None})")
        return None
    fn, sign, mn = fom3(sn)
    fb, sigb, mb = fom3(sb)
    print(f"{name:22s}  nonBNT FoM3={fn:.3e} sig(S8)={sign[1]:.4f} | "
          f"BNT FoM3={fb:.3e} sig(S8)={sigb[1]:.4f} | FoM3 ratio={fb/fn:.2f}x  "
          f"sigOm*sigS8 ratio={(sigb[0]*sigb[1])/(sign[0]*sign[1]):.3f}")
    return fn, fb


def main():
    print("=== Phase-B: 3-param FoM (Om,S8,w0), 14000, rebin=20/l37 ===\n")
    print(f"{'FISHER hybrid k3':22s}  nonBNT FoM3={FISHER['nonbnt_460']['fom3']:.3e} "
          f"sig(S8)={FISHER['nonbnt_460']['sig'][1]:.4f} | "
          f"BNT FoM3={FISHER['bnt_580']['fom3']:.3e} sig(S8)={FISHER['bnt_580']['sig'][1]:.4f} | "
          f"FoM3 ratio={FISHER['bnt_580']['fom3']/FISHER['nonbnt_460']['fom3']:.2f}x")
    print("-" * 130)
    report("NPE score-hybrid", [f"{SCORE}/posterior_summary_nonbnt_460_14000_mle.npy"],
           [f"{SCORE}/posterior_summary_bnt_580_14000_mle.npy"])
    report("NPE score-sample", [f"{SCORE}/posterior_summary_nonbnt_460_14000_sample.npy"],
           [f"{SCORE}/posterior_summary_bnt_580_14000_sample.npy"])
    # whitening worker outputs: bnt_ vs non-bnt distinguished by 'bnt_ps' token; pool seed runs
    wn = sorted(glob.glob(f"{WHIT}/posterior_samples_ps_*nobaryons_vs_nobaryons*run*.npy"))
    wb = sorted(glob.glob(f"{WHIT}/posterior_samples_bnt_ps_*nobaryons_vs_nobaryons*run*.npy"))
    report("NPE whitening (full)", wn, wb)
    print("\nReference: raw-score(ill-cond)=failed | whitening pilot product-ratio 0.79")


if __name__ == "__main__":
    main()
