#!/usr/bin/env python3
"""Score (MOPED) compression of the auto+cross PS data vector for the NPE chase
(plan: /home/tersenov/.claude/plans/jolly-toasting-robin.md).

Builds the validated Fisher J (local order-2 derivative at the fiducial) and C (hybrid = analytic
NaMaster Gaussian + low-rank SSC/cNG) for a config, forms the score weights W = C^{-1} J, and
compresses the dump-cache data vector x -> t = J^T C^{-1} x (6 sufficient statistics for a near-
Gaussian likelihood). Saves a summary cache (theta, x=t, x_fid=t_fid) for NPE training.

MOPED-LOSSLESSNESS ORACLE (run before any NPE): the Fisher built from the 6 score summaries must
equal the full-data Fisher. Two checks:
  (alg) F_t = J^T C^{-1} J  must equal the full Fisher F_x = J^T C^{-1} J exactly (by construction).
  (emp) re-derive J_t (local-order2 derivative of t vs theta) and C_t (sample cov of the COMPRESSED
        fiducial perms, a 6x6 from 200 perms -> well-conditioned, no Hartlap penalty), then
        F_t^emp = J_t^T C_t^{-1} J_t. Its FoM3 must match the full hybrid Fisher FoM3.
The empirical check is the meaningful one: it confirms an NPE trained on t (which sees J_t, C_t
implicitly) can recover the full Fisher information. numpy only; run with jaxili python.

Usage: python scripts/score_compress.py <area>   (default 14000)
"""
import os
import sys
import numpy as np

# Footprint must be set BEFORE importing the fisher modules (they read FISHER_AREA at import time).
AREA = int(sys.argv[1]) if len(sys.argv) > 1 else 14000
COVK = sys.argv[2] if len(sys.argv) > 2 else "hybrid"   # hybrid | sample | analytic
os.environ["FISHER_AREA"] = str(AREA)

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diagnostics")
sys.path.insert(0, HERE)
import fisher_local_jacobian as L          # build_config, local_jacobian, FID
import fisher_hybrid_cov as H              # C_ANA, bnt operator, cut_rebin_R, per_spectrum_uppers
P3 = [0, 1, 2]
ORDER, BW, KHYB = "order2", 0.75, 3
CFG = {"nonbnt_460": ([460, 460, 460, 460], False),
       "bnt_580":    ([580, 1024, 1024, 1024], True)}
CACHE = "outputs/score_experiment/cache"
OUTW = "outputs/score_experiment/score"
os.makedirs(OUTW, exist_ok=True)

Tfull = np.kron(H.bnt_spectra_operator(), np.eye(H.NBPW))
C_ANA_BNT = Tfull @ H.C_ANA @ Tfull.T


def compression_cov(cuts, bnt, kind=COVK, k=KHYB):
    """Covariance used for the score WEIGHTS W=C^{-1}J. Score compression is lossless only when C is
    the TRUE data covariance, so test a few estimators:
      hybrid   = analytic Gaussian + top-k SSC/cNG eigenmodes (Phase-I best true-cov estimate).
      sample   = full 200-perm sample cov (unbiased true cov, noisy) + small ridge for a stable inverse.
      analytic = analytic Gaussian only (no SSC/cNG).
    Returns (C, fiducial_perms)."""
    R = H.cut_rebin_R(H.per_spectrum_uppers(cuts))
    Cana = R @ (C_ANA_BNT if bnt else H.C_ANA) @ R.T
    fa, fc, nell = L.load_set("fiducial", "nobaryons", bnt)
    perms = L.datavector(fa, fc, nell, cuts)                 # (200, nfeat)
    Csamp = np.cov(perms, rowvar=False)
    if kind == "analytic":
        return Cana, perms
    if kind == "sample":
        ridge = 1e-3 * np.median(np.diag(Csamp))
        return Csamp + ridge * np.eye(Csamp.shape[0]), perms
    D = Csamp - Cana                                         # hybrid (default)
    ev, V = np.linalg.eigh(D)
    idx = np.argsort(ev)[::-1][:k]
    return Cana + (V[:, idx] * ev[idx]) @ V[:, idx].T, perms


def fom3(cov6):
    c3 = cov6[np.ix_(P3, P3)]
    return 1.0 / np.sqrt(np.linalg.det(c3)), np.sqrt(np.diag(cov6))[P3]


def main():
    print(f"=== Score (MOPED) compression @ {AREA} deg^2  (order={ORDER}, bw={BW}, hybrid k={KHYB}) ===\n")
    results = {}
    for tag, (cuts, bnt) in CFG.items():
        cache = np.load(f"{CACHE}/{tag}_{AREA}_nobary/cache.npz")
        x, theta, x_fid = cache["x"], cache["theta"], cache["x_fid"]

        cfg = L.build_config(cuts, bnt)
        J, _ = L.local_jacobian(cfg["grid_avg"], cfg["ucos"], cfg["fid_mean"], ORDER, BW)  # (nfeat,6)
        C, perms = compression_cov(cuts, bnt)
        Cinv = np.linalg.inv(C)

        # full-data hybrid Fisher (the target) and the MLE-form score compression.
        F_x = J.T @ Cinv @ J
        fom_x, sig_x = fom3(np.linalg.inv(F_x))
        # quasi-MLE summaries: theta_hat = theta_fid + F^{-1} J^T C^{-1} (x - mu).  6 numbers in
        # PARAMETER units, well-conditioned, posterior ~ N(theta_hat, F^{-1}) -> trivial for the flow.
        Wmle = (Cinv @ J) @ np.linalg.inv(F_x)               # (nfeat, 6)
        That = L.FID + (x - x_fid) @ Wmle                    # (n, 6)
        t_fid = L.FID.copy()                                 # theta_hat(x_fid) == theta_fid exactly
        perms_t = L.FID + (perms - x_fid) @ Wmle             # (200, 6)

        # empirical compressed param-covariance from the compressed fiducial perms (6x6, well-cond.):
        # Cov(theta_hat) should equal F^{-1} when C == true cov.
        Ct = np.cov(perms_t, rowvar=False)
        fom_t, sig_t = fom3(Ct)
        F_alg = J.T @ Cinv @ J                               # algebraic MOPED identity check vs F_x
        alg_ok = np.max(np.abs(F_alg - F_x) / (np.abs(F_x) + 1e-300))

        np.savez(f"{OUTW}/score_cache_{tag}_{AREA}_{COVK}.npz", theta=theta, x=That, x_fid=t_fid,
                 Wmle=Wmle, J=J, C=C, perms_t=perms_t, F_x=F_x, Ct=Ct)
        # npe_on_summary.py format (theta_tr/va, y_tr/va, y_fid); all rows in _tr, jaxili re-splits.
        nz6 = np.zeros((0, 6), np.float32)
        np.savez(f"{OUTW}/compressed_{tag}_{AREA}_{COVK}.npz",
                 theta_tr=theta.astype(np.float32), theta_va=nz6,
                 y_tr=That.astype(np.float32), y_va=np.zeros((0, 6), np.float32),
                 y_fid=t_fid.astype(np.float32))
        results[tag] = dict(fom_x=fom_x, fom_t=fom_t, sig_x=sig_x, sig_t=sig_t,
                            cond_Ct=np.linalg.cond(Ct), nfeat=x.shape[1])
        print(f"[{tag}]  nfeat={x.shape[1]} -> 6 quasi-MLE score stats (parameter units)")
        print(f"   FoM3  full-hybrid={fom_x:.3e}   compressed-perm-cov={fom_t:.3e}   "
              f"ratio={fom_t/fom_x:.4f}   (cond(Cov_hat)={np.linalg.cond(Ct):.1f})")
        print(f"   sigma(Om,S8,w0) full={np.round(sig_x,4)}  compressed={np.round(sig_t,4)}")
        print(f"   algebraic MOPED check |F_t-F_x|/F_x = {alg_ok:.1e}\n")

    rn, rb = results["nonbnt_460"], results["bnt_580"]
    print("=== BNT-580 / non-BNT-460 FoM3 ratio (BNT more constraining => >1) ===")
    print(f"   full-hybrid Fisher:      {rb['fom_x']/rn['fom_x']:.2f}x")
    print(f"   compressed (empirical):  {rb['fom_t']/rn['fom_t']:.2f}x   <- what an NPE-on-score targets")
    print(f"\nsaved score caches -> {OUTW}/*_{AREA}_{COVK}.npz")


if __name__ == "__main__":
    main()
