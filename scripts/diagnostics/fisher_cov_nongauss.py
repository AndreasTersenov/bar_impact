#!/usr/bin/env python3
"""Phase I close-out: is the sim-vs-Gaussian covariance excess REAL non-Gaussian covariance
(cNG+SSC, which the survey also has -> trust the sim cov) or a sim-suite artifact (correlated/
duplicated perms -> distrust the sim cov)?

Two empirical tests, no provenance needed:
  (T1) PERM INDEPENDENCE: pairwise correlation between the 200 perms' (whitened) data vectors.
       Independent realizations -> off-diagonal perm-perm correlations centered at 0 with spread
       ~1/sqrt(nfeat). Duplicated/dependent perms -> spikes toward 1. (A perm artifact that BIASED
       the covariance would require gross dependence here.)
  (T2) STRUCTURE of D = C_sample - C_analytic (rebinned, full config): leading-eigenvalue fraction
       and whether the leading eigenvector is COHERENT (all bandpowers same sign = SSC-like, real
       super-survey covariance) vs oscillatory. Plus the implied sigma inflation sqrt of the
       variance ratio (what adopting the Gaussian-only cov would cost the Fisher).

Conclusion logic: T1 ~ independent AND D positive-definite/coherent => the excess is real
non-Gaussian covariance; the production Fisher covariance must include it (sim cov + Sellentin-
Heavens), and a pure analytic Gaussian would be over-optimistic by the reported factor.

numpy only.
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from fisher_bnt_vs_nonbnt import load_set, datavector  # noqa: E402
import fisher_gaussian_cov as G  # noqa: E402

AREA = 14000
C = np.load(os.path.join(HERE, "cache_gaussian_cov", f"gaussian_cov_native_{AREA}.npy"))


def main():
    fa, fc, nbpw = load_set("fiducial", "nobaryons", bnt=False)
    cuts = [1024, 1024, 1024, 1024]
    dv = datavector(fa, fc, nbpw, cuts)                          # (200, 120) rebinned+cut
    nperm, nfeat = dv.shape

    # ---- T1: perm independence ----
    # whiten each feature (z-score across perms) so the perm-perm correlation isn't dominated by a
    # few high-variance features, then correlate perms.
    z = (dv - dv.mean(0)) / (dv.std(0, ddof=1) + 1e-300)
    Rp = np.corrcoef(z)                                          # (200,200) perm-perm
    iu = np.triu_indices(nperm, k=1)
    off = Rp[iu]
    print("=== T1: perm-perm correlation (independent => centered 0, spread ~1/sqrt(nfeat)) ===")
    print(f"  nfeat={nfeat}  expected spread 1/sqrt(nfeat)={1/np.sqrt(nfeat):.3f}")
    print(f"  perm-perm corr: mean {off.mean():+.4f}  std {off.std():.4f}  "
          f"max {off.max():.3f}  min {off.min():.3f}")
    print(f"  frac|corr|>0.3 {np.mean(np.abs(off)>0.3):.4f} (>0 would flag dependent/duplicated perms)")

    # ---- T2: structure of the excess D = C_sample - C_analytic (rebinned) ----
    R = G.build_full_R(nbpw, upper=1024)
    C_ana = R @ C @ R.T
    C_samp = np.cov(dv, rowvar=False)
    D = C_samp - C_ana
    print("\n=== T2: excess covariance D = C_sample - C_analytic (rebinned, full config) ===")
    print(f"  median diag ratio  C_ana/C_samp = {np.median(np.diag(C_ana)/np.diag(C_samp)):.3f}")
    print(f"  => adopting Gaussian-only cov shrinks sigma by ~sqrt = "
          f"{np.sqrt(np.median(np.diag(C_ana)/np.diag(C_samp))):.3f} (too optimistic if <1)")
    # is D positive (extra variance) and coherent?
    dD = np.diag(D)
    print(f"  diag(D): frac positive {np.mean(dD>0):.2f}  median(D_ii/C_samp_ii) {np.median(dD/np.diag(C_samp)):.3f}")
    # eigen-decomposition of the correlation-form excess (normalize by sample diag to compare modes)
    s = 1.0 / np.sqrt(np.diag(C_samp))
    Dn = (D * s[:, None]) * s[None, :]
    evals, evecs = np.linalg.eigh(Dn)
    order = np.argsort(evals)[::-1]
    evals = evals[order]; evecs = evecs[:, order]
    lead = evecs[:, 0]
    print(f"  leading eigenvalue fraction of |D_norm| trace: {evals[0]/np.sum(np.abs(evals)):.2f}")
    print(f"  leading eigvec coherence (|mean sign|): {abs(np.mean(np.sign(lead))):.2f} "
          f"(~1 => all bandpowers move together = SSC-like real super-survey covariance)")
    print(f"  top-3 normalized eigenvalues: {np.round(evals[:3],3)}")

    print("\nRead: T1 independent + T2 positive/coherent => the excess is REAL non-Gaussian (cNG+SSC)")
    print("covariance the survey also has. Production Fisher cov = SIM cov (+Sellentin-Heavens),")
    print("validated by the analytic Gaussian on the diagonal. Pure analytic Gaussian is too tight.")


if __name__ == "__main__":
    main()
