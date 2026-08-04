# CROSS-CHECK: BNT vs non-BNT at matched lmax=460 (MOPED, pooled)

MOPED cross-check of the embedding-network flagship (position 6). Same cut, same rebinning, same 5-seed protocol, but the flow is fed 6 score/MOPED summaries instead of the whitened data vector. Gives 1.47x against the embedding's 1.41x. The two methods share no assumptions - MOPED requires Gaussianity, the analytic covariance and the local Jacobian; the embedding requires none - so their agreement to 4 percent is a genuine corroboration rather than a repeat. Both recover the physical Omega_m-S8 degeneracy (r = -0.909/-0.931 here, -0.919/-0.938 for the embedding). The embedding contours are tighter in both arms; that extra tightness sits 1.25-1.36x above the Gaussian Fisher bound and is accompanied by a w0 offset that scales with it, but it is COMMON MODE across the two arms (w0 offset -0.0435 BNT vs -0.0436 non-BNT) and cancels in the ratio. Bias caveat as in the flagship: BNT sits marginally at the 0.3 sigma threshold at this cut, tolerated on its error bar; its own adopted cut is 420.

- **source**: `plots/bnt_flagship_matched_c460_14000_pooled`
- **generator commit**: `unknown`
- **generated**: 2026-08-04T16:20:58Z
- **published**: 2026-08-04T17:00:46+00:00 at repo `06e07006`
- **rows in values.csv**: 2

## Scales included

- **scales_included**: both arms at ell_max=460 (matched); BNT cuts bin-1 only, bins 2-4 to 1024

## Known gaps

- provenance git_commit is unknown — the figure cannot be traced to the code that made it

## Caveats (from provenance)

- Both series are calibrated and on-truth (SBC rank-std 0.28-0.29 vs the ideal 0.289; null means within 0.3 sigma of truth), so the FoM difference is a real difference in information, not one posterior being broken.
- The decisive quantity is r(Omega_m, S8). Weak lensing carries a physical degeneracy near -0.9; raw NPE on the ill-conditioned BNT vector returns -0.03, i.e. it loses the degeneracy structure entirely while keeping plausible (even tighter) marginals. That inflates the 3-param volume without widening any 1-D projection.
- SBC and TARP CANNOT see that failure: both test marginal rank uniformity per parameter, so a posterior with correct marginals and missing correlations passes both.
- MOPED is not free where it is not needed: on the well-conditioned non-BNT vector raw NPE is ~20% tighter than MOPED, presumably non-Gaussian information the Gaussian-optimal compression discards.

*Do not edit these files. Regenerate at the source and re-publish, so the figure and its numbers can never disagree.*
