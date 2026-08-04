# FLAGSHIP: BNT vs non-BNT at matched lmax=460 (embedding network, no MOPED)

THE PAPER FIGURE for the BNT constraining-power result. Both arms at the SAME scale cut (lmax=460, rebin=20), both fed the FULL data vector (92 features BNT, 50 non-BNT) noise-whitened by the analytic covariance, with a 16-dimensional embedding network inside the normalizing flow trained jointly under the NPE loss. No separate compression stage. The only difference between the two contours is the BNT basis: BNT cuts bin-1 only and keeps bins 2-4 to lmax=1024, non-BNT cuts every bin. BNT/non-BNT FoM3 = 1.41x; per parameter sigma(Om) 1.15x, sigma(S8) 1.24x, sigma(w0) 1.04x, so the gain is concentrated where lensing constrains and is negligible in w0. Both posteriors pass TARP (max|ecp-alpha| 0.05, 0.05) and SBC (rank-std 0.29-0.30 against the ideal 0.289). INDEPENDENT CORROBORATION: the MOPED analysis of the same configuration gives 1.47x (see bnt_flagship_matched_c460_pooled). MOPED assumes Gaussianity and needs the analytic covariance and local Jacobian; the embedding needs none of them, so the two share essentially no failure modes and agree to 4 percent. Read as constraining power at fixed cut, NOT baryon mitigation: BNT bin-1 crosses 0.3 sigma at a LOWER lmax than non-BNT cut-all. Bias at this cut: non-BNT safe at 0.17 sigma, BNT marginally at the 0.3 threshold and tolerated on its error bar.

- **source**: `plots/bnt_vs_nonbnt_embedding_14000`
- **generator commit**: `unknown`
- **generated**: 2026-08-04T16:20:54Z
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
