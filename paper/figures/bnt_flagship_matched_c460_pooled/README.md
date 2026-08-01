# FLAGSHIP: BNT vs non-BNT at matched lmax=460 (MOPED, pooled)

THE FLAGSHIP CONSTRAINING-POWER RESULT. Both arms score/MOPED-compressed at the SAME scale cut (lmax=460, rebin=20, hybrid covariance, 5 NDE seeds each), so the only difference between them is the BNT basis: BNT cuts bin-1 only and keeps bins 2-4 to lmax=1024 (92 of 120 columns), non-BNT cuts every bin (50 of 120). BNT gives 1.47x the 3-parameter FoM. Both posteriors are calibrated (SBC rank-std 0.28-0.29 against the ideal 0.289) and on-truth (null means within 0.3 sigma), and both are unbiased at this cut (baryon tension 0.30+/-0.09 sigma for BNT, 0.17+/-0.03 for non-BNT), which is what makes the matched comparison fair. Read this as constraining power at fixed cut, NOT as baryon mitigation: BNT bin-1 actually crosses 0.3 sigma at a LOWER lmax (460) than non-BNT cut-all (620), so cutting only bin 1 does not control baryons better - it retains more information at equal unbiasedness. The 1.47x is also conditional on the compression: under raw NPE the same comparison inverts to 0.33x, because a flow fed the ill-conditioned BNT vector loses the Omega_m-S8 degeneracy entirely (r=-0.03 against the physical -0.9) while passing every standard calibration check.

- **source**: `plots/bnt_flagship_matched_c460_14000_pooled`
- **generator commit**: `5a917568072f4c604bc2203292973bf2f3ba509a`
- **generated**: 2026-08-01T12:15:06Z
- **published**: 2026-08-01T12:21:26+00:00 at repo `5a91756`
- **rows in values.csv**: 2

## Scales included

- **scales_included**: both arms at ell_max=460 (matched); BNT cuts bin-1 only, bins 2-4 to 1024

## Caveats (from provenance)

- Both series are calibrated and on-truth (SBC rank-std 0.28-0.29 vs the ideal 0.289; null means within 0.3 sigma of truth), so the FoM difference is a real difference in information, not one posterior being broken.
- The decisive quantity is r(Omega_m, S8). Weak lensing carries a physical degeneracy near -0.9; raw NPE on the ill-conditioned BNT vector returns -0.03, i.e. it loses the degeneracy structure entirely while keeping plausible (even tighter) marginals. That inflates the 3-param volume without widening any 1-D projection.
- SBC and TARP CANNOT see that failure: both test marginal rank uniformity per parameter, so a posterior with correct marginals and missing correlations passes both.
- MOPED is not free where it is not needed: on the well-conditioned non-BNT vector raw NPE is ~20% tighter than MOPED, presumably non-Gaussian information the Gaussian-optimal compression discards.

*Do not edit these files. Regenerate at the source and re-publish, so the figure and its numbers can never disagree.*
