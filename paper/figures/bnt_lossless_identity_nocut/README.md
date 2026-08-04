# VALIDATION: BNT vs standard basis with NO scale cut (lossless identity)

THE ORACLE for the BNT result. BNT is an invertible linear map of the standard tomographic spectra, so with NO scale cut applied the two bases carry IDENTICAL information and the contours must coincide. They do: per-seed-mean FoM3 4.326e5 (BNT) vs 4.420e5 (standard), ratio 0.979. Both arms at lmax=1024, 120 features each, same embedding-network setup as the flagship at position 6. The earlier score work verified this identity ANALYTICALLY at the summary level (gate 2, agreement to 1e-13); this is the first end-to-end confirmation through trained posteriors. CONVENTION NOTE: the ratio is 0.979 (mean of per-seed FoM3), 1.031 (FoM3 of the seed-averaged covariance) and 1.142 (FoM3 of pooled samples). The spread is seed-level training scatter, not a physical effect - the standard-basis arm has a per-seed spread of 19 percent (4.42e5 +/- 8.2e4) against BNT's 7 percent, so pooling fattens that arm. Report the per-seed mean: it treats each seed as one independent analysis and does not fold training scatter into the contour width. The marginals agree to ~5 percent per parameter with the STANDARD basis marginally tighter, i.e. no systematic advantage to either, exactly as an invertible map requires. WHY THIS MATTERS: it establishes that the 1.41x measured at lmax=460 is the effect of the TARGETED CUT and not an artifact of working in the BNT basis.

- **source**: `plots/bnt_lossless_identity_nocut_14000`
- **generator commit**: `unknown`
- **generated**: 2026-08-04T16:21:26Z
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
