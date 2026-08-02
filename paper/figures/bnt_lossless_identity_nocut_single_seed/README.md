# VALIDATION: no scale cut, lossless identity (single seed)

Single-seed companion to the no-cut oracle at position 6.3. Each arm shows its most representative NDE seed (BNT run 45, standard run 41), chosen independently per arm. Ratio 0.866x. This is the LEAST stable of the four seed conventions precisely because it compounds two independent selections - here the standard basis happened to draw a tight seed (5.32e5 against its 4.15e5 seed-average). Across all four conventions the no-cut ratio spans 0.87 to 1.14, centred on unity, which quantifies the seed-level training scatter (~5 percent per parameter). Do NOT quote this figure's ratio as the oracle result; use the per-seed mean 0.979 from position 6.3. Shown because the contour SHAPES are what the oracle is really about, and they coincide here as clearly as in the pooled version.

- **source**: `plots/bnt_lossless_identity_nocut_14000_single_seed`
- **generator commit**: `80956a42f88866b110521653cd1338b0b0f13f28`
- **generated**: 2026-08-01T19:12:23Z
- **published**: 2026-08-02T12:03:21+00:00 at repo `489dd05`
- **rows in values.csv**: 2

## Scales included

- **scales_included**: both arms at ell_max=460 (matched); BNT cuts bin-1 only, bins 2-4 to 1024

## Caveats (from provenance)

- Both series are calibrated and on-truth (SBC rank-std 0.28-0.29 vs the ideal 0.289; null means within 0.3 sigma of truth), so the FoM difference is a real difference in information, not one posterior being broken.
- The decisive quantity is r(Omega_m, S8). Weak lensing carries a physical degeneracy near -0.9; raw NPE on the ill-conditioned BNT vector returns -0.03, i.e. it loses the degeneracy structure entirely while keeping plausible (even tighter) marginals. That inflates the 3-param volume without widening any 1-D projection.
- SBC and TARP CANNOT see that failure: both test marginal rank uniformity per parameter, so a posterior with correct marginals and missing correlations passes both.
- MOPED is not free where it is not needed: on the well-conditioned non-BNT vector raw NPE is ~20% tighter than MOPED, presumably non-Gaussian information the Gaussian-optimal compression discards.

*Do not edit these files. Regenerate at the source and re-publish, so the figure and its numbers can never disagree.*
