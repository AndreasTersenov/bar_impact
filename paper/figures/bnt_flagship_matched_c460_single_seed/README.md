# CROSS-CHECK: BNT vs non-BNT at matched lmax=460 (MOPED, single seed)

Single-seed companion to the MOPED cross-check at position 6.1; see the embedding flagship at position 6 for the paper result. Each arm shows its most representative NDE seed (scripts/tension/seeds.py, median-referenced on centre and width; both select run 41). Ratio 1.507x against the pooled 1.467x - the difference is seed selection, not physics, since the two arms draw representatives independently. Quote pooled numbers.

- **source**: `plots/bnt_flagship_matched_c460_14000_single_seed`
- **generator commit**: `unknown`
- **generated**: 2026-08-04T16:21:03Z
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
