# FLAGSHIP: BNT vs non-BNT at matched lmax=460 (MOPED, single seed)

Single-seed companion to the flagship. Each arm shows its most representative NDE seed, chosen by scripts/tension/seeds.py against the MEDIAN across seeds on both centre and width (worst-parameter, so a seed representative in Omega_m and S8 but a width off in w0 does not qualify). This is the object a real survey reports: one trained density estimator and its own posterior, rather than a pooled stack whose width carries seed-to-seed training scatter. The ratio here is 1.507x against the pooled figure's 1.467x; both arms happen to select run 41, and the difference is seed selection rather than physics. QUOTE THE POOLED FIGURE (1.47x) for the headline - it averages the training scatter out instead of inheriting one draw of it. The per-seed scores behind the choice are recorded under seed_selection in provenance.json so the pick is auditable.

- **source**: `plots/bnt_flagship_matched_c460_14000_single_seed`
- **generator commit**: `f52188579b6301dc484c779c1440a95a41f1139a`
- **generated**: 2026-07-31T20:59:48Z
- **published**: 2026-07-31T21:00:25+00:00 at repo `f521885`
- **rows in values.csv**: 2

## Scales included

- **scales_included**: both arms at ell_max=460 (matched); BNT cuts bin-1 only, bins 2-4 to 1024

## Caveats (from provenance)

- Both series are calibrated and on-truth (SBC rank-std 0.28-0.29 vs the ideal 0.289; null means within 0.3 sigma of truth), so the FoM difference is a real difference in information, not one posterior being broken.
- The decisive quantity is r(Omega_m, S8). Weak lensing carries a physical degeneracy near -0.9; raw NPE on the ill-conditioned BNT vector returns -0.03, i.e. it loses the degeneracy structure entirely while keeping plausible (even tighter) marginals. That inflates the 3-param volume without widening any 1-D projection.
- SBC and TARP CANNOT see that failure: both test marginal rank uniformity per parameter, so a posterior with correct marginals and missing correlations passes both.
- MOPED is not free where it is not needed: on the well-conditioned non-BNT vector raw NPE is ~20% tighter than MOPED, presumably non-Gaussian information the Gaussian-optimal compression discards.

*Do not edit these files. Regenerate at the source and re-publish, so the figure and its numbers can never disagree.*
