# BNT vs standard basis: fractional baryonic impact (l1_scale1)

- **source**: `outputs/plots/bnt_frac_diff/bnt_frac_diff_l1_scale1`
- **generator commit**: `unknown`
- **generated**: 2026-08-03T09:28:12Z
- **published**: 2026-08-03T21:09:00+00:00 at repo `9ef3572`
- **rows in values.csv**: 276

## Scales included

- **scales_included**: wavelet scale 1 (~10')

## Known gaps

- provenance is missing 'git_commit'
- provenance git_commit is unknown — the figure cannot be traced to the code that made it

## Caveats (from provenance)

- Bin 1 is expected to coincide between the two bases: the BNT matrix's first row is the identity, so BNT bin 1 IS standard bin 1. Agreement there is a correctness check.
- No +5 offset in the HOS denominator; empty bins are masked instead.
- y-range is fitted to the CURVES; the near-empty end bins have bands far larger.
- Band scaled to survey area by 1/sqrt(f_sky) -- indicative for the HOS, which have no mode-counting result behind them.

*Do not edit these files. Regenerate at the source and re-publish, so the figure and its numbers can never disagree.*
