# BNT vs standard basis: fractional baryonic impact (ps)

- **source**: `outputs/plots/bnt_frac_diff/bnt_frac_diff_ps`
- **generator commit**: `unknown`
- **generated**: 2026-08-09T10:03:37Z
- **published**: 2026-08-09T10:03:42+00:00 at repo `b53c9c9f`
- **rows in values.csv**: 80

## Scales included

- **scales_included**: PS: 10 logarithmic bands, lmin=30, lmax=1024

## Known gaps

- provenance is missing 'git_commit'
- provenance git_commit is unknown — the figure cannot be traced to the code that made it

## Caveats (from provenance)

- Bin 1 is expected to coincide between the two bases: the BNT matrix's first row is the identity, so BNT bin 1 IS standard bin 1. Agreement there is a correctness check.
- No +5 offset in the HOS denominator; empty bins are masked instead.
- y-range is fitted to the CURVES; the near-empty end bins have bands far larger.
- Band scaled to survey area by 1/sqrt(f_sky) -- indicative for the HOS, which have no mode-counting result behind them.

*Do not edit these files. Regenerate at the source and re-publish, so the figure and its numbers can never disagree.*
