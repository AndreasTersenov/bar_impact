# Fractional baryonic impact on the starlet peak counts (full sky)

- **source**: `outputs/plots/hos_frac_diff/hos_frac_diff_peaks`
- **generator commit**: `0bbd792c834ac14f516aa4dbd6e7092153578867`
- **generated**: 2026-08-02T13:58:21Z
- **published**: 2026-08-02T13:58:34+00:00 at repo `0bbd792`
- **rows in values.csv**: 353

## Scales included

- **scale 1**: 10'
- **scale 2**: 20'
- **scale 3**: 40'

## Caveats (from provenance)

- No +5 offset in the denominator. The published figure used <stat>+5, which is 100% of the denominator in the empty tails; those bins are masked here instead.
- Peak counts are DISCRETE, so matched noise cancels less for them than for the l1-norm (correlation ~0.65 vs ~0.97) and their bands stay wider. Not a defect.
- The band is scaled to survey area by 1/sqrt(f_sky), which for a HOS has no mode-counting justification -- indicative only.

*Do not edit these files. Regenerate at the source and re-publish, so the figure and its numbers can never disagree.*
