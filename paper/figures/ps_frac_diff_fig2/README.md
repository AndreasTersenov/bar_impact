# Fig. 2 - fractional baryonic impact on the auto power spectra (matched noise, survey band)

- **source**: `outputs/plots/ps_frac_diff/ps_frac_diff_matched_survey`
- **generator commit**: `2340a4ee6f6adf47706ea6e4151a94db3b307a34`
- **generated**: 2026-08-02T11:25:18Z
- **published**: 2026-08-02T11:26:08+00:00 at repo `dd75859`
- **rows in values.csv**: 40

## Known gaps

- _values.csv has no seed-count column; n_seeds is the column that reveals a point averaging a different subset than it used to
- provenance does not state the SCALES included (no 'scales_included', 'conventions', 'cuts', 'ps_edges' or 'lmin') — a figure that does not say which multipoles and wavelet scales went in cannot be interpreted

## Caveats (from provenance)

- THE PER-BAND COMPARISON UNDERSTATES THE IMPACT. The curve sits below the band at most ell, but the baryonic shift is COHERENT while the noise is random, so it accumulates as sqrt(N). See cumulative_SN_of_mismatch: it reaches ~10 sigma (full sky) / ~6 sigma (14000 deg2) by lmax=1000 in the high-z bins. The caption MUST say this, or the figure argues against the paper's own conclusion.
- The denominator is the NOISY DMO spectrum, so this is the baryonic shift relative to the measured spectrum. Fig. B.1 divides by the signal alone, so the same physics appears up to ~34x larger there at high ell. State this in the caption.
- Shape-noise realizations are INDEPENDENT between the baryonified and DMO runs (power_spectrum_processing.py:91 seeds from os.urandom). The survey band is unaffected since it involves no subtraction, but the residual wiggle in the mean curve is caused by this and can only be removed by regenerating with matched seeds.
- The published x-axis is linear while the binning is logarithmic; --ell-axis centres shows where the bins actually sit. Default reproduces the published positions.

*Do not edit these files. Regenerate at the source and re-publish, so the figure and its numbers can never disagree.*
