# Fig. 2 (14000 deg2 variant) - fractional baryonic impact on the auto power spectra

- **source**: `outputs/plots/ps_frac_diff/ps_frac_diff_matched_survey_14000`
- **generator commit**: `cb724b61c20ae63fa66fbe357b1d7042ce54c418`
- **generated**: 2026-08-02T12:45:27Z
- **published**: 2026-08-02T12:45:29+00:00 at repo `cb724b6`
- **rows in values.csv**: 40

## Known gaps

- _values.csv has no seed-count column; n_seeds is the column that reveals a point averaging a different subset than it used to
- provenance does not state the SCALES included (no 'scales_included', 'conventions', 'cuts', 'ps_edges' or 'lmin') — a figure that does not say which multipoles and wavelet scales went in cannot be interpreted

## Caveats (from provenance)

- THE PER-BAND COMPARISON UNDERSTATES THE IMPACT. The curve sits below the band at most ell, but the baryonic shift is COHERENT while the noise is random, so it accumulates as sqrt(N). See cumulative_SN_of_mismatch: it reaches ~10 sigma (full sky) / ~6 sigma (14000 deg2) by lmax=1000 in the high-z bins. The caption MUST say this, or the figure argues against the paper's own conclusion.
- The denominator is the NOISY DMO spectrum, so this is the baryonic shift relative to the measured spectrum. Fig. B.1 divides by the signal alone, so the same physics appears up to ~34x larger there at high ell. State this in the caption.
- Shape-noise realizations are INDEPENDENT between the baryonified and DMO runs (power_spectrum_processing.py:91 seeds from os.urandom). The survey band is unaffected since it involves no subtraction, but the residual wiggle in the mean curve is caused by this and can only be removed by regenerating with matched seeds.
- PUBLISHED BINNING HAS TWO DEFECTS, reproduced by --binning published: the log-spaced band edges are ell VALUES used as INDICES into an array already sliced at lmin, so every band is shifted up by 30 and ell 30-58 never enter the figure (lowest ell plotted is 59); and the points are drawn on a linear axis while the bands are logarithmic. --binning fixed corrects both.

*Do not edit these files. Regenerate at the source and re-publish, so the figure and its numbers can never disagree.*
