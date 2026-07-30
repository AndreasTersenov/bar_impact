# Fisher 68% contours, baryon-safe regime (full sky)

FISHER FORECAST, NOT NPE, and FULL SKY not the masked paper footprint -- do not read these as posteriors. PS at lmax=400 against L1 and peaks with the finest wavelet scale dropped. Recovered after the disk failure destroyed both the pdf and the png; the regeneration reproduces the surviving pre-crash FoM table to 7.3e-13, i.e. floating-point roundoff. The Jacobian is a linear response fit and is the dominant approximation: it can over- OR under-state a probe's sensitivity, so the HOS figure of merit is not a bound on the NPE.

- **source**: `outputs/diagnostics/constraining_power/fisher_contours_baryon_safe`
- **generator commit**: `unknown`
- **generated**: 2026-07-29T21:47:30+00:00
- **published**: 2026-07-30T07:32:09+00:00 at repo `b900627`
- **rows in values.csv**: 4

## Known gaps

- _values.csv has no seed-count column; n_seeds is the column that reveals a point averaging a different subset than it used to
- provenance git_commit is unknown — the figure cannot be traced to the code that made it
- provenance is missing 'mplstyle'

## Caveats (from provenance)

- FISHER, not NPE. The Jacobian is a linear response fit and is the dominant approximation; it can over- OR under-state a probe's sensitivity, so do not read the HOS FoM as a bound on the NPE. See the module docstring.
- The HOS gain is jacobian-sensitive: l1's 6-param FoM lead over PS l100 is ~x70 with a global linear Jacobian but ~x17 with the local derivative used here (JAC_MODE='local').
- 200 fiducial perms may under-estimate the non-Gaussian covariance tails of l1/peaks, which would make the HOS look optimistically tight.
- FULL-SKY, not the masked paper footprint.
- Regenerated after the RAID0 disk failure destroyed both the .pdf and .png (100% zeros) and fisher_covs.npz. Every input .npy was verified readable first; fisher_fom_table_PRECRASH_REFERENCE.json is the surviving pre-crash table, kept for numerical comparison.

*Do not edit these files. Regenerate at the source and re-publish, so the figure and its numbers can never disagree.*
