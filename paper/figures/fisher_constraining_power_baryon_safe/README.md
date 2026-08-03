# Fisher 68% contours, baryon-safe regime (full sky)

- **source**: `outputs/diagnostics/constraining_power/fisher_contours_baryon_safe`
- **generator commit**: `unknown`
- **generated**: 2026-07-29T21:47:30+00:00
- **published**: 2026-08-03T21:10:14+00:00 at repo `9ef3572`
- **rows in values.csv**: 4

## Scales included

- **ps_bandpower_edges**: [37, 68, 100, 140, 200, 280, 400, 560, 760, 1024]
- **hos_scales_full**: [0, 1, 2, 3]
- **hos_scales_baryon_safe**: [1, 2, 3]
- **regime**: baryon-safe: PS lmax=400, HOS drop the finest wavelet scale (scales234)

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
