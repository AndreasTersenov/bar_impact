# Fisher 68% contours, full-resolution regime (full sky)

The full-resolution companion to the baryon-safe Fisher figure: PS to lmax=1024 against the higher-order statistics keeping all four detail scales. Same caveats -- Fisher, full sky, linear Jacobian.

- **source**: `outputs/diagnostics/constraining_power/fisher_contours`
- **generator commit**: `unknown`
- **generated**: 2026-07-29T21:47:29+00:00
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
