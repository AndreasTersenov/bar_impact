# Noiseless fractional baryonic impact on the auto power spectra, standard vs BNT bins

Noiseless companion to Fig. 2: <dC_l>/<C_l> for the 4 standard tomographic bins (solid) and the 4 BNT-nulled bins (dotted), 200 fiducial realisations, lmax 1024, no shape noise. Band is the standard error of the PAIRED difference std(C_bar-C_dmo)/sqrt(200)/<C_dmo> -- both sets are the same 200 perms, so treating them as independent would inflate the band ~10x. The baryonified halves were rebuilt post-crash by SLURM job 597132 from the read-only CosmoGridV1 release; the nobaryons halves are the 2025-10 originals. BNT bin 1 == standard bin 1 by construction (first BNT row is identity).

- **source**: `outputs/plots/ps_frac_diff_noiseless/ps_frac_diff_noiseless`
- **generator commit**: `3d39df81936d88073602be68d6a734b2c4296925 (generator dirty/untracked at run time)`
- **generated**: 2026-08-05T15:05:02.157249+00:00
- **published**: 2026-08-05T15:05:02+00:00 at repo `3d39df81`
- **rows in values.csv**: 8

## Scales included

- **lmin_plotted**: 2
- **lmax**: 1024
- **note**: full multipole range, no scale cut; noiseless product is lmax=1024 only -- never mix with the *_lmax2048 variants

## Known gaps

- _values.csv has no seed-count column; n_seeds is the column that reveals a point averaging a different subset than it used to
- provenance is missing 'figure'

## Caveats (from provenance)

- noiseless -- no shape noise anywhere; row scatter is sample variance only
- BNT bin 1 is standard bin 1 bit-for-bit (first BNT row is the identity); the two curves overlapping exactly is a consistency check, not a duplicate
- baryonified halves rebuilt post-crash from the read-only CosmoGridV1 release (job 597132); nobaryons halves are the 2025-10 originals

*Do not edit these files. Regenerate at the source and re-publish, so the figure and its numbers can never disagree.*
