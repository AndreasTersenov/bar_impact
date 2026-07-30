# Submean masked-peak investigation — plots

One subdirectory per footprint, `<AREA>sqdeg/`. "submean" = footprint mean subtracted before the
starlet transform (fixes the spurious masked-peak tightness; lets the coarse scale land in range).

Per area:
- `tension_scales234.png`          nobaryons vs baryonified, FINEST scale dropped -> baryon-robust (~0 sigma).
                                   dotted red = scales1234 baryonified (finest IN) for reference.
- `tension_submean_vs_orig.png`    (14001 only) submean vs original non-submean contours.
- `coarse_contours_scales2345.png` coarse scale ADDED (scales2345); dotted = scales234 (no coarse) ref.
- `coarse_datavector_vs_S8.png`    (14001 only) coarse data vector, submean vs non-submean, colored by S8.
- `coarse_binning.png`             coarse data vector / tail zoom / per-bin S8 sensitivity (binning diagnostic).

Tension numbers per area (all configs): `../../submean_area_summary.txt` (appended by the monitor).
Pipeline scripts (this session): `/tmp/{npe_area.sh, npe_area_coarse.sh, analyze_area.py, analyze_area_coarse.py}`.
