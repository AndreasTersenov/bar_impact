# HANDOFF — masked peak-count `--submean` fix (overnight 2026-06-18→19)

## TL;DR
The masked peak-count "too-tight contours" pathology is the cosmology-dependent **footprint mean**
creating an edge step that leaks into the detail-scale peaks. **Fix = subtract the footprint mean
before the starlet transform** (`--submean`, mirrors the masked-PS submean). Confirmed on a 6-area
Fisher pilot: peaks σ(S₈)-vs-area slope −0.08 (anomaly) → −0.35, matching l1 −0.38. Full memory:
`masked-peak-counts-too-tight`.

## What is RUNNING overnight
`scripts/diagnostics/run_submean_overnight.sh` (background; 60 workers; OMP=1). Reprocesses the
**non-BNT** masked peaks with `--submean` for all 6 areas (order: 14001, 5001, 2001, 10001, 28001,
35001), each: fid nobaryons + fid baryonified + grid nobaryons, then copies grid `_submean`
aggregates `new_grid/ → grid/`.
- Log: `pgrep -af run_submean_overnight` (running task id `bitg4yit1`). Scope: grid NOBARYONS only,
  fid BOTH (confirmed with user).
- **SAFETY GATE**: each area verifies row counts (grid==16965 aligned with params, fid==200) and
  refuses to copy a short/misaligned aggregate, logging `ROWCOUNT FAIL`. **Grep the log for
  `ROWCOUNT FAIL`** before using any area. 14001 uses `--force-overwrite` (had suspect partials).
  Pre-flight checks done: scope, processing<->NPE filename match, alignment, disk (43T free), gate test.
  iter note: existing non-submean data is iter=3, this run iter=1 (negligible for peaks).
- ~1.5 h/area grid → ~9 h total. **Resumable**: re-run the same script; `process_file` skips files
  that already exist (no `--force-overwrite`). If the night was short, finished areas are fully usable.

## Outputs produced (per area A)
- Grid: `grid/all_peak_counts_grid_nobaryons_bin{1..4}_masked_{A}sqdeg_submean_noisy_s0.26_new_normalization.npy`
  (also in `new_grid/`; the copy to `grid/` is what the NPE reads).
- Fid:  `fiducial/cosmo_fiducial/all_peak_counts_fiducial_{nobaryons,baryonified}_bin{1..4}_masked_{A}sqdeg_submean_...npy`

## NEXT (morning) — run the NPE + tension on the submean data
Env: **jaxili + GPU** (`/home/tersenov/anaconda3/envs/jaxili/bin/python`). Pick a free GPU first
(`nvidia-smi`; default GPUs 0–2, leave 3). Single-area validation (14001, scales1234):
```
PY=/home/tersenov/anaconda3/envs/jaxili/bin/python
# train on submean nobaryons grid + sample at nobaryons fiducial
$PY scripts/run_npe_peak_counts_inference.py --simulation-type nobaryons --fiducial-type nobaryons \
   --bins 1,2,3,4 --scales 0,1,2,3 --noisy --new-normalization --masked --mask-area-sqdeg 14001 \
   --submean --train --gpu 0
# sample at baryonified fiducial (loads the checkpoint)
$PY scripts/run_npe_peak_counts_inference.py --simulation-type nobaryons --fiducial-type baryonified \
   --bins 1,2,3,4 --scales 0,1,2,3 --noisy --new-normalization --masked --mask-area-sqdeg 14001 \
   --submean --gpu 0
```
Then the tension (compare to the existing non-submean posteriors):
```
python scripts/compute_tension_statistics_peak_counts.py --submean     # loops areas+scales
```
**Expected:** the inflated masked-peak baryon tension (the ×12.9 / ×6.0) drops to an honest level;
masked-peak constraining power now scales with area like l1.

## Plumbing already done (this session)
- `--submean` in `peak_counts_processing.py` (process_file: `kg=(kg-kg[mask!=0].mean())*mask` after
  masking; per-file + `--save-combined` outputs tagged `_submean`; gated to `--apply-mask`). Verified
  bit-exact vs manual submean.
- `--submean` in `run_npe_peak_counts_inference.py` (one change at `mask_suffix`/`mask_label` → tags
  both the input aggregates it reads AND the posterior it writes).
- `--submean` in `compute_tension_statistics_peak_counts.py`.

## NOT done — deferred (do with validation, not blindly)
- **BNT submean.** `bnt_peak_counts_processing.py` uses mask→noise→BNT, leaving NOISE outside the
  footprint (no clean step). BNT submean must be combined with **order-A** (noise→mask→BNT, outside=0;
  the `--mask-correction` path in `bnt_l1_peaks_processing.py`). Implement + validate (like non-BNT)
  before running. The user's analysis is BNT-heavy, so this is the main remaining production task.
- Full NPE **sweep** with submean: `submit_npe_peak_counts_parameter_sweep_parallel.py` needs
  `--submean` added to its `BASE_CMD` (run_npe already supports it).
- `--submean` in the combined `l1_peaks_processing.py` / `bnt_l1_peaks_processing.py` for consistency.

## Diagnostics (for reference)
`scripts/diagnostics/masked_peaks_*.py`; pilot data in `outputs/diagnostics/masked_peaks/`.
Erosion / apodization / renormalization / pixel-geometry were all ruled out before submean was found.
