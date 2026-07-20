# Power-spectrum production plan — mean-subtracted, multi-mask, scale-matched

Status: drafted for sign-off (2026-06-16). Branch: `ps-lowell-mean-leakage`.

## Objective

Produce the paper's masked angular power-spectrum NPE contours at the six survey areas, using
the **full practically-safe low-ℓ information** now that the monopole (mass-sheet) leakage is
fixed, plus a consistent full-sky reference. Build the data vectors **once** so any ℓ-range /
binning choice — including the baryon-bias-vs-ℓ-cut study — is a cheap downstream NPE rerun.

## Background already established

- The masked PS leaks the unobservable mask-weighted monopole into ℓ≲120; `--subtract-mean`
  (mask-weighted monopole removal, monopole only) removes it. Implemented in
  `cross_power_spectrum_processing_master.py` (default off, `_submean` tag, records μ).
- Fisher gate confirmed: raw masked low-ℓ is artificially tight (σ(S₈) up to 3.4× tighter than
  full-sky); submean restores the physical ordering (masked looser than full-sky) and leaves
  ℓ>100 untouched. See `outputs/diagnostics/fisher_gate/`.
- Scale-matching to the wavelet HOS resolved (`docs/starlet_scale_ell.md`): the published
  wavelet HOS (`scales1234`) spans ℓ≈30–1535; the largest wavelet scale (wav3) half-power edge
  is ℓ≈37. **No coarse scale is needed** to justify the PS low-ℓ floor; the coarse-scale HOS
  work (it is not mean-centered) is deferred (`memory: l1-peak-coarse-not-mean-centered`).

## Decisions (locked)

| item | value | rationale |
|------|-------|-----------|
| ℓ_min (headline) | **37** | conservative wav3 half-power edge; escapes the coarse/mean regime |
| ℓ_max (headline) | **1024** | reliable range at nside 512 (above is pixel-window-suppressed) |
| storage | **unbinned, nlb=1, lmax=1024** | per-ℓ → any ℓ-range + any binning chosen downstream |
| masks (sq deg) | **2000, 5000, 10000, 14000, 28000, 35000** | the paper's `MASK_AREAS` |
| fix | **`--subtract-mean` on, for every mask** | mass-sheet gauge |
| full-sky | **reuse existing per-ℓ Cls; NPE only** | full-sky is leakage-free; vectors already exist |

Open option: store to lmax=1535 instead of 1024 for completeness. Not recommended (ℓ>1024 is
pixel-window/aliasing-limited and unusable), and it needs an `--nlb 1` override since the
pipeline auto-bins nlb=4 above ℓ=1500. Flag if you want it anyway.

## Design principle — compute once, slice/rebin downstream

The expensive step is the masked PS reprocess. Stored **unbinned per-ℓ**, every later choice is
a cheap NPE rerun on the same `.npy`:
- any ℓ_min (37 headline; explore down to <30 / the coarse regime for the standalone study);
- any ℓ_max (the baryon-bias scan walks this end down);
- any rebinning — the loader already does ℓ-slice + boxcar `rebin_cls`; we add a `log` option.

Nothing about ℓ-range or binning is baked into the heavy run. Only lmax (1024) and "unbinned"
(automatic at 1024) must be fixed beforehand.

## Phase 1 — masked PS reprocessing (the expensive run)

For each of the 6 masks: **fiducial (200 perms) for BOTH `nobaryons` and `baryonified`** (the two
observed-data scenarios for the tension comparison), and the **grid (16965 realizations) for
`nobaryons` only** (the NPE training set — a baryonified grid is not needed). So 6 heavy grid
runs, not 12.

```
python scripts/cross_power_spectrum_processing_master.py \
    --apply-mask --mask-area-sqdeg <AREA> --apodization-scale-deg 2.0 \
    --subtract-mean --lmax 1024 \
    --bin-range 1 2 3 4 --noise-level 0.26 \
    --num-workers 50 --aggregate-for-inference --inference-output-dir <DIR>
# (noise is ON by default in the processing script; --no-noise disables it. There is no --noisy.)
```

Notes / required fixes before launching:
- **Aggregation must propagate the `_submean` tag.** `aggregate_for_inference` currently drops
  it from the aggregated filename (so raw/submean would collide). Fix: add `_submean` to the
  aggregated `.npy` name (one-line change, mirrors `process_file`).
- **Verify provenance is current.** The on-disk masked Cls were made by an older NaMaster; we
  recompute everything with the cosmostat_new venv (pymaster 2.5.2) for internal consistency.
- Sequencing: run **14000 first** (headline), validate end-to-end, then the other five. Stage
  by mask so a problem surfaces on one mask, not all six.

Run environment: `cosmostat_new` venv python (`/home/tersenov/software/cosmostat_new/.../bin/python`),
**50 CPU workers** (titan, no scheduler; cluster-resources default). Rough cost: ~17k×2 maps/mask
× MASTER decouple; ~tens of minutes per mask per sim-type at 50 workers after the one-time MCM
build → order ~1 day for all six, stageable.

## Phase 2 — full-sky reference (NPE only)

No PS recompute. Reuse the existing per-ℓ full-sky grid + fiducial vectors
(`all_cross_cls_grid_nobaryons_bins1234_..._lmax….npy`, shape (16965, 6×n_ell)), confirming
they use the same ℓ-handling/normalisation as the masked path. Rerun NPE at the matched ℓ range
(≥37) so the comparison is apples-to-apples.

## Phase 3 — NPE production + comparisons

Per mask, run NPE on the masked-submean vectors at the headline cut (ℓ∈[37,1024]) for the paper
comparisons (`nobaryons_vs_baryonified`, `nobaryons_vs_nobaryons`, etc.), plus the full-sky arm.
NPE training runs on GPU (JAX/jaxili): use **GPUs 0, 1, 2** (`CUDA_VISIBLE_DEVICES=0,1,2` /
the submit-script gpu arg), leaving GPU 3 free per the titan cluster-resources convention.
Deliverables: posterior samples, tension stats, and the contour plots at the six mask levels
(getdist), with the full-sky overlay. Reuse the gate plotting utilities.

## Validation — gate per mask (back-pressure)

For each mask, before trusting its contours, run the Fisher gate
(`scripts/diagnostics/fisher_gate_mean_subtraction.py`, parameterised by mask area) on the
submean vectors:
1. masked-submean σ(S₈) at ℓ≥37 is **≥ full-sky** (physical ordering — masking loses info);
2. result is **stable** to ℓ_min within the safe window (insensitive 30↔37↔50);
3. ℓ>100 unchanged vs the old raw result (the fix only touched low-ℓ).
A mask only proceeds to NPE once its gate passes. This makes every scale choice defensible.

## Extension — baryon-bias vs ℓ-cut (reuses everything)

Once the per-ℓ submean vectors exist, the baryon study is a sweep of **ℓ_max** (baryons live at
high ℓ): rerun NPE / tension for `nobaryons_vs_baryonified` over a grid of ℓ_max (e.g. 340→1020,
matching the published `UPPER_CUTS`), holding ℓ_min=37 fixed, and watch how the baryonic bias
collapses as small scales are removed. No new data vectors — pure NPE reruns.

## Code changes needed (small)

1. `aggregate_for_inference`: propagate `_submean` into the aggregated `.npy` filename.
2. NPE loader: add a `log` rebinning option alongside boxcar `rebin_cls` (downstream only).
3. (Only if lmax=1535 chosen) add an `--nlb` override to force unbinned at high lmax.
4. Full-sky NPE: confirm ℓ-offset / normalisation matches the masked path (the `ell_offset`
   2-vs-0 seam) so the comparison is clean.

## Deferred (noted, not in this run)

- Coarse-scale L1/peak rerun with mean-centering + gate validation (only if probing ℓ<30 or
  extending the HOS to large scales). See `memory: l1-peak-coarse-not-mean-centered`.
- BNT masked PS (the BNT coarse is already 0; BNT PS leakage check is a separate pass).

## Verification / success criteria

- Phase-1 regression: with `--subtract-mean` **off**, a recomputed map reproduces itself
  bit-for-bit (no-op check); with it **on**, only ℓ≲120 bandpowers change.
- Each mask passes its gate (physical ordering + stability + ℓ>100 unchanged).
- Recovered ℓ≥37 masked contours are looser than full-sky and tighter than the ℓ>100-only
  published result, with no residual anomaly.
- Provenance: every output records mask, μ per bin, lmax, nlb, mean_subtracted in metadata.
