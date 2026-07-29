# HANDOFF — power-spectrum production (overnight autonomous run)

Started 2026-06-16 ~19:40. Branch `ps-lowell-mean-leakage`. Plan: `docs/POWER_SPECTRUM_PRODUCTION_PLAN.md`.

## What is running now (autonomous)

**Phase 1 reprocess** — `scripts/diagnostics/run_overnight_reprocess.sh` (background task `b7m2ee0qe`).
Masked PS, **mean-subtracted (`--subtract-mean`)**, per-ℓ (nlb=1), **lmax 1024**, for all 6 masks
(14000 first, then 10000, 5000, 2000, 28000, 35000). Per mask: **fiducial** for BOTH nobaryons
and baryonified (the two observed-data scenarios), **grid** for nobaryons ONLY (the NPE training
set — no baryonified grid needed). Task `bp2p5u0or`.
- Live status: `outputs/diagnostics/prod_overnight/STATUS.log` (one START/DONE line per run).
- Per-run logs: `outputs/diagnostics/prod_overnight/<tag>.log`.
- Outputs: per-perm `..._master_submean_noisy_s0.26.npz` next to each map; aggregated
  `all_cross_cls_{grid,fiducial}_{sim}_bins1234_masked_<area>sqdeg_apod2.0_master_submean_noisy_s0.26.npy`
  in `new_grid/` and `fiducial/cosmo_fiducial/`. ~4–5 h total.

## Validated before launch (pilot, 14000 fiducial nobaryons)

Per-ℓ aggregated shape (200, 6×1023); `mean_subtracted=True`; monopoles `[0.0075, 0.0163, 0.036,
0.0544]` = the Phase-A patch means (correct μ removed). Pipeline is sound.

## What happens next (autonomous, on reprocess completion)

When `b7m2ee0qe` completes I am re-engaged and will:
1. Make + **test** the NPE loader change to read the `_submean` aggregates (the inference loader
   currently doesn't look for the `_submean` tag) — tested on **14000 first**.
2. Run the Fisher gate on the 14000 submean production vectors (back-pressure: masked-submean must
   be ≥ full-sky and ℓ>100 unchanged) before trusting it.
3. If 14000 validates → run NPE (GPUs **0,1,2**) for all masks at ℓ∈[37,1024], masked-submean +
   full-sky, generate the six-mask contour figures + tension stats.
4. If NPE testing fails → **stop**, leave all reprocessed data intact, and append a clear
   "NPE blocked on X" note to the bottom of this file rather than produce garbage.

## Realistic scope by morning

- **Reprocess: done** (all per-ℓ submean data vectors ready) — high confidence.
- **NPE/contours: done if the loader change validates on 14000, else staged** with a note here.
  (The NPE loader change is the one untested piece; that's why it's gated on a 14000 test.)

## Deferred (not in this run)

- L1/peak coarse-scale rerun with mean-centering (`memory: l1-peak-coarse-not-mean-centered`).
- BNT masked PS leakage pass.

## Morning checklist

1. `cat outputs/diagnostics/prod_overnight/STATUS.log` — did all 24 runs finish exit=0?
2. Read the bottom of this file for the NPE outcome / any blocker note.
3. Contours (if produced): `outputs/diagnostics/fisher_gate/` and the NPE output dirs.

---
(Autonomous progress notes appended below as the run advances.)
