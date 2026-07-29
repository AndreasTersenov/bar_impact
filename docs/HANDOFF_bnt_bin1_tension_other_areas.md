# HANDOFF — BNT bin-1 baryon-tension-vs-scale-cut plot for the other footprints + full sky

**Goal.** Reproduce the plot `plots/nsigma_vs_upper_cut_bnt_bin1.png` (currently only **14000 deg²**) for
the other five mask areas **{2000, 5000, 10000, 28000, 35000}** and for **full sky**. The plot is the
3-parameter (Ωm, S₈, w₀) Gaussian **Q_DM tension** between the nobaryons NULL and the baryonified
observation, as a function of the upper scale cut ℓmax, for two configs:
- **non-BNT — cut all bins** (grey squares): the standard analysis; tension rises steeply with ℓmax.
- **BNT — cut bin-1 only, bins 2–4 full** (blue circles): BNT decorrelates the bins so only bin-1 needs a
  cut; tension stays ~flat below the 0.3σ threshold while keeping small scales in bins 2–4.

Monopole-subtracted PS, ℓ≥37, rebin=20-ish (paper grid), mean±std over 5 NPE training seeds.

## What is already done vs not
- **14000 deg², BNT bin-1**: DONE — posteriors in `outputs/baryon_tension/bnt_ps_bin1_submean_l37/posteriors/mask_14000/`,
  plot `plots/nsigma_vs_upper_cut_bnt_bin1.{png,pdf}`. This is the figure to replicate per area.
- **Other 5 masked areas, BNT bin-1**: NOT done — need the NPE sweep.
- **Full sky, BNT bin-1**: NOT done — and needs a small code addition (see Gotchas).
- **non-BNT "cut all bins" (the grey curve)**: ALREADY DONE for all 6 masked areas — it is the
  `submean_l37` campaign (`docs/PLAN_tension_submean_l37.md`, "BARYON-TENSION campaign DONE 2026-06-20").
  Reuse its posteriors/tension as the grey comparison; do NOT re-run it. (For full sky, the grey comes from
  the `fullsky` campaign — verify it exists; if not, run it too.)

## Infrastructure (all exists — `scripts/tension/` package + entrypoints)
- **NPE sweep**: `scripts/run_tension_sweep.py` — flags `--bnt-bin1`, `--bnt-cutall`, `--fullsky` (mutually
  exclusive), `--areas`, `--upper-cuts`, `--runs 1..5`, `--gpus`, `--jobs-per-gpu`, `--mem-fraction`,
  `--seed-base`, `--lmin 37`, `--dry-run`. Runs the worker (`run_npe_inference_auto_cross_ps_master.py`)
  per (area × upper_cut × {null,biased} × run). **Env: jaxili** (GPU).
- **Tension + plot**: `scripts/compute_tension.py` — same `--bnt-bin1/--bnt-cutall/--fullsky/--areas`
  flags; reads the posteriors, computes Q_DM nσ (tensiometer), writes tables/CSVs and the
  `nsigma_vs_upper_cut_bnt_bin1` figure. **Env: aname** (tensiometer + getdist 1.4.3) — see
  [[bar-impact-tension-env]].
- **Campaign configs**: `scripts/tension/configs.py` — `bnt_bin1_campaign()` (`bnt=True, cut_bins=(1,)`,
  bins 2–4 kept at `full_cut=1024`), `bnt_cutall_campaign()` (`bnt=True, cut_bins=(1,2,3,4)`),
  `submean_l37_campaign()` (non-BNT cut-all = the grey curve), `fullsky_campaign()`. Upper-cut grid =
  `PAPER_UPPER_CUTS` (the ~340→1020 step-40 x-axis).
- **Monitor**: `scripts/monitor_bnt_bin1.py`. **Worker GPU packing** (from the campaign): `--jobs-per-gpu 4
  --mem-fraction 0.15` ≈ 2.8× throughput (NPE jobs ~30% util; cap JAX preallocation).

## Data availability (verified)
- Masked BNT grids (`all_bnt_cls_grid_…masked_<A>sqdeg_…submean…lmax1535.npy`) exist for all six areas
  (built by `scripts/build_bnt_grids_from_spectra.py`, M C Mᵀ shortcut).
- **Full-sky BNT grids exist** (`new_grid/all_bnt_cls_grid_nobaryons_bin1_noisy_s0.26.npy` etc.) — so
  full-sky BNT is feasible WITHOUT rebuilding grids.

## Plan
1. **Masked areas (the easy 90%).** Run the BNT bin-1 NPE sweep for the five remaining areas, then compute
   tension + plot. Reuse the existing non-BNT-cut-all (grey) tension.
   ```bash
   # jaxili, GPU. (set --gpus / --jobs-per-gpu to the box; campaign used 4/gpu, mem-fraction 0.15)
   /home/tersenov/anaconda3/envs/jaxili/bin/python scripts/run_tension_sweep.py --bnt-bin1 \
       --areas 2000 5000 10000 28000 35000 --runs 1 2 3 4 5 --gpus 0 1 --jobs-per-gpu 4 --mem-fraction 0.15
   # aname, tensiometer + plot
   /home/tersenov/anaconda3/envs/aname/bin/python scripts/compute_tension.py --bnt-bin1 \
       --areas 2000 5000 10000 28000 35000
   ```
   Check first with `--dry-run` (job count = areas × cuts × 2 × runs). Verify the grey (non-BNT cut-all)
   tension is picked up from `submean_l37`; if `compute_tension` doesn't auto-overlay it, point it at those
   posteriors or add the overlay in the plot builder.
2. **Full sky (the gotcha).** `--fullsky` and `--bnt-bin1` are separate `elif` branches in BOTH
   `run_tension_sweep.py` (lines ~54–61) and `compute_tension.py` (~34–39) — they can't currently be
   combined. Options: (a) add a `fullsky_bnt_bin1_campaign()` to `configs.py` and a branch in both
   entrypoints (cleanest), or (b) a one-off: construct `fullsky_campaign()` then set `bnt=True, cut_bins=(1,)`
   on it and drive `sweep.run_sweep` directly. Full-sky uses the healpy spectra (`all_bnt_cls_grid_…_noisy_s0.26.npy`,
   no `masked` tail, `ell_offset=0`); confirm the worker's fullsky+bnt path loads the BNT fullsky files and
   applies the bin-1 cut. Also confirm the full-sky non-BNT (grey) tension exists (`fullsky_campaign`); run
   it if not.
3. **Plots.** Either one panel per area (like the 14000 figure) or a 6-panel + a full-sky panel. The 14000
   builder lives in `compute_tension.py` — extend it to loop areas, or call it per area.

## Verification / back-pressure
- Per area, the figure should show: BNT bin-1 (blue) ≈ flat and below ~0.6σ across ℓmax; non-BNT (grey)
  rising with ℓmax (and rising faster at larger area — bigger area → more S/N → tension crosses 0.3σ at
  lower ℓmax). Sanity vs 14000: BNT crosses 0.3σ around ℓ~620, non-BNT around ℓ~450.
- Error bars = std over the 5 training-seed runs (estimator variance), as in the 14000 plot.
- nσ from tensiometer Gaussian Q_DM on the 3-param (Ωm,S₈,w₀) subset (`scripts/tension/estimators.py`).
- `--dry-run` job count sane before launching; monitor with `monitor_bnt_bin1.py`.

## Env / pointers
- NPE sweep → **jaxili**; tension + plots → **aname** (tensiometer/getdist). Processing (if any) → cosmostat_new.
- Refs: `docs/PLAN_tension_submean_l37.md` (the parent campaign, full record), memory
  [[bar-impact-tension-env]], [[bnt-on-spectra-validated]] (BNT bin-1 localization + the M C Mᵀ shortcut).
- Caveat from the parent campaign: nlb=4 × rebin=10 gave 40-ℓ bins (coarser than the paper's nlb1 10-ℓ);
  keep binning consistent with the 14000 BNT plot so areas are like-for-like.
