# HANDOFF — score-compressed BNT bin-1 tension-vs-cut (start at 14000 deg²)

> **SUPERSEDED (2026-08-01).** This document predates the settled BNT scale-cut result and repeats
> the **retracted** claim that the BNT data vector is ill-conditioned. Measured, it is *better*
> conditioned than non-BNT (correlation cond 8.3e2 vs 4.4e3), and the raw score is 1.2e4, not the
> "~1e8" quoted here. The real cause of the raw-flow failure is **information dilution**. Read
> `NOTES_bnt_compression_for_paper.md` §1 and `HANDOFF_BNT_SETTLED.md` instead; kept for history.

**Read this first, then `docs/NOTES_bnt_compression_for_paper.md` for the conceptual frame.**
This continues a long investigation (2026-06-24/25). Fresh-session handoff.

## The one-sentence goal
Reproduce the masked BNT-bin-1 **tension-vs-scale-cut** plot
(`plots/nsigma_vs_upper_cut_bnt_bin1_allareas_optimal.png`, the blue = BNT-bin-1, grey =
non-BNT-cut-all) **but with the BNT spectra score/MOPED-compressed**, so the BNT tension reflects
*properly extracted* contours. **Start with 14000 deg² only.**

## Why this matters (the scientific point)
The tension is `nσ ≈ bias / σ`. In the current plot the BNT bin-1 contours are **under-extracted
(inflated σ)** — the raw NDE can't handle the ill-conditioned BNT data vector (see
`docs/NOTES_bnt_compression_for_paper.md §1`). So the baryon **bias is real but its significance is
artificially suppressed by the inflated variance**. Today's binning hack (rebin=40) only got 76%
of the way at 14000 (the "no-cut % extracted" annotation). The **score/MOPED compression recovers
the full information** (verified: it recovers the Fisher σ, calibrated — `bnt-npe-score-chase`), so
the score-BNT tension should be **higher** than the inflated version — revealing the true bias
significance of cutting only bin-1. That is the honest version of the plot.

## What already exists (DO NOT rebuild)
The score pipeline is **built and validated at a single fixed cut** (`outputs/score_experiment/`,
plan `~/.claude/plans/jolly-toasting-robin.md`, memory `bnt-npe-score-chase`). Crucially:
- The experiment's **"BNT-580" IS the BNT bin-1 config**: `--bnt --bnt-bins 0,1,2,3 --upper-cuts
  580,1024,1024,1024` = bin1@580, bins2-4@1024. So the machinery already does BNT bin-1 — just at
  ℓmax=580, rebin=20, 14000, **null only**.
- Pipeline (works, calibrated TARP/SBC): worker `--dump-cache` (RAW theta,x,x_fid) →
  `scripts/score_compress.py <area> hybrid` (builds J=local-order-2 + C=hybrid, forms MLE summaries
  `θ̂ = θ_fid + F⁻¹JᵀC⁻¹(x−μ)`, 6 stats, MOPED-lossless oracle) → `scripts/npe_on_summary.py`
  (NPE on the 6 summaries, 5 seeds, samples at `y_fid`, TARP/SBC).
- Validated result: BNT-bin1@580 FoM₃ = 165k vs non-BNT-460 = 113k → **1.46× advantage, calibrated**
  (vs whitening 0.96× = no advantage). Contours: `outputs/score_experiment/fisher_vs_npe_contours_14000.png`.

## What is NEW (the work to do)
Three gaps between "fixed-cut FoM" and "tension-vs-cut":

1. **Parameterize the cut.** The fisher modules are **area-parameterized but cut-HARDCODED**
   (`fisher_gaussian_cov.py`: *"all share the same upper cut in the full config"*; J/C read
   `FISHER_AREA` env at import). The bin-1 cut must vary over the 18 step-40 cuts (340…1020 =
   `[c,1024,1024,1024]`). **Recommended efficient path:** dump-cache **once** at the *full* BNT
   vector (bin1@1024), build J and C **once** at full, then for each cut `c` apply a **selection
   operator** `S_c` (keep auto1[:c], autos2-4[:1024], cross-1j[:c] (x-cut min rule!), cross-jk[:1024])
   → `J_c=S_cJ`, `C_c=S_cC S_cᵀ`, `t_c = F_c⁻¹J_cᵀC_c⁻¹ x_c`. Avoids re-dumping/re-building per cut.
   (Alternative: re-dump + rebuild J,C per cut — correct but ~18× heavier. Either way the **BNT
   cross x-cut = min(c,1024)** must be respected, matching the master worker fix.)
2. **Add the biased (baryonified) fiducial.** Today's score runs only the null. Tension needs the
   posterior at the **nobaryons** fiducial (null) AND the **baryonified** fiducial (biased), both
   from the *same* nobaryons-trained score-NPE. Dump `x_fid` for both fiducials (worker
   `--fiducial-type {nobaryons,baryonified}`), compress both with the same J,C, sample the trained
   NPE at both summaries. (`npe_on_summary.py` currently samples one `y_fid` — extend to two, or run
   twice with the two summaries.)
3. **Compute Q_DM tension.** Apply the existing `scripts/tension/estimators.py` (Gaussian Q_DM,
   3-param Ωm,S8,w0) to the (null, biased) score-NPE posterior pair per cut → nσ vs cut. Mirror the
   error bars (5 training seeds).

## Concrete first-session plan (14000 only)
1. **Sanity / orient:** read `score_compress.py`, `fisher_{local_jacobian,hybrid_cov,gaussian_cov}.py`,
   `npe_on_summary.py`, `run_score_npe_sweep.sh`. Confirm the dump-cache + J + C + score path runs
   for BNT-bin1@580 at 14000 (reproduce one point).
2. **Pilot 2-3 cuts** (e.g. 460, 700, 1020): build the per-cut score (slice path), run null+biased
   score-NPE (3 seeds first), compute Q_DM tension. **Gate:** the no-cut (1020) score-BNT tension
   should be ~equal to the non-BNT tension at 1020 (the lossless identity — at no cut BNT≡non-BNT),
   and each tension should be ≥ the inflated rebin=40 value at that cut. Verify TARP/SBC pass.
3. **Full sweep:** 18 step-40 cuts × {null,biased} × 5 seeds at 14000. Compute tension, plot blue
   (score-BNT-bin1) over grey (standard non-BNT-cut-all, reuse `ps_submean_l37` tables — non-BNT is
   binning-insensitive, no score needed). Compare to the inflated `*_optimal.png` 14000 panel.
4. **Validate + write up:** TARP/SBC per cut; null recovers truth; the score-BNT tension curve vs
   the inflated one (expect higher); state where BNT bin-1 actually keeps tension < 0.3σ once
   properly extracted.

## Key decisions / gotchas
- **rebin = 20** for the score input (matches the validated J/C feature space). Score is lossless so
  the input rebin mainly sets J/C conditioning; don't change it without rebuilding J/C consistently.
- **Non-BNT grey** stays the standard analysis (rebin=10, `ps_submean_l37` tables) — it's
  well-conditioned, needs no score. Comparison = score-BNT-bin1 vs standard-non-BNT-cutall (same as
  the original plot, just blue properly extracted).
- **Calibration is not automatic** — run TARP/SBC at every cut; a tight contour that fails coverage
  is not a win (this killed VMIM). The score path was calibrated at 580; re-verify across cuts.
- **The biased summary** uses the nobaryons J,C on baryonified data — that is correct (the model is
  nobaryons; baryons are the contamination). Don't rebuild J,C on baryonified.
- **Lossless identity = your free sanity check** at every step: at no cut (ℓmax=1020), BNT and
  non-BNT must give the same tension and FoM.

## Environments (titan, GPU 2) — three-env split
- NPE / score / dump-cache / TARP → **jaxili** (`/home/tersenov/anaconda3/envs/jaxili/bin/python`).
- Analytic NaMaster Gaussian cov (if rebuilding C) → **cosmostat_new** (pymaster). [[bar-impact-namaster-venv]]
- Q_DM tension / getdist contours → **aname** (tensiometer). [[bar-impact-tension-env]]

## Context from this session (background, all DONE)
- **Optimal-binning campaign** (the inflated-contour fix via coarser bins): `docs/PLAN_bnt_optimal_binning.md`,
  figures `plots/nsigma_vs_upper_cut_bnt_bin1_{allareas,fullsky}_optimal.png`. rebin*=40 masked / 60
  full-sky; recovers 76-93% (masked) / 29% (full-sky) of the lossless tension — i.e. binning helps
  but does NOT finish, which is *why* we go to score compression. memory `bnt-bin1-tension-all-areas`.
- **Why BNT breaks the NDE + terminology (whitening ≠ score) + "by construction" framing**:
  `docs/NOTES_bnt_compression_for_paper.md` (paper material). Key: whitening under-extracts BNT
  (0.96× = no advantage); only score recovers it (1.46×); compression is a no-op on non-BNT (control
  verified all 6 areas). The score result is a *calibrated realization* of the Fisher-predicted
  advantage, not a discovery; honest and defensible.
- **VMIM (learned neural compression) FAILED** (over-confident, off-truth) — `bnt-npe-score-chase`,
  `outputs/baryon_tension/vmim/`. The principled successor was score/MOPED. If revisiting a learned
  compressor later: anchor it to the score projection (over-complete), RealNVP not MAF on the small
  summary, TARP in the loop.
- **Fisher reference**: `fisher-proper-audit` — BNT 2.5× FoM3 "true info"; score realizes ~1.5×, gap
  is non-BNT-side (non-BNT NPE tighter than its Fisher), not BNT under-extraction.

## Pointers
- Score outputs/caches: `outputs/score_experiment/{cache,score,npe_score,npe_whiten,contours}/`.
- Scripts: `score_compress.py`, `npe_on_summary.py`, `run_score_npe_sweep.sh`,
  `diagnostics/fisher_{local_jacobian,gaussian_cov,hybrid_cov}.py`, `diagnostics/npe_fom_from_samples.py`.
- Tension infra (reuse for Q_DM + plotting): `scripts/tension/` (estimators, aggregate),
  `scripts/build_bnt_bin1_allareas_plot.py`.
- Plan that built the score pipeline: `~/.claude/plans/jolly-toasting-robin.md` (verification oracles).
- Memories: `bnt-npe-score-chase`, `bnt-bin1-tension-all-areas`, `bnt-on-spectra-validated`,
  `fisher-proper-audit`, `bar-impact-tension-env`, `bar-impact-namaster-venv`.

## Open after this (not the first session)
Other 5 masked areas + full sky (repeat at each); the production tension figure rebuild; the
baryon-bias-robustness of score-BNT under baryonification (does the null stay unbiased — separate
secondary check); over-complete compression for any non-Gaussian gain.
