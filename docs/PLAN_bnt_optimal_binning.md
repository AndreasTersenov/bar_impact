# PLAN — optimal BNT data-vector presentation per footprint, then remake the σ-vs-cut figure

> **SUPERSEDED (2026-08-01).** This document predates the settled BNT scale-cut result and repeats
> the **retracted** claim that the BNT data vector is ill-conditioned. Measured, it is *better*
> conditioned than non-BNT (correlation cond 8.3e2 vs 4.4e3), and the raw score is 1.2e4, not the
> "~1e8" quoted here. The real cause of the raw-flow failure is **information dilution**. Read
> `NOTES_bnt_compression_for_paper.md` §1 and `HANDOFF_BNT_SETTLED.md` instead; kept for history.

**Owner:** overnight autonomous run on **GPU 2**, while you sleep.
**Origin:** the full-sky BNT contours are "trash" (bloated 2.3–4.8× vs non-BNT) because the raw
NDE under-extracts the high-dimensional, ill-conditioned BNT data vector. Coarsening the ℓ-binning
is the cheap, direct lever (full-sky vector 980→240 features at rebin=40), and the early signal is
clear: **rebin=40 already halves the full-sky BNT contours** (σ_Ωm 0.034→0.016, σ_S₈ 0.069→0.033).
This plan finds the *optimal* binning per footprint and remakes the headline figure with it.

## Goal
For each footprint (6 masked + full sky), find the ℓ-rebin that gives the **best-extracted BNT
bin-1 posterior** (tightest, still-unbiased contours), then remake the 6-panel masked figure
(`plots/nsigma_vs_upper_cut_bnt_bin1_allareas.png`) + the full-sky panel using that optimal BNT —
testing whether optimally-presented BNT bin-1 **beats** standard non-BNT cut-all after the cuts.

## Gate (checked first; do NOT burn the night on a dead hypothesis)
The rebin=40 run is finishing now. Compute the **fair, same-binning** ratio σ(BNT r40)/σ(non-BNT r40)
at ℓmax=1020 and 500.
- **PASS** (proceed): binning clearly helps — mean σ-ratio drops below ~1.5 (from 4.5), i.e. the
  full-sky BNT contour is no longer pathological. (Early data already ~halves it.)
- **PARTIAL** (proceed, flag): improves but the optimum may be coarser than 40 — the Phase-1 scan
  will find it.
- **FAIL** (stop + report): even the best rebin leaves the ratio >2.5 → binning alone is
  insufficient; the residual is the nulled-direction conditioning, which needs score/MOPED
  compression, not bins. Write that up and stop; do not run Phase 2.

## Phase 1 — per-footprint rebin optimization  (~1 h, ~105 null-only NPE jobs)
For each of the 7 footprints, scan **rebin ∈ {10, 20, 40, 60, 80}** (refine ±one step if the optimum
sits at a grid edge), training the **BNT bin-1 NULL** posterior at **ℓmax=1020** (the most-features /
hardest case), **3 runs** each.
- **Metric:** FoM₃ = 1/√det(Cov₃) of the (Ωm, S₈, w₀) null covariance (higher = tighter), **subject
  to the null being unbiased** (|mean − truth| < 1.5σ per param). `rebin* = argmax FoM₃`.
- Expect a U-shape: too fine → NDE under-extracts (wide); optimal → tight; too coarse → real info
  lost (wide). Log FoM-vs-rebin per footprint; if monotone (no interior min), pick a conservative
  mid value and flag.
- **Decision recorded per area**, e.g. `fullsky: rebin*=60`, `14000: rebin*=10`, …

## Phase 2 — final BNT sweep at rebin*  (adaptive; ≤ ~1260 jobs / ~7.5 h, budget-capped)
For each footprint with **rebin* ≠ 10**: run the full BNT bin-1 **tension-vs-cut** sweep — 18 step-40
cuts × {null, biased} × 5 runs — at rebin*. Footprints with rebin*=10 **reuse the existing** rebin=10
posteriors (no re-run).
- **Budget guard (no silent caps):** if projected re-run > ~4.5 h, keep full-sky + the largest-gain
  footprints at 5 runs and drop the marginal (small-area, small-gain) ones to 3 runs, **logging
  exactly what was reduced/deferred**. Likely outcome: full sky definitely re-runs; most masked
  areas are already near-optimal at rebin=10 (BNT there was only 1.2× inflated and already beats
  non-BNT after cuts), so Phase 2 is probably closer to ~1–3 h than the 7.5 h ceiling.

## Phase 3 — remake figures + validate  (~15 min)
- Recompute BNT tension at rebin* per area; rebuild **`nsigma_vs_upper_cut_bnt_bin1_allareas`** (6
  masked panels) + the **full-sky panel**, now "optimal-BNT vs standard-non-BNT", annotating each
  panel with its rebin*.
- Regenerate contour overlays at rebin* for full sky + 14000 as a visual check.
- Per area: confirm the null recovers truth; report **where optimal-BNT now beats non-BNT after the
  baryon-safe cut**. Honest reporting if any footprint doesn't improve.
- Flag full TARP/SBC calibration of the chosen optima as a **follow-up** (too heavy for the loop).

## Decisions baked in (veto any before approving)
1. **Optimize BNT only; keep non-BNT at its standard rebin=10.** Non-BNT is well-conditioned;
   coarsening only loses it information, so rebin=10 is already its optimum. Comparison = best-BNT
   vs standard-non-BNT (the actual science question).
2. **One rebin\* per area**, optimized at ℓmax=1020 and applied to all cuts (conservative — mildly
   over-binned at low cuts, where it matters least).
3. **Metric = FoM₃ subject to unbiased null.** Not "tightest at any cost" (guards over-confidence).
4. **rebin only** as the knob this round. `--bnt-cross-abs` (abs of the nulled cross-spectra) is a
   plausible secondary knob; left as a fast follow-up if rebin's optimum still trails non-BNT.
5. **GPU 2, 5 jobs/gpu, mem 0.15** (proven). Fully resumable; every output carries a `_r{N}` tag so
   nothing collides with the shipped rebin=10 deliverables. Nothing committed.

## How I'll know it worked (back-pressure)
- Phase 1: clear interior FoM optimum per area; null unbiased at rebin*.
- Phase 2: sweeps finish 0-failed, QA clean.
- Phase 3: optimal-BNT contours ≤ the rebin=10 BNT contours (tighter), and the remade panels show
  BNT at/below non-BNT after the cuts. A per-area `rebin*` + FoM table written to the campaign dir.

## RESULTS (live)
- **Gate PASSED** (rebin=40): full-sky BNT/non-BNT σ-ratio 4.5→2.0 (ℓmax1020), 2.8→1.46 (cut500).
- **Phase 1 scan** (FoM₃ of BNT null, ℓmax=1020): coarser binning helps *every* footprint.
  Full sky has a clean interior optimum (FoM 4k→51k→92k→**136k**(r60)→132k(r80)). Masked FoM rises
  monotonically to r40 (the grid edge).
- **Calibration guard** (BNT σ must stay ≥ non-BNT σ at ℓmax=1020, since the rotation is lossless):
  at r40 every masked area has BNT σ ≥ non-BNT σ in all 3 params → well-extracted, NOT over-confident.
  So **rebin\*: masked = 40, full sky = 60.** (Didn't push masked to 60/80: r40 is already converged
  near the non-BNT floor; coarser would risk crossing it.)
- **Consequence:** properly-extracted (tighter) BNT contours mean the SAME bias now reads as HIGHER
  tension — the remade BNT curves will sit ABOVE the rebin=10 ones. The original near-zero BNT
  tension was partly the under-extraction hiding the bias. The remade figure is the honest version.
- **Phase 2 running** at rebin\*, 18 step-40 cuts, **3 runs** (overnight budget; extensible to 5),
  all 7 footprints. Non-BNT stays at rebin=10 (its own optimum).

## FINAL RESULTS (2026-06-25)
**Lossless validation (BNT − non-BNT tension at no-cut, same binning; must be 0 if fully extracted):**
- Masked r40 % extracted: 2000=87, 5000=85, 10000=82, 14000=76, 28000=93, 35000=93 (residual 7-24%,
  worst at mid-area 14000). rebin=10 was ~25-75% short → r40 is a big correction but not complete.
- Full sky r60: BNT 1.10 vs non-BNT 3.83 = **29% extracted** (was 0.05 = ~1% at r10; 22× better but
  binning can't finish the 10-ℓ healpy vector → needs score/MOPED compression).

**Scientific correction (the headline):** the shipped rebin=10 BNT figure substantially OVERSTATED
BNT bin-1's baryon mitigation — the "very low BNT tension" was largely NPE under-extraction bloating
the contours and hiding the bias. Properly extracted (r40), BNT tension is 2-6× higher and CONVERGES
to non-BNT at no-cut (the lossless identity), exactly as it must. At large areas BNT bin-1 barely
beats non-BNT; cutting bin-1 alone leaves significant residual contamination in the full-scale bins
2-4 at high S/N. BNT's real benefit is much narrower than the original figure implied.

**Figures:** `plots/nsigma_vs_upper_cut_bnt_bin1_allareas_optimal.{png,pdf}` (masked r40, per-panel
%-extracted annotation), `plots/nsigma_vs_upper_cut_bnt_bin1_fullsky_optimal.{png,pdf}` (r60, caveat).
Tables: `…/tables_r40/`, `…/tables_r60/`. rebin scan + lossless legs logged in the campaign logs.

**Caveats / open:** 3-run means (noisy); no TARP/coverage yet; r40 coarsens cut resolution (staircase);
masked residual 7-24% not closed (would need coarser bins — risk overshoot — or compression). The
at-cut comparison is best-vs-best (BNT r40 vs non-BNT r10, different binning).

## Deliverables
Remade `nsigma_vs_upper_cut_bnt_bin1_allareas.{png,pdf}` + full-sky panel (optimal BNT); per-area
`rebin*`/FoM table + FoM-vs-rebin logs; refreshed contour overlays; updated
`docs/HANDOFF_bnt_bin1_other_areas_PROGRESS.md` + memory. A morning summary of what changed.
