# Paper notes & results — central index

Single place to find the paper-relevant results and where each is documented in detail. Numbers below
are pulled from the per-topic docs/memory and should be checked against the linked source before they go
into the manuscript. **Status:** ✅ settled / paper-ready · 🔄 in progress · 📝 plan (not yet run).

Last updated: 2026-06-25.

---

## 0. Headline results (paper-ready)

- **Masked HOS need footprint-mean subtraction (submean); once applied, constraining power scales with
  area as expected for every statistic.** The masked peak counts were anomalously flat without it; submean
  restores peak scaling to match L1 and the Fisher. → §1.
- **Baryon-bias tension vs scale cut:** the ℓmax at which the null↔biased tension crosses 0.3σ scales with
  survey area (≈ℓ900 at 2000 deg² → ≈ℓ380 at 35000 deg²), on monopole-subtracted PS from ℓ=37. → §2.
- **BNT localizes the baryon bias to tomographic bin 1**, and a bin-1 scale cut suffices over a
  footprint-dependent range. Reliability caveat: raw NPE inflates BNT contours, so flat BNT nulls are
  inference-limited, not information loss (BNT is lossless) — re-derived with score compression. → §4.
- **Proper sim-based Fisher:** BNT gives ≈**2.5× FoM₃** ("true information") over non-BNT; realized NPE
  advantage is currently ≈1.5×, gap under investigation. → §5.

---

## 1. Masked-statistic correctness: submean + the coarse scale + scaling validation

**Status ✅ (this thread is essentially closed).**

- **Masked peaks were anomalously "too tight"** under masking (data-level σ-vs-area slope ≈ −0.08 vs L1
  −0.4). Root cause: the cosmology-dependent **footprint mean** creates a boundary step that leaks into
  the detail scales (spurious local maxima). **Fix:** subtract the footprint mean before the starlet
  transform (`--submean`). Reprocessed all 6 masked footprints + full-sky.
  → `HANDOFF_masked_peaks_submean.md`, memory `masked-peak-counts-too-tight.md`.
- **L1 norm:** submean is essentially a **no-op for L1** (it never had the peak pathology — the boundary
  step makes spurious *maxima*, which peaks count but the L1 magnitude-sum ignores). Submean only modestly
  *tightens* the smallest footprints (σ(S₈) 0.047→0.028 at 2000 deg²); large footprints unchanged. Verified
  submean-vs-non-submean L1 directly (contours overlap).
- **Coarse starlet scale: dropped from the analysis.** It's baryon-safe but ≈neutral (adds no constraining
  power as binned) for both peaks and L1; full-sky it needs submean even to be non-zero. **Paper analysis
  uses scales1234 (coarse excluded).**
- **Scaling validation (the capstone):** σ(S₈)/FoM₃ vs area, all three statistics, seed-averaged —
  peaks **−0.38**, L1 **−0.41**, PS **−0.45** (σ-slope; **no cuts** — HOS scales1234, PS full range at
  the **ℓmin=37** low-ℓ-recovered floor, l37–1020). All three scale in lockstep at ≈−0.4; peaks/L1 match
  the data-level Fisher (−0.35/−0.38). The masked-peak anomaly is gone. Bonus: the PS slope *depends on*
  the ℓmin=37 recovery (ℓmin=100 → spuriously shallow −0.29), so the plot also demonstrates the low-ℓ
  recovery's value.
  → **`scaling_vs_area_submean.md`** + plot `outputs/plots/submean_masked_peaks/scaling_vs_area_all_stats_submean_seedavg.png`.
- **NPE robustness fixes (enabling the above):** the `train_with_nan_retry` was inert (dict-vs-attribute
  bug → silent prior-collapsed posteriors); fixed in both NPE scripts. Root NaN trigger = near-constant
  bins blowing up under z-scoring; fixed with a relative zero-variance filter (l1 script). After fixes the
  l1 NPE converges 28/28 with 0 NaN. → memory `npe-nan-retry-was-inert.md`. **Any NPE posterior made
  before these fixes (2026-06-20) could be a silent collapse — check σ(param) vs prior.**

## 2. Baryon-bias tension vs scale cut (monopole-subtracted PS)

**Status ✅ (campaign done 2026-06-20).** Null (nobaryons) vs biased (baryonified) Q_DM tension vs ℓmax
scale-cut, monopole-subtracted PS from ℓ=37, all 6 footprints × step-40 cuts × 5 NPE seeds (error bars =
training-seed variance). Result: the **0.3σ crossing ℓmax scales with area** (2000:ℓ900 → 35000:ℓ380).
New module `scripts/tension/`; reproduces paper CSVs to 5e-6. Tables + 6-panel plot
`plots/nsigma_vs_upper_cut_masks.{png,pdf}`. Caveat: nlb4×rebin10 = 40-ℓ bins (4× coarser than the
paper's nlb1 10-ℓ → magnitudes not like-for-like). Open: observation-noise error bars (deferred).
→ `PLAN_tension_submean_l37.md`.

## 3. Low-ℓ PS recovery + PS-vs-HOS

**Status 🔄 (handoff active).** All 6 masked footprints reprocessed to monopole-subtracted nlb=4/lmax1535
PS; null plots (full-sky + 6-mask × full-ℓ/baryon-safe × {PS, l1, peaks}) in
`outputs/diagnostics/lmin_compare/`. Full-sky-without-monopole done (PS ℓ2 replaces ℓ0; full-ℓ σ(S8)
ℓ2=0.011, inside ℓ37=0.014, no collapse). Resume = masked-peaks (now done, see §1), baryon-bias panels,
full-sky peaks scales1234 + seed-averaging. → `HANDOFF_lmin_recovery_PS_vs_HOS.md`.
Constraining-power side (Fisher): HOS ≫ PS; PS+HOS super-additive on w0; low-ℓ recovery robust (×1.9
full-ℓ / ×3.8 baryon-safe), but HOS magnitude is linearization-limited → NPE decisive. → memory
`fisher-ps-vs-hos-constraining-power.md`, `ps-vs-hos-w0-degeneracy-real.md`.

## 4. BNT (nulling tomographic cross-correlations)

**Status 🔄.**
- **BNT on the (masked) power spectra:** C̃ = M C Mᵀ on the produced PS grids is exact (oracle 1e-11;
  masked-MASTER commutation — a gap the BNT literature leaves open). → `BNT_on_spectra.md`, memory
  `bnt-on-spectra-validated.md`.
- **Baryon bias localizes to BNT bin 1.** Extended to all 6 masked areas + full sky; the bin-1 cut value
  scales with area (unneeded ≤5000, helps 10–14k, insufficient ≥28000).
  → `HANDOFF_bnt_bin1_other_areas_PROGRESS.md`, memory `bnt-bin1-tension-all-areas.md`.
- **Reliability finding (important):** raw NPE **inflates BNT contours** (masked 1.2–1.6×, full-sky
  2.3–4.8×), so the flat BNT nulls are **inference-limited, not information loss** (BNT is lossless) →
  must re-derive nσ with score/MOPED-compressed NPE.
- **Score/MOPED compression:** MLE-form NPE works + is calibrated, recovers the Fisher per-param σ;
  realized BNT advantage ≈**1.5× FoM₃** (σS8 ratio 0.83), short of the Fisher 2.5×. Twist: BNT NPE ==
  Fisher; the gap is non-BNT NPE being *tighter* than its Fisher. → memory `bnt-npe-score-chase.md`,
  `NOTES_bnt_compression_for_paper.md`.
- **Active (2026-06-25):** fold score compression into the BNT bin-1 tension-vs-cut plot (start 14000) so
  the BNT tension reflects properly-extracted contours. → `HANDOFF_score_tension_foldin.md`,
  `PLAN_score_bnt_tension_14000.md`, `PLAN_bnt_optimal_binning.md`.
- Compression method plans: `PLAN_bnt_neural_compression.md` (VMIM-MLP), `PLAN_bnt_npe_whitening.md`.

## 5. Fisher forecasts (proper)

**Status ✅ (audit done).** Proper sim-based Fisher (local order-2 Jacobian + analytic-NaMaster-Gaussian/
hybrid covariance, all validated, oracle 2e-13): BNT-580 / non-BNT-460 ≈ **2.5× FoM₃** ("true info").
Error budget: 0.34 biased / 0.45–0.48 true / 0.72 finite-sim / 0.79 NPE. → `PLAN_fisher_proper.md`,
`PLAN_fisher_audit.md`, memory `fisher-proper-audit.md`.

---

## Environments & gotchas (for reproducing)

- **tensiometer** only in the `aname` conda env; **pymaster** only in `cosmostat_new`; **jaxili** NPE env
  is GPU. Three-env split. → memory `bar-impact-tension-env.md`, `bar-impact-namaster-venv.md`.
- NaMaster **nlb=1 is ill-conditioned at low f_sky → use nlb=4**. → memory `nlb1-decoupling-breaks-small-masks.md`.
- Paper full-sky = **healpy**, not NaMaster full-sphere; don't compare across estimators. → memory
  `fullsky-two-pipelines-and-lowell-gain.md`.
- Before any big reprocess: pre-flight checks (scope, write↔read filename match, row-count gate, partials,
  disk, test the gate). → memory `check-before-big-runs.md`.
