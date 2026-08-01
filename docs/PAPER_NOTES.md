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
- **BNT localizes the baryon bias to tomographic bin 1, so the scale cut can be targeted at that bin
  alone** — retaining 92 of 120 bandpowers against 50 for a uniform cut at the same ℓmax, and giving
  **1.47× the 3-param FoM** (19% tighter σ(S₈)) at 14000 deg², matched ℓmax=460, both arms calibrated
  and on-truth (BNT marginally at the 0.3σ threshold and tolerated on its error bar; non-BNT safe at
  0.17σ). Real but not dramatic. This is *information retention*, **not** better baryon control —
  BNT actually crosses 0.3σ at a lower ℓmax than a uniform cut. Requires MOPED compression: fed the
  raw BNT vector, a flow loses the Ωm–S₈ degeneracy entirely (r = −0.03 vs −0.9) and the comparison
  inverts to 0.33× — because nulling dilutes the information across many weak modes, NOT because the
  vector is ill-conditioned (measured: it is better conditioned than non-BNT). → §4.
- **Proper sim-based Fisher:** at the same matched cut the information-level BNT gain is **2.10×**; the
  calibrated NPE realizes **1.47×**. The shortfall is density estimation, not physics — and it comes
  from both ends (BNT realizes 0.81 of its Fisher; non-BNT lands 1.16× tighter than its own). → §5.

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

**Status ✅ settled 2026-07-31.** Flagship figures:
`paper/figures/bnt_flagship_matched_c460_{pooled,single_seed}/`. Full record and the numbers below:
`PLAN_score_bnt_tension_14000.md` (ADDENDUM 2026-07-31).

**The story, in one paragraph.** BNT isolates the baryon sensitivity into essentially a single
tomographic bin. For the power spectrum that means the scale cut can be *targeted* — applied to bin 1
alone, leaving bins 2–4 at full range — so less information is discarded than a uniform cut at the
same ℓmax. That yields tighter contours: real, but not dramatic. Extracting it requires compressing
the data vector first: nulling removes the dominant common mode and leaves differences, so the
information that the standard basis concentrates in a few high-S/N bandpowers is **diluted across
many individually weak modes**, and a normalizing flow fed the raw vector cannot learn the required
relative weighting from a finite simulation suite. (It is *not* ill-conditioning — measured, the BNT
vector is better conditioned than the standard one; see the table below.) The fair comparison is
therefore both arms at the **same ℓmax, same rebinning, same compression**, differing only in basis.

- **BNT on the (masked) power spectra:** C̃ = M C Mᵀ on the produced PS grids is exact (oracle 1e-11;
  masked-MASTER commutation — a gap the BNT literature leaves open). → `BNT_on_spectra.md`, memory
  `bnt-on-spectra-validated.md`.
- **Baryon bias localizes to BNT bin 1** — the premise, and it holds. Extended to all 6 masked areas +
  full sky; the bin-1 cut value scales with area. (Source doc is 100% NUL; memory
  `bnt-bin1-tension-all-areas.md` survives.)
- **FLAGSHIP — matched cut, 14000 deg², rebin 20, MOPED, 5 NDE seeds each:**

  | | σ(Ωm, S8, w0) | features | FoM₃ | baryon tension |
  |---|---|---|---|---|
  | BNT bin-1 @460 (bins 2–4 full) | 0.01597, 0.02759, 0.08665 | 92/120 | 1.637e5 | 0.30 ± 0.09σ |
  | non-BNT cut-all @460 | 0.01871, 0.03403, 0.08795 | 50/120 | 1.098e5 | 0.17 ± 0.03σ |

  **BNT/non-BNT = 1.47×** (19% tighter σ(S₈), 13% on Ωm). Both calibrated (SBC 0.28–0.29 vs the ideal
  0.289) and on-truth. **Bias at this cut is asymmetric and must be stated honestly:** non-BNT is
  comfortably safe (0.17σ), BNT sits *marginally at* the 0.3 threshold (0.304 ± 0.091) and is
  tolerated on its error bar (mean − σ = 0.21), not comfortably below it. BNT's own adopted cut is
  **420**. ℓmax=460 is chosen because it is the adopted cut of the main PS analysis
  (`ps_submean_l37`: 460 → 0.288σ, 500 → 0.413σ).

- **The gain is information-level; our estimator realizes part of it.** Fisher gives 2.10× at the same
  matched cut against the NPE's 1.47×. The shortfall is density estimation, not physics. It comes from
  both ends: BNT realizes 0.81 of its own Fisher, while non-BNT lands 1.16× *tighter* than its Fisher
  (non-Gaussian information the Gaussian bound cannot see, or a slightly conservative hybrid C — SBC
  says it is not over-confidence).

- **NOT a baryon-mitigation result — corrected 2026-07-31.** Earlier framing had BNT controlling
  baryons better. It does not: BNT bin-1 crosses 0.3σ at ℓmax **460** while non-BNT cut-all crosses at
  **620**, and at matched 460 BNT sits at 0.30σ against non-BNT's 0.17σ. The targeted cut trades
  slightly more residual bias for substantially more retained information, and wins on that trade. The
  old raw-NPE figure suggested otherwise because it under-extracted BNT (see below).

- **Compression is required, and the control proves it.** Same cut/rebin/seeds, flow fed the z-scored
  data vector instead of the 6 MOPED summaries: raw NPE on BNT returns r(Ωm,S₈) = **−0.03** where the
  physical lensing degeneracy is ≈ **−0.9**. Its *marginals* are fine — tighter than MOPED's — so the
  failure is in the **degeneracy structure**, inflating the 3-param volume 3.6×. **SBC and TARP cannot
  see this** (both test marginal rank uniformity per parameter, and the posterior is calibrated and
  on-truth). Consequence: BNT/non-BNT is **1.47× under MOPED and 0.33× under raw** — the sign of the
  conclusion flips on the compression, which likely explains the contradictory historical BNT results.

- **WHY the flow fails — measured, with the alternatives excluded** (`why_compression_is_needed.py`,
  `outputs/diagnostics/why_compression_is_needed.csv`). The paper needs this, and the standard
  explanation is wrong:

  | explanation | verdict |
  |---|---|
  | ill-conditioning | **WRONG.** BNT correlation-matrix cond **8.3e2** vs non-BNT **4.4e3** — BNT is *better* conditioned on what a z-scored flow sees. (`NOTES_bnt_compression_for_paper.md` quotes ~1e8 for the raw score; measured 1.2e4.) Do not repeat this claim. |
  | dynamic range / sign changes | real (24× range, 29/92 negative) but **irrelevant** — z-scoring removes both before the flow sees anything |
  | dimension (92 vs 50) | **EXCLUDED** by control: raw non-BNT at rebin 10 = **100 features** keeps r = −0.946 (vs −0.947 at 50) and *improves* FoM₃ 1.39e5 → 1.58e5 |
  | **information dilution** | **SUPPORTED — this is the explanation** |

  Fraction of Fisher information in the highest-S/N **10%** of features:
  σ(S₈): **0.64 non-BNT vs 0.05 BNT**; Ωm: 0.67 vs 0.07. Median per-feature S/N falls **73 → 6** (S₈);
  for w₀ only **39%** of BNT features reach S/N > 1. Nulling removes the dominant common mode and
  leaves differences, so the large amplitudes cancel and the signal survives only in small residuals
  spread across many modes. A flow conditioned on the raw vector must learn the correct *relative*
  weighting of ~90 individually weak features from a finite simulation suite; small errors in that
  weighting damage the joint structure far more than the marginals — exactly the observed failure.
  MOPED supplies the weighting analytically as C⁻¹J F⁻¹.

- **Binning is a side issue, measured.** MOPED at rebin 20/10/5/2/1 (matched @460, NPE trained and
  calibrated at every rung): Fisher predicts 2.1–2.6× going to native, the posteriors deliver +3%
  (BNT) and +11% (non-BNT), and native is *worse* than r20 for BNT. rebin=20 is production.
  → `outputs/diagnostics/score_rebin_ladder_fom.csv`.

- **Three ratios, three questions — do not conflate:** 1.47× (BNT at equal cut — flagship);
  3.62× (compression rescuing the ill-conditioned vector); 1.20× (this pipeline vs standard raw
  analysis — what a referee asks for, and the most modest).

- **Retraction carried forward:** the shipped rebin=10 BNT figure overstated BNT's baryon mitigation;
  the honest version is rebin=40 (`nsigma_vs_lmax_bnt_bin1_allareas_optimal`). Its per-panel
  "% extracted" annotation is not reproducible from this repo. → `PLAN_bnt_optimal_binning.md`.
- Compression method plans: `PLAN_bnt_neural_compression.md` (VMIM-MLP), `PLAN_bnt_npe_whitening.md`;
  VMIM v2 (learned compressor) agrees with MOPED near the adopted cut and is the robustness check.

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
