# BNT data vectors and neural compression — methods notes (paper-relevant)

**Rewritten 2026-08-01.** The previous version of this file led with an ill-conditioning explanation
that is measurably **false**; it is corrected in §1 below and the retraction is kept explicit in §5 so
the claim does not get reintroduced from an older draft. Intended as raw material for the methods
section.

Sources: the settled analysis in `PAPER_NOTES.md` §4 and `HANDOFF_BNT_SETTLED.md`; the measurement in
`scripts/diagnostics/why_compression_is_needed.py` → `outputs/diagnostics/why_compression_is_needed.csv`.
All numbers below are 14000 deg², rebin=20, **matched** ℓmax=460, 5 NDE seeds, mean of per-seed FoM₃.

---

## 1. Why a flow fed the raw BNT vector fails — measured, with the alternatives excluded

The summary statistics are the tomographic auto + cross angular power spectra. The BNT transform is a
fixed nulling operator `M` on the tomographic bins, acting as `C̃ = M C Mᵀ` per multipole. It is an
**invertible linear map**, so it neither adds nor destroys information — a fact the no-cut control in
§4 confirms end to end.

Fed the rebin=20 BNT vector raw, a plain conditional RealNVP nevertheless fails, and fails in a
specific and initially silent way:

| configuration | r(Ωm, S₈) | det(R) | FoM₃ |
|---|---|---|---|
| raw plain flow, BNT | **−0.03** | 0.995 | 4.61e4 |
| raw plain flow, non-BNT | −0.95 | 0.021 | 1.39e5 |
| MOPED, BNT | −0.91 | 0.024 | 1.67e5 |

Weak lensing carries a physical Ωm–S₈ degeneracy near −0.9. The raw BNT posterior returns essentially
**uncorrelated** parameters while keeping plausible — in fact slightly tighter — marginals. The
failure is entirely in the *joint* structure, and it inflates the 3-parameter volume 3.6×. It also
**passes SBC and TARP**, which test marginal rank uniformity per parameter and are structurally blind
to a missing correlation (see §6).

### 1.1 Four candidate explanations, three excluded by measurement

| explanation | verdict |
|---|---|
| ill-conditioning | **WRONG** — BNT is *better* conditioned than non-BNT |
| dynamic range / sign changes | real but **irrelevant** — removed by z-scoring before the network |
| dimension (92 vs 50 features) | **EXCLUDED** by a direct control |
| **information dilution** | **SUPPORTED — this is the explanation** |

**Ill-conditioning is wrong.** On the quantity a z-scored flow actually sees — the correlation matrix
of the input — BNT measures condition **8.3e2** against non-BNT's **4.4e3**. BNT is better
conditioned by a factor of 5. The raw score `C⁻¹J` measures **1.2e4**, not the ~1e8 that earlier
drafts of this file asserted. Nothing here is near a numerical precision limit.

**Dynamic range and sign changes are real but irrelevant.** BNT does have a 24× wider
feature-amplitude range (1.7e3 vs 7.2e1) and 29 of 92 features with negative mean, since cross-spectra
are differences. The flow input is z-scored per feature, which removes both before the network sees
anything.

**Dimension is excluded by control.** Raw non-BNT at rebin 10 gives 100 features — *more* than BNT's
92 — and keeps r(Ωm,S₈) = −0.946 (against −0.947 at 50 features), while improving FoM₃ from 1.39e5 to
1.58e5. More features do not break the flow. The BNT *basis* does.

### 1.2 Information dilution — the explanation that survives

Nulling cancels the dominant common mode and leaves differences. The large amplitudes cancel; the
cosmological signal survives only in small residuals spread across many modes. So the information
that the standard basis **concentrates** in a few high-S/N bandpowers is **redistributed** almost
uniformly:

| | non-BNT | BNT |
|---|---|---|
| fraction of S₈ Fisher in the top 10% of features | **0.64** | **0.05** |
| fraction of Ωm Fisher in the top 10% of features | 0.67 | 0.07 |
| median per-feature S/N (S₈) | **73.4** | **5.8** |
| fraction of features with S/N > 1 (w₀) | 1.00 | **0.39** |

A flow conditioned on the raw vector must therefore learn the correct **relative weighting** of ~90
individually weak features from a finite simulation suite (16,965 sims). Getting those weights
slightly wrong damages the joint structure far more than the marginals — which is exactly the observed
failure. MOPED supplies that weighting analytically as `C⁻¹J F⁻¹`; an embedding network learns it
under the NPE loss; a plain flow on z-scored input does neither.

**This is the justification to give in the paper for why the data vector is not fed to the density
estimator raw.** It is a statement about where the information sits, not about numerical conditioning.

---

## 2. Terminology: whitening vs MOPED / score compression (they are NOT the same)

- **MOPED ≡ score compression.** Same construction: the score `t = JᵀC⁻¹(x−μ)`, used in its
  parameter-MLE form `θ̂ = θ_fid + F⁻¹JᵀC⁻¹(x−μ)` (6 numbers, in parameter units). For a (near-)
  Gaussian likelihood these are *sufficient* statistics — the compression is **lossless** (verified
  `Fisher(t) = Fisher(x)` to ~1e-10). The `F⁻¹` parameter-units rotation matters because it puts the
  six summaries on comparable scales; it is **not** rescuing a cond-1e8 pathology (the raw score
  measures 1.2e4).
- **Whitening is different and is NOT a compression.** It is a full-rank decorrelation `L⁻¹(x−μ)` that
  keeps *all* features — it rotates and decorrelates but does not reduce dimension, so the flow must
  still find the projection in the full space.

---

## 3. Three ways to make it work, and what the paper uses

All three address dilution; they differ in what they assume.

1. **Compress first (MOPED).** Hand the flow the analytic projection; it learns only a 6-dimensional
   density. Needs the analytic covariance and the local Jacobian, and is Gaussian-optimal by
   construction — measurably lossy where the flow can already cope (on non-BNT, raw NPE beats MOPED,
   1.39e5 vs 1.11e5).
2. **Bin coarsely enough to re-concentrate the signal.** Raw BNT at rebin 40 recovers r = −0.93 and
   matches MOPED's FoM. But it starves the standard vector, which drops to 20 features at rebin 40 —
   so it cannot be applied symmetrically to both arms, which is what a fair comparison requires.
3. **Give the density estimator an embedding network** (what the paper uses). A 16-dim MLP inside the
   flow, trained **jointly** with it under the same NPE loss. It needs no covariance and no Jacobian,
   so it is not restricted to Gaussian-optimal projections, and it is part of the density estimator
   rather than a separate analysis stage needing its own justification.

**Preprocessing decides whether the embedding works** — this is the single most load-bearing
implementation detail:

| extractor | ratio | BNT r(Ωm,S₈) | det(R) |
|---|---|---|---|
| MOPED | 1.470 | −0.909 | 0.026 |
| embedding + **analytic whitening** (`ana_whiten`) | **1.405** | −0.919 | 0.026 |
| embedding + z-score | 1.026 | −0.884 | 0.109 (partial) |
| raw plain flow | 0.331 | −0.025 | 0.995 (lost) |

Z-scoring leaves the noise correlated and inflates noise-dominated features to the scale of
signal-carrying ones. Whitening by the analytic `C⁻¹ᐟ²` first makes the noise isotropic, so the ~6
signal directions stand out. Same data, same cut, same flow architecture — only the preprocessing
differs, and it moves the answer from 1.03 to 1.41.

---

## 4. The controls that license the result

**The no-cut oracle.** With no scale cut at all, BNT and non-BNT give the same constraints:
FoM₃ 4.326e5 vs 4.420e5, **ratio 0.979**. This is the end-to-end confirmation that BNT is an
invertible linear map carrying identical information, and that the pipeline is not manufacturing an
advantage. Any BNT gain must therefore come from the interaction with the *cut*, not from the
transform.

**The non-BNT control on compression.** Applying the same compression to the non-BNT vector reproduces
the uncompressed posterior (FoM ratio score/whiten = 1.00–1.08, σ ratios 0.98–1.04, both on-truth,
contours coincide). Compression is a **no-op on non-BNT** and does not generically tighten contours;
it only rescues the diluted vector.

**Two extractors that share no failure modes.** MOPED needs Gaussianity, the analytic covariance and
the local Jacobian; the embedding needs none of them. They agree to 4% (1.470 vs 1.405).

---

## 5. What the result is — and the honest caveats

**The claim.** BNT localizes the baryon sensitivity to tomographic bin 1, so the scale cut can be
applied to that bin alone. Less information is discarded than by a uniform cut at the same ℓmax, and
the contours tighten by **~1.4×**. It is information **retention**, not better baryon control.

The cleanest statement is the decomposition:

```
              uncut      cut@460    cost of the cut
BNT          4.326e5     2.565e5        1.69x
non-BNT      4.420e5     1.826e5        2.42x
ratio of costs = 2.42/1.69 = 1.435   vs directly measured 1.405   (agree to 2%)
```

Both bases start from identical information; the cut costs the standard analysis 2.42× and BNT 1.69×.
**BNT does not add information; it loses less when you cut.**

**Caveats to state, not hide:**

1. **Bias at ℓmax=460 is asymmetric.** non-BNT 0.172 ± 0.027σ (safe); BNT 0.304 ± 0.091σ —
   *marginally at* the 0.3 threshold, tolerated on its error bar (mean − σ = 0.21). BNT's own adopted
   cut is 420; 460 is used because it is the adopted cut of the main PS analysis. Do not write "both
   unbiased at this cut."
2. **Gaussian-sufficiency (MOPED arm only).** Score is optimal only up to the Gaussian likelihood. The
   embedding arm does not carry this assumption, which is part of why it is the primary.
3. **The embedding exceeds the Gaussian Fisher** by 1.25–1.36× with a w₀ offset that scales with FoM.
   It is common-mode across both arms and cancels in the ratio, so it does not threaten the headline,
   but it is unresolved: it may be real non-Gaussian information or fiducial-local over-confidence.
   (The fiducial "observation" is the mean of 200 permutations while every training row is a single
   realization.)
4. **A w₀ offset of ≈ −0.025 is present in every method**, including the plain flow and MOPED. It
   predates this work and is not caused by BNT, the cut, or the embedding.
5. **Scope.** 14000 deg², rebin=20, matched ℓmax=460. This is the constraining-power methods result;
   it is distinct from the baryon-*bias*-mitigation question.

**Retracted claims — do not reintroduce from older drafts.** (i) "The BNT data vector is
ill-conditioned" — measured false, §1.1. (ii) "The raw score has condition number ~1e8" — measured
1.2e4. (iii) "The BNT vector is unsuitable for direct NPE" — it is suitable given an embedding network
and analytic whitening, which is what the paper now uses. Several superseded planning documents in
`docs/` still contain (i) and (ii); they carry a banner pointing here.

---

## 6. A methods finding worth stating in its own right

**TARP and SBC cannot see a wrong degeneracy.** Every configuration in §3 passes both — *including*
the raw plain flow whose posterior has r(Ωm,S₈) = −0.03 against the physical −0.9 (TARP dev 0.115,
SBC 0.285/0.282/0.290). The best TARP score in the entire set (0.0325) belongs to an embedding run
with a doubled w₀ offset. Both tests average coverage over the prior and test marginal rank uniformity
per parameter, so neither sees joint structure, nor local behaviour at the single fiducial where every
FoM is measured. **Passing SBC/TARP is not evidence that a posterior's degeneracy structure is
correct.**
