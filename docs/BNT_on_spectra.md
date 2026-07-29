# Applying BNT directly to the (masked) power spectra — prescription, literature check, caveats

**Status:** prescription verified against the BNT bibliography **and** the masked-harmonic
equivalence is now numerically **confirmed** by the oracle (§9, 2026-06-21): worst relative
diff between BNT-on-maps and M·C·Mᵀ = 1.1e-11 (machine roundoff) across 3 perms × 10 spectra ×
383 bandpowers. The no-rerun shortcut is proven for our pipeline (14000 deg², apod 2.0, lmax
1535, nlb 4, f_sky 0.339). Next: build BNT grids (§9.2), then the bin-1 scale-cut sweep (§9.3).

**Goal.** For the baryon-bias scale-cut tension analysis, apply the BNT (Bernardeau–Nishimichi–
Taruya) nulling transform to the *already-produced* monopole-subtracted auto+cross power-spectrum
grids, rather than reprocessing BNT-transformed maps through NaMaster. If valid, this turns a full
grid reprocess (thousands of maps × pymaster) into a cheap linear algebra step on existing arrays.
The motivating physics: BNT decorrelates the redshift bins so the baryonic contamination
concentrates in the lowest BNT bin, and we then only need to scale-cut the spectra involving that
bin to recover unbiased contours.

---

## TL;DR

- **Prescription (confirmed by the literature):** the BNT-transformed tomographic power spectrum is
  the quadratic form **C̃(ℓ) = M · C(ℓ) · Mᵀ**, i.e. `C̃_ab(ℓ) = Σ_ij M_ai M_bj C_ij(ℓ)`, summed over
  the original tomographic pairs (i,j). Autos *and* crosses are inputs; you cannot build the BNT
  data vector without the original cross-spectra.
- **Cross-spectrum scale cut (confirmed):** a BNT cross a×b is cut at the *more restrictive* of the
  two bins' cuts (the lower-z bin governs). For us → cut every spectrum involving BNT bin 1.
- **On our data (measured):** after BNT the high-ℓ baryon suppression collapses from
  +0.5/1.0/1.5/1.6 % across the original autos to **+0.5/0.1/0.1/0.1 %** across the BNT autos — the
  bias localizes to **BNT bin 1**, exactly as hoped.
- **The catch (the important caveat):** *no* BNT paper applies the transform to **masked, measured
  pseudo-Cℓ / MASTER-decoupled** spectra. They all BNT the maps/kernels before estimation, or work
  on theory Cℓ. x-cut deliberately went to configuration space to *avoid* mask deconvolution. So
  "apply M·C·Mᵀ to our decoupled masked grids" is **our own extension**, justified by a commutation
  argument (§6) but not citable — hence the oracle gate (§9) is mandatory, not optional polish.

---

## 1. The prescription

BNT is a constant linear transform on the tomographic convergence maps,
`κ̃_a(n̂) = Σ_i M_ai κ_i(n̂)`, with M a fixed 4×4 matrix (independent of ℓ and of sky position).
The angular power spectrum is bilinear in the maps, so the full N×N tomographic matrix at each ℓ
transforms as

```
C̃(ℓ) = M · C(ℓ) · Mᵀ          C̃_ab(ℓ) = Σ_ij M_ai M_bj C_ij(ℓ)
```

where `C_ij(ℓ)` is the 4×4 matrix carrying the 4 autos on the diagonal and the 6 crosses
off-diagonal. The BNT autos (diagonal of C̃) mix all original autos and crosses; the BNT crosses
(off-diagonal of C̃) are the transformed cross-spectra. **Nothing is discarded and nothing is
"cut" at this stage** — the cut comes later (§4).

## 2. Our BNT matrix and its structure

Hard-coded (identically) in all `bnt_*` scripts, e.g.
`scripts/bnt_cross_power_spectrum_processing_master.py:54`:

```
M = [[ 1.        ,  0.        ,  0.        ,  0.       ],
     [-1.        ,  1.        ,  0.        ,  0.       ],
     [ 0.4521097 , -1.4521097 ,  1.        ,  0.       ],
     [ 0.        ,  0.25127807, -1.251278  ,  1.       ]]
```

Structure checks against the literature (all pass): **lower-triangular, ones on the diagonal**
(x-cut Eq. 21 form), and **banded** — row 4 has a zero in column 1, i.e. each BNT bin a combines
only bins {a−2, a−1, a} (the review's banded 3-bin construction). Rows 1–3 of the *nulling* rows
sum to zero (mass-sheet / kernel nulling); row 1 is the untouched κ₁.

> ⚠️ The off-diagonal weights (0.4521…, −1.4521…, 0.2513…, −1.2513…) come from the source n(z)
> via the `n_i^0 = ∫dχ n_i`, `n_i^1 = ∫dχ n_i/χ` integrals. They are **n(z)-dependent** and were
> computed for the CosmoGrid source redshifts of this analysis. If the n(z) ever changes, M must be
> recomputed. (Not re-derived here; structure verified, values taken as given.)

## 3. Literature verification

| Paper | What it gives | Confirms |
|---|---|---|
| BNT 2024 review, [arXiv:2412.14704](https://arxiv.org/html/2412.14704), **Eq. 8** | `Ĉ_ab(ℓ) = p_i^a p_j^b C_ℓ^ij` (harmonic) | C̃ = M C Mᵀ exactly; autos+crosses transform identically; banded matrix |
| x-cut (baryons), [arXiv:2007.00675](https://arxiv.org/abs/2007.00675), **Eq. 22** | `ξ̃±^ij(θ) = M^ik ξ±^kl (Mᵀ)^lj` (config space) | same quadratic form; 4×4 lower-tri, ones on diagonal (Eq. 21) |
| k-cut, [arXiv:1809.03515](https://arxiv.org/abs/1809.03515) | redshift-dependent ℓ-cut after BNT (harmonic) | the per-bin ℓ-cut idea; ℓ ↔ k localization |
| Comparison of nulling methods, [arXiv:2512.15604](https://arxiv.org/html/2512.15604v2), **Eq. 10, 13** | `C^ij → Ĉ_ab`; `ℓ^ij(k) = k χ^ij` | matrix reorganization; per-bin-pair ℓ-cut |

Origin: Bernardeau, Nishimichi & Taruya (2014), arXiv:1312.0430.

**What is confirmed:** the prescription (C̃ = M C Mᵀ), the matrix form, that crosses are required
inputs and are kept, and the cross-spectrum cut rule (§4).

**What is *not* covered by any of them:** application to masked / pseudo-Cℓ / MASTER-decoupled
measured spectra (§7).

## 4. Cross-spectra: required inputs, and how they are cut

Two distinct questions, both answered by the literature:

1. **Do we need / keep the original crosses?** Yes — they are summed into every BNT auto
   (`C̃_aa = Σ_ij M_ai M_aj C_ij`), so the BNT data vector cannot be formed without them. The full
   transformed set (4 BNT autos + 6 BNT crosses) is retained. x-cut: "the majority of the
   information lies in the tomographic autocorrelation data points since the BNT transformation
   removes cross-bin correlations by construction" — but the crosses are *not discarded*.

2. **How is a BNT cross scale-cut?** At the **more restrictive of the two bins' cuts**.
   x-cut **Eq. 23**: for bins i,j, cut all `θ < min{x/d_A^i, x/d_A^j}` (the lower-z, smaller-distance
   bin dominates). Harmonic analog (comparison paper Eq. 13): `ℓ^ij = k χ^ij` with the lower bin
   setting ℓ_max. **For us:** only BNT bin 1 is contaminated, so cut {auto-1, 1×2, 1×3, 1×4} at bin
   1's ℓ_max and keep {auto-2/3/4, 2×3, 2×4, 3×4} at full range. This is the x-cut rule *and* the
   user's "cut bin-1's spectra" — they coincide here, not by approximation.

The principled general form is the **full triangular cut**: every BNT bin gets its own
`ℓ_max(a) = k_max · χ(z_a)`, cross a×b at `min(ℓ_max^a, ℓ_max^b)`. "Cut bin 1 only" is the special
case that applies *because* we measured bin 1 to be the only appreciably contaminated bin (§5).

## 5. Numerical findings on our data (14000 deg², monopole-subtracted, l_max 1535)

Computed by applying M·C·Mᵀ to the existing submean fiducial grids (`/tmp/bnt_on_spectra.py`):

**Baryon suppression (baryonified/nobaryons fiducial mean), high ℓ (600–1024):**

```
                 bin1    bin2    bin3    bin4
original autos:  +0.5%   +1.0%   +1.5%   +1.6%    (all biased, grows with bin)
BNT autos:       +0.5%   +0.1%   +0.1%   +0.1%    (bias localized to BNT bin 1)
```

Physical reading: BNT bin 1 is the untouched κ₁ (lowest z → smallest physical scale at fixed ℓ →
most baryon-sensitive); BNT bins 2–4 are difference combinations that cancel the common low-z
baryon load. **This is the central result that motivates the whole approach.**

**BNT cross structure** (correlation coefficient at ℓ≈300): separated-bin crosses 1×3 (+0.25),
1×4 (+0.00), 2×4 (+0.11) are ~nulled; adjacent crosses 1×2 (−0.59), 2×3 (−0.68), 3×4 (−0.51)
retain strong (anti)correlation — expected, since the banded transform overlaps neighbors and
shape noise is shared between adjacent BNT bins. Consistent with x-cut's "removes cross-bin
correlations by construction" for *separated* bins. The BNT-cross baryon ratios are dominated by
near-zero denominators and are not individually informative; the baryon signal lives in the BNT
bin-1 auto (and the crosses that involve it).

**Low-ℓ / monopole:** after BNT the first bandpower keeps S/N(mean/std) ≈ 2 (noise-like) — the
monopole subtraction propagates correctly through BNT (§6); no re-leakage.

## 6. Why it is exact on *our* pipeline (commutation argument)

Every stage downstream of the maps is a linear operator that is the **same for every tomographic
bin pair** because all four bins share one footprint mask. M is a constant matrix, so it commutes
with each:

- **mask multiply:** `w·(Σ M_ai κ_i) = Σ M_ai (w·κ_i)` — masking ∘ BNT = BNT ∘ masking.
- **MASTER decouple:** one coupling matrix per mask (same for all pairs) → linear → commutes.
- **bandpower bin:** same linear binning for all pairs → commutes.
- **monopole subtract:** `μ̃_a = Σ M_ai μ_i`, so submean-then-BNT = BNT-then-submean (verified
  numerically: BNT bin-0 S/N ≈ 2, not re-inflated).
- **noise:** added to the original maps *before* BNT (`bnt_cross_power_spectrum_processing_master.py`
  docstring line 9, "before BNT transform for physical consistency"), and **autos and crosses are
  measured from one shared noisy field per map** (`compute_power_spectra_master`, the same NmtField
  `f` feeds both `compute_coupled_cell(f,f)` and `(f_i,f_j)`). So the bilinear identity holds *per
  realization*, not just in the mean. This is also the only **data-reproducible** noise model — on
  real data you cannot inject noise into BNT maps independently — so M·C(noisy)·Mᵀ is not merely
  equivalent, it is the correct model. (Related: the masked-BNT-HOS note, `[[masked-bnt-hos-treatment]]`.)

⟹ For a single shared mask, `M·C·Mᵀ` on the MASTER-decoupled binned submean grids equals
reprocessing the BNT maps through the same MASTER pipeline. **No rerun needed — *if* the commutation
survives the actual decoupling numerically (§7).**

## 7. Caveats — read before trusting this

1. **Masked harmonic-space BNT is unvalidated in the literature.** All BNT papers BNT the
   maps/kernels *before* estimation, or use theory Cℓ. None apply M·C·Mᵀ to masked measured
   pseudo-Cℓ. The 2024 review and 2025 comparison paper do not mention masks, pseudo-Cℓ, or
   mode-coupling at all. **x-cut explicitly switched to configuration space to avoid mask
   deconvolution** ("There is no need to deconvolve the mask which could lead to a loss of
   information"). So §6 is *our* derivation, not a citable result. → **oracle gate (§9).**
2. **Mitigant:** we cut on *MASTER-decoupled* bandpowers, not raw pseudo-Cℓ. The mode-coupling
   leakage x-cut worried about (an ℓ-cut on coupled pseudo-Cℓ doesn't cleanly remove physical
   scales) is largely undone by the decoupling. We are in a better regime than the naive one — but
   bandpower decoupling is imperfect at low f_sky (small footprints), a residual concern.
3. **The ℓ↔k localization after BNT is approximate**, and *more* approximate on a cut sky. The
   "bin 1 carries all the bias" result is measured on *our* data (§5), not assumed from theory — so
   the empirical localization is what we rely on, which is the right footing.
4. **M is n(z)-specific** (§2). Valid only for this analysis's source redshifts.
5. **Binning is coarse** (nlb=4 → 40-ℓ effective bandpowers after rebin=10). The per-bin ℓ_max grid
   is correspondingly coarse; fine-tuning bin-1's cut is limited by this granularity.

## 8. Implications for the analysis

- Build the BNT data vector = M·C·Mᵀ → 4 BNT autos + 6 BNT crosses (full set, nothing dropped).
- Scale-cut **only the spectra involving BNT bin 1** ({auto-1, 1×2, 1×3, 1×4}) at a scanned ℓ_max;
  keep bins 2–4 at full range. This is simultaneously the x-cut Eq. 23 rule and the user's bin-1
  hypothesis.
- Headline test: does a mild bin-1 cut remove the baryon bias (contours return to unbiased) while
  retaining bins 2–4's full constraining power? Compare against the original "cut everything"
  nσ-vs-ℓmax curve.
- The fiducial-ratio result (§5) is necessary but not sufficient; the contour-level sweep is the
  proof (covariance matters too).

**Mechanism / worker change.** The cut is specified via the worker's per-bin `--upper-cuts`
(e.g. `"500,1024,1024,1024"` = cut BNT bin 1 at ℓ=500, bins 2–4 at full). Autos use their own
bin's cut (`upper_cuts[i]`). **Cross loaders had a latent bug:** they used
`max(upper_cuts[i], upper_cuts[j])` — the *less* restrictive bin — which would leave the
contaminated bin-1 crosses (1×2, 1×3, 1×4) at full ℓ. Fixed to `min(...)` (the x-cut Eq. 23 rule)
in both the data and fiducial cross loaders of `run_npe_inference_auto_cross_ps_master.py`. The
change is a **no-op for uniform single-`--upper-cut` runs** (min=max), so the existing non-BNT
submean tension results are unchanged; it only affects per-bin cuts (i.e. this BNT sweep).

## 9. Plan (oracle promoted to a gate)

1. **Oracle (mandatory, the literature substitute) — ✅ PASSED 2026-06-21.**
   `scripts/diagnostics/bnt_on_spectra_oracle.py` runs both paths on the *same* noisy maps for 3
   fiducial perms (14000 deg², apod 2.0, lmax 1535, nlb 4): Path A = MASTER(M @ kgs_noisy),
   Path B = M · MASTER(kgs_noisy) · Mᵀ. Result: **worst relative diff 1.1e-11** (median ~7e-16;
   abs diff ~1e-18); the only ~1e-11 cases sit on the nulled crosses where the value is ~1e-14.
   ⟹ BNT commutes with mask + MASTER decouple + bandpower bin on our exact pipeline; M·C·Mᵀ on the
   produced grids is byte-equivalent (to roundoff) to reprocessing BNT maps. The no-rerun shortcut
   is proven; the masked-harmonic gap the literature left open (§7.1) is closed empirically here.
2. **Build BNT grids** via the transform: read the 10 submean grids per area, assemble C(ℓ), write
   `all_bnt_cls_grid_*` + `all_bnt_cross_cls_grid_*` in the exact format the worker's existing
   `--bnt` path expects. Numpy only, all 6 areas, fast.
3. **BNT bin-1 scale-cut sweep:** re-run the nσ-vs-ℓmax tension plot with per-bin cuts touching only
   BNT-bin-1 spectra (worker `--upper-cuts`), vs the original cut-everything curve.

**Verification oracles:** step 1 is itself the oracle; step 2 — BNT grid round-trips (e.g.
`M⁻¹ C̃ M⁻ᵀ` recovers C; high-ℓ BNT autos unchanged by submean) and the §5 baryon-localization
reproduced from the saved grids; step 3 — the QA gate + estimator-variance error bars already in the
tension package.

## References

- Bernardeau, Nishimichi & Taruya 2014, arXiv:1312.0430 (original BNT).
- Taylor, Bernardeau & Huff 2020 — x-cut cosmic shear (baryons), [arXiv:2007.00675](https://arxiv.org/abs/2007.00675).
- k-cut cosmic shear, [arXiv:1809.03515](https://arxiv.org/abs/1809.03515).
- BNT review 2024, [arXiv:2412.14704](https://arxiv.org/html/2412.14704).
- Comparing nulling methods for Stage-IV, 2025, [arXiv:2512.15604](https://arxiv.org/html/2512.15604v2).
