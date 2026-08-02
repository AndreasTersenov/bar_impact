# Response to the referee — Fig. 2

> **Referee:** *Fig. 2; are the error bars (err(PS_baryons)/PS_DMO) or are they
> err(PS_baryons−PS_DMO)/PS_DMO? (It should be the former). Also, the ratio seems noisy; are you
> using the same shape noise realization for DMO and baryons? The cosmic variance should cancel
> almost exactly as I assume you're using the same simulation with and without baryons. You should
> do the same for the shape noise.*

We thank the referee — both points were correct, and we have made both changes. Fig. 2 has been
regenerated.

---

## 1. The error bars

They were the second form, `err(PS_bar − PS_DMO)/PS_DMO`, computed as the scatter of the
per-realization ratio across the simulation suite.

**They are now `err(PS_bar)/PS_DMO`, as the referee suggests.**

We agree this is the appropriate choice. In our analysis the DMO prediction plays the role of the
*model* — it is what the density estimator is trained on — while the baryonified spectrum plays the
role of the *observation*. The uncertainty that belongs on the figure is therefore the statistical
error of a single measurement, which is exactly `err(PS_bar)`. The band now has the direct
interpretation: **the 1σ range within which a single (full-sky) realization would measure this
fractional difference.**

We verified that the band is a genuine cosmic-variance-plus-shape-noise error: it agrees with the
Gaussian expectation `σ(C_ℓ)/C_ℓ = sqrt(2/N_modes)` to a few percent across the full multipole
range.

## 2. The shape noise

The referee is right that the shape noise was **not** matched. The simulations were indeed shared —
the baryonified and DMO maps for each realization come from the same shell permutation, so the
cosmic variance was already common — but the noise was drawn independently in the two runs.

**We have regenerated the fiducial spectra with the same shape-noise realization added to the
baryonified and DMO maps**, for all 200 permutations and all four tomographic bins. The noise seed
is now set deterministically from the permutation and bin index, so the suite is exactly
reproducible.

The effect is as the referee anticipated:

| | before | after |
|---|---|---|
| correlation between the baryonified and DMO realizations | 0.10 → −0.005 | **1.0000 → 0.9996** |
| per-realization scatter of the ratio | — | **10–30× smaller** |

The mean curve is statistically unchanged (e.g. bin 4 at ℓ ≈ 900–1000: −0.0162 after, −0.0169
before), as expected since the mean noise power cancels either way. What changed is the scatter:
the ratio is no longer noisy, and the curves in the new figure are smooth.

## 3. Two further corrections we found while making these changes

**(a) The lowest multipoles were being dropped.** The logarithmic band edges were specified as
multipole values but used as array indices into a spectrum already truncated at ℓ_min = 30. Every
band was therefore shifted upward by 30, and multipoles ℓ = 30–58 did not enter the figure at all.
This has been corrected; all multipoles ℓ ≥ 30 are now included.

**(b) The points were plotted at the wrong multipoles.** The bands are logarithmically spaced, but
the points were drawn at linearly spaced positions (ℓ = 30, 130, …, 930). They are now drawn at the
mean multipole of the band each point actually averages (ℓ = 36, 52, 74, 104, 148, 212, 302, 430,
612, 871).

## 4. Effect on the results

**None of the conclusions change.** The corrections in §3 affect only the lowest multipoles, where
the baryonic suppression is negligible, and the change of error bar and noise seeding does not move
the mean curve. In particular, the adopted scale cuts are derived from the parameter-level bias
measured on the unbinned spectra, which was never affected by the binning used for this figure.

## 5. A clarification we have added to the caption

Fig. 2 is now accompanied by a statement that the per-multipole comparison understates the total
impact of baryons. The baryonic shift is coherent across multipoles while the noise is random, so it
accumulates: although the curve lies within the 1σ band at most multipoles, the cumulative
significance of the DMO-model / baryonified-data mismatch grows to

| ℓ_max | bin 1 | bin 2 | bin 3 | bin 4 |
|---|---|---|---|---|
| 460 | 1.7σ | 2.6σ | 2.8σ | 2.6σ |
| 1000 | 3.5σ | 6.7σ | 9.6σ | 10.1σ |

for a full-sky measurement. This is the reason a scale cut is required, and we felt it should be
stated explicitly so the figure is not read as implying that baryons are negligible.

> **Internal note (not for the referee).** The figure exists in two variants and the numbers above
> are the full-sky ones. If the 14000 deg² variant is used instead, the band widens by
> 1/√f_sky = 1.72 and the table becomes 1.0/1.5/1.7/1.5 σ at ℓ_max = 460 and 2.1/3.9/5.6/5.9 σ at
> ℓ_max = 1000. **Update this section to match whichever figure ships.** Slugs:
> `paper/figures/ps_frac_diff_fig2` (full sky) and `ps_frac_diff_fig2_14000`.

---

### Note on Fig. B.1

Fig. B.1 is computed on strictly noiseless convergence maps and is unchanged. Its vertical scale
differs from Fig. 2 by the signal fraction S/(S+N): Fig. B.1 divides by the signal alone, whereas
Fig. 2 divides by the measured (noisy) spectrum. The two therefore show the same physics on
different scales, and the ordering of the tomographic bins differs between them — bin 1 has the
largest intrinsic suppression but, being the noisiest bin, the smallest fractional impact on the
observed spectrum. This is now stated in the captions.
