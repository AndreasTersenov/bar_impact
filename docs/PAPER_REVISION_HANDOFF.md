# Paper revision handoff — editing the draft locally, evidence lives here

**Written 2026-08-05**, at the close of the figure-remake campaign. Audience: Andreas editing
the manuscript on the laptop; the JZ Claude session for anything this doc doesn't answer.

The division of labour: **every figure-level fact is on the cluster, in machine-checked
form** — this doc is the index and the delta list, not a re-derivation. When a caption or a
number is in doubt, the lookup order is:

1. `paper/figures/<slug>/README.md` — what the figure is, source, publish date
2. `paper/figures/<slug>/values.csv` — every number plotted, exactly
3. `paper/figures/<slug>/provenance.json` — command, inputs, conventions, caveats list
4. `docs/PAPER_FIGURE_MAP.md` §3 — the current state of the science (scale cuts, FoM
   comparisons, caption traps), rewritten 2026-07-31 with measured answers
5. `docs/PAPER_NOTES.md` — central results index (**last updated 2026-06-25**; §0, §2, §4
   still carry the pre-correction BNT framing and the crossing-as-cut error — trust
   PAPER_FIGURE_MAP where they disagree)
6. Ask the JZ session — it holds the working memory of *why* choices were made

## 1. What to download

One bundle, everything needed to drop figures into the draft:

    scp jean-zay:/lustre/fswork/projects/rech/prk/ulx34io/transfers/paper_figures_2026-08-05.tar.gz .

Contents: the full `paper/figures/` tree (61 figures, each with `figure.pdf`, `figure.png`,
`values.csv`, `provenance.json`, `meta.json`, `README.md`, indexed by `MANIFEST.md`) plus
this doc, `PAPER_FIGURE_MAP.md` and `PAPER_NOTES.md`. All 61 pass
`scripts/paper/figures.py verify` as of today.

The noiseless power-spectrum data vectors (16 × `.npy`, standard+BNT × DMO+baryonified)
are a separate bundle, already delivered:
`$WORK/transfers/noiseless_cls_2026-08-05.tar.gz` — its README documents provenance,
the lmax-1024-only warning, and the paired-difference band.

## 2. Global changes affecting many figures (vs the submitted version)

These conventions changed after submission; every regenerated figure carries them, and any
text describing methodology must match:

- **Footprint-mean subtraction (`submean`) for masked HOS.** Root-cause fix for the
  anomalously tight masked peak counts. Full-sky detail scales are monopole-invariant, so
  full-sky peaks stay non-submean — *exact*, but a different noise realisation than the
  other statistics in the same figure (one caption line; PAPER_FIGURE_MAP §3.5–3.6).
- **PS multipole floor ℓmin = 37** with monopole subtraction / MASTER low-ℓ recovery.
  The σ-vs-area slope depends on it (−0.45 vs a spurious −0.29 at ℓmin=100).
- **Coarse starlet scale dropped: the analysis is `scales1234`.** The coarse scale is
  baryon-safe but adds no constraining power; `scales2345` variants exist as robustness
  checks only (PAPER_NOTES §1, PAPER_FIGURE_MAP §3.2).
- **Pooled and single-seed variants** are published side by side (`_pooled` /
  `_single_seed`). Pooling widens σ by only 1–5%; FoM₃ differs up to ~1.27× because of the
  det^(−1/2). A centre-outlier guard now rejects broken seeds (right width, wrong place —
  two were caught; PAPER_FIGURE_MAP §3.3–3.4).
- **Okabe–Ito colourblind-safe palette** in the regenerated figures — deliberately does NOT
  match figures kept from the submitted version.
- **NPE NaN-retry fix (2026-06-20).** Any posterior trained before it may be a silent
  prior collapse; everything published post-dates the fix.

## 3. Figure-level deltas from this campaign (Aug 2026)

- **`hos_bnt_peaks_vs_scale_cut`, `hos_bnt_l1_vs_scale_cut`** (new figures): three-arm
  triangles — BNT all-scales (outline), standard with the [20′,40′,80′] cut, standard all
  scales. Legend carries **only the three series labels**; FoM₃ and posterior moments live
  in `values.csv`, not the legend. **Both BNT arms are the baryonified runs**
  (like-for-like with the other arms; an earlier no-baryons ℓ₁ iteration is superseded).
  Caveat recorded in provenance: these NPE runs predate the ℓmin=37/submean conventions —
  fine standalone, do not overlay on current-convention contours.

  | statistic | BNT all scales | cut [20′,40′,80′] | all scales |
  |---|---|---|---|
  | peaks | 6.08×10⁴ | 6.06×10⁵ (10.0×) | 1.33×10⁶ (21.9×) |
  | ℓ₁ | 1.47×10⁵ | 1.31×10⁶ (8.9×) | 3.62×10⁶ (24.6×) |

- **`ps_frac_diff_noiseless`** (new figure, published today): noiseless ⟨ΔC_ℓ⟩/⟨C_ℓ⟩,
  standard bins solid, BNT bins dotted, ℓmax 1024. The baryonified spectra had to be
  **rebuilt from the read-only CosmoGridV1 release** (SLURM job 597132) — the originals did
  not survive the disk failure; the nobaryons halves are the 2025-10 originals. Band =
  paired-difference SEM (same 200 perms in both sets — independent-sets errors would be
  ~10× too wide). BNT bin 1 ≡ standard bin 1 (first BNT row is the identity).
  dC/C at ℓ=1000: standard −0.162/−0.129/−0.093/−0.077; BNT −0.162/−0.092/−0.033/−0.008.
- **`starlet_scale_ell`** (new appendix figure): measured starlet scale→multipole response
  (commit `cd2a48f3`), replacing the recollected mapping.
- Everything else in `paper/figures/` was regenerated post-crash under the §2 conventions;
  each README/provenance records its own history.

## 4. Manuscript text corrections (accumulated list)

Checked against the pipeline, with sources — the draft must say:

1. **Peak counts: 30 SNR bins per scale, SNR ∈ [−2, 6]** — verified on the datavector,
   shape `(200, 5, 30)`; the submitted text says 40. Generator default
   `scripts/peak_counts_processing.py:124` (`nbins=31` is the edge count).
2. **ℓ₁-norm: 40 bins over SNR ∈ [−13, 13]** — datavector shape `(200, 5, 40)`,
   `scripts/l1_norm_processing.py:123`. Check the range quoted in the draft against this.
3. **Five starlet scales computed, scales 1–4 used** (coarse excluded) — anywhere the text
   or a caption implies all five enter the analysis, it's wrong.
4. **Full-sky ℓmax=340 comparison figure**: all three statistics sit *above* nominal 0.3σ,
   justified by error bars — the caption must say "matched, marginally tolerated bias",
   not "baryon-safe" in the 14000 deg² sense (PAPER_FIGURE_MAP §3.1).
5. **The peaks-vs-PS claim is footprint-dependent**: at full sky peaks gain ~nothing over
   PS (1.03× FoM₃) while ℓ₁ gains 2.33×; at 14000 deg² peaks do edge past PS. Never quote
   one footprint's ordering as the general result (§3.2).
6. **BNT framing**: information *retention* under a targeted bin-1 cut (1.47× FoM₃ at
   matched ℓmax=460, 92/120 bandpowers kept vs 50) — **not** better baryon control (BNT
   crosses 0.3σ at a *lower* ℓmax than a uniform cut), and it **requires MOPED**: a flow on
   the raw BNT vector loses the Ωm–S₈ degeneracy and the comparison inverts
   (PAPER_NOTES §0 headline, which is current on this point).
7. **Full-sky peaks non-submean** caption line (see §2).

## 5. Open decisions — yours, not mine

- **The 2.2σ headline vs the 1.43 ± 0.30 controlled re-run** (raw-NPE high draw vs
  score-compressed; PAPER_FIGURE_MAP §1.1). Propagates into the adopted cut: ℓmax 460
  (raw r10) / ~500–540 (VMIM) / 580 (score) at 14000 deg². The paper must pick and justify.
- **Headline footprint**: 14000 deg² vs full sky for the constraining-power story.
- **TARP/SBC calibration**: still the largest referee exposure; needs retraining
  (no flow checkpoints survive for any corrected config).
- **ℓ₁ BNT baryonified NPE**: single run; a retrain would firm the triangle's BNT arm.
- **prk STORE cleanup**: 93,785 dead inodes awaiting a decision.

## 6. Uncommitted state (as of writing)

`scripts/plot_ps_frac_diff_noiseless.py`, this doc, and the
`paper/figures/ps_frac_diff_noiseless/` + regenerated manifest are not yet committed.
The provenance gate's remaining warnings on `ps_frac_diff_noiseless` clear once the
generator is committed (it currently records "dirty/untracked at run time").

## 7. Asking the JZ session

For anything not answered by the chain in the preamble: state the figure slug or the
manuscript claim, and ask. The session keeps persistent memory of the recovery forensics,
the corruption signatures, and the conventions — and can re-derive any number from the
sidecars. Rule of thumb: **if a number can't be traced to a `values.csv`, don't put it in
the paper; ask instead.**
