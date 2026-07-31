# HANDOFF — restarting the BNT + VMIM compression thread

**Written 2026-07-31.** For a fresh session picking up the BNT power-spectrum / neural-compression
work. Section 1 is the state of everything else (so nothing silently rots); sections 2–5 are the
BNT thread in detail.

Read alongside `docs/PAPER_FIGURE_MAP.md` (the survey of the figure tree and the measured science
results) and `paper/README.md` (the curated figure system).

---

## 1. Where everything else stands

### 1.1 Figures — done and self-maintaining

**30 published figures in `paper/figures/`, all passing `verify`.** One directory per figure holding
`figure.pdf`, `figure.png`, `values.csv`, `provenance.json`, `meta.json`, `README.md`, and for
contour figures a `fom.csv`. Three commands maintain it:

```bash
PY=/lustre/fswork/projects/rech/nzu/ulx34io/envs/aname/bin/python
$PY scripts/paper/figures.py republish   # re-copy from recorded sources; clears drift
$PY scripts/paper/figures.py manifest    # rebuild the index
$PY scripts/paper/figures.py verify      # re-gate everything
```

`publish` REFUSES a figure lacking a valid vector PDF, a non-empty `values.csv`, or a parseable
`provenance.json`. Do not weaken that gate; it is the reason the directory is trustworthy.

Git: branch `tension-submean-l37`, HEAD `ac4e80b`, pushed, working tree clean.

### 1.2 Science settled this session

- **Adopted scale cuts, measured** (largest step-40 cut with mean tension < 0.3σ):
  14000 deg² — PS ℓmax **460** (0.288σ); peaks/L1 `scales234` (0.079 / 0.091σ).
  Full sky — PS ℓmax **300** (0.206σ) or **340** (0.358σ, borderline); peaks 0.344σ, L1 0.379σ.
- **The adopted cut is NOT the crossing.** The crossing is the first cut that fails; the adoptable
  cut is the last that passes. They differ by one grid step. Every doc except this one and the
  figure map still reports the crossing as if it were the cut.
- **Full sky, matched bias:** L1 gains **2.33×** in FoM₃ over the PS; **peaks gain 1.03×, i.e.
  nothing.** At 14000 deg² peaks *do* edge past the PS (1.52e5 vs 1.44e5). The claim is
  L1-specific and footprint-dependent; quoting one footprint without the other misleads.
- **Cost of baryon safety at 14000:** PS retains 58% of FoM₃, peaks 38%, L1 39%.
- **`scales2345` (coarse scale included)** is a robustness result only: bias drops
  (peaks 0.079→0.034, L1 0.091→0.082) and seed scatter collapses ~10×, but it costs peaks a third
  of their FoM and buys L1 8%.

### 1.3 Still open, unchanged

| item | status |
|---|---|
| **TARP/SBC calibration** of the corrected NPE posteriors | Largest referee exposure. No flow checkpoints survive for any corrected config, and it **cannot** be computed from saved sample files — those are 3000×6 draws at one fixed observation and cannot produce an ECP curve. Needs a retrain. |
| **Baryon-model dependence** | One baryonification prescription throughout. Needs new sims. |
| **2.22σ vs 1.43σ** | `PLAN_score_bnt_tension_14000.md` records that the published 2.22σ PS headline at 14000 "does not reproduce" (controlled re-run: 1.43 ± 0.30). Unresolved; a scientific decision. |
| **`PAPER_NOTES.md` §0, §2, §4** | Still carry the crossing-as-cut error and the pre-correction BNT framing. |
| **`peak_counts_processing.py`** | No full-sky submean branch (its submean block sits inside `if apply_mask:`). Not currently blocking anything — see §5.3. |
| **`docs/HANDOFF_bnt_bin1_other_areas_PROGRESS.md`** | 100% NUL, unrecoverable. `PAPER_NOTES §4` cites it. |

---

## 2. The BNT thread — what it is

**The physics argument.** In the standard frame the baryon-safe scale cut must be applied roughly
uniformly across tomographic bins: if ℓmax = 400, that is ℓmax for every bin. Under the BNT
(nulling) transform, the baryon sensitivity concentrates in the first BNT bin, so in principle only
bin 1 needs cutting and bins 2–4 can keep their full range. That should retain much more
information and give tighter contours at equal baryon safety.

**Why it became hard.** BNT is a fixed lower-triangular nulling operator `M` acting as
`C̃ = M C Mᵀ` per multipole. By design it produces near-cancellations, so the transformed vector is
**ill-conditioned**: strong anti-correlations, near-degenerate directions, sign changes, wide dynamic
range, and its information sitting in many low-S/N high-ℓ modes. A normalizing-flow NPE asked to
find the `JᵀC⁻¹` projection in that space **under-learns it** and returns a posterior that is too
wide. Critically the failure is **silent** — the posterior stays calibrated, it is just less
informative than the data allow. The well-conditioned non-BNT vector does not suffer this.

Consequence: the *rebinning* of the BNT spectra mattered enormously, which is what made direct
inference on BNT vectors unattractive.

**The two fixes, and how they differ.**

- **Score / MOPED compression** (`t = JᵀC⁻¹(x−μ)`, used as `θ̂ = θ_fid + F⁻¹JᵀC⁻¹(x−μ)`): analytic,
  6 numbers, lossless for a Gaussian likelihood (`Fisher(t)=Fisher(x)` to ~1e-10). The `F⁻¹`
  parameter-units rotation is essential — the raw score is itself cond ~1e8 and the NPE fails on it.
- **Whitening is NOT compression.** Full-rank Cholesky decorrelation keeps all ~96 dimensions, so the
  flow must still find the projection. It leaves BNT looking no better than non-BNT (FoM ratio 0.96)
  while score reveals the advantage (1.46). They behave **oppositely** on the BNT vector.
- **VMIM neural compression** — a *learned* compressor. The first attempt failed (over-confident,
  off-truth). **v2 corrected it and PASSES.**

**The non-BNT control matters.** The same compression applied to the well-conditioned non-BNT vector
reproduces the uncompressed posterior (FoM ratio 1.00–1.08 across all six footprints). That is what
licenses attributing the BNT gain to information rather than to the method.

Sources: `docs/NOTES_bnt_compression_for_paper.md` (paper-facing, best single read),
`docs/BNT_on_spectra.md` (the transform and the cross-spectra treatment),
`docs/PLAN_bnt_optimal_binning.md` (why rebinning matters; and the retraction in §5.2 below).

---

## 3. VMIM v2 — the pilot PASSED

`docs/PLAN_bnt_vmim_v2.md`, RESULTS LOG 2026-06-29. Pilot P0→P2 complete.

**P2f, the primary pass:** identity σ-ratio 1.08/1.12 ✓; both arms on-truth (BNT S₈ 0.846, non-BNT
0.843 against 0.84); TARP ≤0.06; SBC rank-std 0.30–0.31 (uniform) both. Only Gate-3 w0/floor
(0.63–0.67) "fails", and that is a **documented over-estimated w0 Fisher floor** from a shallow
local w0 Jacobian — confirmed not over-confidence by the uniform w0 SBC.

**The methodological findings that made it work — do not relearn these:**

1. Stage-2 must be the **sbi_lens RealNVP** port, not jaxili; plus **by-cosmology** train/val split
   (leakage = 0 across 2424 cosmologies), no summary-noise, H0/100.
2. A plain learned compressor on z-scored ill-conditioned input **under-extracts BNT** — same failure
   mode as raw NPE.
3. The fix is **analytic-covariance noise-whitening** (`ana_whiten`, IMNN-style): whiten by the
   regularized analytic `C^{-1/2}` so noise is isotropic and the MLP need only find the ~6 signal
   directions.
4. **CRUX — the whitening clip must be PER-FEATURE-RELATIVE.** After `C^{-1/2}` the
   parameter-sensitive directions carry std up to ~4.6; an ABSOLUTE ±5 clip (correct for a
   unit-variance z-score) lops that signal and **biases S₈ by ~1σ**. This was the entire difference
   between "biased" (P2c/d) and "on-truth" (P2e).
5. **A 3-seed compressor deep-ensemble** (common split, varied init) is the over-confidence remedy:
   SBC 0.35→0.30, identity 0.40σ→0.15σ.

---

## 4. The scale-cut sweep — COMPLETE, with a caveat

`docs/PLAN_vmim_scalecuts.md` describes it; **it ran and finished.** This is the thing previously
believed blocked.

- **Grid:** ℓmax 340…1020 step 40 (18 cuts) × 2 configs × 3 compressor seeds = 36 job dirs, all
  present under `outputs/baryon_tension/vmim_v2/scalecuts/`.
- **Configs:** `nonbnt` = cut ALL bins, `cuts=[c,c,c,c]`. `bnt` = cut bin-1 ONLY,
  `cuts=[c,1024,1024,1024]`.
- **`tension_agg.csv` is INTACT** (1909 B, 0% NUL), `n_seeds=9` per row (3 compressor × 3 NDE seeds).
- **Deliverable figure exists:** `plots/nsigma_vs_upper_cut_compressed_14000.png` (Jun 30).

**Result, VMIM-compressed at 14000 deg²:**

| config | adopted ℓmax | tension there | first failing cut |
|---|---|---|---|
| non-BNT (cut all bins) | 500 | 0.213 | 540 (0.382) |
| BNT (cut bin-1 only) | 420 | 0.202 | 460 (0.311) |

**Do not read "420 < 500" as BNT being worse.** They are not the same quantity: BNT cuts only bin 1
and keeps bins 2–4 at ℓ=1024, so at c=420 it retains far more information than non-BNT does at
c=500. **The comparison that settles the argument is the FoM at each config's adopted cut, and no
figure reports it yet.** That is the first thing to produce.

### 4.1 THE CAVEAT — the per-seed posteriors are partly damaged

`tension_agg.csv` was aggregated **before** the disk failure and is complete and trustworthy. The
per-seed `.npy` posteriors underneath it are **not**. Of 108 seed slots (36 dirs × 3):

- **19 unreadable** (disk damage — `np.load` raises ValueError; NEVER "fix" with `allow_pickle=True`)
- **3 collapsed** — readable but prior-width, σ(S₈) = 0.36–0.41 against a 0.08 guard:
  `nonbnt_c460/cs41`, `nonbnt_c460/cs43`, `nonbnt_c500/cs43`

Everything else is healthy (σ(S₈) 0.016–0.039).

**Why this specifically matters:** the collapsed seeds sit at ℓmax 460–500, which is exactly where
the non-BNT curve crosses 0.3σ. **The non-BNT adopted cut of 500 is therefore the least trustworthy
number in the sweep**, and a naive FoM re-derivation from surviving posteriors gives a nonsense 72×
BNT advantage driven entirely by the collapsed non-BNT fit. Retrain those three before quoting any
FoM comparison.

### 4.2 Adjacent cuts share values

`tension_agg.csv` has identical entries at adjacent cuts (non-BNT 540/580, 620/660; BNT 460/500).
Most likely the rebin=20 grid makes adjacent ℓmax values select the same columns, so the curve is a
staircase rather than smooth. **Confirm this rather than assuming it** — it changes how the curve
should be drawn and whether "adopted cut = 420" is really distinguishable from 460.

---

## 5. What survived, what did not

### 5.1 Intact

**Docs:** `BNT_on_spectra.md`, `PLAN_bnt_vmim_v2.md`, `PLAN_vmim_scalecuts.md`,
`HANDOFF_bnt_vmim_reimplement.md`, `NOTES_bnt_compression_for_paper.md`,
`PLAN_bnt_neural_compression.md`, `PLAN_bnt_npe_whitening.md`, `PLAN_score_bnt_tension_14000.md`,
`PLAN_bnt_optimal_binning.md`, `FIRST_PROMPT_bnt_vmim.md`,
`HANDOFF_bnt_bin1_tension_other_areas.md`.

**Scripts (all 8 pipeline pieces):** `vmim_compress.py`, `nde_realnvp.py`,
`nde_realnvp_from_summary.py`, `vmim_gate.py`, `pool_ensemble.py`, `p0_verify_nde_port.py`,
`vmim_scalecut_compute.py`, plus `plot_cut_contours.py` / `plot_biased_contours.py`.

**Data:** 52 `cache.npz` score caches under `outputs/score_experiment/` (48 were regenerated and
verified this recovery — `bnt_full` reproduces the documented pre-crash shape `x[16965,120]`
exactly). TARP checkpoints `ckpt_tarp_bnt_580` and `ckpt_tarp_nonbnt_460` (orbax dirs) survive —
these are the ONLY flow checkpoints in the repo.

### 5.2 Lost

| what | consequence |
|---|---|
| **`scripts/score_cut_utils.py`** (5144 B, 100% NUL, untracked) | Provided `keep_indices(cuts)` and `build_score(cuts, bnt)`. Blocks **re-running or extending** the sweep; does NOT block using the completed results. Interface documented in `PLAN_vmim_scalecuts.md` §1. ~5 KB to rewrite. |
| **`run_vmim_pilot.py`** | Absent. Pilot already complete, so low priority. |
| **`docs/HANDOFF_bnt_bin1_other_areas_PROGRESS.md`** | 100% NUL. Cited by `PAPER_NOTES §4`. |
| **`~/.claude/plans/jolly-toasting-robin.md`** | The score/MOPED experiment plan. Its findings are distilled in `NOTES_bnt_compression_for_paper.md`, so this is recoverable in substance. |
| **19 of 108 sweep seed posteriors** | See §4.1. |
| **`bnt_ps_bin1_fullsky_l37/tables/`** — all 8 files | 100% NUL. The "0.05 → 1.10, 22× better" full-sky BNT claim has **no recoverable source**. |

### 5.3 A retraction to carry forward

`PLAN_bnt_optimal_binning.md` "FINAL RESULTS" records that the shipped **rebin=10** BNT figure
substantially **OVERSTATED** BNT's baryon mitigation — the low BNT tension was NPE under-extraction
inflating the contours and hiding the bias, not baryon control. The honest version is rebin=40
(`nsigma_vs_lmax_bnt_bin1_allareas_optimal`, published). `PAPER_NOTES §4` still carries the
pre-correction framing.

Also: the per-panel "% extracted" annotation on that figure (87/85/82/76/93/93) is **not
reproducible from this repo** — it divides by a non-BNT-at-rebin-40 tension that was never saved.
Recorded in its sidecar as `reproducible_from_repo: false`.

---

## 6. Suggested order of work

1. **Retrain the 3 collapsed non-BNT seeds** (`nonbnt_c460/cs41`, `nonbnt_c460/cs43`,
   `nonbnt_c500/cs43`). Pipeline intact; one compressor + NDE each. Sizing: read the epoch rate
   from a live log, do not guess.
2. **Confirm or refute the adjacent-cut degeneracy** (§4.2) by checking which columns
   `keep_indices` selects at 460 vs 500 — this needs `score_cut_utils.py` rewritten, or can be
   inferred from the cached column structure.
3. **Produce the FoM-at-adopted-cut comparison** — BNT (bin-1 only, c=420) vs non-BNT (all bins,
   c=500). This is the number the whole thread exists to produce and no figure reports it.
4. **Then** a contour figure at those cuts, published through `scripts/paper/figures.py`.
5. Optionally extend the sweep to other footprints (needs `score_cut_utils.py`).

---

## 7. Traps — every one of these cost real time

- **A damaged `.npy` raises `ValueError`, not an IOError** (numpy reads the mangled header as a
  pickle stream). Guards catching only `(FileNotFoundError, IndexError)` let it through. Never use
  `allow_pickle=True` to "fix" it.
- **A 100%-NUL CSV reads as an empty DataFrame without raising** — `pandas.read_csv` returns
  `shape (0,1)`. 38 CSVs in this repo are affected. Assert row counts, not existence.
- **Any comparison between two separately-processed datavectors compares shape-noise realisations
  too** — `l1_norm_processing.py:46` seeds from `os.urandom`. This confounded a measurement and led
  me to a wrong conclusion about full-sky peaks.
- **`sbatch --export` uses COMMAS as separators**, so `--export=ALL,SCALES=1,2,3` silently becomes
  `SCALES=1`. Export from the calling shell instead. Cost 40 GPU-minutes.
- **Use `qos_gpu_a100-dev`** (2 h cap) for short GPU jobs. A 20 h request on `-t3` bought a 34 h
  queue wait for a 35-minute job.
- **getdist 1.4.3 (aname) cannot draw filled contours** under matplotlib ≥3.8. Contour work runs in
  **jaxili**; table work stays in **aname** (jaxili has no pandas, no tarp).
- **The σ(S₈) collapse guard cannot see a centre outlier** — a posterior of the right width in the
  wrong place. `plot_contours_vs_area.py` has a centre guard requiring both a robust MAD-z and a
  full posterior width of offset; a MAD-only version over-cuts where seeds agree tightly.
- **The pooled/per-seed FoM ratio is a good detector**: ~1.1 is geometry, far above means a broken
  seed.
- **`monitor_tension_pilot.py` REWRITES `tension_3param_agg.csv`**, the only surviving record of the
  full-seed campaign. 44 tables are archived read-only under
  `outputs/baryon_tension/_TABLES_ARCHIVE_precrash_20260730/`.
