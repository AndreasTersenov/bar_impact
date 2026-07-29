# HANDOFF — low-ℓ PS recovery + PS-vs-HOS across footprints (2026-06-17/18)

Continue from here. This documents the full low-ℓ-recovery / constraining-power campaign run this
session, the corrected data production, the results, and the open threads. **Immediate next task is at
the bottom (`## RESUME HERE`).**

---

## 0. One-paragraph summary
The masked power spectrum was previously forced to ℓ>100 by a **mass-sheet / monopole leakage** bug
(masking a κ map with a non-zero footprint-mean injects the unobservable monopole into low-ℓ). The fix
is **monopole subtraction** (subtract the mask-weighted mean of each map before NaMaster). This session
(a) **re-produced the proper monopole-subtracted PS for all 6 masked footprints** at the correct
binning, (b) ran NPE to compare **PS (ℓ100 vs ℓ37 vs ℓ0) against the HOS (l1-norm, peak counts)** on
full-sky + every masked footprint, both a full-ℓ and a baryon-safe regime, and (c) computed FoMs. Main
findings: low-ℓ recovery ~doubles the PS FoM; the masked pipeline reproduces full-sky cleanly (σ scales
∝ f_sky^-½); the HOS still beat the *corrected* PS but by ~half of what they beat the ℓ100 PS; and the
**masked peak counts are anomalously too-tight** (a real boundary artifact, see memory).

---

## 1. METHODOLOGY — do not re-confuse these (all verified this session)
- **nlb (bandpower width): use nlb=4, never nlb=1.** nlb=1 (unbinned) decoupling is numerically
  singular at low f_sky (measured cond(MCM): 2000=1e17, 5000=3e14, 10000/14000≈1e9 at nlb=1; all drop
  to O(1–20) at nlb=4). It only "works" for 28000/35000. The June-16/17 submean production was
  mistakenly run at nlb=1 (1023-col files) and broke 2000–14000; **this session redid it at nlb=4.**
- **Processing lmax = 1535** (= 3·nside−1, Nyquist for nside 512) → the pipeline AUTO-selects nlb=4
  (`lmax_effective>1500`). **Analysis ℓ-cut ≤ 1024** (applied at NPE via `--upper-cut`). These are TWO
  different lmaxes: process at 1535, analyze ≤1024. Setting processing `--lmax 1024` silently gives
  nlb=1 — that was the bug in `docs/POWER_SPECTRUM_PRODUCTION_PLAN.md` (now stale/wrong on this point).
- **Monopole subtraction** = `--subtract-mean`: subtract μ = Σ(w·κ)/Σw (the mask-weighted footprint
  mean = the ℓ=0 mass-sheet mode) before `NmtField`. **Agreed name: "monopole-subtracted"** (file tag
  `_submean`; "raw" for the un-subtracted version). No-op for ℓ>~150 (≤1% bulk), reshapes ℓ<100.
- **Full-sky is NOT affected and must NOT be redone.** On the full sphere the monopole is the single
  ℓ=0 mode (orthogonal to ℓ≥1), so any ℓ_min>0 already removes it — proven on a real bin-4 map
  (C₀ removed cleanly, every Cℓ ℓ≥1 unchanged to 1e-16). The paper full-sky uses the **healpy** pipeline
  (`run_npe_inference_auto_cross_ps.py`), June 2025, untouched. (A separate NaMaster full-sphere version
  exists but is NOT comparable — different estimator.)
- **pymaster**: current 2.5.2, only in the `cosmostat_new` venv:
  `/home/tersenov/software/cosmostat_new/cosmostat/cosmostat_new/bin/python`. Use it for ALL processing.
- **NPE env**: `/home/tersenov/anaconda3/envs/jaxili/bin/python` (JAX). NPE training is fast (~2–3 min).
- **HOS scale pairing** (starlet detail scales; idx0=finest/highest-ℓ, idx4=coarse=mass-sheet, excluded):
  full-ℓ regime ↔ **scales1234** (idx 0,1,2,3, keeps smallest); baryon-safe ↔ **scales234** (idx 1,2,3,
  drops smallest). Naming adds +1: idx[1,2,3]→"scales234".

---

## 2. DATA STATE (all on disk now)
Base: `/home/tersenov/CosmoGridV1/stage3_forecast/`. Params: `grid/cosmo_params.npy` (16965,6) =
[Ωm, S8, w0, H0, ns, Ωb]. Fiducial truth = [0.26, 0.84, −1, 67.36, 0.9649, 0.0493].
- **Proper monopole-subtracted nlb=4/lmax1535 PS GRIDS: ALL 6 masks DONE** (2000/5000/10000/14000/
  28000/35000). Files `new_grid/all_cls_grid_nobaryons_bin{b}_masked_{A}sqdeg_apod2.0_master_submean_
  noisy_s0.26_lmax1535.npy` (16965, 383) + `all_cross_cls_grid_..._submean_..._lmax1535.npy` (16965,
  2298). Fiducials same in `fiducial/` (200 rows), both nobaryons AND baryonified.
  - **Driver: `scripts/reprocess_submean_nlb4_production.py`** (grid=nobaryons only; fiducial=both
    sim types; 50 workers; finished 2026-06-18 05:38). Reusable for re-runs.
  - The old broken nlb=1 submean files (1023 cols, no lmax tag) still coexist — IGNORE them; use the
    `_lmax1535` (383-col) ones.
- **HOS (l1 + peaks) masked NPE posteriors**: already existed for all masks at `{A}+1` sqdeg (e.g.
  5001, 14001, 35001), scales1234 AND scales234, in `outputs/samples/posterior_samples_{,"pc_"}
  nobaryons_vs_nobaryons_bins1234_scales{1234,234}_noisy_s0.26_masked_{A}sqdeg_new_normalization*npe.npy`.
  (HOS footprint is {A}+1 deg² vs PS's {A} — a 1 deg² label difference, cosmologically nil.)

---

## 3. WHAT WAS PRODUCED (the plots) — `outputs/diagnostics/lmin_compare/`
For **full-sky + 6 masks**, each in **full-ℓ (lmax 1024, HOS scales1234)** and **baryon-safe (lmax 400,
HOS scales234)**, 4-probe (now 5-probe) **null (nobaryons_vs_nobaryons)** triangle plots in (Ωm,S8,w0):
PS ℓ100 (paper), PS ℓ37 (recovered), **PS ℓ0 (all low modes)**, l1, peaks.
- full-sky: `fullsky_baseline/samples/` (l37/l100-1024), `fullsky_l400/` (l37/l100-400, peaks sc234, the
  4-probe fig), 
- 5000: `gate_nlb4/samples/` (l37/l100-1024) + `masked5000_l400/` (l37/l100-400) + fig
  `masked5000_baryonsafe_PS_l1_peaks.png`, `masked5000_fulll_PS_l1_peaks.png`
- 35000: `masked_35000/` ; 14000: `masked_14000_npe/` ; 2000/10000/28000: `masked_others/`
- **PS ℓ0 runs + the 5-probe "withl0" plots**: `ps_l0/` (14 plots `withl0_{FS,2000,...}_{fulll,
  baryonsafe}.png`). The builder is the inline python in the session; key gotcha: the masked PS dirs are
  under `outputs/diagnostics/lmin_compare/` (not `outputs/diagnostics/`).

---

## 4. KEY RESULTS
- **Low-ℓ recovery ≈ doubles the PS FoM₃D** in every regime (×1.6–2.6); σ(S8) tightens ~20% (full-ℓ) to
  ~35% (baryon-safe) going ℓ100→ℓ37, and the PS (Ωm,w0) degeneracy rotates toward the HOS direction
  (r −0.4 → ~−0.1).
- **σ(S8) ladder is monotonic in f_sky** (validates the masked monopole-subtracted PS), PS ℓ37 baryon-
  safe: full-sky 0.017 → 35000 0.026 → 28000 0.028 → 14000 0.038 → 10000 0.042 → 5000 0.055 → 2000 0.081.
- **HOS over the CORRECTED (ℓ37) PS, FoM₃D**: l1 ×1.7–2.8 (steady), peaks ×1.2–6.0 — vs ℓ100 it was
  l1 ×4.5–5.6, peaks ×3–13. So ~half the apparent paper HOS advantage was the PS's missing low-ℓ.
- **PS ℓ0 (down to ℓ≈2)**: masked = ~15–20% tighter than ℓ37, CLEAN, consistent across footprints →
  submean recovered the *whole* low-ℓ band; ℓ37 was conservative. Full-sky ℓ0 = ×0.43 (anomalously
  tight) because it includes the ℓ=0 monopole (mass-sheet) — illustrative, NOT a real constraint.
- FoM convention used: FoM = 1/√det(C) (2D pairs and full 3-param). Tables computed inline (search the
  transcript for "figures of merit"). NPE FoMs are single-run (~15–20% width scatter — seed-average for
  final paper numbers; means agree with the paper, ratios are robust).

---

## 5. OPEN THREADS / CAVEATS
- **Masked peak counts are TOO TIGHT** (degrade only ×1.3 under masking vs l1 ×2.4 / PS ×3). Confirmed
  data-level (not NPE). Separate-session investigation pinned the mechanism: a **hard (un-apodized)
  binary mask edge** injects edge ringing into every wavelet scale → a perimeter-scaling boundary-peak
  channel. Fix = reprocess masked peaks with **apodized and/or eroded** mask (the `--mask-correction`
  path). See memory `masked-peak-counts-too-tight`. **Until fixed: masked peaks are PROVISIONAL** —
  all `peaks` contours in the masked plots are flagged. Full-sky peaks and all l1/PS are fine.
- All plots are **null (nvn)** = constraint width. The **baryon-bias (nvb)** versions (e.g. the user's
  `posterior_samples_ps_auto_cross_nobaryons_vs_baryonified_..._l100-400_..._npe.npy`) are a separate
  story to build if wanted (show the baryon impact + whether low-ℓ recovery shifts the bias).
- Full-sky peaks **scales1234** was never run (only scales234) — that's the one missing cell in the
  full-sky full-ℓ 4-probe (currently 4 probes there). Run it if you want full symmetry.

---

## 6. RELEVANT SCRIPTS
- Processing: `scripts/cross_power_spectrum_processing_master.py` (`--apply-mask --mask-area-sqdeg A
  --subtract-mean --lmax 1535 --num-workers N --aggregate-for-inference --inference-output-dir ...`;
  add `--fiducial` for fiducial, `--baryonified` for baryon). Production driver:
  `scripts/reprocess_submean_nlb4_production.py`.
- NPE masked PS: `scripts/run_npe_inference_auto_cross_ps_master.py` (`--masked --mask-area-sqdeg A
  --subtract-mean --lmax 1535 --lower-cut LC --upper-cut UC --rebin 10 --simulation-type nobaryons
  --fiducial-type nobaryons --noisy --noise-level 0.26 --train --gpu G --samples-dir D`).
- NPE full-sky PS (healpy): `scripts/run_npe_inference_auto_cross_ps.py` (no `--masked`/`--subtract-mean`/
  `--lmax`; `--lower-cut`/`--upper-cut`/`--rebin`). Cross-fiducial symlink gotcha already fixed.
- Fisher (validated proxy): `scripts/diagnostics/fisher_constraining_power.py` (+ verify_*).
- The 35000/2000/10000/28000 NPE were driven by inline scripts; replicate the master-NPE command above.

---

## DONE 2026-06-19 — full-sky plot WITHOUT the monopole (PS ℓ2)
Built `outputs/diagnostics/lmin_compare/ps_l0/withl2_FS_{fulll,baryonsafe}.png` via
`scripts/diagnostics/build_fullsky_nomonopole_plots.py` (readable standalone builder; `getdist` in the
`jaxili` env). PS ℓ2 (lower-cut 2, excludes ℓ=0 monopole + ℓ=1 dipole) **replaces** PS ℓ0 on the full
sky. Originals `withl0_FS_*` left intact for comparison. Pre-staged `l2-{1024,400}` posteriors verified
present (marker `ps_l0/L2DONE`). σ(S8):
- **full-ℓ** (upper 1024 / HOS scales1234, 4 probes — no full-sky peaks at scales1234):
  PS ℓ100=0.0158, ℓ37=0.0139, **ℓ2=0.0111**, l1=0.0094. ℓ2 lands just inside ℓ37 → legitimate low-mode
  gain, NOT the ℓ0 collapse (0.006). Confirms full-sky and masked agree once the monopole is excluded.
- **baryon-safe** (upper 400 / scales234, 5 probes — peaks scales234 from `fullsky_l400/`):
  PS ℓ100=0.0351, ℓ37=0.0174, **ℓ2=0.0195**, l1=0.0113, peaks=0.0135. ℓ2 ≈ ℓ37 (within NPE scatter).
- Builder-paths gotcha (recorded): full-sky PS ℓ100/ℓ37 are NOT in `ps_l0/` — they live in
  `fullsky_baseline/samples` (1024) and `fullsky_l400` (400); HOS in `outputs/samples` (+ peaks234 in
  `fullsky_l400`). The builder searches all four dirs.

## RESUME HERE (next tasks — deferred, pick with user)
- **Masked peak counts are provisional/too-tight** (the user is fixing in a separate session via
  footprint-mean subtraction — see memory `masked-peak-counts-too-tight`). Keep peaks flagged on every
  masked plot until that lands, then reprocess masked peaks + re-run NPE/tension.
- **Baryon-bias plots**: regenerate the `withl0`/`withl2` panels for `nobaryons_vs_baryonified` (bias
  direction), not just the `nobaryons_vs_nobaryons` null.
- **Full-sky peaks scales1234** (the missing 4th probe in the full-ℓ full-sky plot) — run NPE if a
  full-ℓ peaks comparison is wanted; and seed-average the NPE FoMs (single-run width scatter ~15–20%)
  before any paper number.
- Plot builder paths gotcha: masked PS dirs are under `outputs/diagnostics/lmin_compare/` (include
  `outputs/samples` for the HOS).
