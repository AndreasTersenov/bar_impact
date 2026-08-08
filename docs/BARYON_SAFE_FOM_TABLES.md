# Baryon-safe FoM tables (Table 3 candidates)

**Generated** 2026-08-07 — repo git commit `3d39df81` plus 9 fresh NPE seeds from
SLURM job 687543 (see [Provenance](#provenance)).

Two cut rules, both defensible, laid out side by side so the referee response can
pick one after seeing both. Numbers use `NULL` posteriors only (nobaryons vs
nobaryons — constraining power, not bias); the `NULL`-to-biased shift at the same
cut is the separate baryon-safety verification and is reported in the
`bar_impact_paper_figure_map §3.1`.

Companion CSV files live alongside this doc under
[`docs/tables/`](tables/) — one per (rule × errbar) combination, keyed
`(statistic, area)` with `fom3_pooled`, `fom3_mean`, `fom3_std`, `n_seeds`, and
the exact `runs` used per cell.

---

## The two rules

For each area, the PS scale cut is the largest step-40 upper cut that still
counts as baryon-safe under the rule.

| rule | criterion | when it kicks in |
|---|---|---|
| `mean`     (**strict**) | cell's mean bias < 0.3 σ | matches the audit script's default |
| `errorbar` (**loose**)  | mean − 1σ < 0.3 σ (interval still reaches the threshold) | the rule the submitted table quietly used at 2000 / 5000 / full-sky |

The HOS cut is `scales234` (finest wavelet scale dropped) regardless of rule —
wavelet scales are quantised, so there is no next-finer setting to try. Whether
`scales234` really clears 0.3σ at every area is a separate check (see
[Safety verification](#safety-verification-hos-scales234) below).

### PS cuts by rule

| area | `mean` (strict) | `errorbar` (loose) |
|---|---:|---:|
| 2,000  deg² | ℓ37–**860** | ℓ37–940 |
| 5,000  deg² | ℓ37–**580** | ℓ37–620 |
| 10,000 deg² | ℓ37–**540** | ℓ37–540 |
| 14,000 deg² | ℓ37–**460** | ℓ37–460 |
| 28,000 deg² | ℓ37–**380** | ℓ37–380 |
| 35,000 deg² | ℓ37–**340** | ℓ37–380 |
| full sky    | ℓ37–**300** | ℓ37–340 |

Rules agree at 10,000, 14,000, 28,000 — only the small (2,000; 5,000; 35,000)
and full-sky columns move.

---

## Table 3 — STRICT rule (`--rule mean`)

Units: FoM₃ / 10⁴. Numbers in parentheses are the HOS / PS ratio ± propagated
error.

### SEM error bars (`std / √n`, "uncertainty on the quoted value")

| Statistic | 2,000 deg² | 5,000 deg² | 10,000 deg² | 14,000 deg² | 28,000 deg² | 35,000 deg² | Full Sky |
|---|---|---|---|---|---|---|---|
| PS | **2.3 ± 0.1** | **5.0 ± 0.3** | 11.7 ± 0.7 | 14.5 ± 0.4 | 22.0 ± 0.6 | **20.6 ± 0.7** | **39.1 ± 2.7** |
| Starlet peak counts | 1.1 ± 0.1 (×**0.46 ± 0.03**) | 4.1 ± 0.2 (×**0.81 ± 0.06**) | 9.5 ± 0.8 (×0.81 ± 0.09) | 15.5 ± 1.4 (×1.07 ± 0.10) | 39.6 ± 4.5 (×1.80 ± 0.21) | 69.7 ± 7.0 (×**3.38 ± 0.36**) | 64.0 ± 3.2 (×**1.63 ± 0.14**) |
| Starlet ℓ₁-norm | 2.7 ± 0.1 (×**1.15 ± 0.07**) | 10.3 ± 0.2 (×**2.05 ± 0.12**) | 21.1 ± 3.0 (×1.80 ± 0.28) | 26.1 ± 3.7 (×1.80 ± 0.26) | 79.5 ± 7.1 (×3.62 ± 0.34) | 113.9 ± 6.0 (×**5.53 ± 0.35**) | 143.4 ± 9.6 (×**3.67 ± 0.35**) |

### STD error bars (per-seed scatter, "what one retraining would give")

| Statistic | 2,000 deg² | 5,000 deg² | 10,000 deg² | 14,000 deg² | 28,000 deg² | 35,000 deg² | Full Sky |
|---|---|---|---|---|---|---|---|
| PS | 2.3 ± 0.3 | 5.0 ± 0.6 | 11.7 ± 1.6 | 14.5 ± 0.9 | 22.0 ± 1.3 | 20.6 ± 1.6 | 39.1 ± 6.1 |
| Starlet peak counts | 1.1 ± 0.1 (×0.46 ± 0.08) | 4.1 ± 0.5 (×0.81 ± 0.14) | 9.5 ± 2.1 (×0.81 ± 0.21) | 15.5 ± 3.5 (×1.07 ± 0.25) | 39.6 ± 10.1 (×1.80 ± 0.47) | 69.7 ± 15.7 (×3.38 ± 0.81) | 64.0 ± 7.2 (×1.63 ± 0.31) |
| Starlet ℓ₁-norm | 2.7 ± 0.2 (×1.15 ± 0.18) | 10.3 ± 0.4 (×2.05 ± 0.26) | 21.1 ± 6.8 (×1.80 ± 0.63) | 26.1 ± 8.2 (×1.80 ± 0.58) | 79.5 ± 15.8 (×3.62 ± 0.75) | 113.9 ± 13.3 (×5.53 ± 0.78) | 143.4 ± 21.4 (×3.67 ± 0.79) |

CSV: [`tables/baryon_safe_fom_mean_sem.csv`](tables/baryon_safe_fom_mean_sem.csv) ·
[`tables/baryon_safe_fom_mean_std.csv`](tables/baryon_safe_fom_mean_std.csv).
Full audit log with per-cell diagnostics: [`tables/_run_mean_sem.log`](tables/_run_mean_sem.log).

---

## Table 3 — LOOSE rule (`--rule errorbar`, matches the SUBMITTED table)

### SEM error bars

| Statistic | 2,000 deg² | 5,000 deg² | 10,000 deg² | 14,000 deg² | 28,000 deg² | 35,000 deg² | Full Sky |
|---|---|---|---|---|---|---|---|
| PS | 2.7 ± 0.1 | 6.4 ± 0.3 | 11.7 ± 0.7 | 14.5 ± 0.4 | 22.0 ± 0.6 | 28.9 ± 1.8 | 54.9 ± 1.3 |
| Starlet peak counts | 1.1 ± 0.1 (×0.39 ± 0.03) | 4.1 ± 0.2 (×0.63 ± 0.05) | 9.5 ± 0.8 (×0.81 ± 0.09) | 15.5 ± 1.4 (×1.07 ± 0.10) | 39.6 ± 4.5 (×1.80 ± 0.21) | 69.7 ± 7.0 (×2.41 ± 0.28) | 64.0 ± 3.2 (×1.17 ± 0.06) |
| Starlet ℓ₁-norm | 2.7 ± 0.1 (×0.99 ± 0.05) | 10.3 ± 0.2 (×1.60 ± 0.09) | 21.1 ± 3.0 (×1.80 ± 0.28) | 26.1 ± 3.7 (×1.80 ± 0.26) | 79.5 ± 7.1 (×3.62 ± 0.34) | 113.9 ± 6.0 (×3.94 ± 0.32) | 143.4 ± 9.6 (×2.61 ± 0.18) |

### STD error bars

| Statistic | 2,000 deg² | 5,000 deg² | 10,000 deg² | 14,000 deg² | 28,000 deg² | 35,000 deg² | Full Sky |
|---|---|---|---|---|---|---|---|
| PS | 2.7 ± 0.2 | 6.4 ± 0.7 | 11.7 ± 1.6 | 14.5 ± 0.9 | 22.0 ± 1.3 | 28.9 ± 4.0 | 54.9 ± 2.9 |
| Starlet peak counts | 1.1 ± 0.1 (×0.39 ± 0.06) | 4.1 ± 0.5 (×0.63 ± 0.11) | 9.5 ± 2.1 (×0.81 ± 0.21) | 15.5 ± 3.5 (×1.07 ± 0.25) | 39.6 ± 10.1 (×1.80 ± 0.47) | 69.7 ± 15.7 (×2.41 ± 0.64) | 64.0 ± 7.2 (×1.17 ± 0.14) |
| Starlet ℓ₁-norm | 2.7 ± 0.2 (×0.99 ± 0.12) | 10.3 ± 0.4 (×1.60 ± 0.18) | 21.1 ± 6.8 (×1.80 ± 0.63) | 26.1 ± 8.2 (×1.80 ± 0.58) | 79.5 ± 15.8 (×3.62 ± 0.75) | 113.9 ± 13.3 (×3.94 ± 0.71) | 143.4 ± 21.4 (×2.61 ± 0.41) |

CSV: [`tables/baryon_safe_fom_errorbar_sem.csv`](tables/baryon_safe_fom_errorbar_sem.csv) ·
[`tables/baryon_safe_fom_errorbar_std.csv`](tables/baryon_safe_fom_errorbar_std.csv).
Full audit log: [`tables/_run_errorbar_sem.log`](tables/_run_errorbar_sem.log).

---

## What moves between rules

The HOS FoMs are the same in both tables — `scales234` is the same set of
wavelet coefficients regardless of the PS cut choice. So only the PS row and
the HOS/PS ratios change, and they change in the columns where the rules
disagree on the PS cut:

| column | strict PS | loose PS | Δ PS | peaks ratio Δ | L1 ratio Δ |
|---|---:|---:|---:|---:|---:|
| 2,000  | 2.3 | 2.7 | −0.4 (×0.85) | 0.46 → 0.39 (+18%) | 1.15 → 0.99 (+16%) |
| 5,000  | 5.0 | 6.4 | −1.4 (×0.78) | 0.81 → 0.63 (+29%) | 2.05 → 1.60 (+28%) |
| 35,000 | 20.6 | 28.9 | −8.3 (×0.71) | 3.38 → 2.41 (+40%) | 5.53 → 3.94 (+40%) |
| full sky | 39.1 | 54.9 | −15.8 (×0.71) | 1.63 → 1.17 (+39%) | 3.67 → 2.61 (+41%) |

**Direction is the same everywhere:** stricter PS cut → smaller PS FoM → larger
HOS advantage. The magnitude at full sky (peaks ratio 1.17 → 1.63; L1 2.61 →
3.67) is the biggest single number in this document — the L1 → 3.67× headline
under strict is the same underlying data as the L1 → 2.61× headline under loose;
only the denominator changed.

---

## Safety verification (HOS scales234)

The HOS row is only meaningful if `scales234` is itself baryon-safe at each
area. From `scripts/diagnostics/check_hos_cut_is_safe.py` (verdicts in
`outputs/diagnostics/hos_cut_safety_14001.json` and per-area siblings), the
measured 3-param `Q_DM` bias in σ:

| area | peaks (scales234) | L1 (scales234) | note |
|---|---:|---:|---|
| 14,000 | **0.079 ± 0.047** | **0.091 ± 0.073** | reference footprint — both strictly < 0.3σ |
| full sky | 0.344 ± 0.183 | 0.379 ± 0.151 | above 0.3σ mean, tolerated only on the error-bar rule |

At full sky the mean bias for the HOS matches the PS on the loose rule (both
around 0.34–0.38σ), and drops below 0.3σ only on the mean − σ interpretation.
So the full-sky HOS/PS comparison is **only self-consistent under the loose
rule** — mixing rules there would compare a strict-safe PS to a
loose-safe HOS, which is what the previous table incidentally did.

---

## Per-cell diagnostics

Both the pooled and per-seed FoMs, plus the number of seeds and which runs
were used. Runs surviving after RAID0 disk-failure drops. Same numbers as in
the CSVs, but visible here for the referee response draft.

### `mean` rule

| statistic | area | cut | n_seeds | runs | FoM₃ (pooled) | FoM₃ (per-seed mean ± std) |
|---|---|---|---:|---|---:|---:|
| PS     | 2,000  | ℓ37–860  | 8 | 1 2 3 4 5 6 7 8 | 2.28×10⁴ | (2.33 ± 0.29)×10⁴ |
| peaks  | 2,000  | scales234 | 6 | 1 1 2 3 4 5 | 1.04×10⁴ | (1.06 ± 0.14)×10⁴ |
| L1     | 2,000  | scales234 | 5 | 1 2 3 4 5 | 2.61×10⁴ | (2.68 ± 0.24)×10⁴ |
| PS     | 5,000  | ℓ37–580  | 5 | 1 2 3 4 5 | 4.88×10⁴ | (5.01 ± 0.61)×10⁴ |
| peaks  | 5,000  | scales234 | 6 | 1 1 2 3 4 5 | 3.91×10⁴ | (4.06 ± 0.53)×10⁴ |
| L1     | 5,000  | scales234 | 5 | 1 2 3 4 5 | 9.91×10⁴ | (10.29 ± 0.38)×10⁴ |
| PS     | 10,000 | ℓ37–540  | 5 | 1 2 3 4 5 | 1.13×10⁵ | (1.17 ± 0.16)×10⁵ |
| peaks  | 10,000 | scales234 | 6 | 1 1 2 3 4 5 | 8.86×10⁴ | (9.51 ± 2.07)×10⁴ |
| L1     | 10,000 | scales234 | 5 | 1 2 3 4 5 | 1.80×10⁵ | (2.11 ± 0.68)×10⁵ |
| PS     | 14,000 | ℓ37–460  | 5 | 1 2 3 4 5 | 1.42×10⁵ | (1.45 ± 0.09)×10⁵ |
| peaks  | 14,000 | scales234 | 6 | 1 1 2 3 4 5 | 1.42×10⁵ | (1.55 ± 0.35)×10⁵ |
| L1     | 14,000 | scales234 | 5 | 1 2 3 4 5 | 2.24×10⁵ | (2.61 ± 0.82)×10⁵ |
| PS     | 28,000 | ℓ37–380  | 5 | 1 2 3 4 5 | 2.11×10⁵ | (2.20 ± 0.13)×10⁵ |
| peaks  | 28,000 | scales234 | 5 (r1 dropped) | 2 3 4 5 6 | 3.55×10⁵ | (3.96 ± 1.01)×10⁵ |
| L1     | 28,000 | scales234 | 5 | 1 2 3 4 5 | 6.71×10⁵ | (7.95 ± 1.58)×10⁵ |
| PS     | 35,000 | ℓ37–340  | 5 | 1 2 3 4 5 | 2.01×10⁵ | (2.06 ± 0.16)×10⁵ |
| peaks  | 35,000 | scales234 | 5 (r1 dropped) | 2 3 4 5 6 | 6.28×10⁵ | (6.97 ± 1.57)×10⁵ |
| L1     | 35,000 | scales234 | 5 | 1 2 3 4 5 | 1.09×10⁶ | (1.14 ± 0.13)×10⁶ |
| PS     | full sky | ℓ37–300 | 5 | 1 2 3 4 5 | 3.54×10⁵ | (3.91 ± 0.61)×10⁵ |
| peaks  | full sky | scales234 | 5 | 1 2 3 4 5 | 5.93×10⁵ | (6.40 ± 0.72)×10⁵ |
| L1     | full sky | scales234 | 5 | 1 2 3 4 5 | 1.31×10⁶ | (1.43 ± 0.21)×10⁶ |

### `errorbar` rule — only the PS row where the cut differs from the `mean` rule

Cells not shown here are identical to the strict table above.

| statistic | area | cut | n_seeds | runs | FoM₃ (pooled) | FoM₃ (per-seed mean ± std) |
|---|---|---|---:|---|---:|---:|
| PS | 2,000 | ℓ37–**940** | 4 (r3 corrupted) | 1 2 4 5 | 2.60×10⁴ | (2.70 ± 0.20)×10⁴ |
| PS | 5,000 | ℓ37–**620** | 4 (r3 corrupted) | 1 2 4 5 | 6.26×10⁴ | (6.43 ± 0.69)×10⁴ |
| PS | 35,000 | ℓ37–**380** | 5 | 1 2 3 4 5 | 2.71×10⁵ | (2.89 ± 0.40)×10⁵ |
| PS | full sky | ℓ37–**340** | 5 | 1 2 3 4 5 | 4.80×10⁵ | (5.49 ± 0.29)×10⁵ |

---

## Provenance

**Generator.** `scripts/paper/audit_baryon_safe_fom.py` (see docstring for the
FoM definition, the PS-cut rule, and the two error-bar conventions).

**How this run was launched.**
```
for RULE in mean errorbar; do
  for ERR in sem std; do
    PYTHONNOUSERSITE=1 /lustre/fswork/projects/rech/nzu/ulx34io/envs/jaxili/bin/python \
      scripts/paper/audit_baryon_safe_fom.py \
      --areas 2000 5000 10000 14000 28000 35000 fullsky \
      --rule $RULE --errbar $ERR --table --latex \
      --csv docs/tables/baryon_safe_fom_${RULE}_${ERR}.csv \
      > docs/tables/_run_${RULE}_${ERR}.log 2>&1
  done
done
```

**Repo state.** git commit `3d39df81`, "appendix: publish the starlet
scale→ell figure, and rebuild the noiseless spectra" (2026-08-05).

**Software.** jaxili conda env at
`/lustre/fswork/projects/rech/nzu/ulx34io/envs/jaxili`; numpy 1.26.4, scipy
1.17.1, matplotlib 3.11.1.

**Seed top-up.** SLURM job `687543` on 2026-08-07 ran 9 fresh NPE trainings:
regenerated 3 RAID0-destroyed posteriors (2000/860 null r4, 2000/860 biased
r3, 5000/580 biased r1) and trained 3 fresh pairs at 2000/860 (runs 6, 7, 8).
That took the 2000/860 seed pool from 3 → **8 pairs** and 5000/580 from 4 →
**5 pairs**. See `scripts/jz/npe_ps_2000_lmax860_more_seeds.slurm` for the
job script and `logs/npe_ps_2000_more_687543.out` for the run log.

**Motivation for the top-up.** In the 3-seed pool, the biased-arm FoM₃ error
at 2000/860 came out at 0.04×10⁴ — implausibly tight compared with the null
(0.12) and full-res (0.16). With 8 seeds it is **0.20×10⁴**, comparable to
its neighbours. The tightness was a sampling fluke, not a real feature.

**PS cut source.** `outputs/baryon_tension/ps_submean_l37/tables/tension_3param_agg.csv`
(masked areas) and `outputs/baryon_tension/ps_fullsky_l37/tables/tension_3param_agg.csv`
(full sky). The `crossings.csv` sitting next to
`paper/figures/ps_bias_vs_lmax/values.csv` is the same information in a more
readable form.

**Posterior source.**
- PS masked: `outputs/baryon_tension/ps_submean_l37/posteriors/mask_XXXXX/null/`
- PS full sky: `outputs/baryon_tension/ps_fullsky_l37/posteriors/fullsky/null/`
- Peaks / L1: `outputs/samples/posterior_samples_pc_..._scales234_..._new_normalization*_npe.npy`
  and the same pattern without the `pc_` prefix for L1

**Prior-collapse guard.** Any seed with σ(σ8) ≥ 0.08 is dropped from that
cell's pool. This is the same threshold `plot_nsigma_vs_area.py` uses.

**Caveats specific to this table** (from `audit_baryon_safe_fom.py`):

- FoM₃ is `1 / √det Cov[Ωm, σ8, w₀]`. It is not a bound, and it treats the
  three parameters as of equal analytical interest — which the paper argues
  they are for a w-CDM baryonic-feedback study.
- Pooled and per-seed-mean FoM answer different questions. Per-seed-mean is
  used in the tables; pooled is in the CSV so it can be cited separately.
- Ratios treat the peaks/PS and L1/PS training scatters as uncorrelated —
  they are trained independently, so this is right. What it does not capture
  is that the biased arm at every column shares one fiducial realisation; a
  fiducial-perturbation covariance would enlarge every error bar coherently.
- The mixed rule (strict for 2000 / 5000, loose for full sky) is exactly what
  the SUBMITTED Table 3 quietly did and is what the `errorbar` rule reproduces
  when full-sky ℓmax=340 is honestly labelled loose. This document keeps rule
  consistency within each block so the referee can compare like for like.
