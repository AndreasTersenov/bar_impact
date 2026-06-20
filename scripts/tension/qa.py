"""Posterior quality-assurance gate for the unattended sweep.

Two levels:

1. `assess_posterior` — single-posterior diagnostics. Cheap checks that catch a failed NPE
   fit before its samples are trusted: non-finite, collapsed (σ≈0), unconstrained (σ≈prior),
   prior-railing, and — for the NULL only — mean far from truth (the observation is the
   ≈noiseless perm-mean, so a biased null is a failed fit, not physics).

2. `flag_outlier_runs` — across the runs of one configuration, flag any whose summary is far
   from the run-median (modified-z / MAD rule). This is the principled replacement for the
   old hard-coded "always use run3 for mask 5000" patch.

Thresholds are deliberately loose (catch gross failures, not natural NPE scatter) and are
tuned on the Phase-3 pilot; they live here as named constants.
"""
from typing import Optional, Sequence

import numpy as np

# Single-posterior thresholds (fractions of the prior range unless noted).
COLLAPSE_FRAC = 0.005      # σ < COLLAPSE_FRAC · prior_width            -> collapsed fit
# (0.02 was too aggressive: the largest footprints (28000/35000) constrain S8 to ~1.5% of the
#  prior legitimately and were false-flagged. A real collapse has σ≈0. Calibrated on the
#  all-six run 2026-06-20: tightest real S8 σ=0.0144 vs prior 0.997 → 1.4%, so 0.5% is safe.)
UNCONSTRAINED_FRAC = 0.90  # σ > UNCONSTRAINED_FRAC · σ_uniform(prior)  -> ≈ prior (no info)
RAIL_EPS_FRAC = 0.02       # "at an edge" = within this · prior_width of a bound
RAIL_MASS_FRAC = 0.30      # > this fraction of samples at an edge      -> railing
NULL_TRUTH_NSIGMA = 5.0    # null |mean − truth| must be < this · σ

# Cross-run outlier rule.
OUTLIER_NMAD = 3.5         # modified z-score (0.6745·|x−med|/MAD) above this -> outlier


def assess_posterior(
    samples: np.ndarray,
    *,
    role: str,
    truth: Sequence[float],
    prior_lo: Optional[Sequence[float]] = None,
    prior_hi: Optional[Sequence[float]] = None,
    subset: Sequence[int] = (0, 1, 2),
) -> dict:
    """Diagnose one posterior. Returns {status, ok, reasons, mean, std} for the subset params."""
    idx = list(subset)
    reasons = []

    if not np.all(np.isfinite(samples)):
        # Non-finite samples poison every other statistic; stop here.
        return _record("FLAG", ["nonfinite_samples"], np.full(len(idx), np.nan),
                       np.full(len(idx), np.nan))

    s = samples[:, idx]
    t = np.asarray(truth, float)[idx]
    mean, std = s.mean(0), s.std(0)

    if np.any(std <= 0):
        reasons.append("zero_std")

    if prior_lo is not None and prior_hi is not None:
        lo = np.asarray(prior_lo, float)[idx]
        hi = np.asarray(prior_hi, float)[idx]
        width = hi - lo
        sigma_uniform = width / np.sqrt(12.0)
        if np.any(std < COLLAPSE_FRAC * width):
            reasons.append("collapsed")
        if np.any(std > UNCONSTRAINED_FRAC * sigma_uniform):
            reasons.append("unconstrained")
        eps = RAIL_EPS_FRAC * width
        at_edge = np.maximum((s <= lo + eps).mean(0), (s >= hi - eps).mean(0))
        if np.any(at_edge > RAIL_MASS_FRAC):
            reasons.append("prior_railing")

    if role == "null":
        safe_std = np.where(std > 0, std, np.inf)
        if np.any(np.abs(mean - t) / safe_std > NULL_TRUTH_NSIGMA):
            reasons.append("null_off_truth")

    status = "OK" if not reasons else "FLAG"
    return _record(status, reasons, mean, std)


def _record(status, reasons, mean, std) -> dict:
    return {
        "status": status,
        "ok": status == "OK",
        "reasons": ";".join(reasons),
        "mean": np.asarray(mean).tolist(),
        "std": np.asarray(std).tolist(),
    }


def flag_outlier_runs(values: Sequence[float], n_mad: float = OUTLIER_NMAD) -> np.ndarray:
    """Boolean mask over runs: True where a run is a MAD outlier vs the run-median.

    Needs ≥3 runs to judge; returns all-False otherwise. MAD==0 (identical runs) -> all-False.
    """
    v = np.asarray(values, float)
    if v.size < 3:
        return np.zeros(v.size, dtype=bool)
    med = np.median(v)
    mad = np.median(np.abs(v - med))
    if mad == 0:
        return np.zeros(v.size, dtype=bool)
    modified_z = 0.6745 * np.abs(v - med) / mad
    return modified_z > n_mad


def flag_outlier_posteriors(means, stds, n_mad: float = OUTLIER_NMAD) -> np.ndarray:
    """Flag runs whose posterior summary is an outlier vs siblings.

    `means`, `stds` are (n_runs, n_subset) arrays of per-run subset means and widths. A run
    is flagged if ANY parameter's mean OR width is a MAD outlier across the runs. Width is the
    most discriminating signal: a failed NPE fit is typically several× too broad (e.g. the
    stale 14000 l37-1024 null with σ(S8)=0.20 vs the good 0.024) and stands out immediately.
    """
    means = np.atleast_2d(np.asarray(means, float))
    stds = np.atleast_2d(np.asarray(stds, float))
    flagged = np.zeros(means.shape[0], dtype=bool)
    for col in range(means.shape[1]):
        flagged |= flag_outlier_runs(means[:, col], n_mad)
        flagged |= flag_outlier_runs(stds[:, col], n_mad)
    return flagged
