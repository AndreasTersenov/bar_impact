#!/usr/bin/env python3
"""Cut-aware score/MOPED machinery for the BNT-bin-1 tension-vs-scale-cut sweep.

The validated single-cut pipeline (`score_compress.py`) builds the Fisher J (local order-2) and
hybrid covariance C at ONE hard-coded cut. This module parameterizes the cut so the BNT bin-1
ℓmax can be swept, while reusing the *exact same* fisher code paths (`fisher_local_jacobian`,
`fisher_hybrid_cov`) — only the `cuts` argument changes.

Two pieces:
  - `keep_indices(cuts)`: which columns of the FULL (ℓmax=1024 on every bin) rebinned data vector
    survive a given per-bin cut. Cutting only truncates the tail rebinned bins of each spectrum
    (the rebin is anchored at the ℓ-floor, so a cut keeps the first n_s rebinned bins of block s).
    This lets us dump the grid ONCE at full and slice to any cut instead of re-dumping 18×.
  - `build_score(cuts, bnt, covk)`: J, C, Wmle for the MLE-form score compression at `cuts`,
    identical in construction to `score_compress.compression_cov` + the Wmle line.

FISHER_AREA must be set in the environment before import (the fisher modules read it at import).
numpy only; runs under the jaxili interpreter.
"""
import os
import sys

import numpy as np

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diagnostics")
sys.path.insert(0, HERE)
import fisher_local_jacobian as L          # noqa: E402  build_config, local_jacobian, FID, datavector, load_set
import fisher_hybrid_cov as H              # noqa: E402  C_ANA, bnt_spectra_operator, cut_rebin_R, per_spectrum_uppers, NBPW

FID = L.FID
ORDER, BW, KHYB = "order2", 0.75, 3
FULL_CUTS = [1024, 1024, 1024, 1024]

# Exact BNT native analytic covariance (T⊗I applied to the native Gaussian cov), as in score_compress.
_Tfull = np.kron(H.bnt_spectra_operator(), np.eye(H.NBPW))
C_ANA_BNT = _Tfull @ H.C_ANA @ _Tfull.T


def _nbins(upper):
    """Number of rebinned bins kept for a single spectrum cut at multipole `upper`.

    Matches `fisher_hybrid_cov.cut_rebin_R` / `fisher_bnt_vs_nonbnt.cut_rebin`: native bins
    [lo, hi) with lo from the ℓ-floor and hi from `upper`, grouped REBIN at a time (floor).
    """
    lo = max(0, int((H.LOWER - H.ELL_OFFSET) / H.ELL_PER_BIN))
    hi = min(H.NBPW, int((upper - H.ELL_OFFSET) / H.ELL_PER_BIN))
    return max(0, (hi - lo) // H.REBIN)


def keep_indices(cuts):
    """Column indices of the full (FULL_CUTS) rebinned data vector that survive per-bin `cuts`.

    Spectrum order is `fisher_hybrid_cov.SPECTRA` = 4 autos + 6 cross PAIRS, which is exactly the
    concatenation order of `fisher_local_jacobian.datavector`. Each block keeps its first n_s(cut)
    rebinned bins; bins 2-4 autos and the 2-3-4 crosses are full, bin-1 auto and the three bin-1
    crosses are truncated (cross x-cut = min, handled by `per_spectrum_uppers`).
    """
    ups_full = H.per_spectrum_uppers(FULL_CUTS)
    ups_cut = H.per_spectrum_uppers(list(cuts))
    keep, off = [], 0
    for uf, uc in zip(ups_full, ups_cut):
        nf, nc = _nbins(uf), _nbins(uc)
        keep.extend(range(off, off + nc))
        off += nf
    return np.asarray(keep, dtype=int)


def hybrid_cov(cuts, bnt, k=KHYB):
    """Hybrid covariance C = analytic Gaussian + top-k SSC/cNG eigenmodes, at `cuts`.

    Identical to `score_compress.compression_cov(..., kind='hybrid')`. Returns (C, perms) where
    `perms` are the 200 nobaryons fiducial realizations (compressed later for the empirical check).
    """
    R = H.cut_rebin_R(H.per_spectrum_uppers(list(cuts)))
    Cana = R @ (C_ANA_BNT if bnt else H.C_ANA) @ R.T
    fa, fc, nell = L.load_set("fiducial", "nobaryons", bnt)
    perms = L.datavector(fa, fc, nell, list(cuts))                 # (200, nfeat)
    Csamp = np.cov(perms, rowvar=False)
    D = Csamp - Cana
    ev, V = np.linalg.eigh(D)
    idx = np.argsort(ev)[::-1][:k]
    C = Cana + (V[:, idx] * ev[idx]) @ V[:, idx].T
    return C, perms


def build_score(cuts, bnt, covk="hybrid"):
    """J, C, Wmle, F for the MLE-form score compression at `cuts`.

    covk='hybrid' (default, the validated production estimator: analytic Gaussian + top-k SSC/cNG)
    or 'analytic' (Gaussian only — used for the exact lossless-identity gate, since the analytic
    cov transforms exactly as A C A^T whereas the hybrid eigen-truncation is basis-dependent).

    Wmle = C^{-1} J F^{-1} maps a (centered) data vector to the 6 quasi-MLE parameter summaries:
    theta_hat = FID + (x - x_fid) @ Wmle. By construction F = J^T C^{-1} J is the full Fisher.
    """
    cfg = L.build_config(list(cuts), bnt)
    J, _ = L.local_jacobian(cfg["grid_avg"], cfg["ucos"], cfg["fid_mean"], ORDER, BW)  # (nfeat,6)
    if covk == "hybrid":
        C, perms = hybrid_cov(cuts, bnt)
    elif covk == "analytic":
        R = H.cut_rebin_R(H.per_spectrum_uppers(list(cuts)))
        C = R @ (C_ANA_BNT if bnt else H.C_ANA) @ R.T
        perms = None
    else:
        raise ValueError(f"covk={covk!r} not supported (use 'hybrid' or 'analytic')")
    Cinv = np.linalg.inv(C)
    F = J.T @ Cinv @ J
    Wmle = (Cinv @ J) @ np.linalg.inv(F)
    return dict(J=J, C=C, F=F, Wmle=Wmle, perms=perms, nfeat=J.shape[0])
