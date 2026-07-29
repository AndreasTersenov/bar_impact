#!/usr/bin/env python3
"""Phase I follow-up: characterize the native band-to-band covariance that the rebinned check (V2)
flagged. V1 showed the analytic Gaussian DIAGONAL matches the sims (1-2% at ell>100); V2 showed the
REBINNED variance is ~23% larger in the sims and far more correlated. That gap must live in the
OFF-diagonal (band-to-band) covariance the Gaussian model omits. Two candidate origins, opposite
implications:
  (a) NON-GAUSSIANITY (trispectrum): real, grows with ell -> the analytic Gaussian UNDER-estimates the
      true covariance, the Fisher would be over-optimistic, and we must add a cNG/sample term.
  (b) CORRELATED PERMUTATIONS (sim suite shares LSS across perms): artifact, ell-independent-ish ->
      the 200-perm sample OVER-estimates the covariance and the analytic Gaussian is the cleaner choice.

Discriminator: the mean ADJACENT-band correlation vs ell, analytic vs sample. Averaging over many band
pairs beats down the 1/sqrt(200)~0.07 sample noise to <1%, so a true rising-with-ell signal is
measurable. Non-Gaussianity rises with ell; a perm artifact is roughly flat.

Also: a self-consistency check (sample rebinned var from native-sample R C R^T == direct rebinned var),
to confirm the V2 gap is genuinely the native off-diagonals and not a rebin-operator bug.

numpy only; uses the saved analytic native covariance + the per-perm fiducial bandpowers.
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from fisher_bnt_vs_nonbnt import load_set, PAIRS, REBIN, LOWER, ELL_OFFSET, ELL_PER_BIN  # noqa: E402
import fisher_gaussian_cov as G  # noqa: E402

AREA = 14000
C = np.load(os.path.join(HERE, "cache_gaussian_cov", f"gaussian_cov_native_{AREA}.npy"))
NBPW = 383
EFF = ELL_OFFSET + ELL_PER_BIN * (np.arange(NBPW) + 0.5)        # bandpower effective ell ~ 3.5,7.5,...
SPECTRA = G.SPECTRA


def perm_data():
    fa, fc, nell = load_set("fiducial", "nobaryons", bnt=False)
    d = {}
    for k, (i, _) in enumerate(G.AUTOS):
        d[(i, i)] = fa[k]
    for k, (i, j) in enumerate(PAIRS):
        d[(i, j)] = fc[:, k * nell:(k + 1) * nell]
    return d


def mean_offdiag_corr(cov, lag, ell_lo, ell_hi):
    """mean correlation between bandpowers k and k+lag for eff_ell in [ell_lo,ell_hi]."""
    d = np.sqrt(np.diag(cov))
    sel = np.where((EFF >= ell_lo) & (EFF < ell_hi))[0]
    sel = sel[sel + lag < NBPW]
    if len(sel) == 0:
        return np.nan
    r = cov[sel, sel + lag] / (d[sel] * d[sel + lag] + 1e-300)
    return np.mean(r)


def main():
    pdata = perm_data()
    bands = [(37, 100), (100, 200), (200, 400), (400, 700), (700, 1100), (1100, 1531)]

    print("=== adjacent-band (lag=1) correlation: analytic Gaussian vs 200-perm sample ===")
    print("(non-Gaussianity => sample rises with ell; perm artifact => flat. analytic ~ mask coupling)")
    print(f"{'spectrum':>9} | " + " ".join(f"{lo:>4}-{hi:<4}" for lo, hi in bands))
    for a, (i, j) in enumerate(SPECTRA):
        blk_a = C[a * NBPW:(a + 1) * NBPW, a * NBPW:(a + 1) * NBPW]
        blk_s = np.cov(pdata[(min(i, j), max(i, j))], rowvar=False)
        ra = [mean_offdiag_corr(blk_a, 1, lo, hi) for lo, hi in bands]
        rs = [mean_offdiag_corr(blk_s, 1, lo, hi) for lo, hi in bands]
        print(f"  ({i},{j}) ana | " + " ".join(f"{x:>9.3f}" for x in ra))
        print(f"  ({i},{j}) sim | " + " ".join(f"{x:>9.3f}" for x in rs))

    # self-consistency: sample rebinned var two ways, + analytic, for auto(4,4)
    print("\n=== self-consistency (auto(4,4), rebin=20, ell>=37): isolate the gap to native off-diags ===")
    a = SPECTRA.index((4, 4))
    blk_a = C[a * NBPW:(a + 1) * NBPW, a * NBPW:(a + 1) * NBPW]
    x = pdata[(4, 4)]                                            # (200, 383)
    lo = max(0, int((LOWER - ELL_OFFSET) / ELL_PER_BIN)); hi = int((1024 - ELL_OFFSET) / ELL_PER_BIN)
    n = (hi - lo) // REBIN
    R = np.zeros((n, NBPW))
    for k in range(n):
        R[k, lo + k * REBIN: lo + (k + 1) * REBIN] = 1.0 / REBIN
    dv = x @ R.T
    var_direct = dv.var(axis=0, ddof=1)                         # rebinned var from data
    var_samp_native = np.diag(R @ np.cov(x, rowvar=False) @ R.T)  # from native sample cov
    var_ana = np.diag(R @ blk_a @ R.T)                          # from analytic native cov
    print(f"  rebin var direct vs native-sample R C R^T  max|rel diff| {np.max(np.abs(var_direct/var_samp_native-1)):.2e} (==0 => no rebin bug)")
    print(f"  analytic / sample rebinned var: median {np.median(var_ana/var_direct):.3f}")
    # diag-only analytic (drop native off-diags) vs full analytic: how much does mask coupling add?
    var_ana_diagonly = np.diag(R @ np.diag(np.diag(blk_a)) @ R.T)
    print(f"  analytic(diag-only) / analytic(full): median {np.median(var_ana_diagonly/var_ana):.3f} "
          f"(near 1 => analytic has little band coupling; the gap is the SIMS' extra coupling)")


if __name__ == "__main__":
    main()
