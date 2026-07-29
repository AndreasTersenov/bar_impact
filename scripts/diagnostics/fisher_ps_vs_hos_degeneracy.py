"""Is the w0 degeneracy sign-flip (PS vs HOS) in the DATA or introduced by the NPE?

Build a Gaussian Fisher forecast directly from the grid data vectors + cosmo_params for
PS (l100-400, auto+cross, full-sky healpy) and for the HOS (l1 / peaks, scales234), using
the SAME deterministic walk alignment. Compare the Fisher parameter CORRELATION matrices
(esp. the sign of Om-w0 and S8-w0) to the NPE posteriors. If Fisher reproduces the flip,
it is in the data (physical); if Fisher gives the same sign for PS and HOS but the NPE
flips, the flip is NPE/processing-side.

Moderate rebinning keeps n_features < n_fid-2 so the covariance is invertible (Hartlap).
"""
import numpy as np

BASE = "/lustre/fsn1/projects/rech/prk/ulx34io/cosmogrid_products/stage3_forecast"
GRID = f"{BASE}/new_grid"; GRIDg = f"{BASE}/grid"
FID = f"{BASE}/fiducial/cosmo_fiducial"
params = np.load(f"{BASE}/grid/cosmo_params.npy")            # (16965, 6)
PN = ["Om", "S8", "w0", "H0", "ns", "Ob"]


def rebin_last(a, f):
    n = a.shape[-1] // f
    return a[..., :n * f].reshape(*a.shape[:-1], n, f).mean(-1)


def fisher_corr(Dg, par, Dfid, label):
    """Linear Jacobian (lstsq) + Hartlap sample cov -> param correlation matrix."""
    # drop near-constant features (variance floor) so C is well-conditioned
    v = Dfid.var(0)
    keep = v > (v.max() * 1e-8)
    Dg, Dfid = Dg[:, keep], Dfid[:, keep]
    n_fid, n_data = Dfid.shape
    X = np.column_stack([np.ones(len(par)), par - par.mean(0)])
    coef, *_ = np.linalg.lstsq(X, Dg, rcond=None)
    J = coef[1:].T                                          # (n_data, 6)
    C = np.cov(Dfid, rowvar=False)
    if n_data >= n_fid - 2:
        # fall back to diagonal cov if too many features
        Cinv = np.diag(1.0 / np.diag(C)); hint = "DIAG cov"
    else:
        Cinv = np.linalg.inv(C) * ((n_fid - n_data - 2) / (n_fid - 1)); hint = "full cov+Hartlap"
    F = J.T @ Cinv @ J
    cov = np.linalg.inv(F)
    d = np.sqrt(np.diag(cov))
    corr = cov / np.outer(d, d)
    print(f"\n=== {label}  (n_feat={n_data}, {hint}) ===")
    print("  sigmas:", {PN[i]: round(float(d[i]), 4) for i in range(6)})
    print(f"  corr Om-S8 = {corr[0,1]:+.2f}   Om-w0 = {corr[0,2]:+.2f}   S8-w0 = {corr[1,2]:+.2f}")
    return corr


# ---------------- PS: l100-400, auto+cross, full-sky healpy --------------------
LO, HI, RB = 100, 400, 15   # ell window + rebin -> ~20 bandpowers/spectrum
def ps_vec(kind):  # kind: 'grid' or fiducial perms
    autos = []
    for b in [1, 2, 3, 4]:
        if kind == "grid":
            a = np.load(f"{GRID}/all_cls_grid_nobaryons_bin{b}_noisy_s0.26.npy")
        else:
            a = np.load(f"{FID}/all_cls_fiducial_nobaryons_bin{b}_noisy_s0.26.npy")
        autos.append(a[:, LO:HI])
    if kind == "grid":
        cr = np.load(f"{GRID}/all_cross_cls_grid_nobaryons_bins1234_noisy_s0.26.npy")
    else:
        cr = np.load(f"{FID}/all_cross_cls_fiducial_nobaryons_bins1234_noisy_s0.26.npy")
    cr = cr.reshape(cr.shape[0], 6, 1025)[:, :, LO:HI]
    blocks = [rebin_last(x, RB) for x in autos] + [rebin_last(cr[:, p, :], RB) for p in range(6)]
    return np.concatenate(blocks, axis=1)

corr_ps = fisher_corr(ps_vec("grid"), params, ps_vec("fid"), "PS l100-400 (auto+cross)")

# ---------------- HOS: l1 / peaks, scales234 (0-indexed 1,2,3) ------------------
SC = [1, 2, 3]; SNR_RB = 5    # 40 SNR bins -> 8 ; 4 bins x 3 scales x 8 = 96 features
def hos_vec(prefix, kind):
    bins = []
    for b in [1, 2, 3, 4]:
        if kind == "grid":
            a = np.load(f"{GRIDg}/all_{prefix}_grid_nobaryons_bin{b}_noisy_s0.26_new_normalization.npy")
        else:
            a = np.load(f"{FID}/all_{prefix}_fiducial_nobaryons_bin{b}_noisy_s0.26_new_normalization.npy")
        a = np.asarray(a, float)[:, SC, :]                  # (n, 3, 40)
        bins.append(rebin_last(a, SNR_RB).reshape(a.shape[0], -1))
    return np.concatenate(bins, axis=1)

corr_l1 = fisher_corr(hos_vec("l1_norms", "grid"), params, hos_vec("l1_norms", "fid"), "l1 scales234")
corr_pk = fisher_corr(hos_vec("peak_counts", "grid"), params, hos_vec("peak_counts", "fid"), "peaks scales234")

print("\n================ SUMMARY: Fisher (data) vs NPE posterior signs ================")
print("                       Om-S8        Om-w0        S8-w0")
print(f"  NPE  PS    l100-400   -0.97        -0.38        +0.57")
print(f"  NPE  peaks sc234      -0.83        +0.72        -0.24")
print(f"  NPE  l1    sc234      -0.83        +0.82        -0.40")
print("  ----- Fisher from grid data: -----")
for nm, c in [("PS", corr_ps), ("peaks", corr_pk), ("l1", corr_l1)]:
    print(f"  FISH {nm:5s}            {c[0,1]:+.2f}        {c[0,2]:+.2f}        {c[1,2]:+.2f}")
