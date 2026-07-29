#!/usr/bin/env python3
"""Fisher cross-check: does BNT (cut bin-1 only) beat non-BNT (cut all bins) at the INFORMATION
level? Fisher is basis-invariant for the full vector, so it cleanly separates a genuine BNT
de-biasing/constraining advantage from NPE representation artifacts (the off-truth null, wider
contours we saw in the pilot).

Built-in oracle: non-BNT-full and BNT-full must give IDENTICAL Fisher σ (BNT is information-
preserving). If they match, the cut comparison (non-BNT ℓ≤460 vs BNT bin-1 ℓ≤580, bins 2-4 full)
is trustworthy.

Jacobian = lstsq(C vs params) over the 16965-cosmo grid; covariance from the 200 fiducial perms
(nobaryons), Hartlap-corrected. rebin chosen so n_features < n_perm (well-conditioned inverse).
14000 deg², monopole-subtracted, masked nlb=4/lmax1535. numpy only.
"""
import os
import numpy as np

DD = "/lustre/fsn1/projects/rech/prk/ulx34io/cosmogrid_products/stage3_forecast"
A = int(os.environ.get("FISHER_AREA", "14000"))   # footprint selector (env-threaded for the rollout)
NPERM = 200
REBIN = int(os.environ.get("FISHER_REBIN", "20"))  # ell-rebin; 20 keeps n_feat<200 for the SAMPLE/hybrid
                                                   # cov. Finer (e.g. 10) needs the ANALYTIC cov (any dim).
ELL_OFFSET, ELL_PER_BIN = 2, 4   # masked, lmax>1500
LOWER = 37
PAIRS = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
NAMES = ["Om", "S8", "w0", "H0", "ns", "Ob"]
SUB = [0, 1, 2]
TAIL = f"masked_{A}sqdeg_apod2.0_master_submean_noisy_s0.26_lmax1535"


def gname(kind, sim, b=None, cross=False, bnt=False):
    pre = ("all_bnt_cross_cls" if bnt else "all_cross_cls") if cross else \
          ("all_bnt_cls" if bnt else "all_cls")
    sub = "bins1234" if cross else f"bin{b}"
    return f"{DD}/{kind}/{pre}_{('grid' if kind=='new_grid' else 'fiducial')}_{sim}_{sub}_{TAIL}.npy"


def load_set(kind, sim, bnt):
    autos = [np.load(gname(kind, sim, b=b, bnt=bnt), allow_pickle=True) for b in (1, 2, 3, 4)]
    cross = np.load(gname(kind, sim, cross=True, bnt=bnt), allow_pickle=True)
    nell = autos[0].shape[1]
    return autos, cross, nell


def cut_rebin(arr, upper, nell):
    lo = max(0, int((LOWER - ELL_OFFSET) / ELL_PER_BIN))
    hi = min(arr.shape[1], int((upper - ELL_OFFSET) / ELL_PER_BIN))
    a = arr[:, lo:hi]
    n = a.shape[1] // REBIN
    if n == 0:
        return a[:, :0]
    return a[:, :n * REBIN].reshape(a.shape[0], n, REBIN).mean(axis=2)


def datavector(autos, cross, nell, bin_cuts):
    parts = [cut_rebin(autos[b], bin_cuts[b], nell) for b in range(4)]
    for k, (i, j) in enumerate(PAIRS):
        blk = cross[:, k * nell:(k + 1) * nell]
        parts.append(cut_rebin(blk, min(bin_cuts[i - 1], bin_cuts[j - 1]), nell))
    return np.concatenate(parts, axis=1)


def fisher_sigma(bin_cuts, bnt, params):
    ga, gc, nell = load_set("new_grid", "nobaryons", bnt)
    fa, fc, _ = load_set("fiducial", "nobaryons", bnt)
    dv_grid = datavector(ga, gc, nell, bin_cuts)
    dv_fid = datavector(fa, fc, nell, bin_cuts)
    # drop non-finite grid rows
    ok = np.isfinite(dv_grid).all(1) & np.isfinite(params).all(1)
    dvg, th = dv_grid[ok], params[ok]
    nfeat = dvg.shape[1]
    # Jacobian: dDV/dtheta via lstsq on centered grid
    thc = th - th.mean(0)
    dvc = dvg - dvg.mean(0)
    J, *_ = np.linalg.lstsq(thc, dvc, rcond=None)        # (6, nfeat) = dDV/dtheta
    # covariance from fiducial perms (one-realization), Hartlap-corrected inverse
    cov = np.cov(dv_fid, rowvar=False)
    cinv = np.linalg.inv(cov)
    h = (NPERM - nfeat - 2) / (NPERM - 1)
    F = J @ (h * cinv) @ J.T
    sig = np.sqrt(np.diag(np.linalg.inv(F)))
    return sig, nfeat


def main():
    params = np.load(f"{DD}/grid/cosmo_params.npy", allow_pickle=True)
    if params.ndim == 1:  # structured/object -> stack
        params = np.vstack([np.asarray(params[n]) for n in NAMES]).T if params.dtype.names else \
                 np.array([list(r) for r in params])
    params = np.asarray(params, float)
    print(f"params {params.shape}; rebin={REBIN}, Nperm={NPERM}, 14000 deg² submean\n")
    configs = [
        ("non-BNT  full          ", [1024, 1024, 1024, 1024], False),
        ("BNT      full (oracle)  ", [1024, 1024, 1024, 1024], True),
        ("non-BNT  cut-all ℓ460   ", [460, 460, 460, 460], False),
        ("BNT      bin1-ℓ580 2-4f ", [580, 1024, 1024, 1024], True),
    ]
    print(f"{'config':26s} {'nfeat':>5} {'σ(Ωm)':>8} {'σ(S8)':>8} {'σ(w0)':>8}  area(Ωm-S8)")
    res = {}
    for label, cuts, bnt in configs:
        sig, nf = fisher_sigma(cuts, bnt, params)
        res[label.strip()] = sig
        print(f"{label:26s} {nf:>5} {sig[0]:>8.4f} {sig[1]:>8.4f} {sig[2]:>8.4f}  {sig[0]*sig[1]:.3e}")
    # oracle + science readouts
    o1, o2 = res["non-BNT  full"], res["BNT      full (oracle)"]
    print(f"\n[oracle] |BNT-full − nonBNT-full| / nonBNT-full  (should be ~0):")
    print("   " + "  ".join(f"{n}:{abs(o2[i]-o1[i])/o1[i]:.1e}" for i, n in enumerate(NAMES)))
    c, b = res["non-BNT  cut-all ℓ460"], res["BNT      bin1-ℓ580 2-4f"]
    print(f"\n[science] BNT-bin1-580 vs non-BNT-460 (ratio <1 => BNT tighter):")
    print("   " + "  ".join(f"{NAMES[i]}:{b[i]/c[i]:.3f}" for i in SUB) +
          f"   area ratio {b[0]*b[1]/(c[0]*c[1]):.3f}")


if __name__ == "__main__":
    main()
