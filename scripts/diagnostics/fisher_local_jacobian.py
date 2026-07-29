#!/usr/bin/env python3
"""Phase II of the proper Fisher (docs/PLAN_fisher_proper.md): the LOCAL derivative at the fiducial,
with convergence tests. Replaces the global lstsq-over-prior Jacobian (the documented Euclid-IST:F
instability) that drove the 0.37<->0.5 wobble.

Phase I settled the covariance: keep the SIM covariance (it carries the real SSC+cNG the survey has),
validated by the analytic Gaussian on the diagonal. So the only remaining lever is the Jacobian.

METHOD: J = d(signal)/dtheta estimated LOCALLY at the CosmoGrid fiducial
theta_fid = [Om,S8,w0,H0,ns,Ob] = [0.26,0.84,-1.0,67.36,0.9649,0.0493], by kernel-weighted polynomial
regression of the noise-averaged grid data vectors in WHITENED parameter space (u = (theta-fid)/std):
  - order1_anchored : slope through (fid, fid_mean), no intercept (repo-consistent, low variance).
  - order1_free     : local linear regression with intercept.
  - order2          : local quadratic (intercept + linear + squares + cross); gradient = linear
                      coefficients at u=0 -> removes the leading CURVATURE bias of a finite neighborhood.
Realizations (~7/cosmo) are averaged FIRST (sqrt(7) noise cut). Gaussian kernel, bandwidth h in
whitened param-std units; N_eff = (sum w)^2/sum w^2 reported per h.

CONVERGENCE (the back-pressure): the gradient -> Fisher sigma must PLATEAU as h shrinks and across
order. order2 should be flat in h (curvature removed); order1 should converge to order2 as h->0.
If the BNT-580/non-BNT-460 area ratio is stable across (order, h), it is trustworthy.

COVARIANCE: the 200-perm sim covariance (Phase I), Hartlap-corrected, on whitened features
(Fisher-invariant). Oracle: BNT-full == non-BNT-full (BNT is information-preserving). numpy only.
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from fisher_bnt_vs_nonbnt import load_set, datavector, DD, NPERM, NAMES  # noqa: E402

FID = np.array([0.26, 0.84, -1.0, 67.36, 0.9649, 0.0493])
PARAMS = np.load(f"{DD}/grid/cosmo_params.npy", allow_pickle=True).astype(float)
PS = PARAMS.std(0)                                              # param-std for whitening param coords
SUB = [0, 1, 2]                                                 # Om, S8, w0 for the headline area


def avg_realizations(dv, th):
    """Mean over the ~7 realizations of each unique cosmology -> (ncos, nfeat), (ncos, 6)."""
    keys = np.round(th, 8)
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    out = np.zeros((uniq.shape[0], dv.shape[1]))
    np.add.at(out, inv, dv)
    out /= np.bincount(inv)[:, None]
    return out, uniq


def design(u, order):
    """Polynomial design matrix in whitened param coords u (n,6). Returns (X, idx_linear)."""
    n = u.shape[0]
    if order == "order1_anchored":
        return u, np.arange(6)                                 # no intercept; cols 0..5 are the slope
    cols = [np.ones(n), u]                                     # intercept + linear
    if order == "order2":
        sq = u ** 2
        cross = [u[:, a] * u[:, b] for a in range(6) for b in range(a + 1, 6)]
        cols += [sq, np.column_stack(cross)]
    X = np.column_stack(cols)
    return X, np.arange(1, 7)                                  # linear coeffs are cols 1..6


def local_jacobian(grid_avg, ucos, fid_mean, order, h):
    """Kernel-weighted polynomial gradient at u=0. Returns J = dDV/dtheta (nfeat, 6)."""
    d2 = (ucos ** 2).sum(1)
    w = np.exp(-0.5 * d2 / h ** 2)
    sw = np.sqrt(w)[:, None]
    X, lin = design(ucos, order)
    Y = grid_avg - fid_mean if order == "order1_anchored" else grid_avg
    beta, *_ = np.linalg.lstsq(X * sw, Y * sw, rcond=None)     # (ncoef, nfeat)
    slope_u = beta[lin]                                        # dDV/du  (6, nfeat)
    J = (slope_u / PS[:, None]).T                              # dDV/dtheta (nfeat, 6)
    neff = (w.sum() ** 2) / (w ** 2).sum()
    return J, neff


def fisher_sigma(J, Cw, s_keep, nperm):
    """F = J^T (hartlap Cinv) J on whitened features; J already in dtheta units (nfeat,6)."""
    nfeat = J.shape[0]
    h = (nperm - nfeat - 2) / (nperm - 1)
    if h <= 0:
        return None, nfeat, h
    Jw = J / s_keep[:, None]                                   # whiten features (Fisher-invariant)
    F = Jw.T @ (h * np.linalg.inv(Cw)) @ Jw
    sig = np.sqrt(np.diag(np.linalg.inv(F)))
    return sig, nfeat, h


def build_config(bin_cuts, bnt):
    ga, gc, nell = load_set("new_grid", "nobaryons", bnt)
    fa, fc, _ = load_set("fiducial", "nobaryons", bnt)
    dv_g = datavector(ga, gc, nell, bin_cuts)
    dv_f = datavector(fa, fc, nell, bin_cuts)
    ok = np.isfinite(dv_g).all(1) & np.isfinite(PARAMS).all(1)
    dv_g, th = dv_g[ok], PARAMS[ok]
    grid_avg, thcos = avg_realizations(dv_g, th)
    ucos = (thcos - FID) / PS
    s = dv_f.std(0)
    keep = s > 0
    grid_avg, dv_f, s = grid_avg[:, keep], dv_f[:, keep], s[keep]
    fid_mean = dv_f.mean(0)
    Cw = np.cov(dv_f / s, rowvar=False)
    return dict(grid_avg=grid_avg, ucos=ucos, fid_mean=fid_mean, s=s, Cw=Cw)


def sig_for(cfg, order, h):
    J, neff = local_jacobian(cfg["grid_avg"], cfg["ucos"], cfg["fid_mean"], order, h)
    sig, nfeat, hart = fisher_sigma(J, cfg["Cw"], cfg["s"], NPERM)
    return sig, nfeat, hart, neff


def area(sig):
    return sig[0] * sig[1] if sig is not None else np.nan


def main():
    print("=== Phase II: LOCAL Jacobian at the fiducial, masked nlb=4/lmax1535 submean, 14000 ===")
    print(f"theta_fid = {dict(zip(NAMES, FID))}\n")
    configs = {
        "nonbnt_full":  ([1024, 1024, 1024, 1024], False),
        "bnt_full":     ([1024, 1024, 1024, 1024], True),
        "nonbnt_460":   ([460, 460, 460, 460], False),
        "bnt_580":      ([580, 1024, 1024, 1024], True),
    }
    C = {k: build_config(cuts, bnt) for k, (cuts, bnt) in configs.items()}

    orders = ["order1_anchored", "order1_free", "order2"]
    bws = [0.5, 0.75, 1.0, 1.5, 2.0]

    # N_eff per bandwidth (same grid for all configs ~ use nonbnt_460)
    print("N_eff(grid cosmologies feeding the local derivative) per bandwidth h:")
    for h in bws:
        _, neff = local_jacobian(C["nonbnt_460"]["grid_avg"], C["nonbnt_460"]["ucos"],
                                 C["nonbnt_460"]["fid_mean"], "order1_free", h)
        print(f"  h={h:<4}  N_eff={neff:6.0f} / {C['nonbnt_460']['ucos'].shape[0]} cosmologies")

    # ---- oracle: BNT-full == non-BNT-full under the local Jacobian ----
    print("\n=== ORACLE  BNT-full vs non-BNT-full (must match; BNT is information-preserving) ===")
    for order in orders:
        sa, *_ = sig_for(C["nonbnt_full"], order, 1.0)
        sb, *_ = sig_for(C["bnt_full"], order, 1.0)
        rel = np.max(np.abs(sb - sa) / np.abs(sa)) if sa is not None else np.nan
        print(f"  {order:16s}  max|rel diff| over 6 params = {rel:.2e}")

    # ---- CONVERGENCE: sigma(S8) and BNT/non-BNT area ratio vs (order, bandwidth) ----
    print("\n=== CONVERGENCE: BNT-580 / non-BNT-460  (NPE=0.79; global-lstsq=0.37; noiseavg-global~0.5) ===")
    hdr = "order \\ h        " + "".join(f"{h:>10}" for h in bws)
    print(hdr); print("-" * len(hdr))
    for order in orders:
        cells = []
        for h in bws:
            sb, nfb, *_ = sig_for(C["bnt_580"], order, h)
            sc, nfc, *_ = sig_for(C["nonbnt_460"], order, h)
            r = area(sb) / area(sc)
            cells.append(f"{r:>10.3f}")
        print(f"{order:16s}" + "".join(cells))
    print("\n(area ratio = (sigOm*sigS8)_BNT580 / (sigOm*sigS8)_nonBNT460. Stable across order & h => trustworthy.)")

    # ---- detail at a converged setting (order2, h=0.75): sigma's + nfeat/Hartlap ----
    print("\n=== DETAIL  order2, h=0.75 ===")
    print(f"{'config':14s} {'nfeat':>5} {'hart':>6} {'sigOm':>8} {'sigS8':>8} {'sigw0':>8} {'area(Om,S8)':>12}")
    for k in configs:
        sig, nfeat, hart, _ = sig_for(C[k], "order2", 0.75)
        if sig is None:
            print(f"{k:14s}  (Hartlap invalid: nfeat>=nperm)"); continue
        print(f"{k:14s} {nfeat:>5} {hart:>6.3f} {sig[0]:>8.4f} {sig[1]:>8.4f} {sig[2]:>8.4f} {sig[0]*sig[1]:>12.3e}")


if __name__ == "__main__":
    main()
