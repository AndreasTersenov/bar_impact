#!/usr/bin/env python3
"""Phase I of the proper Fisher (docs/PLAN_fisher_proper.md): the analytic, mask-aware
Gaussian covariance of the auto+cross power-spectrum bandpowers, cross-checked against the
200-perm sample covariance.

WHY: the ad-hoc Fisher's covariance was a 200-perm sample matrix with a large, config-
differential Hartlap factor (the documented Euclid-IST:F instability). The analytic NaMaster
Gaussian covariance is perfectly conditioned, has no perm noise, and is self-consistent with a
Gaussian-likelihood Fisher. This script builds it and validates it against the sims.

WHAT it computes, for the 14000 deg^2 apodized mask (nside=512, lmax=1535, nlb=4):
  - one NaMaster field (the shared mask), one workspace w (the production MCM), one covariance
    workspace cw; both cached to disk (cw is the expensive step).
  - the full Gaussian covariance of the 10 tomographic spectra [(1,1),(2,2),(3,3),(4,4),
    (1,2),(1,3),(1,4),(2,3),(2,4),(3,4)] x 383 native bandpowers = 3830 x 3830.
  - INPUT theory Cls = the 200-perm fiducial MEAN of the *measured* decoupled bandpowers
    (autos already include shape noise; cross are noise-free) -> fully sim-based, apples-to-apples
    with the NPE. Each is unbinned to per-ell via the bandpower windows.

VALIDATION (back-pressure):
  (V1) native per-bandpower variance: analytic diag vs sample var over 200 perms. The diagonal is
       well estimated from 200 samples, so this is the robust, high-resolution oracle. Expect
       agreement to ~sqrt(2/200) ~ 10% scatter in the Gaussian regime; watch low-ell (few modes,
       mild non-Gaussianity) vs high-ell (noise-dominated, Gaussian-exact).
  (V2) the Fisher's rebinned+cut covariance (full config, ell>=37, rebin=20): apply the SAME
       linear cut_rebin operator R to the analytic native covariance (R C R^T) and compare the
       full 120x120 matrix to np.cov of the rebinned data vector -> tests the off-diagonal
       structure where the sample matrix is conditioned.

Run with the cosmostat_new venv python (pymaster):
  /home/tersenov/software/cosmostat_new/cosmostat/cosmostat_new/bin/python \
      scripts/diagnostics/fisher_gaussian_cov.py
"""
import os
import sys
import time
import numpy as np
import healpy as hp
import pymaster as nmt

# Reuse the exact data layout / cut_rebin from the ad-hoc Fisher so the comparison is like-for-like.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from fisher_bnt_vs_nonbnt import (  # noqa: E402
    load_set, datavector, DD, NPERM, REBIN, LOWER, ELL_OFFSET, ELL_PER_BIN, PAIRS,
)

# ---- geometry / binning (must match cross_power_spectrum_processing_master.py) ----
NSIDE = 512
LMAX = 1535                 # = 3*nside - 1 (Nyquist), as used in production
NLB = 4
AREA = float(os.environ.get("FISHER_AREA", "14000"))   # footprint selector (env-threaded for the rollout)
APOD_DEG = 2.0
CENTER = (0.0, 90.0)
APOD_TYPE = "C2"

# the 10 tomographic spectra in DATA-VECTOR order (4 autos, then the 6 cross pairs)
AUTOS = [(1, 1), (2, 2), (3, 3), (4, 4)]
SPECTRA = AUTOS + PAIRS                       # 10 spectra
NBINS_TOMO = 4

CACHE = os.path.join(HERE, "cache_gaussian_cov")
os.makedirs(CACHE, exist_ok=True)
COV_OUT = os.path.join(CACHE, f"gaussian_cov_native_{int(AREA)}.npy")


# --------------------------------------------------------------------------------------------
# mask (replicated EXACTLY from cross_power_spectrum_processing_master.create_apodized_mask)
# --------------------------------------------------------------------------------------------
def create_apodized_mask(nside, target_area_sqdeg, center_coords, apod_type, apod_scale_deg):
    total_area_sqdeg = 41252.96125
    angular_radius_rad = np.arccos(1 - (target_area_sqdeg / total_area_sqdeg) * 2)
    angular_radius_deg = np.rad2deg(angular_radius_rad)
    theta_center = np.deg2rad(90.0 - center_coords[1])
    phi_center = np.deg2rad(center_coords[0])
    center_vec = hp.ang2vec(theta_center, phi_center)
    npix = hp.nside2npix(nside)
    vx, vy, vz = hp.pix2vec(nside, np.arange(npix))
    dots = np.clip(center_vec[0] * vx + center_vec[1] * vy + center_vec[2] * vz, -1.0, 1.0)
    ang_sep_deg = np.rad2deg(np.arccos(dots))
    mask = np.zeros(npix, dtype=np.float32)
    inner_radius = angular_radius_deg - apod_scale_deg
    if inner_radius > 0:
        mask[ang_sep_deg <= inner_radius] = 1.0
    outer_radius = angular_radius_deg + apod_scale_deg
    in_transition = (ang_sep_deg > max(0, inner_radius)) & (ang_sep_deg <= outer_radius)
    if np.any(in_transition):
        width = outer_radius - max(0, inner_radius)
        x = (ang_sep_deg[in_transition] - max(0, inner_radius)) / width
        if apod_type == "C2":
            taper = np.where(x < 0.5, 1.0 - 2 * x**2, 2 * (1 - x)**2)
        elif apod_type == "C1":
            taper = 0.5 * (1.0 + np.cos(np.pi * x))
        else:
            raise ValueError(apod_type)
        mask[in_transition] = taper.astype(np.float32)
    return mask.astype(np.float64), float(np.mean(mask))


# --------------------------------------------------------------------------------------------
# workspaces (cached: cw is the expensive coupling-coefficient computation)
# --------------------------------------------------------------------------------------------
def build_workspaces(mask):
    b = nmt.NmtBin.from_lmax_linear(LMAX, nlb=NLB)
    try:
        f = nmt.NmtField(mask, None, spin=0, lmax=LMAX)
    except Exception:
        f = nmt.NmtField(mask, [mask], purify_b=False, lmax=LMAX)

    w = nmt.NmtWorkspace()
    w_path = os.path.join(CACHE, f"w_{int(AREA)}.fits")
    if os.path.exists(w_path):
        w.read_from(w_path)
        print(f"[w] loaded {w_path}")
    else:
        t0 = time.time()
        w.compute_coupling_matrix(f, f, b)
        w.write_to(w_path)
        print(f"[w] computed MCM in {time.time()-t0:.1f}s -> {w_path}")

    cw = nmt.NmtCovarianceWorkspace()
    cw_path = os.path.join(CACHE, f"cw_{int(AREA)}.fits")
    if os.path.exists(cw_path):
        cw.read_from(cw_path)
        print(f"[cw] loaded {cw_path}")
    else:
        t0 = time.time()
        cw.compute_coupling_coefficients(f, f, f, f)
        cw.write_to(cw_path)
        print(f"[cw] computed covariance coupling in {time.time()-t0:.1f}s -> {cw_path}")
    return b, w, cw


# --------------------------------------------------------------------------------------------
# input theory Cls = 200-perm fiducial MEAN of the measured decoupled bandpowers, per spectrum,
# unbinned to per-ell. Autos already carry shape noise; cross are ~noise-free.
# --------------------------------------------------------------------------------------------
def load_fiducial_spectra():
    """Return (per-perm data dict, mean per-ell theory dict). Keys are sorted tomo pairs (i,j)."""
    fa, fc, nell = load_set("fiducial", "nobaryons", bnt=False)   # autos: 4x(200,383); cross (200,6*383)
    assert nell == fa[0].shape[1]
    perbin = {}
    for b_idx, (i, _) in enumerate(AUTOS):
        perbin[(i, i)] = fa[b_idx]                                # (200, 383)
    for k, (i, j) in enumerate(PAIRS):
        perbin[(i, j)] = fc[:, k * nell:(k + 1) * nell]           # (200, 383)
    means = {key: v.mean(axis=0) for key, v in perbin.items()}   # (383,)
    return perbin, means, nell


def unbin_to_per_ell(b, cl_binned):
    """Expand binned bandpowers to a per-ell array of length LMAX+1 via the bandpower windows."""
    per_ell = b.unbin_cell(np.atleast_2d(cl_binned))[0]          # length b.lmax()+1
    out = np.zeros(LMAX + 1)
    n = min(len(per_ell), LMAX + 1)
    out[:n] = per_ell[:n]
    if n < LMAX + 1:                                             # fill unbinned tail with last value
        out[n:] = per_ell[-1]
    return out


# --------------------------------------------------------------------------------------------
# assemble the full native Gaussian covariance (3830 x 3830)
# --------------------------------------------------------------------------------------------
def assemble_native_cov(b, w, cw, theory_per_ell, nbpw):
    nspec = len(SPECTRA)
    ndim = nspec * nbpw
    C = np.zeros((ndim, ndim))

    def cl(i, j):
        return theory_per_ell[(min(i, j), max(i, j))]

    t0 = time.time()
    nblk = 0
    for a in range(nspec):
        (i1, i2) = SPECTRA[a]
        for c in range(a, nspec):
            (j1, j2) = SPECTRA[c]
            covar = nmt.gaussian_covariance(
                cw, 0, 0, 0, 0,
                [cl(i1, j1)], [cl(i1, j2)], [cl(i2, j1)], [cl(i2, j2)],
                w, w,
            )
            blk = np.asarray(covar).reshape(nbpw, nbpw)
            C[a * nbpw:(a + 1) * nbpw, c * nbpw:(c + 1) * nbpw] = blk
            if c != a:
                C[c * nbpw:(c + 1) * nbpw, a * nbpw:(a + 1) * nbpw] = blk.T
            nblk += 1
    print(f"[cov] assembled {nblk} blocks ({ndim}x{ndim}) in {time.time()-t0:.1f}s")
    return C


# --------------------------------------------------------------------------------------------
# the Fisher's cut_rebin as an explicit linear operator R (per spectrum block), so we can map
# the analytic native covariance into the rebinned+cut space: C_rebin = R C R^T.
# --------------------------------------------------------------------------------------------
def cut_rebin_operator(nbpw, upper):
    lo = max(0, int((LOWER - ELL_OFFSET) / ELL_PER_BIN))
    hi = min(nbpw, int((upper - ELL_OFFSET) / ELL_PER_BIN))
    n = (hi - lo) // REBIN
    R = np.zeros((n, nbpw))
    for k in range(n):
        s = lo + k * REBIN
        R[k, s:s + REBIN] = 1.0 / REBIN
    return R                                                      # (n_rebinned, nbpw)


def build_full_R(nbpw, upper):
    """Block-diagonal R over the 10 spectra (all share the same upper cut in the full config)."""
    Rb = cut_rebin_operator(nbpw, upper)
    n = Rb.shape[0]
    R = np.zeros((len(SPECTRA) * n, len(SPECTRA) * nbpw))
    for a in range(len(SPECTRA)):
        R[a * n:(a + 1) * n, a * nbpw:(a + 1) * nbpw] = Rb
    return R


# --------------------------------------------------------------------------------------------
def main():
    print(f"=== Phase I: analytic NaMaster Gaussian covariance @ {int(AREA)} deg^2 ===")
    mask, fsky = create_apodized_mask(NSIDE, AREA, CENTER, APOD_TYPE, APOD_DEG)
    print(f"mask: f_sky={fsky:.4f}  (lmax={LMAX}, nlb={NLB})")

    b, w, cw = build_workspaces(mask)
    eff_ell = b.get_effective_ells()
    nbpw = len(eff_ell)
    print(f"bandpowers: {nbpw} (eff_ell {eff_ell[0]:.1f}..{eff_ell[-1]:.1f})")

    perbin, means, nell = load_fiducial_spectra()
    assert nell == nbpw, f"data nbpw {nell} != workspace nbpw {nbpw}"
    theory = {key: unbin_to_per_ell(b, means[key]) for key in means}

    if os.path.exists(COV_OUT):
        C = np.load(COV_OUT)
        print(f"[cov] loaded {COV_OUT}  shape {C.shape}")
    else:
        C = assemble_native_cov(b, w, cw, theory, nbpw)
        np.save(COV_OUT, C)
        print(f"[cov] saved {COV_OUT}")

    # ---- V1: native per-bandpower variance, analytic vs sample (robust diagonal) ----
    print("\n=== V1: native per-bandpower variance  analytic / sample  (200 perms) ===")
    print(f"{'spectrum':>10} {'med ratio':>10} {'ratio ell<100':>14} {'ratio ell>400':>14}")
    dlow = (eff_ell < 100)
    dhigh = (eff_ell > 400)
    for a, (i, j) in enumerate(SPECTRA):
        sl = slice(a * nbpw, (a + 1) * nbpw)
        ana = np.diag(C)[sl]
        samp = perbin[(min(i, j), max(i, j))].var(axis=0, ddof=1)
        ratio = ana / (samp + 1e-300)
        tag = f"({i},{j})"
        print(f"{tag:>10} {np.median(ratio):>10.3f} "
              f"{np.median(ratio[dlow]):>14.3f} {np.median(ratio[dhigh]):>14.3f}")

    # ---- V2: rebinned+cut full covariance (full config, ell>=37, rebin=20) ----
    print("\n=== V2: rebinned+cut covariance (full config 1024, ell>=37, rebin=20) ===")
    R = build_full_R(nbpw, upper=1024)
    C_rb = R @ C @ R.T
    fa, fc, _ = load_set("fiducial", "nobaryons", bnt=False)
    dv = datavector(fa, fc, nbpw, [1024, 1024, 1024, 1024])      # (200, 120)
    C_samp = np.cov(dv, rowvar=False)
    da, ds = np.diag(C_rb), np.diag(C_samp)
    print(f"  features: analytic {C_rb.shape[0]}  sample {C_samp.shape[0]}")
    print(f"  diag ratio analytic/sample: median {np.median(da/ds):.3f}  "
          f"IQR [{np.percentile(da/ds,25):.3f}, {np.percentile(da/ds,75):.3f}]")
    # correlation-matrix agreement (off-diagonal structure)
    Da, Ds = np.sqrt(np.outer(da, da)), np.sqrt(np.outer(ds, ds))
    corr_a, corr_s = C_rb / Da, C_samp / Ds
    iu = np.triu_indices_from(corr_a, k=1)
    print(f"  off-diag correlation: mean|analytic| {np.mean(np.abs(corr_a[iu])):.3f}  "
          f"mean|sample| {np.mean(np.abs(corr_s[iu])):.3f}  "
          f"RMS(analytic-sample) {np.sqrt(np.mean((corr_a[iu]-corr_s[iu])**2)):.3f}")
    np.savez(os.path.join(CACHE, f"cov_rebinned_full_{int(AREA)}.npz"),
             C_analytic=C_rb, C_sample=C_samp, eff_ell=eff_ell)
    print(f"\n[done] native cov: {COV_OUT}")
    print("Read: V1 ratios ~1 (esp. ell>400) => analytic Gaussian matches the sims; "
          "low-ell departures = few-mode/non-Gaussian, expected.")


if __name__ == "__main__":
    main()
