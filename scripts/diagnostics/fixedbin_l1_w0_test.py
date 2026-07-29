"""Verification #1: does the HOS w0 degeneracy flip survive FIXED (cosmology-independent) binning?

The on-disk l1 data uses auto per-scale binning (min_snr/max_snr = None), so the SNR bin edges are
cosmology-dependent -- a possible source of a spurious degeneracy. Here we recompute l1 for a grid
subset (perm_0000 walk) + fiducial perms with FIXED min_snr=-13, max_snr=13 bins, scales234, and
compute the Fisher Om-w0. If it still flips positive (like the auto-binned data), the binning is not
the cause -> the flip is real. Run with the cosmostat_new venv (pycs).
"""
import os, sys, time
import numpy as np
import h5py, healpy as hp
from multiprocessing import Pool
from pycs.astro.wl.hos_peaks_l1 import get_wtl1_sphere

BASE = "/lustre/fsn1/projects/rech/prk/ulx34io/cosmogrid_products/stage3_forecast"
NG = f"{BASE}/new_grid"; FIDd = f"{BASE}/fiducial/cosmo_fiducial"
PARAMS = np.load(f"{BASE}/grid/cosmo_params.npy")
OUT = "/lustre/fsn1/projects/rech/prk/ulx34io/bar_impact/outputs/diagnostics/fixedbin_l1"
os.makedirs(OUT, exist_ok=True)
NGRID, NFID, NWORK = 260, 190, 40
SCALES = [1, 2, 3]; NBINS = 40; NSCALES = 5; NOISE = 0.26; NOISE_STD = 0.0146
SNR_RB = 8  # 40 -> 5 SNR bins ; 3 scales x 5 x 4 tomo = 60 features < NFID-2

def seed_worker():
    np.random.seed(int.from_bytes(os.urandom(4), "little"))

def add_shape_noise(kg, sigma_e=NOISE, gal=6.75, nside=512):
    pix_arcmin2 = hp.nside2pixarea(nside, degrees=True) * 3600.0
    sig = sigma_e / np.sqrt(gal * pix_arcmin2)
    return kg + np.random.normal(0, sig, len(kg))

def l1vec(path):
    out = []
    with h5py.File(path, "r") as f:
        for b in [1, 2, 3, 4]:
            kg = np.array(f[f"kg/stage3_lensing{b}"], float)
            kg = add_shape_noise(kg)
            _, l = get_wtl1_sphere(kg, nscales=NSCALES, nbins=NBINS,
                                   min_snr=-13, max_snr=13, noise_std=NOISE_STD)  # FIXED bins
            a = np.asarray(l)[SCALES]                      # (3,40)
            n = a.shape[1] // SNR_RB
            out.append(a[:, :n * SNR_RB].reshape(3, n, SNR_RB).mean(2).ravel())
    return np.concatenate(out)

def grid_walk(n):
    cosmos = sorted(d for d in os.listdir(NG) if d.startswith("cosmo_"))
    perms = [f"perm_{i:04d}" for i in range(7)]
    row = 0; picked = []
    for c in cosmos:
        for p in perms:
            mp = f"{NG}/{c}/{p}/projected_probes_maps_nobaryons512.h5"
            if os.path.exists(mp):
                if p == "perm_0000":
                    picked.append((mp, row))
                row += 1
        if len(picked) >= n:
            break
    return picked[:n]

if __name__ == "__main__":
    t0 = time.time()
    picked = grid_walk(NGRID)
    gpaths = [p for p, _ in picked]; gparams = PARAMS[[r for _, r in picked]]
    fpaths = [f"{FIDd}/perm_{i:04d}/projected_probes_maps_nobaryons512.h5" for i in range(NFID)]
    fpaths = [p for p in fpaths if os.path.exists(p)]
    print(f"grid={len(gpaths)} fid={len(fpaths)}  workers={NWORK}", flush=True)
    with Pool(NWORK, initializer=seed_worker) as pool:
        G = np.array(pool.map(l1vec, gpaths)); print(f"grid done {time.time()-t0:.0f}s", flush=True)
        F = np.array(pool.map(l1vec, fpaths)); print(f"fid done {time.time()-t0:.0f}s", flush=True)
    np.savez(f"{OUT}/fixedbin_l1.npz", G=G, F=F, gparams=gparams)

    # Fisher Om-w0 (fixed bins)
    v = F.var(0); keep = v > v.max() * 1e-8; Gk, Fk = G[:, keep], F[:, keep]
    nf, nd = Fk.shape
    X = np.column_stack([np.ones(len(gparams)), gparams - gparams.mean(0)])
    coef, *_ = np.linalg.lstsq(X, Gk, rcond=None); J = coef[1:].T
    C = np.cov(Fk, rowvar=False); Cinv = np.linalg.inv(C) * ((nf - nd - 2) / (nf - 1))
    cov = np.linalg.inv(J.T @ Cinv @ J); d = np.sqrt(np.diag(cov)); corr = cov / np.outer(d, d)
    print(f"\n=== FIXED-BIN l1 scales234 (n_feat={nd}, n_fid={nf}) ===")
    print(f"  Om-w0 = {corr[0,2]:+.2f}   S8-w0 = {corr[1,2]:+.2f}   Om-S8 = {corr[0,1]:+.2f}")
    print(f"  compare: AUTO-binned (on-disk) Fisher  Om-w0=+0.60  S8-w0=-0.27  Om-S8=-0.90")
    print(f"  PS l100-400 Fisher                     Om-w0=-0.80")
    print(f"[done {time.time()-t0:.0f}s]")
