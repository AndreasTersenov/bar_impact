"""Fixed-bin l1 at FULL SNR resolution (40 bins), all 5 scales, 4 tomo bins, for a grid subset
(perm_0000 walk) + fiducial perms. Saves (n, 4, 5, 40) so we can decompose the w0 degeneracy by
SNR region / scale / tomo to understand WHERE the PS-vs-HOS direction difference comes from.
Fixed -13/13 bins => 'SNR bin i' is a fixed SNR (fixed kind of structure) across cosmologies.
Run with cosmostat_new venv."""
import os, time
import numpy as np, h5py, healpy as hp
from multiprocessing import Pool
from pycs.astro.wl.hos_peaks_l1 import get_wtl1_sphere

BASE = "/lustre/fsn1/projects/rech/prk/ulx34io/cosmogrid_products/stage3_forecast"
NG = f"{BASE}/new_grid"; FIDd = f"{BASE}/fiducial/cosmo_fiducial"
PARAMS = np.load(f"{BASE}/grid/cosmo_params.npy")
OUT = "/lustre/fsn1/projects/rech/prk/ulx34io/bar_impact/outputs/diagnostics/fixedbin_l1"
os.makedirs(OUT, exist_ok=True)
NGRID, NFID, NWORK = 300, 190, 40
NBINS, NSCALES, NOISE, NOISE_STD = 40, 5, 0.26, 0.0146
NOISELESS = os.environ.get("NOISELESS", "0") == "1"
TAG = "_noiseless" if NOISELESS else ""

def seed_worker():
    np.random.seed(int.from_bytes(os.urandom(4), "little"))

def add_shape_noise(kg, sigma_e=NOISE, gal=6.75, nside=512):
    sig = sigma_e / np.sqrt(gal * hp.nside2pixarea(nside, degrees=True) * 3600.0)
    return kg + np.random.normal(0, sig, len(kg))

def l1full(path):
    out = np.zeros((4, NSCALES, NBINS))
    with h5py.File(path, "r") as f:
        for bi, b in enumerate([1, 2, 3, 4]):
            kg = np.array(f[f"kg/stage3_lensing{b}"], float)
            if not NOISELESS:
                kg = add_shape_noise(kg)
            _, l = get_wtl1_sphere(kg, nscales=NSCALES, nbins=NBINS,
                                   min_snr=-13, max_snr=13, noise_std=NOISE_STD)
            out[bi] = np.asarray(l)
    return out

def grid_walk(n):
    cosmos = sorted(d for d in os.listdir(NG) if d.startswith("cosmo_"))
    perms = [f"perm_{i:04d}" for i in range(7)]; row = 0; picked = []
    for c in cosmos:
        for p in perms:
            mp = f"{NG}/{c}/{p}/projected_probes_maps_nobaryons512.h5"
            if os.path.exists(mp):
                if p == "perm_0000": picked.append((mp, row))
                row += 1
        if len(picked) >= n: break
    return picked[:n]

if __name__ == "__main__":
    t0 = time.time()
    picked = grid_walk(NGRID)
    gpaths = [p for p, _ in picked]; gparams = PARAMS[[r for _, r in picked]]
    fpaths = [f"{FIDd}/perm_{i:04d}/projected_probes_maps_nobaryons512.h5" for i in range(NFID)]
    fpaths = [p for p in fpaths if os.path.exists(p)]
    print(f"grid={len(gpaths)} fid={len(fpaths)} workers={NWORK}", flush=True)
    with Pool(NWORK, initializer=seed_worker) as pool:
        G = np.array(pool.map(l1full, gpaths)); print(f"grid done {time.time()-t0:.0f}s", flush=True)
        F = np.array(pool.map(l1full, fpaths)); print(f"fid done {time.time()-t0:.0f}s", flush=True)
    np.savez(f"{OUT}/fixedbin_l1_full{TAG}.npz", G=G, F=F, gparams=gparams,
             snr=np.linspace(-13, 13, NBINS))
    print(f"saved fixedbin_l1_full{TAG} G{G.shape} F{F.shape} [done {time.time()-t0:.0f}s]")
