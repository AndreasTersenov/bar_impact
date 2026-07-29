"""
Fisher gate for the monopole-leakage fix.

Question: does removing the mask-weighted monopole make the masked 30<=ell<100 band
trustworthy -- i.e. does it loosen the artificially-tight masked low-ell contours to be
>= the config-matched full-sky reference, while leaving the 100+ band unchanged?

This is the cheap, faithful stand-in for re-running the full NPE: the leakage is a
Gaussian-information effect (a near-noiseless, cosmology-dependent monopole), so a Fisher
forecast captures the contour tightening and its removal. We recompute everything with the
CURRENT code (the on-disk paper Cls were made by an older library version and do not
reproduce bit-for-bit), so raw vs submean is a clean, controlled comparison.

Alignment: the grid datavector / cosmo_params are built by the deterministic walk
sorted(cosmo_dirs) x perm_0000..6 over existing maps. We replicate it, take the perm_0000
rows of the first N cosmologies, and read their params from cosmo_params.npy at the matching
row indices -- no misalignment.

Run with the cosmostat venv python (has pymaster).
"""
import os
import sys
import json
import argparse
import importlib.util

import numpy as np
import h5py
import healpy as hp

SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(SCRIPTS_DIR)

spec = importlib.util.spec_from_file_location(
    "psm", os.path.join(SCRIPTS_DIR, "cross_power_spectrum_processing_master.py"))
PSM = importlib.util.module_from_spec(spec)
spec.loader.exec_module(PSM)
assert PSM.HAS_NAMASTER, "needs NaMaster (run with the cosmostat venv python)"

COSMOGRID = "/lustre/fsn1/projects/rech/prk/ulx34io/cosmogrid_products/stage3_forecast"
GRID_DIR = os.path.join(COSMOGRID, "new_grid")
FID_DIR = os.path.join(COSMOGRID, "fiducial", "cosmo_fiducial")
MAP_NAME = "projected_probes_maps_nobaryons512.h5"
PARAMS_NPY = os.path.join(REPO, "cosmoGRID_datavectors", "cosmo_params.npy")
PARAM_NAMES = ["Omega_m", "S8", "w0", "H0", "ns", "Omega_b"]

BINS = [1, 2, 3, 4]
CROSS_PAIRS = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
SPEC_KEYS = [(b, b) for b in BINS] + CROSS_PAIRS  # auto then cross (inference ordering)
NPERM_GRID = 7   # perm_0000..perm_0006 (matches main()'s walk)
NPERM_FID = 200

CFG = dict(nside=512, mask_area=14000.0, center=(0.0, 90.0), apod_type="C2",
           apod_deg=2.0, lmax=1535, noise=0.26, gal_density=6.75)

# Globals set in run() before the Pool forks; workers inherit them (Linux fork + COW),
# including the parent-built MCM cache, so no per-worker coupling-matrix rebuild.
_MASK = None
_UNIT = None


# --------------------------------------------------------------------------- #
def grid_walk_perm0(n_cosmo):
    """Replicate main()'s grid walk and return the first n_cosmo (cosmo, perm_0000, path,
    global_row_index) tuples that have an existing map. global_row_index indexes into
    cosmo_params.npy (perm-expanded, same walk order)."""
    cosmo_dirs = sorted(d for d in os.listdir(GRID_DIR) if d.startswith("cosmo_"))
    perm_dirs = [f"perm_{i:04d}" for i in range(NPERM_GRID)]
    row = 0
    picked = []
    for cosmo in cosmo_dirs:
        for perm in perm_dirs:
            path = os.path.join(GRID_DIR, cosmo, perm, MAP_NAME)
            if not os.path.exists(path):
                continue
            if perm == "perm_0000":
                picked.append((cosmo, perm, path, row))
            row += 1
        if len(picked) >= n_cosmo:
            break
    return picked[:n_cosmo]


def load_noisy_maps(path, seed):
    rng = np.random.default_rng(seed)
    maps = {}
    with h5py.File(path, "r") as f:
        for b in BINS:
            kg = np.array(f[f"kg/stage3_lensing{b}"], dtype=np.float64)
            kg = PSM.add_shape_noise(kg, sigma_e=CFG["noise"],
                                     galaxy_density=CFG["gal_density"],
                                     nside=CFG["nside"], rng=rng)
            maps[b] = kg
    return maps


def datavector(maps, mask, subtract_mean):
    """Concatenated decoupled auto+cross bandpowers for one realization."""
    if subtract_mean:
        wsum = float(np.sum(mask))
        maps = {b: maps[b] - float(np.sum(mask * maps[b]) / wsum) for b in maps}
    cls, ells = PSM.compute_power_spectra_master(maps, mask, lmax=CFG["lmax"],
                                                 use_namaster=True, verbose=False)
    vec = np.concatenate([np.asarray(cls[k]) for k in SPEC_KEYS])
    return vec, np.asarray(ells)


def _worker(path_seed):
    """Compute all three arms (masked-raw, masked-submean, fullsky) for one map,
    loading it once. Returns (raw, sub, fullsky, ells)."""
    path, seed = path_seed
    maps = load_noisy_maps(path, seed)
    raw, ells = datavector(maps, _MASK, False)
    sub, _ = datavector(maps, _MASK, True)
    fs, _ = datavector(maps, _UNIT, False)
    return raw, sub, fs, ells


def build_arms(paths_seeds, workers, label):
    """Parallel map over realizations -> (raw, sub, fullsky) matrices + ells."""
    from multiprocessing import Pool
    print(f"  [{label}] {len(paths_seeds)} maps on {workers} workers ...", flush=True)
    if workers <= 1:
        out = [_worker(ps) for ps in paths_seeds]
    else:
        with Pool(workers) as pool:
            out = pool.map(_worker, paths_seeds, chunksize=1)
    raw = np.array([o[0] for o in out])
    sub = np.array([o[1] for o in out])
    fs = np.array([o[2] for o in out])
    return raw, sub, fs, out[0][3]


# --------------------------------------------------------------------------- #
def rebin_per_spectrum(matrix, ells, factor):
    """Rebin each of the 10 spectra (concatenated along axis 1) by `factor`, and return
    the rebinned matrix plus the rebinned per-spectrum ell centers (one spectrum's worth)."""
    nspec = len(SPEC_KEYS)
    nell = len(ells)
    assert matrix.shape[1] == nspec * nell
    blocks = []
    for s in range(nspec):
        block = matrix[:, s * nell:(s + 1) * nell]
        ncoarse = nell // factor
        block = block[:, :ncoarse * factor].reshape(matrix.shape[0], ncoarse, factor).mean(axis=2)
        blocks.append(block)
    ec = ells[:(nell // factor) * factor].reshape(-1, factor).mean(axis=1)
    return np.concatenate(blocks, axis=1), ec


def select_band(matrix, ell_centers, lo, hi):
    """Keep coarse bandpowers with lo <= center < hi in every spectrum."""
    keep = np.where((ell_centers >= lo) & (ell_centers < hi))[0]
    nspec = len(SPEC_KEYS)
    ncoarse = len(ell_centers)
    cols = np.concatenate([keep + s * ncoarse for s in range(nspec)])
    return matrix[:, cols], len(keep)


def fisher_sigmas(grid_vecs, params, fid_vecs):
    """Linear Jacobian (lstsq) + Hartlap-corrected sample covariance -> param sigmas."""
    n_fid, n_data = fid_vecs.shape
    if n_data >= n_fid - 2:
        return None  # covariance not invertible / Hartlap undefined
    # Jacobian: fit data = a + params @ J^T  ->  J is (n_data, n_params)
    X = np.column_stack([np.ones(len(params)), params - params.mean(0)])
    coef, *_ = np.linalg.lstsq(X, grid_vecs, rcond=None)   # (1+n_params, n_data)
    J = coef[1:].T                                          # (n_data, n_params)
    C = np.cov(fid_vecs, rowvar=False)                      # (n_data, n_data)
    Cinv = np.linalg.inv(C)
    hartlap = (n_fid - n_data - 2) / (n_fid - 1)            # Hartlap 2007
    Cinv *= hartlap
    F = J.T @ Cinv @ J
    cov = np.linalg.inv(F)
    sig = np.sqrt(np.diag(cov))
    # 2D (Omega_m, S8) figure of merit: 1/sqrt(det) of the 2x2 sub-cov
    sub = cov[np.ix_([0, 1], [0, 1])]
    area = np.pi * np.sqrt(max(np.linalg.det(sub), 0.0))    # ~1-sigma ellipse area
    return dict(sigmas=sig.tolist(), n_data=int(n_data), hartlap=float(hartlap),
                area_Om_S8=float(area), param_cov=cov.tolist())


# --------------------------------------------------------------------------- #
def run(args):
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    global _MASK, _UNIT
    mask, f_sky, radius = PSM.get_cached_mask(
        nside=CFG["nside"], target_area_sqdeg=CFG["mask_area"], center_coords=CFG["center"],
        apodization_type=CFG["apod_type"], apodization_scale_deg=CFG["apod_deg"])
    unit_mask = np.ones_like(mask)
    _MASK, _UNIT = mask, unit_mask
    print(f"[mask] f_sky={f_sky:.4f} radius={radius:.1f}")

    # ---- assemble grid (perm_0000) + params -------------------------------
    picked = grid_walk_perm0(args.n_grid)
    all_params = np.load(PARAMS_NPY)
    rows = [p[3] for p in picked]
    params = all_params[rows]              # (N, 6) aligned via the walk
    grid_ps = [(p[2], PSM.get_deterministic_seed(p[2], 42)) for p in picked]
    print(f"[grid] {len(picked)} cosmologies; param rows {rows[0]}..{rows[-1]}; "
          f"Om range {params[:,0].min():.3f}-{params[:,0].max():.3f}, "
          f"S8 range {params[:,1].min():.3f}-{params[:,1].max():.3f}")

    # ---- assemble fiducial perms ------------------------------------------
    fid_ps = []
    for i in range(min(args.n_fid, NPERM_FID)):
        path = os.path.join(FID_DIR, f"perm_{i:04d}", MAP_NAME)
        if os.path.exists(path):
            fid_ps.append((path, PSM.get_deterministic_seed(path, 42)))
    print(f"[fid] {len(fid_ps)} perms")

    # Pre-build both coupling matrices in the parent so forked workers inherit them
    # (copy-on-write) instead of each rebuilding -- big saving at lmax 1535.
    print("[mcm] pre-building masked + unit coupling matrices in parent ...", flush=True)
    _ = datavector(load_noisy_maps(grid_ps[0][0], grid_ps[0][1]), mask, False)
    _ = datavector(load_noisy_maps(grid_ps[0][0], grid_ps[0][1]), unit_mask, False)

    grid_raw, grid_sub, grid_fs, ells = build_arms(grid_ps, args.workers, "grid")
    fid_raw, fid_sub, fid_fs, _ = build_arms(fid_ps, args.workers, "fid")

    # Save raw datavectors so contours / alternate cuts can be re-plotted without recomputing.
    np.savez_compressed(
        os.path.join(out_dir, "datavectors.npz"),
        ells=ells, params=params, fid_params=all_params[0],
        grid_raw=grid_raw, grid_sub=grid_sub, grid_fs=grid_fs,
        fid_raw=fid_raw, fid_sub=fid_sub, fid_fs=fid_fs)

    # ---- Fisher for each arm x band, rebinned ------------------------------
    arms = {"masked_raw": (grid_raw, fid_raw), "masked_sub": (grid_sub, fid_sub),
            "fullsky": (grid_fs, fid_fs)}
    # (lo, hi, rebin): fine bins for the narrow recovery band, coarse for the wide bands
    # so the covariance stays invertible with n_fid perms.
    bands = {"30-100": (30, 100, args.rebin),
             "100-1024": (100, 1024, args.rebin_wide),
             "30-1024": (30, 1024, args.rebin_wide)}
    results = {}
    for aname, (g, fd) in arms.items():
        for bname, (lo, hi, rb) in bands.items():
            gr, ec = rebin_per_spectrum(g, ells, rb)
            fr, _ = rebin_per_spectrum(fd, ells, rb)
            gband, nb = select_band(gr, ec, lo, hi)
            fband, _ = select_band(fr, ec, lo, hi)
            res = fisher_sigmas(gband, params, fband)
            key = f"{aname}|{bname}"
            if res is None:
                results[key] = dict(skipped="cov not invertible", n_coarse_bandpowers=int(nb))
                print(f"  {key:28s} SKIP (n_data too large: {nb} bp x10)")
            else:
                results[key] = {**res, "n_coarse_bandpowers": int(nb)}
                print(f"  {key:28s} sigma(Om)={res['sigmas'][0]:.4f} "
                      f"sigma(S8)={res['sigmas'][1]:.4f} area(Om,S8)={res['area_Om_S8']:.3e}")

    summary = dict(config=CFG, f_sky=f_sky, n_grid=len(picked), n_fid=len(fid_ps),
                   rebin=args.rebin, results=results)
    with open(os.path.join(out_dir, "fisher_gate_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    # ---- verdict ----------------------------------------------------------
    print("\n=== GATE VERDICT (sigma(S8), area) ===")
    def g(k): return results.get(k, {})
    for band in ["30-1024", "30-100", "100-1024"]:
        mr, ms, fs = g(f"masked_raw|{band}"), g(f"masked_sub|{band}"), g(f"fullsky|{band}")
        print(f"  band {band}:")
        for nm, r in [("masked_raw", mr), ("masked_sub", ms), ("fullsky", fs)]:
            if "sigmas" in r:
                print(f"     {nm:12s} sigma(S8)={r['sigmas'][1]:.4f}  area={r['area_Om_S8']:.3e}")
    print(f"\n[done] wrote {out_dir}/fisher_gate_summary.json")
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-grid", type=int, default=300)
    p.add_argument("--n-fid", type=int, default=200)
    p.add_argument("--workers", type=int, default=40,
                   help="Parallel worker processes (titan has 128 cores).")
    p.add_argument("--rebin", type=int, default=8,
                   help="Rebin factor for the narrow 30-100 recovery band.")
    p.add_argument("--rebin-wide", type=int, default=24,
                   help="Coarser rebin for the wide 100-1024 / 30-1024 bands (cov invertibility).")
    p.add_argument("--pilot", action="store_true", help="Tiny run to validate the pipeline.")
    p.add_argument("--out-dir", type=str,
                   default=os.path.join(REPO, "outputs", "diagnostics", "fisher_gate"))
    args = p.parse_args()
    if args.pilot:
        args.n_grid, args.n_fid = 20, 30
    run(args)


if __name__ == "__main__":
    main()
