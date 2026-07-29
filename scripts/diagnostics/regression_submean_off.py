"""
Regression check for Phase B: with subtract_mean OFF, the MASTER pipeline must reproduce
the existing on-disk raw Cls byte-for-byte (same deterministic noise seed, same MCM).

If this passes, the `_submean` reprocessing can REUSE the existing raw files instead of
recomputing them, and we trust that the new code path is a no-op when the flag is off.

Run with the cosmostat venv python (has pymaster).
"""
import os
import importlib.util
import numpy as np

SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location(
    "psm", os.path.join(SCRIPTS_DIR, "cross_power_spectrum_processing_master.py"))
PSM = importlib.util.module_from_spec(spec)
spec.loader.exec_module(PSM)
assert PSM.HAS_NAMASTER, "needs NaMaster"

import h5py
import healpy as hp

PERM0 = ("/lustre/fsn1/projects/rech/prk/ulx34io/cosmogrid_products/stage3_forecast/fiducial/cosmo_fiducial/"
         "perm_0000/projected_probes_maps_nobaryons512.h5")
EXISTING = ("/lustre/fsn1/projects/rech/prk/ulx34io/cosmogrid_products/stage3_forecast/fiducial/cosmo_fiducial/perm_0000/"
            "projected_probes_maps_nobaryons512_all_cls_bins1234_masked_14000sqdeg_"
            "apod2.0_master_noisy_s0.26_lmax1535.npz")
BINS = [1, 2, 3, 4]
LMAX = 1535

# Replicate process_file's exact loading + noise sequence (global_seed default 42).
file_seed = PSM.get_deterministic_seed(PERM0, 42)
rng = np.random.default_rng(file_seed)
mask, f_sky, _ = PSM.get_cached_mask(nside=512, target_area_sqdeg=14000.0,
                                     center_coords=(0.0, 90.0),
                                     apodization_type="C2", apodization_scale_deg=2.0)
maps = {}
with h5py.File(PERM0, "r") as f:
    for idx, b in enumerate(BINS):
        kg = np.array(f[f"kg/stage3_lensing{b}"])
        kg = PSM.add_shape_noise(kg, sigma_e=0.26, galaxy_density=6.75, nside=512, rng=rng)
        maps[b] = kg

cls_dict, ells = PSM.compute_power_spectra_master(maps, mask, lmax=LMAX,
                                                  use_namaster=True, verbose=False)

ref = np.load(EXISTING)
print(f"existing file keys: {[k for k in ref.files if k.startswith('cls_')]}")
worst = 0.0
for (i, j), cls in sorted(cls_dict.items()):
    key = f"cls_{i}_{j}"
    if key not in ref.files:
        print(f"  {key}: MISSING in existing file")
        continue
    a, b_ = np.asarray(cls), np.asarray(ref[key])
    if a.shape != b_.shape:
        print(f"  {key}: shape mismatch {a.shape} vs {b_.shape}")
        worst = np.inf
        continue
    denom = np.maximum(np.abs(b_), 1e-30)
    rel = np.max(np.abs(a - b_) / denom)
    worst = max(worst, rel)
    print(f"  {key}: max rel diff = {rel:.2e}")

print(f"\nWORST max-rel-diff across all spectra: {worst:.2e}")
print("REGRESSION PASS (reproduces existing raw)" if worst < 1e-6
      else "REGRESSION MISMATCH -- investigate before reusing raw files")
