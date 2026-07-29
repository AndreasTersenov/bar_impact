#!/usr/bin/env python3
"""
BNT combined L1-norm + Peak-count processing.

BNT analogue of l1_peaks_processing.py: applies the BNT transform to the four
tomographic convergence maps, then computes BOTH wavelet L1 norms and multiscale
peak counts for a selected BNT bin from a SINGLE starlet transform per map —
instead of running bnt_l1_norm_processing.py and bnt_peak_counts_processing.py as
two separate jobs that each redo the BNT prep and the (dominant) spherical-
harmonic transform.

Per-map outputs match the two standalone BNT scripts byte-for-byte and use the
same filenames:
  * L1   : <file>_bnt_l1_norms_bin{B+1}{mask}{noise}_new_normalization.npy
  * Peaks: <file>_bnt_peak_counts_bin{B+1}{mask}{noise}_new_normalization.npy
so existing aggregation / NPE steps are unaffected.

Map prep mirrors the standalone BNT scripts exactly: load all 4 bins -> apply
mask to all (if requested) -> add shape noise to all (if requested) -> apply
BNT_MATRIX -> select the requested BNT bin. (Note this order masks BEFORE adding
noise, matching the BNT scripts.)

NOTE ON NOISE: because L1 and peaks now come from one transform of one prepared
map, they share the SAME shape-noise realization for a given (file, BNT bin).
The two standalone scripts, run separately, drew INDEPENDENT noise for L1 vs
peaks. This is irrelevant if L1 and peaks are analyzed separately (each marginal
is statistically identical), but it changes the L1<->peaks cross-covariance if
you ever build a JOINT L1+peaks data vector. Use the standalone scripts if you
need independent L1/peak noise. The reused stats core is identical to
l1_peaks_processing.compute_l1_and_peaks (verified bit-identical to
get_wtl1_sphere / get_wtpeaks_sphere).

The per-bin coarse-SNR arguments of bnt_peak_counts_processing.py are not applied
there (parsed but never passed to get_wtpeaks_sphere), so this script likewise
uses uniform peak bins, reproducing the standalone output.
"""

import os
import argparse
import multiprocessing as mp
from functools import partial

import h5py
import healpy as hp
import numpy as np
from tqdm import tqdm

# Speed up the pycs spherical starlet transform (map2alm iter, neighbour cache).
import pycs_speedups
pycs_speedups.enable(starlet_iter=1)

# Reuse the stats core + helpers from the non-BNT combined script (one source of
# truth; guarantees outputs stay consistent with l1_peaks_processing.py).
from l1_peaks_processing import (
    compute_l1_and_peaks,
    add_shape_noise,
    create_euclid_mask,
    get_cached_mask,
    seed_worker,
)

# BNT transformation matrix (hard-coded, identical to bnt_*_processing.py)
BNT_MATRIX = np.array([[ 1.        ,  0.        ,  0.        ,  0.        ],
                       [-1.        ,  1.        ,  0.        ,  0.        ],
                       [ 0.4521097 , -1.4521097 ,  1.        ,  0.        ],
                       [ 0.        ,  0.25127807, -1.251278  ,  1.        ]])


def process_file(
    file_path,
    bnt_bin=3,
    noise_level=0.26,
    add_noise=True,
    nscales=5,
    noise_std=0.0146,
    l1_nbins=40,
    l1_min_snr=-13,
    l1_max_snr=13,
    pk_nbins=31,
    pk_min=-2,
    pk_max=10,
    apply_mask=False,
    mask_area_sqdeg=14000.0,
    mask_center=(0.0, 90.0),
    mask_correction=False,
    force_overwrite=False,
    verbose=False,
):
    """Load 4 bins, BNT-transform, compute BOTH stats for one BNT bin, save.

    Faithful (default): mask -> noise -> BNT, statistics over the whole sphere
    (matches the standalone BNT scripts).

    With ``mask_correction`` (only when ``apply_mask``): noise -> mask -> BNT, so
    the unobserved region is zero (order A, reproducible on real data), and the
    statistics are measured only inside the footprint (mask passed to
    ``compute_l1_and_peaks``). Corrected outputs carry a ``_maskcorr`` tag so they
    never collide with the faithful outputs.
    """
    mask_suffix = ""
    if apply_mask:
        area_tag = int(round(mask_area_sqdeg)) if mask_area_sqdeg else "mask"
        mask_suffix = f"_masked_{area_tag}sqdeg"
    corr_suffix = "_maskcorr" if (apply_mask and mask_correction) else ""
    noise_tag = f"_noisy_s{noise_level:.2f}" if add_noise else ""

    l1_path = file_path.replace(
        ".h5", f"_bnt_l1_norms_bin{bnt_bin+1}{mask_suffix}{corr_suffix}{noise_tag}_new_normalization.npy"
    )
    pk_path = file_path.replace(
        ".h5", f"_bnt_peak_counts_bin{bnt_bin+1}{mask_suffix}{corr_suffix}{noise_tag}_new_normalization.npy"
    )

    # Skip only if BOTH outputs already exist (recompute if either is missing).
    if os.path.exists(l1_path) and os.path.exists(pk_path) and not force_overwrite:
        if verbose:
            print(f"Skipping {os.path.basename(file_path)}, both BNT outputs exist.")
        return (l1_path, pk_path)

    try:
        mask_arr = None
        if apply_mask:
            mask_arr, _, _ = get_cached_mask(
                nside=512, target_area_sqdeg=mask_area_sqdeg, center_coords=mask_center
            )

        # Load all 4 kappa maps
        kgs = []
        with h5py.File(file_path, "r") as f:
            for i in range(4):
                kgs.append(np.array(f[f"kg/stage3_lensing{i+1}"]))

        if mask_correction:
            # order A: noise -> mask (outside footprint = 0) -> BNT.
            # Reproducible on real data; mask commutes with BNT (common mask).
            if add_noise:
                kgs = [add_shape_noise(kg, sigma_e=noise_level) for kg in kgs]
            if apply_mask:
                kgs = [kg * mask_arr for kg in kgs]
        else:
            # faithful: mask -> noise -> BNT (matches the standalone BNT scripts)
            if apply_mask:
                kgs = [kg * mask_arr for kg in kgs]
            if add_noise:
                kgs = [add_shape_noise(kg, sigma_e=noise_level) for kg in kgs]

        # Apply BNT transform and select the requested BNT bin
        kgs = np.array(kgs)
        kgs_bnt = BNT_MATRIX @ kgs

        count_mask = mask_arr if (apply_mask and mask_correction) else None
        _, l1norms, pk_counts, _ = compute_l1_and_peaks(
            kgs_bnt[bnt_bin],
            nscales=nscales,
            noise_std=noise_std,
            l1_nbins=l1_nbins,
            l1_min_snr=l1_min_snr,
            l1_max_snr=l1_max_snr,
            pk_nbins=pk_nbins,
            pk_min=pk_min,
            pk_max=pk_max,
            mask=count_mask,
        )

        if pk_counts.size == 0 or l1norms.size == 0:
            if verbose:
                print(f"Warning: empty result for {os.path.basename(file_path)}")
            return None

        np.save(l1_path, l1norms)
        np.save(pk_path, pk_counts)
        if verbose:
            print(f"Processed: {os.path.basename(file_path)} -> BNT bin {bnt_bin+1} L1 + peaks")
        return (l1_path, pk_path)

    except Exception as e:
        if verbose:
            print(f"Error processing {os.path.basename(file_path)}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Apply BNT transform, then compute L1 norms AND peak counts in one starlet transform."
    )

    parser.add_argument("--fiducial", action="store_true",
                        help="Process fiducial cosmology instead of grid cosmologies.")
    parser.add_argument("--base-dir", help="Override default base directory for data.")
    parser.add_argument("--baryonified", action="store_true",
                        help="Use baryonified maps instead of nobaryons maps.")

    # BNT bin selection (0-indexed; output filenames are 1-indexed)
    bin_group = parser.add_mutually_exclusive_group()
    bin_group.add_argument("--bnt-bin", type=int, default=None,
                           help="Single BNT bin to analyze (0-3).")
    bin_group.add_argument("--bnt-bins", type=str, default=None,
                           help="Comma-separated list of BNT bins (e.g. '0,1,2,3').")

    # Noise
    parser.add_argument("--noise-level", type=float, default=0.26, help="Shape noise (sigma_e).")
    parser.add_argument("--no-noise", action="store_true", help="Don't add shape noise.")
    parser.add_argument("--noise-std", type=float, default=0.0146,
                        help="Noise std for wavelet normalization.")

    # Mask
    parser.add_argument("--apply-mask", action="store_true",
                        help="Apply Euclid-like sky mask before the BNT transform.")
    parser.add_argument("--mask-area-sqdeg", type=float, default=14000.0,
                        help="Area of the Euclid-like mask in sq deg (default: 14000).")
    parser.add_argument("--mask-center", type=float, nargs=2, metavar=("LON", "LAT"),
                        default=(0.0, 90.0), help="Mask centre (lon, lat) deg (default: 0 90).")
    parser.add_argument("--mask-correction", action="store_true",
                        help="Corrected masked treatment: noise->mask->BNT (outside footprint "
                             "= 0, reproducible on data) AND measure the statistics only inside "
                             "the footprint. Requires --apply-mask. Writes outputs with a "
                             "'_maskcorr' tag so they do not collide with the faithful outputs.")

    # Transform
    parser.add_argument("--nscales", type=int, default=5, help="Number of wavelet scales.")

    # L1 params (defaults match bnt_l1_norm_processing.py)
    parser.add_argument("--l1-nbins", type=int, default=40, help="Number of L1 bins.")
    parser.add_argument("--l1-min-snr", type=float, default=-13, help="L1 min SNR.")
    parser.add_argument("--l1-max-snr", type=float, default=13, help="L1 max SNR.")

    # Peak params (defaults match bnt_peak_counts_processing.py)
    parser.add_argument("--pk-nbins", type=int, default=31, help="Number of peak histogram bins.")
    parser.add_argument("--pk-min", type=float, default=-2, help="Peak histogram min.")
    parser.add_argument("--pk-max", type=float, default=10, help="Peak histogram max.")

    # Execution
    parser.add_argument("--num-workers", type=int, default=70, help="Number of worker processes.")
    parser.add_argument("--verbose", action="store_true", help="Print detailed progress.")
    parser.add_argument("--force-overwrite", action="store_true",
                        help="Reprocess even if outputs already exist.")

    args = parser.parse_args()

    if args.mask_correction and not args.apply_mask:
        print("Warning: --mask-correction has no effect without --apply-mask; ignoring.")
        args.mask_correction = False

    # Parse BNT bins (default to bin 3, the 4th bin)
    if args.bnt_bins:
        bnt_bin_numbers = [int(b.strip()) for b in args.bnt_bins.split(",")]
    elif args.bnt_bin is not None:
        bnt_bin_numbers = [args.bnt_bin]
    else:
        bnt_bin_numbers = [3]
    for bnt_bin in bnt_bin_numbers:
        if bnt_bin < 0 or bnt_bin > 3:
            print(f"Error: BNT bin {bnt_bin} is out of range [0, 3].")
            return

    mask_center = tuple(args.mask_center)

    if args.base_dir:
        base_dir = args.base_dir
    elif args.fiducial:
        base_dir = "/lustre/fsn1/projects/rech/prk/ulx34io/cosmogrid_products/stage3_forecast/fiducial/cosmo_fiducial/"
    else:
        base_dir = "/lustre/fsn1/projects/rech/prk/ulx34io/cosmogrid_products/stage3_forecast/new_grid/"

    filename = (
        "projected_probes_maps_baryonified512.h5"
        if args.baryonified
        else "projected_probes_maps_nobaryons512.h5"
    )

    if args.fiducial:
        perm_dirs = [f"perm_{i:04d}" for i in range(200)]
        file_paths = [
            os.path.join(base_dir, perm, filename)
            for perm in perm_dirs
            if os.path.exists(os.path.join(base_dir, perm, filename))
        ]
    else:
        cosmo_dirs = sorted([d for d in os.listdir(base_dir) if d.startswith("cosmo_")])
        perm_dirs = [f"perm_{i:04d}" for i in range(7)]
        file_paths = [
            os.path.join(base_dir, cosmo, perm, filename)
            for cosmo in cosmo_dirs
            for perm in perm_dirs
            if os.path.exists(os.path.join(base_dir, cosmo, perm, filename))
        ]

    map_type = "baryonified" if args.baryonified else "nobaryons"
    dataset_type = "fiducial" if args.fiducial else "grid"
    print(f"Processing {len(file_paths)} {map_type} files from {dataset_type} dataset")
    print(f"Computing BNT L1 norms + peak counts for BNT bins: {[b+1 for b in bnt_bin_numbers]} (1-indexed)")
    print("Both statistics share a single starlet transform per map.")

    if args.apply_mask:
        _, f_sky, angular_radius_deg = get_cached_mask(
            nside=512, target_area_sqdeg=args.mask_area_sqdeg, center_coords=mask_center
        )
        print(
            f"Applying Euclid-like mask: {args.mask_area_sqdeg:.0f} sq deg "
            f"(f_sky≈{f_sky:.3f}, radius≈{angular_radius_deg:.2f}°) "
            f"centered at lon={mask_center[0]:.1f}°, lat={mask_center[1]:.1f}°"
        )
        if args.mask_correction:
            print("Mask correction ON: noise->mask->BNT and statistics measured inside "
                  "the footprint; outputs tagged '_maskcorr'.")

    for bnt_bin in bnt_bin_numbers:
        print(f"\n{'='*60}\nProcessing BNT bin {bnt_bin+1} (0-indexed: {bnt_bin})\n{'='*60}")

        with mp.Pool(processes=args.num_workers, initializer=seed_worker) as pool:
            process_func = partial(
                process_file,
                bnt_bin=bnt_bin,
                noise_level=args.noise_level,
                add_noise=not args.no_noise,
                nscales=args.nscales,
                noise_std=args.noise_std,
                l1_nbins=args.l1_nbins,
                l1_min_snr=args.l1_min_snr,
                l1_max_snr=args.l1_max_snr,
                pk_nbins=args.pk_nbins,
                pk_min=args.pk_min,
                pk_max=args.pk_max,
                apply_mask=args.apply_mask,
                mask_area_sqdeg=args.mask_area_sqdeg,
                mask_center=mask_center,
                mask_correction=args.mask_correction,
                force_overwrite=args.force_overwrite,
                verbose=args.verbose,
            )
            results = list(
                tqdm(
                    pool.imap(process_func, file_paths),
                    total=len(file_paths),
                    desc=f"BNT bin {bnt_bin+1}",
                )
            )

        successful = [r for r in results if r is not None]
        l1_done = len([1 for r in successful if os.path.exists(r[0])])
        pk_done = len([1 for r in successful if os.path.exists(r[1])])
        print(
            f"BNT bin {bnt_bin+1} complete: {len(successful)}/{len(file_paths)} maps "
            f"(L1 files: {l1_done}, peak files: {pk_done})"
        )

    print(f"\n{'='*60}\nAll BNT bins processing complete\n{'='*60}")


if __name__ == "__main__":
    main()
