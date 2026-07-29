#!/usr/bin/env python3
"""
Combined L1-norm + Peak-count processing.

Computes BOTH wavelet L1 norms and multiscale peak counts from each convergence
map in a SINGLE starlet transform, instead of running l1_norm_processing.py and
peak_counts_processing.py as two separate jobs that each pay the (dominant)
spherical-harmonic-transform cost.

The per-map outputs are byte-for-byte the same as the two standalone scripts:
  * L1   : <file>_l1_norms_bin{B}{mask}{noise}_new_normalization.npy   shape (nscales, l1_nbins)
  * Peaks: <file>_peak_counts_bin{B}{mask}{noise}_new_normalization.npy shape (nscales, pk_nbins-1)
so existing aggregation / NPE steps work unchanged. The only difference is that
the starlet coefficients are computed once and shared between the two statistics
(plus the coef/TabNorm/noise_std normalization, which is identical in both).

See scripts/pycs_speedups.py for the transform speedups (applied below).
"""

import os
import io
import sys
import argparse
import contextlib
import multiprocessing as mp
from functools import partial

import h5py
import healpy as hp
import numpy as np
from tqdm import tqdm

from pycs.sparsity.mrs.mrs_starlet import CMRStarlet
from pycs.astro.wl.hos_peaks_l1 import get_peaks_sphere

# Speed up the pycs spherical starlet transform (map2alm iter, neighbour cache).
# See scripts/pycs_speedups.py. Must run before the multiprocessing Pool is
# created so forked workers inherit the patches.
import pycs_speedups
pycs_speedups.enable(starlet_iter=1)

# Global mask cache
MASK_CACHE = {}


@contextlib.contextmanager
def suppress_stdout():
    """Context manager to suppress stdout output."""
    saved_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = saved_stdout


def seed_worker():
    """Initializer for multiprocessing pool to ensure unique random seeds."""
    np.random.seed(int.from_bytes(os.urandom(4), byteorder="little"))


def add_shape_noise(kg, sigma_e=0.26, galaxy_density=6.75, nside=512):
    """Add shape noise to a full-sky Healpix convergence (kappa) map.

    Convergence is a scalar field, so there is no factor of 2 (unlike shear).
    """
    npix = hp.nside2npix(nside)
    pixel_area_arcmin2 = hp.nside2pixarea(nside, degrees=True) * 3600
    sigma_pix = sigma_e / np.sqrt(galaxy_density * pixel_area_arcmin2)
    noise = np.random.normal(loc=0, scale=sigma_pix, size=npix)
    return kg + noise


def create_euclid_mask(nside=512, target_area_sqdeg=14000.0, center_coords=(0.0, 90.0)):
    """Create a contiguous Euclid-like disk mask of a specific sky area."""
    total_area_sqdeg = 41252.96125  # 4 * pi * (180/pi)^2
    angular_radius_rad = np.arccos(1 - (target_area_sqdeg / total_area_sqdeg) * 2)
    angular_radius_deg = np.rad2deg(angular_radius_rad)

    theta_center = np.deg2rad(90.0 - center_coords[1])
    phi_center = np.deg2rad(center_coords[0])
    center_vec = hp.ang2vec(theta_center, phi_center)

    disc_pixels = hp.query_disc(nside, center_vec, angular_radius_rad)

    npix = hp.nside2npix(nside)
    mask = np.zeros(npix, dtype=np.float32)
    mask[disc_pixels] = 1.0

    f_sky = mask.mean()
    return mask, f_sky, angular_radius_deg


def get_cached_mask(nside=512, target_area_sqdeg=14000.0, center_coords=(0.0, 90.0)):
    """Return a cached Euclid-like mask to avoid recomputation in each worker."""
    key = (int(nside), float(target_area_sqdeg), float(center_coords[0]), float(center_coords[1]))
    if key not in MASK_CACHE:
        MASK_CACHE[key] = create_euclid_mask(
            nside=nside, target_area_sqdeg=target_area_sqdeg, center_coords=center_coords
        )
    return MASK_CACHE[key]


def compute_l1_and_peaks(
    kappa_map,
    nscales=5,
    noise_std=0.0146,
    l1_nbins=40,
    l1_min_snr=-13,
    l1_max_snr=13,
    l1_min_snr_coarse=None,
    l1_max_snr_coarse=None,
    pk_nbins=31,
    pk_min=-2,
    pk_max=10,
    mask=None,
):
    """Compute wavelet L1 norms AND multiscale peak counts from one transform.

    Mirrors ``get_wtl1_sphere`` and ``get_wtpeaks_sphere`` exactly (same
    normalization, binning and peak logic), but performs the expensive starlet
    transform a single time and reuses the per-scale normalized coefficients for
    both statistics.

    Parameters
    ----------
    mask : array_like or None
        If None (default), every pixel on the sphere is binned -- the faithful
        behavior of the standalone scripts. If a HEALPix mask is given, the
        statistics are restricted to ``mask != 0`` exactly as pycs's ``Mask=``
        argument does: for L1 only the in-mask coefficients are binned; for peaks
        the full coefficient map is used for the local-maximum test but only
        in-mask pixels can be counted as peaks. Use for masked runs so the
        statistic is measured only where there is data.

    Returns
    -------
    l1_bins   : (nscales, l1_nbins) bin centers for the L1 norms
    l1_norms  : (nscales, l1_nbins) L1 norm per bin
    pk_counts : (nscales, pk_nbins-1) peak-count histogram per scale
    pk_centers: (pk_nbins-1,) peak histogram bin centers
    """
    Nside = hp.npix2nside(kappa_map.shape[0])

    # --- The single expensive step: starlet transform (the SHTs) ---
    C = CMRStarlet()
    C.init_starlet(Nside, nscale=nscales)
    C.transform(kappa_map)

    # Peak-count histogram bins (shared across scales; matches get_wtpeaks_sphere)
    pk_bins = np.linspace(pk_min, pk_max, pk_nbins)
    pk_centers = 0.5 * (pk_bins[:-1] + pk_bins[1:])

    l1_bins_coll = []
    l1_norm_coll = []
    pk_counts = []

    for i in range(nscales):
        # Normalized coefficients (identical input for L1 and peaks)
        if C.TabNorm[i] == 0:
            base = C.coef[i].copy()
        else:
            base = C.coef[i] / C.TabNorm[i]
        if noise_std is not None:
            base = base / noise_std

        # --- L1 norm --- matches get_wtl1_sphere (mask restricts the binned pixels)
        l1_vals = base if mask is None else base[mask != 0]
        is_coarse = i == nscales - 1
        if is_coarse:
            if l1_min_snr_coarse is not None:
                lo = l1_min_snr_coarse
            elif l1_min_snr is not None:
                lo = l1_min_snr
            else:
                lo = np.min(l1_vals)
            if l1_max_snr_coarse is not None:
                hi = l1_max_snr_coarse
            elif l1_max_snr is not None:
                hi = l1_max_snr
            else:
                hi = np.max(l1_vals)
        else:
            lo = l1_min_snr if l1_min_snr is not None else np.min(l1_vals)
            hi = l1_max_snr if l1_max_snr is not None else np.max(l1_vals)

        thresholds = np.linspace(lo, hi, l1_nbins + 1)
        l1_bins_coll.append(0.5 * (thresholds[:-1] + thresholds[1:]))
        digitized = np.digitize(l1_vals, thresholds)
        l1_norm_coll.append(
            [np.sum(np.abs(l1_vals[digitized == j])) for j in range(1, len(thresholds))]
        )

        # --- Peak counts --- matches get_wtpeaks_sphere (mask gates which pixels
        # can be peaks; the full map is still used for the local-maximum test)
        _, peak_heights = get_peaks_sphere(base, Nside, threshold=None, ordered=True, mask=mask)
        counts, _ = np.histogram(peak_heights, bins=pk_bins)
        pk_counts.append(counts)

    return (
        np.array(l1_bins_coll),
        np.array(l1_norm_coll),
        np.array(pk_counts),
        pk_centers,
    )


def process_file(
    file_path,
    bin_number=1,
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
    """Process one file: extract kappa map, optional mask, compute BOTH stats, save.

    With ``mask_correction`` (only meaningful when ``apply_mask``), the statistics
    are measured only inside the footprint (mask passed to ``compute_l1_and_peaks``)
    instead of over the whole sphere. The non-BNT order is already noise->mask
    (outside = 0), so only the binning restriction is added here. Corrected outputs
    carry a ``_maskcorr`` tag so they never collide with the faithful outputs.
    """
    mask_suffix = ""
    if apply_mask:
        mask_suffix = f"_masked_{int(round(mask_area_sqdeg))}sqdeg"
    corr_suffix = "_maskcorr" if (apply_mask and mask_correction) else ""
    noise_tag = f"_noisy_s{noise_level:.2f}" if add_noise else ""

    l1_path = file_path.replace(
        ".h5", f"_l1_norms_bin{bin_number}{mask_suffix}{corr_suffix}{noise_tag}_new_normalization.npy"
    )
    pk_path = file_path.replace(
        ".h5", f"_peak_counts_bin{bin_number}{mask_suffix}{corr_suffix}{noise_tag}_new_normalization.npy"
    )

    # Skip only if BOTH outputs already exist (recompute if either is missing).
    if os.path.exists(l1_path) and os.path.exists(pk_path) and not force_overwrite:
        if verbose:
            print(f"Skipping {os.path.basename(file_path)}, both outputs exist.")
        return (l1_path, pk_path)

    map_key = f"kg/stage3_lensing{bin_number}"
    try:
        with h5py.File(file_path, "r") as f:
            kg = np.array(f[map_key])

        if add_noise:
            kg = add_shape_noise(kg, sigma_e=noise_level)

        count_mask = None
        if apply_mask:
            nside = hp.get_nside(kg)
            mask, _, _ = get_cached_mask(
                nside=nside, target_area_sqdeg=mask_area_sqdeg, center_coords=mask_center
            )
            kg = kg * mask  # order A: outside footprint = 0
            if mask_correction:
                count_mask = mask  # measure the statistic only inside the footprint

        _, l1norms, pk_counts, _ = compute_l1_and_peaks(
            kg,
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
            print(f"Processed: {os.path.basename(file_path)} -> L1 + peaks")
        return (l1_path, pk_path)

    except Exception as e:
        if verbose:
            print(f"Error processing {os.path.basename(file_path)}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Compute wavelet L1 norms AND peak counts in a single starlet transform."
    )

    # Dataset selection (mirrors l1_norm_processing.py / peak_counts_processing.py)
    parser.add_argument("--fiducial", action="store_true",
                        help="Process fiducial cosmology instead of grid cosmologies.")
    parser.add_argument("--base-dir", help="Override default base directory for data.")
    parser.add_argument("--baryonified", action="store_true",
                        help="Use baryonified maps instead of nobaryons maps.")

    bin_group = parser.add_mutually_exclusive_group()
    bin_group.add_argument("--bin-number", type=int, default=1,
                           help="Single bin number to process.")
    bin_group.add_argument("--bins", type=str,
                           help="Comma-separated list of bin numbers (e.g. '1,2,3,4').")

    # Noise
    parser.add_argument("--noise-level", type=float, default=0.26, help="Shape noise (sigma_e).")
    parser.add_argument("--no-noise", action="store_true", help="Don't add shape noise.")
    parser.add_argument("--noise-std", type=float, default=0.0146,
                        help="Noise std for wavelet normalization.")

    # Mask
    parser.add_argument("--apply-mask", action="store_true",
                        help="Apply Euclid-like sky mask before computing statistics.")
    parser.add_argument("--mask-area-sqdeg", type=float, default=14000.0,
                        help="Area of the Euclid-like mask in sq deg (default: 14000).")
    parser.add_argument("--mask-center", type=float, nargs=2, metavar=("LON", "LAT"),
                        default=(0.0, 90.0), help="Mask centre (lon, lat) deg (default: 0 90).")
    parser.add_argument("--mask-correction", action="store_true",
                        help="Measure the statistics only inside the footprint (pass the "
                             "mask to the HOS binning) instead of over the whole sphere. "
                             "Requires --apply-mask. Writes outputs with a '_maskcorr' tag "
                             "so they do not collide with the faithful (default) outputs.")

    # Transform
    parser.add_argument("--nscales", type=int, default=5, help="Number of wavelet scales.")

    # L1 algorithm params (defaults match l1_norm_processing.py)
    parser.add_argument("--l1-nbins", type=int, default=40, help="Number of L1 bins.")
    parser.add_argument("--l1-min-snr", type=float, default=-13, help="L1 min SNR.")
    parser.add_argument("--l1-max-snr", type=float, default=13, help="L1 max SNR.")

    # Peak algorithm params (defaults match peak_counts_processing.py)
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

    mask_center = tuple(args.mask_center)

    if args.bins:
        bin_numbers = [int(b.strip()) for b in args.bins.split(",")]
        print(f"Processing multiple bins: {bin_numbers}")
    else:
        bin_numbers = [args.bin_number]
        print(f"Processing single bin: {args.bin_number}")

    map_type = "baryonified" if args.baryonified else "nobaryons"
    dataset_type = "fiducial" if args.fiducial else "grid"
    print(f"Processing {len(file_paths)} {map_type} files from {dataset_type} dataset")
    print("Computing BOTH L1 norms and peak counts per map (single transform).")

    if args.apply_mask:
        _, mask_f_sky, mask_radius = get_cached_mask(
            nside=512, target_area_sqdeg=args.mask_area_sqdeg, center_coords=mask_center
        )
        print(
            f"Applying Euclid-like mask: {args.mask_area_sqdeg:.0f} sq deg "
            f"(f_sky≈{mask_f_sky:.3f}, radius≈{mask_radius:.2f}°) "
            f"centered at lon={mask_center[0]:.1f}°, lat={mask_center[1]:.1f}°"
        )
        if args.mask_correction:
            print("Mask correction ON: statistics measured inside the footprint; "
                  "outputs tagged '_maskcorr'.")

    for bin_number in bin_numbers:
        print(f"\n{'='*60}\nProcessing bin {bin_number}\n{'='*60}")

        with mp.Pool(processes=args.num_workers, initializer=seed_worker) as pool:
            process_func = partial(
                process_file,
                bin_number=bin_number,
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
                    desc=f"Processing bin {bin_number}",
                )
            )

        successful = [r for r in results if r is not None]
        l1_done = len([1 for r in successful if os.path.exists(r[0])])
        pk_done = len([1 for r in successful if os.path.exists(r[1])])
        print(
            f"Bin {bin_number} complete: {len(successful)}/{len(file_paths)} maps "
            f"(L1 files: {l1_done}, peak files: {pk_done})"
        )

    print(f"\n{'='*60}\nAll bins processing complete\n{'='*60}")


if __name__ == "__main__":
    main()
