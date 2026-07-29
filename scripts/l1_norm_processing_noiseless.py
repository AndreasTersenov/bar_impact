#!/usr/bin/env python3
"""
L1 norm processing script with a robust noiseless mode.

This script supports two definitions of wavelet L1 norms:
1) SNR-binned L1 norms (default for noisy maps):
   Uses pycs.get_wtl1_sphere with user-provided noise normalization.
2) Coefficient-binned L1 norms (default for noiseless maps):
   Computes starlet coefficients and bins directly in wavelet-coefficient space.

The coefficient-binned mode avoids relying on an external shape-noise scale,
which is ill-defined for noiseless simulations.
"""

import argparse
import contextlib
import io
import multiprocessing as mp
import os
import sys
import tempfile
from functools import partial

import h5py
import healpy as hp
import numpy as np
from pycs.astro.wl.hos_peaks_l1 import get_wtl1_sphere
from pycs.sparsity.mrs.mrs_starlet import mrs_uwttrans
from tqdm import tqdm

# Speed up the pycs spherical starlet transform (map2alm iter, neighbour cache).
# See scripts/pycs_speedups.py. Must run before the multiprocessing Pool is
# created so forked workers inherit the patches.
import pycs_speedups
pycs_speedups.enable(starlet_iter=1)

MASK_CACHE = {}


@contextlib.contextmanager
def suppress_stdout():
    """Context manager to suppress noisy third-party stdout output."""
    saved_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = saved_stdout


def seed_worker():
    """Initializer for multiprocessing workers with unique RNG seeds."""
    np.random.seed(int.from_bytes(os.urandom(4), byteorder="little"))


def add_shape_noise(kg, sigma_e=0.26, galaxy_density=6.75, nside=512):
    """Add shape noise to a full-sky HEALPix convergence map."""
    npix = hp.nside2npix(nside)
    pixel_area_arcmin2 = hp.nside2pixarea(nside, degrees=True) * 3600
    sigma_pix = sigma_e / np.sqrt(galaxy_density * pixel_area_arcmin2)
    noise = np.random.normal(loc=0.0, scale=sigma_pix, size=npix)
    return kg + noise


def create_euclid_mask(nside=512, target_area_sqdeg=14000.0, center_coords=(0.0, 90.0)):
    """Create a contiguous Euclid-like disk mask of a target sky area."""
    total_area_sqdeg = 41252.96125
    angular_radius_rad = np.arccos(1 - (target_area_sqdeg / total_area_sqdeg) * 2)
    angular_radius_deg = np.rad2deg(angular_radius_rad)

    theta_center = np.deg2rad(90.0 - center_coords[1])
    phi_center = np.deg2rad(center_coords[0])
    center_vec = hp.ang2vec(theta_center, phi_center)

    disc_pixels = hp.query_disc(nside, center_vec, angular_radius_rad)

    npix = hp.nside2npix(nside)
    mask = np.zeros(npix, dtype=np.float32)
    mask[disc_pixels] = 1.0

    f_sky = float(mask.mean())
    return mask, f_sky, angular_radius_deg


def get_cached_mask(nside=512, target_area_sqdeg=14000.0, center_coords=(0.0, 90.0)):
    """Return a cached Euclid-like mask to avoid recomputation in workers."""
    key = (int(nside), float(target_area_sqdeg), float(center_coords[0]), float(center_coords[1]))
    if key not in MASK_CACHE:
        MASK_CACHE[key] = create_euclid_mask(
            nside=nside,
            target_area_sqdeg=target_area_sqdeg,
            center_coords=center_coords,
        )
    return MASK_CACHE[key]


def compute_coeff_l1_norms(
    kg,
    nscales,
    nbins,
    coeff_ranges,
    mask=None,
    normalize_energy=False,
    temp_path="/.",
):
    """
    Compute L1 norms by binning starlet coefficients directly.

    Parameters
    ----------
    kg : np.ndarray
        Input convergence map.
    nscales : int
        Number of wavelet scales.
    nbins : int
        Number of coefficient bins.
    coeff_ranges : list[tuple(float, float)]
        Per-scale (min_coeff, max_coeff) used for fixed, comparable bins.
    mask : np.ndarray | None
        Optional binary mask (1=valid) applied in coefficient selection only.
    normalize_energy : bool
        If True, scale coefficients at each wavelet scale by sqrt(sum(c^2)).
    temp_path : str
        Path prefix used by pycs for temporary files.

    Returns
    -------
    bins_arr, l1_arr : tuple[np.ndarray, np.ndarray]
        Arrays of shape (nscales, nbins).
    """
    with suppress_stdout():
        wt = mrs_uwttrans(kg, nscale=nscales, verbose=False, path=temp_path)

    bins_coll = []
    l1_coll = []

    for scale_idx in range(nscales):
        coeffs = np.asarray(wt[scale_idx], dtype=np.float64)

        if mask is not None:
            coeffs = coeffs[mask != 0]

        if normalize_energy:
            norm = np.sqrt(np.sum(coeffs * coeffs))
            if norm > 0:
                coeffs = coeffs / norm

        min_coeff, max_coeff = coeff_ranges[scale_idx]
        if not np.isfinite(min_coeff) or not np.isfinite(max_coeff) or min_coeff >= max_coeff:
            raise ValueError(
                f"Invalid coeff range at scale {scale_idx}: [{min_coeff}, {max_coeff}]"
            )

        thresholds = np.linspace(min_coeff, max_coeff, nbins + 1)
        bin_centers = 0.5 * (thresholds[:-1] + thresholds[1:])

        # Clip out-of-range values into edge bins for stable total counts.
        digitized = np.digitize(coeffs, thresholds)
        digitized = np.clip(digitized, 1, nbins)

        l1_per_bin = np.bincount(
            digitized,
            weights=np.abs(coeffs),
            minlength=nbins + 1,
        )[1: nbins + 1]

        bins_coll.append(bin_centers)
        l1_coll.append(l1_per_bin)

    return np.asarray(bins_coll), np.asarray(l1_coll)


def estimate_coeff_ranges(
    file_paths,
    map_key,
    nscales,
    add_noise,
    noise_level,
    apply_mask,
    mask_area_sqdeg,
    mask_center,
    normalize_energy,
    lower_pct,
    upper_pct,
    sample_size,
):
    """Estimate robust per-scale coefficient ranges from a sample of files."""
    if lower_pct < 0 or upper_pct > 100 or lower_pct >= upper_pct:
        raise ValueError("Invalid percentile range for coefficient calibration.")

    sample_paths = file_paths[: min(sample_size, len(file_paths))]
    if not sample_paths:
        raise ValueError("No files available for coefficient range estimation.")

    lows = [[] for _ in range(nscales)]
    highs = [[] for _ in range(nscales)]

    for path in tqdm(sample_paths, desc="Estimating coeff ranges"):
        with h5py.File(path, "r") as f:
            kg = np.array(f[map_key])

        if add_noise:
            kg = add_shape_noise(kg, sigma_e=noise_level, nside=hp.get_nside(kg))

        if apply_mask:
            mask, _, _ = get_cached_mask(
                nside=hp.get_nside(kg),
                target_area_sqdeg=mask_area_sqdeg,
                center_coords=mask_center,
            )
        else:
            mask = None

        with tempfile.TemporaryDirectory() as tmp_dir:
            with suppress_stdout():
                wt = mrs_uwttrans(kg, nscale=nscales, verbose=False, path=tmp_dir)

        for i in range(nscales):
            coeffs = np.asarray(wt[i], dtype=np.float64)
            if mask is not None:
                coeffs = coeffs[mask != 0]

            if normalize_energy:
                norm = np.sqrt(np.sum(coeffs * coeffs))
                if norm > 0:
                    coeffs = coeffs / norm

            low = np.percentile(coeffs, lower_pct)
            high = np.percentile(coeffs, upper_pct)
            lows[i].append(low)
            highs[i].append(high)

    coeff_ranges = []
    for i in range(nscales):
        min_coeff = float(np.min(lows[i]))
        max_coeff = float(np.max(highs[i]))
        if min_coeff >= max_coeff:
            eps = 1e-12
            min_coeff -= eps
            max_coeff += eps
        coeff_ranges.append((min_coeff, max_coeff))

    return coeff_ranges


def process_file(
    file_path,
    bin_number=1,
    metric="snr",
    nscales=5,
    nbins=40,
    noise_level=0.26,
    add_noise=True,
    min_snr=-13.0,
    max_snr=13.0,
    noise_std=0.0146,
    verbose=False,
    apply_mask=False,
    mask_area_sqdeg=14000.0,
    mask_center=(0.0, 90.0),
    force_overwrite=False,
    min_snr_coarse=100.0,
    max_snr_coarse=200.0,
    coeff_ranges=None,
    normalize_energy=False,
):
    """Process one file and save L1 norms in either SNR or coefficient mode."""
    mask_suffix = ""
    if apply_mask:
        area_tag = int(round(mask_area_sqdeg))
        mask_suffix = f"_masked_{area_tag}sqdeg"

    metric_suffix = "_coeffbins" if metric == "coeff" else ""
    if add_noise:
        suffix = (
            f"_l1_norms_bin{bin_number}{mask_suffix}{metric_suffix}"
            f"_noisy_s{noise_level:.2f}_new_normalization.npy"
        )
    else:
        suffix = f"_l1_norms_bin{bin_number}{mask_suffix}{metric_suffix}_new_normalization.npy"

    save_path = file_path.replace(".h5", suffix)
    map_key = f"kg/stage3_lensing{bin_number}"

    if os.path.exists(save_path) and not force_overwrite:
        if verbose:
            print(f"Skipping {os.path.basename(file_path)} (exists)")
        return save_path

    try:
        with h5py.File(file_path, "r") as f:
            kg = np.array(f[map_key])

        if add_noise:
            kg = add_shape_noise(kg, sigma_e=noise_level, nside=hp.get_nside(kg))

        if apply_mask:
            mask, _, _ = get_cached_mask(
                nside=hp.get_nside(kg),
                target_area_sqdeg=mask_area_sqdeg,
                center_coords=mask_center,
            )
        else:
            mask = None

        if metric == "snr":
            # Keep pycs SNR-binned implementation for noisy-map compatibility.
            _, l1norms = get_wtl1_sphere(
                kg,
                nscales=nscales,
                nbins=nbins,
                Mask=mask,
                min_snr=min_snr,
                max_snr=max_snr,
                noise_std=noise_std,
                min_snr_coarse=min_snr_coarse,
                max_snr_coarse=max_snr_coarse,
            )
        elif metric == "coeff":
            if coeff_ranges is None:
                raise ValueError("coeff_ranges must be provided in coeff mode")
            with tempfile.TemporaryDirectory() as tmp_dir:
                _, l1norms = compute_coeff_l1_norms(
                    kg=kg,
                    nscales=nscales,
                    nbins=nbins,
                    coeff_ranges=coeff_ranges,
                    mask=mask,
                    normalize_energy=normalize_energy,
                    temp_path=tmp_dir,
                )
        else:
            raise ValueError(f"Unsupported metric: {metric}")

        np.save(save_path, l1norms)
        if verbose:
            print(f"Processed {os.path.basename(file_path)} -> {os.path.basename(save_path)}")
        return save_path

    except Exception as exc:
        if verbose:
            print(f"Error processing {os.path.basename(file_path)}: {exc}")
        return None


def discover_input_files(base_dir, fiducial, filename):
    """Discover input map files from either fiducial or grid layouts."""
    if fiducial:
        perm_dirs = [f"perm_{i:04d}" for i in range(200)]
        return [
            os.path.join(base_dir, perm, filename)
            for perm in perm_dirs
            if os.path.exists(os.path.join(base_dir, perm, filename))
        ]

    cosmo_dirs = sorted([d for d in os.listdir(base_dir) if d.startswith("cosmo_")])
    perm_dirs = [f"perm_{i:04d}" for i in range(7)]
    return [
        os.path.join(base_dir, cosmo, perm, filename)
        for cosmo in cosmo_dirs
        for perm in perm_dirs
        if os.path.exists(os.path.join(base_dir, cosmo, perm, filename))
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Compute wavelet L1 norms with SNR bins or coefficient bins."
    )

    parser.add_argument("--fiducial", action="store_true")
    parser.add_argument("--base-dir")
    parser.add_argument("--baryonified", action="store_true")

    bin_group = parser.add_mutually_exclusive_group()
    bin_group.add_argument("--bin-number", type=int, default=1)
    bin_group.add_argument("--bins", type=str)

    parser.add_argument("--noise-level", type=float, default=0.26)
    parser.add_argument("--no-noise", action="store_true")

    parser.add_argument("--metric", choices=["auto", "snr", "coeff"], default="auto")
    parser.add_argument("--nscales", type=int, default=5)
    parser.add_argument("--nbins", type=int, default=40)

    parser.add_argument("--apply-mask", action="store_true")
    parser.add_argument("--mask-area-sqdeg", type=float, default=14000.0)
    parser.add_argument(
        "--mask-center",
        type=float,
        nargs=2,
        metavar=("LON", "LAT"),
        default=(0.0, 90.0),
    )

    parser.add_argument("--min-snr", type=float, default=-13.0)
    parser.add_argument("--max-snr", type=float, default=13.0)
    parser.add_argument("--min-snr-coarse", type=str, default="10,40,100,150")
    parser.add_argument("--max-snr-coarse", type=str, default="50,100,200,300")
    parser.add_argument("--noise-std", type=float, default=0.0146)

    parser.add_argument("--coeff-min", type=float, default=None)
    parser.add_argument("--coeff-max", type=float, default=None)
    parser.add_argument("--coeff-lower-pct", type=float, default=0.5)
    parser.add_argument("--coeff-upper-pct", type=float, default=99.5)
    parser.add_argument("--coeff-calibration-files", type=int, default=64)
    parser.add_argument("--normalize-energy", action="store_true")

    parser.add_argument("--num-workers", type=int, default=70)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--force-overwrite", action="store_true")

    args = parser.parse_args()

    if args.base_dir:
        base_dir = args.base_dir
    elif args.fiducial:
        base_dir = "/lustre/fsn1/projects/rech/prk/ulx34io/cosmogrid_products/stage3_forecast/fiducial/cosmo_fiducial/"
    else:
        base_dir = "/lustre/fsn1/projects/rech/prk/ulx34io/cosmogrid_products/stage3_forecast/new_grid/"

    filename = "projected_probes_maps_baryonified512.h5" if args.baryonified else "projected_probes_maps_nobaryons512.h5"
    file_paths = discover_input_files(base_dir=base_dir, fiducial=args.fiducial, filename=filename)

    if not file_paths:
        raise RuntimeError("No input files found. Check base-dir/fiducial/baryonified options.")

    if args.bins:
        bin_numbers = [int(b.strip()) for b in args.bins.split(",")]
    else:
        bin_numbers = [args.bin_number]

    min_snr_coarse_list = [float(x.strip()) for x in args.min_snr_coarse.split(",")]
    max_snr_coarse_list = [float(x.strip()) for x in args.max_snr_coarse.split(",")]
    coarse_snr_min = {i + 1: min_snr_coarse_list[i] for i in range(len(min_snr_coarse_list))}
    coarse_snr_max = {i + 1: max_snr_coarse_list[i] for i in range(len(max_snr_coarse_list))}

    add_noise = not args.no_noise
    if args.metric == "auto":
        metric = "snr" if add_noise else "coeff"
    else:
        metric = args.metric

    mask_center = tuple(args.mask_center)
    dataset_type = "fiducial" if args.fiducial else "grid"
    map_type = "baryonified" if args.baryonified else "nobaryons"

    print(f"Processing {len(file_paths)} files ({dataset_type}, {map_type})")
    print(f"Bins: {bin_numbers}")
    print(f"Noise: {'on' if add_noise else 'off'}")
    print(f"Metric mode: {metric}")

    if args.apply_mask:
        _, mask_f_sky, mask_radius = get_cached_mask(
            nside=512,
            target_area_sqdeg=args.mask_area_sqdeg,
            center_coords=mask_center,
        )
        print(
            f"Mask: {args.mask_area_sqdeg:.0f} sqdeg "
            f"(f_sky~{mask_f_sky:.3f}, radius~{mask_radius:.2f} deg)"
        )

    for bin_number in bin_numbers:
        print("\n" + "=" * 60)
        print(f"Processing bin {bin_number}")
        print("=" * 60)

        bin_min_snr_coarse = coarse_snr_min.get(bin_number, 100.0)
        bin_max_snr_coarse = coarse_snr_max.get(bin_number, 200.0)

        coeff_ranges = None
        if metric == "coeff":
            map_key = f"kg/stage3_lensing{bin_number}"
            if args.coeff_min is not None and args.coeff_max is not None:
                coeff_ranges = [(args.coeff_min, args.coeff_max)] * args.nscales
                print(
                    f"Coeff bins: fixed manual range [{args.coeff_min}, {args.coeff_max}] for all scales"
                )
            elif args.coeff_min is None and args.coeff_max is None:
                print(
                    f"Coeff bins: calibrating ranges from first "
                    f"{min(args.coeff_calibration_files, len(file_paths))} files"
                )
                coeff_ranges = estimate_coeff_ranges(
                    file_paths=file_paths,
                    map_key=map_key,
                    nscales=args.nscales,
                    add_noise=add_noise,
                    noise_level=args.noise_level,
                    apply_mask=args.apply_mask,
                    mask_area_sqdeg=args.mask_area_sqdeg,
                    mask_center=mask_center,
                    normalize_energy=args.normalize_energy,
                    lower_pct=args.coeff_lower_pct,
                    upper_pct=args.coeff_upper_pct,
                    sample_size=args.coeff_calibration_files,
                )
                for i, (cmin, cmax) in enumerate(coeff_ranges):
                    print(f"  Scale {i}: [{cmin:.6e}, {cmax:.6e}]")
            else:
                raise ValueError("Provide both --coeff-min and --coeff-max, or neither.")

        process_func = partial(
            process_file,
            bin_number=bin_number,
            metric=metric,
            nscales=args.nscales,
            nbins=args.nbins,
            noise_level=args.noise_level,
            add_noise=add_noise,
            min_snr=args.min_snr,
            max_snr=args.max_snr,
            noise_std=args.noise_std,
            verbose=args.verbose,
            apply_mask=args.apply_mask,
            mask_area_sqdeg=args.mask_area_sqdeg,
            mask_center=mask_center,
            force_overwrite=args.force_overwrite,
            min_snr_coarse=bin_min_snr_coarse,
            max_snr_coarse=bin_max_snr_coarse,
            coeff_ranges=coeff_ranges,
            normalize_energy=args.normalize_energy,
        )

        with mp.Pool(processes=args.num_workers, initializer=seed_worker) as pool:
            results = list(
                tqdm(
                    pool.imap(process_func, file_paths),
                    total=len(file_paths),
                    desc=f"Bin {bin_number}",
                )
            )

        successful = [r for r in results if r is not None]
        done = len([r for r in successful if os.path.exists(r)])
        print(f"Bin {bin_number} complete: {done}/{len(file_paths)} files")


if __name__ == "__main__":
    main()
