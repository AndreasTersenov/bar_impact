#!/usr/bin/env python3
"""Compute Fisher forecasts for CosmoGRID auto and cross power spectra using JAX."""

import os
import argparse
from typing import List, Optional, Sequence, Tuple

import numpy as np
import jax

# Enable double precision for improved numerical stability
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from getdist import plots, MCSamples


DEFAULT_LATEX_LABELS = [
    r"$\Omega_{m}$",
    r"$\sigma_8$",
    r"$w_0$",
    r"$H_0$",
    r"$n_s$",
    r"$\Omega_b$",
]

PARAM_LABEL_MAP = {
    "Omega_m": DEFAULT_LATEX_LABELS[0],
    "S8": DEFAULT_LATEX_LABELS[1],
    r"\sigma_8": DEFAULT_LATEX_LABELS[1],
    "w0": DEFAULT_LATEX_LABELS[2],
    "w_0": DEFAULT_LATEX_LABELS[2],
    "H0": DEFAULT_LATEX_LABELS[3],
    "H_0": DEFAULT_LATEX_LABELS[3],
    "n_s": DEFAULT_LATEX_LABELS[4],
    "ns": DEFAULT_LATEX_LABELS[4],
    "Omega_b": DEFAULT_LATEX_LABELS[5],
    "omega_b": DEFAULT_LATEX_LABELS[5],
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Fisher forecasts on CosmoGRID auto + cross power spectra",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data configuration
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/lustre/fsn1/projects/rech/prk/ulx34io/cosmogrid_products/stage3_forecast",
        help="Base directory for data",
    )
    parser.add_argument(
        "--simulation-type",
        type=str,
        choices=["baryonified", "nobaryons"],
        default="baryonified",
        help="Type of simulation to use for the grid",
    )

    # Analysis configuration (auto + cross requires multiple bins)
    parser.add_argument(
        "--bins",
        type=str,
        default="1,2,3,4",
        help="Comma-separated list of redshift bins to analyze",
    )

    # BNT configuration
    parser.add_argument(
        "--bnt",
        action="store_true",
        help="Use BNT-transformed power spectra",
    )
    parser.add_argument(
        "--bnt-bins",
        type=str,
        default="0,1,2,3",
        help="Comma-separated list of BNT bins to analyze",
    )

    # Power spectrum processing options
    parser.add_argument(
        "--lower-cut",
        type=int,
        default=30,
        help="Lower multipole cut for the power spectrum (l_min)",
    )
    parser.add_argument(
        "--upper-cut",
        type=int,
        default=1024,
        help="Upper multipole cut for the power spectrum (l_max)",
    )
    parser.add_argument(
        "--rebin",
        type=int,
        default=1,
        help="Rebinning factor for the power spectrum (1 = no rebinning)",
    )

    parser.add_argument(
        "--noisy",
        action="store_true",
        help="Use noisy datavectors",
    )
    parser.add_argument(
        "--noise-level",
        type=float,
        default=0.26,
        help="Noise level when --noisy is set",
    )

    # Cross power spectra configuration
    parser.add_argument(
        "--cross-data-dir",
        type=str,
        help="Directory containing aggregated cross power spectra files. Defaults to data-dir if unspecified.",
    )
    parser.add_argument(
        "--cross-pairs",
        type=str,
        default=None,
        help="Semicolon-separated list of cross power spectrum pairs to include, e.g., '1,3;1,4'.",
    )
    parser.add_argument(
        "--auto-only",
        action="store_true",
        help="Use only auto power spectra",
    )
    parser.add_argument(
        "--cross-only",
        action="store_true",
        help="Use only cross power spectra",
    )

    # Fiducial configuration
    parser.add_argument(
        "--fiducial-type",
        type=str,
        choices=["baryonified", "nobaryons"],
        default=None,
        help="Type of fiducial (defaults to simulation type if not provided)",
    )
    parser.add_argument(
        "--fiducial-params",
        type=str,
        default="0.26,0.84,-1.0,67.36,0.9649,0.0493",
        help="Comma-separated fiducial cosmological parameters used to expand the Fisher matrix",
    )

    parser.add_argument(
        "--param-names",
        type=str,
        default="Omega_m,S8,w0,H0,n_s,Omega_b",
        help="Comma-separated parameter names used for reporting uncertainties",
    )
    parser.add_argument(
        "--plot-samples",
        type=int,
        default=10000,
        help="Number of Gaussian samples drawn from the Fisher covariance for plotting",
    )
    parser.add_argument(
        "--save-samples",
        action="store_true",
        help="Save the Gaussian samples drawn from the Fisher covariance to disk",
    )
    parser.add_argument(
        "--samples-dir",
        type=str,
        default=None,
        help="Directory to store saved Gaussian samples (defaults to output-dir)",
    )

    # Output configuration
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/lustre/fsn1/projects/rech/prk/ulx34io/bar_impact/outputs/fisher",
        help="Directory to save Fisher matrix outputs",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="GPU index to use (set to 'cpu' to force CPU execution)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=1,
        help="Random seed used for drawing Gaussian samples when generating plots",
    )
    parser.add_argument(
        "--cov-epsilon",
        type=float,
        default=1e-6,
        help="Relative diagonal jitter added to the covariance matrix before inversion",
    )
    parser.add_argument(
        "--fisher-epsilon",
        type=float,
        default=1e-8,
        help="Relative diagonal jitter added to the Fisher matrix before inversion",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed information about data loading and processing",
    )

    args = parser.parse_args()

    if args.fiducial_type is None:
        args.fiducial_type = args.simulation_type

    if args.cross_data_dir is None:
        args.cross_data_dir = args.data_dir

    if args.auto_only and args.cross_only:
        parser.error("Cannot specify both --auto-only and --cross-only")

    return args


def parse_cross_pairs(cross_pairs_str: Optional[str]) -> Optional[List[Tuple[int, int]]]:
    if cross_pairs_str is None:
        return None
    pairs: List[Tuple[int, int]] = []
    for pair_str in cross_pairs_str.split(';'):
        i, j = pair_str.split(',')
        pairs.append((int(i.strip()), int(j.strip())))
    return pairs


def get_cross_indices_for_pairs(
    bin_indices: Sequence[int], cross_pairs: Optional[Sequence[Tuple[int, int]]]
) -> Optional[List[int]]:
    if cross_pairs is None:
        return None

    all_cross_pairs: List[Tuple[int, int]] = []
    for i in range(len(bin_indices)):
        for j in range(i + 1, len(bin_indices)):
            all_cross_pairs.append((bin_indices[i], bin_indices[j]))

    selected_indices: List[int] = []
    for pair in cross_pairs:
        try:
            idx = all_cross_pairs.index(pair)
            selected_indices.append(idx)
        except ValueError:
            print(
                f"Warning: Cross pair {pair} not found in available pairs {all_cross_pairs}"
            )

    if not selected_indices:
        raise ValueError(
            "No cross pairs matched the requested selection. Check --cross-pairs and --bins."
        )
    return selected_indices


def resolve_file(base_dir: str, subdirs: Sequence[str], filename: str) -> str:
    root, ext = os.path.splitext(filename)
    candidate_names = []
    if not root.endswith("_new"):
        candidate_names.append(f"{root}_new{ext}")
    candidate_names.append(filename)

    for subdir in subdirs:
        for candidate in candidate_names:
            path = os.path.join(base_dir, subdir, candidate) if subdir else os.path.join(base_dir, candidate)
            if os.path.exists(path):
                return path
    raise FileNotFoundError(
        f"None of the files {[os.path.join(subdir, name) for subdir in subdirs for name in candidate_names]} exist in {base_dir}"
    )


def construct_auto_paths(args: argparse.Namespace) -> Tuple[str, List[str], List[str], str]:
    params_filename = (
        f"cosmo_params{'_baryonified' if args.simulation_type == 'baryonified' else ''}.npy"
    )
    params_path = resolve_file(args.data_dir, ["grid"], params_filename)

    if args.bnt:
        bin_indices = [int(b.strip()) for b in args.bnt_bins.split(',')]
        bin_desc = f"bins{''.join([str(b + 1) for b in bin_indices])}"
        data_prefix = "all_bnt_cls"
        bin_prefix = "bin"
        bin_suffix_list = [f"{b + 1}" for b in bin_indices]
    else:
        bin_indices = [int(b.strip()) for b in args.bins.split(',')]
        bin_desc = f"bins{''.join([str(b) for b in bin_indices])}"
        data_prefix = "all_cls"
        bin_prefix = "bin"
        bin_suffix_list = bin_indices

    noise_suffix = f"_noisy_s{args.noise_level:.2f}" if args.noisy else ""

    auto_data_paths = []
    auto_fiducial_paths = []

    for idx, bin_idx in enumerate(bin_indices):
        bin_spec = f"{bin_prefix}{bin_suffix_list[idx]}"
        data_filename = f"{data_prefix}_grid_{args.simulation_type}_{bin_spec}{noise_suffix}.npy"
        data_path = resolve_file(args.data_dir, ["new_grid", "grid"], data_filename)
        auto_data_paths.append(data_path)

        fid_filename = f"{data_prefix}_fiducial_{args.fiducial_type}_{bin_spec}{noise_suffix}.npy"
        fid_path = resolve_file(
            args.data_dir,
            [os.path.join("fiducial", "cosmo_fiducial")],
            fid_filename,
        )
        auto_fiducial_paths.append(fid_path)

    return params_path, auto_data_paths, auto_fiducial_paths, bin_desc


def construct_cross_paths(args: argparse.Namespace, bin_desc: str) -> Tuple[str, str]:
    noise_suffix = f"_noisy_s{args.noise_level:.2f}" if args.noisy else ""

    if args.bnt:
        data_filename = f"all_bnt_cross_cls_grid_{args.simulation_type}_{bin_desc}{noise_suffix}.npy"
        fid_filename = f"all_bnt_cross_cls_fiducial_{args.fiducial_type}_{bin_desc}{noise_suffix}.npy"
    else:
        data_filename = f"all_cross_cls_grid_{args.simulation_type}_{bin_desc}{noise_suffix}.npy"
        fid_filename = f"all_cross_cls_fiducial_{args.fiducial_type}_{bin_desc}{noise_suffix}.npy"

    data_path = resolve_file(args.cross_data_dir, ["new_grid", "grid"], data_filename)
    fid_path = resolve_file(args.cross_data_dir, ["fiducial/cosmo_fiducial"], fid_filename)

    return data_path, fid_path


def compute_ps_desc(lower_cut: int, upper_cut: int, rebin_factor: int) -> str:
    desc = f"l{lower_cut}-{upper_cut}"
    if rebin_factor > 1:
        desc += f"_r{rebin_factor}"
    return desc


def rebin_array(data: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return data
    length = data.shape[-1]
    trimmed_length = (length // factor) * factor
    if trimmed_length != length:
        data = data[..., :trimmed_length]
    new_shape = data.shape[:-1] + (trimmed_length // factor, factor)
    return data.reshape(new_shape).mean(axis=-1)


def load_auto_simulations(auto_data_paths: Sequence[str], verbose: bool = False) -> List[np.ndarray]:
    arrays = []
    for path in auto_data_paths:
        arr = np.load(path, allow_pickle=True)
        if arr.ndim != 2:
            raise ValueError(
                f"Expected simulation grid data with shape (n_sims, n_ells); got {arr.shape} for {path}"
            )
        if verbose:
            print(f"Loaded auto simulations from {path}, shape: {arr.shape}")
        arrays.append(arr)
    return arrays


def load_auto_fiducial_samples(auto_fid_paths: Sequence[str], verbose: bool = False) -> List[np.ndarray]:
    arrays = []
    for path in auto_fid_paths:
        arr = np.load(path, allow_pickle=True)
        if arr.ndim != 2:
            raise ValueError(
                f"Expected fiducial data with shape (n_realizations, n_ells); got {arr.shape} for {path}"
            )
        if verbose:
            print(f"Loaded auto fiducial samples from {path}, shape: {arr.shape}")
        arrays.append(arr)
    return arrays


def process_auto_simulation_data(
    auto_arrays: Sequence[np.ndarray], lower_cut: int, upper_cut: int, rebin_factor: int
) -> np.ndarray:
    processed = []
    for arr in auto_arrays:
        cls_cut = arr[:, lower_cut:upper_cut]
        cls_proc = rebin_array(cls_cut, rebin_factor)
        processed.append(cls_proc)
    return np.concatenate(processed, axis=1)


def process_auto_fiducial_samples(
    auto_fid_arrays: Sequence[np.ndarray], lower_cut: int, upper_cut: int, rebin_factor: int
) -> np.ndarray:
    processed = []
    for arr in auto_fid_arrays:
        fid_cut = arr[:, lower_cut:upper_cut]
        fid_proc = rebin_array(fid_cut, rebin_factor)
        processed.append(fid_proc)
    return np.concatenate(processed, axis=1)


def process_cross_simulation_data(
    cross_array: np.ndarray,
    lower_cut: int,
    upper_cut: int,
    rebin_factor: int,
    cross_indices: Optional[Sequence[int]] = None,
    verbose: bool = False,
) -> np.ndarray:
    if cross_array.ndim != 2:
        raise ValueError(
            f"Expected cross simulation data with shape (n_sims, n_columns); got {cross_array.shape}"
        )
    cross_cut = cross_array[:, lower_cut:upper_cut]
    n_multipoles = upper_cut - lower_cut

    if cross_indices is not None:
        total_pairs = cross_cut.shape[1] // n_multipoles if n_multipoles else 0
        if verbose:
            print(
                f"Selecting cross indices {cross_indices} out of {total_pairs} available pairs; "
                f"each pair has {n_multipoles} multipoles"
            )
        selected = []
        for idx in cross_indices:
            start = idx * n_multipoles
            end = start + n_multipoles
            selected.append(cross_cut[:, start:end])
        cross_cut = np.concatenate(selected, axis=1)

    cross_proc = rebin_array(cross_cut, rebin_factor)
    return cross_proc


def process_cross_fiducial_samples(
    cross_fid_array: np.ndarray,
    lower_cut: int,
    upper_cut: int,
    rebin_factor: int,
    cross_indices: Optional[Sequence[int]] = None,
    verbose: bool = False,
) -> np.ndarray:
    if cross_fid_array.ndim != 2:
        raise ValueError(
            f"Expected cross fiducial data with shape (n_realizations, n_columns); got {cross_fid_array.shape}"
        )
    cross_cut = cross_fid_array[:, lower_cut:upper_cut]
    n_multipoles = upper_cut - lower_cut

    if cross_indices is not None:
        total_pairs = cross_cut.shape[1] // n_multipoles if n_multipoles else 0
        if verbose:
            print(
                f"Selecting fiducial cross indices {cross_indices} out of {total_pairs} available pairs"
            )
        selected = []
        for idx in cross_indices:
            start = idx * n_multipoles
            end = start + n_multipoles
            selected.append(cross_cut[:, start:end])
        cross_cut = np.concatenate(selected, axis=1)

    cross_proc = rebin_array(cross_cut, rebin_factor)
    return cross_proc


def covariance_from_samples(samples: jnp.ndarray) -> jnp.ndarray:
    centered = samples - jnp.mean(samples, axis=0, keepdims=True)
    denom = samples.shape[0] - 1
    return centered.T @ centered / denom


def compute_jacobian(
    params: jnp.ndarray,
    data_vectors: jnp.ndarray,
    reference_params: jnp.ndarray,
    reference_vector: jnp.ndarray,
) -> jnp.ndarray:
    X = params - reference_params
    Y = data_vectors - reference_vector
    solution, residuals, rank, singular_vals = jnp.linalg.lstsq(X, Y, rcond=None)
    if rank < solution.shape[0]:
        print(
            "Warning: Parameter design matrix is rank-deficient (rank = %d < %d)."
            % (rank, solution.shape[0])
        )
    if residuals.size:
        rss = residuals.sum()
        print(f"Residual sum of squares from linear fit: {rss:.3e}")
    return solution


def add_relative_jitter(matrix: jnp.ndarray, rel_epsilon: float) -> jnp.ndarray:
    if rel_epsilon <= 0:
        return matrix
    diag_mean = jnp.mean(jnp.diag(matrix))
    jitter = rel_epsilon * diag_mean
    return matrix + jitter * jnp.eye(matrix.shape[0], dtype=matrix.dtype)


def main() -> None:
    args = parse_arguments()

    if args.gpu != "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    cross_pairs = parse_cross_pairs(args.cross_pairs)

    params_path, auto_data_paths, auto_fid_paths, bin_desc = construct_auto_paths(args)

    if args.bnt:
        bin_indices = [int(b.strip()) + 1 for b in args.bnt_bins.split(',')]
    else:
        bin_indices = [int(b.strip()) for b in args.bins.split(',')]

    cross_indices = get_cross_indices_for_pairs(bin_indices, cross_pairs) if (cross_pairs and not args.auto_only) else None

    cross_data_path, cross_fid_path = construct_cross_paths(args, bin_desc)

    print(f"Using parameters file: {params_path}")
    print(f"Using auto datavector files: {auto_data_paths}")
    print(f"Using cross datavector file: {cross_data_path}")
    print(f"Using auto fiducial files: {auto_fid_paths}")
    print(f"Using cross fiducial file: {cross_fid_path}")
    if cross_pairs:
        print(f"Requested cross pairs: {cross_pairs}")

    params = np.load(params_path, allow_pickle=True)
    auto_arrays = load_auto_simulations(auto_data_paths, verbose=args.verbose)
    auto_fid_arrays = load_auto_fiducial_samples(auto_fid_paths, verbose=args.verbose)
    cross_array = np.load(cross_data_path, allow_pickle=True)
    cross_fid_array = np.load(cross_fid_path, allow_pickle=True)

    data_parts = []
    fid_parts = []

    if not args.cross_only:
        auto_data_matrix = process_auto_simulation_data(
            auto_arrays, args.lower_cut, args.upper_cut, args.rebin
        )
        auto_fid_matrix = process_auto_fiducial_samples(
            auto_fid_arrays, args.lower_cut, args.upper_cut, args.rebin
        )
        data_parts.append(auto_data_matrix)
        fid_parts.append(auto_fid_matrix)
        print(f"Auto simulation matrix shape: {auto_data_matrix.shape}")
        print(f"Auto fiducial samples shape: {auto_fid_matrix.shape}")

    if not args.auto_only:
        cross_data_matrix = process_cross_simulation_data(
            cross_array,
            args.lower_cut,
            args.upper_cut,
            args.rebin,
            cross_indices,
            verbose=args.verbose,
        )
        cross_fid_matrix = process_cross_fiducial_samples(
            cross_fid_array,
            args.lower_cut,
            args.upper_cut,
            args.rebin,
            cross_indices,
            verbose=args.verbose,
        )
        data_parts.append(cross_data_matrix)
        fid_parts.append(cross_fid_matrix)
        print(f"Cross simulation matrix shape: {cross_data_matrix.shape}")
        print(f"Cross fiducial samples shape: {cross_fid_matrix.shape}")

    if not data_parts:
        raise ValueError("No data selected: enable auto, cross, or both.")

    combined_data_matrix = np.concatenate(data_parts, axis=1)
    combined_fid_matrix = np.concatenate(fid_parts, axis=1)

    print(f"Combined simulation matrix shape: {combined_data_matrix.shape}")
    print(f"Combined fiducial samples shape: {combined_fid_matrix.shape}")

    fid_mean_vector = combined_fid_matrix.mean(axis=0)

    params_jax = jnp.array(params)
    sims_jax = jnp.array(combined_data_matrix)
    fid_samples_jax = jnp.array(combined_fid_matrix)
    fid_mean_jax = jnp.array(fid_mean_vector)

    fiducial_params_values = np.array(
        [float(x.strip()) for x in args.fiducial_params.split(',')],
        dtype=np.float64,
    )
    fiducial_params_jax = jnp.array(fiducial_params_values)

    if fiducial_params_jax.shape[-1] != params_jax.shape[1]:
        raise ValueError(
            "Number of fiducial parameters does not match grid parameter dimension"
            f" ({fiducial_params_jax.shape[-1]} vs {params_jax.shape[1]})"
        )

    jacobian = compute_jacobian(
        params_jax,
        sims_jax,
        fiducial_params_jax,
        fid_mean_jax,
    )

    covariance_matrix = covariance_from_samples(fid_samples_jax)
    covariance_matrix = add_relative_jitter(covariance_matrix, args.cov_epsilon)

    inv_covariance = jnp.linalg.inv(covariance_matrix)

    fisher_matrix = jacobian @ inv_covariance @ jacobian.T
    fisher_matrix = add_relative_jitter(fisher_matrix, args.fisher_epsilon)

    param_covariance = jnp.linalg.inv(fisher_matrix)
    param_sigmas = jnp.sqrt(jnp.diag(param_covariance))

    param_names = [name.strip() for name in args.param_names.split(',')]
    if len(param_names) != param_sigmas.shape[0]:
        if len(param_names) == 1:
            param_names = [f"theta_{i}" for i in range(param_sigmas.shape[0])]
        else:
            raise ValueError("Parameter names must match the number of parameters")

    plot_labels = [PARAM_LABEL_MAP.get(name, rf"${name}$") for name in param_names]

    correlation = param_covariance / jnp.sqrt(
        jnp.outer(param_sigmas, param_sigmas)
    )

    ps_desc = compute_ps_desc(args.lower_cut, args.upper_cut, args.rebin)

    if args.auto_only:
        spectra_type = "auto"
    elif args.cross_only:
        spectra_type = "cross"
    else:
        spectra_type = "auto_cross"

    if cross_pairs and not args.auto_only:
        cross_pairs_str = "_".join([f"{i}-{j}" for i, j in cross_pairs])
        if args.cross_only:
            spectra_type = f"cross_{cross_pairs_str}"
        else:
            spectra_type = f"auto_cross_{cross_pairs_str}"

    if args.bnt:
        datavector_desc = f"{args.simulation_type}_bnt_{bin_desc}_{ps_desc}_{spectra_type}"
    else:
        datavector_desc = f"{args.simulation_type}_{bin_desc}_{ps_desc}_{spectra_type}"

    if args.noisy:
        datavector_desc += f"_noisy_s{args.noise_level:.2f}"

    base_filename = (
        f"fisher_ps_{args.simulation_type}_vs_{args.fiducial_type}_{datavector_desc}"
    )

    os.makedirs(args.output_dir, exist_ok=True)

    samples_dir = (
        os.path.abspath(args.samples_dir)
        if args.samples_dir is not None
        else args.output_dir
    )
    if args.save_samples:
        os.makedirs(samples_dir, exist_ok=True)

    outputs = {
        f"{base_filename}_jacobian.npy": np.array(jacobian),
        f"{base_filename}_data_cov.npy": np.array(covariance_matrix),
        f"{base_filename}_fisher.npy": np.array(fisher_matrix),
        f"{base_filename}_param_cov.npy": np.array(param_covariance),
        f"{base_filename}_correlation.npy": np.array(correlation),
    }

    for filename, array in outputs.items():
        path = os.path.join(args.output_dir, filename)
        np.save(path, array)
        print(f"Saved {array.shape} array to {path}")

    rng = np.random.default_rng(seed=args.random_seed)
    gaussian_samples = rng.multivariate_normal(
        mean=fiducial_params_values,
        cov=np.array(param_covariance),
        size=args.plot_samples,
    )

    fisher_samples = MCSamples(
        samples=gaussian_samples,
        names=plot_labels,
        label=f"Fisher ({datavector_desc})",
        settings={"ignore_rows": 0.0},
    )

    samples_filename: Optional[str] = None
    if args.save_samples:
        samples_filename = f"{base_filename}_gaussian_samples.npy"
        samples_path = os.path.join(samples_dir, samples_filename)
        np.save(samples_path, gaussian_samples)
        print(f"Saved Gaussian samples to {samples_path}")

    subplotter = plots.get_subplot_plotter()
    subplotter.settings.figure_legend_frame = False
    subplotter.settings.alpha_filled_add = 0.4

    markers = {label: val for label, val in zip(plot_labels, fiducial_params_values)}

    subplotter.triangle_plot(
        [fisher_samples],
        filled=True,
        line_args=[{"color": "crimson"}],
        contour_colors=["crimson"],
        markers=markers,
    )

    plot_filename = os.path.join(
        args.output_dir, f"{base_filename}_fisher_triangle.pdf"
    )
    subplotter.export(plot_filename)
    print(f"Saved Fisher GetDist triangle plot to {plot_filename}")

    summary_path = os.path.join(args.output_dir, f"{base_filename}_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as fh:
        fh.write("Fisher forecast summary\n")
        fh.write("=======================\n\n")
        fh.write("Configuration:\n")
        fh.write(f"  Simulation type: {args.simulation_type}\n")
        fh.write(f"  Fiducial type:   {args.fiducial_type}\n")
        fh.write(f"  Datavector:      {datavector_desc}\n")
        fh.write(f"  BNT:             {args.bnt}\n")
        fh.write(f"  Rebin factor:    {args.rebin}\n")
        fh.write(f"  Lower cut:       {args.lower_cut}\n")
        fh.write(f"  Upper cut:       {args.upper_cut}\n")
        if cross_pairs:
            fh.write(f"  Cross pairs:     {cross_pairs}\n")
        fh.write(f"  Auto only:       {args.auto_only}\n")
        fh.write(f"  Cross only:      {args.cross_only}\n")
        if args.save_samples and samples_filename is not None:
            fh.write(f"  Samples file:    {samples_filename}\n")
        fh.write("\n")

        fh.write("1-sigma parameter uncertainties:\n")
        for name, sigma in zip(param_names, np.array(param_sigmas)):
            fh.write(f"  {name:>10s}: {sigma:.4e}\n")

        fh.write("\nCorrelation matrix:\n")
        corr_np = np.array(correlation)
        header = "        " + " ".join(f"{name:>10s}" for name in param_names)
        fh.write(header + "\n")
        for name, row in zip(param_names, corr_np):
            row_values = " ".join(f"{val:>10.3f}" for val in row)
            fh.write(f"{name:>8s} {row_values}\n")

    print("\nFisher forecast results:")
    for name, sigma in zip(param_names, np.array(param_sigmas)):
        print(f"  {name:>10s}: σ = {sigma:.4e}")

    print(f"Saved summary to {summary_path}")
    print("Done!")


if __name__ == "__main__":
    main()
