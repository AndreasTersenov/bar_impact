#!/usr/bin/env python3

"""Compute Fisher forecasts for CosmoGRID power spectra using JAX."""

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
    r"$S_8$",
    r"$w_0$",
    r"$H_0$",
    r"$n_s$",
    r"$\Omega_b$",
]

PARAM_LABEL_MAP = {
    "Omega_m": DEFAULT_LATEX_LABELS[0],
    "S8": DEFAULT_LATEX_LABELS[1],
    "S_8": DEFAULT_LATEX_LABELS[1],
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
        description="Run Fisher forecasts on CosmoGRID power spectra",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data configuration
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/home/tersenov/CosmoGridV1/stage3_forecast",
        help="Base directory for data",
    )
    parser.add_argument(
        "--simulation-type",
        type=str,
        choices=["baryonified", "nobaryons"],
        default="baryonified",
        help="Type of simulation to use for the grid",
    )

    # Analysis configuration
    bin_group = parser.add_mutually_exclusive_group(required=False)
    bin_group.add_argument(
        "--bin",
        type=int,
        default=2,
        help="Which redshift bin to analyze",
    )
    bin_group.add_argument(
        "--bins",
        type=str,
        help="Comma-separated list of redshift bins to analyze for tomographic inference",
    )

    # BNT configuration
    parser.add_argument(
        "--bnt",
        action="store_true",
        help="Use BNT-transformed power spectra",
    )

    bnt_bin_group = parser.add_mutually_exclusive_group(required=False)
    bnt_bin_group.add_argument(
        "--bnt-bin",
        type=int,
        default=3,
        help="Which BNT bin to analyze (0-3, default=3 corresponds to bin4)",
    )
    bnt_bin_group.add_argument(
        "--bnt-bins",
        type=str,
        help="Comma-separated list of BNT bins to analyze for tomographic inference",
    )

    # Power spectrum processing options
    parser.add_argument(
        "--lower-cut",
        type=int,
        default=30,
        help="Lower multipole cut for the power spectrum (l_min)",
    )

    upper_cut_group = parser.add_mutually_exclusive_group(required=False)
    upper_cut_group.add_argument(
        "--upper-cut",
        type=int,
        default=1024,
        help="Upper multipole cut for the power spectrum (l_max)",
    )
    upper_cut_group.add_argument(
        "--upper-cuts",
        type=str,
        help="Comma-separated list of upper multipole cuts for each bin (l_max)",
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

    parser.add_argument(
        "--masked",
        action="store_true",
        help="Use masked power spectra (14300 sq deg disk mask)",
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
        default="/home/tersenov/software/bar_impact/outputs/fisher",
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

    args = parser.parse_args()

    # Match fiducial to simulation type if unspecified
    if args.fiducial_type is None:
        args.fiducial_type = args.simulation_type

    return args


def parse_upper_cuts(args: argparse.Namespace) -> List[int]:
    """Parse upper cuts and validate against number of bins."""
    if args.bnt:
        if args.bnt_bins:
            num_bins = len([int(b.strip()) for b in args.bnt_bins.split(",")])
        else:
            num_bins = 1
    else:
        if args.bins:
            num_bins = len([int(b.strip()) for b in args.bins.split(",")])
        else:
            num_bins = 1

    if args.upper_cuts:
        upper_cuts = [int(cut.strip()) for cut in args.upper_cuts.split(",")]
        if len(upper_cuts) != num_bins:
            raise ValueError(
                f"Number of upper cuts ({len(upper_cuts)}) must match number of bins ({num_bins})"
            )
    else:
        upper_cuts = [args.upper_cut] * num_bins

    return upper_cuts


def construct_paths(args: argparse.Namespace) -> Tuple[str, List[str], List[str], str]:
    """Construct file paths for power spectra based on provided arguments."""
    params_filename = (
        f"cosmo_params{'_baryonified' if args.simulation_type == 'baryonified' else ''}.npy"
    )
    params_path = os.path.join(args.data_dir, "grid", params_filename)

    if args.bnt:
        if args.bnt_bins:
            bin_indices = [int(b.strip()) for b in args.bnt_bins.split(",")]
            bin_desc = f"bntbins{''.join([str(b + 1) for b in bin_indices])}"
        else:
            bin_indices = [args.bnt_bin]
            bin_desc = f"bnt{args.bnt_bin + 1}"
        data_prefix = "all_bnt_cls"
        bin_prefix = "bin"
        bin_suffix_list = [f"{b + 1}" for b in bin_indices]
    else:
        if args.bins:
            bin_indices = [int(b.strip()) for b in args.bins.split(",")]
            bin_desc = f"bins{''.join([str(b) for b in bin_indices])}"
        else:
            bin_indices = [args.bin]
            bin_desc = f"bin{args.bin}"
        data_prefix = "all_cls"
        bin_prefix = "bin"
        bin_suffix_list = bin_indices

    noise_suffix = f"_noisy_s{args.noise_level:.2f}" if args.noisy else ""
    mask_suffix = "_masked_14300sqdeg" if args.masked else ""

    data_paths = []
    fiducial_paths = []

    for i, _ in enumerate(bin_indices):
        bin_spec = f"{bin_prefix}{bin_suffix_list[i]}"
        data_filename = (
            f"{data_prefix}_grid_{args.simulation_type}_{bin_spec}{mask_suffix}{noise_suffix}.npy"
        )
        data_path = os.path.join(args.data_dir, "new_grid", data_filename)
        if not os.path.exists(data_path):
            data_path = os.path.join(args.data_dir, "grid", data_filename)
        data_paths.append(data_path)

        fiducial_filename = (
            f"{data_prefix}_fiducial_{args.fiducial_type}_{bin_spec}{mask_suffix}{noise_suffix}.npy"
        )
        fiducial_path = os.path.join(
            args.data_dir,
            "fiducial",
            "cosmo_fiducial",
            fiducial_filename,
        )
        fiducial_paths.append(fiducial_path)

    return params_path, data_paths, fiducial_paths, bin_desc


def compute_ps_desc(lower_cut: int, upper_cuts: Sequence[int], rebin_factor: int) -> str:
    if len(set(upper_cuts)) == 1:
        ps_desc = f"l{lower_cut}-{upper_cuts[0]}"
    else:
        ps_desc = f"l{lower_cut}-{'-'.join(map(str, upper_cuts))}"
    if rebin_factor > 1:
        ps_desc += f"_r{rebin_factor}"
    return ps_desc


def rebin_array(data: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return data
    length = data.shape[-1]
    trimmed_length = (length // factor) * factor
    if trimmed_length != length:
        data = data[..., :trimmed_length]
    new_shape = data.shape[:-1] + (trimmed_length // factor, factor)
    return data.reshape(new_shape).mean(axis=-1)


def process_simulation_data(
    cls_full_bins: Sequence[np.ndarray],
    lower_cut: int,
    upper_cuts: Sequence[int],
    rebin_factor: int,
) -> np.ndarray:
    processed = []
    for i, cls_full in enumerate(cls_full_bins):
        if cls_full.ndim != 2:
            raise ValueError(
                f"Expected simulation grid data with shape (n_sims, n_ells); got {cls_full.shape}"
            )
        cls_cut = cls_full[:, lower_cut:upper_cuts[i]]
        cls_proc = rebin_array(cls_cut, rebin_factor)
        processed.append(cls_proc)
    return np.concatenate(processed, axis=1)


def process_fiducial_samples(
    fid_full_bins: Sequence[np.ndarray],
    lower_cut: int,
    upper_cuts: Sequence[int],
    rebin_factor: int,
) -> np.ndarray:
    processed = []
    for i, fid_full in enumerate(fid_full_bins):
        if fid_full.ndim != 2:
            raise ValueError(
                f"Expected fiducial data with shape (n_realizations, n_ells); got {fid_full.shape}"
            )
        fid_cut = fid_full[:, lower_cut:upper_cuts[i]]
        fid_proc = rebin_array(fid_cut, rebin_factor)
        processed.append(fid_proc)
    return np.concatenate(processed, axis=1)


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

    upper_cuts = parse_upper_cuts(args)
    params_path, data_paths, fiducial_paths, bin_spec = construct_paths(args)
    ps_desc = compute_ps_desc(args.lower_cut, upper_cuts, args.rebin)

    print(f"Using parameters file: {params_path}")
    print(f"Using datavector files: {data_paths}")
    print(f"Using fiducial files: {fiducial_paths}")

    if not os.path.exists(params_path):
        raise FileNotFoundError(f"Parameters file not found: {params_path}")

    for path in data_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Simulation datavector file not found: {path}")

    for path in fiducial_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Fiducial datavector file not found: {path}")

    params = np.load(params_path, allow_pickle=True)
    cls_full_bins = [np.load(path, allow_pickle=True) for path in data_paths]
    fid_full_bins = [np.load(path, allow_pickle=True) for path in fiducial_paths]
    print(f"Loaded grid parameters with shape {params.shape}")

    sims_matrix = process_simulation_data(
        cls_full_bins,
        args.lower_cut,
        upper_cuts,
        args.rebin,
    )
    print(f"Processed simulation grid to shape {sims_matrix.shape}")

    fid_samples_matrix = process_fiducial_samples(
        fid_full_bins,
        args.lower_cut,
        upper_cuts,
        args.rebin,
    )
    print(
        "Processed fiducial realizations to shape"
        f" {fid_samples_matrix.shape} (expected ~200 realizations)"
    )

    fid_mean_vector = fid_samples_matrix.mean(axis=0)

    params_jax = jnp.array(params)
    sims_jax = jnp.array(sims_matrix)
    fid_samples_jax = jnp.array(fid_samples_matrix)
    fid_mean_jax = jnp.array(fid_mean_vector)

    fiducial_params_values = np.array(
        [float(x.strip()) for x in args.fiducial_params.split(",")],
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

    param_names = [name.strip() for name in args.param_names.split(",")]
    if len(param_names) != param_sigmas.shape[0]:
        if len(param_names) == 1:
            param_names = [f"theta_{i}" for i in range(param_sigmas.shape[0])]
        else:
            raise ValueError("Parameter names must match the number of parameters")

    plot_labels = [
        PARAM_LABEL_MAP.get(name, rf"${name}$") for name in param_names
    ]

    correlation = param_covariance / jnp.sqrt(
        jnp.outer(param_sigmas, param_sigmas)
    )

    datavector_desc = f"{args.simulation_type}_{bin_spec}_{ps_desc}"
    if args.masked:
        datavector_desc += "_masked_14300sqdeg"
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

    # Generate GetDist Fisher contour plot
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

    plot_filename = os.path.join(args.output_dir, f"{base_filename}_fisher_triangle.pdf")
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
        fh.write(f"  Upper cuts:      {upper_cuts}\n")
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
