#!/usr/bin/env python3
"""
Submit NPE inference jobs for peak counts parameter sweep - PARALLEL VERSION.

This script submits jobs with different combinations of:
- fiducial-type: nobaryons, baryonified
- mask-area-sqdeg: 2001.0, 5001.0, 10001.0, 14001.0, 28001.0, 35001.0
- scales (wavelets): "0,1,2,3" and "1,2,3"

Jobs are distributed across GPUs 0, 1, 2 and run in parallel (one job per GPU at a time).

Usage:
  python scripts/submit_npe_peak_counts_parameter_sweep_parallel.py [--run N] [--rerun] [--bnt]
  
  --run N: Optional run number to append to output filenames (for multiple runs)
  --rerun: Rerun all jobs even if output files already exist
  --bnt: Use BNT-transformed data
"""

import subprocess
import itertools
import time
import argparse
from pathlib import Path
from datetime import datetime

# Fixed parameters
BASE_CMD = "python scripts/run_npe_peak_counts_inference.py"
FIXED_PARAMS = {
    "simulation-type": "nobaryons",
    "bins": "1,2,3,4",
    "noisy": True,
    "noise-level": "0.26",
    "masked": True,
    "train": True,
    "new-normalization": True,
}

# Variable parameters
FIDUCIAL_TYPES = ["nobaryons", "baryonified"]
MASK_AREAS = [2001.0, 5001.0, 10001.0, 14001.0, 28001.0, 35001.0]
SCALE_CONFIGS = ["0,1,2,3", "1,2,3"]  # Wavelet configurations
GPUS = [0, 1, 2]

# Log directory
LOG_DIR = Path("logs/npe_peak_counts_parameter_sweep")

# Samples directory (where outputs are saved)
SAMPLES_DIR = Path("/home/tersenov/software/bar_impact/outputs/samples")

# Run number (set via command line)
RUN_NUMBER = None

# BNT flag (set via command line)
USE_BNT = False

def get_expected_output_filename(fiducial_type, mask_area, scales):
    """Construct the expected output filename for a job."""
    # Construct scale description (scales are 1-indexed in filename)
    scale_indices = [int(s.strip()) for s in scales.split(',')]
    scale_desc = f"scales{''.join([str(s+1) for s in scale_indices])}"
    
    # For BNT mode with "1,2,3" scales, we use per-bin scales so add "_perbin" suffix
    if USE_BNT and scales == "1,2,3":
        scale_desc += "_perbin"
    
    # Construct bin description
    # For BNT mode: use "bntbins1234", for standard mode: use "bins1234"
    bin_spec = "bntbins1234" if USE_BNT else "bins1234"
    
    run_suffix = f"_run{RUN_NUMBER}" if RUN_NUMBER is not None else ""
    bnt_prefix = "bnt_" if USE_BNT else ""
    
    # Format: posterior_samples_[bnt_]pc_{simulation_type}_vs_{fiducial_type}_{bin_spec}_{scale_desc}_noisy_s{noise}_masked_{mask}sqdeg_new_normalization_npe.npy
    filename = (
        f"posterior_samples_{bnt_prefix}pc_nobaryons_vs_{fiducial_type}_{bin_spec}_{scale_desc}_"
        f"noisy_s0.26_masked_{int(mask_area)}sqdeg_new_normalization{run_suffix}_npe.npy"
    )
    return SAMPLES_DIR / filename

def check_output_exists(fiducial_type, mask_area, scales):
    """Check if the output file for this job already exists."""
    output_path = get_expected_output_filename(fiducial_type, mask_area, scales)
    return output_path.exists()

def build_command(fiducial_type, mask_area, scales, gpu):
    """Build the command string for a specific parameter combination."""
    cmd_parts = [BASE_CMD]
    
    # Add fixed parameters
    for key, value in FIXED_PARAMS.items():
        if isinstance(value, bool):
            if value:
                cmd_parts.append(f"--{key}")
        else:
            cmd_parts.append(f"--{key} {value}")
    
    # Add variable parameters
    cmd_parts.append(f"--fiducial-type {fiducial_type}")
    cmd_parts.append(f"--mask-area-sqdeg {mask_area}")
    
    # Handle scale configuration
    if USE_BNT and scales == "1,2,3":
        # For BNT with no 0th scale: bin 1 uses "1,2,3", bins 2,3,4 use "0,1,2,3"
        cmd_parts.append(f'--scales-per-bin "1,2,3;0,1,2,3;0,1,2,3;0,1,2,3"')
    else:
        # Standard case: same scales for all bins
        cmd_parts.append(f"--scales {scales}")
    
    cmd_parts.append(f"--gpu {gpu}")
    
    # Add BNT flag if specified
    if USE_BNT:
        cmd_parts.append("--bnt")
        cmd_parts.append("--bnt-bins 0,1,2,3")
    
    # Add run number if specified
    if RUN_NUMBER is not None:
        cmd_parts.append(f"--run {RUN_NUMBER}")
    
    return " ".join(cmd_parts)

def get_log_filename(fiducial_type, mask_area, scales, gpu):
    """Generate a log filename for this job."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    scales_tag = scales.replace(",", "")
    return LOG_DIR / f"job_{fiducial_type}_mask{int(mask_area)}_scales{scales_tag}_gpu{gpu}_{timestamp}.log"

def submit_job(command, gpu, fiducial_type, mask_area, scales, job_index, total_jobs):
    """Submit a job to run in the background with logging."""
    log_file = get_log_filename(fiducial_type, mask_area, scales, gpu)
    
    print(f"[Job {job_index}/{total_jobs}] GPU {gpu}: fid={fiducial_type}, mask={int(mask_area)}, scales={scales}")
    print(f"  Log: {log_file}")
    
    # Run the command in the background with output redirected to log file
    with open(log_file, 'w') as f:
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True
        )
    
    return process, log_file

def check_processes(active_jobs):
    """Check which processes have completed and return updated active jobs."""
    completed = []
    for gpu, (process, log_file, job_info) in list(active_jobs.items()):
        if process.poll() is not None:  # Process has finished
            completed.append((gpu, process.returncode, log_file, job_info))
            del active_jobs[gpu]
    
    return completed

def main():
    global RUN_NUMBER, LOG_DIR, USE_BNT
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Submit NPE peak counts inference parameter sweep jobs")
    parser.add_argument("--run", type=int, default=None,
                        help="Run number to append to output filenames (for multiple runs)")
    parser.add_argument("--rerun", action="store_true",
                        help="Rerun all jobs even if output files already exist")
    parser.add_argument("--bnt", action="store_true",
                        help="Use BNT-transformed data")
    args = parser.parse_args()
    
    RUN_NUMBER = args.run
    USE_BNT = args.bnt
    
    # Update log directory with run number and BNT if specified
    bnt_suffix = "_bnt" if USE_BNT else ""
    if RUN_NUMBER is not None:
        LOG_DIR = Path(f"logs/npe_peak_counts_parameter_sweep{bnt_suffix}_run{RUN_NUMBER}")
    elif USE_BNT:
        LOG_DIR = Path(f"logs/npe_peak_counts_parameter_sweep{bnt_suffix}")
    
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    # Generate all parameter combinations
    all_combinations = list(itertools.product(
        FIDUCIAL_TYPES,
        MASK_AREAS,
        SCALE_CONFIGS
    ))
    
    # Filter out jobs that already have output files (unless --rerun is specified)
    jobs_to_run = []
    skipped_jobs = []
    if args.rerun:
        jobs_to_run = list(all_combinations)
    else:
        for combo in all_combinations:
            fiducial_type, mask_area, scales = combo
            if check_output_exists(fiducial_type, mask_area, scales):
                skipped_jobs.append(combo)
            else:
                jobs_to_run.append(combo)
    
    total_jobs = len(jobs_to_run)
    total_original = len(all_combinations)
    print("="*80)
    print("NPE Peak Counts Inference Parameter Sweep - PARALLEL EXECUTION")
    if USE_BNT:
        print("MODE: BNT-transformed data")
    if RUN_NUMBER is not None:
        print(f"RUN NUMBER: {RUN_NUMBER}")
    if args.rerun:
        print("RERUN MODE: Running all jobs regardless of existing outputs")
    print("="*80)
    print(f"Total parameter combinations: {total_original}")
    if not args.rerun:
        print(f"  Already completed (skipping): {len(skipped_jobs)}")
    print(f"  Jobs to run: {total_jobs}")
    print(f"  Fiducial types: {FIDUCIAL_TYPES}")
    print(f"  Mask areas: {MASK_AREAS}")
    print(f"  Scale configs: {SCALE_CONFIGS}")
    print(f"  GPUs: {GPUS} (running {len(GPUS)} jobs in parallel)")
    if total_jobs > 0:
        print(f"  Batches: {(total_jobs + len(GPUS) - 1) // len(GPUS)}")
    print(f"  Log directory: {LOG_DIR}")
    print("="*80)
    
    if total_jobs == 0:
        print("\nAll jobs already completed! Nothing to run.")
        return
    
    # Ask for confirmation
    response = input("\nProceed with job submission? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("Aborted.")
        return
    
    print("\nStarting parallel job execution...\n")
    
    # Track active jobs: gpu -> (process, log_file, job_info)
    active_jobs = {}
    job_queue = list(enumerate(jobs_to_run, start=1))
    completed_count = 0
    
    start_time = time.time()
    
    while job_queue or active_jobs:
        # Fill available GPU slots
        while job_queue and len(active_jobs) < len(GPUS):
            job_index, (fiducial_type, mask_area, scales) = job_queue.pop(0)
            
            # Find available GPU
            available_gpus = [g for g in GPUS if g not in active_jobs]
            if not available_gpus:
                break
            
            gpu = available_gpus[0]
            
            # Submit job
            command = build_command(fiducial_type, mask_area, scales, gpu)
            process, log_file = submit_job(
                command, gpu, fiducial_type, mask_area, scales, job_index, total_jobs
            )
            
            job_info = {
                'fiducial_type': fiducial_type,
                'mask_area': mask_area,
                'scales': scales,
                'index': job_index
            }
            active_jobs[gpu] = (process, log_file, job_info)
        
        # Check for completed jobs
        completed = check_processes(active_jobs)
        for gpu, returncode, log_file, job_info in completed:
            completed_count += 1
            status = "✓ SUCCESS" if returncode == 0 else "✗ FAILED"
            print(f"\n[{status}] GPU {gpu}: Job {job_info['index']}/{total_jobs} completed")
            print(f"  fid={job_info['fiducial_type']}, mask={int(job_info['mask_area'])}, scales={job_info['scales']}")
            print(f"  Log: {log_file}")
            if returncode != 0:
                print(f"  Exit code: {returncode}")
            print(f"  Progress: {completed_count}/{total_jobs} jobs completed")
        
        # Small delay to avoid busy waiting
        time.sleep(2)
    
    elapsed_time = time.time() - start_time
    
    print("\n" + "="*80)
    print("All jobs completed!")
    print(f"Total time: {elapsed_time/60:.1f} minutes")
    print(f"Logs saved to: {LOG_DIR}")
    print("="*80)

if __name__ == "__main__":
    main()
