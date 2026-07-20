#!/usr/bin/env python3
"""
Submit NPE inference jobs for parameter sweep - PARALLEL VERSION - FULL SKY.

This script submits jobs with different combinations of:
- fiducial-type: nobaryons, baryonified
- ell-max: 340 to 1020 (step 20)
- rebin-factor: 10, 20

Jobs are distributed across GPUs 0, 1, 2 and run in parallel (one job per GPU at a time).

Usage:
  python scripts/submit_npe_inference_parameter_sweep_parallel_fullsky.py [--run N] [--rerun] [--bnt]
  
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
BASE_CMD = "python scripts/run_npe_inference_auto_cross_ps_master.py"
FIXED_PARAMS = {
    "simulation-type": "nobaryons",
    "bins": "1,2,3,4",
    "lower-cut": "30",
    "noisy": True,
    "noise-level": "0.26",
    "masked": False,  # Full sky - no masking
    "train": True,
}

# Variable parameters for full sky
FIDUCIAL_TYPES = ["nobaryons", "baryonified"]
ELL_MAX_VALUES = list(range(1020, 1021, 20))  # 340, 360, ..., 1000, 1020 (same as masked version)
REBIN_FACTORS = [10, 20]  # Rebinning factors
GPUS = [0, 1]

# Log directory
LOG_DIR = Path("logs/npe_parameter_sweep_fullsky")

# Samples directory (where outputs are saved)
SAMPLES_DIR = Path("/home/tersenov/software/bar_impact/outputs/samples")

# Run number (set via command line)
RUN_NUMBER = None

# BNT flag (set via command line)
USE_BNT = False

def get_expected_output_filename(fiducial_type, ell_max, rebin_factor):
    """Construct the expected output filename for a job."""
    run_suffix = f"_run{RUN_NUMBER}" if RUN_NUMBER is not None else ""
    
    if USE_BNT:
        # BNT samples use 'bntbins1234' instead of 'bins1234' and different prefix
        filename = (
            f"posterior_samples_ps_nobaryons_vs_{fiducial_type}_"
            f"bntbins1234_l30-{ell_max}_r{rebin_factor}_noisy_s0.26_npe{run_suffix}.npy"
        )
    else:
        # Normal full sky samples
        filename = (
            f"posterior_samples_ps_auto_cross_nobaryons_vs_{fiducial_type}_"
            f"bins1234_l30-{ell_max}_r{rebin_factor}_noisy_s0.26_npe{run_suffix}.npy"
        )
    return SAMPLES_DIR / filename

def check_output_exists(fiducial_type, ell_max, rebin_factor):
    """Check if the output file for this job already exists."""
    output_path = get_expected_output_filename(fiducial_type, ell_max, rebin_factor)
    return output_path.exists()

def build_command(fiducial_type, ell_max, rebin_factor, gpu):
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
    cmd_parts.append(f"--lmax {ell_max}")
    
    # Handle upper cut configuration
    if USE_BNT:
        # For BNT: variable ell_max for bin 1, fixed 1024 for bins 2,3,4
        cmd_parts.append(f"--upper-cuts {ell_max},1024,1024,1024")
    else:
        # Standard case: same upper cut for all bins
        cmd_parts.append(f"--upper-cut {ell_max}")
    
    cmd_parts.append(f"--rebin {rebin_factor}")
    cmd_parts.append(f"--gpu {gpu}")
    
    # Add BNT flag if specified
    if USE_BNT:
        cmd_parts.append("--bnt")
        cmd_parts.append("--bnt-bins 0,1,2,3")
    
    # Add run number if specified
    if RUN_NUMBER is not None:
        cmd_parts.append(f"--run {RUN_NUMBER}")
    
    return " ".join(cmd_parts)

def get_log_filename(fiducial_type, ell_max, rebin_factor, gpu):
    """Generate a log filename for this job."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return LOG_DIR / f"job_{fiducial_type}_lmax{ell_max}_r{rebin_factor}_gpu{gpu}_{timestamp}.log"

def submit_job(command, gpu, fiducial_type, ell_max, rebin_factor, job_index, total_jobs):
    """Submit a job to run in the background with logging."""
    log_file = get_log_filename(fiducial_type, ell_max, rebin_factor, gpu)
    
    print(f"[Job {job_index}/{total_jobs}] GPU {gpu}: fid={fiducial_type}, lmax={ell_max}, r={rebin_factor}")
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
    global LOG_DIR, USE_BNT, RUN_NUMBER
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Submit NPE inference parameter sweep jobs for full sky")
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
        LOG_DIR = Path(f"logs/npe_parameter_sweep_fullsky{bnt_suffix}_run{RUN_NUMBER}")
    elif USE_BNT:
        LOG_DIR = Path(f"logs/npe_parameter_sweep_fullsky{bnt_suffix}")
    
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    # Generate all parameter combinations
    all_combinations = list(itertools.product(
        FIDUCIAL_TYPES,
        ELL_MAX_VALUES,
        REBIN_FACTORS
    ))
    
    # Filter out jobs that already have output files (unless --rerun is specified)
    jobs_to_run = []
    skipped_jobs = []
    if args.rerun:
        jobs_to_run = list(all_combinations)
    else:
        for combo in all_combinations:
            fiducial_type, ell_max, rebin_factor = combo
            if check_output_exists(fiducial_type, ell_max, rebin_factor):
                skipped_jobs.append(combo)
            else:
                jobs_to_run.append(combo)
    
    total_jobs = len(jobs_to_run)
    total_original = len(all_combinations)
    print("="*80)
    print("NPE Inference Parameter Sweep - PARALLEL EXECUTION - FULL SKY")
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
    print(f"  Ell-max values: {len(ELL_MAX_VALUES)} values ({ELL_MAX_VALUES[0]} to {ELL_MAX_VALUES[-1]})")
    print(f"  Rebin factors: {REBIN_FACTORS}")
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
            job_index, (fiducial_type, ell_max, rebin_factor) = job_queue.pop(0)
            
            # Find available GPU
            available_gpus = [g for g in GPUS if g not in active_jobs]
            if not available_gpus:
                break
            
            gpu = available_gpus[0]
            
            # Submit job
            command = build_command(fiducial_type, ell_max, rebin_factor, gpu)
            process, log_file = submit_job(
                command, gpu, fiducial_type, ell_max, rebin_factor, job_index, total_jobs
            )
            
            job_info = {
                'fiducial_type': fiducial_type,
                'ell_max': ell_max,
                'rebin_factor': rebin_factor,
                'index': job_index
            }
            active_jobs[gpu] = (process, log_file, job_info)
        
        # Check for completed jobs
        completed = check_processes(active_jobs)
        for gpu, returncode, log_file, job_info in completed:
            completed_count += 1
            status = "✓ SUCCESS" if returncode == 0 else "✗ FAILED"
            print(f"\n[{status}] GPU {gpu}: Job {job_info['index']}/{total_jobs} completed")
            print(f"  fid={job_info['fiducial_type']}, lmax={job_info['ell_max']}, r={job_info['rebin_factor']}")
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
