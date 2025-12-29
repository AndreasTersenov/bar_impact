#!/usr/bin/env python3
"""
Submit NPE inference jobs for parameter sweep - PARALLEL VERSION.

This script submits jobs with different combinations of:
- fiducial-type: nobaryons, baryonified
- upper-cut: 520 to 1020 (step 20)
- mask-area-sqdeg: 2000.0, 5000.0, 10000.0, 14000.0, 28000.0

Jobs are distributed across GPUs 0, 1, 2 and run in parallel (one job per GPU at a time).

Usage:
  python scripts/submit_npe_inference_parameter_sweep_parallel.py [--run N]
  
  --run N: Optional run number to append to output filenames (for multiple runs)
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
    "lower-cut": "100",
    "noisy": True,
    "noise-level": "0.26",
    "masked": True,
    "apodization-scale-deg": "2.0",
    "train": True,
    "rebin": "10",
}

# Variable parameters
FIDUCIAL_TYPES = ["nobaryons", "baryonified"]
UPPER_CUTS = list(range(520, 1021, 20))  # 520, 540, ..., 1000, 1020
MASK_AREAS = [2000.0, 5000.0, 10000.0, 14000.0, 28000.0]
GPUS = [0, 1, 2]

# Log directory
LOG_DIR = Path("logs/npe_parameter_sweep")

# Run number (set via command line)
RUN_NUMBER = None

def build_command(fiducial_type, upper_cut, mask_area, gpu):
    """Build the command string for a specific parameter combination."""
    cmd_parts = [BASE_CMD]
    
    # Add fixed parameters
    for key, value in FIXED_PARAMS.items():
        if isinstance(value, bool):
            if value:
                cmd_parts.append(f"--{key}")
        else:
            cmd_parts.append(f"--{key} {value}")
    
    # Add lmax based on mask_area: 1530 for 14000.0, 1535 for others
    lmax = "1530" if mask_area == 14000.0 else "1535"
    cmd_parts.append(f"--lmax {lmax}")
    
    # Add variable parameters
    cmd_parts.append(f"--fiducial-type {fiducial_type}")
    cmd_parts.append(f"--upper-cut {upper_cut}")
    cmd_parts.append(f"--mask-area-sqdeg {mask_area}")
    cmd_parts.append(f"--gpu {gpu}")
    
    # Add run number if specified
    if RUN_NUMBER is not None:
        cmd_parts.append(f"--run {RUN_NUMBER}")
    
    return " ".join(cmd_parts)

def get_log_filename(fiducial_type, upper_cut, mask_area, gpu):
    """Generate a log filename for this job."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return LOG_DIR / f"job_{fiducial_type}_ucut{upper_cut}_mask{int(mask_area)}_gpu{gpu}_{timestamp}.log"

def submit_job(command, gpu, fiducial_type, upper_cut, mask_area, job_index, total_jobs):
    """Submit a job to run in the background with logging."""
    log_file = get_log_filename(fiducial_type, upper_cut, mask_area, gpu)
    
    print(f"[Job {job_index}/{total_jobs}] GPU {gpu}: fid={fiducial_type}, ucut={upper_cut}, mask={int(mask_area)}")
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
    global RUN_NUMBER, LOG_DIR
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Submit NPE inference parameter sweep jobs")
    parser.add_argument("--run", type=int, default=None,
                        help="Run number to append to output filenames (for multiple runs)")
    args = parser.parse_args()
    
    RUN_NUMBER = args.run
    
    # Update log directory with run number if specified
    if RUN_NUMBER is not None:
        LOG_DIR = Path(f"logs/npe_parameter_sweep_run{RUN_NUMBER}")
    
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    # Generate all parameter combinations
    all_combinations = list(itertools.product(
        FIDUCIAL_TYPES,
        UPPER_CUTS,
        MASK_AREAS
    ))
    
    total_jobs = len(all_combinations)
    print("="*80)
    print("NPE Inference Parameter Sweep - PARALLEL EXECUTION")
    if RUN_NUMBER is not None:
        print(f"RUN NUMBER: {RUN_NUMBER}")
    print("="*80)
    print(f"Total jobs to submit: {total_jobs}")
    print(f"  Fiducial types: {FIDUCIAL_TYPES}")
    print(f"  Upper cuts: {len(UPPER_CUTS)} values ({UPPER_CUTS[0]} to {UPPER_CUTS[-1]})")
    print(f"  Mask areas: {MASK_AREAS}")
    print(f"  GPUs: {GPUS} (running {len(GPUS)} jobs in parallel)")
    print(f"  Batches: {(total_jobs + len(GPUS) - 1) // len(GPUS)}")
    print(f"  Log directory: {LOG_DIR}")
    print("="*80)
    
    # Ask for confirmation
    response = input("\nProceed with job submission? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("Aborted.")
        return
    
    print("\nStarting parallel job execution...\n")
    
    # Track active jobs: gpu -> (process, log_file, job_info)
    active_jobs = {}
    job_queue = list(enumerate(all_combinations, start=1))
    completed_count = 0
    
    start_time = time.time()
    
    while job_queue or active_jobs:
        # Fill available GPU slots
        while job_queue and len(active_jobs) < len(GPUS):
            job_index, (fiducial_type, upper_cut, mask_area) = job_queue.pop(0)
            
            # Find available GPU
            available_gpus = [g for g in GPUS if g not in active_jobs]
            if not available_gpus:
                break
            
            gpu = available_gpus[0]
            
            # Submit job
            command = build_command(fiducial_type, upper_cut, mask_area, gpu)
            process, log_file = submit_job(
                command, gpu, fiducial_type, upper_cut, mask_area, job_index, total_jobs
            )
            
            job_info = {
                'fiducial_type': fiducial_type,
                'upper_cut': upper_cut,
                'mask_area': mask_area,
                'index': job_index
            }
            active_jobs[gpu] = (process, log_file, job_info)
        
        # Check for completed jobs
        completed = check_processes(active_jobs)
        for gpu, returncode, log_file, job_info in completed:
            completed_count += 1
            status = "✓ SUCCESS" if returncode == 0 else "✗ FAILED"
            print(f"\n[{status}] GPU {gpu}: Job {job_info['index']}/{total_jobs} completed")
            print(f"  fid={job_info['fiducial_type']}, ucut={job_info['upper_cut']}, mask={int(job_info['mask_area'])}")
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
