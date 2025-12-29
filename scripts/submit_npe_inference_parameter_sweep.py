#!/usr/bin/env python3
"""
Submit NPE inference jobs for parameter sweep across multiple configurations.

This script submits jobs with different combinations of:
- fiducial-type: nobaryons, baryonified
- upper-cut: 520 to 1020 (step 20)
- mask-area-sqdeg: 2000.0, 5000.0, 10000.0, 14000.0, 28000.0

Jobs are distributed across GPUs 0, 1, 2 for parallel execution.
"""

import subprocess
import itertools
import time
from pathlib import Path

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
    
    return " ".join(cmd_parts)

def submit_job(command, gpu, job_index, total_jobs):
    """Submit a job to run in the background."""
    print(f"[Job {job_index}/{total_jobs}] Submitting to GPU {gpu}")
    print(f"  Command: {command}")
    
    # Run the command in the background
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    return process

def main():
    # Generate all parameter combinations
    all_combinations = list(itertools.product(
        FIDUCIAL_TYPES,
        UPPER_CUTS,
        MASK_AREAS
    ))
    
    total_jobs = len(all_combinations)
    print("="*80)
    print("NPE Inference Parameter Sweep")
    print("="*80)
    print(f"Total jobs to submit: {total_jobs}")
    print(f"  Fiducial types: {FIDUCIAL_TYPES}")
    print(f"  Upper cuts: {len(UPPER_CUTS)} values ({UPPER_CUTS[0]} to {UPPER_CUTS[-1]})")
    print(f"  Mask areas: {MASK_AREAS}")
    print(f"  GPUs: {GPUS}")
    print(f"  Jobs per GPU: ~{total_jobs // len(GPUS)}")
    print("="*80)
    
    # Ask for confirmation
    response = input("\nProceed with job submission? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("Aborted.")
        return
    
    # Submit jobs in round-robin fashion across GPUs
    active_processes = {}  # gpu -> process
    job_index = 0
    
    for fiducial_type, upper_cut, mask_area in all_combinations:
        job_index += 1
        
        # Round-robin GPU selection
        gpu = GPUS[(job_index - 1) % len(GPUS)]
        
        # Wait if this GPU already has a running job
        if gpu in active_processes:
            print(f"\n[GPU {gpu}] Waiting for previous job to complete...")
            active_processes[gpu].wait()
            print(f"[GPU {gpu}] Previous job completed.")
            del active_processes[gpu]
        
        # Build and submit command
        command = build_command(fiducial_type, upper_cut, mask_area, gpu)
        process = submit_job(command, gpu, job_index, total_jobs)
        active_processes[gpu] = process
        
        print(f"  Job submitted (PID: {process.pid})")
        print()
        
        # Small delay to avoid overwhelming the system
        time.sleep(1)
    
    # Wait for all remaining jobs to complete
    print("\n" + "="*80)
    print("All jobs submitted. Waiting for completion...")
    print("="*80)
    
    for gpu, process in active_processes.items():
        print(f"[GPU {gpu}] Waiting for final job to complete...")
        process.wait()
        print(f"[GPU {gpu}] Job completed.")
    
    print("\n" + "="*80)
    print("All jobs completed!")
    print("="*80)

if __name__ == "__main__":
    main()
