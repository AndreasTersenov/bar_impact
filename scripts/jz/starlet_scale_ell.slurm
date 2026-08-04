#!/bin/bash
#SBATCH --job-name=starlet_ell
#SBATCH --account=nzu@cpu
#SBATCH --partition=cpu_p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=01:00:00
#SBATCH --output=logs/starlet_ell_%j.out
#SBATCH --error=logs/starlet_ell_%j.out
#
# Starlet scale -> multipole response, for the paper appendix.
#
# WHY SLURM. --method both is 48 spherical-harmonic transforms at nside 512 (8 delta inputs +
# 40 white-noise realizations), each with anafast to lmax 1535. That is comfortably past the
# login node's 30-minute CPU cap, and the cap kills silently mid-run.
#
# pycs is a SOURCE CHECKOUT, not an installed package: it needs PYTHONPATH. jaxili supplies
# healpy and numpy; no env has both, which is why this combination is spelled out here rather
# than left to be rediscovered.
set -uo pipefail
cd /lustre/fsn1/projects/rech/prk/ulx34io/bar_impact || exit 1
export PYTHONNOUSERSITE=1
export PYTHONPATH=/lustre/fswork/projects/rech/nzu/ulx34io/cosmostat_src
export OMP_NUM_THREADS=8
PY=/lustre/fswork/projects/rech/nzu/ulx34io/envs/jaxili/bin/python
"$PY" -u scripts/diagnostics/starlet_scale_ell.py --method both --ndelta 8 --nreal 40
