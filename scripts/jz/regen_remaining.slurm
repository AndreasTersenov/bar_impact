#!/bin/bash
#SBATCH --job-name=regen_rest
#SBATCH --account=nzu@cpu
#SBATCH --partition=cpu_p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=01:30:00
#SBATCH --output=logs/regen_rest_%j.out
#SBATCH --error=logs/regen_rest_%j.out
#
# Rebuild the 43 published figures not yet on the current style, via
# scripts/paper/regen_remaining_figures.py. Serial on purpose: several generators write into
# shared output directories and a couple emit two figures per run, so parallelism here risks
# two processes racing on the same stem for no real wall-clock gain at this size.
set -uo pipefail
cd /lustre/fsn1/projects/rech/prk/ulx34io/bar_impact || exit 1
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=/lustre/fswork/projects/rech/nzu/ulx34io/envs/jaxili/bin/python
"$PY" -u scripts/paper/regen_remaining_figures.py
