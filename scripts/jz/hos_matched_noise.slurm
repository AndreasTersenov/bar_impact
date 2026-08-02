#!/bin/bash
#SBATCH --job-name=hos_matchnoise
#SBATCH --account=nzu@cpu
#SBATCH --partition=cpu_p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --hint=nomultithread
#SBATCH --time=06:00:00
#SBATCH --output=logs/hos_matchnoise_%j.out
#SBATCH --error=logs/hos_matchnoise_%j.out
#
# Starlet l1-norm + peak counts recomputed with the shape noise MATCHED between the baryonified and
# DMO maps, and the SNR bin centres saved. See scripts/hos_matched_noise_processing.py for the
# three defects this fixes and how the statistic is obtained without the pysap bindings.
#
# WALLTIME sized from a measured smoke test rather than a guess: 1 permutation x 1 bin on one
# login core ran 4 spherical starlet transforms (2 variants x 2 statistics) at ~43 s each, ~3 min.
# The full job is 200 perms x 4 bins = 800 such units ~ 40 core-hours, so ~1 h on 40 cores. 6 h is
# requested because this is PURE PYTHON (no C++ bindings), unlike the power-spectrum job whose
# anafast is compiled -- so the compute-node speedup that made that one finish in 3 minutes should
# NOT be assumed here.
#
# PYTHONPATH must carry the cosmostat checkout: pycs is not installed, and cannot be, because
# pycs.sparsity.sparse2d.starlet raises NameError without the pysap C++ bindings. Only the
# SPHERICAL transform is needed and that has a working pure-python path.
# PYTHONNOUSERSITE=1 is required: ~/.local has an astropy built against numpy<2.
#
#   sbatch scripts/jz/hos_matched_noise.slurm
#   NPERM=4 sbatch scripts/jz/hos_matched_noise.slurm     # short validation run
set -euo pipefail
cd /lustre/fsn1/projects/rech/prk/ulx34io/bar_impact
mkdir -p logs

NPERM="${NPERM:-200}"
BINS="${BINS:-1,2,3,4}"
NPROC="${NPROC:-40}"
TAG="${TAG:-matchednoise}"

PY=/lustre/fswork/projects/rech/nzu/ulx34io/envs/jaxili/bin/python
export COSMOSTAT_SRC=/lustre/fswork/projects/rech/nzu/ulx34io/cosmostat_src
export PYTHONPATH=$COSMOSTAT_SRC
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1        # numpy must not thread against the process pool

echo "=== matched-noise HOS (l1-norm + peak counts) ==="
echo "perms : $NPERM   bins: $BINS   nproc: $NPROC   tag: $TAG"
echo "start : $(date -Is)"
echo

srun $PY -u scripts/hos_matched_noise_processing.py \
    --nperm "$NPERM" --bins "$BINS" --nproc "$NPROC" --tag "$TAG" ${BNT:+--bnt}
rc=$?

echo
echo "end   : $(date -Is)   rc=$rc"
exit $rc
