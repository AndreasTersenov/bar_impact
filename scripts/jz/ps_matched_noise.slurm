#!/bin/bash
#SBATCH --job-name=ps_matchnoise
#SBATCH --account=nzu@cpu
#SBATCH --partition=cpu_p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --hint=nomultithread
#SBATCH --time=04:00:00
#SBATCH --output=logs/ps_matchnoise_%j.out
#SBATCH --error=logs/ps_matchnoise_%j.out
#
# Recompute the fiducial auto power spectra with the shape noise MATCHED between the baryonified
# and DMO maps, answering the referee's second point on Fig. 2. See
# scripts/ps_matched_noise_processing.py for the physics and the measured justification.
#
# WALLTIME sized from a measured smoke test, not a guess: 2 permutations x 1 bin on 2 login-node
# cores took 3m14s, i.e. ~1.6 min per spectrum. The full job is 200 perms x 4 bins x 2 variants =
# 1600 spectra ~ 43 core-hours, so ~1.1 h wall on 40 cores. 4 h is requested because compute-node
# throughput is unmeasured and the shared-dataset filesystem may throttle 400 concurrent h5 reads.
#
# READS the read-only shared dataset /lustre/fsmisc/dataset/CosmoGridV1 -- no download, no quota
# cost, and the baryonified and nobaryons maps for a permutation sit in the SAME directory and
# share the shell permutation, so the ICs are common exactly as the referee assumed.
#
# WRITES with tag `matchednoise`, alongside the published spectra rather than over them. The old
# files stay valid inputs for the published figure; nothing is destroyed and the two can be
# compared directly.
#
# PYTHONNOUSERSITE=1 IS REQUIRED. ~/.local holds an astropy built against numpy<2 which shadows the
# env and dies with "module 'numpy' has no attribute 'in1d'" before healpy is even reached.
# jaxili is the interpreter with healpy AND h5py (aname has neither).
#
#   sbatch scripts/jz/ps_matched_noise.slurm
#   NPERM=8 sbatch scripts/jz/ps_matched_noise.slurm     # short validation run
set -euo pipefail
cd /lustre/fsn1/projects/rech/prk/ulx34io/bar_impact
mkdir -p logs

NPERM="${NPERM:-200}"
BINS="${BINS:-1,2,3,4}"
NPROC="${NPROC:-40}"
TAG="${TAG:-matchednoise}"

PY=/lustre/fswork/projects/rech/nzu/ulx34io/envs/jaxili/bin/python
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1        # healpy/numpy must not thread against the process pool

echo "=== matched-noise power spectra ==="
echo "perms : $NPERM   bins: $BINS   nproc: $NPROC   tag: $TAG"
echo "start : $(date -Is)"
echo

srun $PY -u scripts/ps_matched_noise_processing.py \
    --nperm "$NPERM" --bins "$BINS" --nproc "$NPROC" --tag "$TAG"
rc=$?

echo
echo "end   : $(date -Is)   rc=$rc"
exit $rc
