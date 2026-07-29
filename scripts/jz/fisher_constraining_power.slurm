#!/bin/bash
#SBATCH --job-name=fisher_cp
#SBATCH --account=nzu@cpu
#SBATCH --partition=cpu_p1
#SBATCH --qos=qos_cpu-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=logs/fisher_cp_%j.out
#SBATCH --error=logs/fisher_cp_%j.out
#
# Regenerate the PS-vs-HOS Fisher constraining-power figures, including
# `fisher_contours_baryon_safe` (PS lmax=400 vs l1/peaks scales234) — the three summary
# statistics at their required scale cuts. The disk failure destroyed that figure's .pdf
# AND .png (both 100% zeros) plus fisher_covs.npz, so there is no surviving image.
#
# Runs as a batch job, not on the login node: the grid arrays are ~2.4 GB for the PS alone
# (4 x 16965x1025 autos + 16965x6150 cross, float64) before the HOS, and the login node is
# capped at ~5 GB / 30 min CPU. 10 cores on cpu_p1 gives ample memory headroom.
#
# fisher_fom_table_PRECRASH_REFERENCE.json is the surviving pre-crash FoM table; compare
# the regenerated fisher_fom_table.json against it to confirm the numbers reproduce. That
# comparison is the whole point of keeping it — do not delete it.
#
# Submit from the repo root:  sbatch scripts/jz/fisher_constraining_power.slurm

set -uo pipefail

cd /lustre/fsn1/projects/rech/prk/ulx34io/bar_impact || exit 1
PY=/lustre/fswork/projects/rech/nzu/ulx34io/envs/aname/bin/python

echo "=== Fisher constraining power ==="
echo "start : $(date -Is)"
echo "python: $PY"
echo

srun $PY -u scripts/diagnostics/fisher_constraining_power.py
rc=$?

echo
echo "exit  : $rc"
echo "end   : $(date -Is)"
exit $rc
