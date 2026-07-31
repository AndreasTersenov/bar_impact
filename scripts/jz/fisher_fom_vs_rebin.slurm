#!/bin/bash
#SBATCH --job-name=fom_vs_rebin
#SBATCH --account=nzu@cpu
#SBATCH --partition=cpu_p1
#SBATCH --qos=qos_cpu-dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=01:30:00
#SBATCH --output=logs/fom_vs_rebin_%j.out
#SBATCH --error=logs/fom_vs_rebin_%j.out
#
# Fisher FoM3 vs ell-rebinning, both arms, fixed cuts. Answers "what does rebin=20 cost us?" without
# training anything — pure numpy, so it belongs on a CPU node, not a GPU.
#
# FISHER_REBIN is read at IMPORT time by the fisher_* modules, so each value needs its own process.
# The analytic covariance is the only one that can follow to native resolution (the hybrid's low-rank
# term comes from 200 permutations and becomes noise past a few hundred features), so the analytic
# leg is the primary; the hybrid leg runs only at r20/r10 as a cross-check that the two agree where
# both are valid.
#
# RIDGE_REL is threaded through because covk=analytic is singular at r20 as shipped — set it from
# diag_analytic_cov_rank.py's step [5] output before running, or leave 0 to record the failures.
#
# Submit:  sbatch scripts/jz/fisher_fom_vs_rebin.slurm
set -euo pipefail
cd /lustre/fsn1/projects/rech/prk/ulx34io/bar_impact
mkdir -p logs outputs/diagnostics

PY=/lustre/fswork/projects/rech/nzu/ulx34io/envs/jaxili/bin/python
OUT=outputs/diagnostics/fisher_fom_vs_rebin.csv
RIDGE_REL="${RIDGE_REL:-0}"
export FISHER_AREA=14000
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

# --- pre-flight: the module must import and its cut oracle must pass, or the run is meaningless ---
echo "[pre-flight] score_cut_utils validation"
FISHER_REBIN=20 $PY scripts/score_cut_utils.py || { echo "[fatal] validation failed"; exit 1; }

rm -f "$OUT"

echo "=== analytic covariance (primary; the only path that reaches native) ==="
for R in 20 10 5 2 1; do
  echo "--- FISHER_REBIN=$R"
  FISHER_REBIN=$R $PY scripts/diagnostics/fisher_fom_vs_rebin.py \
      --out "$OUT" --covk analytic --ridge-rel "$RIDGE_REL" || echo "[warn] rebin=$R failed, continuing"
done

echo "=== hybrid covariance (cross-check, only where 200 perms can still support the low-rank term) ==="
for R in 20 10; do
  echo "--- FISHER_REBIN=$R (hybrid)"
  FISHER_REBIN=$R $PY scripts/diagnostics/fisher_fom_vs_rebin.py \
      --out "$OUT" --covk hybrid || echo "[warn] hybrid rebin=$R failed, continuing"
done

echo "=== RESULT ==="
column -s, -t < "$OUT"
echo "wrote $OUT"
