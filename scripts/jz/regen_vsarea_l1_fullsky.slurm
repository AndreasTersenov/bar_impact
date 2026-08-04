#!/bin/bash
#SBATCH --job-name=regen_l1fs
#SBATCH --account=nzu@cpu
#SBATCH --partition=cpu_p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --hint=nomultithread
#SBATCH --time=00:30:00
#SBATCH --output=logs/regen_l1fs_%j.out
#SBATCH --error=logs/regen_l1fs_%j.out
#
# EXPLORATORY: add a full-sky 7th contour to the l1 contours_vs_area figures.
#
# The full-sky l1 posteriors DO exist (unlike peaks, whose submean branch is gated on
# apply_mask so no full-sky submean peak product was ever produced), but the two roles are
# not equally supplied:
#     null   nobaryons_vs_nobaryons    5 seeds
#     biased nobaryons_vs_baryonified  1 seed
# The masked areas pool 5-9 seeds each, so on the BIASED figures the full-sky contour would
# carry one seed's training scatter against six pooled ones -- narrower for a reason that has
# nothing to do with survey area, which is the one thing the figure is about. Generated here
# so the effect can be looked at; NOT published without that being judged acceptable.
#
# Unlike the PS full-sky leg, this is like-for-like in pipeline terms: the l1 full-sky product
# carries _fullsky_submean_ and comes from the same wavelet pipeline as the masked ones. The PS
# full-sky contour, by contrast, skips MASTER decoupling and is not magnitude-comparable.

set -uo pipefail
cd /lustre/fsn1/projects/rech/prk/ulx34io/bar_impact || exit 1
PY=/lustre/fswork/projects/rech/nzu/ulx34io/envs/jaxili/bin/python
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p logs
pids=()
for spec in "null pooled" "biased pooled" "biased single"; do
  set -- $spec
  echo "[run ] l1 --role $1 --seed-mode $2 --include-fullsky"
  "$PY" -u scripts/diagnostics/plot_contours_vs_area.py \
      --statistic l1 --role "$1" --seed-mode "$2" --include-fullsky \
      > "logs/regen_l1fs_${SLURM_JOB_ID}_${1}_${2}.log" 2>&1 &
  pids+=($!)
done
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
echo; echo "=== seed counts actually used (full sky row is the one to check) ==="
grep -h "fullsky\|full sky\|seed(s)" logs/regen_l1fs_${SLURM_JOB_ID}_*.log | head -30
echo; ls -la outputs/plots/contours_vs_area/*l1*with_fullsky*.pdf 2>/dev/null
echo "failed_procs=$fail"
