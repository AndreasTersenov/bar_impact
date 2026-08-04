#!/bin/bash
#SBATCH --job-name=regen_vsall
#SBATCH --account=nzu@cpu
#SBATCH --partition=cpu_p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --time=00:40:00
#SBATCH --output=logs/regen_vsall_%j.out
#SBATCH --error=logs/regen_vsall_%j.out
#
# Every contours_vs_area variant the paper publishes, in one pass: 15 figures.
#
#   WITHOUT full sky   ps, peaks, l1  x  {null pooled, biased pooled, biased single}  =  9
#   WITH    full sky   ps, l1         x  {null pooled, biased pooled, biased single}  =  6
#
# peaks has NO with-fullsky variant and cannot have one: there are zero full-sky peak-count
# posteriors of any tag, because peak_counts_processing.py gates its submean branch on
# apply_mask so no full-sky peaks product was ever produced. --include-fullsky on peaks now
# exits 1 rather than writing a figure tagged _with_fullsky containing only six contours.
#
# Regenerating the with-fullsky figures even though jobs 582931/583074 already made them:
# those ran BEFORE the under-seeding caveat was added, so their provenance does not carry it.
# The l1 biased full-sky contour rests on ONE seed against 5-9 for the masked areas and is
# ~1.8x tighter than area^1.5 allows; that has to be recorded in the figure's own provenance,
# not just known in conversation.

set -uo pipefail
cd /lustre/fsn1/projects/rech/prk/ulx34io/bar_impact || exit 1
PY=/lustre/fswork/projects/rech/nzu/ulx34io/envs/jaxili/bin/python
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p logs
STAMP=$(date +%s)

pids=(); tags=()
run() {  # run <stat> <role> <seedmode> <fullsky-flag-or-empty> <tag>
  "$PY" -u scripts/diagnostics/plot_contours_vs_area.py \
      --statistic "$1" --role "$2" --seed-mode "$3" $4 \
      > "logs/regen_vsall_${SLURM_JOB_ID}_$5.log" 2>&1 &
  pids+=($!); tags+=("$5")
}

for spec in "null pooled" "biased pooled" "biased single"; do
  set -- $spec
  run all "$1" "$2" ""                 "plain_${1}_${2}"
  run ps  "$1" "$2" "--include-fullsky" "fs_ps_${1}_${2}"
  run l1  "$1" "$2" "--include-fullsky" "fs_l1_${1}_${2}"
done

fail=0
for i in "${!pids[@]}"; do
  wait "${pids[$i]}" || { echo "[FAIL] ${tags[$i]}"; fail=1; }
done

echo "=== under-seeding warnings raised ==="
grep -h "\[WARN\]" logs/regen_vsall_${SLURM_JOB_ID}_*.log || echo "  (none)"

echo; echo "=== verify: 15 expected outputs ==="
D=outputs/plots/contours_vs_area; bad=0
check() { # check <file>
  if [ ! -f "$1" ]; then echo "  MISSING $(basename "$1")"; bad=1
  elif [ "$(stat -c %Y "$1")" -lt "$STAMP" ]; then echo "  STALE   $(basename "$1")"; bad=1
  else echo "  ok      $(basename "$1")"; fi
}
for spec in "null _pooled" "biased _pooled" "biased _single_seed"; do
  set -- $spec
  for stat in ps peaks l1; do check "$D/contours_vs_area_${stat}_${1}_l37-1020${2}.pdf"; done
  for stat in ps l1;       do check "$D/contours_vs_area_${stat}_${1}_l37-1020_with_fullsky${2}.pdf"; done
done
echo; echo "failed_procs=$fail bad=$bad"
[ $fail -eq 0 ] && [ $bad -eq 0 ] || exit 1
