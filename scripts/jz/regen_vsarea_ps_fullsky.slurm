#!/bin/bash
#SBATCH --job-name=regen_psfs
#SBATCH --account=nzu@cpu
#SBATCH --partition=cpu_p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --hint=nomultithread
#SBATCH --time=00:30:00
#SBATCH --output=logs/regen_psfs_%j.out
#SBATCH --error=logs/regen_psfs_%j.out
#
# The THREE published power-spectrum contours_vs_area figures source from the
# _with_fullsky_ variants, not the plain ones -- they carry a SEVENTH contour (full sky)
# on top of the six masked areas. Job 582498 regenerated only the six-area variants, so
# without this the PS figures would stay on the old style while peaks and l1 moved.
#
# Confirmed from each slug's recorded source in paper/figures/*/README.md:
#   contours_vs_area_ps_null_pooled          -> ..._ps_null_l37-1020_with_fullsky_pooled
#   contours_vs_area_ps_biased_pooled        -> ..._ps_biased_l37-1020_with_fullsky_pooled
#   contours_vs_area_ps_biased_single_seed   -> ..._ps_biased_l37-1020_with_fullsky_single_seed
#
# The 7th series is why CYCLE gained a 7th colour (#000000): cycle_colors refuses to wrap,
# so this job would have died at the pre-flight rather than emit two areas in one colour.
#
# Full sky is PS-only in practice: peak_counts_processing.py has no full-sky submean branch,
# so there is no peaks/l1 full-sky product to add a 7th contour from.

set -uo pipefail
cd /lustre/fsn1/projects/rech/prk/ulx34io/bar_impact || exit 1
PY=/lustre/fswork/projects/rech/nzu/ulx34io/envs/jaxili/bin/python
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p logs
STAMP=$(date +%s)

"$PY" -c "
import sys; sys.path.insert(0,'scripts')
import paper_contour_style as PCS
c = PCS.cycle_colors(7)
print(f'[pre-flight] 7 colours for 6 areas + full sky: {c}')
sys.exit(0 if len(set(c)) == 7 else 1)
" || { echo '[fatal] palette cannot supply 7 distinct colours'; exit 1; }

pids=()
for spec in "null pooled" "biased pooled" "biased single"; do
  set -- $spec
  echo "[run ] ps --role $1 --seed-mode $2 --include-fullsky"
  "$PY" -u scripts/diagnostics/plot_contours_vs_area.py \
      --statistic ps --role "$1" --seed-mode "$2" --include-fullsky \
      > "logs/regen_psfs_${SLURM_JOB_ID}_${1}_${2}.log" 2>&1 &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done

echo; echo "=== verify ==="
D=outputs/plots/contours_vs_area
bad=0
for f in "$D/contours_vs_area_ps_null_l37-1020_with_fullsky_pooled.pdf" \
         "$D/contours_vs_area_ps_biased_l37-1020_with_fullsky_pooled.pdf" \
         "$D/contours_vs_area_ps_biased_l37-1020_with_fullsky_single_seed.pdf"; do
  if [ ! -f "$f" ]; then echo "  MISSING $(basename "$f")"; bad=1
  elif [ "$(stat -c %Y "$f")" -lt "$STAMP" ]; then echo "  STALE   $(basename "$f")"; bad=1
  else echo "  ok      $(basename "$f")"; fi
done
echo "failed_procs=$fail bad=$bad"
[ $fail -eq 0 ] && [ $bad -eq 0 ] || exit 1
