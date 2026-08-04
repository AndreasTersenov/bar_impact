#!/bin/bash
#SBATCH --job-name=regen_vsarea
#SBATCH --account=nzu@cpu
#SBATCH --partition=cpu_p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=logs/regen_vsarea_%j.out
#SBATCH --error=logs/regen_vsarea_%j.out
#
# Regenerate all nine published contours_vs_area figures on the paper's contour style.
#
# WHAT CHANGED IN THE GENERATOR (2026-08-04, scripts/diagnostics/plot_contours_vs_area.py):
#   * FILLED contours (were unfilled)
#   * flat Okabe-Ito cycle colours from paper_contour_style.cycle_colors (was a light->dark
#     single-hue ramp built from a hardcoded copy of the palette)
#   * sigma_8 axis label (was mislabelled S_8 -- the samples were always sigma_8; see
#     scripts/paper/fix_sigma8_labels.py for the verification against cosmo_params.npy)
# Nothing about the posteriors, the seeds, the collapse guards or the FoM values changes. This
# job only redraws.
#
# WHY SLURM. One figure takes ~18 min of pure getdist KDE. Nine of them is ~2.7 h serial,
# and the login node has a 30-minute CPU cap -- this cannot run there. Nine independent
# renders, so they go in parallel on one node and the job finishes in roughly the time of the
# slowest single figure.
#
# NINE FIGURES = 3 statistics x {null pooled, biased pooled, biased single-seed}. There is no
# published null single-seed figure, so it is not generated here; add it deliberately if the
# paper ever needs one rather than by widening this loop.
#
# THREADS. numpy/OpenMP would otherwise grab the whole node per process and 9 of them would
# thrash. One thread each, 12 cores requested for 9 workers plus headroom.
#
# Submit from the repo root:  sbatch scripts/jz/regen_contours_vs_area.slurm

set -uo pipefail

cd /lustre/fsn1/projects/rech/prk/ulx34io/bar_impact || exit 1

PY=/lustre/fswork/projects/rech/nzu/ulx34io/envs/jaxili/bin/python

# ~/.local carries an astropy built against numpy 1.x that shadows this env's numpy 2.
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

mkdir -p logs outputs/plots/contours_vs_area
STAMP=$(date +%s)

echo "=== regenerate contours_vs_area (9 figures) ==="
echo "start : $(date -Is)"
echo "python: $PY"
echo

# PRE-FLIGHT: the palette must resolve to six DISTINCT colours before spending an hour of CPU.
# cycle_colors raises if the palette is short, but a silent duplicate would only show up in the
# finished figure, where two survey areas would share a colour.
"$PY" -c "
import sys
sys.path.insert(0, 'scripts')
import paper_contour_style as PCS
c = PCS.cycle_colors(6)
print(f'[pre-flight] palette={PCS.palette_name()} colours={c}')
if len(set(c)) != 6:
    print('[fatal] palette gives duplicate colours for six areas'); sys.exit(1)
" || exit 1
echo

pids=(); names=()
launch() {   # launch <role> <seed-mode> <tag-for-log>
  local role=$1 seed=$2 tag=$3
  echo "[run ] role=$role seed-mode=$seed  -> logs/regen_vsarea_${SLURM_JOB_ID}_${tag}.log"
  "$PY" -u scripts/diagnostics/plot_contours_vs_area.py \
      --statistic all --role "$role" --seed-mode "$seed" \
      > "logs/regen_vsarea_${SLURM_JOB_ID}_${tag}.log" 2>&1 &
  pids+=($!); names+=("$tag")
}

# --statistic all loops ps/peaks/l1 inside one process. Three processes rather than nine keeps
# the parallelism at 3; splitting per statistic would be faster still but the three roles are
# what the published set is organised by, and 3 x ~18 min fits the walltime comfortably.
launch null   pooled null_pooled
launch biased pooled biased_pooled
launch biased single biased_single

fail=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "[ok  ] ${names[$i]}"
  else
    echo "[FAIL] ${names[$i]} -- see logs/regen_vsarea_${SLURM_JOB_ID}_${names[$i]}.log"
    fail=1
  fi
done

# VERIFY: every expected file must exist AND be newer than job start. A generator that skips a
# statistic (missing posteriors for one footprint, say) exits 0 having written nothing, so
# "rc=0" alone does not mean nine figures were produced.
echo
echo "=== verify ==="
D=outputs/plots/contours_vs_area
missing=0; stale=0
for stat in ps peaks l1; do
  for spec in "null _pooled" "biased _pooled" "biased _single_seed"; do
    set -- $spec
    f="$D/contours_vs_area_${stat}_${1}_l37-1020${2}.pdf"
    if [ ! -f "$f" ]; then
      echo "  MISSING  $(basename "$f")"; missing=$((missing+1))
    elif [ "$(stat -c %Y "$f")" -lt "$STAMP" ]; then
      echo "  STALE    $(basename "$f")  (not rewritten by this job)"; stale=$((stale+1))
    else
      echo "  ok       $(basename "$f")"
    fi
  done
done

echo
echo "missing=$missing stale=$stale failed_procs=$fail"
echo "end   : $(date -Is)"
[ $missing -eq 0 ] && [ $stale -eq 0 ] && [ $fail -eq 0 ] || exit 1
echo
echo "Next: scripts/paper/figures.py republish   (then verify)"
