#!/bin/bash
#SBATCH --job-name=restyle_ct
#SBATCH --account=nzu@cpu
#SBATCH --partition=cpu_p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --hint=nomultithread
#SBATCH --time=04:00:00
#SBATCH --output=logs/restyle_ct_%j.out
#SBATCH --error=logs/restyle_ct_%j.out
#
# Restyle every contour figure to the Okabe-Ito palette and the shared getdist type
# (scripts/paper_contour_style.py), regenerate, diff values.csv against what is currently
# published, and republish.
#
# WHY SLURM AND NOT THE LOGIN NODE. Each figure is a getdist KDE over pooled posteriors and
# takes ~12 min at 94% of one core; 21 of them is ~4 h serial. The login node caps a process
# at 30 min CPU, so a serial run there gets killed partway through, leaving some figures
# restyled and some not -- the worst possible outcome for a figure set that must be
# internally consistent. The figures are independent, so a compute node runs them in
# parallel and the whole batch lands in roughly the time of the slowest one.
#
# OMP_NUM_THREADS=1 is required: getdist's numpy would otherwise open a thread pool per
# process and 12 concurrent jobs would oversubscribe the node into thrashing.
#
# THE values.csv DIFF IS THE POINT, not a formality. Regenerating a figure is NOT guaranteed
# to be style-only: bias_vs_area_three_stats had its errorbar default silently flipped
# std->sem underneath it by an unrelated commit, which would have shrunk every drawn bar by
# sqrt(n) under the guise of "just the font". Every figure's numbers are therefore compared
# BEFORE it is allowed to replace a published one, and any that move are reported as DIFF
# rather than absorbed.
#
#   sbatch scripts/jz/restyle_contour_figures.slurm
set -uo pipefail
cd /lustre/fsn1/projects/rech/prk/ulx34io/bar_impact
mkdir -p logs

JAX=/lustre/fswork/projects/rech/nzu/ulx34io/envs/jaxili/bin/python
ANA=/lustre/fswork/projects/rech/nzu/ulx34io/envs/aname/bin/python
C3=scripts/diagnostics/plot_contours_three_stats.py
CVA=scripts/diagnostics/plot_contours_vs_area.py
CT=outputs/plots/contours_three_stats
CV=outputs/plots/contours_vs_area

export PALETTE=okabe
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONNOUSERSITE=1

JOBS=$(mktemp)   # slug \t stem \t command

add () { printf '%s\t%s\t%s\n' "$1" "$2" "$3" >> "$JOBS"; }

# ---------------- plot_contours_three_stats (12) ----------------
add contours_three_stats_biased_pooled $CT/contours_PS_peaks_L1_biased_14000 \
  "$JAX $C3 --area 14000 --mode biased --cut-mode full --seed-mode pooled --out $CT/contours_PS_peaks_L1_biased_14000"
add contours_three_stats_null_pooled $CT/contours_PS_peaks_L1_null_14000 \
  "$JAX $C3 --area 14000 --mode null --cut-mode full --seed-mode pooled --out $CT/contours_PS_peaks_L1_null_14000"
add contours_three_stats_both_pooled $CT/contours_PS_peaks_L1_both_14000 \
  "$JAX $C3 --area 14000 --mode both --cut-mode full --seed-mode pooled --out $CT/contours_PS_peaks_L1_both_14000"
add contours_baryon_safe_biased_14000_pooled $CT/contours_PS_peaks_L1_biased_14000_bsafe_l460_scales234 \
  "$JAX $C3 --area 14000 --mode biased --cut-mode baryon-safe --hos-scales scales234 --seed-mode pooled --out $CT/contours_PS_peaks_L1_biased_14000_bsafe_l460_scales234"
add contours_baryon_safe_biased_14000_single_seed $CT/contours_PS_peaks_L1_biased_14000_bsafe_l460_scales234_single_seed \
  "$JAX $C3 --area 14000 --mode biased --cut-mode baryon-safe --hos-scales scales234 --seed-mode single --out $CT/contours_PS_peaks_L1_biased_14000_bsafe_l460_scales234_single_seed"
add contours_baryon_safe_biased_14000_scales2345_pooled $CT/contours_PS_peaks_L1_biased_14000_bsafe_l460_scales2345_pooled \
  "$JAX $C3 --area 14000 --mode biased --cut-mode baryon-safe --hos-scales scales2345 --seed-mode pooled --out $CT/contours_PS_peaks_L1_biased_14000_bsafe_l460_scales2345_pooled"
add contours_baryon_safe_biased_14000_scales2345_single_seed $CT/contours_PS_peaks_L1_biased_14000_bsafe_l460_scales2345_single_seed \
  "$JAX $C3 --area 14000 --mode biased --cut-mode baryon-safe --hos-scales scales2345 --seed-mode single --out $CT/contours_PS_peaks_L1_biased_14000_bsafe_l460_scales2345_single_seed"
add contours_baryon_safe_both_14000_pooled $CT/contours_PS_peaks_L1_both_14000_bsafe_l460_scales234 \
  "$JAX $C3 --area 14000 --mode both --cut-mode baryon-safe --hos-scales scales234 --seed-mode pooled --out $CT/contours_PS_peaks_L1_both_14000_bsafe_l460_scales234"
for L in 300 340; do
  add contours_baryon_safe_biased_fullsky_l${L}_pooled $CT/contours_PS_peaks_L1_biased_fullsky_bsafe_l${L}_scales234_pooled \
    "$JAX $C3 --area fullsky --mode biased --cut-mode baryon-safe --ps-lmax $L --hos-scales scales234 --seed-mode pooled --out $CT/contours_PS_peaks_L1_biased_fullsky_bsafe_l${L}_scales234_pooled"
  add contours_baryon_safe_biased_fullsky_l${L}_single_seed $CT/contours_PS_peaks_L1_biased_fullsky_bsafe_l${L}_scales234_single_seed \
    "$JAX $C3 --area fullsky --mode biased --cut-mode baryon-safe --ps-lmax $L --hos-scales scales234 --seed-mode single --out $CT/contours_PS_peaks_L1_biased_fullsky_bsafe_l${L}_scales234_single_seed"
done

# ---------------- plot_contours_vs_area (9) ----------------
for S in peaks l1; do
  add contours_vs_area_${S}_null_pooled $CV/contours_vs_area_${S}_null_l37-1020 \
    "$JAX $CVA --statistic $S --role null --seed-mode pooled"
  add contours_vs_area_${S}_biased_pooled $CV/contours_vs_area_${S}_biased_l37-1020 \
    "$JAX $CVA --statistic $S --role biased --seed-mode pooled"
  add contours_vs_area_${S}_biased_single_seed $CV/contours_vs_area_${S}_biased_l37-1020_single_seed \
    "$JAX $CVA --statistic $S --role biased --seed-mode single"
done
add contours_vs_area_ps_null_pooled $CV/contours_vs_area_ps_null_l37-1020_with_fullsky \
  "$JAX $CVA --statistic ps --role null --include-fullsky --seed-mode pooled"
add contours_vs_area_ps_biased_pooled $CV/contours_vs_area_ps_biased_l37-1020_with_fullsky \
  "$JAX $CVA --statistic ps --role biased --include-fullsky --seed-mode pooled"
add contours_vs_area_ps_biased_single_seed $CV/contours_vs_area_ps_biased_l37-1020_with_fullsky_single_seed \
  "$JAX $CVA --statistic ps --role biased --include-fullsky --seed-mode single"

N=$(wc -l < "$JOBS")
echo "=== restyle $N contour figures to PALETTE=$PALETTE ==="
echo "start : $(date -Is)"
echo

# ---- phase 1: generate, in parallel -----------------------------------------
echo "--- phase 1: generating (12-way parallel) ---"
cut -f1,3 "$JOBS" | while IFS=$'\t' read -r slug cmd; do
  printf '%s\t%s\n' "$slug" "$cmd"
done | xargs -P 12 -d '\n' -I{} bash -c '
  slug=$(printf "%s" "{}" | cut -f1)
  cmd=$(printf "%s" "{}" | cut -f2-)
  if eval "$cmd" > logs/restyle_$slug.log 2>&1; then
    echo "  [gen ok  ] $slug"
  else
    echo "  [gen FAIL] $slug  (see logs/restyle_$slug.log)"
  fi
'
echo
echo "--- phase 2: diff vs published, then publish ---"

PASS=0; DIFF=0; FAIL=0
while IFS=$'\t' read -r slug stem cmd; do
  pub="paper/figures/$slug/values.csv"
  if [ ! -f "$stem.png" ]; then
    echo "  [FAIL] $slug — no output at $stem.png"; FAIL=$((FAIL+1)); continue
  fi
  if [ -f "$pub" ] && [ -f "${stem}_values.csv" ]; then
    if diff -q "$pub" "${stem}_values.csv" > /dev/null; then
      echo "  [ok  ] $slug — values.csv identical"; PASS=$((PASS+1))
    else
      echo "  [DIFF] $slug — values.csv CHANGED:"
      diff "$pub" "${stem}_values.csv" | head -6 | sed 's/^/         /'
      DIFF=$((DIFF+1))
    fi
  else
    echo "  [new ] $slug — no published values.csv to compare"; PASS=$((PASS+1))
  fi
  $ANA scripts/paper/figures.py publish "$stem" --slug "$slug" --force 2>&1 \
     | grep -E "published|warn|refus" | sed 's/^/         /'
done < "$JOBS"

echo
echo "--- rebuilding manifest + verifying ---"
$ANA scripts/paper/figures.py manifest 2>&1 | tail -1
$ANA scripts/paper/figures.py verify 2>&1 | tail -3

echo
echo "=== SUMMARY  identical=$PASS  values-changed=$DIFF  failed=$FAIL  (of $N) ==="
echo "end   : $(date -Is)"
rm -f "$JOBS"
[ $FAIL -eq 0 ]
