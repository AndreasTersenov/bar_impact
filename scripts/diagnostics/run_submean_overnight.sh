#!/bin/bash
# Overnight: footprint-mean-subtracted (--submean) masked PEAK reprocess, NON-BNT, all 6 areas.
# SCOPE (confirmed with user): grid = NOBARYONS only; fiducial = BOTH nobaryons + baryonified.
# Per area: fid nobaryons + fid baryonified + grid nobaryons, then (only if the grid aggregate has
# the full 16965 rows) copy the grid _submean aggregate new_grid/ -> grid/ (where the NPE reads).
#
# SAFETY: after each combine we verify the aggregate row count (grid==16965 aligned with params,
# fid==200). If a per-file fails, the combine silently produces FEWER, MISALIGNED rows -> corrupt
# NPE training. The check below refuses to copy a short grid aggregate and logs a loud FAIL.
# Resumable (process_file skips existing files); NO --force-overwrite. Excludes BNT (needs order-A).
# NOTE: existing non-submean aggregates were made at the pycs default iter=3; this run uses the
# pycs_speedups iter=1 (peak histograms were bit-identical iter=1 vs iter=3 in testing) -> the
# submean-vs-nonsubmean validation comparison has at most a negligible iter confound.
cd /lustre/fsn1/projects/rech/prk/ulx34io/bar_impact
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=/home/tersenov/anaconda3/bin/python
B=/lustre/fsn1/projects/rech/prk/ulx34io/cosmogrid_products/stage3_forecast
W=60
AREAS="14001 5001 2001 10001 28001 35001"   # Euclid + gate first

rows() {  # echo row count of a .npy (or -1 on load failure)
  $PY -c "import numpy as np,sys
try: print(np.load(sys.argv[1],allow_pickle=True).shape[0])
except Exception: print(-1)" "$1"
}

echo "######## SUBMEAN OVERNIGHT REPROCESS START: $(date) ########"
for A in $AREAS; do
  echo ""; echo "############ AREA ${A} sqdeg — START $(date) ############"
  # 14001 has suspect partials from the killed single-area job -> force a clean reprocess.
  FORCE=""; [ "$A" = "14001" ] && FORCE="--force-overwrite"
  COM="--bins 1,2,3,4 --apply-mask --mask-area-sqdeg ${A} --submean --save-combined --num-workers ${W} ${FORCE}"

  echo ">>> [${A}] fiducial nobaryons $(date)"
  $PY scripts/peak_counts_processing.py --fiducial $COM 2>&1 | grep -v "pysap\|slow python" | tail -3
  echo ">>> [${A}] fiducial baryonified $(date)"
  $PY scripts/peak_counts_processing.py --fiducial --baryonified $COM 2>&1 | grep -v "pysap\|slow python" | tail -3
  echo ">>> [${A}] grid nobaryons (long) $(date)"
  $PY scripts/peak_counts_processing.py $COM 2>&1 | grep -v "pysap\|slow python" | tail -5

  echo ">>> [${A}] VERIFY row counts"
  ok=1
  for b in 1 2 3 4; do
    gf="$B/new_grid/all_peak_counts_grid_nobaryons_bin${b}_masked_${A}sqdeg_submean_noisy_s0.26_new_normalization.npy"
    fn="$B/fiducial/cosmo_fiducial/all_peak_counts_fiducial_nobaryons_bin${b}_masked_${A}sqdeg_submean_noisy_s0.26_new_normalization.npy"
    fb="$B/fiducial/cosmo_fiducial/all_peak_counts_fiducial_baryonified_bin${b}_masked_${A}sqdeg_submean_noisy_s0.26_new_normalization.npy"
    g=$(rows "$gf"); n=$(rows "$fn"); m=$(rows "$fb")
    echo "    bin${b}: grid=$g (want 16965)  fid_nobar=$n  fid_bar=$m (want 200)"
    [ "$g" = "16965" ] && [ "$n" = "200" ] && [ "$m" = "200" ] || ok=0
  done
  if [ "$ok" = "1" ]; then
    echo ">>> [${A}] OK — copy grid aggregates new_grid/ -> grid/"
    for b in 1 2 3 4; do
      f="all_peak_counts_grid_nobaryons_bin${b}_masked_${A}sqdeg_submean_noisy_s0.26_new_normalization.npy"
      cp -f "$B/new_grid/$f" "$B/grid/$f" && echo "    copied $f"
    done
  else
    echo "!!!!!!!! [${A}] ROWCOUNT FAIL — NOT copying to grid/. Investigate before using this area. !!!!!!!!"
  fi
  echo "############ AREA ${A} sqdeg — DONE $(date) ############"
done
echo ""; echo "######## SUBMEAN OVERNIGHT REPROCESS COMPLETE: $(date) ########"
