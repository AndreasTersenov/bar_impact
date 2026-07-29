#!/bin/bash
# Overnight Phase-1 reprocess: masked PS, mean-subtracted, per-l (nlb=1), lmax 1024,
# for all 6 masks x {nobaryons, baryonified} x {fiducial, grid}. 14000 first.
# Deterministic, validated by the fiducial pilot. Continues on error; logs to STATUS.log.
set -u
CST=/home/tersenov/software/cosmostat_new/cosmostat/cosmostat_new/bin/python
SCRIPT=/lustre/fsn1/projects/rech/prk/ulx34io/bar_impact/scripts/cross_power_spectrum_processing_master.py
LOGDIR=/lustre/fsn1/projects/rech/prk/ulx34io/bar_impact/outputs/diagnostics/prod_overnight
mkdir -p "$LOGDIR"
STATUS="$LOGDIR/STATUS.log"

run() {  # $1 = tag ; rest = extra flags for this run
  local tag="$1"; shift
  echo "$(date '+%F %T') START $tag" >> "$STATUS"
  $CST -u "$SCRIPT" "$@" \
       --apply-mask --apodization-scale-deg 2.0 --subtract-mean \
       --lmax 1024 --bin-range 1 2 3 4 --noise-level 0.26 \
       --num-workers 50 --aggregate-for-inference \
       > "$LOGDIR/${tag}.log" 2>&1
  echo "$(date '+%F %T') DONE  $tag exit=$?" >> "$STATUS"
}

echo "$(date '+%F %T') ===== OVERNIGHT REPROCESS START =====" >> "$STATUS"
for AREA in 14000 10000 5000 2000 28000 35000; do
  # fiducial = the "observed data" scenarios: need BOTH nobaryons and baryonified
  run "fid_${AREA}_nobar"  --fiducial               --mask-area-sqdeg "$AREA"
  run "fid_${AREA}_baryon" --fiducial --baryonified --mask-area-sqdeg "$AREA"
  # grid = the NPE training set: nobaryons ONLY (no baryonified grid needed)
  run "grid_${AREA}_nobar"                           --mask-area-sqdeg "$AREA"
done
echo "$(date '+%F %T') ===== OVERNIGHT REPROCESS COMPLETE =====" >> "$STATUS"
