#!/bin/bash
# Clean full-sky control: paper's OWN healpy pipeline (run_npe_inference_auto_cross_ps.py),
# at the paper cut (l100) and the low-ell-inclusive cut (l30), null + baryon.
# Output to an ISOLATED samples dir so the paper _npe files are never overwritten.
JPY=/home/tersenov/anaconda3/envs/jaxili/bin/python
SCRIPT=/lustre/fsn1/projects/rech/prk/ulx34io/bar_impact/scripts/run_npe_inference_auto_cross_ps.py
OUT=/lustre/fsn1/projects/rech/prk/ulx34io/bar_impact/outputs/diagnostics/fullsky_baseline
SAMP="$OUT/samples"; CKPT="$OUT/ckpt"; LOG="$OUT/logs"
mkdir -p "$SAMP" "$CKPT" "$LOG"
STATUS="$LOG/STATUS.log"

run() {  # lowercut fidtype gpu
  local LC=$1 FT=$2 G=$3
  echo "$(date '+%T') START healpy fullsky l$LC fid=$FT gpu=$G" >> "$STATUS"
  $JPY "$SCRIPT" --simulation-type nobaryons --fiducial-type "$FT" \
    --bins 1,2,3,4 --lower-cut "$LC" --upper-cut 1024 --lmax 1024 \
    --noisy --noise-level 0.26 --rebin 10 --train \
    --samples-dir "$SAMP" --checkpoint-dir "$CKPT/l${LC}_${FT}" --gpu "$G" \
    > "$LOG/healpy_fullsky_l${LC}_${FT}.log" 2>&1
  echo "$(date '+%T') DONE  healpy fullsky l$LC fid=$FT exit=$?" >> "$STATUS"
}

echo "$(date '+%T') ===== HEALPY FULLSKY CONTROL START =====" >> "$STATUS"
# Round 1: l100 null (gpu2) + l100 baryon (gpu3)
run 100 nobaryons 2 &
run 100 baryonified 3 &
wait
# Round 2: l30 null (gpu2) + l30 baryon (gpu3)
run 30 nobaryons 2 &
run 30 baryonified 3 &
wait
echo "$(date '+%T') ===== HEALPY FULLSKY CONTROL COMPLETE =====" >> "$STATUS"
