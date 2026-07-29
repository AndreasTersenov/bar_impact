#!/bin/bash
# NPE fan-out: masked mean-subtracted PS, ell in [37,1024], rebin 10, for each
# (mask, fiducial_type). nobaryons grid trains; fiducial nobaryons/baryonified = the two
# observed-data scenarios. Up to 3 concurrent jobs on GPUs 0,1,2.
JPY=/home/tersenov/anaconda3/envs/jaxili/bin/python
SCRIPT=/lustre/fsn1/projects/rech/prk/ulx34io/bar_impact/scripts/run_npe_inference_auto_cross_ps_master.py
LOGDIR=/lustre/fsn1/projects/rech/prk/ulx34io/bar_impact/outputs/diagnostics/npe_prod
mkdir -p "$LOGDIR"
STATUS="$LOGDIR/NPE_STATUS.log"

LCUT="${LOWERCUT:-37}"   # ell_min, override with LOWERCUT=100 env var
run_npe() {  # area fidtype gpu
  local A=$1 FT=$2 G=$3
  echo "$(date '+%T') START npe mask=$A fid=$FT lcut=$LCUT gpu=$G" >> "$STATUS"
  $JPY "$SCRIPT" --simulation-type nobaryons --fiducial-type "$FT" \
    --bins 1,2,3,4 --lower-cut "$LCUT" --upper-cut 1024 --lmax 1024 \
    --noisy --noise-level 0.26 --masked --mask-area-sqdeg "$A" \
    --apodization-scale-deg 2.0 --subtract-mean --train --rebin 10 --gpu "$G" \
    > "$LOGDIR/npe_${A}_${FT}_l${LCUT}.log" 2>&1
  echo "$(date '+%T') DONE  npe mask=$A fid=$FT lcut=$LCUT exit=$?" >> "$STATUS"
}

# Job list passed as args ("AREA:FIDTYPE" ...); default = the 5 clean masks both scenarios
# (10000 baryonified already produced by the validation test).
JOBS=("$@")
if [ ${#JOBS[@]} -eq 0 ]; then
  JOBS=("10000:nobaryons" \
        "2000:nobaryons" "2000:baryonified" \
        "5000:nobaryons" "5000:baryonified" \
        "28000:nobaryons" "28000:baryonified" \
        "35000:nobaryons" "35000:baryonified")
fi

echo "$(date '+%T') ===== NPE FANOUT START (${#JOBS[@]} jobs) =====" >> "$STATUS"
i=0
for job in "${JOBS[@]}"; do
  A=${job%%:*}; FT=${job##*:}; G=$((i % 3))
  run_npe "$A" "$FT" "$G" &
  i=$((i + 1))
  while [ "$(jobs -r | wc -l)" -ge 3 ]; do wait -n; done
done
wait
echo "$(date '+%T') ===== NPE FANOUT COMPLETE =====" >> "$STATUS"
