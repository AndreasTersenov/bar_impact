#!/bin/bash
# Wait for the 14000 monopole-subtracted nlb=4 GRID to finish, then run the NPE l37-vs-l100
# comparison (paper footprint): nobaryons_vs_nobaryons (constraint) + nobaryons_vs_baryonified (bias),
# at lower-cut 37 and 100, on the submean/lmax1535 data. Sequential, free-GPU pick, isolated samples dir.
set -u
BASE=/home/tersenov/CosmoGridV1/stage3_forecast
GRIDCROSS=$BASE/new_grid/all_cross_cls_grid_nobaryons_bins1234_masked_14000sqdeg_apod2.0_master_submean_noisy_s0.26_lmax1535.npy
PY=/home/tersenov/anaconda3/envs/jaxili/bin/python
REPO=/mnt/home/tersenov/software/bar_impact
SAMP=$REPO/outputs/diagnostics/lmin_compare/samples_14000
LOG=$REPO/outputs/diagnostics/lmin_compare/watch_14000.log
mkdir -p "$SAMP"
cd "$REPO"

echo "$(date '+%F %T') waiting for 14000 submean grid: $GRIDCROSS" >> "$LOG"
until [ -f "$GRIDCROSS" ]; do sleep 120; done
sleep 60   # let aggregation flush all auto bins + cross
echo "$(date '+%F %T') grid ready -> launching NPE jobs" >> "$LOG"

pick_gpu() { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1 | tr -d ' '; }

for CUT in 37 100; do
  for FID in nobaryons baryonified; do
    GPU=$(pick_gpu)
    echo "$(date '+%F %T') START l${CUT} fid=${FID} gpu=${GPU}" >> "$LOG"
    $PY scripts/run_npe_inference_auto_cross_ps_master.py \
        --simulation-type nobaryons --fiducial-type "$FID" \
        --masked --mask-area-sqdeg 14000 --apodization-scale-deg 2.0 \
        --noisy --noise-level 0.26 \
        --subtract-mean --lmax 1535 \
        --lower-cut "$CUT" --upper-cut 1024 --rebin 10 \
        --train --gpu "$GPU" --samples-dir "$SAMP" \
        > "$REPO/outputs/diagnostics/lmin_compare/npe_l${CUT}_${FID}.log" 2>&1
    echo "$(date '+%F %T') DONE  l${CUT} fid=${FID} exit=$?" >> "$LOG"
  done
done
echo "$(date '+%F %T') ===== ALL 14000 NPE DONE =====" >> "$LOG"
touch "$REPO/outputs/diagnostics/lmin_compare/DONE_14000"
