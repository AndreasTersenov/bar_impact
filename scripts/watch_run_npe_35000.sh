#!/bin/bash
# When the 35000 monopole-subtracted PS grid finishes, run the 4 PS NPE cuts (l37/l100 x 1024/400),
# masked 35000, nobaryons_vs_nobaryons, submean/lmax1535. HOS (l1 & peaks, scales1234/234, masked 35001)
# already exist, so only the PS is generated here. Free-GPU pick, sequential.
set -u
BASE=/lustre/fsn1/projects/rech/prk/ulx34io/cosmogrid_products/stage3_forecast
GRIDCROSS=$BASE/new_grid/all_cross_cls_grid_nobaryons_bins1234_masked_35000sqdeg_apod2.0_master_submean_noisy_s0.26_lmax1535.npy
PY=/home/tersenov/anaconda3/envs/jaxili/bin/python
REPO=/lustre/fsn1/projects/rech/prk/ulx34io/bar_impact
OUT=$REPO/outputs/diagnostics/lmin_compare/masked_35000
LOG=$OUT/watch.log
mkdir -p "$OUT"; cd "$REPO"
pick_gpu() { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1 | tr -d ' '; }

echo "$(date '+%F %T') waiting for 35000 PS submean grid" >> "$LOG"
until [ -f "$GRIDCROSS" ]; do sleep 120; done
sleep 60
echo "$(date '+%F %T') grid ready -> running 4 PS NPE cuts" >> "$LOG"
for CUT in "37 1024" "100 1024" "37 400" "100 400"; do
  set -- $CUT; LC=$1; UC=$2; GPU=$(pick_gpu)
  echo "$(date '+%F %T') START l${LC}-${UC} gpu=$GPU" >> "$LOG"
  $PY scripts/run_npe_inference_auto_cross_ps_master.py \
    --simulation-type nobaryons --fiducial-type nobaryons \
    --masked --mask-area-sqdeg 35000 --apodization-scale-deg 2.0 \
    --noisy --noise-level 0.26 --subtract-mean --lmax 1535 \
    --lower-cut "$LC" --upper-cut "$UC" --rebin 10 \
    --train --gpu "$GPU" --samples-dir "$OUT" \
    > "$OUT/npe_l${LC}-${UC}.log" 2>&1
  echo "$(date '+%F %T') DONE  l${LC}-${UC} exit=$?" >> "$LOG"
done
touch "$OUT/DONE"
echo "$(date '+%F %T') ===== ALL 35000 PS NPE DONE =====" >> "$LOG"
