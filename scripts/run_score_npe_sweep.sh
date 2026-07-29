#!/usr/bin/env bash
# Six-footprint rollout of the calibrated score-compression NPE (plan jolly-toasting-robin.md).
# Per footprint x {non-BNT-460, BNT-580}: dump-cache -> score-compress (MLE, hybrid cov) -> NPE 5 seeds (+TARP/SBC).
# Analytic covs (gaussian_cov_native_<A>.npy) must already exist (built in cosmostat_new). Runs in jaxili.
set -u
ROOT=/lustre/fsn1/projects/rech/prk/ulx34io/bar_impact
PY=/home/tersenov/anaconda3/envs/jaxili/bin/python
CACHE=$ROOT/outputs/score_experiment/cache
SC=$ROOT/outputs/score_experiment/score
NPE=$ROOT/outputs/score_experiment/npe_score
mkdir -p "$CACHE" "$SC" "$NPE"
export OMP_NUM_THREADS=4
COMMON="--simulation-type nobaryons --fiducial-type nobaryons --masked --apodization-scale-deg 2.0 \
  --noisy --noise-level 0.26 --subtract-mean --lmax 1535 --lower-cut 37 --rebin 20"

for A in 2000 5000 10000 14000 28000 35000; do
  echo "############ AREA $A ############"
  # ---- dump-cache (skip if present) ----
  for spec in "nonbnt_460|--upper-cut 460" "bnt_580|--bnt --bnt-bins 0,1,2,3 --upper-cuts 580,1024,1024,1024"; do
    tag=${spec%%|*}; flags=${spec#*|}
    d=$CACHE/${tag}_${A}_nobary
    if [ -f "$d/cache.npz" ]; then echo "[dump] $tag $A exists, skip"; else
      echo "[dump] $tag $A"
      $PY scripts/run_npe_inference_auto_cross_ps_master.py $COMMON --mask-area-sqdeg ${A}.0 \
        $flags --dump-cache "$d" >/dev/null 2>&1 || { echo "DUMP FAIL $tag $A"; continue; }
    fi
  done
  # ---- score-compress (MLE, hybrid) — sets FISHER_AREA internally ----
  echo "[score] $A"
  $PY scripts/score_compress.py $A hybrid 2>&1 | grep -E "compressed \(|MOPED" | sed "s/^/   /"
  # ---- NPE on score summaries, 5 seeds, both configs ----
  for tag in nonbnt_460 bnt_580; do
    echo "[npe] $tag $A"
    $PY scripts/npe_on_summary.py --compressed $SC/compressed_${tag}_${A}_hybrid.npz \
      --out $NPE --tag ${tag}_${A}_mle --seeds 41,42,43,44,45 --epochs 300 --gpu 0 \
      2>&1 | grep -E "TARP|SBC|NPE OK|no usable" | sed "s/^/   /"
  done
done
echo "############ ROLLOUT DONE ############"
