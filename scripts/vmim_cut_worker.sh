#!/bin/bash
# One (cut, config, compressor-seed) job: retrain the compressor on the cut data vector, then NDE
# with several seeds -> per-NDE-seed null+biased. args:
#   $1=cut  $2=config(nonbnt|bnt)  $3=comp_seed  $4=gpu  $5=outroot  $6=nde_seeds(comma)
set -e
cut=$1; config=$2; cseed=$3; gpu=$4; root=$5; seeds=$6
cd /mnt/home/tersenov/software/bar_impact
PY=/home/tersenov/anaconda3/envs/jaxili/bin/python
C=outputs/score_experiment/cache
if [ "$config" = "nonbnt" ]; then cuts="$cut,$cut,$cut,$cut"; arm=nonbnt;
else cuts="$cut,1024,1024,1024"; arm=bnt; fi
od=$root/${config}_c$cut/cs$cseed
mkdir -p $od

FISHER_AREA=14000 XLA_PYTHON_CLIENT_MEM_FRACTION=0.2 $PY scripts/vmim_compress.py \
  --cache $C/${arm}_full_14000_nobary/cache.npz \
  --biased-cache $C/${arm}_full_14000_bary/cache.npz \
  --out $od/comp --preproc ana_whiten --analytic-cov $arm --cuts "$cuts" \
  --lr 5e-4 --steps 20000 --val-every 500 --max-minutes 12 --seed $cseed --split-seed 0 \
  --gpu $gpu > $od/log_comp.txt 2>&1

$PY scripts/nde_realnvp_from_summary.py --compressed $od/comp \
  --out $od/nde --tag $config --seeds $seeds --total-steps 20000 --save-every 1500 \
  --patience 15 --num-samples 3000 --save-per-seed --no-tarp \
  --gpu $gpu --mem-fraction 0.2 > $od/log_nde.txt 2>&1
echo "DONE $config c$cut cs$cseed"
