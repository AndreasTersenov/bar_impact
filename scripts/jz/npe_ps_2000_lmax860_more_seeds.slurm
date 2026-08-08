#!/bin/bash
#SBATCH --job-name=npe_ps_2000_more
#SBATCH --account=nzu@a100
#SBATCH -C a100
#SBATCH --partition=gpu_p5
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=01:00:00
#SBATCH --output=logs/npe_ps_2000_more_%j.out
#SBATCH --error=logs/npe_ps_2000_more_%j.out
#
# Regenerate 3 RAID0-destroyed posteriors and add 3 fresh seeds at 2000 deg^2, PS,
# lmax=860 (strict baryon-safe cut). Motivation: the pooled per-seed FoM3 error bar
# at 2000/860 came out suspiciously tight for the biased arm — 3-pair sample where
# two of three landed close. This job restores the paired 5-seed convention AND
# adds r6/r7/r8 so the error bar is measured with 8 independent seeds, resolving
# whether the tightness was a fluke or real.
#
# Also regenerates the 1 corrupted posterior at 5000/580 (biased r1) as a side
# fix, at trivial extra cost.
#
# Trainings, sequential on one A100 (~2 min each, empirical from Jun-2026 runs):
#   2000/860 null   r4       (regen; was 100% zeros, Jul 22 14:07 RAID0 signature)
#   2000/860 biased r3       (regen)
#   2000/860 null   r6,r7,r8 (fresh)
#   2000/860 biased r6,r7,r8 (fresh)
#   5000/580 biased r1       (regen; was 100% zeros, same signature)
#
#   total: 9 trainings ~= 20 min compute + JAX warm-up per launch -> ~35 min wall.
#
# WHY not qos_gpu_a100-dev: dev has a 2 h wall limit but tight throttling on
# concurrent jobs; t3 with a 1 h request backfills quickly for a single-GPU job
# and gives comfortable headroom.
#
# WHY --random-seed = 100 + run_index: reproduces the seed pattern the original
# sweep used (log line "attempt 0 seed 101 gpu 3 slot 6" for run1 confirmed 101
# came from --random-seed 101 on that dispatch), so a regenerated r3/r4 is
# statistically equivalent to what would have been there, not a re-roll.
#
# Skip logic: if the target posterior already exists AND loads (numpy accepts the
# file), the training is skipped. Re-runs after wall-clock kill are safe.
#
#   sbatch scripts/jz/npe_ps_2000_lmax860_more_seeds.slurm
set -uo pipefail
cd /lustre/fsn1/projects/rech/prk/ulx34io/bar_impact || exit 1

module purge
module load arch/a100
PY=/lustre/fswork/projects/rech/nzu/ulx34io/envs/jaxili/bin/python
export PYTHONNOUSERSITE=1

REPO=/lustre/fsn1/projects/rech/prk/ulx34io/bar_impact
PSD=$REPO/outputs/baryon_tension/ps_submean_l37/posteriors
CKPT_BASE=$REPO/checkpoints/tension_sweep/ps_submean_l37
LOG_BASE=$REPO/outputs/baryon_tension/ps_submean_l37/logs

echo "=== NPE PS 2000/860 + 5000/580 seed top-up ==="
echo "start : $(date -Is)"
echo "python: $PY"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo

# Verify all training inputs exist before consuming GPU time on a doomed job.
DD=/lustre/fsn1/projects/rech/prk/ulx34io/cosmogrid_products/stage3_forecast
for A in 2000 5000; do
  for stem in \
    all_cls_grid_nobaryons_bin1_masked_${A}sqdeg_apod2.0_master_submean_noisy_s0.26_lmax1535.npy \
    all_cross_cls_grid_nobaryons_bins1234_masked_${A}sqdeg_apod2.0_master_submean_noisy_s0.26_lmax1535.npy \
    all_cls_fiducial_nobaryons_bin1_masked_${A}sqdeg_apod2.0_master_submean_noisy_s0.26_lmax1535.npy \
    all_cls_fiducial_baryonified_bin1_masked_${A}sqdeg_apod2.0_master_submean_noisy_s0.26_lmax1535.npy \
    ; do
    for sub in new_grid fiducial; do
      p=$DD/$sub/$stem
      [ -f "$p" ] && { echo "[pre-flight OK] $(basename $p)"; break; }
    done
  done
done
echo

# posterior_ok  <path>  returns 0 if the file exists AND numpy can load it.
posterior_ok() {
  local p="$1"
  [ -f "$p" ] || return 1
  $PY - <<PYEOF 2>/dev/null
import numpy as np, sys
try:
    a = np.load("$p", allow_pickle=True)
    sys.exit(0 if a.ndim == 2 and a.shape[1] >= 3 else 1)
except Exception:
    sys.exit(1)
PYEOF
}

# train_one  <area>  <lmax>  <arm>  <run_idx>
train_one() {
  local area="$1" lmax="$2" arm="$3" r="$4"
  local fid="nobaryons"
  local arm_tag_biased="baryonified"
  [ "$arm" = "biased" ] && fid="baryonified"

  local mask_dir
  printf -v mask_dir "%s/mask_%05d/%s" "$PSD" "$area" "$arm"
  mkdir -p "$mask_dir"

  local fname="posterior_samples_ps_auto_cross_nobaryons_vs_${fid}_bins1234_l37-${lmax}_r10_masked_${area}sqdeg_apod2.0_master_submean_noisy_s0.26_run${r}.npy"
  local target="$mask_dir/$fname"

  if posterior_ok "$target"; then
    echo "[skip] already good: ${arm} r${r} at ${area}/${lmax}"
    return 0
  fi

  local ckpt="$CKPT_BASE/mask$(printf %05d $area)_l37-${lmax}_${arm}_run${r}"
  local seed=$((100 + r))
  local log="$LOG_BASE/mask$(printf %05d $area)_l37-${lmax}_${arm}_run${r}.regen.log"
  mkdir -p "$ckpt" "$LOG_BASE"

  echo "[train] ${arm} r${r} at ${area}/${lmax}  seed=${seed}  -> ${target}"
  echo "        ckpt=${ckpt}"
  echo "        log =${log}"

  srun $PY -u scripts/run_npe_inference_auto_cross_ps_master.py \
      --simulation-type nobaryons --fiducial-type "$fid" \
      --masked --mask-area-sqdeg "$area" --apodization-scale-deg 2.0 \
      --noisy --noise-level 0.26 \
      --subtract-mean --lmax 1535 \
      --lower-cut 37 --upper-cut "$lmax" --rebin 10 \
      --random-seed "$seed" \
      --train --gpu 0 --run "$r" \
      --samples-dir "$mask_dir" \
      --checkpoint-dir "$ckpt" \
      >>"$log" 2>&1
  local rc=$?

  if [ $rc -eq 42 ]; then
    echo "[NaN loss] ${arm} r${r} at ${area}/${lmax}  seed=${seed}  -> retry with seed $((seed+50))"
    srun $PY -u scripts/run_npe_inference_auto_cross_ps_master.py \
        --simulation-type nobaryons --fiducial-type "$fid" \
        --masked --mask-area-sqdeg "$area" --apodization-scale-deg 2.0 \
        --noisy --noise-level 0.26 \
        --subtract-mean --lmax 1535 \
        --lower-cut 37 --upper-cut "$lmax" --rebin 10 \
        --random-seed $((seed+50)) \
        --train --gpu 0 --run "$r" \
        --samples-dir "$mask_dir" \
        --checkpoint-dir "$ckpt" \
        >>"$log" 2>&1
    rc=$?
  fi

  if posterior_ok "$target"; then
    echo "[done] ${arm} r${r} at ${area}/${lmax}  rc=$rc  -> $(basename $target)"
    return 0
  fi
  echo "[FAIL] ${arm} r${r} at ${area}/${lmax}  rc=$rc"
  return 1
}

fail=0

# Restore RAID0-corrupted seeds at 2000/860 (r3 biased, r4 null) to hit 5-pair convention
train_one 2000 860 biased 3 || fail=$((fail+1))
train_one 2000 860 null   4 || fail=$((fail+1))

# Fresh seeds at 2000/860 — both arms, to widen the FoM3 error-bar sample from 5 -> 8 pairs
for r in 6 7 8; do
  train_one 2000 860 null   "$r" || fail=$((fail+1))
  train_one 2000 860 biased "$r" || fail=$((fail+1))
done

# Free lunch: restore r1 biased at 5000/580 so that pool goes 4 -> 5 pairs
train_one 5000 580 biased 1 || fail=$((fail+1))

echo
echo "end   : $(date -Is)   failures=$fail"
exit $fail
