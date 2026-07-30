#!/bin/bash
#SBATCH --job-name=npe_bsafe
#SBATCH --account=nzu@a100
#SBATCH -C a100
#SBATCH --partition=gpu_p5
#SBATCH --qos=qos_gpu_a100-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=01:00:00
#SBATCH --output=logs/npe_bsafe_%j.out
#SBATCH --error=logs/npe_bsafe_%j.out
#
# NPE at the BARYON-SAFE scale cut for the higher-order statistics, 14000 deg^2, submean.
#
# WHY THIS EXISTS. The paper needs NPE contours for all three statistics at their required
# scale cuts, in the corrected convention (PS lmin=37; peaks/l1 on zero-mean maps). Only
# the PS side exists:
#   PS    lmax cuts, submean, lmin=37 : 5 seeds x 6 footprints x 18 cuts   -> HAVE
#   peaks scales234 submean           : 1 seed, and 3 of 12 files destroyed -> INSUFFICIENT
#   l1    scales234 submean           : 305 scales234 posteriors exist, ZERO submean -> ABSENT
# l1 at a reduced scale set was never rerun after the submean correction. This job closes
# that gap at the reference footprint with a 5-seed ensemble, matching the PS.
#
# scales234 == --scales "1,2,3": the file tag is 1-based (idx+1) while --scales is 0-based,
# so idx [1,2,3] -> "scales234". idx 0 is the finest wavelet scale (dropped here, it is the
# baryon-contaminated one); idx 4 (coarse/mass-sheet) is excluded throughout the analysis.
#
# PREREQUISITE, already applied: run_npe_peak_counts_inference.py on this branch had NO
# --submean flag AND a dead NaN-retry (it tested hasattr(metrics,'val_loss'), but jaxili
# returns a dict keyed 'val/loss', so the check was always False and a NaN validation loss
# silently produced a collapsed prior-width posterior reported as successful). Both were
# fixed on the unmerged branch collect-uncommitted-bar-2026-07-20 for the l1 runner only;
# they are now ported here. Without that port this job would quietly emit collapsed seeds.
#
# 20 trainings = 2 statistics x 2 roles x 5 seeds. Checkpointed per run; reruns skip work
# that already produced a posterior, so requeueing after the wall clock is safe.
#
# WALLTIME. A 20 h request bought a 34 h queue wait on a full A100 partition, so keep it
# short. But the first sizing was too optimistic: log-timestamp spacing from an old sweep
# suggested ~1.7 min/training, while a measured run here is ~8 min (1000 epochs at ~2.4
# it/s plus setup), i.e. ~2.7 h for all 20. Submit 1 h to get scheduled quickly and chain a
# 3 h follow-up with --dependency=afterany:<first>; the skip logic means the second job only
# does what the first did not reach. Do NOT size this from filename spacing again -- read
# the epoch rate out of a live log.
#
# Submit from the repo root:  sbatch scripts/jz/npe_hos_baryonsafe_14000.slurm

set -uo pipefail

cd /lustre/fsn1/projects/rech/prk/ulx34io/bar_impact || exit 1

module purge
module load arch/a100
PY=/lustre/fswork/projects/rech/nzu/ulx34io/envs/jaxili/bin/python

# The higher-order footprint tag IS 14001, and --mask-area-sqdeg is used VERBATIM to build
# both the input datavector names and the output posterior names -- the runner does not add
# the +1 itself. Job 426843 passed 14000 here and all 20 trainings died instantly with
# FileNotFoundError on all_l1_norms_grid_..._masked_14000sqdeg_submean_... (only the 14001
# files exist). Pass the tag, not the nominal area.
AREA_TAG=14001             # peaks/l1 datavector + posterior tag for the 14000 deg^2 footprint
SCALES="1,2,3"             # -> the "scales234" file tag
RUNS="1 2 3 4 5"
DD=/lustre/fsn1/projects/rech/prk/ulx34io/cosmogrid_products/stage3_forecast
SAMPLES=/lustre/fsn1/projects/rech/prk/ulx34io/bar_impact/outputs/samples
CKPT=/lustre/fsn1/projects/rech/prk/ulx34io/bar_impact/checkpoints/npe_bsafe_14000

mkdir -p "$CKPT"

echo "=== NPE baryon-safe HOS, tag ${AREA_TAG} (14000 deg2 footprint), scales234, submean ==="
echo "start : $(date -Is)"
echo "python: $PY"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo

done_n=0; skip_n=0; fail_n=0

for stat in l1 peaks; do
  case "$stat" in
    l1)    SCRIPT=scripts/run_npe_inference.py            ; TAG=""     ;;
    peaks) SCRIPT=scripts/run_npe_peak_counts_inference.py; TAG="pc_"  ;;
  esac

  for role in nobaryons baryonified; do
    # role names the FIDUCIAL (the observed data vector); the model is always nobaryons-trained.
    for r in $RUNS; do
      out="${SAMPLES}/posterior_samples_${TAG}nobaryons_vs_${role}_bins1234_scales234_noisy_s0.26_masked_${AREA_TAG}sqdeg_submean_new_normalization_run${r}_npe.npy"

      # Skip only if the posterior exists AND loads: a disk-damaged .npy raises ValueError
      # (numpy reads the mangled header as pickled data), so -f alone would skip a corrupt file.
      if [ -f "$out" ] && "$PY" -c "
import sys, numpy as np
try:
    a = np.load(sys.argv[1])
    sys.exit(0 if a.ndim == 2 and a.shape[0] > 0 else 1)
except Exception:
    sys.exit(1)" "$out" 2>/dev/null; then
        echo "[skip] $stat $role run$r (readable posterior already present)"
        skip_n=$((skip_n+1)); continue
      fi
      [ -f "$out" ] && { echo "  (replacing unreadable $out)"; rm -f "$out"; }

      echo "[run ] $stat $role run$r   ($(date -Is))"
      srun "$PY" -u "$SCRIPT" \
        --data-dir "$DD" \
        --simulation-type nobaryons \
        --fiducial-type "$role" \
        --bins 1,2,3,4 \
        --scales "$SCALES" \
        --noisy --noise-level 0.26 \
        --masked --mask-area-sqdeg "$AREA_TAG" --submean \
        --new-normalization \
        --train --epochs 1000 \
        --num-samples 3000 \
        --run "$r" --random-seed "$r" \
        --checkpoint-dir "$CKPT" \
        --samples-dir "$SAMPLES"
      rc=$?

      if [ $rc -ne 0 ]; then
        echo "[FAIL] $stat $role run$r rc=$rc"; fail_n=$((fail_n+1)); continue
      fi
      if [ ! -f "$out" ]; then
        echo "[FAIL] $stat $role run$r exited 0 but wrote no posterior at:"
        echo "       $out"
        fail_n=$((fail_n+1)); continue
      fi
      # A collapsed posterior is the silent failure this analysis guards against downstream
      # (sigma_S8 >= 0.08 => prior-width). Flag it here rather than discovering it in a figure.
      "$PY" -c "
import sys, numpy as np
a = np.load(sys.argv[1])
s = float(np.sqrt(np.cov(a[:, :3], rowvar=False)[1, 1]))
print(f'       sigma_S8={s:.4f}' + ('   *** COLLAPSED (>=0.08) ***' if s >= 0.08 else ''))
" "$out"
      done_n=$((done_n+1))
    done
  done
done

echo
echo "=== summary ==="
echo "trained $done_n   skipped $skip_n   failed $fail_n"
echo "end   : $(date -Is)"
echo
echo "Next: build the figure with"
echo "  scripts/diagnostics/plot_contours_three_stats.py --area 14000 --cut-mode baryon-safe"
echo "(the --cut-mode leg still needs writing; it must pair PS at its 0.3-sigma lmax with"
echo " these scales234 HOS posteriors)"
[ $fail_n -eq 0 ] || exit 1
