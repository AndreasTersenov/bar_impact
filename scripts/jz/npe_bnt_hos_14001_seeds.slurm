#!/bin/bash
#SBATCH --job-name=npe_bnt_hos_14001
#SBATCH --account=nzu@a100
#SBATCH -C a100
#SBATCH --partition=gpu_p5
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=logs/npe_bnt_hos_14001_%j.out
#SBATCH --error=logs/npe_bnt_hos_14001_%j.out
#
# 5-seed BNT NPE trainings at the 14000 deg^2 (mask tag 14001) footprint, for
# both L1 and peaks, baryonified arm, scales1234 (all four detail scales), NON-submean
# (matches the full-sky BNT arm convention in the current paper figures — there is
# no submean BNT L1/peaks training grid at 14001, and building one would need a
# ~thousands-of-maps CPU sweep, which we deliberately do NOT do here).
#
# WHY. To rebuild paper/figures/hos_bnt_{l1,peaks}_vs_scale_cut at 14000 deg^2 rather
# than full sky, the BNT arm is the bottleneck: only 1 posterior exists at 14001
# today. Pool over 5 seeds so the BNT contour width carries training scatter, like
# the other pooled 14000 figures.
#
# NO DATA-VECTOR REGENERATION. Fiducial and grid BNT datavectors already exist at
# 14001 for both statistics; this job only re-trains the flows. Verified with the
# pre-flight check below (refuses to launch if any bin is missing).
#
# TRAININGS: 2 stats x 1 arm x 5 seeds = 10 trainings. Empirical from similar HOS
# jobs: ~2-8 min each depending on early-stopping (worst case ~80 min total on
# one A100). Request 2 h in a100-dev to backfill quickly.
#
# Naming: the driver builds output filenames as
#   L1:    posterior_samples_nobaryons_vs_baryonified_bntbins1234_scales1234_
#          noisy_s0.26_masked_14001sqdeg_new_normalization_run{N}_npe.npy
#   peaks: posterior_samples_bnt_pc_nobaryons_vs_baryonified_bntbins1234_scales1234_
#          noisy_s0.26_masked_14001sqdeg_new_normalization_run{N}_npe.npy
# (peaks gets `bnt_pc_` prefix; L1 gets `bnt_` implicit via bntbins tag). The pre-
# existing single-seed files have no _runN suffix, so they don't collide.
#
#   sbatch scripts/jz/npe_bnt_hos_14001_seeds.slurm
set -uo pipefail
cd /lustre/fsn1/projects/rech/prk/ulx34io/bar_impact || exit 1

module purge
module load arch/a100
PY=/lustre/fswork/projects/rech/nzu/ulx34io/envs/jaxili/bin/python
export PYTHONNOUSERSITE=1

REPO=/lustre/fsn1/projects/rech/prk/ulx34io/bar_impact
DD=/lustre/fsn1/projects/rech/prk/ulx34io/cosmogrid_products/stage3_forecast
SAMPLES=$REPO/outputs/samples
CKPT_BASE=$REPO/checkpoints/bnt_hos_14001

AREA_TAG=14001
SCALES="0,1,2,3"       # all four detail scales; internal 0-based; tag scales1234
SCALES_TAG=scales1234
BNT_BINS="0,1,2,3"     # all four tomographic z-bins in BNT space; tag bntbins1234
BNT_BIN_TAG=bntbins1234
# Runs per statistic. L1 needs a larger pool because its BNT training frequently
# diverges to NaN on the first random init and the 20-attempt in-driver retry
# does not always recover (job 703621: only seeds 3 and 5 survived out of 5).
# Peaks does not have that pathology, so a smaller pool suffices there.
L1_RUNS="${L1_RUNS:-6 7 8 9 10 11 12 13 14 15}"
PEAKS_RUNS="${PEAKS_RUNS:-1 2 3 4 5}"

echo "=== NPE BNT HOS at 14001 (5 seeds x 2 stats) ==="
echo "start : $(date -Is)"
echo "python: $PY"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo

# PRE-FLIGHT: refuse to launch if any grid or fiducial datavector is missing. The
# driver only errors at grid load, ~15 s in, so a batch of doomed trainings would
# eat GPU time for nothing. The paths mirror what construct_paths() in each driver
# builds -- note the L1 driver's BNT branch reads grids from new_grid/, while the
# peaks driver's BNT branch reads from grid/. (For masked areas the actual files
# live under new_grid/; the grid/ path for BNT peaks is served by symlinks that
# point at the new_grid/ file. If the symlink is ever cleaned, this pre-flight
# will catch it before any GPU time is used.)
missing=0
for stat in l1 peaks; do
  case "$stat" in
    l1)    grid_stem="all_bnt_l1_norms_grid"    ; fid_stem="all_bnt_l1_norms_fiducial"    ; grid_dir="new_grid" ;;
    peaks) grid_stem="all_bnt_peak_counts_grid" ; fid_stem="all_bnt_peak_counts_fiducial" ; grid_dir="grid"     ;;
  esac
  for b in 1 2 3 4; do
    g=$DD/${grid_dir}/${grid_stem}_nobaryons_bin${b}_masked_${AREA_TAG}sqdeg_noisy_s0.26_new_normalization.npy
    f=$DD/fiducial/cosmo_fiducial/${fid_stem}_baryonified_bin${b}_masked_${AREA_TAG}sqdeg_noisy_s0.26_new_normalization.npy
    [ -e "$g" ] || { echo "[fatal] MISSING grid ($grid_dir): $g"; missing=$((missing+1)); }
    [ -f "$f" ] || { echo "[fatal] MISSING fid  : $f"; missing=$((missing+1)); }
  done
done
[ $missing -gt 0 ] && { echo "abort: $missing input(s) missing"; exit 1; }
echo "[pre-flight OK] all 16 BNT grid+fiducial inputs at ${AREA_TAG}sqdeg present"
echo

posterior_ok() {
  local p="$1"
  [ -f "$p" ] || return 1
  "$PY" - <<'PYEOF' "$p" 2>/dev/null
import sys, numpy as np
try:
    a = np.load(sys.argv[1])
    ok = a.ndim == 2 and a.shape[0] > 0 and a.shape[1] >= 3
    # prior-collapse guard (sig(S8) < 0.08), same threshold as the audit script
    import numpy as _np
    if ok:
        s = float(_np.sqrt(_np.cov(a[:, :3], rowvar=False)[1, 1]))
        ok = s < 0.08
    sys.exit(0 if ok else 1)
except Exception:
    sys.exit(1)
PYEOF
}

done_n=0; skip_n=0; fail_n=0

for stat in l1 peaks; do
  case "$stat" in
    l1)    SCRIPT=scripts/run_npe_inference.py            ; TAG=""       ; STAT_RUNS="$L1_RUNS"    ;;
    peaks) SCRIPT=scripts/run_npe_peak_counts_inference.py; TAG="bnt_pc_"; STAT_RUNS="$PEAKS_RUNS" ;;
  esac
  # L1 driver names the sample file with no leading tag (BNT is signalled by bntbins);
  # peaks driver prefixes bnt_pc_ when --bnt is set. See build_output_filename in each.

  for r in $STAT_RUNS; do
    out="${SAMPLES}/posterior_samples_${TAG}nobaryons_vs_baryonified_${BNT_BIN_TAG}_${SCALES_TAG}_noisy_s0.26_masked_${AREA_TAG}sqdeg_new_normalization_run${r}_npe.npy"

    if posterior_ok "$out"; then
      echo "[skip] $stat baryonified run$r (readable posterior sigma(S8)<0.08 present)"
      skip_n=$((skip_n+1)); continue
    fi
    [ -f "$out" ] && { echo "  (replacing unreadable/collapsed $out)"; rm -f "$out"; }

    ckpt="$CKPT_BASE/${stat}_bntbins1234_scales1234_${AREA_TAG}_run${r}"
    mkdir -p "$ckpt"

    echo "[run ] $stat baryonified run$r seed=$r ($(date -Is))"
    srun "$PY" -u "$SCRIPT" \
        --data-dir "$DD" \
        --simulation-type nobaryons \
        --fiducial-type baryonified \
        --bnt --bnt-bins "$BNT_BINS" \
        --scales "$SCALES" \
        --noisy --noise-level 0.26 \
        --masked --mask-area-sqdeg "$AREA_TAG" \
        --new-normalization \
        --train --epochs 1000 \
        --num-samples 3000 \
        --run "$r" --random-seed "$r" \
        --checkpoint-dir "$ckpt" \
        --samples-dir "$SAMPLES"
    rc=$?

    if [ $rc -ne 0 ]; then
      echo "[FAIL] $stat baryonified run$r rc=$rc"; fail_n=$((fail_n+1)); continue
    fi
    if [ ! -f "$out" ]; then
      echo "[FAIL] $stat baryonified run$r wrote no posterior at:"
      echo "       $out"
      fail_n=$((fail_n+1)); continue
    fi
    "$PY" -c "
import sys, numpy as np
a = np.load(sys.argv[1])
s = float(np.sqrt(np.cov(a[:, :3], rowvar=False)[1, 1]))
print(f'       sigma_S8={s:.4f}' + ('   *** COLLAPSED (>=0.08) ***' if s >= 0.08 else ''))
" "$out"
    done_n=$((done_n+1))
  done
done

echo
echo "=== summary ==="
echo "trained $done_n   skipped $skip_n   failed $fail_n"
echo "end   : $(date -Is)"
[ $fail_n -eq 0 ] || exit 1
