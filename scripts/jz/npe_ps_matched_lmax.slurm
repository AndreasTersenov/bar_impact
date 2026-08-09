#!/bin/bash
#SBATCH --job-name=npe_ps_matched
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
#SBATCH --output=logs/npe_ps_matched_%j.out
#SBATCH --error=logs/npe_ps_matched_%j.out
#
# PS NPE at an arbitrary upper cut on a MASKED footprint, both roles, 5 seeds.
#
# WHY. The masked PS campaign only ever ran cuts from lmax=340 upward, because that is where
# the baryon-safe cut lives. Comparing the PS against a WAVELET scale set on a matched ell
# range needs cuts well below that: the starlet bands measured in
# paper/figures/starlet_scale_ell/values.csv are
#
#     wav0  ell 364-1535     wav1  ell 142-336     wav2  ell 71-168     wav3  ell 36-84
#
# and the file tag is 1-based, so the tag "scales34" = wav2 + wav3 = ell 36-168. With the PS
# already starting at lmin=37 that is almost exactly l37-168; 180 is used because it is the
# nearest cut on the full-sky grid, which keeps the two footprints on the same value. (For
# reference the scales234 equivalent is ell 36-336, and the Fisher baryon-safe figure paired
# that with PS lmax=400 -- a looser round-up than the 180-vs-168 here.)
#
# The half-power ranges OVERLAP, so "scales34 covers 36-168" is where those bands dominate,
# not a sharp window -- see the caveats in that figure's provenance. And the bands were
# measured on the full sphere; a mask couples multipoles, so the effective coverage of the
# masked maps is somewhat broader than the table implies. The matched cut is therefore a
# well-motivated approximation, not an identity.
#
#   AREA=14000 LMAX=180 sbatch scripts/jz/npe_ps_matched_lmax.slurm
#
# Both roles are trained because plot_contours_three_stats.py pairs null/biased runs and drops
# a run unreadable on either side, even when only the biased arm is drawn.
set -uo pipefail
cd /lustre/fsn1/projects/rech/prk/ulx34io/bar_impact || exit 1

module purge
module load arch/a100
PY=/lustre/fswork/projects/rech/nzu/ulx34io/envs/jaxili/bin/python
export PYTHONNOUSERSITE=1

AREA="${AREA:?set AREA, e.g. 14000 (the nominal masked area, NOT the +1 HOS tag)}"
LMAX="${LMAX:?set LMAX, e.g. 180}"
RUNS="${RUNS:-1 2 3 4 5}"

REPO=/lustre/fsn1/projects/rech/prk/ulx34io/bar_impact
DD=/lustre/fsn1/projects/rech/prk/ulx34io/cosmogrid_products/stage3_forecast
PSD=$REPO/outputs/baryon_tension/ps_submean_l37/posteriors
CKPT_BASE=$REPO/checkpoints/tension_sweep/ps_submean_l37
LOG_BASE=$REPO/outputs/baryon_tension/ps_submean_l37/logs

# Every masked SUBMEAN product is tagged lmax1535, including 14000. The 1530 figure that
# appears in submit_npe_inference_parameter_sweep_parallel.py applies to the older non-submean
# 14000 spectra only -- assuming it here made the pre-flight fail on a file that never existed.
SPEC_LMAX=1535

echo "=== PS NPE, area ${AREA} deg^2, l37-${LMAX}, r10, both roles, runs: ${RUNS} ==="
echo "start : $(date -Is)   spectra lmax=${SPEC_LMAX}"
nvidia-smi --query-gpu=name --format=csv,noheader
echo

probe=$DD/new_grid/all_cls_grid_nobaryons_bin1_masked_${AREA}sqdeg_apod2.0_master_submean_noisy_s0.26_lmax${SPEC_LMAX}.npy
[ -f "$probe" ] || { echo "[fatal] missing grid: $probe"; exit 1; }
echo "[pre-flight] $(basename "$probe")"
echo

posterior_ok() {
  [ -f "$1" ] || return 1
  "$PY" - "$1" <<'PYEOF' 2>/dev/null
import sys, numpy as np
try:
    a = np.load(sys.argv[1], allow_pickle=True)
    s = float(np.sqrt(np.cov(a[:, :3], rowvar=False)[1, 1]))
    sys.exit(0 if (a.ndim == 2 and a.shape[1] >= 3 and s < 0.08) else 1)
except Exception:
    sys.exit(1)
PYEOF
}

done_n=0; skip_n=0; fail_n=0
for role in null biased; do
  fid=nobaryons; [ "$role" = "biased" ] && fid=baryonified
  mask_dir=$(printf "%s/mask_%05d/%s" "$PSD" "$AREA" "$role")
  mkdir -p "$mask_dir"
  for r in $RUNS; do
    out="$mask_dir/posterior_samples_ps_auto_cross_nobaryons_vs_${fid}_bins1234_l37-${LMAX}_r10_masked_${AREA}sqdeg_apod2.0_master_submean_noisy_s0.26_run${r}.npy"
    if posterior_ok "$out"; then
      echo "[skip] $role run$r"; skip_n=$((skip_n+1)); continue
    fi
    [ -f "$out" ] && { echo "  (replacing unreadable $out)"; rm -f "$out"; }
    ckpt="$CKPT_BASE/mask$(printf %05d "$AREA")_l37-${LMAX}_${role}_run${r}"
    log="$LOG_BASE/mask$(printf %05d "$AREA")_l37-${LMAX}_${role}_run${r}.log"
    mkdir -p "$ckpt" "$LOG_BASE"
    echo "[run ] $role run$r  seed=$((100+r))  ($(date -Is))"
    srun "$PY" -u scripts/run_npe_inference_auto_cross_ps_master.py \
        --simulation-type nobaryons --fiducial-type "$fid" \
        --masked --mask-area-sqdeg "$AREA" --apodization-scale-deg 2.0 \
        --noisy --noise-level 0.26 \
        --subtract-mean --lmax "$SPEC_LMAX" \
        --lower-cut 37 --upper-cut "$LMAX" --rebin 10 \
        --random-seed $((100+r)) \
        --train --gpu 0 --run "$r" \
        --samples-dir "$mask_dir" --checkpoint-dir "$ckpt" \
        >>"$log" 2>&1
    rc=$?
    if posterior_ok "$out"; then
      echo "[done] $role run$r"; done_n=$((done_n+1))
    else
      echo "[FAIL] $role run$r rc=$rc  (see $log)"; fail_n=$((fail_n+1))
    fi
  done
done

echo
echo "=== summary ==="
echo "trained $done_n   skipped $skip_n   failed $fail_n"
echo "end   : $(date -Is)"
[ $fail_n -eq 0 ] || exit 1
