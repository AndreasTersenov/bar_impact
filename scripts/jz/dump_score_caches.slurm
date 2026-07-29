#!/bin/bash
#SBATCH --job-name=dump_score_caches
#SBATCH --account=prk@cpu
#SBATCH --partition=cpu_p1
#SBATCH --qos=qos_cpu-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=04:00:00
#SBATCH --output=logs/jz/dump_score_caches_%j.out
#SBATCH --error=logs/jz/dump_score_caches_%j.out
#
# Regenerate the score-experiment caches destroyed by the Titan disk failure.
#
# CPU, not GPU, on purpose: run_npe_inference_auto_cross_ps_master.py returns
# immediately after writing cache.npz (line 971-983) — the dump path is pure
# numpy load/cut/rebin/save and never touches the NDE. No GPU hours needed.
# (jax is imported at module scope, so the env is still required; it just runs on CPU.)
#
#   sbatch scripts/jz/dump_score_caches.slurm
#
# Idempotent: any (tag, area) whose cache.npz is *readable* is skipped, so a
# re-submit after a timeout resumes rather than redoing work.
#
# NB the skip test opens the zip — it does NOT just check the file exists. The 20
# pre-existing caches here are 13.8 MB each but zero-filled by the disk failure, so
# an existence/size test would skip every one of them and silently leave corrupt
# input in place.

set -uo pipefail

REPO=/lustre/fsn1/projects/rech/prk/ulx34io/bar_impact
PY=/lustre/fswork/projects/rech/nzu/ulx34io/envs/jaxili/bin/python
DD=/lustre/fsn1/projects/rech/prk/ulx34io/cosmogrid_products/stage3_forecast
CACHE=$REPO/outputs/score_experiment/cache

cd "$REPO" || exit 1
mkdir -p "$CACHE" logs/jz

# keep JAX on CPU and stop it grabbing memory it will not use
export JAX_PLATFORMS=cpu
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

# Matches run_score_npe_sweep.sh COMMON exactly. --subtract-mean is what makes
# these the *_submean vectors; lower-cut 37 is the recovered low-ell floor.
COMMON="--simulation-type nobaryons --masked --apodization-scale-deg 2.0 \
        --noisy --noise-level 0.26 --subtract-mean --lmax 1535 --lower-cut 37 --rebin 20 \
        --data-dir $DD"

echo "=== score-cache regeneration ==="
echo "host   : $(hostname)"
echo "started: $(date -u +%FT%TZ)"
echo "data   : $DD"
echo

ok=0; skip=0; fail=0
for A in 2000 5000 10000 14000 28000 35000; do
  for spec in "nonbnt_460|--upper-cut 460" \
              "bnt_580|--bnt --bnt-bins 0,1,2,3 --upper-cuts 580,1024,1024,1024" \
              "nonbnt_full|--upper-cut 1024" \
              "bnt_full|--bnt --bnt-bins 0,1,2,3 --upper-cuts 1024,1024,1024,1024"; do
    tag=${spec%%|*}; flags=${spec#*|}

    # PLAN_bnt_vmim_v2.md needs BOTH fiducials: nobaryons = the null,
    # baryonified = the biased observation the tension is measured against.
    for fid in nobaryons baryonified; do
      sfx=$([ "$fid" = "nobaryons" ] && echo nobary || echo bary)
      out="$CACHE/${tag}_${A}_${sfx}"

      # readable-zip test, not an existence test — see header note
      if [ -f "$out/cache.npz" ] && "$PY" -c "
import sys, zipfile
try:
    z = zipfile.ZipFile(sys.argv[1])
    sys.exit(1 if z.testzip() else 0)
except Exception:
    sys.exit(1)" "$out/cache.npz" 2>/dev/null; then
        echo "[skip] $tag $A $sfx (existing cache is readable)"; skip=$((skip+1)); continue
      fi
      # stale/corrupt cache present -> remove so the dump starts clean
      [ -f "$out/cache.npz" ] && { echo "  (replacing corrupt cache)"; rm -f "$out/cache.npz"; }

      echo "[dump] $tag $A $sfx"
      if srun --ntasks=1 "$PY" -u scripts/run_npe_inference_auto_cross_ps_master.py \
            $COMMON --mask-area-sqdeg ${A}.0 --fiducial-type "$fid" \
            $flags --dump-cache "$out" 2>&1 | tail -3; then
        if [ -s "$out/cache.npz" ]; then ok=$((ok+1))
        else echo "  !! no cache.npz written"; fail=$((fail+1)); fi
      else
        echo "  !! worker failed"; fail=$((fail+1))
      fi
    done
  done
done

echo
echo "=== done $(date -u +%FT%TZ) ==="
echo "written: $ok   skipped: $skip   failed: $fail"
echo
echo "verify with:"
echo "  python3 /lustre/fsn1/projects/rech/prk/ulx34io/recovery_handoff/check_integrity.py \\"
echo "     <<< \"\$(ls $CACHE/*/cache.npz)\""
