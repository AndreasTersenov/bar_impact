#!/bin/bash
#SBATCH --job-name=npe_bsafe_area
#SBATCH --account=nzu@a100
#SBATCH -C a100
#SBATCH --partition=gpu_p5
#SBATCH --qos=qos_gpu_a100-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=logs/npe_bsafe_area_%j.out
#SBATCH --error=logs/npe_bsafe_area_%j.out
#
# GENERALISED from npe_hos_baryonsafe_14000.slurm, which hardcoded the 14000 footprint. Fills
# the missing cells of the baryon-safe FoM table (the paper's Table 3) at any footprint.
#
# AREA_TAG is the HOS footprint tag, which is the nominal area PLUS ONE (2001, 5001, 10001,
# 14001, 28001, 35001). It is used VERBATIM to build both input and output names -- the runner
# does not add the +1 itself -- so passing 2000 here dies on FileNotFoundError. Required, with
# no default, precisely because a wrong-but-plausible value silently trains the wrong thing.
#
#   AREA_TAG=2001 sbatch scripts/jz/npe_hos_baryonsafe_area.slurm
#
# BOTH ROLES are trained, not just null. The FoM table needs the NULL posteriors, but the
# column is headed "baryon-safe", and that claim is only verified by measuring the null-to-biased
# tension at the SAME footprint (scripts/diagnostics/check_hos_cut_is_safe.py). scales234 has so
# far been verified at 14000 only -- 0.079+/-0.047 sigma for peaks, 0.091+/-0.073 for l1 -- and
# was carried elsewhere as an assumption. The biased half costs the same as the null half and
# turns that assumption into a measurement.
#
# WALLTIME IS DELIBERATELY SHORT. 20 trainings per footprint (2 statistics x 2 roles x 5 seeds)
# at ~8 min each is ~2.7 h, but gpu_p5 routinely carries ~800 pending jobs and backfills SHORT
# requests first: 1 h jobs here have started in under a minute while a 20 h request once bought
# a 34 h wait. So split by statistic (STATS=l1 / STATS=peaks, ~10 trainings = ~80 min each) and
# request 2 h. Four such jobs also run concurrently, so the wall clock is ~80 min rather than
# 2.7 h. The skip logic makes requeueing after a wall-clock kill safe, so shorter is strictly
# better than longer here -- do not pad the request 'to be safe'.
#
#   for A in 2001 5001; do for S in l1 peaks; do
#       AREA_TAG=$A STATS=$S sbatch --export=ALL,AREA_TAG=$A,STATS=$S <this>
#   done; done
#
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
AREA_TAG="${AREA_TAG:?set AREA_TAG, e.g. 2001 or 5001 -- the HOS footprint tag, NOT the nominal area}"

# Which wavelet scales. --scales is 0-BASED indices; the filename tag is 1-based (idx+1).
#   idx 0        finest/highest-ell detail scale -- the baryon-contaminated one, dropped here
#   idx 1,2,3    detail scales -> tag "scales234"   (the baryon-safe set)
#   idx 4        COARSE / mass-sheet scale, normally excluded
#   idx 1,2,3,4  detail + coarse -> tag "scales2345"
# Override both together. Set them in the CALLING SHELL, never via --export=ALL,VAR=...:
#
#   SCALES="1,2,3,4" SCALES_TAG=scales2345 sbatch scripts/jz/npe_hos_baryonsafe_14000.slurm
#
# sbatch --export uses COMMAS to separate variables, so --export=ALL,SCALES=1,2,3,4 silently
# becomes SCALES=1 plus three junk names. That happened (job 436470): it trained 20 models on
# a single wavelet scale, wrote them as "scales2", and wasted 40 GPU-minutes. --export=ALL is
# the default, so simply exporting from the caller works.
SCALES="${SCALES:-1,2,3}"
SCALES_TAG="${SCALES_TAG:-scales234}"
RUNS="1 2 3 4 5"

# FOOTPRINT: "masked" (uses AREA_TAG) or "fullsky". STATS: which statistics to train.
#   FOOTPRINT=fullsky STATS=l1 SCALES="1,2,3" SCALES_TAG=scales234 sbatch <this>
# peaks has NO full-sky submean grid: peak_counts_processing.py gates its submean branch on
# apply_mask (line ~142, submean_suffix = "_submean" if (apply_mask and submean)), so a
# full-sky peaks run silently produces an untagged, un-mean-subtracted product. l1 does have
# the full-sky branch. So FOOTPRINT=fullsky must be run with STATS=l1 until that is fixed.
FOOTPRINT="${FOOTPRINT:-masked}"
STATS="${STATS:-l1 peaks}"

# SUBMEAN=0 trains on the NON-submean product. Only legitimate for DETAIL-ONLY scale sets
# on the FULL SKY, and there it is exact rather than approximate: full-sky "submean" is a
# monopole subtraction, and a starlet detail coefficient is a difference of successively
# smoothed maps, w_j = c_{j-1} - c_j, so a constant cancels identically and only the coarse
# scale c_J retains it. Hence scales234 (detail only) is monopole-invariant BY CONSTRUCTION.
# Do NOT use this for a scale set that includes the coarse scale (e.g. scales2345), where the
# monopole genuinely matters, nor for masked runs, where submean subtracts a FOOTPRINT mean
# rather than a constant and is a real correction.
#
# This exists because peak_counts_processing.py has no full-sky submean branch (its submean
# block sits inside `if apply_mask:`), so no full-sky submean peak-count datavector was ever
# produced. Making one means reprocessing ~69,000 maps; the invariance above makes that
# unnecessary for detail-only cuts.
SUBMEAN="${SUBMEAN:-1}"
if [ "$SUBMEAN" = "1" ]; then SUBMEAN_ARG="--submean"; else SUBMEAN_ARG=""; fi

if [ "$FOOTPRINT" = "fullsky" ]; then
  MASK_ARGS=""
  if [ "$SUBMEAN" = "1" ]; then OUT_TAG="fullsky_submean"; else OUT_TAG=""; fi
  GRID_TAG=""
else
  MASK_ARGS="--masked --mask-area-sqdeg $AREA_TAG"
  OUT_TAG="masked_${AREA_TAG}sqdeg_submean"
  GRID_TAG="_masked_${AREA_TAG}sqdeg"
fi
DD=/lustre/fsn1/projects/rech/prk/ulx34io/cosmogrid_products/stage3_forecast
SAMPLES=/lustre/fsn1/projects/rech/prk/ulx34io/bar_impact/outputs/samples
CKPT=/lustre/fsn1/projects/rech/prk/ulx34io/bar_impact/checkpoints/npe_bsafe_area

# PRE-FLIGHT: the filename tag is idx+1 per index, so derive it and refuse to run if it
# disagrees with SCALES_TAG. This is what catches a mangled SCALES before 20 trainings go into
# the wrong filename -- the failure mode above was invisible until the very end of the job.
expected_tag="scales$(echo "$SCALES" | tr ',' '\n' | awk '{printf "%d", $1+1}')"
if [ "$expected_tag" != "$SCALES_TAG" ]; then
  echo "[fatal] SCALES='$SCALES' implies tag '$expected_tag' but SCALES_TAG='$SCALES_TAG'."
  echo "        The two must agree. If SCALES looks truncated, you passed it through"
  echo "        'sbatch --export=ALL,SCALES=...' -- commas separate variables there."
  exit 1
fi
echo "[pre-flight] SCALES='$SCALES' -> tag '$expected_tag' == SCALES_TAG. OK."

# PRE-FLIGHT 2: the training INPUT must exist. Job 426843 trained nothing because
# --mask-area-sqdeg is used verbatim to build input names and 14000 was passed where the
# datavectors are tagged 14001 -- 20 launches, all dying on FileNotFoundError. One stat call
# up front turns that into an immediate, legible failure.
for stat in $STATS; do
  case "$stat" in
    l1)    inpref="all_l1_norms" ;;
    peaks) inpref="all_peak_counts" ;;
    *)     echo "[fatal] unknown statistic '$stat' in STATS"; exit 1 ;;
  esac
  if [ "$SUBMEAN" = "1" ]; then sm="_submean"; else sm=""; fi
  probe="${DD}/grid/${inpref}_grid_nobaryons_bin1${GRID_TAG}${sm}_noisy_s0.26_new_normalization.npy"
  if [ ! -f "$probe" ]; then
    echo "[fatal] missing training input for '$stat' on footprint '$FOOTPRINT':"
    echo "        $probe"
    [ "$FOOTPRINT" = "fullsky" ] && [ "$stat" = "peaks" ] && \
      echo "        (expected: peak_counts_processing.py has no full-sky submean branch)"
    exit 1
  fi
  echo "[pre-flight] $stat input present: $(basename "$probe")"
done

mkdir -p "$CKPT"

echo "=== NPE HOS [${STATS}], footprint ${FOOTPRINT} (${OUT_TAG}), ${SCALES_TAG} (idx ${SCALES}) ==="
echo "start : $(date -Is)"
echo "python: $PY"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo

done_n=0; skip_n=0; fail_n=0

for stat in $STATS; do
  case "$stat" in
    l1)    SCRIPT=scripts/run_npe_inference.py            ; TAG=""     ;;
    peaks) SCRIPT=scripts/run_npe_peak_counts_inference.py; TAG="pc_"  ;;
  esac

  for role in nobaryons baryonified; do
    # role names the FIDUCIAL (the observed data vector); the model is always nobaryons-trained.
    for r in $RUNS; do
      out="${SAMPLES}/posterior_samples_${TAG}nobaryons_vs_${role}_bins1234_${SCALES_TAG}_noisy_s0.26${OUT_TAG:+_}${OUT_TAG}_new_normalization_run${r}_npe.npy"

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
        $MASK_ARGS $SUBMEAN_ARG \
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
