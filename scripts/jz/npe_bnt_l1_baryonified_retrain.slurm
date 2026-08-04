#!/bin/bash
#SBATCH --job-name=npe_bntl1
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
#SBATCH --output=logs/npe_bntl1_%j.out
#SBATCH --error=logs/npe_bntl1_%j.out
#
# Retrain the BNT l1 arm whose baryonified posterior came back prior-dominated.
#
# WHAT FAILED. outputs/samples/posterior_samples_nobaryons_vs_baryonified_bntbins1234_
# scales1234_noisy_s0.26_new_normalization_npe.npy (2026-02-08 21:53) spans the FULL prior
# in all three parameters (Om 0.016-0.443, S8 0.434-1.224, w0 -1.37..-0.30) against
# Om 0.215-0.298 for a converged run. It is unusable, so hos_bnt_l1_vs_scale_cut currently
# has to show the nobaryons arm instead -- which is NOT like-for-like with the two
# baryonified comparison arms in the same figure. This job closes that gap.
#
# WHY IT FAILED (diagnosis, not yet proof -- this job is the proof).
# The checkpoint name is built from simulation_type only, NOT fiducial_type
# (run_npe_inference.py:679,696), so ONE trained NDE served both arms that evening: the
# good nobaryons_vs_nobaryons contour at 19:47 and the prior-wide baryonified one at 21:53.
# Training was therefore fine; the OBSERVATION broke the evaluation.
#
# Standardising the baryonified fiducial mean against the nobaryons training grid:
#   588 of 589 live features lie at |z| < 1.8   -- comfortably inside the training manifold
#   1 feature -- BNT bin4, finest scale, SNR bin 36 -- lies at |z| = 27
# That bin is nonzero in 2 of 16,965 training sims; its std (0.42) is set entirely by those
# two outliers. Handing a flow a 27-sigma input on a near-degenerate feature is a standard
# route to a flat posterior.
#
# WHY RERUNNING SHOULD FIX IT. filter_zero_variance_bins originally used an ABSOLUTE floor
# only (min_variance=1e-5), which that bin survives (variance 0.179). Commit 3d92555
# (2026-07-29) added a RELATIVE floor, 1e-6 x median(variance) = 0.65 here, which drops it:
#   old absolute-only floor -> 589 features kept, the 27-sigma bin among them
#   current filter          -> 569 features kept, that bin removed
# The broken run predates the fix by five months. Watch the "Zero-variance bin filtering"
# block in the log: it must report 569 valid bins. If it reports 589, the fix is not on the
# checked-out tree and this job will reproduce the failure rather than repair it.
#
# --train IS REQUIRED, not optional. The filter change alters the input dimension (569 vs
# 589), so the existing checkpoint cannot be reused -- loading it would fail or, worse,
# silently mis-map features.
#
# TWO RUNS, ONE TRAINING. Run 1 trains and samples at the baryonified fiducial. Run 2 omits
# --train, so it LOADS the checkpoint just written and samples the nobaryons fiducial. That
# is not a spare figure arm -- it is the control. The nobaryons contour is known-good from
# February, so if the retrained NDE reproduces it, the training is sound and any change in
# the baryonified arm is physical rather than an artefact of retraining. It also exactly
# mirrors the original setup: one NDE, two observations.
#
# NON-DESTRUCTIVE BY CONSTRUCTION:
#   --checkpoint-dir points at a NEW directory, so the 589-feature checkpoint that the
#     February figures were built from stays intact and those figures stay reproducible.
#   --run 1 tags the output _run1, so the broken posterior is preserved as evidence rather
#     than overwritten. Nothing already on disk is replaced by this job.
#
# WALLTIME. The comparable HOS job measured ~8 min per training (1000 epochs at ~2.4 it/s
# plus setup) at a similar 16,965 x ~570 problem size; this is one training plus two
# sampling passes. 1 h is requested to schedule quickly on a busy A100 partition -- do NOT
# raise it speculatively, a 20 h request on this partition bought a 34 h queue wait.
#
# Submit from the repo root:  sbatch scripts/jz/npe_bnt_l1_baryonified_retrain.slurm

set -uo pipefail

cd /lustre/fsn1/projects/rech/prk/ulx34io/bar_impact || exit 1

module purge
module load arch/a100
PY=/lustre/fswork/projects/rech/nzu/ulx34io/envs/jaxili/bin/python

# ~/.local holds an astropy built against numpy 1.x that shadows this env's numpy 2 and
# fails at import. Every python invocation here must carry this.
export PYTHONNOUSERSITE=1

DD=/lustre/fsn1/projects/rech/prk/ulx34io/cosmogrid_products/stage3_forecast
SAMPLES=/lustre/fsn1/projects/rech/prk/ulx34io/bar_impact/outputs/samples
CKPT=/lustre/fsn1/projects/rech/prk/ulx34io/bar_impact/checkpoints/npe_bnt_l1_retrain

# --bnt-bins and --scales are 0-BASED; both filename tags are 1-based (idx+1).
#   --bnt-bins 0,1,2,3 -> tag bntbins1234   (all four BNT bins, as in the figure)
#   --scales   0,1,2,3 -> tag scales1234    ("all scales": the four detail scales.
#                                            idx 4, coarse/mass-sheet, is excluded
#                                            throughout the analysis and never used.)
BNT_BINS="0,1,2,3"
SCALES="0,1,2,3"
RUN=1

# PRE-FLIGHT 1: every input must exist. The runner builds paths verbatim from the flags, so
# a missing file surfaces as a FileNotFoundError partway in; one stat call up front turns
# that into an immediate, legible failure. BNT grids live in new_grid/, BNT fiducials in
# fiducial/cosmo_fiducial/ (run_npe_inference.py:499-506).
missing=0
for b in 1 2 3 4; do
  for f in \
    "${DD}/new_grid/all_bnt_l1_norms_grid_nobaryons_bin${b}_noisy_s0.26_new_normalization.npy" \
    "${DD}/fiducial/cosmo_fiducial/all_bnt_l1_norms_fiducial_baryonified_bin${b}_noisy_s0.26_new_normalization.npy" \
    "${DD}/fiducial/cosmo_fiducial/all_bnt_l1_norms_fiducial_nobaryons_bin${b}_noisy_s0.26_new_normalization.npy"
  do
    [ -f "$f" ] || { echo "[fatal] missing input: $f"; missing=1; }
  done
done
[ -f "${DD}/grid/cosmo_params.npy" ] || { echo "[fatal] missing ${DD}/grid/cosmo_params.npy"; missing=1; }
[ $missing -eq 0 ] || exit 1
echo "[pre-flight] all 13 inputs present (4 grid + 8 fiducial + params)"

# PRE-FLIGHT 2: confirm the variance-floor fix is actually on the checked-out tree. Without
# it this job silently reproduces the February failure, and the only symptom would be
# another prior-wide posterior an hour from now.
# Importing run_npe_inference pulls in jax/jaxili, which is slow and pointless here, so
# lift just the filter function out of the source with AST and run it standalone. This still
# tests the REAL code on the tree, not a reimplementation of it.
DD="$DD" "$PY" -c "
import sys, os, ast, types
import numpy as np
text = open('scripts/run_npe_inference.py').read()
tree = ast.parse(text)
src = {n.name: ast.get_source_segment(text, n) for n in tree.body
       if isinstance(n, ast.FunctionDef)}
if 'filter_zero_variance_bins' not in src:
    print('[fatal] filter_zero_variance_bins not found in scripts/run_npe_inference.py')
    sys.exit(1)
m = types.SimpleNamespace()
ns = {'np': np}
exec(src['filter_zero_variance_bins'], ns)
m.filter_zero_variance_bins = ns['filter_zero_variance_bins']
D = os.environ['DD']
G = np.concatenate([
    np.load(f'{D}/new_grid/all_bnt_l1_norms_grid_nobaryons_bin{b}_noisy_s0.26_new_normalization.npy')[:, [0,1,2,3], :].reshape(16965, -1)
    for b in [1,2,3,4]], 1)
mask, n_removed = m.filter_zero_variance_bins(G, min_variance=1e-5, verbose=False)
kept = int(mask.sum())
print(f'[pre-flight] variance filter keeps {kept} / {G.shape[1]} features')
if kept == 569:
    print('[pre-flight] OK: relative floor active, the 27-sigma bin is filtered out.')
    sys.exit(0)
print(f'[fatal] expected 569 kept. {kept} means the relative variance floor (commit 3d92555)')
print('        is NOT on this tree -- this job would reproduce the prior-wide posterior.')
sys.exit(1)
" || exit 1

mkdir -p "$CKPT" logs

OUT_BAR="${SAMPLES}/posterior_samples_nobaryons_vs_baryonified_bntbins1234_scales1234_noisy_s0.26_new_normalization_run${RUN}_npe.npy"
OUT_NOB="${SAMPLES}/posterior_samples_nobaryons_vs_nobaryons_bntbins1234_scales1234_noisy_s0.26_new_normalization_run${RUN}_npe.npy"

echo
echo "=== BNT l1 retrain | bntbins1234 scales1234 | full sky | noisy s0.26 ==="
echo "start : $(date -Is)"
echo "python: $PY"
echo "ckpt  : $CKPT   (new dir; the February checkpoint is untouched)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo

# --- 1. TRAIN, and sample at the BARYONIFIED fiducial (the arm the paper needs) -----------
# Idempotent: job 570333 already produced this arm, so a resubmit for the control below must
# not spend 8 GPU-minutes reproducing it. Skip only if the posterior exists AND loads -- a
# disk-damaged .npy raises on np.load, so -f alone would skip a corrupt file.
if [ -f "$OUT_BAR" ] && "$PY" -c "
import sys, numpy as np
try:
    a = np.load(sys.argv[1]); sys.exit(0 if a.ndim == 2 and a.shape[0] > 0 else 1)
except Exception:
    sys.exit(1)" "$OUT_BAR" 2>/dev/null; then
  echo "[skip] baryonified arm (readable posterior already present)"
  rc_bar=0
else
echo "[run ] train + baryonified fiducial   ($(date -Is))"
srun "$PY" -u scripts/run_npe_inference.py \
  --data-dir "$DD" \
  --bnt --bnt-bins "$BNT_BINS" \
  --scales "$SCALES" \
  --simulation-type nobaryons \
  --fiducial-type baryonified \
  --noisy --noise-level 0.26 \
  --new-normalization \
  --train --epochs 1000 \
  --num-samples 3000 \
  --run "$RUN" --random-seed "$RUN" \
  --checkpoint-dir "$CKPT" \
  --samples-dir "$SAMPLES"
rc_bar=$?
[ $rc_bar -eq 0 ] || { echo "[FAIL] training/baryonified sampling rc=$rc_bar"; exit 1; }
fi

# --- 2. Sample at the NOBARYONS fiducial (the control) ------------------------------------
# --train IS PASSED HERE ON PURPOSE, though it looks redundant. The intent was to LOAD the
# checkpoint written above, but run_npe_inference.py's no-train branch calls inference.load()
# and this jaxili's NPE has no such method: job 570333 printed
#   "Failed to load model: 'NPE' object has no attribute 'load'"
# and then `return`ed -- with EXIT CODE 0. The step consumed 20 s, wrote no posterior, and
# was reported as successful. Do not "simplify" this back to a bare load until that branch
# either gains a working loader or exits non-zero.
#
# Retraining does not weaken the control: --random-seed is the same, the training data are
# identical, so this reproduces the same NDE and the comparison against the known-good
# February nobaryons contour still tests exactly what it was meant to test.
echo
echo "[run ] nobaryons fiducial (control)   ($(date -Is))"
srun "$PY" -u scripts/run_npe_inference.py \
  --train --epochs 1000 \
  --data-dir "$DD" \
  --bnt --bnt-bins "$BNT_BINS" \
  --scales "$SCALES" \
  --simulation-type nobaryons \
  --fiducial-type nobaryons \
  --noisy --noise-level 0.26 \
  --new-normalization \
  --num-samples 3000 \
  --run "$RUN" --random-seed "$RUN" \
  --checkpoint-dir "$CKPT" \
  --samples-dir "$SAMPLES"
rc_nob=$?
[ $rc_nob -eq 0 ] || echo "[warn] control sampling rc=$rc_nob (the baryonified arm above may still be fine)"

# --- 3. VERDICT ---------------------------------------------------------------------------
# The whole point of the job is whether the posterior is still prior-dominated, so decide it
# here rather than in a figure. Prior width is the reference: a converged run is a small
# fraction of it, the February failure was ~all of it.
echo
echo "=== verdict ==="
"$PY" - "$OUT_BAR" "$OUT_NOB" <<'PYEOF'
import sys, os, numpy as np
OLD = ("outputs/samples/posterior_samples_nobaryons_vs_baryonified_"
       "bntbins1234_scales1234_noisy_s0.26_new_normalization_npe.npy")
NAMES = ["Om", "S8", "w0"]
def describe(tag, path):
    if not os.path.exists(path):
        print(f"{tag:28s} MISSING: {path}")
        return None
    s = np.load(path)
    rng = [(float(s[:, i].min()), float(s[:, i].max())) for i in range(3)]
    sd = [float(s[:, i].std()) for i in range(3)]
    print(f"{tag:28s} n={s.shape[0]:5d}  " +
          "  ".join(f"{n} {lo:6.3f}..{hi:6.3f} (sd {d:.3f})"
                    for n, (lo, hi), d in zip(NAMES, rng, sd)))
    return s
print()
old = describe("OLD baryonified (broken)", OLD)
new = describe("NEW baryonified (run1)", sys.argv[1])
ctl = describe("NEW nobaryons  (control)", sys.argv[2])
print()
if new is None:
    sys.exit(1)
# sigma_S8 >= 0.08 is the prior-width tripwire used across this analysis.
s8 = float(new[:, 1].std())
if s8 >= 0.08:
    print(f"*** STILL PRIOR-DOMINATED: sigma_S8 = {s8:.4f} >= 0.08 ***")
    print("    The variance-floor hypothesis does NOT explain the failure on its own.")
    print("    Next: read the training log above for the validation loss and the reported")
    print("    'Removed bins' count before assuming the data are at fault.")
else:
    print(f"CONVERGED: sigma_S8 = {s8:.4f} < 0.08")
    if old is not None:
        print(f"    prior-wide old run had sigma_S8 = {float(old[:,1].std()):.4f}")
    print("    The l1 BNT arm is now like-for-like baryonified. Rebuild the figure with:")
    print("      scripts/plot_hos_bnt_triangle.py --stat l1 --bnt-arm baryonified \\")
    print("          --outline-bnt --name hos_v2_l1")
    print("    and REMOVE the l1 entry from NON_CONVERGED in that script, plus point FILES")
    print("    ['l1']['bnt_baryonified'] at the _run1 file.")
PYEOF

echo
echo "end   : $(date -Is)"
