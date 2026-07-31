#!/bin/bash
#SBATCH --job-name=rebuild_gcov
#SBATCH --account=nzu@cpu
#SBATCH --partition=cpu_p1
#SBATCH --qos=qos_cpu-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --output=logs/rebuild_gcov_%j.out
#SBATCH --error=logs/rebuild_gcov_%j.out
#
# Regenerate the analytic mask-aware Gaussian covariance destroyed by the RAID0 failure.
#
# WHAT WAS LOST: every large binary in scripts/diagnostics/cache_gaussian_cov/ —
# gaussian_cov_native_*.npy (6), cw_*.fits (6), w_*.fits (6) — all carrying the 3 MiB stripe
# signature, ~20% zeroed. A zeroed covariance is INDEFINITE, not merely singular, so C^-1 against it
# returns plausible-looking nonsense (it gave a Fisher ratio of 1.377 where the truth is 1.455).
# Only cov_rebinned_full_*.npz survived, being 234 KB — under the stripe period.
#
# THIS JOB rebuilds area 14000 only (the sole footprint in scope). The expensive step is NaMaster's
# covariance coupling coefficients at nside=512 / lmax=1535; the MCM is comparatively cheap.
#
# TWO HAZARDS THIS SCRIPT DEFENDS AGAINST — both would silently corrupt the result:
#   1. fisher_gaussian_cov.py caches on os.path.exists(), so it would happily read_from() the
#      DAMAGED w_/cw_ fits and np.load() the damaged native cov and skip regeneration entirely.
#      => the damaged files are moved aside first, never read.
#   2. at its line ~273 the script OVERWRITES cov_rebinned_full_{AREA}.npz — the one intact
#      artifact we have. => backed up read-only to _INTACT_PRECRASH_BACKUP/ before the run, and the
#      post-flight compares the regenerated version against it.
#
# THE VALIDATION ORACLE: rebinning the regenerated native covariance must reproduce the surviving
# pre-crash rebinned covariance. That is an independent, pre-crash reference the job cannot fake.
#
# Submit:  sbatch scripts/jz/rebuild_gaussian_cov.slurm
set -euo pipefail
cd /lustre/fsn1/projects/rech/prk/ulx34io/bar_impact
mkdir -p logs

AREA=${AREA:-14000}
# NaMaster 2.x, NOT 3.0: in 3.0 NmtCovarianceWorkspace.compute_coupling_coefficients and .read_from
# became private (_-prefixed), so fisher_gaussian_cov.py does not run against it. Porting was
# rejected deliberately — the only validation is "reproduce the pre-crash reference", and a port
# would make a numerical disagreement ambiguous between a port bug and a version difference.
PY=/lustre/fswork/projects/rech/nzu/ulx34io/envs/namaster2/bin/python
CACHE=scripts/diagnostics/cache_gaussian_cov
BACKUP=$CACHE/_INTACT_PRECRASH_BACKUP
QUARANTINE=$CACHE/_DAMAGED_RAID0
export FISHER_AREA=$AREA
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-40}
# a stale ~/.local astropy shadows the env's and dies on numpy 2 (np.in1d removed)
export PYTHONNOUSERSITE=1

echo "=== rebuild_gaussian_cov  AREA=$AREA  threads=$OMP_NUM_THREADS  $(date) ==="

# --- pre-flight 1: the environment has NaMaster AND the pre-3.0 public API --------------------
$PY -c "
import pymaster as nmt, healpy, numpy, sys
print('[env] pymaster', nmt.__version__, '| healpy', healpy.__version__, '| numpy', numpy.__version__)
missing = [n for n, o in [
    ('NmtBin.from_lmax_linear', getattr(nmt.NmtBin, 'from_lmax_linear', None)),
    ('NmtWorkspace.compute_coupling_matrix', getattr(nmt.NmtWorkspace, 'compute_coupling_matrix', None)),
    ('NmtWorkspace.read_from', getattr(nmt.NmtWorkspace, 'read_from', None)),
    ('NmtCovarianceWorkspace.compute_coupling_coefficients',
     getattr(nmt.NmtCovarianceWorkspace, 'compute_coupling_coefficients', None)),
    ('NmtCovarianceWorkspace.read_from', getattr(nmt.NmtCovarianceWorkspace, 'read_from', None)),
    ('gaussian_covariance', getattr(nmt, 'gaussian_covariance', None))] if o is None]
if missing:
    print('[fatal] NaMaster API mismatch (this is 3.0+?): ' + ', '.join(missing)); sys.exit(1)
print('[env] pre-3.0 public API present')
" || { echo "[fatal] namaster env broken or wrong version: $PY"; exit 1; }

# --- pre-flight 2: the intact reference must exist and be sane -------------------------------
REF=$BACKUP/cov_rebinned_full_${AREA}.npz
[ -f "$REF" ] || { echo "[fatal] missing intact reference $REF — cannot validate; refusing to run"; exit 1; }
$PY - <<PYEOF || { echo "[fatal] reference covariance is not sane"; exit 1; }
import numpy as np
C = np.load("$REF")["C_analytic"]
ev = np.linalg.eigvalsh(C)
assert ev.min() > 0, "reference not PSD"
assert np.isclose(np.trace(C), ev.sum()), "reference trace != sum(eig)"
print(f"[ref] intact reference OK: {C.shape} cond={ev.max()/ev.min():.3e}")
PYEOF

# --- quarantine the damaged inputs so the cache logic cannot resurrect them -------------------
mkdir -p $QUARANTINE
for f in w_${AREA}.fits cw_${AREA}.fits gaussian_cov_native_${AREA}.npy; do
  if [ -f "$CACHE/$f" ]; then
    mv -v "$CACHE/$f" "$QUARANTINE/$f"
  fi
done

# --- the actual rebuild (hours: cw coupling coefficients dominate) ----------------------------
echo "[run] $(date) starting fisher_gaussian_cov.py"
$PY -u scripts/diagnostics/fisher_gaussian_cov.py
echo "[run] $(date) finished"

# --- post-flight: the regenerated covariance must reproduce the pre-crash rebinned reference ---
$PY - <<PYEOF
import numpy as np, sys, os
sys.path.insert(0, "scripts/diagnostics")
os.environ.setdefault("FISHER_AREA", "$AREA")
new = np.load("$CACHE/cov_rebinned_full_${AREA}.npz")["C_analytic"]
ref = np.load("$REF")["C_analytic"]
nat = np.load("$CACHE/gaussian_cov_native_${AREA}.npy")

ev = np.linalg.eigvalsh(nat)
print(f"[check] native {nat.shape} eig[min,max]=[{ev.min():.3e},{ev.max():.3e}] "
      f"PSD={ev.min() > -1e-10*abs(ev.max())} zerofrac={(nat==0).mean():.4f}")

if new.shape != ref.shape:
    print(f"[FAIL] shape {new.shape} != reference {ref.shape}"); sys.exit(1)
den = np.abs(ref).max()
amax = np.abs(new - ref).max()
rel = amax / den
# diagonal is the robust, high-signal comparison; off-diagonals span many orders of magnitude
dd = np.abs(np.diag(new) - np.diag(ref)) / np.abs(np.diag(ref))
print(f"[check] vs pre-crash rebinned reference: max|dC|/max|C| = {rel:.3e}")
print(f"[check] diagonal relative diff: median={np.median(dd):.3e} max={dd.max():.3e}")
ok = rel < 1e-6 and dd.max() < 1e-6
print("[RESULT] " + ("PASS - regenerated covariance reproduces the pre-crash reference"
                     if ok else
                     "FAIL - regenerated covariance DISAGREES with the pre-crash reference; "
                     "do not use it until this is understood"))
sys.exit(0 if ok else 1)
PYEOF

echo "=== done $(date) ==="
