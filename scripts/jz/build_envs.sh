#!/usr/bin/env bash
# Rebuild the two bar_impact environments that did not survive the Titan disk failure.
#
# RUN ON A LOGIN OR PREPOST NODE — compute nodes have no internet.
# This is a package install, not compute, so it respects the login-node limits.
#
#   bash scripts/jz/build_envs.sh            # both
#   bash scripts/jz/build_envs.sh namaster   # just the pymaster env
#   bash scripts/jz/build_envs.sh tension    # just the tensiometer env
#
# ---------------------------------------------------------------------------
# Why two, and why not one:
#
# The recovered Titan memory `bar-impact-tension-env` records that bar_impact has
# THREE environments that must not be mixed, each missing the others' deps:
#
#   1. NPE / jaxili / plotting     -> already rebuilt at
#                                     $WORK/../nzu/ulx34io/envs/jaxili  (getdist 1.6.1)
#   2. Power spectra (pymaster)    -> "cosmostat_new", built here
#   3. Tension stats (tensiometer) -> "aname", built here  (getdist 1.4.3)
#
# Note the getdist version differs between (1) and (3) — 1.6.1 vs 1.4.3. That is not
# an accident to be tidied up; the memory calls it out explicitly. Keep them separate.
# ---------------------------------------------------------------------------

set -euo pipefail

WHICH=${1:-both}
# Envs go under nzu, NOT prk. prk's $WORK is at ~94% of its 500k INODE quota (storage is
# only ~69% full, but a conda env is tens of thousands of small files, so it is inodes
# that run out — the failure surfaces as a misleading "Disk quota exceeded"). nzu $WORK
# sits near 55% and already hosts the jaxili env.
ENV_ROOT=${ENV_ROOT:-/lustre/fswork/projects/rech/nzu/ulx34io/envs}
export https_proxy=${https_proxy:-http://prodprox.idris.fr:3128}
export http_proxy=${http_proxy:-http://prodprox.idris.fr:3128}

case "$(hostname)" in
  *jean-zay*|*login*|*prepost*) ;;
  *) echo "WARNING: this looks like a compute node — no internet, the install will fail." ;;
esac

if ! command -v conda >/dev/null 2>&1; then
  echo "[env] loading conda module"
  module load miniforge 2>/dev/null || module load anaconda-py3 2>/dev/null || {
    echo "ERROR: no conda module. Try: module avail 2>&1 | grep -i -e miniforge -e conda"; exit 1; }
fi
eval "$(conda shell.bash hook)"

# --------------------------------------------------------------------------
build_namaster () {
  local P="$ENV_ROOT/cosmostat_new"
  if [ -d "$P" ]; then echo "[namaster] $P exists — reusing"; else
    echo "[namaster] creating (conda-forge namaster 2.5.2)"
    # 2.5.2 matches what the original Titan venv carried
    # (/home/tersenov/software/cosmostat_new/cosmostat/cosmostat_new/bin/python).
    # The memory also warns the on-disk masked MASTER products came from an even
    # older build and do NOT reproduce bit-for-bit — so expect small numerical
    # differences and do not mix old and new outputs.
    conda create -y -p "$P" -c conda-forge \
        python=3.11 "namaster=2.5.2" "healpy=1.18" numpy scipy astropy matplotlib-base \
      || { echo "[namaster] 2.5.2 unavailable — falling back to latest (EXPECT differences)"
           conda create -y -p "$P" -c conda-forge python=3.11 namaster healpy numpy scipy astropy; }
  fi
  echo "[namaster] verifying — exercising the exact call fisher_gaussian_cov.py makes"
  "$P/bin/python" - <<'PY'
import pymaster as nmt, healpy as hp, numpy as np
print("   pymaster", getattr(nmt, "__version__", "?"), "| healpy", hp.__version__)
b = nmt.NmtBin.from_lmax_linear(255, nlb=4)
m = np.ones(hp.nside2npix(64))
f = nmt.NmtField(m, None, spin=0, lmax=255)
w = nmt.NmtWorkspace(); w.compute_coupling_matrix(f, f, b)
cw = nmt.NmtCovarianceWorkspace(); cw.compute_coupling_coefficients(f, f)
cl = np.ones(256)
c = nmt.gaussian_covariance(cw, 0, 0, 0, 0, [cl], [cl], [cl], [cl], w, w)
print("   smoke test: gaussian_covariance ->", np.asarray(c).shape, "OK")
PY
}

# --------------------------------------------------------------------------
build_tension () {
  local P="$ENV_ROOT/aname"
  if [ -d "$P" ]; then echo "[tension] $P exists — reusing"; else
    echo "[tension] creating (tensiometer + getdist 1.4.3)"
    # matplotlib-base, not matplotlib: the full package pulls Qt6 (~1 GB and many
    # thousands of files) for GUI backends we never use — these scripts all run Agg.
    # On an inode-constrained filesystem that is the difference between fitting and not.
    conda create -y -p "$P" -c conda-forge python=3.11 numpy scipy matplotlib-base pandas
    # getdist pinned to 1.4.3 to match the original aname env — tensiometer pulls
    # GetDist as a dependency, so install it explicitly first and let pip resolve.
    "$P/bin/pip" install --no-input "getdist==1.4.3" tensiometer
  fi
  echo "[tension] verifying — importing what tension/estimators.py actually uses"
  "$P/bin/python" - <<'PY'
import tensiometer, getdist
print("   tensiometer", getattr(tensiometer, "__version__", "?"), "| getdist", getdist.__version__)
import tensiometer.utilities as utilities          # the import that failed on jaxili
print("   tensiometer.utilities OK")
try:
    from tensiometer import gaussian_tension
    print("   gaussian_tension OK (Q_DM path)")
except Exception as e:
    print("   [warn] gaussian_tension import:", type(e).__name__, e)
PY
}

# --------------------------------------------------------------------------
case "$WHICH" in
  namaster) build_namaster ;;
  tension)  build_tension ;;
  both)     build_namaster; build_tension ;;
  *) echo "usage: $0 [both|namaster|tension]"; exit 2 ;;
esac

cat <<EOF

=== next ===
Covariance rebuild (real compute — submit it, do not run it on a login node):

  ENV=$ENV_ROOT/cosmostat_new
  for A in 2000 5000 10000 14000 28000 35000; do
    FISHER_AREA=\$A \$ENV/bin/python scripts/diagnostics/fisher_gaussian_cov.py
  done

Let it regenerate w_<A>.fits / cw_<A>.fits — the damaged ones must not be reused,
and workspace files are NaMaster-version-specific anyway.

Tension / nsigma plots:

  $ENV_ROOT/aname/bin/python scripts/diagnostics/plot_nsigma_vs_area.py
  $ENV_ROOT/aname/bin/python scripts/compute_tension.py ...

Use the jaxili interpreter for NPE and general plotting; do not mix the three.
EOF
