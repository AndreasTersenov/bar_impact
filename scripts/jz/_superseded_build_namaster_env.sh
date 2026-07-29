#!/usr/bin/env bash
# Build the NaMaster (pymaster) environment needed for the covariance rebuild.
#
# RUN THIS ON A LOGIN OR PREPOST NODE — compute nodes have no internet.
# It is a package install, not compute, so it respects the login-node limits.
#
#   bash scripts/jz/build_namaster_env.sh
#
# Why this is needed: all 19 files in scripts/diagnostics/cache_gaussian_cov/
# (gaussian_cov_native_*.npy, w_*.fits, cw_*.fits) were destroyed by the Titan
# disk failure and are damaged identically at source, so they cannot be
# re-transferred — only recomputed. fisher_gaussian_cov.py needs pymaster to do
# that, and no environment on Jean Zay currently has it (the jaxili env has jax,
# healpy and numpy but not pymaster).

set -euo pipefail

ENV_PREFIX=${ENV_PREFIX:-/lustre/fswork/projects/rech/prk/ulx34io/envs/cosmostat_new}
export https_proxy=${https_proxy:-http://prodprox.idris.fr:3128}
export http_proxy=${http_proxy:-http://prodprox.idris.fr:3128}

echo "=== NaMaster env build ==="
echo "prefix : $ENV_PREFIX"
echo "host   : $(hostname)"

case "$(hostname)" in
  *jean-zay*|*login*|*prepost*) ;;
  *) echo "WARNING: this looks like a compute node — it has no internet and the install will fail." ;;
esac

# conda is not on PATH by default on Jean Zay
if ! command -v conda >/dev/null 2>&1; then
  echo "[env] loading miniforge"
  module load miniforge 2>/dev/null || module load anaconda-py3 2>/dev/null || {
    echo "ERROR: no conda module found. Try: module avail 2>&1 | grep -i -e miniforge -e conda"
    exit 1; }
fi
eval "$(conda shell.bash hook)"

if [ -d "$ENV_PREFIX" ]; then
  echo "[env] $ENV_PREFIX already exists — reusing"
else
  echo "[env] creating (conda-forge namaster 2.5.2, linux-64)"
  # Version pinned deliberately. The recovered Titan memory `bar-impact-namaster-venv`
  # records that the original venv
  #   /home/tersenov/software/cosmostat_new/cosmostat/cosmostat_new/bin/python
  # carried **pymaster 2.5.2 with python 3.11**, and warns that the on-disk masked
  # MASTER products were made by an even older build and do not reproduce bit-for-bit
  # across versions. Matching 2.5.2 keeps the rebuilt covariances as close to the
  # originals as possible; conda-forge 3.0 would add a second version discontinuity.
  # namaster pulls its compiled deps (gsl, fftw, cfitsio) from conda-forge, so there
  # is no need to build from source.
  conda create -y -p "$ENV_PREFIX" -c conda-forge \
      python=3.11 "namaster=2.5.2" "healpy=1.18" numpy scipy astropy \
    || { echo "[env] 2.5.2 unavailable — falling back to latest, EXPECT small numerical differences"
         conda create -y -p "$ENV_PREFIX" -c conda-forge python=3.11 namaster healpy numpy scipy astropy; }
fi

conda activate "$ENV_PREFIX"

echo
echo "=== verify ==="
python - <<'PY'
import pymaster as nmt, healpy as hp, numpy as np
print("  pymaster", getattr(nmt, "__version__", "?"))
print("  healpy  ", hp.__version__)
print("  numpy   ", np.__version__)
# smoke-test the exact call fisher_gaussian_cov.py makes
b = nmt.NmtBin.from_lmax_linear(255, nlb=4)
m = np.ones(hp.nside2npix(64))
f = nmt.NmtField(m, None, spin=0, lmax=255)
w = nmt.NmtWorkspace(); w.compute_coupling_matrix(f, f, b)
cw = nmt.NmtCovarianceWorkspace(); cw.compute_coupling_coefficients(f, f)
cl = np.ones(256)
c = nmt.gaussian_covariance(cw, 0, 0, 0, 0, [cl], [cl], [cl], [cl], w, w)
print("  smoke test: gaussian_covariance ->", np.asarray(c).shape, "OK")
PY

cat <<EOF

=== next ===
The covariance rebuild is a real compute job — submit it, do not run it here:

  ENV=$ENV_PREFIX
  for A in 2000 5000 10000 14000 28000 35000; do
    FISHER_AREA=\$A \$ENV/bin/python scripts/diagnostics/fisher_gaussian_cov.py
  done

Each area recomputes w_<A>.fits (mode coupling) and cw_<A>.fits (covariance
coupling — the expensive one), then gaussian_cov_native_<A>.npy. Let it
regenerate the .fits workspaces: the damaged ones must not be reused, and
workspace files are NaMaster-version-specific anyway.

Wrap that loop in a cpu_p1 SLURM job with a generous walltime; the coupling
coefficients dominate.
EOF
