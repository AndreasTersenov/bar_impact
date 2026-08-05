#!/bin/bash
#SBATCH --job-name=ps_noiseless
#SBATCH --account=nzu@cpu
#SBATCH --partition=cpu_p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --hint=nomultithread
#SBATCH --time=03:00:00
#SBATCH --output=logs/ps_noiseless_%j.out
#SBATCH --error=logs/ps_noiseless_%j.out
#
# Rebuild the NOISELESS BARYONIFIED power spectra -- the missing half of the noiseless
# fractional-difference figure (Delta C_ell / C_ell, standard and BNT bins).
#
# WHY THEY ARE MISSING. Every noiseless product on SCRATCH is `nobaryons`: 0 of 1057 cls files
# under cosmogrid_products are noiseless AND baryonified. The derived .npy survive but the raw
# projected_probes_maps_*.h5 do not, so this cannot be recomputed from our own tree.
#
# WHERE THE MAPS ARE. The shared, read-only CosmoGridV1 release:
#   /lustre/fsmisc/dataset/CosmoGridV1/stage3_forecast/fiducial/cosmo_fiducial/perm_{0000..0199}
# 200 realisations, both baryonified and nobaryons, 122 MB each. Nothing needs downloading.
#
# READ-ONLY IS THE CATCH. Both processing scripts wrote outputs beside their input map, which
# fails on the shared dataset. They now take --out-dir, which also redirects the combined file;
# without that the combine step would try to write into /lustre/fsmisc and die at the very end,
# after all the compute.
#
# lmax 1024 to match the surviving noiseless nobaryons files, which are (200, 1025). The
# lmax2048 variants exist too but are a different product -- do not mix them in one figure.
#
# EIGHT PASSES: standard bins 1-4 and BNT bins 1-4, each reading all 200 maps. The h5 holds
# every tomographic bin, so a pass re-reads the same 122 MB file for one bin; that is I/O
# bound, hence 40 workers.

set -uo pipefail
cd /lustre/fsn1/projects/rech/prk/ulx34io/bar_impact || exit 1

export PYTHONNOUSERSITE=1
PY=/lustre/fswork/projects/rech/nzu/ulx34io/envs/jaxili/bin/python
SRC=/lustre/fsmisc/dataset/CosmoGridV1/stage3_forecast/fiducial/cosmo_fiducial
DEST=/lustre/fsn1/projects/rech/prk/ulx34io/cosmogrid_products/stage3_forecast/fiducial/cosmo_fiducial
WORKDIR=/lustre/fsn1/projects/rech/prk/ulx34io/cosmogrid_products/_noiseless_baryonified_work
mkdir -p "$WORKDIR" logs

echo "=== noiseless baryonified power spectra ==="
echo "start : $(date -Is)"
echo "maps  : $SRC  ($(ls -d $SRC/perm_* 2>/dev/null | wc -l) perms)"
echo "dest  : $DEST"
echo

# PRE-FLIGHT: the shared dataset must be readable and hold the baryonified maps, and the
# destination must be writable. Checked before spending hours of I/O.
n=$(ls $SRC/perm_*/projected_probes_maps_baryonified512.h5 2>/dev/null | wc -l)
[ "$n" -eq 200 ] || { echo "[fatal] expected 200 baryonified maps, found $n"; exit 1; }
[ -w "$DEST" ] || { echo "[fatal] destination not writable: $DEST"; exit 1; }
"$PY" -c "import h5py, healpy, numpy, tqdm" || { echo "[fatal] missing deps"; exit 1; }
echo "[pre-flight] 200 maps present, destination writable, deps import"
echo

fail=0

echo "--- standard bins 1-4 ---"
for B in 1 2 3 4; do
  OUT="$DEST/all_cls_fiducial_baryonified_bin${B}.npy"
  if [ -f "$OUT" ]; then echo "[skip] bin $B (exists)"; continue; fi
  echo "[run ] standard bin $B   ($(date -Is))"
  "$PY" -u scripts/power_spectrum_processing.py \
      --fiducial --baryonified --no-noise --lmax 1024 \
      --bin-number "$B" --base-dir "$SRC/" \
      --out-dir "$WORKDIR" --num-workers 40 \
      --save-combined --combined-output "$OUT" || { echo "[FAIL] standard bin $B"; fail=1; }
done

echo
echo "--- BNT bins 1-4 ---"
# --bnt-bin is 0-BASED and the output filename is 1-based (bnt_bin+1), matching the existing
# all_bnt_cls_fiducial_nobaryons_bin{1..4}.npy.
for J in 0 1 2 3; do
  B=$((J+1))
  OUT="$DEST/all_bnt_cls_fiducial_baryonified_bin${B}.npy"
  if [ -f "$OUT" ]; then echo "[skip] BNT bin $B (exists)"; continue; fi
  echo "[run ] BNT bin $B (--bnt-bin $J)   ($(date -Is))"
  "$PY" -u scripts/bnt_power_spectrum_processing.py \
      --fiducial --baryonified --no-noise --lmax 1024 \
      --bnt-bin "$J" --base-dir "$SRC/" \
      --out-dir "$WORKDIR" --num-workers 40 \
      --save-combined --combined-output "$OUT" || { echo "[FAIL] BNT bin $B"; fail=1; }
done

echo
echo "=== verify: shape and lmax must match the surviving nobaryons files ==="
"$PY" - <<'PYEOF'
import os, numpy as np
D = "/lustre/fsn1/projects/rech/prk/ulx34io/cosmogrid_products/stage3_forecast/fiducial/cosmo_fiducial"
bad = 0
for pre in ("all_cls", "all_bnt_cls"):
    for b in (1, 2, 3, 4):
        new = f"{D}/{pre}_fiducial_baryonified_bin{b}.npy"
        ref = f"{D}/{pre}_fiducial_nobaryons_bin{b}.npy"
        if not os.path.exists(new):
            print(f"  MISSING {os.path.basename(new)}"); bad += 1; continue
        a, r = np.load(new), np.load(ref)
        ok = a.shape == r.shape and np.isfinite(a).all()
        # A baryonified spectrum must sit BELOW the dark-matter-only one at high ell -- that
        # suppression is the entire physical content of the figure. If it does not, the wrong
        # map or the wrong bin was read, and the shapes alone would not reveal it.
        frac = (a.mean(0)[200:1000] - r.mean(0)[200:1000]) / r.mean(0)[200:1000]
        sup = frac.mean() < 0
        print(f"  {'ok  ' if ok and sup else 'BAD '} {os.path.basename(new):46s} "
              f"{a.shape}  mean dC/C(200-1000) = {frac.mean():+.4f}")
        if not (ok and sup):
            bad += 1
print("\nverify:", "all good" if bad == 0 else f"{bad} problem(s)")
raise SystemExit(1 if bad else 0)
PYEOF
vrc=$?

echo
echo "end   : $(date -Is)"
echo "failed_passes=$fail verify_rc=$vrc"
echo "(per-realisation intermediates are in $WORKDIR and can be deleted once the"
echo " combined files are verified -- they are ~200 x 8 small .npy)"
[ $fail -eq 0 ] && [ $vrc -eq 0 ] || exit 1
