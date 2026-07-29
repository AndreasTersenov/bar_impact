#!/bin/bash
#SBATCH --job-name=arch_store
#SBATCH --account=nzu@cpu
#SBATCH --partition=archive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --output=logs/archive_store_%j.out
#SBATCH --error=logs/archive_store_%j.out
#
# Archive $SCRATCH/cosmogrid_products/stage3_forecast to $STORE as TARBALLS.
#
# WHY TAR, NOT rsync.  The first attempt copied the tree file-for-file. stage3_forecast
# holds 350,578 files; a project's STORE inode quota is 100,000. The copy reached 93,483
# files -- 94% of the whole team-shared prk quota -- while transferring only 3.73 GiB of
# 83 GB, and would have died with "Disk quota exceeded" long before finishing. That error
# reports as a quota failure while `df` shows tens of TB free: it is inodes, not bytes.
# This is the documented purpose of STORE -- a small number of large files. Four tarballs
# use four inodes.
#
# Writes to nzu (prk's STORE inode quota is currently exhausted by the failed raw copy;
# that partial must be removed separately, and only AFTER these tarballs verify).
#
# MUST run on --partition=archive. $STORE is not mounted on the cpu_p1 compute nodes; a
# job submitted there fails instantly with "mkdir: cannot create directory '/lustre/fsstor':
# Permission denied" -- mkdir -p walking up to a path component that does not exist on
# that node. archive is the partition with the STORE mount (20 h limit, 1 core, no
# allocation charge), and it is I/O bound anyway.
#
# Each group produces:   <g>.tar  <g>.tar.list  <g>.tar.sha256
# The .list is a full member index so a single file can be located without reading the
# tarball; extract one with:  tar -xf <g>.tar -C <dest> <member-path>
#
# Submit from the repo root:  sbatch scripts/jz/archive_science_to_store.slurm

set -uo pipefail

SRC=/lustre/fsn1/projects/rech/prk/ulx34io/cosmogrid_products/stage3_forecast
DEST=/lustre/fsstor/projects/rech/nzu/ulx34io/titan_recovery/science_data

mkdir -p "$DEST" || { echo "[fatal] cannot create $DEST"; exit 1; }

echo "=== archive stage3_forecast -> STORE (tar) ==="
echo "src   : $SRC"
echo "dest  : $DEST"
echo "start : $(date -Is)"
df -h "$DEST" | tail -1
echo

# Top-level loose files go in their own tarball; each big subdir gets one.
GROUPS=(grid new_grid fiducial)

rc_all=0

# --- the loose files at the tree root (figures, .pkl, .npz) --------------------
tag=toplevel
if [ -f "$DEST/$tag.tar.sha256" ]; then
  echo "[skip] $tag (already has a checksum -- delete it to force a rebuild)"
else
  echo "[tar ] $tag"
  # -maxdepth 1 -type f: only the root-level files, never the subdirs handled below.
  ( cd "$SRC" && find . -maxdepth 1 -type f -print0 \
      | tar --null -T - -cvf "$DEST/$tag.tar" ) > "$DEST/$tag.tar.list" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then echo "[FAIL] tar $tag rc=$rc"; rc_all=1; else
    echo "[ver ] $tag"
    tar -tf "$DEST/$tag.tar" > /dev/null || { echo "[FAIL] unreadable $tag"; rc_all=1; }
    ( cd "$DEST" && sha256sum "$tag.tar" > "$tag.tar.sha256" )
    ls -lh "$DEST/$tag.tar"
  fi
fi
echo

for g in "${GROUPS[@]}"; do
  if [ ! -d "$SRC/$g" ]; then echo "[warn] $SRC/$g missing -- skipped"; continue; fi
  if [ -f "$DEST/$g.tar.sha256" ]; then
    echo "[skip] $g (already has a checksum -- delete it to force a rebuild)"
    continue
  fi
  echo "[tar ] $g   ($(date -Is))"
  # -v output IS the index; it goes to .list, so tar's own stdout must not be the tarball.
  ( cd "$SRC" && tar -cvf "$DEST/$g.tar" "$g" ) > "$DEST/$g.tar.list" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "[FAIL] tar $g rc=$rc -- see $DEST/$g.tar.list"; rc_all=1; continue
  fi
  echo "[ver ] $g   ($(date -Is))"
  if ! tar -tf "$DEST/$g.tar" > /dev/null; then
    echo "[FAIL] $g.tar is not readable back"; rc_all=1; continue
  fi
  ( cd "$DEST" && sha256sum "$g.tar" > "$g.tar.sha256" )
  n_src=$(find "$SRC/$g" -type f | wc -l)
  n_tar=$(grep -cv '/$' "$DEST/$g.tar.list")
  echo "[ok  ] $g: $n_src files on disk, $n_tar members in tar"
  [ "$n_src" -eq "$n_tar" ] || { echo "[WARN] member count differs for $g"; rc_all=1; }
  ls -lh "$DEST/$g.tar"
  echo
done

echo "=== summary ==="
ls -lh "$DEST"
du -sh "$DEST"
echo "inodes used in DEST: $(find "$DEST" | wc -l)"
echo "end   : $(date -Is)"
if [ $rc_all -eq 0 ]; then
  echo "ALL GROUPS OK."
  echo "Next (manual, only now that the tarballs verify): reclaim the 93,785 inodes the"
  echo "failed raw copy left on prk:"
  echo "  rm -rf /lustre/fsstor/projects/rech/prk/ulx34io/titan_recovery/science_data"
else
  echo "SOME GROUPS FAILED -- do NOT delete anything at the source or on prk."
fi
exit $rc_all
