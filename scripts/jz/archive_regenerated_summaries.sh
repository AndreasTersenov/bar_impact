#!/bin/bash
# Archive the regenerated matched-noise summaries to $STORE.
#
# WHY THESE NEED ARCHIVING. They live under cosmogrid_products/ on $SCRATCH, which purges anything
# untouched for 30 days, and they are NOT covered by the July stage3_forecast tarballs -- those
# predate this regeneration. They are reproducible from committed scripts with deterministic seeds
# (~3 min for the power spectra, ~20 min for the HOS), so this is convenience rather than rescue.
# Archiving them anyway costs 34 MB and means a rerun is never on the critical path.
#
# NOT A SLURM JOB, unlike the other two archive scripts here. Those move tens of GB and must run on
# --partition=archive because $STORE is not mounted on cpu_p1. $STORE IS reachable from a login
# node, and 34 MB is well inside the login-node budget, so this just runs.
#
# STILL TAR, THOUGH. 68 files against an nzu STORE inode quota that is 82% used and team-shared.
# Three inodes instead of 68 is free to do and consistent with how everything else here is stored.
#
#   bash scripts/jz/archive_regenerated_summaries.sh
set -uo pipefail

SRC=/lustre/fsn1/projects/rech/prk/ulx34io/cosmogrid_products/stage3_forecast/fiducial/cosmo_fiducial
DEST=/lustre/fsstor/projects/rech/nzu/ulx34io/titan_recovery/regenerated_summaries
TAG=matchednoise

mkdir -p "$DEST" || { echo "[fatal] cannot create $DEST"; exit 1; }
cd "$SRC" || exit 1

mapfile -t FILES < <(ls | grep "$TAG" | sort)
if [ "${#FILES[@]}" -eq 0 ]; then
  echo "[fatal] no files matching '$TAG' in $SRC -- refusing to write an empty archive"
  exit 1
fi
echo "=== archive regenerated summaries -> STORE ==="
echo "src   : $SRC"
echo "dest  : $DEST"
echo "files : ${#FILES[@]}   ($(du -ch "${FILES[@]}" | tail -1 | cut -f1))"
echo "start : $(date -Is)"

STAMP=$(date +%Y%m%d)
OUT="$DEST/${TAG}_${STAMP}.tar"
if [ -f "$OUT.sha256" ]; then
  echo "[skip] $OUT already has a checksum -- delete it to force a rebuild"
  exit 0
fi

tar -cvf "$OUT" "${FILES[@]}" > "$OUT.list" 2>&1
rc=$?
[ $rc -ne 0 ] && { echo "[FAIL] tar rc=$rc -- see $OUT.list"; exit 1; }

# Read it back before trusting it: a tar can write cleanly and still be unreadable if the
# destination filled mid-write.
tar -tf "$OUT" > /dev/null || { echo "[FAIL] $OUT is not readable back"; exit 1; }
( cd "$DEST" && sha256sum "$(basename "$OUT")" > "$(basename "$OUT").sha256" )

n_tar=$(grep -cv '/$' "$OUT.list")
echo "[ok  ] ${#FILES[@]} files on disk, $n_tar members in tar   ($(du -h "$OUT" | cut -f1))"
[ "${#FILES[@]}" -eq "$n_tar" ] || { echo "[WARN] member count differs"; exit 1; }

echo
ls -lh "$DEST"
echo "end   : $(date -Is)"
echo "ALL OK. The source stays put -- this is a backup, not a migration."
