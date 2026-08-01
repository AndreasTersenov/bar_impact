#!/bin/bash
#SBATCH --job-name=arch_score
#SBATCH --account=nzu@cpu
#SBATCH --partition=archive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --time=04:00:00
#SBATCH --output=logs/archive_score_%j.out
#SBATCH --error=logs/archive_score_%j.out
#
# Archive outputs/score_* -- every trained NDE posterior behind the BNT scale-cut result --
# to $STORE as one tarball per experiment directory.
#
# WHY THIS EXISTS. These trees are gitignored by design (they are outputs, not source), they
# live on $SCRATCH, and nothing else holds them. That is two independent ways to lose them:
# the Titan RAID0 failure that this whole recovery effort exists to undo, and the routine
# 30-day-idle purge that $SCRATCH applies to anything untouched. Every FoM in the paper is
# read out of these directories.
#
# WHY TAR, NOT rsync.  outputs/score_* is 18,798 files -- 15,857 of them tiny orbax
# checkpoint .json shards. The nzu STORE inode quota is 100,000, TEAM-SHARED, and already
# 81,934 used (82%). A file-for-file copy would consume 18,798 of the ~18,000 remaining
# inodes and take the whole team to zero, to store 1.5 GiB against a 50 TiB allowance that
# is only 36% full. Bytes are not the constraint here; inodes are. One tarball per
# experiment costs 3 inodes (tar + list + sha256) instead of thousands.
#
# MUST run on --partition=archive. $STORE is not mounted on cpu_p1 compute nodes; a job
# submitted there dies instantly with a Permission denied on /lustre/fsstor. archive is the
# partition carrying the STORE mount (20 h limit, 1 core, no allocation charge), and this
# job is pure I/O anyway.
#
# ONE TARBALL PER EXPERIMENT, not one for everything. Restoring a single run should not
# require unpacking 1.5 GiB, and the per-experiment .list lets you find a file without
# reading any tarball at all. Extract one member with:
#     tar -xf <exp>.tar -C <dest> <member-path>
#
# Re-running is safe: an experiment with a verified .sha256 is skipped. Delete that file to
# force a rebuild.
#
# Submit from the repo root:  sbatch scripts/jz/archive_score_outputs_to_store.slurm

set -uo pipefail

SRC=/lustre/fsn1/projects/rech/prk/ulx34io/bar_impact/outputs
DEST=/lustre/fsstor/projects/rech/nzu/ulx34io/titan_recovery/bar_impact_posteriors

mkdir -p "$DEST" || { echo "[fatal] cannot create $DEST -- are you on --partition=archive?"; exit 1; }

echo "=== archive outputs/score_* -> STORE (tar) ==="
echo "src   : $SRC"
echo "dest  : $DEST"
echo "start : $(date -Is)"
df -h "$DEST" | tail -1
echo

# Discover the experiments rather than hardcoding them, so a run added later is not silently
# skipped by a stale list. Bare names only (no path), which is what tar's -C form wants.
#
# NOT named GROUPS: that is a bash builtin array holding the invoking user's numeric group
# IDs. Assigning to it is silently ignored, and a loop over it then iterates the GIDs,
# tarring nothing while still reporting success. This bit us once already.
mapfile -t EXPS < <(cd "$SRC" && find . -maxdepth 1 -mindepth 1 -type d -name 'score_*' -printf '%f\n' | sort)

if [ "${#EXPS[@]}" -eq 0 ]; then
  echo "[fatal] no score_* directories found under $SRC -- refusing to report success"
  exit 1
fi

echo "found ${#EXPS[@]} experiment directories:"
printf '  %s\n' "${EXPS[@]}"
echo

n_src_total=$(cd "$SRC" && find "${EXPS[@]}" -type f | wc -l)
echo "total files to archive: $n_src_total"
echo

rc_all=0
n_done=0
n_skip=0

for g in "${EXPS[@]}"; do
  if [ -f "$DEST/$g.tar.sha256" ]; then
    echo "[skip] $g (already has a checksum -- delete it to force a rebuild)"
    n_skip=$((n_skip + 1))
    continue
  fi

  echo "[tar ] $g   ($(date -Is))"
  # -v output IS the index; it goes to .list, so tar's own stdout must not be the tarball.
  ( cd "$SRC" && tar -cvf "$DEST/$g.tar" "$g" ) > "$DEST/$g.tar.list" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "[FAIL] tar $g rc=$rc -- see $DEST/$g.tar.list"
    rc_all=1
    continue
  fi

  # Read the archive back before trusting it. A tar that writes cleanly can still be
  # unreadable if the destination filled mid-write.
  echo "[ver ] $g   ($(date -Is))"
  if ! tar -tf "$DEST/$g.tar" > /dev/null; then
    echo "[FAIL] $g.tar is not readable back"
    rc_all=1
    continue
  fi

  ( cd "$DEST" && sha256sum "$g.tar" > "$g.tar.sha256" )

  # Member count must match the source. Catches a tar that stopped early with rc=0.
  n_src=$(cd "$SRC" && find "$g" -type f | wc -l)
  n_tar=$(grep -cv '/$' "$DEST/$g.tar.list")
  echo "[ok  ] $g: $n_src files on disk, $n_tar members in tar   ($(du -h "$DEST/$g.tar" | cut -f1))"
  if [ "$n_src" -ne "$n_tar" ]; then
    echo "[WARN] member count differs for $g -- do not delete the source"
    rc_all=1
  fi
  n_done=$((n_done + 1))
done

echo
echo "=== summary ==="
echo "archived : $n_done"
echo "skipped  : $n_skip"
ls -lh "$DEST"
echo
du -sh "$DEST"
echo "inodes used in DEST: $(find "$DEST" | wc -l)   (a raw copy would have used $n_src_total)"
echo "end   : $(date -Is)"

if [ $rc_all -eq 0 ]; then
  echo "ALL EXPERIMENTS OK."
  echo "The source stays where it is -- this is a backup, not a migration. outputs/ is on"
  echo "\$SCRATCH and will be purged if it goes 30 days untouched; these tarballs are what"
  echo "survives that."
else
  echo "SOME EXPERIMENTS FAILED -- do NOT delete anything at the source."
fi
exit $rc_all
