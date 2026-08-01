#!/bin/bash
#SBATCH --job-name=score_ladder
#SBATCH --account=nzu@a100
#SBATCH -C a100
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=logs/score_ladder_%j.out
#SBATCH --error=logs/score_ladder_%j.out
#
# MOPED-compress at several ell-rebinnings, train an NPE on each, and measure the resulting
# posterior. Answers end-to-end what the Fisher scan could only bound: does skipping the rebinning
# before compression actually buy tighter, still-calibrated contours?
#
# WHY THIS IS CHEAP: MOPED emits 6 numbers regardless of input dimension, so every NDE here trains
# on a 6-dim summary and is identically sized whether the input had 50 features or 2470. Only the
# numpy compression step scales with rebin.
#
# WHY IT BEATS THE FISHER SCAN: the Fisher result assumes C is the true covariance, and at fine
# rebin only the analytic Gaussian is available (the hybrid's low-rank SSC/cNG term comes from 200
# permutations and is noise past a few hundred features). The NDE learns p(theta|t) from sims, so an
# imperfect C costs INFORMATION without BIASING the posterior — the contour is honest and directly
# measurable, and TARP/SBC checks it.
#
# CUTS ARE MATCHED (both arms at the same ell_max). Comparing BNT@460 against non-BNT@580 mixes two
# analyses and is what made the BNT advantage appear to evaporate at native resolution in the Fisher
# scan; at matched cuts it does not.
#
# Probe first, then scale: run with REBINS=20 to time one rung against the existing r20 result
# (which it should reproduce), then submit the full ladder with a walltime sized from that log.
#
#   sbatch scripts/jz/score_rebin_ladder.slurm
#   REBINS=20 sbatch scripts/jz/score_rebin_ladder.slurm      # timing/validation probe
set -euo pipefail
cd /lustre/fsn1/projects/rech/prk/ulx34io/bar_impact
mkdir -p logs

REBINS="${REBINS:-20 10 5 2 1}"
ARMS="${ARMS:-nonbnt bnt}"
CUT="${CUT:-460}"
COVK="${COVK:-analytic}"      # the only covariance valid at every rung; keeps the ladder comparable
MODE="${MODE:-moped}"         # moped | raw. raw feeds the flow the z-scored data vector itself
                              # (no compression) and is the CONTROL for the compression claim.
SEEDS="${SEEDS:-41,42,43,44,45}"
EMBED_DIM="${EMBED_DIM:-0}"   # >0 inserts a learned embedding net in the NDE (use with MODE=raw)
EMBED_HIDDEN="${EMBED_HIDDEN:-256,256}"
ROOT="${ROOT:-outputs/score_rebin_ladder}"

PY=/lustre/fswork/projects/rech/nzu/ulx34io/envs/jaxili/bin/python
export FISHER_AREA=14000
export PYTHONNOUSERSITE=1
export ANAW_CLIP="${ANAW_CLIP:-5}"   # ana_whiten clip; 0 disables (the documented bias suspect)
mkdir -p "$ROOT"

echo "=== score rebin ladder  mode=$MODE cut=$CUT covk=$COVK seeds=$SEEDS  $(date) ==="
echo "    rebins: $REBINS   arms: $ARMS"

# --- pre-flight: the cut machinery must still reproduce its oracle ---------------------------
FISHER_REBIN=20 $PY scripts/score_cut_utils.py >/dev/null \
  || { echo "[fatal] score_cut_utils validation failed"; exit 1; }
echo "[pre-flight] score_cut_utils OK"

for RB in $REBINS; do
  for ARM in $ARMS; do
    if [ "$ARM" = "bnt" ]; then CUTS="$CUT,1024,1024,1024"; BNTFLAG="--bnt";
    else CUTS="$CUT,$CUT,$CUT,$CUT"; BNTFLAG=""; fi
    OD="$ROOT/${ARM}_r${RB}_c${CUT}"
    mkdir -p "$OD"

    if [ -f "$OD/nde/null_pooled_${ARM}.npy" ]; then
      echo "--- $ARM r$RB already done, skipping"; continue
    fi

    echo "--- [$(date +%H:%M:%S)] compress($MODE) $ARM rebin=$RB cuts=$CUTS"
    FISHER_REBIN=$RB $PY -u scripts/score_compress_at_rebin.py \
        --cuts "$CUTS" $BNTFLAG --covk "$COVK" --mode "$MODE" --out "$OD/comp" > "$OD/log_comp.txt" 2>&1 \
      || { echo "[warn] compress failed for $ARM r$RB — see $OD/log_comp.txt"; continue; }
    grep -E "nfeat|cond|FoM3|shift" "$OD/log_comp.txt" | sed 's/^/      /'

    echo "--- [$(date +%H:%M:%S)] NDE $ARM rebin=$RB (5 seeds, 6-dim summaries)"
    $PY -u scripts/nde_realnvp_from_summary.py --compressed "$OD/comp" \
        --out "$OD/nde" --tag "$ARM" --seeds "$SEEDS" \
        --total-steps 20000 --save-every 1500 --patience 15 --num-samples 3000 \
        --save-per-seed --tarp-points 400 --tarp-draws 200 \
        --embed-dim "$EMBED_DIM" --embed-hidden "$EMBED_HIDDEN" \
        --gpu 0 --mem-fraction 0.9 > "$OD/log_nde.txt" 2>&1 \
      || { echo "[warn] NDE failed for $ARM r$RB — see $OD/log_nde.txt"; continue; }
    grep -E "NULL|BIASED|theta_tr" "$OD/log_nde.txt" | sed 's/^/      /'
  done
done

echo "=== [$(date +%H:%M:%S)] ladder complete; summarising ==="
$PY - <<'PYEOF'
import glob, json, os
import numpy as np
ROOT = os.environ.get("ROOT", "outputs/score_rebin_ladder")
CUT = os.environ.get("CUT", "460")
def fom(a):
    return 1.0 / np.sqrt(np.linalg.det(np.cov(a[:, :3], rowvar=False)))
rows = {}
for d in sorted(glob.glob(f"{ROOT}/*_r*_c{CUT}")):
    arm, rb = os.path.basename(d).split("_")[0], int(os.path.basename(d).split("_r")[1].split("_")[0])
    per = []
    for f in sorted(glob.glob(f"{d}/nde/null_s*_{arm}.npy")):
        try:
            a = np.load(f)
        except Exception:
            continue
        if (a == 0).mean() > 0.5 or not np.isfinite(a).all():
            continue
        per.append(a)
    if not per:
        continue
    q = json.load(open(f"{d}/comp/compress_qa.json"))
    rows[(arm, rb)] = dict(n=len(per), nfeat=q["nfeat"], fisher=q["fisher_fom3"],
                           npe_mean=float(np.mean([fom(a) for a in per])),
                           npe_pool=float(fom(np.concatenate(per))))
print(f"\n{'arm':8s}{'rebin':>6}{'nfeat':>7}{'seeds':>6}{'Fisher FoM3':>14}{'NPE FoM3 (mean)':>18}{'NPE (pooled)':>15}{'NPE/Fisher':>12}")
for (arm, rb), v in sorted(rows.items(), key=lambda kv: (-kv[0][1], kv[0][0])):
    print(f"{arm:8s}{rb:>6}{v['nfeat']:>7}{v['n']:>6}{v['fisher']:>14.4e}"
          f"{v['npe_mean']:>18.4e}{v['npe_pool']:>15.4e}{v['npe_mean']/v['fisher']:>12.3f}")
print(f"\n{'rebin':>6}{'MATCHED BNT/nonBNT (NPE)':>28}{'(Fisher)':>12}")
for rb in sorted({k[1] for k in rows}, reverse=True):
    if ("bnt", rb) in rows and ("nonbnt", rb) in rows:
        b, n = rows[("bnt", rb)], rows[("nonbnt", rb)]
        print(f"{rb:>6}{b['npe_mean']/n['npe_mean']:>28.3f}{b['fisher']/n['fisher']:>12.3f}")
PYEOF
echo "=== done $(date) ==="
