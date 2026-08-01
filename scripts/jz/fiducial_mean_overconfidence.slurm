#!/bin/bash
#SBATCH --job-name=fidmean
#SBATCH --account=nzu@a100
#SBATCH -C a100
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=01:00:00
#SBATCH --output=logs/fidmean_%j.out
#SBATCH --error=logs/fidmean_%j.out
#
# Settle the one open question in the BNT thread: is the embedding's 1.25-1.36x excess over the
# Gaussian Fisher real non-Gaussian information, or over-confidence from evaluating the flow at an
# observation 14x cleaner than anything it trained on?
#
# NO TRAINING HAPPENS HERE. The flows in $ROOT are loaded from their checkpoints and merely
# re-evaluated at different conditioning values. That is the whole point: holding the network fixed
# isolates the evaluation point from training noise, which a retrain would confound. It is also why
# an hour on the dev QoS is ample where the original run needed the full ladder.
#
# Cost is dominated by loading the 16,965-row grid twice (once per arm), not by the ~220 posterior
# draws, each of which is a forward pass through a 173k-parameter flow.
#
#   sbatch scripts/jz/fiducial_mean_overconfidence.slurm
#   N_OBS=50 sbatch scripts/jz/fiducial_mean_overconfidence.slurm    # tighter percentiles
set -euo pipefail
cd /lustre/fsn1/projects/rech/prk/ulx34io/bar_impact
mkdir -p logs outputs/diagnostics

ROOT="${ROOT:-outputs/score_embed_anaw16}"   # the run behind the paper figure
ARMS="${ARMS:-bnt,nonbnt}"
SEEDS="${SEEDS:-41,42,43,44,45}"
N_OBS="${N_OBS:-20}"
OUT="${OUT:-outputs/diagnostics/fiducial_mean_overconfidence.json}"

PY=/lustre/fswork/projects/rech/nzu/ulx34io/envs/jaxili/bin/python
export FISHER_AREA=14000
export FISHER_REBIN=20
export PYTHONNOUSERSITE=1

echo "=== fiducial-mean over-confidence test ==="
echo "root  : $ROOT"
echo "arms  : $ARMS   seeds: $SEEDS   n_obs: $N_OBS"
echo "start : $(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo

srun $PY -u scripts/diagnostics/fiducial_mean_overconfidence.py \
    --root "$ROOT" --arms "$ARMS" --seeds "$SEEDS" --n-obs "$N_OBS" --out "$OUT"
rc=$?

echo
echo "end   : $(date -Is)   rc=$rc"
exit $rc
