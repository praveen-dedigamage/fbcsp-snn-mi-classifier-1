#!/bin/bash
# ============================================================================
#  Roihu — BNCI2015-001 training array (12 subjects x 5 folds = 60 tasks)
#
#      sbatch roihu/01_train_array.sh
#
#  Partition/GRES verified against CSC docs (batch-job-partitions, Aug 2026):
#      gputest    15 min   1-2 nodes   1-4 GPUs
#      gpumedium  36 h     1-4 nodes   1-4 GPUs   <- used here
#      gpularge   36 h     1-10 nodes  4 GPUs only (whole nodes)
#  Each reserved GH200 grants 72 CPU cores and 217 GiB total memory.
#
#  Walltime rationale: measured 33 s/epoch on an RTX 3060 laptop, with the
#  fold needing ~250 epochs under the paper protocol (patience 100) -> ~2.3 h.
#  H100 should be faster, but the LIF loop is sequential over T=2561 and was
#  only 17-21 % GPU-utilised locally, i.e. launch-overhead bound rather than
#  compute bound, so do NOT assume a large speedup. 4 h is a safe margin;
#  gpumedium permits up to 36 h if a fold overruns.
# ============================================================================
#SBATCH --job-name=fbcsp_snn_train
#SBATCH --account=project_XXXXXXX          # <-- EDIT: your CSC project
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=1-60
#SBATCH --output=logs/train_%A_%a.out
#SBATCH --error=logs/train_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

# ---- configuration ---------------------------------------------------------
DATASET="BNCI2015_001"
RESULTS_DIR="${RESULTS_DIR:-Results_bnci2015}"
N_FOLDS=5
SEED=42
PYTORCH_MODULE="${PYTORCH_MODULE:-pytorch}"   # set to what `module spider pytorch` reports

# ---- map array index -> (subject, fold) ------------------------------------
TASK=${SLURM_ARRAY_TASK_ID}
SUBJECT_ID=$(( (TASK - 1) / N_FOLDS + 1 ))
FOLD_IDX=$(( (TASK - 1) % N_FOLDS ))

module purge
module load "${PYTORCH_MODULE}"

echo "=============================================="
echo "  Roihu task ${TASK}: subject ${SUBJECT_ID}, fold ${FOLD_IDX}"
echo "  node    : $(hostname)   arch: $(uname -m)"
echo "  dataset : ${DATASET}"
echo "  seed    : ${SEED}"
echo "  started : $(date)"
echo "=============================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Flags below reproduce the configuration that produced the paper's results
# (worktree run_puhti_array.sh): fixed six-band bank, dual-end CSP with m=4,
# adaptive MIBIF at mi_fraction=0.1. --seed is new and makes runs reproducible.
srun python -u main.py train \
    --source moabb \
    --moabb-dataset "${DATASET}" \
    --subject-id "${SUBJECT_ID}" \
    --fold "${FOLD_IDX}" \
    --n-folds "${N_FOLDS}" \
    --freq-bands "[(4,8),(8,14),(12,18),(16,24),(20,30),(26,40)]" \
    --csp-components-per-band 8 \
    --hidden-neurons 64 \
    --population-per-class 20 \
    --beta 0.95 \
    --dropout-prob 0.5 \
    --lr 1e-3 \
    --weight-decay 0.1 \
    --epochs 1000 \
    --early-stopping-patience 100 \
    --early-stopping-warmup 100 \
    --spiking-prob 0.7 \
    --feature-selection-method mibif \
    --mi-fraction 0.1 \
    --seed "${SEED}" \
    --results-dir "${RESULTS_DIR}"

echo "finished: $(date)"
