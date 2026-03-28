#!/bin/bash
#SBATCH --job-name=fbcsp_snn_subj
#SBATCH --account=project_2003397
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=0-4                    # one task per fold (5 folds in parallel)
#SBATCH --output=logs/fbcsp_snn_S%x_f%a_%j.out
#SBATCH --error=logs/fbcsp_snn_S%x_f%a_%j.err

# ============================================================
# FBCSP-SNN — single-subject, fold-parallel SLURM array job
# Each array task runs ONE fold for the specified subject.
#
# Usage:
#   sbatch --job-name=S2 run_puhti_subject.sh 2
#
# The subject ID is passed as the first positional argument.
# SLURM_ARRAY_TASK_ID (0–9) maps directly to --fold.
#
# Monitor:
#   squeue -u $USER
#   for F in $(seq 0 9); do echo "=== fold $F ===" && tail -3 logs/fbcsp_snn_S2_f${F}_*.out 2>/dev/null; done
# ============================================================

set -euo pipefail

SUBJECT_ID=${1:?Usage: sbatch --job-name=S<N> run_puhti_subject.sh <subject_id>}
FOLD_ID=${SLURM_ARRAY_TASK_ID}
PROJECT_DIR=/scratch/project_2003397/praveen/fbcsp-snn-mi-classifier-1

echo "=============================================="
echo "  FBCSP-SNN  Subject ${SUBJECT_ID}  Fold ${FOLD_ID}"
echo "  Node:   $(hostname)"
echo "  GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "  Start:  $(date)"
echo "=============================================="

module purge
source "${PROJECT_DIR}/.venv/bin/activate"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export MNE_DATA=/scratch/project_2003397/praveen/mne_data

cd "${PROJECT_DIR}"

python main.py train \
    --source moabb \
    --moabb-dataset BNCI2014_001 \
    --subject-id "${SUBJECT_ID}" \
    --fold "${FOLD_ID}" \
    --n-folds 5 \
    --adaptive-bands \
    --n-adaptive-bands 6 \
    --csp-components-per-band 4 \
    --hidden-neurons 64 \
    --population-per-class 20 \
    --beta 0.95 \
    --dropout-prob 0.5 \
    --epochs 1000 \
    --early-stopping-patience 100 \
    --early-stopping-warmup 100 \
    --lr 1e-3 \
    --weight-decay 0.1 \
    --spiking-prob 0.7 \
    --feature-selection-method mibif \
    --feature-percentile 50.0 \
    --results-dir Results

EXIT_CODE=$?

echo ""
echo "Fold ${FOLD_ID} finished with exit code ${EXIT_CODE}"
echo "End: $(date)"
exit ${EXIT_CODE}
