#!/bin/bash
#SBATCH --job-name=fbcsp_dstest
#SBATCH --account=project_2003397
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --array=1-3
#SBATCH --output=logs/fbcsp_dstest_%a_%j.out
#SBATCH --error=logs/fbcsp_dstest_%a_%j.err

# ============================================================
# FBCSP-SNN — Multi-dataset pipeline compatibility test
#
# Runs subject 1, fold 0 for each of the three secondary datasets
# to verify that data loading, CSP, spike encoding, and SNN
# all work correctly before committing to a full 9-subject run.
#
# Task 1 → PhysionetMI  (4-class, 64 ch, 160 Hz, single-session)
# Task 2 → Cho2017      (2-class, 64 ch, 512 Hz, single-session)
# Task 3 → BNCI2015_001 (2-class, 13 ch, 512 Hz, two sessions)
#
# Submit:
#   cd /scratch/project_2003397/praveen/fbcsp-snn-mi-classifier-1
#   sbatch run_puhti_dataset_test.sh
# ============================================================

set -euo pipefail

# Map task ID to dataset name
declare -A DATASET_MAP
DATASET_MAP[1]="PhysionetMI"
DATASET_MAP[2]="Cho2017"
DATASET_MAP[3]="BNCI2015_001"

DATASET="${DATASET_MAP[${SLURM_ARRAY_TASK_ID}]}"
RESULTS_DIR="Results_dataset_test"

PROJECT_DIR=/scratch/project_2003397/praveen/fbcsp-snn-mi-classifier-1

echo "=============================================="
echo "  FBCSP-SNN  dataset compatibility test"
echo "  Task:    ${SLURM_ARRAY_TASK_ID}  →  ${DATASET}"
echo "  Subject: 1,  Fold: 0"
echo "  Results: ${RESULTS_DIR}/${DATASET}/"
echo "  Node:    $(hostname)"
echo "  GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "  Start:   $(date)"
echo "=============================================="

unset SINGULARITY_BIND
unset APPTAINER_BIND

module purge
source "${PROJECT_DIR}/.venv/bin/activate"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export MNE_DATA=/scratch/project_2003397/praveen/mne_data

cd "${PROJECT_DIR}"

python main.py train \
    --source moabb \
    --moabb-dataset "${DATASET}" \
    --subject-id 1 \
    --fold 0 \
    --n-folds 5 \
    --adaptive-bands \
    --n-adaptive-bands 12 \
    --min-fisher-fraction 0.15 \
    --csp-components-per-band 8 \
    --hidden-neurons 64 \
    --population-per-class 20 \
    --beta 0.95 \
    --dropout-prob 0.5 \
    --lr 1e-3 \
    --weight-decay 0.1 \
    --epochs 200 \
    --early-stopping-patience 50 \
    --early-stopping-warmup 50 \
    --spiking-prob 0.7 \
    --feature-selection-method mibif \
    --mi-fraction 0.1 \
    --augment-windows \
    --results-dir "${RESULTS_DIR}/${DATASET}"

EXIT_CODE=$?

echo ""
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "✓ ${DATASET}: pipeline ran successfully"
else
    echo "✗ ${DATASET}: FAILED with exit code ${EXIT_CODE}"
fi
echo "End: $(date)"
exit ${EXIT_CODE}
