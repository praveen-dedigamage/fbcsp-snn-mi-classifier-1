#!/bin/bash
#SBATCH --job-name=fbcsp_agg
#SBATCH --account=project_2003397
#SBATCH --partition=small
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --array=1-9                    # one task per subject
#SBATCH --output=logs/fbcsp_agg_S%a_%j.out
#SBATCH --error=logs/fbcsp_agg_S%a_%j.err

# ============================================================
# FBCSP-SNN — Per-subject aggregation (array job)
# Each task reads fold JSONs for one subject and writes
# summary.csv + confusion matrix plots.
#
# Runs automatically after the training array via submit_puhti.sh
# ============================================================

set -euo pipefail

SUBJECT_ID=${SLURM_ARRAY_TASK_ID}
PROJECT_DIR=/scratch/project_2003397/praveen/fbcsp-snn-mi-classifier-1

# Read N_FOLDS from the training array script — single source of truth
N_FOLDS=$(grep '^N_FOLDS=' "${PROJECT_DIR}/run_puhti_array.sh" | cut -d= -f2)

echo "=============================================="
echo "  FBCSP-SNN Aggregation — Subject ${SUBJECT_ID}"
echo "  N_FOLDS: ${N_FOLDS}"
echo "  Node:    $(hostname)"
echo "  Start:   $(date)"
echo "=============================================="

source "${PROJECT_DIR}/.venv/bin/activate"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export MNE_DATA=/scratch/project_2003397/praveen/mne_data

cd "${PROJECT_DIR}"

python main.py aggregate \
    --source moabb \
    --moabb-dataset BNCI2014_001 \
    --subject-id "${SUBJECT_ID}" \
    --n-folds "${N_FOLDS}"

echo ""
echo "Subject ${SUBJECT_ID} aggregation complete."
echo "End: $(date)"
