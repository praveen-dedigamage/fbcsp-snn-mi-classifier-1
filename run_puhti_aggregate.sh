#!/bin/bash
#SBATCH --job-name=fbcsp_agg
#SBATCH --account=project_2003397
#SBATCH --partition=small
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/fbcsp_agg_S%x_%j.out
#SBATCH --error=logs/fbcsp_agg_S%x_%j.err

# ============================================================
# FBCSP-SNN — Per-subject aggregation
# SUBJECT_ID is injected by submit_puhti.sh via --export.
# Dependency is set to only the fold tasks for this subject,
# so each subject aggregates as soon as its folds complete.
# ============================================================

set -euo pipefail

PROJECT_DIR=/scratch/project_2003397/praveen/fbcsp-snn-mi-classifier-1

# SUBJECT_ID and RESULTS_DIR are injected by submit_puhti.sh via --export.
if [ -z "${SUBJECT_ID:-}" ]; then
    echo "ERROR: SUBJECT_ID not set. Use sbatch --export=ALL,SUBJECT_ID=N" >&2
    exit 1
fi
RESULTS_DIR="${RESULTS_DIR:-Results}"

# Read N_FOLDS from the training array script — single source of truth
N_FOLDS=$(grep '^N_FOLDS=' "${PROJECT_DIR}/run_puhti_array.sh" | cut -d= -f2)

echo "=============================================="
echo "  FBCSP-SNN Aggregation — Subject ${SUBJECT_ID}"
echo "  N_FOLDS:     ${N_FOLDS}"
echo "  RESULTS_DIR: ${RESULTS_DIR}"
echo "  Node:        $(hostname)"
echo "  Start:       $(date)"
echo "=============================================="

unset SINGULARITY_BIND
unset APPTAINER_BIND

source "${PROJECT_DIR}/.venv/bin/activate"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export MNE_DATA=/scratch/project_2003397/praveen/mne_data

cd "${PROJECT_DIR}"

python main.py aggregate \
    --source moabb \
    --moabb-dataset BNCI2014_001 \
    --subject-id "${SUBJECT_ID}" \
    --n-folds "${N_FOLDS}" \
    --results-dir "${RESULTS_DIR}"

echo ""
echo "Subject ${SUBJECT_ID} aggregation complete."
echo "End: $(date)"
