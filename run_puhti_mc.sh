#!/bin/bash
#SBATCH --job-name=fbcsp_mc
#SBATCH --account=project_2003397
#SBATCH --partition=small
#SBATCH --array=1-9
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:35:00
#SBATCH --output=logs/fbcsp_mc_S%a_%j.out
#SBATCH --error=logs/fbcsp_mc_S%a_%j.err

# ============================================================
# FBCSP-SNN — Butterworth Monte Carlo (Gm-C mismatch)
#
# Array job: one task per subject (task ID = subject ID).
# No GPU needed — small partition.
# Wall time: ~25 min per subject (5 folds × 300 draws × ~1s)
#
# Usage:
#   sbatch run_puhti_mc.sh
#
#   # Custom results dir or subset of subjects:
#   sbatch --array=1,3,5 --export=ALL,RESULTS_DIR=Results_adm_static6_ptq run_puhti_mc.sh
#
# Override defaults via env vars:
#   RESULTS_DIR=Results_adm_static6_ptq   (default)
#   N_DRAWS=100                           (default)
#   SIGMAS="0.01 0.02 0.05"              (default)
#   OUTPUT_DIR=Results_butterworth_mc     (default)
# ============================================================

set -euo pipefail

SUBJECT_ID=${SLURM_ARRAY_TASK_ID}

RESULTS_DIR="${RESULTS_DIR:-Results_adm_static6_ptq}"
N_DRAWS="${N_DRAWS:-100}"
SIGMAS="${SIGMAS:-0.01 0.02 0.05}"
OUTPUT_DIR="${OUTPUT_DIR:-Results_butterworth_mc}"

PROJECT_DIR=/scratch/project_2003397/praveen/fbcsp-snn-mi-classifier-1

echo "=============================================="
echo "  FBCSP-SNN Butterworth Monte Carlo"
echo "  Subject:     ${SUBJECT_ID}"
echo "  RESULTS_DIR: ${RESULTS_DIR}"
echo "  N_DRAWS:     ${N_DRAWS}"
echo "  SIGMAS:      ${SIGMAS}"
echo "  OUTPUT_DIR:  ${OUTPUT_DIR}"
echo "  Node:        $(hostname)"
echo "  Start:       $(date)"
echo "=============================================="

unset SINGULARITY_BIND
unset APPTAINER_BIND

source "${PROJECT_DIR}/.venv/bin/activate"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export MNE_DATA=/scratch/project_2003397/praveen/mne_data

cd "${PROJECT_DIR}"
mkdir -p logs

python run_butterworth_mc.py \
    --results-dir "${RESULTS_DIR}" \
    --subjects "${SUBJECT_ID}" \
    --n-folds 5 \
    --sigmas ${SIGMAS} \
    --n-draws "${N_DRAWS}" \
    --output-dir "${OUTPUT_DIR}" \
    --moabb-dataset BNCI2014_001

EXIT_CODE=$?
echo ""
echo "Subject ${SUBJECT_ID} Monte Carlo finished with exit code ${EXIT_CODE}"
echo "End: $(date)"
exit ${EXIT_CODE}
