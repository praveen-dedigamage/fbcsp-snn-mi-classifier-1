#!/bin/bash
#SBATCH --job-name=fbcsp_e2e
#SBATCH --account=project_2003397
#SBATCH --partition=small
#SBATCH --array=1-9
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
# Wall-time budget: 5 folds × 100 draws × ~11 s/draw ≈ 92 min
#                  + data load / quant overhead         ≈  5 min
#                  = ~97 min; 2 h gives ~25% buffer
#SBATCH --output=logs/fbcsp_e2e_S%a_%j.out
#SBATCH --error=logs/fbcsp_e2e_S%a_%j.err

# ============================================================
# FBCSP-SNN — End-to-End Hardware Stress Test (item 7)
#
# Loads FP32-trained artifacts and evaluates test accuracy with:
#   σ=2% correlated filter mismatch + 4-bit CSP + INT8 SNN
#
# Array job: one task per subject (task ID = subject ID).
# No GPU needed — small partition.
#
# Override defaults via env vars:
#   RESULTS_DIR=Results_adm_static6_ptq   (default)
#   OUTPUT_DIR=Results_e2e_stress          (default)
#   SIGMA=0.02                             (default)
#   CSP_BITS=4                             (default)
#   SNN_BITS=8                             (default)
#   N_DRAWS=100                            (default)
# ============================================================

set -euo pipefail

SUBJECT_ID=${SLURM_ARRAY_TASK_ID}

RESULTS_DIR="${RESULTS_DIR:-Results_adm_static6_ptq}"
OUTPUT_DIR="${OUTPUT_DIR:-Results_e2e_stress}"
SIGMA="${SIGMA:-0.02}"
CSP_BITS="${CSP_BITS:-4}"
SNN_BITS="${SNN_BITS:-8}"
N_DRAWS="${N_DRAWS:-100}"

PROJECT_DIR=/scratch/project_2003397/praveen/fbcsp-snn-mi-classifier-1

echo "=============================================="
echo "  FBCSP-SNN End-to-End Hardware Stress Test"
echo "  Subject:     ${SUBJECT_ID}"
echo "  RESULTS_DIR: ${RESULTS_DIR}"
echo "  OUTPUT_DIR:  ${OUTPUT_DIR}"
echo "  SIGMA:       ${SIGMA}"
echo "  CSP_BITS:    ${CSP_BITS}"
echo "  SNN_BITS:    ${SNN_BITS}"
echo "  N_DRAWS:     ${N_DRAWS}"
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

python run_e2e_stress.py \
    --results-dir "${RESULTS_DIR}" \
    --subjects "${SUBJECT_ID}" \
    --n-folds 5 \
    --sigma "${SIGMA}" \
    --csp-bits "${CSP_BITS}" \
    --snn-bits "${SNN_BITS}" \
    --n-draws "${N_DRAWS}" \
    --output-dir "${OUTPUT_DIR}" \
    --moabb-dataset BNCI2014_001

EXIT_CODE=$?
echo ""
echo "Subject ${SUBJECT_ID} E2E stress test finished with exit code ${EXIT_CODE}"
echo "End: $(date)"
exit ${EXIT_CODE}
