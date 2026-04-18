#!/bin/bash
#SBATCH --job-name=fbcsp_lava
#SBATCH --account=project_2003397
#SBATCH --partition=small
#SBATCH --array=1-9
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
# Wall-time: step 1 (save spikes, 5 folds) ~3 min + step 2 (Lava infer) ~5 min = ~10 min
# 30 min gives 3× buffer
#SBATCH --output=logs/fbcsp_lava_S%a_%j.out
#SBATCH --error=logs/fbcsp_lava_S%a_%j.err

# ============================================================
# FBCSP-SNN — Lava simulation (item 8)
#
# Two-step pipeline in one SLURM task:
#   Step 1 (.venv)      — save test spike tensors to fold dirs
#   Step 2 (.venv_lava) — run SLAYER inference + .net export
#
# Array job: one task per subject (task ID = subject ID).
# No GPU needed — CPU-only.
#
# Override defaults via env vars:
#   RESULTS_DIR=Results_adm_static6_ptq   (default)
#   OUTPUT_DIR=Results_lava                (default)
# ============================================================

set -euo pipefail

SUBJECT_ID=${SLURM_ARRAY_TASK_ID}

RESULTS_DIR="${RESULTS_DIR:-Results_adm_static6_ptq}"
OUTPUT_DIR="${OUTPUT_DIR:-Results_lava}"

PROJECT_DIR=/scratch/project_2003397/praveen/fbcsp-snn-mi-classifier-1
VENV_MAIN="${PROJECT_DIR}/.venv"
VENV_LAVA="/scratch/project_2003397/praveen/.venv_lava"

echo "=============================================="
echo "  FBCSP-SNN Lava Simulation (item 8)"
echo "  Subject:     ${SUBJECT_ID}"
echo "  RESULTS_DIR: ${RESULTS_DIR}"
echo "  OUTPUT_DIR:  ${OUTPUT_DIR}"
echo "  Node:        $(hostname)"
echo "  Start:       $(date)"
echo "=============================================="

unset SINGULARITY_BIND
unset APPTAINER_BIND
cd "${PROJECT_DIR}"
mkdir -p logs "${OUTPUT_DIR}"

# ---- Step 1: save test spikes (.venv, has moabb/snnTorch) ----
echo ""
echo "--- Step 1: saving test spikes (main venv) ---"
source "${VENV_MAIN}/bin/activate"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export MNE_DATA=/scratch/project_2003397/praveen/mne_data

python save_test_spikes.py \
    --results-dir "${RESULTS_DIR}" \
    --subjects "${SUBJECT_ID}" \
    --n-folds 5 \
    --moabb-dataset BNCI2014_001

deactivate
echo "Step 1 done: $(date)"

# ---- Step 2: Lava SLAYER inference (.venv_lava, has lava-dl) ----
echo ""
echo "--- Step 2: Lava inference (lava venv) ---"
source "${VENV_LAVA}/bin/activate"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

python run_lava_infer.py \
    --results-dir "${RESULTS_DIR}" \
    --subjects "${SUBJECT_ID}" \
    --n-folds 5 \
    --output-dir "${OUTPUT_DIR}"

EXIT_CODE=$?
deactivate

echo ""
echo "Subject ${SUBJECT_ID} Lava simulation finished — exit code ${EXIT_CODE}"
echo "End: $(date)"
exit ${EXIT_CODE}
