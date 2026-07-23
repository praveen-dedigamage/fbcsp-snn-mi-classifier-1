#!/bin/bash
#SBATCH --job-name=ann_twin
#SBATCH --account=project_2003397
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --array=1-45
#SBATCH --output=logs/ann_twin_S%a_%j.out
#SBATCH --error=logs/ann_twin_S%a_%j.err

# ============================================================
# ANN twin (non-spiking) -- CSC Puhti SLURM array job
#
# Trains ANNClassifier -- architecturally identical to SNNClassifier
# (same layer sizes, same leaky-integration recurrence, same optimiser,
# same input features) but with the LIF spike-and-reset replaced by a
# continuous ReLU -- to isolate whether spiking itself contributes
# anything. Reuses an ALREADY-COMPLETED SNN training run's saved CSP
# filters, z-norm stats, and MIBIF selector (same encoded input the SNN
# received), so only the model + optionally the loss differ.
#
# Submit via submit_ann_twin.sh, which sets RESULTS_DIR, MOABB_DATASET,
# and LOSS_TYPE for you.
# ============================================================

set -euo pipefail

N_FOLDS="${N_FOLDS:-5}"
TASK_ID=${SLURM_ARRAY_TASK_ID}

RESULTS_DIR="${RESULTS_DIR:?Set RESULTS_DIR to an existing completed results dir}"
MOABB_DATASET="${MOABB_DATASET:?Set MOABB_DATASET (e.g. BNCI2014_001)}"
LOSS_TYPE="${LOSS_TYPE:-van_rossum}"

SUBJECT_ID=$(( (TASK_ID - 1) / N_FOLDS + 1 ))
FOLD_IDX=$(( (TASK_ID - 1) % N_FOLDS ))

PROJECT_DIR=/scratch/project_2003397/praveen/fbcsp-snn-mi-classifier-1

echo "=============================================="
echo "  ANN twin (non-spiking, loss=${LOSS_TYPE})"
echo "  Task:    ${TASK_ID}  ->  Subject ${SUBJECT_ID}, Fold ${FOLD_IDX}"
echo "  Dataset: ${MOABB_DATASET}"
echo "  Results: ${RESULTS_DIR}"
echo "  Start:   $(date)"
echo "=============================================="

unset SINGULARITY_BIND
unset APPTAINER_BIND

module purge
source "${PROJECT_DIR}/.venv/bin/activate"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export MNE_DATA=/scratch/project_2003397/praveen/mne_data

cd "${PROJECT_DIR}"

python run_ann_twin.py \
    --results-dir "${RESULTS_DIR}" \
    --subject-id "${SUBJECT_ID}" \
    --fold "${FOLD_IDX}" \
    --moabb-dataset "${MOABB_DATASET}" \
    --loss-type "${LOSS_TYPE}" \
    --n-folds "${N_FOLDS}"

EXIT_CODE=$?
echo ""
echo "Task ${TASK_ID} finished with exit code ${EXIT_CODE}"
echo "End: $(date)"
exit ${EXIT_CODE}
