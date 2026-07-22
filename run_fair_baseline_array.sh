#!/bin/bash
#SBATCH --job-name=fair_baseline
#SBATCH --account=project_2003397
#SBATCH --partition=small          # CPU-only -- no GPU needed (sklearn PCA/LDA/LinearSVC)
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G                  # flattened vectors are large (~288K dims for 4-class)
#SBATCH --time=0:30:00             # pilot: ~2 min/fold; generous margin for larger datasets
#SBATCH --array=1-45               # default 9 subjects x 5 folds; submit_fair_baseline.sh overrides
#SBATCH --output=logs/fair_baseline_S%a_%j.out
#SBATCH --error=logs/fair_baseline_S%a_%j.err

# ============================================================
# Fairness-controlled baseline -- CSC Puhti SLURM array job
#
# Reuses an ALREADY-COMPLETED training run's saved CSP filters and
# z-norm stats (no SNN retraining, no GPU). Requires RESULTS_DIR to point
# at a results directory that already has Subject_<id>/fold_<n>/ with
# pipeline_params.json, csp_filters.pkl, znorm.pkl for every task's
# subject/fold.
#
# Submit via submit_fair_baseline.sh, which sets RESULTS_DIR,
# MOABB_DATASET, and the array size for you. Direct sbatch also works if
# these are exported first:
#   RESULTS_DIR=Results_verify MOABB_DATASET=BNCI2014_001 \
#     sbatch --array=1-45 --export=ALL,RESULTS_DIR,MOABB_DATASET run_fair_baseline_array.sh
# ============================================================

set -euo pipefail

N_FOLDS="${N_FOLDS:-5}"
TASK_ID=${SLURM_ARRAY_TASK_ID}

RESULTS_DIR="${RESULTS_DIR:?Set RESULTS_DIR to an existing completed results dir}"
MOABB_DATASET="${MOABB_DATASET:?Set MOABB_DATASET (e.g. BNCI2014_001)}"

SUBJECT_ID=$(( (TASK_ID - 1) / N_FOLDS + 1 ))
FOLD_IDX=$(( (TASK_ID - 1) % N_FOLDS ))

PROJECT_DIR=/scratch/project_2003397/praveen/fbcsp-snn-mi-classifier-1

echo "=============================================="
echo "  Fair baseline (full time-series LDA/SVM)"
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

python run_fair_baseline.py \
    --results-dir "${RESULTS_DIR}" \
    --subject-id "${SUBJECT_ID}" \
    --fold "${FOLD_IDX}" \
    --moabb-dataset "${MOABB_DATASET}" \
    --n-folds "${N_FOLDS}"

EXIT_CODE=$?
echo ""
echo "Task ${TASK_ID} finished with exit code ${EXIT_CODE}"
echo "End: $(date)"
exit ${EXIT_CODE}
