#!/bin/bash
#SBATCH --job-name=fbcsp_snn
#SBATCH --account=project_2003397
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4              # 4 CPU threads for data loading / scipy
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --array=1-45                   # 9 subjects × 5 folds = 45 tasks
#SBATCH --output=logs/fbcsp_snn_S%a_%j.out
#SBATCH --error=logs/fbcsp_snn_S%a_%j.err

# ============================================================
# FBCSP-SNN — CSC Puhti SLURM array job  (V6.4)
# Array index encodes both subject and fold:
#   task 1-5  → subject 1, folds 0-4
#   task 6-10 → subject 2, folds 0-4
#   ...
#   task 41-45 → subject 9, folds 0-4
#
# Submit:
#   cd /scratch/project_2003397/praveen/fbcsp-snn-mi-classifier-1
#   mkdir -p logs
#   sbatch run_puhti_array.sh
#
# Monitor:
#   squeue -u $USER
#   sacct -j <JOBID> --format=JobID,State,Elapsed,MaxRSS
#
# After all 45 tasks complete, aggregate:
#   sbatch --dependency=afterok:<ARRAY_JOBID> run_puhti_aggregate.sh
# ============================================================

set -euo pipefail

N_FOLDS=5
TASK_ID=${SLURM_ARRAY_TASK_ID}

# Allow submit_puhti.sh to inject a custom results directory and extra flags.
# Defaults keep the script independently runnable with `sbatch run_puhti_array.sh`.
RESULTS_DIR="${RESULTS_DIR:-Results}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# Derive 1-indexed subject and 0-indexed fold from task ID
SUBJECT_ID=$(( (TASK_ID - 1) / N_FOLDS + 1 ))
FOLD_IDX=$(( (TASK_ID - 1) % N_FOLDS ))

PROJECT_DIR=/scratch/project_2003397/praveen/fbcsp-snn-mi-classifier-1

echo "=============================================="
echo "  FBCSP-SNN  V6.4"
echo "  Task:    ${TASK_ID}  →  Subject ${SUBJECT_ID}, Fold ${FOLD_IDX}"
echo "  Results: ${RESULTS_DIR}"
echo "  Extra:   ${EXTRA_ARGS:-<none>}"
echo "  Node:    $(hostname)"
echo "  GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "  Start:   $(date)"
echo "  Dir:     ${PROJECT_DIR}"
echo "=============================================="

# ---- Singularity bind fix -------------------------------------------------
# Puhti sets SINGULARITY_BIND/APPTAINER_BIND system-wide to include
# /local_scratch/<user>. Without --gres=nvme that directory doesn't exist
# and Singularity fatally fails. We use a plain .venv — clear the bind vars.
unset SINGULARITY_BIND
unset APPTAINER_BIND

# ---- Environment ----------------------------------------------------------
module purge
source "${PROJECT_DIR}/.venv/bin/activate"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

# MOABB/MNE data cache — keep on scratch, not home (home quota is small)
export MNE_DATA=/scratch/project_2003397/praveen/mne_data

cd "${PROJECT_DIR}"

# ---- Training (single fold) — V4.1 proven configuration ------------------
# V6 features (activity_reg, 3-layer SNN, LR scheduler) are disabled here.
# They collapsed accuracy to ~25% (chance) when combined. Add back one at a
# time only after confirming V4.1 baseline is reproduced.
python main.py train \
    --source moabb \
    --moabb-dataset BNCI2014_001 \
    --subject-id "${SUBJECT_ID}" \
    --fold "${FOLD_IDX}" \
    --n-folds "${N_FOLDS}" \
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
    --epochs 1000 \
    --early-stopping-patience 100 \
    --early-stopping-warmup 100 \
    --spiking-prob 0.7 \
    --feature-selection-method mibif \
    --mi-fraction 0.1 \
    --results-dir "${RESULTS_DIR}" \
    ${EXTRA_ARGS}

EXIT_CODE=$?

echo ""
echo "Fold ${FOLD_IDX} finished with exit code ${EXIT_CODE}"
echo "End: $(date)"
exit ${EXIT_CODE}
