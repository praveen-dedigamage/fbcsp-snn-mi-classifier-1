#!/bin/bash
#SBATCH --job-name=fbcsp_snn
#SBATCH --account=project_2003397
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --array=1-45                   # 9 subjects × 5 folds = 45 tasks
#SBATCH --output=logs/fbcsp_snn_S%a_%j.out
#SBATCH --error=logs/fbcsp_snn_S%a_%j.err

# ============================================================
# FBCSP-SNN — Static 6 Hz filter-bank experiment
#
# Bands: 6 Hz wide, 4 Hz step, 2 Hz overlap, covering 4–30 Hz
#   (4,10)  (8,14)  (12,18)  (16,22)  (20,26)  (24,30)
#
# Band list is hardcoded here (not via SLURM --export) so that
# commas and parentheses are never passed through --export, which
# would silently corrupt the env-var string.
#
# EXTRA_ARGS (no-comma flags only):
#   --augment-windows           sliding-window CSP augmentation
#   --freq-shift-augment        ±2 Hz spectral-drift augmentation
#   --recurrent-hidden          replace LIF hidden layer with RLeaky
#
# Submit via submit_puhti.sh:
#   ARRAY_SCRIPT=run_puhti_static6.sh bash submit_puhti.sh \
#       Results_static6_freqshift --freq-shift-augment --augment-windows
#
#   ARRAY_SCRIPT=run_puhti_static6.sh bash submit_puhti.sh \
#       Results_static6_rlif --recurrent-hidden --augment-windows
# ============================================================

set -euo pipefail

N_FOLDS=5
TASK_ID=${SLURM_ARRAY_TASK_ID}

RESULTS_DIR="${RESULTS_DIR:-Results}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

SUBJECT_ID=$(( (TASK_ID - 1) / N_FOLDS + 1 ))
FOLD_IDX=$(( (TASK_ID - 1) % N_FOLDS ))

PROJECT_DIR=/scratch/project_2003397/praveen/fbcsp-snn-mi-classifier-1

echo "=============================================="
echo "  FBCSP-SNN  static-6Hz filter bank"
echo "  Task:    ${TASK_ID}  →  Subject ${SUBJECT_ID}, Fold ${FOLD_IDX}"
echo "  Bands:   (4,10) (8,14) (12,18) (16,22) (20,26) (24,30)"
echo "  Results: ${RESULTS_DIR}"
echo "  Extra:   ${EXTRA_ARGS:-<none>}"
echo "  Node:    $(hostname)"
echo "  GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "  Start:   $(date)"
echo "  Dir:     ${PROJECT_DIR}"
echo "=============================================="

unset SINGULARITY_BIND
unset APPTAINER_BIND

module purge
source "${PROJECT_DIR}/.venv/bin/activate"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export MNE_DATA=/scratch/project_2003397/praveen/mne_data

cd "${PROJECT_DIR}"

# Static 6 Hz bands — hardcoded to avoid SLURM --export comma issue.
# --band-range is included so freq-shift augmentation clips shifted
# bands correctly when --freq-shift-augment is in EXTRA_ARGS.
python main.py train \
    --source moabb \
    --moabb-dataset BNCI2014_001 \
    --subject-id "${SUBJECT_ID}" \
    --fold "${FOLD_IDX}" \
    --n-folds "${N_FOLDS}" \
    --no-adaptive-bands \
    --freq-bands "[(4,10),(8,14),(12,18),(16,22),(20,26),(24,30)]" \
    --band-range 4.0:30.0 \
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
