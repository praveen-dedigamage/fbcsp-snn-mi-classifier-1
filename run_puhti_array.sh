#!/bin/bash
#SBATCH --job-name=fbcsp_snn
#SBATCH --account=project_2003397
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4              # 4 CPU threads for data loading / scipy
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --array=1-9                    # one task per BCI-IV-2a subject
#SBATCH --output=logs/fbcsp_snn_S%a_%j.out
#SBATCH --error=logs/fbcsp_snn_S%a_%j.err

# ============================================================
# FBCSP-SNN — CSC Puhti SLURM array job
# Each task trains one subject (SLURM_ARRAY_TASK_ID = subject ID)
#
# Submit:
#   cd /scratch/project_2003397/praveen/fbcsp-snn-mi-classifier-1
#   mkdir -p logs
#   sbatch run_puhti_array.sh
#
# Monitor:
#   squeue -u $USER
#   sacct -j <JOBID> --format=JobID,State,Elapsed,MaxRSS
# ============================================================

set -euo pipefail

SUBJECT_ID=${SLURM_ARRAY_TASK_ID}
PROJECT_DIR=/scratch/project_2003397/praveen/fbcsp-snn-mi-classifier-1

echo "=============================================="
echo "  FBCSP-SNN  Subject ${SUBJECT_ID}"
echo "  Node:   $(hostname)"
echo "  GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "  Start:  $(date)"
echo "  Dir:    ${PROJECT_DIR}"
echo "=============================================="

# ---- Environment ----------------------------------------------------------
module purge
source "${PROJECT_DIR}/.venv/bin/activate"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

# MOABB/MNE data cache — keep on scratch, not home (home quota is small)
export MNE_DATA=/scratch/project_2003397/praveen/mne_data

cd "${PROJECT_DIR}"

# ---- Training -------------------------------------------------------------
python main.py train \
    --source moabb \
    --moabb-dataset BNCI2014_001 \
    --subject-id "${SUBJECT_ID}" \
    --adaptive-bands \
    --n-adaptive-bands 6 \
    --csp-components-per-band 4 \
    --hidden-neurons 64 \
    --population-per-class 20 \
    --beta 0.95 \
    --dropout-prob 0.5 \
    --epochs 1000 \
    --early-stopping-patience 100 \
    --early-stopping-warmup 100 \
    --lr 1e-3 \
    --weight-decay 0.1 \
    --spiking-prob 0.7 \
    --feature-selection-method mibif \
    --feature-percentile 50.0 \
    --n-folds 10 \
    --results-dir Results

EXIT_CODE=$?

echo ""
echo "Training finished with exit code ${EXIT_CODE}"
echo "End: $(date)"
exit ${EXIT_CODE}

# ============================================================
# After all 9 array tasks complete, run aggregation + analysis:
#
#   cd /scratch/project_2003397/praveen/fbcsp-snn-mi-classifier-1
#   source .venv/bin/activate
#   for S in 1 2 3 4 5 6 7 8 9; do
#       python main.py aggregate --subject-id $S --n-folds 10
#   done
#   python analyze_results.py --results-dir Results --subjects 1 2 3 4 5 6 7 8 9
#
# Or submit as a dependent job (runs automatically after all subjects finish):
#
#   sbatch --dependency=afterok:<ARRAY_JOBID> run_puhti_aggregate.sh
# ============================================================
