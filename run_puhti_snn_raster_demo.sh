#!/bin/bash
#SBATCH --job-name=fbcsp_snn_raster_demo
#SBATCH --account=project_2003397
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --output=logs/fbcsp_snn_raster_demo_%j.out
#SBATCH --error=logs/fbcsp_snn_raster_demo_%j.err

# ============================================================
# Single-fold training run (Subject 1, Fold 0) using the CURRENT
# documented pipeline config (fixed six-band, csp_components_per_band=8,
# mi_fraction=0.1 -- matches run_puhti_array.sh's proven configuration
# exactly, just a single task instead of the full 45-task array).
#
# Purpose: produce ONE real, representative trained fold whose
# best_model.pt / csp_filters.pkl / znorm.pkl / mibif.pkl /
# pipeline_params.json can be pulled back and used locally to plot a
# real SNN spike raster (input -> hidden -> output layers) for one
# genuine BNCI2014-001 trial -- unlike the stale Results_v3/Subject_2
# checkpoint (old adaptive-band config, csp_m=3, 51% acc), this uses
# the SAME pipeline the paper actually reports 66% mean accuracy for.
#
# Submit:
#   cd /scratch/project_2003397/praveen/fbcsp-snn-mi-classifier-1
#   mkdir -p logs
#   sbatch run_puhti_snn_raster_demo.sh
#
# Monitor:
#   squeue -u $USER
#   sacct -j <JOBID> --format=JobID,State,Elapsed,MaxRSS
#
# After it completes, pull back (from your local machine):
#   scp -r puhti:/scratch/project_2003397/praveen/fbcsp-snn-mi-classifier-1/Results_snn_raster_demo \
#       D:/snn_pipeline_for_mi_eeg_classification/.claude/worktrees/hungry-neumann/
# ============================================================

set -euo pipefail

PROJECT_DIR=/scratch/project_2003397/praveen/fbcsp-snn-mi-classifier-1
RESULTS_DIR=Results_snn_raster_demo

echo "=============================================="
echo "  FBCSP-SNN — SNN raster demo training run"
echo "  Subject: 1   Fold: 0"
echo "  Results: ${RESULTS_DIR}"
echo "  Node:    $(hostname)"
echo "  GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader)"
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

# ---- Training (single fold, Subject 1, Fold 0) — matches
# run_puhti_array.sh's proven V4.1 configuration exactly ----
python main.py train \
    --source moabb \
    --moabb-dataset BNCI2014_001 \
    --subject-id 1 \
    --fold 0 \
    --n-folds 5 \
    --freq-bands "[(4,8),(8,14),(12,18),(16,24),(20,30),(26,40)]" \
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
    --results-dir "${RESULTS_DIR}"

EXIT_CODE=$?

echo ""
echo "Finished with exit code ${EXIT_CODE}"
echo "End: $(date)"
echo ""
echo "Pull back with (run from your local machine):"
echo "  scp -r <puhti-host>:${PROJECT_DIR}/${RESULTS_DIR} <local-repo-path>/"
exit ${EXIT_CODE}
