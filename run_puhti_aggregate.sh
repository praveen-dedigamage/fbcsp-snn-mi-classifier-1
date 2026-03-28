#!/bin/bash
#SBATCH --job-name=fbcsp_snn_aggregate
#SBATCH --account=project_2003397
#SBATCH --partition=small
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/fbcsp_snn_aggregate_%j.out
#SBATCH --error=logs/fbcsp_snn_aggregate_%j.err

# ============================================================
# FBCSP-SNN — Aggregation + cross-subject analysis
# Run after all array job tasks have completed.
#
# Submit as a dependent job:
#   sbatch --dependency=afterok:<ARRAY_JOBID> run_puhti_aggregate.sh
#
# Or run interactively after the array finishes:
#   srun --account=project_2003397 --partition=small --mem=16G \
#        --time=00:30:00 bash run_puhti_aggregate.sh
# ============================================================

set -euo pipefail

PROJECT_DIR=/scratch/project_2003397/praveen/fbcsp-snn-mi-classifier-1

echo "=============================================="
echo "  FBCSP-SNN Aggregation"
echo "  Node:  $(hostname)"
echo "  Start: $(date)"
echo "=============================================="

source "${PROJECT_DIR}/.venv/bin/activate"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export MNE_DATA=/scratch/project_2003397/praveen/mne_data

cd "${PROJECT_DIR}"

# Per-subject aggregation (collect fold JSONs → summary.csv + confusion plots)
for S in 1 2 3 4 5 6 7 8 9; do
    echo ""
    echo "--- Aggregating Subject ${S} ---"
    python main.py aggregate \
        --source moabb \
        --moabb-dataset BNCI2014_001 \
        --subject-id "${S}" \
        --n-folds 10
done

# Cross-subject summary table + bar chart
echo ""
echo "--- Cross-subject analysis ---"
python analyze_results.py \
    --results-dir Results \
    --subjects 1 2 3 4 5 6 7 8 9

echo ""
echo "Aggregation complete. Results in ${PROJECT_DIR}/Results/"
echo "End: $(date)"
