#!/bin/bash
#SBATCH --job-name=fbcsp_reliability
#SBATCH --account=project_2003397
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:30:00
# 30 min was measured too short 2026-07-09: a real run only got through
# 4/9 sweeps (csp/snn/beta_noise + 3/5 severities of filter_bank_noise)
# before hitting the wall. joint_noise_all_sources (all 7 sources, incl.
# the filter bank) is likely the most expensive sweep and hadn't even
# started. Bumped generously — pure inference, low GPU-hour cost either way.
#SBATCH --array=1-45                   # 9 subjects x 5 folds = 45 tasks (match run_puhti_array.sh)
#SBATCH --output=logs/fbcsp_rel_S%a_%j.out
#SBATCH --error=logs/fbcsp_rel_S%a_%j.err

# ============================================================
# FBCSP-SNN — Reliability / hardware-noise sweep (B15)
#
# Runs the Monte Carlo noise-robustness sweep on an ALREADY-TRAINED fold's
# saved checkpoint. No retraining -- pure inference, so this only needs to
# run once per fold, after the corresponding training task has completed
# and written pipeline_params.json + best_model.pt.
#
# Array index encodes both subject and fold, same mapping as
# run_puhti_array.sh:
#   task 1-5  -> subject 1, folds 0-4
#   task 6-10 -> subject 2, folds 0-4
#   ...
#   task 41-45 -> subject 9, folds 0-4
#
# Submit (after training + aggregate have finished for these subjects):
#   cd /scratch/project_2003397/praveen/fbcsp-snn-mi-classifier-1
#   mkdir -p logs
#   sbatch run_puhti_reliability.sh
#
#   # Or scoped to specific subjects already trained, matching submit_puhti.sh:
#   RESULTS_DIR=Results_static6 sbatch --array=1-5 run_puhti_reliability.sh   # subject 1 only
#
# Monitor:
#   squeue -u $USER
#   ls Results/Subject_*/fold_*/reliability_results.json
# ============================================================

set -euo pipefail

N_FOLDS=5
TASK_ID=${SLURM_ARRAY_TASK_ID}

RESULTS_DIR="${RESULTS_DIR:-Results}"
RELIABILITY_SEVERITIES="${RELIABILITY_SEVERITIES:-[0.0,0.05,0.1,0.2,0.3]}"
RELIABILITY_N_REPEATS="${RELIABILITY_N_REPEATS:-20}"

SUBJECT_ID=$(( (TASK_ID - 1) / N_FOLDS + 1 ))
FOLD_IDX=$(( (TASK_ID - 1) % N_FOLDS ))

PROJECT_DIR=/scratch/project_2003397/praveen/fbcsp-snn-mi-classifier-1

echo "=============================================="
echo "  FBCSP-SNN Reliability Sweep (B15)"
echo "  Task:    ${TASK_ID}  ->  Subject ${SUBJECT_ID}, Fold ${FOLD_IDX}"
echo "  Results: ${RESULTS_DIR}"
echo "  Severities: ${RELIABILITY_SEVERITIES}   Repeats: ${RELIABILITY_N_REPEATS}"
echo "  Node:    $(hostname)"
echo "  Start:   $(date)"
echo "=============================================="

unset SINGULARITY_BIND
unset APPTAINER_BIND

module purge
source "${PROJECT_DIR}/.venv/bin/activate"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export MNE_DATA=/scratch/project_2003397/praveen/mne_data

cd "${PROJECT_DIR}"

python main.py reliability \
    --source moabb \
    --moabb-dataset "${MOABB_DATASET:-BNCI2014_001}" \
    --subject-id "${SUBJECT_ID}" \
    --fold "${FOLD_IDX}" \
    --reliability-severities "${RELIABILITY_SEVERITIES}" \
    --reliability-n-repeats "${RELIABILITY_N_REPEATS}" \
    --results-dir "${RESULTS_DIR}"

EXIT_CODE=$?

echo ""
echo "Fold ${FOLD_IDX} reliability sweep finished with exit code ${EXIT_CODE}"
echo "End: $(date)"
exit ${EXIT_CODE}
