#!/bin/bash
# ============================================================
# FBCSP-SNN — One-shot submit: train → aggregate → analyze
#
# Usage:
#   cd /scratch/project_2003397/praveen/fbcsp-snn-mi-classifier-1
#   bash submit_puhti.sh
#
# Pipeline:
#   1. Training array  (45 tasks: 9 subjects × 5 folds, parallel)
#   2. Aggregate array (9 tasks:  one per subject, parallel, afterok train)
#   3. Analyze job     (1 task:   cross-subject summary, afterok all agg)
# ============================================================

set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs

# Read N_FOLDS from the array script — single source of truth
N_FOLDS=$(grep '^N_FOLDS=' run_puhti_array.sh | cut -d= -f2)
if [ -z "${N_FOLDS}" ]; then
    echo "ERROR: could not read N_FOLDS from run_puhti_array.sh" >&2
    exit 1
fi
echo "N_FOLDS=${N_FOLDS} (read from run_puhti_array.sh)"

# Remove fold directories with index >= N_FOLDS (stale from a previous higher-fold run)
echo "Cleaning up stale fold directories (fold_${N_FOLDS} and above)..."
REMOVED=0
for S in $(seq 1 9); do
    for F in Results/Subject_${S}/fold_*/; do
        [ -d "${F}" ] || continue
        FOLD_NUM=$(basename "${F}" | sed 's/fold_//')
        if [ "${FOLD_NUM}" -ge "${N_FOLDS}" ]; then
            echo "  Removing ${F}"
            rm -rf "${F}"
            REMOVED=$((REMOVED + 1))
        fi
    done
done
[ "${REMOVED}" -eq 0 ] && echo "  Nothing to remove." || echo "  Removed ${REMOVED} director(ies)."
echo ""

# --- Stage 1: Training array (45 tasks) ------------------------------------
TRAIN_OUTPUT=$(sbatch run_puhti_array.sh)
TRAIN_JOBID=$(echo "${TRAIN_OUTPUT}" | awk '{print $4}')
echo "${TRAIN_OUTPUT}"
echo ""

# --- Stage 2: Aggregate array (9 tasks, one per subject) -------------------
AGG_OUTPUT=$(sbatch --dependency=afterok:${TRAIN_JOBID} run_puhti_aggregate.sh)
AGG_JOBID=$(echo "${AGG_OUTPUT}" | awk '{print $4}')
echo "${AGG_OUTPUT}"
echo ""

# --- Stage 3: Cross-subject analysis (1 task) ------------------------------
ANALYZE_OUTPUT=$(sbatch --dependency=afterok:${AGG_JOBID} run_puhti_analyze.sh)
ANALYZE_JOBID=$(echo "${ANALYZE_OUTPUT}" | awk '{print $4}')
echo "${ANALYZE_OUTPUT}"
echo ""

echo "=============================================="
echo "  All jobs submitted"
echo "  Train:    ${TRAIN_JOBID}   (45 tasks)"
echo "  Aggregate:${AGG_JOBID}    (9 tasks, after train)"
echo "  Analyze:  ${ANALYZE_JOBID}    (1 task,  after aggregate)"
echo ""
echo "  Monitor:  squeue -u \$USER"
echo "  Results will appear in: Results/"
echo "  Final summary in: logs/fbcsp_analyze_${ANALYZE_JOBID}.out"
echo "=============================================="
