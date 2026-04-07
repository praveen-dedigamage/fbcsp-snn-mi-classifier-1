#!/bin/bash
# ============================================================
# FBCSP-SNN — One-shot submit: array training + auto-aggregate
#
# Usage:
#   cd /scratch/project_2003397/praveen/fbcsp-snn-mi-classifier-1
#   bash submit_puhti.sh
#
# This script:
#   1. Submits the 45-task array job (9 subjects × 5 folds)
#   2. Submits the aggregate job with afterok dependency —
#      it runs automatically once ALL 45 tasks succeed
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

# Submit array job and capture job ID
ARRAY_OUTPUT=$(sbatch run_puhti_array.sh)
ARRAY_JOBID=$(echo "${ARRAY_OUTPUT}" | awk '{print $4}')

echo "${ARRAY_OUTPUT}"
echo "Array job ID: ${ARRAY_JOBID}"

# Submit aggregate job as dependent
AGG_OUTPUT=$(sbatch --dependency=afterok:${ARRAY_JOBID} run_puhti_aggregate.sh)
AGG_JOBID=$(echo "${AGG_OUTPUT}" | awk '{print $4}')

echo "${AGG_OUTPUT}"
echo "Aggregate job ID: ${AGG_JOBID}"

echo ""
echo "All jobs submitted. Monitor with:"
echo "  squeue -u \$USER"
echo "  sacct -j ${ARRAY_JOBID} --format=JobID,State,Elapsed,MaxRSS"
echo ""
echo "Aggregate (job ${AGG_JOBID}) will run automatically when all 45 tasks succeed."
echo "Results will be in: Results/"
