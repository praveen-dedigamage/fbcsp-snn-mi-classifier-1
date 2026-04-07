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
