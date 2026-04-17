#!/bin/bash
# ============================================================
# FBCSP-SNN — Submit Butterworth Monte Carlo jobs
#
# Usage:
#   bash submit_mc.sh [RESULTS_DIR [OUTPUT_DIR]]
#
# Examples:
#   bash submit_mc.sh
#   bash submit_mc.sh Results_adm_static6_ptq Results_butterworth_mc
#   SUBJECTS="1 2 3" bash submit_mc.sh Results_adm_static6_ptq
#
# Stage 1: MC array   — 9 tasks (one per subject, small partition, no GPU)
# Stage 2: MC analyze — 1 job   (afterok all 9 tasks, merges CSVs + plots)
# ============================================================

set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs

RESULTS_DIR="${1:-Results_adm_static6_ptq}"
OUTPUT_DIR="${2:-Results_butterworth_mc}"
SUBJECTS="${SUBJECTS:-1 2 3 4 5 6 7 8 9}"
N_DRAWS="${N_DRAWS:-50}"
SIGMAS="${SIGMAS:-0.01 0.02 0.05}"

echo "RESULTS_DIR: ${RESULTS_DIR}"
echo "OUTPUT_DIR:  ${OUTPUT_DIR}"
echo "SUBJECTS:    ${SUBJECTS}"
echo "N_DRAWS:     ${N_DRAWS}"
echo "SIGMAS:      ${SIGMAS}"
echo ""

# Build array task IDs from subject list (task ID = subject ID, 1-9)
ARRAY_TASKS=""
for S in ${SUBJECTS}; do
    ARRAY_TASKS="${ARRAY_TASKS:+${ARRAY_TASKS},}${S}"
done
echo "Array tasks: ${ARRAY_TASKS}"

# Stage 1: MC array
MC_OUTPUT=$(sbatch \
    --array="${ARRAY_TASKS}" \
    --export="ALL,RESULTS_DIR=${RESULTS_DIR},OUTPUT_DIR=${OUTPUT_DIR},N_DRAWS=${N_DRAWS},SIGMAS=${SIGMAS}" \
    run_puhti_mc.sh)
MC_JOBID=$(echo "${MC_OUTPUT}" | awk '{print $4}')
echo "${MC_OUTPUT}"
echo ""

# Stage 2: Analyze — afterok all array tasks
DEP="afterok"
for S in ${SUBJECTS}; do
    DEP="${DEP}:${MC_JOBID}_${S}"
done

ANALYZE_OUTPUT=$(sbatch \
    --dependency=${DEP} \
    --export="ALL,OUTPUT_DIR=${OUTPUT_DIR},SUBJECTS=${SUBJECTS}" \
    run_puhti_mc_analyze.sh)
ANALYZE_JOBID=$(echo "${ANALYZE_OUTPUT}" | awk '{print $4}')
echo "${ANALYZE_OUTPUT}"
echo ""

N_SUBJECTS=$(echo ${SUBJECTS} | wc -w)
echo "=============================================="
echo "  Monte Carlo jobs submitted"
echo "  Results dir:  ${RESULTS_DIR}"
echo "  Output dir:   ${OUTPUT_DIR}"
echo "  MC array:     ${MC_JOBID} (${N_SUBJECTS} subjects × 60 min each)"
echo "  MC analyze:   ${ANALYZE_JOBID} (after all tasks)"
echo ""
echo "  Monitor: squeue -u \$USER"
echo "  Results: ${OUTPUT_DIR}/mc_raw.csv"
echo "  Plot:    ${OUTPUT_DIR}/butterworth_mc.png"
echo "  Log:     logs/fbcsp_mc_analyze_${ANALYZE_JOBID}.out"
echo "=============================================="
