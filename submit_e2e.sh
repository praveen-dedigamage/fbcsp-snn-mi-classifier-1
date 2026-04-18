#!/bin/bash
# ============================================================
# FBCSP-SNN — Submit End-to-End Hardware Stress Test (item 7)
#
# Usage:
#   bash submit_e2e.sh [RESULTS_DIR [OUTPUT_DIR]]
#
# Examples:
#   bash submit_e2e.sh
#   bash submit_e2e.sh Results_adm_static6_ptq Results_e2e_stress
#   SUBJECTS="1 2 3" bash submit_e2e.sh
#
# Stage 1: E2E array   — 9 tasks (one per subject, ~2 h each)
# Stage 2: E2E analyze — 1 job   (afterok all 9 tasks, merges CSVs + plots)
# ============================================================

set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs

RESULTS_DIR="${1:-Results_adm_static6_ptq}"
OUTPUT_DIR="${2:-Results_e2e_stress}"
SUBJECTS="${SUBJECTS:-1 2 3 4 5 6 7 8 9}"
SIGMA="${SIGMA:-0.02}"
CSP_BITS="${CSP_BITS:-4}"
SNN_BITS="${SNN_BITS:-8}"
N_DRAWS="${N_DRAWS:-100}"

echo "RESULTS_DIR: ${RESULTS_DIR}"
echo "OUTPUT_DIR:  ${OUTPUT_DIR}"
echo "SUBJECTS:    ${SUBJECTS}"
echo "SIGMA:       ${SIGMA}"
echo "CSP_BITS:    ${CSP_BITS}"
echo "SNN_BITS:    ${SNN_BITS}"
echo "N_DRAWS:     ${N_DRAWS}"
echo ""

# Build array task IDs from subject list
ARRAY_TASKS=""
for S in ${SUBJECTS}; do
    ARRAY_TASKS="${ARRAY_TASKS:+${ARRAY_TASKS},}${S}"
done
echo "Array tasks: ${ARRAY_TASKS}"

# Stage 1: E2E array
E2E_OUTPUT=$(sbatch \
    --array="${ARRAY_TASKS}" \
    --export="ALL,RESULTS_DIR=${RESULTS_DIR},OUTPUT_DIR=${OUTPUT_DIR},SIGMA=${SIGMA},CSP_BITS=${CSP_BITS},SNN_BITS=${SNN_BITS},N_DRAWS=${N_DRAWS}" \
    run_puhti_e2e.sh)
E2E_JOBID=$(echo "${E2E_OUTPUT}" | awk '{print $4}')
echo "${E2E_OUTPUT}"
echo ""

# Stage 2: Analyze — afterok all array tasks
DEP="afterok"
for S in ${SUBJECTS}; do
    DEP="${DEP}:${E2E_JOBID}_${S}"
done

ANALYZE_OUTPUT=$(sbatch \
    --dependency=${DEP} \
    --export="ALL,OUTPUT_DIR=${OUTPUT_DIR},SUBJECTS=${SUBJECTS},SIGMA=${SIGMA},CSP_BITS=${CSP_BITS},SNN_BITS=${SNN_BITS}" \
    run_puhti_e2e_analyze.sh)
ANALYZE_JOBID=$(echo "${ANALYZE_OUTPUT}" | awk '{print $4}')
echo "${ANALYZE_OUTPUT}"
echo ""

N_SUBJECTS=$(echo ${SUBJECTS} | wc -w)
echo "=============================================="
echo "  E2E stress test jobs submitted"
echo "  Results dir:  ${RESULTS_DIR}"
echo "  Output dir:   ${OUTPUT_DIR}"
echo "  E2E array:    ${E2E_JOBID} (${N_SUBJECTS} subjects × 2 h each)"
echo "  E2E analyze:  ${ANALYZE_JOBID} (after all tasks)"
echo ""
echo "  Monitor: squeue -u \$USER"
echo "  Results: ${OUTPUT_DIR}/e2e_summary.csv"
echo "  Plot:    ${OUTPUT_DIR}/e2e_stress.png"
echo "  Log:     logs/fbcsp_e2e_analyze_${ANALYZE_JOBID}.out"
echo "=============================================="
