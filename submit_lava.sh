#!/bin/bash
# ============================================================
# FBCSP-SNN — Submit Lava simulation jobs (item 8)
#
# Usage:
#   bash submit_lava.sh [RESULTS_DIR [OUTPUT_DIR]]
#
# Stage 1: Lava array   — 9 tasks (one per subject, 30 min each)
# Stage 2: Lava analyze — 1 job   (afterok all 9 tasks)
# ============================================================

set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs

RESULTS_DIR="${1:-Results_adm_static6_ptq}"
OUTPUT_DIR="${2:-Results_lava}"
SUBJECTS="${SUBJECTS:-1 2 3 4 5 6 7 8 9}"

echo "RESULTS_DIR: ${RESULTS_DIR}"
echo "OUTPUT_DIR:  ${OUTPUT_DIR}"
echo "SUBJECTS:    ${SUBJECTS}"
echo ""

ARRAY_TASKS=""
for S in ${SUBJECTS}; do
    ARRAY_TASKS="${ARRAY_TASKS:+${ARRAY_TASKS},}${S}"
done
echo "Array tasks: ${ARRAY_TASKS}"

# Stage 1: Lava array
LAVA_OUTPUT=$(sbatch \
    --array="${ARRAY_TASKS}" \
    --export="ALL,RESULTS_DIR=${RESULTS_DIR},OUTPUT_DIR=${OUTPUT_DIR}" \
    run_puhti_lava.sh)
LAVA_JOBID=$(echo "${LAVA_OUTPUT}" | awk '{print $4}')
echo "${LAVA_OUTPUT}"
echo ""

# Stage 2: Analyze — afterok all array tasks
DEP="afterok"
for S in ${SUBJECTS}; do
    DEP="${DEP}:${LAVA_JOBID}_${S}"
done

ANALYZE_OUTPUT=$(sbatch \
    --dependency=${DEP} \
    --export="ALL,OUTPUT_DIR=${OUTPUT_DIR},SUBJECTS=${SUBJECTS}" \
    run_puhti_lava_analyze.sh)
ANALYZE_JOBID=$(echo "${ANALYZE_OUTPUT}" | awk '{print $4}')
echo "${ANALYZE_OUTPUT}"
echo ""

N_SUBJECTS=$(echo ${SUBJECTS} | wc -w)
echo "=============================================="
echo "  Lava simulation jobs submitted"
echo "  Results dir:  ${RESULTS_DIR}"
echo "  Output dir:   ${OUTPUT_DIR}"
echo "  Lava array:   ${LAVA_JOBID} (${N_SUBJECTS} subjects × 30 min)"
echo "  Lava analyze: ${ANALYZE_JOBID} (after all tasks)"
echo ""
echo "  Monitor: squeue -u \$USER"
echo "  Results: ${OUTPUT_DIR}/lava_summary.csv"
echo "  Log:     logs/fbcsp_lava_analyze_${ANALYZE_JOBID}.out"
echo "=============================================="
