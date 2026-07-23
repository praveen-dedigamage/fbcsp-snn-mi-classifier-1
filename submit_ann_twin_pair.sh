#!/bin/bash
# ============================================================
# ANN twin -- submit BOTH loss variants for a dataset, chained
#
# Submits van_rossum first, then cross_entropy with
# --dependency=afterany:<van_rossum_jobid> so it only starts once every
# van_rossum task has finished (success or failure) -- guaranteeing
# cross_entropy's tasks find ann_twin_inputs.pt already cached by
# van_rossum for every fold, instead of racing and possibly computing
# the shared preprocessing twice (see run_ann_twin.py's caching).
#
# Usage:
#   bash submit_ann_twin_pair.sh <RESULTS_DIR> <MOABB_DATASET> [N_SUBJECTS]
#
# Examples:
#   bash submit_ann_twin_pair.sh Results_verify BNCI2014_001 9
#   SUBJECTS="2 3 4 5 6 7 8 9" bash submit_ann_twin_pair.sh Results_verify BNCI2014_001
#   bash submit_ann_twin_pair.sh Results_cho2017 Cho2017 52
#   bash submit_ann_twin_pair.sh Results_bnci2015 BNCI2015_001 12
#   SBATCH_TIME=16:00:00 bash submit_ann_twin_pair.sh Results_schirrmeister_verify Schirrmeister2017 14
# ============================================================

set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs

RESULTS_DIR="${1:?Usage: bash submit_ann_twin_pair.sh RESULTS_DIR MOABB_DATASET [N_SUBJECTS]}"
MOABB_DATASET="${2:?Usage: bash submit_ann_twin_pair.sh RESULTS_DIR MOABB_DATASET [N_SUBJECTS]}"
N_SUBJECTS="${3:-9}"
N_FOLDS="${N_FOLDS:-5}"
SBATCH_TIME="${SBATCH_TIME:-}"

if [ ! -d "${RESULTS_DIR}" ]; then
    echo "ERROR: ${RESULTS_DIR} does not exist." >&2
    exit 1
fi

if [ -z "${SUBJECTS:-}" ]; then
    SUBJECTS=$(seq 1 "${N_SUBJECTS}" | tr '\n' ' ' | sed 's/ $//')
fi

N_TASKS=0
ARRAY_TASKS=""
for S in ${SUBJECTS}; do
    START=$(( (S - 1) * N_FOLDS + 1 ))
    END=$(( S * N_FOLDS ))
    ARRAY_TASKS="${ARRAY_TASKS:+${ARRAY_TASKS},}${START}-${END}"
    N_TASKS=$(( N_TASKS + N_FOLDS ))
done

TIME_FLAG=""
[ -n "${SBATCH_TIME}" ] && TIME_FLAG="--time=${SBATCH_TIME}"

echo "=============================================="
echo "  ANN twin -- chained pair (van_rossum -> cross_entropy)"
echo "  Results dir: ${RESULTS_DIR}"
echo "  Dataset:     ${MOABB_DATASET}"
echo "  Subjects:    ${SUBJECTS}"
echo "  Array tasks: ${ARRAY_TASKS}  (${N_TASKS} tasks per loss type)"
[ -n "${SBATCH_TIME}" ] && echo "  Wall time:   ${SBATCH_TIME}"
echo "=============================================="

VR_OUTPUT=$(sbatch \
    --array="${ARRAY_TASKS}" \
    ${TIME_FLAG} \
    --export="ALL,RESULTS_DIR=${RESULTS_DIR},MOABB_DATASET=${MOABB_DATASET},LOSS_TYPE=van_rossum,N_FOLDS=${N_FOLDS}" \
    run_ann_twin_array.sh)
VR_JOBID=$(echo "${VR_OUTPUT}" | awk '{print $4}')
echo "van_rossum:    ${VR_OUTPUT}"

CE_OUTPUT=$(sbatch \
    --array="${ARRAY_TASKS}" \
    --dependency=afterany:${VR_JOBID} \
    ${TIME_FLAG} \
    --export="ALL,RESULTS_DIR=${RESULTS_DIR},MOABB_DATASET=${MOABB_DATASET},LOSS_TYPE=cross_entropy,N_FOLDS=${N_FOLDS}" \
    run_ann_twin_array.sh)
CE_JOBID=$(echo "${CE_OUTPUT}" | awk '{print $4}')
echo "cross_entropy: ${CE_OUTPUT}  (starts only after ${VR_JOBID} fully finishes)"

echo ""
echo "Monitor: squeue -u \$USER"
echo "Once BOTH are done, aggregate:"
echo "  python aggregate_ann_twin.py --results-dir ${RESULTS_DIR} --loss-type van_rossum --subjects ${SUBJECTS}"
echo "  python aggregate_ann_twin.py --results-dir ${RESULTS_DIR} --loss-type cross_entropy --subjects ${SUBJECTS}"
