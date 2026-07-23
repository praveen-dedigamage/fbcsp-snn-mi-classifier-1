#!/bin/bash
# ============================================================
# ANN twin (non-spiking) -- submit across subjects
#
# Reuses an EXISTING completed SNN training run's saved CSP/z-norm/MIBIF
# artifacts (same encoded input the SNN received) and trains
# ANNClassifier -- identical architecture/optimiser/protocol, LIF
# replaced by a continuous ReLU -- via the SAME train_fold() the SNN
# uses. Needs a real GPU job (backprop through 1000 epochs), unlike the
# CPU-only fair baseline.
#
# Usage:
#   bash submit_ann_twin.sh <EXISTING_RESULTS_DIR> <MOABB_DATASET> <LOSS_TYPE> [N_SUBJECTS]
#
# LOSS_TYPE: van_rossum (max parity with the SNN's own loss) or
#            cross_entropy (the "conventional" ANN loss)
#
# Examples:
#   bash submit_ann_twin.sh Results_verify BNCI2014_001 van_rossum 9
#   bash submit_ann_twin.sh Results_verify BNCI2014_001 cross_entropy 9
#   SUBJECTS="1 2 3" bash submit_ann_twin.sh Results_verify BNCI2014_001 van_rossum
#
# Aggregate once the array finishes:
#   python aggregate_ann_twin.py --results-dir <EXISTING_RESULTS_DIR> \
#       --loss-type van_rossum --subjects 1 2 3 4 5 6 7 8 9
# ============================================================

set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs

RESULTS_DIR="${1:?Usage: bash submit_ann_twin.sh RESULTS_DIR MOABB_DATASET LOSS_TYPE [N_SUBJECTS]}"
MOABB_DATASET="${2:?Usage: bash submit_ann_twin.sh RESULTS_DIR MOABB_DATASET LOSS_TYPE [N_SUBJECTS]}"
LOSS_TYPE="${3:?Usage: bash submit_ann_twin.sh RESULTS_DIR MOABB_DATASET LOSS_TYPE [N_SUBJECTS]}"
N_SUBJECTS="${4:-9}"
N_FOLDS="${N_FOLDS:-5}"
SBATCH_TIME="${SBATCH_TIME:-}"

if [ ! -d "${RESULTS_DIR}" ]; then
    echo "ERROR: ${RESULTS_DIR} does not exist -- this script reuses an" >&2
    echo "       already-completed SNN training run's saved artifacts." >&2
    exit 1
fi

if [ "${LOSS_TYPE}" != "van_rossum" ] && [ "${LOSS_TYPE}" != "cross_entropy" ]; then
    echo "ERROR: LOSS_TYPE must be van_rossum or cross_entropy, got '${LOSS_TYPE}'" >&2
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

echo "=============================================="
echo "  ANN twin (non-spiking, loss=${LOSS_TYPE})"
echo "  Results dir reused: ${RESULTS_DIR}"
echo "  Dataset:             ${MOABB_DATASET}"
echo "  Subjects:            ${SUBJECTS}"
echo "  Array tasks:         ${ARRAY_TASKS}  (${N_TASKS} total)"
[ -n "${SBATCH_TIME}" ] && echo "  Wall time override:  ${SBATCH_TIME}"
echo "=============================================="

TIME_FLAG=""
[ -n "${SBATCH_TIME}" ] && TIME_FLAG="--time=${SBATCH_TIME}"

sbatch \
    --array="${ARRAY_TASKS}" \
    ${TIME_FLAG} \
    --export="ALL,RESULTS_DIR=${RESULTS_DIR},MOABB_DATASET=${MOABB_DATASET},LOSS_TYPE=${LOSS_TYPE},N_FOLDS=${N_FOLDS}" \
    run_ann_twin_array.sh

echo ""
echo "Monitor: squeue -u \$USER"
echo "Once all tasks finish, aggregate with:"
echo "  python aggregate_ann_twin.py --results-dir ${RESULTS_DIR} --loss-type ${LOSS_TYPE} --subjects ${SUBJECTS}"
