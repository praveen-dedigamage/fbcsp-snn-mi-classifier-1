#!/bin/bash
# ============================================================
# Fair baseline (full time-series LDA/SVM) -- submit across subjects
#
# Reuses an EXISTING completed training run's saved CSP/z-norm artifacts
# (no SNN retraining, no GPU, ~2 min/fold on the BNCI2014-001 pilot).
#
# Usage:
#   bash submit_fair_baseline.sh <EXISTING_RESULTS_DIR> <MOABB_DATASET> [N_SUBJECTS]
#
# Examples:
#   bash submit_fair_baseline.sh Results_verify BNCI2014_001 9
#   bash submit_fair_baseline.sh Results_schirrmeister_verify Schirrmeister2017 14
#   bash submit_fair_baseline.sh Results_cho2017 Cho2017 52
#   bash submit_fair_baseline.sh Results_bnci2015 BNCI2015_001 12
#   SUBJECTS="1 2 3" bash submit_fair_baseline.sh Results_verify BNCI2014_001
#
# Override wall time (default 30 min; bump for high-dimensional datasets --
# Schirrmeister2017's ~576K-dim flattened vectors are much slower to PCA/fit
# than BNCI2014-001's ~288K or Cho2017's ~74K):
#   SBATCH_TIME=1:30:00 bash submit_fair_baseline.sh Results_schirrmeister_verify Schirrmeister2017 14
#
# Aggregate once the array finishes:
#   python aggregate_fair_baseline.py --results-dir <EXISTING_RESULTS_DIR> \
#       --subjects 1 2 3 4 5 6 7 8 9
# ============================================================

set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs

RESULTS_DIR="${1:?Usage: bash submit_fair_baseline.sh RESULTS_DIR MOABB_DATASET [N_SUBJECTS]}"
MOABB_DATASET="${2:?Usage: bash submit_fair_baseline.sh RESULTS_DIR MOABB_DATASET [N_SUBJECTS]}"
N_SUBJECTS="${3:-9}"
N_FOLDS="${N_FOLDS:-5}"
SBATCH_TIME="${SBATCH_TIME:-}"

if [ ! -d "${RESULTS_DIR}" ]; then
    echo "ERROR: ${RESULTS_DIR} does not exist -- this script reuses an" >&2
    echo "       already-completed training run's saved artifacts, it" >&2
    echo "       does not create them." >&2
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
echo "  Fair baseline (full time-series LDA/SVM)"
echo "  Results dir reused: ${RESULTS_DIR}"
echo "  Dataset:             ${MOABB_DATASET}"
echo "  Subjects:            ${SUBJECTS}"
echo "  N_FOLDS:             ${N_FOLDS}"
echo "  Array tasks:         ${ARRAY_TASKS}  (${N_TASKS} total)"
[ -n "${SBATCH_TIME}" ] && echo "  Wall time override:  ${SBATCH_TIME}"
echo "=============================================="

TIME_FLAG=""
[ -n "${SBATCH_TIME}" ] && TIME_FLAG="--time=${SBATCH_TIME}"

sbatch \
    --array="${ARRAY_TASKS}" \
    ${TIME_FLAG} \
    --export="ALL,RESULTS_DIR=${RESULTS_DIR},MOABB_DATASET=${MOABB_DATASET},N_FOLDS=${N_FOLDS}" \
    run_fair_baseline_array.sh

echo ""
echo "Monitor: squeue -u \$USER"
echo "Once all tasks finish, aggregate with:"
echo "  python aggregate_fair_baseline.py --results-dir ${RESULTS_DIR} --subjects ${SUBJECTS}"
