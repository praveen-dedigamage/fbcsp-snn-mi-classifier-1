#!/bin/bash
# ============================================================
# FBCSP-SNN — One-shot submit with smart dependency chaining
#
# Usage:
#   cd /scratch/project_2003397/praveen/fbcsp-snn-mi-classifier-1
#   bash submit_puhti.sh [RESULTS_DIR [EXTRA_TRAIN_ARGS...]]
#
# Examples:
#   bash submit_puhti.sh                                    # default: Results/
#   bash submit_puhti.sh Results_mif005 --mi-fraction 0.05
#   bash submit_puhti.sh Results_mif010 --mi-fraction 0.10
#   bash submit_puhti.sh Results_mif015 --mi-fraction 0.15
#
# Each experiment writes to its own RESULTS_DIR so parallel runs
# never overwrite each other.
#
# Dependency chain:
#   - Each subject's aggregate job triggers as soon as its own
#     N_FOLDS training tasks complete (not waiting for all 45)
#   - The analyze job triggers once all 9 aggregate jobs finish
#
# Stage 1: Training     45 tasks  (9 subjects × N_FOLDS, all parallel)
# Stage 2: Aggregate     9 jobs   (one per subject, afterok its folds)
# Stage 3: Analyze       1 job    (afterok all 9 aggregate jobs)
# ============================================================

set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs

# --- Parse arguments --------------------------------------------------------
RESULTS_DIR="${1:-Results}"
shift 2>/dev/null || true   # remove RESULTS_DIR; OK if no args were given
EXTRA_ARGS="${*}"           # remaining args forwarded to every training task

echo "RESULTS_DIR: ${RESULTS_DIR}"
echo "EXTRA_ARGS:  ${EXTRA_ARGS:-<none>}"
echo ""

# Read N_FOLDS from the array script — single source of truth
N_FOLDS=$(grep '^N_FOLDS=' run_puhti_array.sh | cut -d= -f2)
if [ -z "${N_FOLDS}" ]; then
    echo "ERROR: could not read N_FOLDS from run_puhti_array.sh" >&2
    exit 1
fi
echo "N_FOLDS=${N_FOLDS} (read from run_puhti_array.sh)"

# Remove fold directories with index >= N_FOLDS (stale from a previous higher-fold run)
echo "Cleaning up stale fold directories (fold_${N_FOLDS} and above) in ${RESULTS_DIR}/..."
REMOVED=0
for S in $(seq 1 9); do
    for F in "${RESULTS_DIR}"/Subject_${S}/fold_*/; do
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

# --- Stage 1: Training array (N_FOLDS × 9 tasks) ---------------------------
# RESULTS_DIR and EXTRA_ARGS are exported into every array task environment.
TRAIN_OUTPUT=$(sbatch \
    --export="ALL,RESULTS_DIR=${RESULTS_DIR},EXTRA_ARGS=${EXTRA_ARGS}" \
    run_puhti_array.sh)
TRAIN_JOBID=$(echo "${TRAIN_OUTPUT}" | awk '{print $4}')
echo "${TRAIN_OUTPUT}"
echo ""

# --- Stage 2: One aggregate job per subject ---------------------------------
# Each subject S owns task IDs: (S-1)*N_FOLDS+1 .. S*N_FOLDS
# We depend only on those specific array tasks, not the full array.
AGG_JOBIDS=()
for S in $(seq 1 9); do
    START=$(( (S - 1) * N_FOLDS + 1 ))
    END=$(( S * N_FOLDS ))

    # Build afterok dependency for just this subject's fold tasks
    DEP="afterok"
    for T in $(seq ${START} ${END}); do
        DEP="${DEP}:${TRAIN_JOBID}_${T}"
    done

    AGG_OUTPUT=$(sbatch --dependency=${DEP} \
                        --job-name="fbcsp_agg_S${S}" \
                        --export="ALL,SUBJECT_ID=${S},RESULTS_DIR=${RESULTS_DIR}" \
                        run_puhti_aggregate.sh)
    AGG_JOBID=$(echo "${AGG_OUTPUT}" | awk '{print $4}')
    AGG_JOBIDS+=("${AGG_JOBID}")
    echo "Subject ${S}: aggregate job ${AGG_JOBID} (depends on tasks ${START}-${END})"
done
echo ""

# --- Stage 3: Cross-subject analysis (after all 9 aggregate jobs) ----------
ANALYZE_DEP="afterok"
for JID in "${AGG_JOBIDS[@]}"; do
    ANALYZE_DEP="${ANALYZE_DEP}:${JID}"
done

ANALYZE_OUTPUT=$(sbatch --dependency=${ANALYZE_DEP} \
    --export="ALL,RESULTS_DIR=${RESULTS_DIR}" \
    run_puhti_analyze.sh)
ANALYZE_JOBID=$(echo "${ANALYZE_OUTPUT}" | awk '{print $4}')
echo "${ANALYZE_OUTPUT}"
echo ""

echo "=============================================="
echo "  All jobs submitted"
echo "  Results dir: ${RESULTS_DIR}"
echo "  Extra args:  ${EXTRA_ARGS:-<none>}"
echo "  Train job:   ${TRAIN_JOBID} ($(( 9 * N_FOLDS )) tasks)"
echo "  Aggregate:   ${AGG_JOBIDS[*]} (9 jobs)"
echo "  Analyze:     ${ANALYZE_JOBID} (after all aggregates)"
echo ""
echo "  Monitor: squeue -u \$USER"
echo "  Final summary in: logs/fbcsp_analyze_${ANALYZE_JOBID}.out"
echo "=============================================="
