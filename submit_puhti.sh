#!/bin/bash
# ============================================================
# FBCSP-SNN — One-shot submit with smart dependency chaining
#
# Usage:
#   cd /scratch/project_2003397/praveen/fbcsp-snn-mi-classifier-1
#   bash submit_puhti.sh [RESULTS_DIR [EXTRA_TRAIN_ARGS...]]
#
# Examples:
#   bash submit_puhti.sh                                    # all 9 subjects
#   bash submit_puhti.sh Results_ptq_test                  # all 9 subjects
#   SUBJECTS="1 2" bash submit_puhti.sh Results_ptq_test   # subjects 1-2 only
#   SUBJECTS="1" bash submit_puhti.sh Results_ptq_test     # subject 1 only
#   bash submit_puhti.sh Results_mif010 --mi-fraction 0.10
#
# Override subject list:
#   SUBJECTS="1 2 3"   — space-separated subject IDs (default: 1..9)
#
# Each experiment writes to its own RESULTS_DIR so parallel runs
# never overwrite each other.
#
# Dependency chain:
#   - Each subject's aggregate job triggers as soon as its own
#     N_FOLDS training tasks complete (not waiting for all tasks)
#   - The analyze job triggers once all aggregate jobs finish
#
# Stage 1: Training     N tasks  (selected subjects × N_FOLDS, all parallel)
# Stage 2: Aggregate    1 job per subject (afterok its own folds only)
# Stage 3: Analyze      1 job    (afterok all aggregate jobs)
# ============================================================

set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs

# --- Parse arguments --------------------------------------------------------
RESULTS_DIR="${1:-Results}"
shift 2>/dev/null || true   # remove RESULTS_DIR; OK if no args were given
EXTRA_ARGS="${*}"           # remaining args forwarded to every training task

# Override the array script via environment variable, e.g.:
#   ARRAY_SCRIPT=run_puhti_static6.sh bash submit_puhti.sh Results_static6 --augment-windows
ARRAY_SCRIPT="${ARRAY_SCRIPT:-run_puhti_array.sh}"

# Subject list — override with: SUBJECTS="1 2 3" bash submit_puhti.sh ...
SUBJECTS="${SUBJECTS:-1 2 3 4 5 6 7 8 9}"

echo "RESULTS_DIR:  ${RESULTS_DIR}"
echo "ARRAY_SCRIPT: ${ARRAY_SCRIPT}"
echo "SUBJECTS:     ${SUBJECTS}"
echo "EXTRA_ARGS:   ${EXTRA_ARGS:-<none>}"
echo ""

# Read N_FOLDS from the chosen array script — single source of truth
N_FOLDS=$(grep '^N_FOLDS=' "${ARRAY_SCRIPT}" | cut -d= -f2)
if [ -z "${N_FOLDS}" ]; then
    echo "ERROR: could not read N_FOLDS from ${ARRAY_SCRIPT}" >&2
    exit 1
fi
echo "N_FOLDS=${N_FOLDS} (read from ${ARRAY_SCRIPT})"

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

# --- Stage 1: Training array (N_FOLDS × selected subjects) -----------------
# Compute the SLURM array task IDs for the chosen subjects.
# Task ID encoding: subject S, fold F → task (S-1)*N_FOLDS + F + 1
ARRAY_TASKS=""
for S in ${SUBJECTS}; do
    START=$(( (S - 1) * N_FOLDS + 1 ))
    END=$(( S * N_FOLDS ))
    ARRAY_TASKS="${ARRAY_TASKS:+${ARRAY_TASKS},}${START}-${END}"
done
echo "Array tasks: ${ARRAY_TASKS}"
echo ""

ENCODER_TYPE="${ENCODER_TYPE:-delta}"

TRAIN_OUTPUT=$(sbatch \
    --array="${ARRAY_TASKS}" \
    --export="ALL,RESULTS_DIR=${RESULTS_DIR},ENCODER_TYPE=${ENCODER_TYPE},EXTRA_ARGS=${EXTRA_ARGS}" \
    "${ARRAY_SCRIPT}")
TRAIN_JOBID=$(echo "${TRAIN_OUTPUT}" | awk '{print $4}')
echo "${TRAIN_OUTPUT}"
echo ""

# --- Stage 2: One aggregate job per subject ---------------------------------
# Each subject S owns specific task IDs — depend only on those.
AGG_JOBIDS=()
for S in ${SUBJECTS}; do
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
N_SUBJECTS=$(echo ${SUBJECTS} | wc -w)
echo "  Results dir:  ${RESULTS_DIR}"
echo "  Array script: ${ARRAY_SCRIPT}"
echo "  Subjects:     ${SUBJECTS}"
echo "  Extra args:   ${EXTRA_ARGS:-<none>}"
echo "  Train job:    ${TRAIN_JOBID} (${N_SUBJECTS} subjects × ${N_FOLDS} folds = $(( N_SUBJECTS * N_FOLDS )) tasks)"
echo "  Aggregate:   ${AGG_JOBIDS[*]} (${N_SUBJECTS} jobs)"
echo "  Analyze:     ${ANALYZE_JOBID} (after all aggregates)"
echo ""
echo "  Monitor: squeue -u \$USER"
echo "  Final summary in: logs/fbcsp_analyze_${ANALYZE_JOBID}.out"
echo "=============================================="
