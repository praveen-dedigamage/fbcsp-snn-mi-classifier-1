#!/bin/bash
# ============================================================
# FBCSP-SNN — Submit Cho2017 cross-dataset run (B2 — Table V)
#
# Dataset: Cho2017
#   2-class MI (left/right hand), 52 subjects, 64 channels,
#   512 Hz, single session -> stratified 80/20 split.
#
# Usage:
#   bash submit_cho2017.sh [RESULTS_DIR [N_SUBJECTS]]
#
# Default results dir: Results_cho2017
# Default subjects:    1..52
#
# Examples:
#   bash submit_cho2017.sh                              # subjects 1-52
#   bash submit_cho2017.sh Results_cho2017 5           # subjects 1-5
#   SUBJECTS="1 2 3" bash submit_cho2017.sh            # specific subjects
# ============================================================

set -euo pipefail

cd "$(dirname "$0")"

RESULTS_DIR="${1:-Results_cho2017}"
N_SUBJECTS="${2:-52}"

export MOABB_DATASET="Cho2017"

# Build subject list unless already set by caller
if [ -z "${SUBJECTS:-}" ]; then
    SUBJECTS=$(seq 1 "${N_SUBJECTS}" | tr '\n' ' ' | sed 's/ $//')
fi
export SUBJECTS

# 512 Hz x ~3 s epoch -> ~1500+ samples/trial, 64 channels. Similar cost
# profile to BNCI2015_001; request 4-hour wall time to be safe.
export SBATCH_TIME="4:00:00"

echo "=============================================="
echo "  FBCSP-SNN — Cho2017 cross-dataset run"
echo "  Dataset:     ${MOABB_DATASET}"
echo "  Subjects:    ${SUBJECTS}"
echo "  Results dir: ${RESULTS_DIR}"
echo "  Wall time:   ${SBATCH_TIME} per fold (512 Hz, 64 ch)"
echo "=============================================="
echo ""

bash submit_puhti.sh "${RESULTS_DIR}"
