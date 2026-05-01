#!/bin/bash
# ============================================================
# FBCSP-SNN — Submit Schirrmeister2017 cross-dataset run
#
# Dataset: Schirrmeister2017 (High Gamma Dataset)
#   4-class MI (right_hand, left_hand, rest, feet)
#   14 subjects, 128 channels, 500 Hz, single session.
#   ~963 trials/subject (~240/class) → full-rank 128×128 CSP covariances.
#   Single session → StratifiedShuffleSplit 80/20.
#
# Usage:
#   bash submit_schirrmeister.sh [RESULTS_DIR [N_SUBJECTS]]
#
# Default results dir: Results_schirrmeister
# Default subjects:    1..14
#
# Examples:
#   bash submit_schirrmeister.sh                              # subjects 1-14
#   bash submit_schirrmeister.sh Results_schirrmeister 5     # subjects 1-5
#   SUBJECTS="1 2 3" bash submit_schirrmeister.sh            # specific subjects
# ============================================================

set -euo pipefail

cd "$(dirname "$0")"

RESULTS_DIR="${1:-Results_schirrmeister}"
N_SUBJECTS="${2:-14}"

export MOABB_DATASET="Schirrmeister2017"

# Build subject list unless already set by caller
if [ -z "${SUBJECTS:-}" ]; then
    SUBJECTS=$(seq 1 "${N_SUBJECTS}" | tr '\n' ' ' | sed 's/ $//')
fi
export SUBJECTS

# 500 Hz × ~4 s epoch → ~2000 samples. Riemannian mean on 128×128 is
# compute-intensive; request 4-hour wall time to be safe.
export SBATCH_TIME="4:00:00"

echo "=============================================="
echo "  FBCSP-SNN — Schirrmeister2017 cross-dataset run"
echo "  Dataset:     ${MOABB_DATASET}"
echo "  Subjects:    ${SUBJECTS}"
echo "  Results dir: ${RESULTS_DIR}"
echo "  Wall time:   ${SBATCH_TIME} per fold (500 Hz, 128 ch)"
echo "=============================================="
echo ""

bash submit_puhti.sh "${RESULTS_DIR}"
