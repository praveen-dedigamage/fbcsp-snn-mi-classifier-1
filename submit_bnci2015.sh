#!/bin/bash
# ============================================================
# FBCSP-SNN — Submit BNCI2015_001 cross-dataset run (item 10)
#
# Dataset: BNCI2015_001
#   2-class MI (left/right hand), 12 subjects, 13 channels,
#   512 Hz, 2 sessions → cross-session evaluation.
#   Strongest generalisation proof: different class count,
#   channel count, sampling rate AND cross-session (same
#   evaluation rigour as BNCI2014_001).
#
# Usage:
#   bash submit_bnci2015.sh [RESULTS_DIR]
#
# Default results dir: Results_bnci2015
# ============================================================

set -euo pipefail

cd "$(dirname "$0")"

RESULTS_DIR="${1:-Results_bnci2015}"

export MOABB_DATASET="BNCI2015_001"
export SUBJECTS="1 2 3 4 5 6 7 8 9 10 11 12"

# 512 Hz → T ≈ 2048 timesteps (vs 1001 for BNCI2014_001 at 250 Hz).
# Training wall time bumped from 2h → 4h to be safe.
export SBATCH_TIME="4:00:00"

echo "=============================================="
echo "  FBCSP-SNN — BNCI2015_001 cross-dataset run"
echo "  Dataset:     ${MOABB_DATASET}"
echo "  Subjects:    ${SUBJECTS}"
echo "  Results dir: ${RESULTS_DIR}"
echo "  Wall time:   ${SBATCH_TIME} per fold (512 Hz)"
echo "=============================================="
echo ""

bash submit_puhti.sh "${RESULTS_DIR}"
