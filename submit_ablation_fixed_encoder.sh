#!/bin/bash
# ============================================================
# FBCSP-SNN — Ablation: fixed-threshold spike encoder (B11b)
#
# Trains with --encoder-type fixed (constant threshold, no adapt_inc/decay
# dynamics) instead of the paper's adaptive-threshold delta encoder.
# Isolates whether threshold adaptivity itself matters, independent of the
# underlying delta-encoding scheme.
#
# Usage:
#   bash submit_ablation_fixed_encoder.sh [RESULTS_DIR [N_SUBJECTS]]
#
# Default results dir: Results_ablation_fixed_encoder
# Default subjects:    1..9 (BNCI2014-001)
# ============================================================

set -euo pipefail

cd "$(dirname "$0")"

RESULTS_DIR="${1:-Results_ablation_fixed_encoder}"
N_SUBJECTS="${2:-9}"

if [ -z "${SUBJECTS:-}" ]; then
    SUBJECTS=$(seq 1 "${N_SUBJECTS}" | tr '\n' ' ' | sed 's/ $//')
fi
export SUBJECTS

echo "=============================================="
echo "  FBCSP-SNN — Ablation: fixed-threshold encoder (B11b)"
echo "  Subjects:    ${SUBJECTS}"
echo "  Results dir: ${RESULTS_DIR}"
echo "=============================================="
echo ""

bash submit_puhti.sh "${RESULTS_DIR}" --encoder-type fixed
