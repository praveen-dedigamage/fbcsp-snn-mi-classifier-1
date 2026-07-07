#!/bin/bash
# ============================================================
# FBCSP-SNN — Ablation: cross-entropy loss (B11c)
#
# Trains with --loss-type cross_entropy (standard softmax cross-entropy on
# population-summed spike counts) instead of the paper's Van Rossum
# spike-train loss. Isolates whether the Van Rossum loss's spike-timing
# sensitivity contributes, independent of the SNN architecture and
# population-coded readout, which are unchanged.
#
# Usage:
#   bash submit_ablation_cross_entropy.sh [RESULTS_DIR [N_SUBJECTS]]
#
# Default results dir: Results_ablation_cross_entropy
# Default subjects:    1..9 (BNCI2014-001)
# ============================================================

set -euo pipefail

cd "$(dirname "$0")"

RESULTS_DIR="${1:-Results_ablation_cross_entropy}"
N_SUBJECTS="${2:-9}"

if [ -z "${SUBJECTS:-}" ]; then
    SUBJECTS=$(seq 1 "${N_SUBJECTS}" | tr '\n' ' ' | sed 's/ $//')
fi
export SUBJECTS

echo "=============================================="
echo "  FBCSP-SNN — Ablation: cross-entropy loss (B11c)"
echo "  Subjects:    ${SUBJECTS}"
echo "  Results dir: ${RESULTS_DIR}"
echo "=============================================="
echo ""

bash submit_puhti.sh "${RESULTS_DIR}" --loss-type cross_entropy
