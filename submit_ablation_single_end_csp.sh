#!/bin/bash
# ============================================================
# FBCSP-SNN — Ablation: single-end CSP (B11a)
#
# Trains with --csp-single-end (2m eigenvectors from the largest-eigenvalue
# end only, same total filter count as the default dual-end design) instead
# of the paper's dual-end CSP. Isolates whether dual-end extraction actually
# contributes, holding filter budget constant.
#
# Usage:
#   bash submit_ablation_single_end_csp.sh [RESULTS_DIR [N_SUBJECTS]]
#
# Default results dir: Results_ablation_single_end_csp
# Default subjects:    1..9 (BNCI2014-001)
# ============================================================

set -euo pipefail

cd "$(dirname "$0")"

RESULTS_DIR="${1:-Results_ablation_single_end_csp}"
N_SUBJECTS="${2:-9}"

if [ -z "${SUBJECTS:-}" ]; then
    SUBJECTS=$(seq 1 "${N_SUBJECTS}" | tr '\n' ' ' | sed 's/ $//')
fi
export SUBJECTS

echo "=============================================="
echo "  FBCSP-SNN — Ablation: single-end CSP (B11a)"
echo "  Subjects:    ${SUBJECTS}"
echo "  Results dir: ${RESULTS_DIR}"
echo "=============================================="
echo ""

bash submit_puhti.sh "${RESULTS_DIR}" --csp-single-end
