#!/bin/bash
# ============================================================
# FBCSP-SNN — Submit PhysionetMI cross-dataset run (item 10b)
#
# Dataset: PhysionetMI
#   4-class MI (left hand, right hand, feet, tongue)
#   109 subjects, 64 channels, 160 Hz, single session.
#   Cross-dataset proof: same class count as BNCI2014_001 but
#   different electrode layout (64 vs 22) and sampling rate
#   (160 vs 250 Hz). Single session → StratifiedShuffleSplit.
#
# Usage:
#   bash submit_physionet.sh [RESULTS_DIR [N_SUBJECTS]]
#
# Default results dir: Results_physionet
# Default subjects:    1..20  (override with SUBJECTS env var)
#
# Examples:
#   bash submit_physionet.sh                            # subjects 1-20
#   bash submit_physionet.sh Results_physionet 30       # subjects 1-30
#   SUBJECTS="1 2 3" bash submit_physionet.sh           # specific subjects
# ============================================================

set -euo pipefail

cd "$(dirname "$0")"

RESULTS_DIR="${1:-Results_physionet}"
N_SUBJECTS="${2:-20}"

export MOABB_DATASET="PhysionetMI"

# Build subject list unless already set by caller
if [ -z "${SUBJECTS:-}" ]; then
    SUBJECTS=$(seq 1 "${N_SUBJECTS}" | tr '\n' ' ' | sed 's/ $//')
fi
export SUBJECTS

# 160 Hz → T ≈ 640 timesteps (shorter than BNCI2014_001 at 250 Hz).
# Default 2-hour wall time is sufficient; no override needed.

echo "=============================================="
echo "  FBCSP-SNN — PhysionetMI cross-dataset run"
echo "  Dataset:     ${MOABB_DATASET}"
echo "  Subjects:    ${SUBJECTS}"
echo "  Results dir: ${RESULTS_DIR}"
echo "  Wall time:   2:00:00 per fold (160 Hz)"
echo "=============================================="
echo ""

bash submit_puhti.sh "${RESULTS_DIR}"
