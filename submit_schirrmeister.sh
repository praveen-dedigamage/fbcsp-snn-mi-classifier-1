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
#
# If many tasks are submitted at once (e.g. a multi-subject resubmit), cap
# concurrency to avoid shared-node/filesystem I/O contention slowing every
# task down (confirmed 2026-07-10 — see RESULTS_LOG.md):
#   ARRAY_THROTTLE=5 SUBJECTS="2 3 4 7 12" bash submit_schirrmeister.sh ...
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

# 500 Hz x ~4 s epoch -> ~2000 timesteps, 2x BNCI2014-001's ~1001 -> every
# SNN forward/backward pass costs roughly 2x. Measured ~30-36 sec/epoch on
# 2026-07-10; with epochs=1000/patience=100, a fold that doesn't plateau
# quickly can need close to the full epoch cap (~9-10h training alone).
# 4h (first attempt) and 8h (second attempt) both proved insufficient --
# 8/70 tasks TIMEOUT at 4h, then all 25/25 resubmitted tasks TIMEOUT at 8h
# (see RESULTS_LOG.md). Bottleneck is per-epoch compute cost, not node
# contention -- bumping this further, not ARRAY_THROTTLE, is the fix.
# Respect a pre-set SBATCH_TIME (e.g. from the caller's environment)
# instead of always overriding it.
export SBATCH_TIME="${SBATCH_TIME:-8:00:00}"

echo "=============================================="
echo "  FBCSP-SNN — Schirrmeister2017 cross-dataset run"
echo "  Dataset:     ${MOABB_DATASET}"
echo "  Subjects:    ${SUBJECTS}"
echo "  Results dir: ${RESULTS_DIR}"
echo "  Wall time:   ${SBATCH_TIME} per fold (500 Hz, 128 ch)"
echo "=============================================="
echo ""

bash submit_puhti.sh "${RESULTS_DIR}"
