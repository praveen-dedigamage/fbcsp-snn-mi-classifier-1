#!/bin/bash
#SBATCH --job-name=fbcsp_freqanalysis
#SBATCH --account=project_2003397
#SBATCH --partition=small
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --output=logs/fbcsp_freqanalysis_%j.out
#SBATCH --error=logs/fbcsp_freqanalysis_%j.err

# ============================================================
# FBCSP-SNN — Frequency-domain diagnostic analysis
#
# Loads MOABB data for all 9 subjects and produces:
#   A. Per-class PSD overlays (Session 1 + Session 2)
#   B. Fisher discriminability spectrum per subject
#   C. Channel × frequency Fisher heatmap
#   D. STFT time-frequency spectrogram per class
#   E. Session 1 vs Session 2 PSD shift
#   F. Cross-subject Fisher ratio heatmap
#
# Output: Results/FreqAnalysis/  (or custom RESULTS_DIR)
#
# Submit standalone:
#   sbatch run_puhti_freq_analysis.sh
#
# Or after training jobs complete (e.g. via submit_puhti.sh):
#   sbatch --dependency=afterok:<TRAIN_JOBID> run_puhti_freq_analysis.sh
#
# To skip slow STFT plots (saves ~20 min):
#   sbatch --export="ALL,NO_STFT=1" run_puhti_freq_analysis.sh
# ============================================================

set -euo pipefail

PROJECT_DIR=/scratch/project_2003397/praveen/fbcsp-snn-mi-classifier-1
RESULTS_DIR="${RESULTS_DIR:-Results}"
NO_STFT="${NO_STFT:-0}"

echo "=============================================="
echo "  FBCSP-SNN Frequency Analysis"
echo "  RESULTS_DIR: ${RESULTS_DIR}"
echo "  NO_STFT:     ${NO_STFT}"
echo "  Node:        $(hostname)"
echo "  Start:       $(date)"
echo "=============================================="

unset SINGULARITY_BIND
unset APPTAINER_BIND

source "${PROJECT_DIR}/.venv/bin/activate"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export MNE_DATA=/scratch/project_2003397/praveen/mne_data

cd "${PROJECT_DIR}"
mkdir -p logs

# Build optional --no-stft flag
STFT_FLAG=""
if [ "${NO_STFT}" = "1" ]; then
    STFT_FLAG="--no-stft"
fi

python analyze_frequency.py \
    --subjects 1 2 3 4 5 6 7 8 9 \
    --dataset BNCI2014_001 \
    --results-dir "${RESULTS_DIR}" \
    ${STFT_FLAG}

echo ""
echo "Frequency analysis complete."
echo "Figures: ${PROJECT_DIR}/${RESULTS_DIR}/FreqAnalysis/"
echo "End: $(date)"
