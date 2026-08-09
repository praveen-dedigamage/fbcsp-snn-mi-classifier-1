#!/bin/bash
# ============================================================================
#  Roihu — EA/CSP fusion experiment
#
#      sbatch roihu/03_fusion.sh                       # all 12 subjects
#      SUBJECTS="1 2 3" sbatch roihu/03_fusion.sh      # pilot first
#
#  Folds the Euclidean-Alignment whitener into the CSP filters
#  (W_eff = R^T W) and re-runs the bit sweep on both forms, to test whether
#  the 4-bit collapse attributed to the whitener survives when it is no
#  longer a separately quantised stage.
#
#  Post-hoc on saved fold artifacts: no retraining, so any accuracy
#  difference comes from the representation alone. Reads Results_bnci2015
#  read-only and writes Results_fusion, so it is safe to run alongside the
#  quantisation sweep.
#
#  Check the FP32 arm first. The fusion is algebraically exact, so
#  `identity_holds` must be true; if it is not, the assumed transform order
#  is wrong and no bit-width number from the run means anything.
# ============================================================================
#SBATCH --job-name=fbcsp_fusion
# Account comes from SBATCH_ACCOUNT in the environment; add
#     export SBATCH_ACCOUNT=project_XXXXXXX
# to ~/.bashrc. Deliberately not an #SBATCH line: a committed
# project id has to be re-substituted after every pull, and that
# conflicted on four separate occasions.
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/fusion_%j.out
#SBATCH --error=logs/fusion_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

# DATASET / RESULTS_DIR / N_FOLDS / SEED / MNE_DATA / PYTORCH_MODULE
source roihu/env.sh

case "${DATASET}" in
    Cho2017)      _default_subjects="$(seq 1 52 | tr '\n' ' ')" ;;
    BNCI2014_002) _default_subjects="$(seq 1 14 | tr '\n' ' ')" ;;
    *)            _default_subjects="1 2 3 4 5 6 7 8 9 10 11 12" ;;
esac
SUBJECTS="${SUBJECTS:-${_default_subjects}}"
BITS="${BITS:-4 6 8 16 32}"
FUSION_OUT="${FUSION_OUT:-Results_fusion}"

module purge
module load "${PYTORCH_MODULE}"

echo "=============================================="
echo "  EA/CSP fusion — ${DATASET}"
echo "  results  : ${RESULTS_DIR}  (read-only)"
echo "  subjects : ${SUBJECTS}"
echo "  bits     : ${BITS}"
echo "  node     : $(hostname)  arch: $(uname -m)"
echo "=============================================="

srun python -u fusion_experiment.py \
    --results-dir "${RESULTS_DIR}" \
    --dataset "${DATASET}" \
    --subjects ${SUBJECTS} \
    --n-folds "${N_FOLDS}" \
    --bits ${BITS} \
    --seed "${SEED}" \
    --output-dir "${FUSION_OUT}"

echo "finished: $(date)"
echo "Results in ${FUSION_OUT}/"
