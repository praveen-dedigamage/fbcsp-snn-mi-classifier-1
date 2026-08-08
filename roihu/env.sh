# ============================================================================
#  Shared environment for all Roihu jobs.  Sourced, not executed.
#
#  Why this exists: the MOABB/MNE cache defaults to ~/mne_data, i.e. $HOME.
#  BNCI2015-001 (12 subjects, 13 channels) fits there at ~1.6 GB, but Cho2017
#  is 52 subjects at 64 channels and would very likely exhaust a CSC home
#  quota part-way through a download -- leaving a half-populated cache that
#  the training array then races on, failing much later and confusingly.
#  Everything therefore lives on scratch.
#
#  PROJECT_ROOT is derived from the submit directory rather than hard-coded,
#  so this works for any user/project without editing:
#      /scratch/<project>/<user>/fbcsp   ->   /scratch/<project>/<user>
# ============================================================================

_submit_dir="${SLURM_SUBMIT_DIR:-$(pwd)}"
PROJECT_ROOT="${PROJECT_ROOT:-$(dirname "${_submit_dir}")}"

export MNE_DATA="${MNE_DATA:-${PROJECT_ROOT}/mne_data}"
mkdir -p "${MNE_DATA}"

# Verified on roihu-gpu via `module spider python-pytorch` (Aug 2026).
PYTORCH_MODULE="${PYTORCH_MODULE:-python-pytorch/2.10}"

# Dataset selection. Override per job:  DATASET=Cho2017 sbatch ...
DATASET="${DATASET:-BNCI2015_001}"
N_FOLDS="${N_FOLDS:-5}"
SEED="${SEED:-42}"

# Results land in a dataset-specific directory so two datasets cannot
# overwrite each other's fold artifacts.
case "${DATASET}" in
    BNCI2015_001) _tag="bnci2015" ;;
    Cho2017)      _tag="cho2017"  ;;
    *)            _tag="$(echo "${DATASET}" | tr '[:upper:]' '[:lower:]')" ;;
esac
RESULTS_DIR="${RESULTS_DIR:-Results_${_tag}}"
OUT_DIR="${OUT_DIR:-Results_quant}"

echo "--- roihu/env.sh ---"
echo "  PROJECT_ROOT : ${PROJECT_ROOT}"
echo "  MNE_DATA     : ${MNE_DATA}"
echo "  DATASET      : ${DATASET}"
echo "  RESULTS_DIR  : ${RESULTS_DIR}"
echo "  N_FOLDS      : ${N_FOLDS}   SEED: ${SEED}"
echo "--------------------"
