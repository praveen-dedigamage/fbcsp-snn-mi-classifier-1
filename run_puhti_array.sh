#!/bin/bash
#SBATCH --job-name=fbcsp_snn
#SBATCH --account=<YOUR_PROJECT>        # replace with your CSC project, e.g. project_2012345
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4              # 4 CPU threads for data loading / scipy
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=1-9                    # one task per BCI-IV-2a subject
#SBATCH --output=logs/fbcsp_snn_S%a_%j.out
#SBATCH --error=logs/fbcsp_snn_S%a_%j.err

# ============================================================
# FBCSP-SNN — CSC Puhti SLURM array job
# Each task trains one subject (SLURM_ARRAY_TASK_ID = subject ID)
#
# Submit:
#   mkdir -p logs
#   sbatch run_puhti_array.sh
#
# Monitor:
#   squeue -u $USER
#   sacct -j <JOBID> --format=JobID,State,Elapsed,MaxRSS
# ============================================================

set -euo pipefail

SUBJECT_ID=${SLURM_ARRAY_TASK_ID}

echo "=============================================="
echo "  FBCSP-SNN  Subject ${SUBJECT_ID}"
echo "  Node:   $(hostname)"
echo "  GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "  Start:  $(date)"
echo "=============================================="

# ---- Environment ----------------------------------------------------------
module purge
module load pytorch/2.1          # adjust to available module; must include CUDA
module load python-data/3.10-22.09   # or whichever Python module provides scipy/sklearn

# If using a venv instead of modules:
# source /scratch/<project>/<user>/venv/bin/activate

# Ensure the project root is on the Python path
PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

# MOABB downloads data here; point to scratch for quota reasons
export MNE_DATA=/scratch/${SLURM_JOB_ACCOUNT}/mne_data

# ---- Training -------------------------------------------------------------
python main.py train \
    --source moabb \
    --moabb-dataset BNCI2014_001 \
    --subject-id "${SUBJECT_ID}" \
    --adaptive-bands \
    --n-adaptive-bands 6 \
    --csp-components-per-band 4 \
    --hidden-neurons 64 \
    --population-per-class 20 \
    --beta 0.95 \
    --dropout-prob 0.5 \
    --epochs 1000 \
    --early-stopping-patience 100 \
    --early-stopping-warmup 100 \
    --lr 1e-3 \
    --weight-decay 0.1 \
    --spiking-prob 0.7 \
    --feature-selection-method mibif \
    --feature-percentile 50.0 \
    --n-folds 10 \
    --results-dir Results

EXIT_CODE=$?

echo ""
echo "Training finished with exit code ${EXIT_CODE}"
echo "End: $(date)"
exit ${EXIT_CODE}

# ============================================================
# After all array tasks complete, run aggregation:
#
#   python analyze_results.py --results-dir Results --subjects 1 2 3 4 5 6 7 8 9
#
# Or trigger it automatically as a dependent job:
#
#   sbatch --dependency=afterok:<ARRAY_JOBID> <<'EOF'
#   #!/bin/bash
#   #SBATCH --job-name=fbcsp_snn_aggregate
#   #SBATCH --account=<YOUR_PROJECT>
#   #SBATCH --partition=small
#   #SBATCH --time=00:30:00
#   #SBATCH --mem=8G
#   module load python-data/3.10-22.09
#   for S in 1 2 3 4 5 6 7 8 9; do
#       python main.py aggregate --subject-id $S --n-folds 10
#   done
#   python analyze_results.py --results-dir Results --subjects 1 2 3 4 5 6 7 8 9
#   EOF
# ============================================================
