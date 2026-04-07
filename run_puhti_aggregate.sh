#!/bin/bash
#SBATCH --job-name=fbcsp_snn_aggregate
#SBATCH --account=project_2003397
#SBATCH --partition=small
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/fbcsp_snn_aggregate_%j.out
#SBATCH --error=logs/fbcsp_snn_aggregate_%j.err

# ============================================================
# FBCSP-SNN — Aggregation + cross-subject analysis
# Run after all array job tasks have completed.
#
# Submit as a dependent job:
#   sbatch --dependency=afterok:<ARRAY_JOBID> run_puhti_aggregate.sh
#
# Or run interactively after the array finishes:
#   srun --account=project_2003397 --partition=small --mem=16G \
#        --time=00:30:00 bash run_puhti_aggregate.sh
# ============================================================

set -euo pipefail

PROJECT_DIR=/scratch/project_2003397/praveen/fbcsp-snn-mi-classifier-1

echo "=============================================="
echo "  FBCSP-SNN Aggregation"
echo "  Node:  $(hostname)"
echo "  Start: $(date)"
echo "=============================================="

source "${PROJECT_DIR}/.venv/bin/activate"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export MNE_DATA=/scratch/project_2003397/praveen/mne_data

cd "${PROJECT_DIR}"

# Per-subject aggregation (collect fold JSONs → summary.csv + confusion plots)
for S in 1 2 3 4 5 6 7 8 9; do
    echo ""
    echo "--- Aggregating Subject ${S} ---"
    python main.py aggregate \
        --source moabb \
        --moabb-dataset BNCI2014_001 \
        --subject-id "${S}" \
        --n-folds 5
done

# Cross-subject summary table + bar chart
echo ""
echo "--- Cross-subject analysis ---"
python analyze_results.py \
    --results-dir Results \
    --subjects 1 2 3 4 5 6 7 8 9

echo ""
echo "=============================================="
echo "  RESULTS SUMMARY"
echo "=============================================="
python - <<'PYEOF'
import csv, pathlib, statistics

results_dir = pathlib.Path("Results")
subjects = list(range(1, 10))

rows = []
fp32_all = []
for s in subjects:
    csv_path = results_dir / f"Subject_{s}" / "summary.csv"
    if not csv_path.exists():
        print(f"  S{s}: missing")
        continue
    with open(csv_path, newline="") as f:
        folds = [r for r in csv.DictReader(f)
                 if r.get("fold", "").strip().lower() not in ("", "mean")]
    vals = [float(r["test_fp32_acc"]) * 100 for r in folds if r.get("test_fp32_acc")]
    if not vals:
        print(f"  S{s}: no data")
        continue
    mean = statistics.mean(vals)
    std  = statistics.stdev(vals) if len(vals) > 1 else 0.0
    fp32_all.extend(vals)
    rows.append((s, mean, std, len(vals)))

print(f"  {'Subject':<10} {'Mean FP32':>10} {'Std':>8} {'Folds':>6}")
print(f"  {'-'*38}")
for s, mean, std, n in rows:
    print(f"  S{s:<9} {mean:>9.1f}%  {std:>6.1f}  {n:>5}")
print(f"  {'-'*38}")
if fp32_all:
    grand_mean = statistics.mean(fp32_all)
    grand_std  = statistics.stdev(fp32_all) if len(fp32_all) > 1 else 0.0
    print(f"  {'Mean':<10} {grand_mean:>9.1f}%  {grand_std:>6.1f}")
    print()
    target = 70.0
    gap = target - grand_mean
    status = "TARGET MET" if gap <= 0 else f"{gap:.1f}pp below target (70%)"
    print(f"  Status: {status}")
PYEOF

echo ""
echo "Aggregation complete. Results in ${PROJECT_DIR}/Results/"
echo "End: $(date)"
