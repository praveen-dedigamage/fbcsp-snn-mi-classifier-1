#!/bin/bash
#SBATCH --job-name=fbcsp_analyze
#SBATCH --account=project_2003397
#SBATCH --partition=small
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=logs/fbcsp_analyze_%j.out
#SBATCH --error=logs/fbcsp_analyze_%j.err

# ============================================================
# FBCSP-SNN — Cross-subject analysis + result summary
# Runs automatically after all 9 aggregate jobs via submit_puhti.sh
# ============================================================

set -euo pipefail

PROJECT_DIR=/scratch/project_2003397/praveen/fbcsp-snn-mi-classifier-1

echo "=============================================="
echo "  FBCSP-SNN Cross-Subject Analysis"
echo "  Node:  $(hostname)"
echo "  Start: $(date)"
echo "=============================================="

source "${PROJECT_DIR}/.venv/bin/activate"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

cd "${PROJECT_DIR}"

# Cross-subject summary table + bar chart
python analyze_results.py \
    --results-dir Results \
    --subjects 1 2 3 4 5 6 7 8 9

# Print compact results summary
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
    gap = 70.0 - grand_mean
    status = "TARGET MET ✓" if gap <= 0 else f"{gap:.1f}pp below target (70%)"
    print(f"  Status: {status}")
PYEOF

echo ""
echo "Results saved in ${PROJECT_DIR}/Results/"
echo "End: $(date)"
