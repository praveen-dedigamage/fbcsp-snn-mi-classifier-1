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
RESULTS_DIR="${RESULTS_DIR:-Results}"

echo "=============================================="
echo "  FBCSP-SNN Cross-Subject Analysis"
echo "  RESULTS_DIR: ${RESULTS_DIR}"
echo "  Node:        $(hostname)"
echo "  Start:       $(date)"
echo "=============================================="

unset SINGULARITY_BIND
unset APPTAINER_BIND

source "${PROJECT_DIR}/.venv/bin/activate"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

cd "${PROJECT_DIR}"

# Cross-subject summary table + bar chart
python analyze_results.py \
    --results-dir "${RESULTS_DIR}" \
    --subjects 1 2 3 4 5 6 7 8 9

# Print compact results summary
echo ""
echo "=============================================="
echo "  RESULTS SUMMARY  (${RESULTS_DIR})"
echo "=============================================="
python - <<'PYEOF'
import csv, pathlib, statistics, os

results_dir = pathlib.Path(os.environ.get("RESULTS_DIR", "Results"))
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
    lda_vals = [float(r["test_acc_lda"]) * 100 for r in folds if r.get("test_acc_lda")]
    svm_vals = [float(r["test_acc_svm"]) * 100 for r in folds if r.get("test_acc_svm")]
    lda_mean = statistics.mean(lda_vals) if lda_vals else None
    svm_mean = statistics.mean(svm_vals) if svm_vals else None
    rows.append((s, mean, std, len(vals), lda_mean, svm_mean))

print(f"  {'Subject':<10} {'SNN FP32':>10} {'LDA':>8} {'SVM':>8} {'Folds':>6}")
print(f"  {'-'*48}")
for s, mean, std, n, lda, svm in rows:
    lda_str = f"{lda:>7.1f}%" if lda is not None else "    N/A"
    svm_str = f"{svm:>7.1f}%" if svm is not None else "    N/A"
    print(f"  S{s:<9} {mean:>9.1f}%  {lda_str}  {svm_str}  {n:>5}")
print(f"  {'-'*48}")
if fp32_all:
    grand_mean = statistics.mean(fp32_all)
    grand_std  = statistics.stdev(fp32_all) if len(fp32_all) > 1 else 0.0
    lda_vals_all = [r[3] for r in rows if r[3] is not None]
    svm_vals_all = [r[4] for r in rows if r[4] is not None]
    lda_grand = f"{statistics.mean(lda_vals_all):>7.1f}%" if lda_vals_all else "    N/A"
    svm_grand = f"{statistics.mean(svm_vals_all):>7.1f}%" if svm_vals_all else "    N/A"
    print(f"  {'Mean':<10} {grand_mean:>9.1f}%  {lda_grand}  {svm_grand}")
    print()
    gap = 70.0 - grand_mean
    status = "TARGET MET ✓" if gap <= 0 else f"{gap:.1f}pp below target (70%)"
    print(f"  SNN Status: {status}")
PYEOF

echo ""
echo "Results saved in ${PROJECT_DIR}/${RESULTS_DIR}/"
echo "End: $(date)"
