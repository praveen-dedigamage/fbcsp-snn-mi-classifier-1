#!/bin/bash
#SBATCH --job-name=fbcsp_analyze
#SBATCH --account=project_2003397
#SBATCH --partition=small
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/fbcsp_analyze_%j.out
#SBATCH --error=logs/fbcsp_analyze_%j.err

# ============================================================
# FBCSP-SNN — Cross-subject analysis + result summary
# Runs automatically after all 9 aggregate jobs via submit_puhti.sh
#
# Bumped 10min -> 30min 2026-07-11: same small-partition, CPU-only job
# category as run_puhti_aggregate.sh, which showed 10-20+ minute stalls
# before the first Python log line even printed. See RESULTS_LOG.md.
# ============================================================

set -euo pipefail

PROJECT_DIR=/scratch/project_2003397/praveen/fbcsp-snn-mi-classifier-1
RESULTS_DIR="${RESULTS_DIR:-Results}"
SUBJECTS="${SUBJECTS:-1 2 3 4 5 6 7 8 9}"

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
    --subjects ${SUBJECTS} \
    --moabb-dataset "${MOABB_DATASET:-BNCI2014_001}"

# Print compact results summary
echo ""
echo "=============================================="
echo "  RESULTS SUMMARY  (${RESULTS_DIR})"
echo "=============================================="
SUBJECTS="${SUBJECTS}" python - <<'PYEOF'
import csv, pathlib, statistics, os

results_dir = pathlib.Path(os.environ.get("RESULTS_DIR", "Results"))
subjects    = [int(s) for s in os.environ.get("SUBJECTS", "1 2 3 4 5 6 7 8 9").split()]

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
    fp32_vals = [float(r["test_acc_fp32"]) * 100 for r in folds if r.get("test_acc_fp32")]
    if not fp32_vals:
        print(f"  S{s}: no data")
        continue

    fp32_mean = statistics.mean(fp32_vals)
    fp32_std  = statistics.stdev(fp32_vals) if len(fp32_vals) > 1 else 0.0
    fp32_all.extend(fp32_vals)

    lda_vals = [float(r["test_acc_lda"]) * 100 for r in folds if r.get("test_acc_lda")]
    svm_vals = [float(r["test_acc_svm"]) * 100 for r in folds if r.get("test_acc_svm")]
    lda_mean = statistics.mean(lda_vals) if lda_vals else None
    svm_mean = statistics.mean(svm_vals) if svm_vals else None

    rows.append((s, fp32_mean, fp32_std, len(fp32_vals), lda_mean, svm_mean))

# Header
print(f"  {'Subj':<6} {'FP32':>7} {'LDA':>7} {'SVM':>7} {'N':>3}")
print(f"  {'-'*40}")
for s, fp32_mean, fp32_std, n, lda, svm in rows:
    lda_str = f"{lda:>6.1f}%" if lda is not None else "   N/A"
    svm_str = f"{svm:>6.1f}%" if svm is not None else "   N/A"
    print(f"  S{s:<5} {fp32_mean:>6.1f}%  {lda_str}  {svm_str}  {n:>3}")

print(f"  {'-'*40}")
if fp32_all:
    gm   = statistics.mean(fp32_all)
    lda_all = [r[4] for r in rows if r[4] is not None]
    svm_all = [r[5] for r in rows if r[5] is not None]
    lda_g = f"{statistics.mean(lda_all):>6.1f}%" if lda_all else "   N/A"
    svm_g = f"{statistics.mean(svm_all):>6.1f}%" if svm_all else "   N/A"
    print(f"  {'MEAN':<6} {gm:>6.1f}%  {lda_g}  {svm_g}")
    print()
    gap = 70.0 - gm
    status = "TARGET MET" if gap <= 0 else f"{gap:.1f}pp below target (70%)"
    print(f"  SNN FP32: {status}")
PYEOF

echo ""
echo "Results saved in ${PROJECT_DIR}/${RESULTS_DIR}/"
echo "End: $(date)"
