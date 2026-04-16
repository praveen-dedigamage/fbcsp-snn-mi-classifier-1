#!/bin/bash
#SBATCH --job-name=fbcsp_mc_analyze
#SBATCH --account=project_2003397
#SBATCH --partition=small
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=logs/fbcsp_mc_analyze_%j.out
#SBATCH --error=logs/fbcsp_mc_analyze_%j.err

# ============================================================
# FBCSP-SNN — Monte Carlo cross-subject analysis
# Runs after all 9 MC array tasks complete (via submit_mc.sh).
# Merges per-subject CSVs and regenerates the combined plot.
# ============================================================

set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-Results_butterworth_mc}"
SUBJECTS="${SUBJECTS:-1 2 3 4 5 6 7 8 9}"

PROJECT_DIR=/scratch/project_2003397/praveen/fbcsp-snn-mi-classifier-1

echo "=============================================="
echo "  FBCSP-SNN Monte Carlo Analysis"
echo "  OUTPUT_DIR: ${OUTPUT_DIR}"
echo "  SUBJECTS:   ${SUBJECTS}"
echo "  Node:       $(hostname)"
echo "  Start:      $(date)"
echo "=============================================="

unset SINGULARITY_BIND
unset APPTAINER_BIND

source "${PROJECT_DIR}/.venv/bin/activate"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

cd "${PROJECT_DIR}"

# Merge all per-subject mc_raw.csv files and re-run the summary + plot
python - <<'PYEOF'
import csv, sys, os
from pathlib import Path

output_dir = Path(os.environ.get("OUTPUT_DIR", "Results_butterworth_mc"))
subjects   = [int(s) for s in os.environ.get("SUBJECTS", "1 2 3 4 5 6 7 8 9").split()]

# Read all per-subject raw CSVs written by run_butterworth_mc.py
all_rows = []
fieldnames = None
for s in subjects:
    p = output_dir / f"mc_raw_S{s}.csv"
    if not p.exists():
        # Fall back to merged file if subject ran alone
        p = output_dir / "mc_raw.csv"
    if not p.exists():
        print(f"  WARNING: no MC raw CSV for S{s}", file=sys.stderr)
        continue
    with open(p, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        all_rows.extend(r for r in reader if int(r["subject"]) == s)

if not all_rows:
    print("No MC data found.", file=sys.stderr)
    sys.exit(1)

# Write merged mc_raw.csv
merged = output_dir / "mc_raw.csv"
output_dir.mkdir(parents=True, exist_ok=True)
with open(merged, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(all_rows)
print(f"Merged {len(all_rows)} rows → {merged}")
PYEOF

# Regenerate combined summary CSV + plot from the merged raw CSV
python run_butterworth_mc.py \
    --from-csv "${OUTPUT_DIR}/mc_raw.csv" \
    --output-dir "${OUTPUT_DIR}"

# Print the summary table from the merged CSV
python - <<'PYEOF'
import csv, os, sys
import numpy as np
from pathlib import Path
from collections import defaultdict

output_dir = Path(os.environ.get("OUTPUT_DIR", "Results_butterworth_mc"))
subjects   = [int(s) for s in os.environ.get("SUBJECTS", "1 2 3 4 5 6 7 8 9").split()]

raw_path = output_dir / "mc_raw.csv"
if not raw_path.exists():
    print("mc_raw.csv not found", file=sys.stderr); sys.exit(1)

records = []
with open(raw_path, newline="") as f:
    records = list(csv.DictReader(f))

sigmas = sorted({float(r["sigma"]) for r in records})
print()
print("  Butterworth Monte Carlo — Mean Accuracy Drop (pp)")
sep = "-" * (8 + 14 * len(sigmas) + 12)
print(sep)
hdr = f"  {'Subj':<6}" + "".join(f"  {'σ='+str(int(s*100))+'%':>12}" for s in sigmas) + f"  {'Baseline':>9}"
print(hdr)
print(sep)

all_drops = defaultdict(list)
for subj in subjects:
    srecs = [r for r in records if int(r["subject"]) == subj]
    if not srecs:
        print(f"  S{subj:<5}  (no data)")
        continue
    bl = np.mean([float(r["baseline_acc"]) for r in srecs]) * 100
    row = f"  S{subj:<5}"
    for sigma in sigmas:
        drops = [float(r["acc_drop"]) * 100 for r in srecs if float(r["sigma"]) == sigma]
        if drops:
            row += f"  {np.mean(drops):>+6.2f}±{np.std(drops):.2f}"
            all_drops[sigma].extend(drops)
        else:
            row += f"  {'N/A':>12}"
    row += f"  {bl:>8.1f}%"
    print(row)

print(sep)
grand = f"  {'MEAN':<6}"
for sigma in sigmas:
    d = all_drops[sigma]
    grand += f"  {np.mean(d):>+6.2f}±{np.std(d):.2f}" if d else f"  {'N/A':>12}"
grand += f"  {np.mean([float(r['baseline_acc'])*100 for r in records]):>8.1f}%"
print(grand)
print(sep)
print()
for sigma in sigmas:
    d = all_drops[sigma]
    if d:
        p95 = np.percentile(d, 95)
        verdict = "PASS" if p95 < 1.0 else "MARGINAL" if p95 < 2.0 else "FAIL"
        print(f"  σ={sigma*100:.0f}%  mean: {np.mean(d):+.3f}pp  p95: {p95:.3f}pp  → {verdict}")
print()
PYEOF

echo "End: $(date)"
