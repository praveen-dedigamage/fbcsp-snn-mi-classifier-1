#!/bin/bash
#SBATCH --job-name=fbcsp_e2e_analyze
#SBATCH --account=project_2003397
#SBATCH --partition=small
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=logs/fbcsp_e2e_analyze_%j.out
#SBATCH --error=logs/fbcsp_e2e_analyze_%j.err

# ============================================================
# FBCSP-SNN — E2E stress test cross-subject analysis
# Runs after all 9 E2E array tasks complete (via submit_e2e.sh).
# Merges per-subject CSVs and regenerates the combined table + plot.
# ============================================================

set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-Results_e2e_stress}"
SUBJECTS="${SUBJECTS:-1 2 3 4 5 6 7 8 9}"
SIGMA="${SIGMA:-0.02}"
CSP_BITS="${CSP_BITS:-4}"
SNN_BITS="${SNN_BITS:-8}"

PROJECT_DIR=/scratch/project_2003397/praveen/fbcsp-snn-mi-classifier-1

echo "=============================================="
echo "  FBCSP-SNN E2E Stress Test Analysis"
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

# Merge per-subject e2e_raw_S{N}.csv files
python - <<'PYEOF'
import csv, sys, os
from pathlib import Path

output_dir = Path(os.environ.get("OUTPUT_DIR", "Results_e2e_stress"))
subjects   = [int(s) for s in os.environ.get("SUBJECTS", "1 2 3 4 5 6 7 8 9").split()]

all_rows = []
fieldnames = None
missing = []
for s in subjects:
    p = output_dir / f"e2e_raw_S{s}.csv"
    if not p.exists():
        print(f"  ERROR: e2e_raw_S{s}.csv not found — job may have failed or timed out",
              file=sys.stderr)
        missing.append(s)
        continue
    with open(p, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = [r for r in reader if int(r["subject"]) == s]
        if not rows:
            print(f"  ERROR: e2e_raw_S{s}.csv exists but contains no rows for S{s}",
                  file=sys.stderr)
            missing.append(s)
            continue
        all_rows.extend(rows)
        print(f"  S{s}: {len(rows)} rows loaded")

if missing:
    print(f"  WARNING: {len(missing)} subject(s) missing: {missing}", file=sys.stderr)
    print(f"  Proceeding with {len(subjects) - len(missing)}/{len(subjects)} subjects.",
          file=sys.stderr)

if not all_rows:
    print("No E2E data found.", file=sys.stderr)
    sys.exit(1)

merged = output_dir / "e2e_raw.csv"
output_dir.mkdir(parents=True, exist_ok=True)
with open(merged, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(all_rows)
print(f"Merged {len(all_rows)} rows → {merged}")
PYEOF

# Regenerate summary CSV + plot + print table
python run_e2e_stress.py \
    --from-csv "${OUTPUT_DIR}/e2e_raw.csv" \
    --sigma    "${SIGMA}" \
    --csp-bits "${CSP_BITS}" \
    --snn-bits "${SNN_BITS}" \
    --output-dir "${OUTPUT_DIR}"

echo "End: $(date)"
