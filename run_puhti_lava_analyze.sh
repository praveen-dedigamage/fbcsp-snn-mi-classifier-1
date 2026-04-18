#!/bin/bash
#SBATCH --job-name=fbcsp_lava_analyze
#SBATCH --account=project_2003397
#SBATCH --partition=small
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=logs/fbcsp_lava_analyze_%j.out
#SBATCH --error=logs/fbcsp_lava_analyze_%j.err

# ============================================================
# Merge per-subject lava_raw_S{N}.csv and regenerate table.
# Runs after all 9 Lava array tasks complete (via submit_lava.sh).
# ============================================================

set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-Results_lava}"
SUBJECTS="${SUBJECTS:-1 2 3 4 5 6 7 8 9}"

PROJECT_DIR=/scratch/project_2003397/praveen/fbcsp-snn-mi-classifier-1
VENV_LAVA="/scratch/project_2003397/praveen/.venv_lava"

echo "=============================================="
echo "  FBCSP-SNN Lava Analysis"
echo "  OUTPUT_DIR: ${OUTPUT_DIR}"
echo "  SUBJECTS:   ${SUBJECTS}"
echo "  Node:       $(hostname)"
echo "  Start:      $(date)"
echo "=============================================="

unset SINGULARITY_BIND
unset APPTAINER_BIND
cd "${PROJECT_DIR}"

source "${VENV_LAVA}/bin/activate"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

# Merge per-subject CSVs
python - <<'PYEOF'
import csv, sys, os
from pathlib import Path

output_dir = Path(os.environ.get("OUTPUT_DIR", "Results_lava"))
subjects   = [int(s) for s in os.environ.get("SUBJECTS", "1 2 3 4 5 6 7 8 9").split()]

all_rows = []
fieldnames = None
missing = []
for s in subjects:
    p = output_dir / f"lava_raw_S{s}.csv"
    if not p.exists():
        print(f"  ERROR: lava_raw_S{s}.csv not found", file=sys.stderr)
        missing.append(s)
        continue
    with open(p, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = [r for r in reader if int(r["subject"]) == s]
        if not rows:
            print(f"  ERROR: lava_raw_S{s}.csv empty", file=sys.stderr)
            missing.append(s)
            continue
        all_rows.extend(rows)
        print(f"  S{s}: {len(rows)} rows loaded")

if missing:
    print(f"  WARNING: missing subjects: {missing}", file=sys.stderr)
if not all_rows:
    print("No Lava data found.", file=sys.stderr); sys.exit(1)

merged = output_dir / "lava_raw.csv"
output_dir.mkdir(parents=True, exist_ok=True)
with open(merged, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(all_rows)
print(f"Merged {len(all_rows)} rows → {merged}")
PYEOF

# Regenerate summary + print table
python run_lava_infer.py \
    --from-csv "${OUTPUT_DIR}/lava_raw.csv" \
    --output-dir "${OUTPUT_DIR}"

deactivate
echo "End: $(date)"
