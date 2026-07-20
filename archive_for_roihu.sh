#!/bin/bash
#SBATCH --job-name=archive_for_roihu
#SBATCH --account=project_2003397
#SBATCH --partition=small
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=logs/archive_%j.out
#SBATCH --error=logs/archive_%j.err

# ============================================================
# Archive the project directory + MOABB/MNE data cache for
# migration Puhti -> Roihu, per Tuomo's guidance (2026-07-20):
# do this on a compute node via sbatch, not at the login node,
# since login-node I/O is currently slow.
#
# Produces two archives in PROJECT_DIR, ready to rsync/scp to
# Roihu's /scratch/project_2003397/ from a LOGIN node (compute
# nodes typically can't reach another cluster's network directly,
# so the actual cross-cluster transfer step still needs a login
# node -- but that's just moving 1-2 large files, not millions of
# small ones, so it should be tolerable even at reduced I/O).
#
# Submit: sbatch archive_for_roihu.sh
# ============================================================

set -uo pipefail
# Note: NOT using `set -e` here. GNU tar exits 1 (not 0) when a file
# changed while being archived -- a warning, not a fatal error -- which
# is expected if other jobs (e.g. an in-progress training run) are
# actively writing into Results_*/ at the same time. `set -e` would abort
# the whole script on that harmless warning, silently skipping every step
# after it (found 2026-07-20: the mne_data archive never ran because of
# exactly this, while the BNCI2015-001 training job was still writing new
# results concurrently). Each tar call below checks its own exit code
# explicitly instead, tolerating 1 but still failing hard on real errors.

PROJECT_DIR=/scratch/project_2003397/praveen
cd "${PROJECT_DIR}"

# Runs `tar "$@"`, tolerating exit code 1 (non-fatal "file changed as we
# read it" warning) but aborting the script on any other non-zero code.
run_tar() {
    tar "$@"
    local ec=$?
    if [ "${ec}" -gt 1 ]; then
        echo "FATAL: tar exited ${ec} (a real error, not just a changed-file warning)" >&2
        exit "${ec}"
    elif [ "${ec}" -eq 1 ]; then
        echo "NOTE: tar reported changed files during archiving (exit 1) -- likely another job writing concurrently. Continuing; consider re-running once no jobs are actively writing, for a fully consistent snapshot."
    fi
}

echo "=============================================="
echo "  Archive for Roihu migration"
echo "  Start: $(date)"
echo "  Node:  $(hostname)"
echo "=============================================="

echo ""
echo "--- Current sizes (for the record) ---"
du -sh fbcsp-snn-mi-classifier-1/ 2>/dev/null || echo "  (repo dir not found at expected path)"
du -sh fbcsp-snn-mi-classifier-1/.venv/ 2>/dev/null || echo "  (.venv not found)"
du -sh mne_data/ 2>/dev/null || echo "  (mne_data not found)"
echo ""

# ---- Archive 1: the repo (code + every Results_* directory) ----
# Excludes .venv -- it's built for Puhti's x86 CPUs; Roihu-GPU is ARM
# (Grace Hopper), so this venv cannot run there and must be rebuilt
# fresh instead of migrated.
echo "--- Archiving repo (excluding .venv) ---"
run_tar --exclude='fbcsp-snn-mi-classifier-1/.venv' \
    -czf "fbcsp-snn-mi-classifier-1_$(date +%Y%m%d).tar.gz" \
    fbcsp-snn-mi-classifier-1/
echo "Repo archive done: $(date)"
ls -lh fbcsp-snn-mi-classifier-1_*.tar.gz

# ---- Archive 2: MOABB/MNE data cache ----
# Avoids re-downloading BNCI2014-001, Schirrmeister2017, Cho2017, and
# BNCI2015-001 on Roihu.
if [ -d "mne_data" ]; then
    echo ""
    echo "--- Archiving mne_data cache ---"
    run_tar -czf "mne_data_$(date +%Y%m%d).tar.gz" mne_data/
    echo "mne_data archive done: $(date)"
    ls -lh mne_data_*.tar.gz
else
    echo ""
    echo "No mne_data/ directory found at ${PROJECT_DIR} -- skipping."
    echo "(check MNE_DATA env var used during training if this is unexpected)"
fi

echo ""
echo "=============================================="
echo "  All archiving complete: $(date)"
echo "  Next: from a LOGIN node, rsync/scp the .tar.gz files to"
echo "  Roihu's /scratch/project_2003397/, then unpack there."
echo "=============================================="
