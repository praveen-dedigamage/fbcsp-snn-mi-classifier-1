#!/bin/bash
# ============================================================================
#  Submit a whole dataset's experiment chain in one shot
#
#      DATASET=Cho2017 N_SUBJECTS=52 bash roihu/run_chain.sh
#      DATASET=BNCI2014_002 N_SUBJECTS=14 bash roihu/run_chain.sh
#      N_SUBJECTS=3 ... bash roihu/run_chain.sh    # pilot first
#      DRYRUN=1 ...   bash roihu/run_chain.sh      # print the plan only
#
#  This is a submission script, not a batch job: it runs on the login node and
#  issues sbatch calls wired together with Slurm dependencies, so each stage
#  starts only when its prerequisite has succeeded. Nothing needs babysitting
#  between stages.
#
#  Chain
#  -----
#      prepare data  (cache 52 subjects; everything else reads it)
#           |  afterok
#      training array (one task per subject, 5 folds concurrent per GPU)
#           |  afterok  -- fans out, these four are independent
#           +-- quantisation sweep, split front end
#           +-- quantisation sweep, fused front end
#           +-- fusion experiment (split vs fused, per bit-width)
#           +-- fairness-controlled baseline (array, one task per subject)
#
#  The four downstream jobs run concurrently by design. They read the fold
#  artifacts read-only and write to disjoint outputs: the sweeps differ by a
#  '_fused' filename suffix, the fusion experiment writes Results_fusion, and
#  the fair baseline writes a fair_baseline_results.json that nothing else
#  touches. Running them in sequence would cost the same billing units and
#  finish a day later.
#
#  BEFORE RUNNING THE FULL 52
#  --------------------------
#  Cho2017 is 52 subjects at 64 channels against BNCI2015-001's 12 at 13.
#  The cost is estimated, not measured. Run `N_SUBJECTS=3` first, then
#  `seff <trainjob>_1`, and multiply: if a subject costs ~250 BU the full run
#  is ~13k, if it costs ~600 it is ~31k, and that difference should change the
#  decision rather than be discovered halfway through.
#
#  Requires SBATCH_ACCOUNT in the environment, or --account left in the job
#  scripts.
# ============================================================================

set -euo pipefail
cd "$(dirname "$0")/.."

DATASET="${DATASET:-Cho2017}"
N_SUBJECTS="${N_SUBJECTS:-52}"
ARRAY="1-${N_SUBJECTS}"
SUBJECTS="$(seq 1 "${N_SUBJECTS}" | tr '\n' ' ')"

# Wall times. Generous rather than tight: a job killed at the limit loses
# everything, because every stage writes its output only on completion.
T_PREP="${T_PREP:-03:00:00}"
T_TRAIN="${T_TRAIN:-12:00:00}"
T_SWEEP="${T_SWEEP:-24:00:00}"
T_FUSION="${T_FUSION:-12:00:00}"
T_FAIR="${T_FAIR:-06:00:00}"

DRYRUN="${DRYRUN:-0}"

mkdir -p logs

echo "============================================================"
echo "  ${DATASET} experiment chain"
echo "    subjects   : ${N_SUBJECTS}  (array ${ARRAY})"
echo "    account    : ${SBATCH_ACCOUNT:-<from script #SBATCH lines>}"
echo "    wall times : prep ${T_PREP}  train ${T_TRAIN}  sweep ${T_SWEEP}"
echo "                 fusion ${T_FUSION}  fair ${T_FAIR}"
if [ "${N_SUBJECTS}" -eq 52 ]; then
echo "    NOTE: full run. Estimated ~13k BU, unmeasured. Consider"
echo "          N_SUBJECTS=3 first to turn that into a number."
fi
echo "============================================================"

submit() {
    # submit <description> <sbatch args...>  -> echoes the job id
    #
    # Aborts the whole chain on failure. Without this an sbatch error yields
    # an empty job id, every dependent submission then fails with "Job
    # dependency problem", and the summary prints six blank ids as though
    # something had been queued.
    local desc="$1"; shift
    if [ "${DRYRUN}" = "1" ]; then
        echo "DRYRUN  ${desc}:  sbatch $*" >&2
        echo "000000"
        return
    fi
    local jid
    if ! jid="$(sbatch --parsable "$@")" || ! [[ "${jid}" =~ ^[0-9]+$ ]]; then
        echo "" >&2
        echo "!! ${desc} failed to submit; chain aborted, nothing queued." >&2
        echo "!! sbatch args were: $*" >&2
        exit 1
    fi
    echo "submitted  ${desc}  -> ${jid}" >&2
    echo "${jid}"
}

# ---- 1. data preparation ---------------------------------------------------
# Must finish before anything reads the cache. Doing this as its own job, not
# alongside training, is deliberate: a half-populated cache fails at a subject
# boundary hours later and reads as an unrelated error.
PREP=$(DATASET="${DATASET}" N_SUBJECTS="${N_SUBJECTS}" \
       submit "prepare-data" --time="${T_PREP}" roihu/00b_prepare_data.sh)

# ---- 2. training -----------------------------------------------------------
TRAIN=$(DATASET="${DATASET}" \
        submit "train" --dependency=afterok:"${PREP}" \
               --array="${ARRAY}" --time="${T_TRAIN}" roihu/01_train_array.sh)

# ---- 3. downstream, all gated on the whole training array succeeding --------
SWEEP=$(DATASET="${DATASET}" SUBJECTS="${SUBJECTS}" \
        submit "sweep-split" --dependency=afterok:"${TRAIN}" \
               --time="${T_SWEEP}" roihu/02_quant_sweep.sh)

SWEEPF=$(DATASET="${DATASET}" SUBJECTS="${SUBJECTS}" FUSE=1 \
         submit "sweep-fused" --dependency=afterok:"${TRAIN}" \
                --time="${T_SWEEP}" roihu/02_quant_sweep.sh)

FUSION=$(DATASET="${DATASET}" SUBJECTS="${SUBJECTS}" \
         submit "fusion" --dependency=afterok:"${TRAIN}" \
                --time="${T_FUSION}" roihu/03_fusion.sh)

FAIR=$(DATASET="${DATASET}" \
       submit "fair-baseline" --dependency=afterok:"${TRAIN}" \
              --array="${ARRAY}" --time="${T_FAIR}" roihu/04_fair_baseline.sh)

echo
echo "============================================================"
echo "  prepare-data   ${PREP}"
echo "  train          ${TRAIN}   (after prep)"
echo "  sweep-split    ${SWEEP}   (after train)"
echo "  sweep-fused    ${SWEEPF}   (after train)"
echo "  fusion         ${FUSION}   (after train)"
echo "  fair-baseline  ${FAIR}   (after train)"
echo "============================================================"
echo
echo "Monitor:"
echo "  squeue --me"
echo "  seff ${TRAIN}_1        # cost per subject, once training starts finishing"
echo
echo "When everything has finished, collect on the login node:"
echo "  module load python-pytorch/2.10"
echo "  python collect_results.py --dataset ${DATASET} --n-folds 5 \\"
echo "      --expect-subjects ${N_SUBJECTS} --results-dir Results_cho2017"
echo "  python compute_energy.py --from-artifacts --n-channels 64 \\"
echo "      --results-dir Results_cho2017 --subjects ${SUBJECTS}\\"
echo "      --n-folds 5 --output-dir Results_energy_cho2017"
echo "  python aggregate_fair_baseline.py --results-dir Results_cho2017 \\"
echo "      --subjects ${SUBJECTS}--n-folds 5"
echo "  python plot_pergroup.py --quant-dir Results_quant --dataset ${DATASET}"
echo
echo "Cancel the whole chain:  scancel ${PREP} ${TRAIN} ${SWEEP} ${SWEEPF} ${FUSION} ${FAIR}"
