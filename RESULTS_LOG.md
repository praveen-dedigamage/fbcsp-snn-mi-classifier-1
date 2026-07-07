# Results Log — Puhti Submission Cycles

Tracks the "submit → wait → copy" cycle for every experiment implemented on
2026-07-06 (see `PIPELINE_REFERENCE.md` §10 for what was built, `TODO.md`
B2/B11/B15/Tier-0 in the paper folder for why). Paste Puhti output (terminal
summaries, `summary.csv` contents, `reliability_results.json` contents) into
the matching section below as each run completes. Keep unresolved runs at
the top of their section; move completed ones to "Done" once numbers are
final and copied into the paper.

---

## How to submit each experiment

```bash
cd /scratch/project_2003397/praveen/fbcsp-snn-mi-classifier-1
git pull                                   # pick up this session's changes
mkdir -p logs

# B2 — missing binary-classification datasets (Table V)
bash submit_cho2017.sh                     # -> Results_cho2017/
bash submit_bnci2015.sh                    # -> Results_bnci2015/  (script already existed)

# B11 — ablation study (one component at a time vs. full pipeline)
bash submit_ablation_single_end_csp.sh     # -> Results_ablation_single_end_csp/
bash submit_ablation_fixed_encoder.sh      # -> Results_ablation_fixed_encoder/
bash submit_ablation_cross_entropy.sh      # -> Results_ablation_cross_entropy/

# B15 — reliability / hardware-noise sweep (run AFTER the corresponding
# training + aggregate jobs have completed — needs saved checkpoints)
sbatch run_puhti_reliability.sh            # -> Results/Subject_*/fold_*/reliability_results.json

# Baseline Tier 0 tuning is automatic — already baked into every training
# run above (baseline.py), no separate submission needed. Just re-run any
# training experiment and the tuned SVM/LDA numbers come along with it.
```

Monitor with `squeue -u $USER`; per-subject summaries land in
`Results_*/Subject_N/summary.csv`, final cross-subject analysis in
`logs/fbcsp_analyze_<JOBID>.out`.

---

## 1. B2 — Missing binary-classification results (Table V)

**Status:** not yet submitted.

### Cho2017 (2-class, 52 subjects, 64ch, 512Hz)
*(paste `summary.csv` / analyze output here once complete)*

### BNCI2015-001 (2-class, 12 subjects, 13ch, 512Hz)
*(paste `summary.csv` / analyze output here once complete)*

---

## 2. B11 — Ablation study

**Status:** not yet submitted. Compare each against the full-pipeline
baseline (BNCI2014-001, `Results/`) already in the paper's Table III.

### Single-end CSP (`--csp-single-end`)
*(paste `summary.csv` here — compare mean accuracy vs. dual-end baseline)*

### Fixed-threshold encoder (`--encoder-type fixed`)
*(paste `summary.csv` here — compare vs. adaptive-threshold baseline)*

### Cross-entropy loss (`--loss-type cross_entropy`)
*(paste `summary.csv` here — compare vs. Van Rossum baseline)*

---

## 3. B15 — Reliability / hardware-realism sweep

**Status:** not yet submitted. Requires training to have completed first
(reads saved `best_model.pt`/`pipeline_params.json` per fold).

### Joint CSP+SNN quantisation grid
*(this is saved automatically during training as
`test_acc_joint_csp{8,6,4}_snn{8,6,4}` in `pipeline_params.json`/
`summary.csv` — no separate run needed; paste the grid here once training
completes)*

### Noise-robustness sweep (`reliability_results.json` per fold)
*(paste per-fold JSON or a summarised table here: severity → acc_mean ±
acc_std, events_mean ± events_std, for each of csp_weight_noise,
snn_weight_noise, beta_noise)*

### Measured spike-events-per-trial (fixes B8's energy estimate)
*(paste `mean_events_per_trial` from `summary.csv` here — this replaces the
paper's unmeasured "0.15 spikes/neuron/timestep" assumption)*

---

## 4. Baseline Tier 0 — Tuned SVM/LDA re-validation

**Status:** not yet re-run with tuning. Automatic on next training run of
any experiment (no separate submission).

### Re-run B6 significance tests against tuned baseline
*(once a fresh BNCI2014-001 + Schirrmeister2017 training run with tuned
baselines completes, re-run `sig_test.py`-style paired tests — see
`TODO.md` B6 in the paper folder for the original methodology — and paste
updated p-values here)*

---

## Done

*(move completed, paper-ready results here once copied into `main.tex`)*
