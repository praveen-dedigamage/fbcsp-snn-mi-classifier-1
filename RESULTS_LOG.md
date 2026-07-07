# Results Log — Puhti Submission Cycles

Tracks the "submit → wait → copy" cycle for every experiment implemented on
2026-07-06 (see `PIPELINE_REFERENCE.md` §10 for what was built, `TODO.md`
B2/B11/B15/Tier-0 in the paper folder for why). Paste Puhti output (terminal
summaries, `summary.csv` contents, `reliability_results.json` contents) into
the matching section below as each run completes. Keep unresolved runs at
the top of their section; move completed ones to "Done" once numbers are
final and copied into the paper.

---

## Decided run order (2026-07-07)

1. **Verification retrain — `Results_verify` (new dir, do NOT overwrite `Results/`).**
   `bash submit_puhti.sh Results_verify` — no ablation flags, default full
   pipeline. Purpose: confirm the new codebase reproduces the original
   paper-producing numbers before trusting anything built on top of it.
   Original `Results/` must stay untouched as the reference — retraining
   into the same paths would silently overwrite the checkpoints being
   verified against, with no way back.
2. **Reliability sweep** — `sbatch run_puhti_reliability.sh`, once step 1
   confirms no regression (or immediately against untouched `Results/` if
   those checkpoints are still there — the sweep needs no retraining either
   way, it's pure inference on a saved checkpoint).
3. **B2 — Cho2017 + BNCI2015-001** (fills empty Table V).
4. **B11 — the three ablations** (essential evidence for the abstract's
   three claimed contributions).

### What "verification" actually checks — isolated vs. joint vs. new fields

Not everything in `summary.csv` is checked the same way:

- **Isolated fields** (`test_acc_fp32`, `test_acc_int8`,
  `test_acc_csp_{8,6,4}bit`) → **regression check.** These are the only
  fields with a "before" to compare against (they're exactly what's already
  in the original `Results/` and the paper's Tables IV/VI). Should match.
  Kept in the code only for this comparison and because `analyze_results.py`
  reads these exact keys — **not** meant to be the primary hardware-realism
  evidence going forward. That's what the joint sweep is for (B15 Tier D:
  it's supposed to *replace* these scattered numbers in the paper, not sit
  alongside them as equally-weighted results).
- **Joint fields** (`test_acc_joint_csp{X}_snn{Y}`) → **new evidence, no
  prior number to check.** First time this grid has ever been computed —
  nothing tested "CSP=8bit and SNN=6bit simultaneously" before this session.
- **`mean_events_per_trial`** → also brand new, no prior baseline. This is
  the number that should replace B8's unmeasured "0.15 spikes/neuron/
  timestep" assumption once it comes back from Puhti.
- **`val_acc_lda`/`test_acc_lda`/`val_acc_svm`/`test_acc_svm`** → **expected
  to differ from the original run, on purpose.** Tier 0 tuning intentionally
  changed how these are computed (LDA shrinkage, SVM grid search) — a
  mismatch here is the tuning working, not a regression.

---

## Scope gap in the reliability sweep — CLOSED 2026-07-07

Originally flagged: Tier B noise-tested CSP weights, SNN weights, and LIF
beta only — a reasonable minimum-viable set, but not sufficient to fully
back a blanket "this pipeline is analog-circuit-realizable" claim. **All
five gaps below are now implemented and wired into `run_reliability()`.**

1. **Filter bank coefficients (Butterworth/Bessel `sos` values) — DONE.**
   New `bandpass_filter_noisy`/`apply_filter_bank_noisy` in
   `preprocessing.py`, wired as the `filter_bank_noise` sweep. This was the
   most important gap: the paper's hardware-realizability argument
   specifically leans on the bandpass filter being Gm-C-circuit realizable
   (§2a), and that claim had never been stress-tested against actual analog
   component-value variation until now.
   - **Real bug caught and fixed during implementation:** naively adding
     noise to the full `sos` array breaks scipy's requirement that column 3
     (the `a0` coefficient) stay exactly `1.0` — `sosfilt` rejects the array
     otherwise. Fixed by only perturbing the other 5 columns per section
     (also the physically correct choice: `a0=1` is a normalisation
     convention, not itself a component value). Verified with both
     Butterworth and Bessel filters before considering this done.
2. **Spike encoder's `adapt_inc`/`decay` — DONE.** Generalized the noisy
   encoder kernel (`_adaptive_threshold_encode_noisy_jit`) to accept
   per-feature `adapt_inc`/`decay` tensors, not just per-feature threshold.
   Wired as two sweeps: `encoder_threshold_noise` (comparator offset —
   this was actually built in the original Tier B pass but never wired into
   a runnable sweep, caught and fixed now too) and `encoder_adaptation_noise`
   (adapt_inc + decay together).
3. **Z-normalisation mean/std — DONE.** New `inject_znorm_noise` in
   `quantization.py`, wired as the `znorm_noise` sweep.
4. **Euclidean Alignment whitener matrix — DONE.** New
   `inject_ea_whitener_noise` in `quantization.py` (mirrors
   `inject_csp_filter_noise`'s structure exactly, same dict-of-matrices
   shape), wired as the `ea_whitener_noise` sweep. Degrades gracefully to a
   no-op if a fold's CSP was fit with `euclidean_alignment=False`.
5. **SNN biases — DONE.** `inject_model_weight_noise` now perturbs
   `Linear.bias` by default (`include_bias=True`), alongside weights, in the
   same `snn_weight_noise` sweep.
6. **Still correctly out of scope, no gap:** MIBIF feature selection
   (routing/wire selection, not a continuous analog coefficient) and Van
   Rossum loss (training-only, no inference-time circuit).

**New experiment added on top of closing the gaps:** `joint_noise_all_sources`
— every one of the 7 individual noise sources perturbed simultaneously at
matched severity, since a real chip has every stage imperfect at once and
testing sources in isolation doesn't represent the actual deployment
scenario (same principle as `pipeline.py`'s joint CSP+SNN quantisation grid,
extended here to the full noise picture).

`run_reliability()` now runs **9 sweeps total**: `csp_weight_noise`,
`snn_weight_noise`, `beta_noise`, `filter_bank_noise`, `ea_whitener_noise`,
`znorm_noise`, `encoder_threshold_noise`, `encoder_adaptation_noise`,
`joint_noise_all_sources`. Full technical detail: `PIPELINE_REFERENCE.md`
§11.

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
