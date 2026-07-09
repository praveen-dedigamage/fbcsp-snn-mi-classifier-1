# Results Log — Puhti Submission Cycles

Tracks the "submit → wait → copy" cycle for every experiment implemented on
2026-07-06 (see `PIPELINE_REFERENCE.md` §10 for what was built, `TODO.md`
B2/B11/B15/Tier-0 in the paper folder for why). Paste Puhti output (terminal
summaries, `summary.csv` contents, `reliability_results.json` contents) into
the matching section below as each run completes. Keep unresolved runs at
the top of their section; move completed ones to "Done" once numbers are
final and copied into the paper.

---

## Verification retrain — COMPLETE 2026-07-08 (all 9 subjects)

`bash submit_puhti.sh Results_verify` — 9 subjects × 5 folds, code at commit
`0ca8101` (fully pulled and confirmed before submission).

- Train job: `35390599` (45 tasks)
- Aggregate jobs (one per subject, each depends only on its own 5 fold tasks):
  `35390600`–`35390608` (subjects 1–9 respectively)
- Analyze job: `35390609` (depends on all 9 aggregates) — final summary lands
  in `logs/fbcsp_analyze_35390609.out`

**What to check once complete:**
1. `Results_verify/Subject_N/summary.csv`'s `test_acc_fp32`/`test_acc_int8`/
   `test_acc_csp_{8,6,4}bit` vs. the original `Results/` — should match
   (regression check, see "What verification actually checks" section below).
2. `val_acc_lda`/`svm` and `test_acc_lda`/`svm` — expected to differ on
   purpose (Tier 0 tuning).
3. New fields now present: `mean_events_per_trial`,
   `mean_input_events_per_trial`, `mean_hidden_events_per_trial`,
   `n_timesteps` — no "before" to compare, this is new evidence.
4. New joint-PTQ fields: `test_acc_joint_csp{8,6,4}_snn{8,6,4}` (9 combos) —
   also new evidence.

**Outcome, not what was expected going in**: this was NOT a same-pipeline
regression check. The original `Results/` was generated with a 12-band
adaptive Fisher-selection front end (`band_selection.py`), which this
session's own earlier commit `a9ceddf` had already deleted in favour of the
fixed six-band scheme, without saying so in the commit message. Full
diagnostic trail and corrected significance-test results (SNN now
*significantly* beats SVM, no longer significantly beats LDA — a flip from
before) are in `puhti_logs/Results_verify/` (`BAND_SELECTION_FINDING.md`,
`ANALYSIS.md`, `SIGNIFICANCE_TEST_RERUN.md`). `main.tex`, `TODO.md`, and
`PAPER_REWRITE_NOTES.md` (paper folder) have all been updated with the real
numbers. Schirrmeister2017 needs the same retrain — not done yet.

Unblocked now: `sbatch run_puhti_reliability.sh` (needs these checkpoints)
and `python compute_energy.py --results-dir Results_verify ...` (needs the
event-count fields, confirmed present in `Results_verify/Subject_1/
summary.csv`).

---

## Reliability sweep — FIRST ATTEMPT FAILED 2026-07-08 (2 bugs, both fixed), needs resubmit

`sbatch run_puhti_reliability.sh` was run without setting `RESULTS_DIR`
first. Two problems surfaced:

1. **Wrong directory, not a code bug.** The script defaults to
   `RESULTS_DIR="${RESULTS_DIR:-Results}"` — plain `Results`, not
   `Results_verify`. The job's own startup banner confirmed `Results:
   Results`. Worse: `Results/Subject_1/fold_0`'s log showed `Filter bank: 3
   bands` — a *third*, even older pipeline generation (the original
   3-non-overlapping-band static scheme), different from both the 12-band
   adaptive pipeline traced in `BAND_SELECTION_FINDING.md` (found via
   Subject 8) and the current fixed six-band pipeline. `Results/` is an
   inconsistent grab-bag across subjects from different points in the
   project's history, not a single coherent snapshot — never use it as a
   reference again; always pass `RESULTS_DIR=Results_verify` explicitly.

2. **Real bug, now fixed**: `inject_beta_noise` (`fbcsp_snn/quantization.py`)
   crashed every task on the `beta_noise` sweep with `RuntimeError: Expected
   all tensors to be on the same device, but found at least two devices,
   cuda:0 and cpu!`. Root cause: `torch.Generator()` (no `device=` arg) and
   `torch.randn(n_neurons, generator=gen)` (no `device=` arg) always
   allocate on CPU, while `lif.beta` lives on the model's device (`cuda:0`
   on Puhti). This function was written without local CUDA access
   (documented in its own docstring as unverified) — first real GPU run
   caught it immediately. Every sibling noise function was audited for the
   same risk: `inject_csp_filter_noise`/`inject_ea_whitener_noise`/
   `inject_znorm_noise` are pure numpy (no device concept, safe);
   `inject_model_weight_noise` and the encoder threshold/adaptation noise
   functions already correctly derive `device=` from the actual tensor at
   every call site (established pattern in `inject_weight_noise_tensor`).
   `inject_beta_noise` was the only offender. Fixed by generating the noise
   tensor on CPU (unchanged RNG semantics/seeding) then moving it to
   `nominal.device` before combining.

**Next**: commit + push the fix, `git pull` on Puhti, then resubmit with
the directory set explicitly this time:
```bash
RESULTS_DIR=Results_verify sbatch run_puhti_reliability.sh
```

---

## Energy computation — extended, not replaced (2026-07-08)

Discovered mid-session that Puhti already had a real, working
`compute_energy.py` + `run_lava_infer.py` (never git-tracked, so invisible
to this repo) — considerably more developed than assumed: uses **real
measured Lava SynOps** (not an assumption), has **real cited references**
for an analog front-end (Qian 2017, Verhoeven 2007, Sharifshazileh 2021,
Burr 2017), and a **real cited GPU/EEGNet-M4 comparison** (Burrello et al.
2020, 4.28 mJ measured). Decision: extend it, don't replace it.

**Important clarification from the user:** Loihi 2 is a **digital**,
asynchronous, event-driven chip — not analog. The existing script's Gm-C/
ADM/ReRAM front-end estimates describe a *separate*, more speculative
hardware story (a hypothetical all-analog front-end that could pair with a
digital Loihi backend), not a claim that Loihi itself is analog. Now stated
explicitly in the script's docstring and print output so this doesn't get
conflated later.

**What got added:** a cross-check section using B15's `mean_input_events_
per_trial`/`mean_hidden_events_per_trial` (this codebase's own PyTorch-side
spike instrumentation) to compute an *independent* fan-out-weighted SynOps
estimate, reported side-by-side against Lava's measured figure per subject.
Rationale: Lava's number stays primary (it's the real target framework's
own accounting for real, currently-deployable hardware — stronger than
anything derived from PyTorch instrumentation alone), but if the two
independent measurements roughly agree, that's corroborating evidence
before either number is trusted in the paper; if they diverge, that's worth
investigating first. Verified the cross-check's core arithmetic
(`input_events × n_hidden + hidden_events × n_output`) and its graceful
skip-on-missing-data behavior with mock `pipeline_params.json` fixtures
before considering this done.

**Status:** user ran `run_lava_infer.py` and has a `Results_lava/
lava_summary.csv`, but hasn't validated whether the numbers are correct.
Next step once training (step 1 in the run order) produces fresh
`pipeline_params.json` files with the B15 event breakdown: run
```bash
python compute_energy.py --lava-dir Results_lava --results-dir Results_verify \
    --subjects 1 2 3 4 5 6 7 8 9 --n-folds 5
```
and check the cross-check table for agreement before trusting either the
Lava-measured or PyTorch-derived SynOps figure in the paper.

**Also decided:** start fresh on Puhti — dozens of old `Results_*`
directories and loose scripts (`run_butterworth_mc.py`, `run_e2e_stress.py`,
`run_lava_infer.py`, `save_test_spikes.py`, `show_csp_quantized.py`, and the
old untracked `compute_energy.py`) get archived (moved, not deleted) by the
user before the next pull — the new tracked `compute_energy.py` would
otherwise conflict with the untracked one of the same name on `git pull`.

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
