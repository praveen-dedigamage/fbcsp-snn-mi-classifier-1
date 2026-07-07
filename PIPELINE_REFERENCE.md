# Pipeline Reference — Verified Against Code

Everything below was traced directly through `fbcsp_snn/pipeline.py` and
`fbcsp_snn/datasets.py` (this worktree, branch `claude/hungry-neumann`), not
assumed from the paper or CLAUDE.md. Line numbers refer to those files as of
2026-07-06. Re-verify if the code changes materially.

---

## 1. Where the raw EEG actually comes from

- `cfg.source` defaults to `"moabb"` (`config.py`). The alternative, `"hdf5"`,
  is a legacy path for loading old `.mat` files directly — not used for any
  dataset reported in the paper.
- Call chain: `_load_raw(cfg)` (`pipeline.py:77-90`) →
  `load_moabb(cfg.moabb_dataset, cfg.subject_id, cfg.n_classes)`
  (`datasets.py:118`) → `paradigm.get_data(dataset=dataset, subjects=[subject_id])`
  (`datasets.py:236`), where `paradigm` is MOABB's `MotorImagery` class and
  `dataset` is a real MOABB dataset object (e.g. `moabb.datasets.BNCI2014_001()`).
- **This is a live MOABB integration, not custom parsing.** MOABB's dataset
  classes know the URL of each dataset's *original* host (BNCI2014_001 →
  BCI Competition IV / Graz archive; Schirrmeister2017 → the High Gamma
  Dataset's own host, etc.). First access downloads and caches; subsequent
  calls reuse the cache. Code comment confirming this:
  `"Fetching epochs via MOABB paradigm (this may download data)..."`
  (`datasets.py:235`).
- **Cache location:** MNE/MOABB default to `~/mne_data`
  (override via `MNE_DATA` env var). On this local Windows machine,
  `C:\Users\USER\mne_data` exists (created March 2026) but is **empty**, and
  `moabb`/`mne` aren't even installed in the local Python interpreter used for
  quick checks. Confirms: actual data fetching/caching/training happens on
  **Puhti HPC** when the SLURM jobs run `python main.py train ...` there — this
  local worktree only holds the code.
- **Registered datasets** (`datasets.py:31-85`), all with real MOABB class
  names: `BNCI2014_001`, `PhysionetMI`, `Cho2017`, `Schirrmeister2017`,
  `BNCI2015_001`.
- **Train/test split logic** (`datasets.py:256-277`):
  - Two-session datasets (`BNCI2014_001`, `BNCI2015_001`) → session 1 = train,
    session 2 = test.
  - Single-session datasets (`Cho2017`, `PhysionetMI`, `Schirrmeister2017`) →
    80/20 `StratifiedShuffleSplit`, `random_state=42`.
  - This matches what the paper's Datasets subsection and table captions say.
- **PhysionetMI special-casing** (`datasets.py:39-60`) — not currently used in
  any paper table, but already implemented: loads both imagined AND executed
  runs (12 runs/subject) to avoid rank-deficient CSP covariances, and crops to
  the 2-3s imagery window post-filtering (`crop_s`) to avoid fixation-period
  contamination. Ready to use if PhysionetMI is ever added to the paper.

---

## 2. Train pipeline — `run_train()` → `_run_single_fold()`

One call to `_run_single_fold()` per CV fold. In order:

1. **Load raw EEG** — see §1. Session 2 (or the 20% test split) is held out
   entirely and never touched by the fold splitter (`pipeline.py:474-485`).
2. **Filter bank** — fixed 6 overlapping bands, applied separately to
   train/val/test (`pipeline.py:180-182`, implementation in
   `preprocessing.py:57-165`). Details below (§2a).
### 2a. Filter bank — Butterworth vs. Bessel, causal vs. zero-phase, analog realizability

- **Default and actually-used filter: Butterworth.** `Config.filter_type` and
  both `bandpass_filter()`/`apply_filter_bank()` defaults are
  `"butterworth"` (`preprocessing.py:63,173`). `"bessel"` exists as a
  `--filter-type` override but is not the default. Per `EXPERIMENTS.md`:
  *"Bessel filter 63.4% — causal Butterworth wins"* — Bessel was tried and
  set aside for lower accuracy.
- **It is causal, not zero-phase — confirmed by implementation, not just
  the docstring.** `bandpass_filter()` uses `scipy.signal.sosfilt` (single
  forward pass), never `filtfilt` (forward+backward, needed for zero-phase
  but requires the entire signal including "future" samples — impossible in
  real-time hardware). The code docstring states this is deliberate:
  *"Causal filtering is used so that the filter bank maps directly to an
  analog Gm-C circuit implementation on neuromorphic hardware"*
  (`preprocessing.py:8-9`, restated at `:67-68`).
- **Both Butterworth and Bessel are analog-realizable** — they're classical
  analytical filter prototypes with standard continuous-time circuit
  realizations (Sallen-Key, multiple-feedback, or Gm-C/OTA-C ladder
  topologies). At `order=4`, designed via second-order-sections (`sos`),
  this is two cascaded biquad stages — a standard analog complexity. The
  tradeoff between them is response shape, not realizability: Butterworth =
  maximally flat magnitude, more group-delay distortion near cutoff; Bessel
  (`norm='delay'`, `preprocessing.py:97`) = maximally flat group delay
  (linear phase), less sharp magnitude rolloff. A true zero-phase/`filtfilt`
  filter, by contrast, **cannot** be built in real analog hardware at all —
  it needs the whole signal in both directions, which is fundamentally
  incompatible with a real-time circuit. This is exactly why causal
  filtering isn't just a software preference here; it's the only choice
  compatible with the paper's hardware-deployment premise.
- **Found and fixed a documentation error while checking this:** CLAUDE.md's
  pipeline diagram said *"Bandpass filter bank (Butterworth, zero-phase)"* —
  factually wrong, contradicts the actual causal/single-pass implementation
  and its own stated design rationale. Corrected 2026-07-06 (also removed a
  leftover "Per-fold adaptive band selection" line in the same diagram, a
  holdover from before adaptive band selection was removed from the code).

3. **Optional dataset-specific crop** — `pipeline.py:184-198`, driven by an
   optional `crop_s` key in `DATASET_REGISTRY` (currently set only for
   PhysionetMI: `crop_s=(2.0, 3.0)`). Trims each trial to just the imagery
   window *after* filtering but *before* CSP. Two reasons this order matters:
   (a) CSP fits spatial filters by maximizing between-class variance — if the
   covariance is computed over a signal that's mostly class-invariant
   fixation period plus a short imagery period, the fit gets diluted; cropping
   to just the imagery window before CSP keeps the covariance focused on the
   discriminative part. (b) Filters have a settling transient at signal
   start — filtering the *full* epoch first means that transient lands in the
   discarded fixation period, not in the short window you actually keep.
   Code comment confirms this reasoning verbatim (`pipeline.py:185-186`):
   *"Filtering on the full epoch avoids edge effects; cropping afterwards
   gives CSP clean signal from only the imagery period."* Not used by any
   dataset currently in the paper's tables (BNCI2014-001, Schirrmeister2017) —
   only relevant if/when PhysionetMI is added.
4. ~~Optional sliding-window augmentation~~ — **removed 2026-07-06.** Was
   off by default (`augment_windows=False`) and confirmed unused anywhere:
   never enabled in `run_puhti_array.sh`/`run_puhti_subject.sh`, never tried
   per `EXPERIMENTS.md`'s accuracy log, never tested, never in the paper.
   Deleted `Config.augment_windows/window_duration/window_step`, the
   `--augment-windows`/`--window-duration`/`--window-step` CLI flags, the
   `if cfg.augment_windows:` branch in `_run_single_fold`, and
   `preprocessing.window_filter_bank()` entirely. Same rationale as the
   `band_selection.py` removal earlier — dormant code with zero evidence of
   ever having been exercised.
5. **Pairwise CSP** — fit on train only, transform all three splits
   (`pipeline.py:220-233`, class in `preprocessing.py:157-386`). No
   leakage: `fit()` only ever touches the data/labels passed to it,
   `transform()` only reuses stored `filters_`/`ea_whiteners_`. Internals
   are richer than the paper documents — see §5a.
6. **Z-normalisation** — fit on train only, transform all three
   (`pipeline.py:236-239`, class in `preprocessing.py:794-865`). Per-feature
   mean/std computed across the `(n_trials × n_samples)` dimension so each
   CSP-output channel gets zero mean/unit variance; `+1e-8` epsilon on std.
   No leakage — verified same fit/transform separation as CSP.

### 5a. Pairwise CSP internals — confirmed mismatch with the paper's Methods section

Checked `preprocessing.py`'s `PairwiseCSP` class in full, then cross-checked
against every SLURM/submit script to see which optional features actually ran
in practice (not just what defaults to `True` in code). Two real discrepancies
found between what the paper describes and what was actually computed —
tracked as **B9**/**B10** in the paper's `TODO.md`:

- **Euclidean Alignment (EA)** — `euclidean_alignment=True` by default
  (`preprocessing.py:200`). Whitens every trial by a class-agnostic covariance
  (`_compute_ea_whitener`, computed from training data only) *before* CSP
  fitting (`preprocessing.py:265-271`). The whitener is stored and reused on
  val/test — no leakage. **The paper's Pairwise CSP subsection
  (`main.tex:346-370`) never mentions this step at all.**
- **Riemannian (Fréchet) mean vs. arithmetic mean** — `riemannian_mean=True`
  by default (`preprocessing.py:201`), routing to `_riemannian_mean_cov()` →
  `_riemannian_mean_from_covs()`: gradient descent on the SPD manifold
  (Moakher 2005), GPU-accelerated via batched `torch.linalg.eigh` when CUDA
  is available (~30-50× speedup, `preprocessing.py:555-618`), falling back to
  arithmetic mean only on numerical overflow. **The paper's Eq. `cov`
  (`main.tex:355`) states "$\bar{\boldsymbol{\Sigma}}_k^{(b)}$ is the
  arithmetic mean of normalised per-trial covariances"** — this describes
  `_mean_normalised_cov()` (`preprocessing.py:662-684`), the *other* code
  path, not the one that actually ran.
- **Optional Ledoit-Wolf shrinkage** — `csp_ledoit_wolf=False` by default
  (`preprocessing.py:202`); replaces Tikhonov regularisation with per-trial
  analytically-optimal shrinkage (`_ledoit_wolf_covs`, `preprocessing.py:687-712`)
  before averaging. The paper mentions "Ledoit-Wolf covariance shrinkage"
  once (`main.tex:521`) but this flag defaults off and needs
  `--csp-ledoit-wolf` to activate — check whether that mention refers to an
  ablation that was actually run or planned-but-not-executed.
- **Eigenvalue flooring in `_solve_csp`** (`preprocessing.py:734-787`) — before
  solving the generalised eigenvalue problem, the composite covariance
  `Σ_A + Σ_B` gets a relative eigenvalue floor (`max(largest_eigval × 1e-6,
  1e-10)`) to guard against rank-deficient composites — explicitly needed for
  high-channel/low-trial cases like PhysionetMI (64 ch, ~17 samples/class →
  rank-17 covariance). Not a bug, a documented numerical safeguard; not
  mentioned in the paper but also not something a Methods section would
  typically need to cover.
- **Confirmed via script audit, not just code defaults:** neither
  `run_puhti_array.sh` nor `submit_schirrmeister.sh`/`submit_bnci2015.sh`
  (both just call `submit_puhti.sh` with no extra args) ever pass
  `--no-euclidean-alignment`, `--no-riemannian-mean`, or `--csp-ledoit-wolf`.
  So EA=on and Riemannian-mean=on for every reported result.
  `submit_schirrmeister.sh`'s own comment confirms this directly: *"Riemannian
  mean on 128×128 is compute-intensive; request 4-hour wall time to be
  safe"* — the script author budgeting extra time specifically because
  Riemannian mean was running.
7. **Classical baselines (LDA + SVM)** — log-variance of the *same*
   z-normalised CSP output (not the spike-encoded features the SNN uses),
   fit on train, scored on val/test (`pipeline.py:241-252`,
   `baseline.py`). **Not shown in the paper's Fig. 1** — but this is exactly
   where the LDA/SVM numbers in Tables III/IV come from, computed every fold
   alongside the SNN, not from a separate script. See §7a for a plan to make
   this comparison fairer.
8. **Spike encoding** (delta/ADM) of z-normalised features, train/val/test
   (`pipeline.py:255-257`, implementation `encoding.py`). See §8 for a full
   deep dive — clean, no leakage risk (no fitting step at all, purely
   deterministic given fold-separated input).
9. **MIBIF feature selection** — fit on train spikes, applied to val/test
   (`pipeline.py:260-269`). **Actual selection mode used in training
   confirmed to differ from the paper's description (tracked as B10 in
   `TODO.md`):** `run_puhti_array.sh:111` passes `--mi-fraction 0.1`, which
   per `Config.mi_fraction`'s docstring keeps features with
   `MI ≥ 0.1 × max_MI` (threshold relative to the single best feature) and
   *overrides* `--feature-percentile` when set. The paper's Feature Selection
   subsection (`main.tex:380`) says *"The top 10% are retained"* — that
   describes percentile-based ranking, a different algorithm from the
   threshold-based selection that actually ran. The downstream "≈57
   features" estimate in the paper assumed percentile selection and may not
   hold under threshold-based selection either. See §8 for internals
   (spike-count-as-MI-proxy rationale, `mutual_info_classif` k default).
10. **SNN training** (`train_fold`) — AdamW, early stopping on val accuracy,
    best checkpoint kept (`pipeline.py:283-311`, model in `model.py`, loss in
    `losses.py`). See §8 for internals and three confirmed paper mismatches
    (B12-B14 in `TODO.md`).
11. **FP32 evaluation** on val and test (`pipeline.py:313-315`).
12. **INT8 whole-model quantisation + re-evaluation** on val and test
    (`pipeline.py:317-323`).
13. **CSP-only PTQ sweep at 8/6/4-bit** — re-quantise just the CSP filters,
    re-run the *already-trained* FP32 SNN on test data three times
    (`pipeline.py:325-344`). This and step 12 are two independent
    quantisation experiments (whole-SNN-INT8 vs. CSP-filters-only-Nbit) —
    **both missing from Fig. 1**, which shows only one classifier branch.
14. **Diagnostic plots + artifact save** — spike raster, neuron traces, weight
    histograms, confusion matrices, then pickle `csp`/`znorm`/`mibif` +
    `best_model.pt` + `pipeline_params.json` (`pipeline.py:346-427`).

**The test-set numbers that populate the paper's tables come from step 11-13,
inside training** — not from a separate inference run.

---

## 3. Test/inference pipeline — `run_infer()` (standalone `main.py infer`)

A **post-hoc reproduction path** — reloads everything from disk, never
retrains:

1. Load `pipeline_params.json` for the requested fold (bands, feature count,
   CSP `m`) (`pipeline.py:571-581`).
2. Load raw EEG, keep only the test split (`pipeline.py:589`).
3. Unpickle the fold's fitted `csp` and `znorm` (`pipeline.py:593-596`).
4. Optionally re-quantise CSP filters via `--csp-bits`
   (`pipeline.py:598-600`) — **this flag was broken until 2026-07-06**
   (`cfg.csp_bits` referenced but never defined in `Config` or the `infer`
   argparse subparser; fixed same day, verified with a standalone parser test
   since this environment's Python lacks `torch`).
5. Unpickle `mibif` if present (`pipeline.py:602-606`).
6. Re-run the identical preprocessing chain on test data: filter bank → CSP
   transform → z-norm transform → spike encode → MIBIF transform
   (`pipeline.py:608-617`).
7. Load `best_model.pt` into a fresh `SNNClassifier` (dropout forced to 0)
   (`pipeline.py:619-629`).
8. FP32 evaluate, then INT8-quantise-and-evaluate, save confusion matrices
   (`pipeline.py:631-651`).

**Purpose:** re-derive test accuracy for one fold without retraining — e.g.
sanity-check a saved model, or re-run the CSP-bit sweep interactively now
that the flag works.

---

## 4. Aggregate — `run_aggregate()`

Reads every fold's `pipeline_params.json`, writes `summary.csv` (per-fold rows
+ mean row), reloads each fold's saved model + preprocessing to recompute a
**summed** confusion matrix across all folds (`pipeline.py:663-876`).

---

## 5. What each saved metric actually means

Per-fold fields in `pipeline_params.json`:

| Field | Meaning |
|---|---|
| `val_acc_fp32` / `test_acc_fp32` | Held-out accuracy, full 32-bit float weights. `val` used for early-stopping/checkpoint selection (slightly optimistic); `test` is the honest, never-touched number. |
| `test_acc_int8` | Same trained SNN, weights compressed to 8-bit int (whole-model quantisation). Tests whether the *classifier* survives low-precision hardware. |
| `test_acc_csp_8bit/_6bit/_4bit` | CSP spatial filters (not the SNN) quantised to 8/6/4-bit, SNN weights left FP32. Simulates an analog/digital crossbar array's precision limit in the *feature-extraction* stage. 4-bit is a much harsher cut than 8-bit — this is where you'd see the pipeline's real hardware floor. |
| `val_acc_lda`/`test_acc_lda`, `val_acc_svm`/`test_acc_svm` | Classical baselines: log-variance of the same CSP+z-norm output, fit with a plain LDA/SVM. Control group — if the SNN can't beat these, the added complexity (spike encoding, LIF neurons, Van Rossum loss) isn't earning its keep on accuracy grounds. Significance testing on the numbers already in the paper's Tables III/IV shows: SNN significantly beats LDA on both datasets (p<0.005), but only ties SVM on BNCI2014-001 (p≈0.05–0.06, not significant) and loses to SVM on Schirrmeister2017 (p<0.001). |
| `best_epoch` / `stopped_epoch` | Which checkpoint got saved (peak val accuracy) vs. when training gave up (`early_stopping_patience=100` epochs with no improvement). |

Confusion matrices are row-normalised (`pipeline.py:850-852`): each row sums
to 100%, i.e. "of trials that were truly class X, what % got predicted as
each class" — shows *which* classes get confused with which, not just overall
accuracy.

---

## 6. Known gaps between the code and the paper's Fig. 1

Fig. 1 (the two-column Training/Inference TikZ diagram) shows a single
classifier branch: CSP → Z-Norm → Spike Encoder → MIBIF → LIF SNN. The actual
code does three additional things every fold that the figure omits:

1. The parallel LDA/SVM baseline branch (§2 step 7).
2. Whole-model INT8 quantisation (§2 step 12).
3. The separate CSP-only 8/6/4-bit PTQ sweep (§2 step 13).

None of this is incorrect — the figure is just simplified. Worth deciding
whether to extend the figure/caption to acknowledge the baseline branch and
both quantisation experiments, or leave it as an intentional simplification
and describe the rest only in prose.

---

## 7. SLURM/submit script audit (2026-07-06)

Read every training-invocation script to determine which optional CSP/MIBIF
behaviors actually ran, rather than assuming from code defaults alone.

- **`run_puhti_array.sh`** (V6.4) — the active, current script. Passes fixed
  6-band `--freq-bands`, `--csp-components-per-band 8`, `--mi-fraction 0.1`,
  and the Table I hyperparameters. Never passes `--no-euclidean-alignment`,
  `--no-riemannian-mean`, or `--csp-ledoit-wolf` — so those three stay at
  their `Config` defaults (EA on, Riemannian mean on, Ledoit-Wolf off).
- **`submit_puhti.sh`** — orchestrates the 3-stage train→aggregate→analyze
  SLURM dependency chain. Forwards a user-supplied `EXTRA_ARGS` to every
  training task, and lets `ARRAY_SCRIPT` be overridden (default
  `run_puhti_array.sh`).
- **`submit_schirrmeister.sh`**, **`submit_bnci2015.sh`** — dataset-specific
  wrappers, both just call `bash submit_puhti.sh "${RESULTS_DIR}"` with no
  extra args. Confirms the Schirrmeister2017 and BNCI2015-001 runs also use
  the same EA/Riemannian-mean/Ledoit-Wolf defaults as BNCI2014-001.
  `submit_schirrmeister.sh`'s own comment — *"Riemannian mean on 128×128 is
  compute-intensive; request 4-hour wall time to be safe"* — is direct
  evidence the script author expected Riemannian mean to run for this
  dataset, not an oversight.
- **`run_puhti_subject.sh` — stale/broken, do not use as-is.** Passes
  `--no-use-bn`, `--lr-scheduler`, `--surrogate-slope`, `--activity-reg`,
  `--hidden-neurons2` — none of which exist in the current `config.py`
  argparse. Running it today would fail immediately with an argparse error.
  `run_puhti_array.sh`'s own comment explains the history: those were "V6
  features (activity_reg, 3-layer SNN, LR scheduler)... disabled here [because]
  they collapsed accuracy to ~25% (chance) when combined." This script is
  leftover from that abandoned V6 branch. Candidate for deletion, or a rewrite
  to match current `config.py` if a single-subject (non-array) launcher is
  still wanted.
- Other scripts present but not yet audited in detail: `run_puhti_aggregate.sh`,
  `run_puhti_analyze.sh`, `run_puhti_dataset_test.sh`, `run_puhti_e2e*.sh`,
  `run_puhti_lava*.sh`, `run_puhti_mc*.sh`, `run_puhti_static6.sh`,
  `submit_e2e.sh`, `submit_lava.sh`, `submit_mc.sh`, `submit_physionet.sh`.

---

## 7a. Plan — making the classical-baseline comparison fairer (discussed 2026-07-06)

Current `baseline.py` (fixed `C=1, gamma='scale'` SVM, unregularised
`solver='svd'` LDA, both on log-variance of the same z-normalised CSP output
as the SNN, no leakage) is methodologically sound for what it isolates, but
has two real weaknesses: the SVM is deliberately untuned, and the comparison
bundles classifier choice together with feature-representation choice (LDA/SVM
get one log-variance scalar per feature; the SNN gets a full 50-timestep spike
train). **Judged not essential for publication** (tracked as optional in the
paper's `TODO.md`, distinct from B11's ablation study, which *is* essential) —
but here's the implementation plan if/when it's picked up, cheapest first:

- **Tier 0 — tune what exists, no new dependency.** Small grid search over
  SVM's `C`/`gamma` using the val split already computed every fold for model
  selection (mirrors how val already picks the SNN's best checkpoint — see §5
  step 10). Switch LDA to `solver='lsqr', shrinkage='auto'` instead of
  `solver='svd'` (currently unregularised — matters more on Schirrmeister2017's
  128 channels than BNCI2014-001's 22). Both changes are localised entirely to
  `baseline.py`'s `run_baseline_classifiers()`.
- **Tier 1 — add MDM (Minimum Distance to Riemannian Mean), reusing existing
  math.** No new dependency (`pyRiemann`) needed — `preprocessing.py` already
  has `_riemannian_mean_from_covs()` (GPU-accelerated, §5a) and the SPD helpers
  (`_spd_log`, `_spd_sqrt_invsqrt`). MDM is: compute each class's Riemannian
  mean from training-fold covariances (already have the function), classify a
  test trial by nearest mean under the affine-invariant Riemannian distance.
  Would live alongside `extract_logvar`/`run_baseline_classifiers` in
  `baseline.py`, operating on the per-band covariances already computed
  inside `PairwiseCSP.fit()` rather than on log-variance features.
- **Tier 2 — separate classifier from representation.** Add log-variance →
  small dense MLP (same param budget as the SNN, tests whether *any* neural
  net beats classical ML independent of spiking) and rate-coded log-variance
  → the existing `SNNClassifier` architecture (tests whether the specific
  delta/adaptive-threshold encoder matters vs. any spike-based front end).
  New code needed: a small MLP baseline class and a rate-coding function
  (population-code a scalar log-variance value into a Poisson-ish spike
  train, analogous to `make_target_spikes` in `losses.py` but for input
  features instead of targets). No new data collection — reuses features
  already computed every fold. This overlaps with B11's ablation study scope.
- **Tier 3 — re-run significance testing.** Once Tier 0 lands, redo the
  paired t-test/Wilcoxon comparisons (§5 step 7 note, `sig_test.py` in the
  paper-side scratchpad) against the tuned baseline — the current "not
  significant, p≈0.05-0.06" SNN-vs-SVM result on BNCI2014-001 could shift
  either direction once SVM is properly tuned.

---

## 8. Deep dive — spike encoding, MIBIF, and SNN training internals (2026-07-06)

Read `encoding.py`, `mibif.py`, `training.py`, `model.py`, and `losses.py` in
full, then cross-checked every number against the paper's Sections 2.3-2.6
(`main.tex:383-461`). No leakage or correctness bugs found in any of the five
files. Four real paper-vs-code mismatches found, tracked as **B12-B14** in the
paper's `TODO.md` (B10 above already covers the MIBIF selection-mode issue).

### 8a. Spike encoding (`encoding.py`) — clean, no changes needed

- Both `_adaptive_threshold_encode_jit` (delta) and `_adm_encode_jit` (ADM)
  are `@torch.jit.script`-compiled, deterministic functions of already
  fold-separated input — no fitting step at all, so no leakage risk applies
  here structurally (unlike CSP/ZNorm/MIBIF, there's nothing to "fit").
- Delta is confirmed the one actually used for every reported result — no
  script anywhere passes `--encoder-type adm`.
- Both encoders match their paper equations exactly: delta threshold dynamics
  (`θ(t+1) = θ(t)·γ + Δ_+` on spike, `θ(t+1) = θ(t)·γ` otherwise) map 1:1 onto
  `main.tex:396-402`.
- Structural quirk, not a bug: timestep 0 never emits a spike in either
  encoder (delta has no `t-1` to diff against; ADM initialises `v_ref = x[0]`
  so `diff=0` at `t=0`). One of 50 timesteps is always silent by
  construction — negligible given `T=50`, not worth changing.
- `adm_reconstruction_rmse()` (`encoding.py:164-209`) is a validation utility
  for the ADM path only (checks reconstruction error against a ≤5% target
  from an earlier planning doc) — not used in the delta-encoder path that
  actually produces the paper's numbers.

### 8b. MIBIF (`mibif.py`) — correct, one reproducibility fragility

- `fit()` computes MI only from `spikes`/`y` passed to it (training data);
  `transform()` only reuses `selected_indices_` — no leakage.
- Feature representation for MI estimation is **spike count** (sum over T),
  not log-variance and not per-timestep spikes — deliberate, documented
  design choice (`mibif.py:1-9`): "Raw per-timestep spikes are too noisy;
  collapsing to total spike count... gives a stable, scalar representation."
  Reasonable and clearly justified.
- **Confirms B10 at the code level, not just the script-flag level:**
  `mi_fraction`/`feature_percentile` are explicitly "mutually exclusive —
  `mi_fraction` takes priority" per the class docstring (`mibif.py:28-38`).
  Since `run_puhti_array.sh` passes `--mi-fraction 0.1`, every reported result
  used threshold mode, not the percentile mode the paper describes.
- **Fragility, not a bug:** `mutual_info_classif(spike_counts, y,
  discrete_features=False, random_state=self.random_state)` (`mibif.py:103-106`)
  never explicitly passes `n_neighbors`. sklearn's current default is 3,
  which happens to match the paper's claimed "k=3" (`main.tex:379`) — but
  it's not pinned in code. A future sklearn version could silently change
  this default without any diff in this repo. Cheap fix if ever revisited:
  pass `n_neighbors=3` explicitly.

### 8c. SNN training (`training.py`, `model.py`, `losses.py`) — correct, three disclosure gaps

- **No leakage:** val is used only for `EarlyStopping`/checkpoint selection
  (`training.py:314-324`), never for gradient updates. Best-state restoration
  before returning is correct (`training.py:341`).
- **Correctly implemented:** AMP guard (CUDA-only, `training.py:270-271`),
  per-epoch trial shuffling, Van Rossum FFT convolution (`losses.py:82-110`,
  matches the paper's Eq. `loss` exactly), population-coded Bernoulli target
  generation (`losses.py:166-211`), winner-take-all decode
  (`model.py:152-174`, matches Eq. `decode`).
- **B12 — hidden hyperparameter:** `surrogate.fast_sigmoid(slope=25)`
  (`model.py:84`) is hardcoded. The paper's surrogate-gradient equation
  (`main.tex:427`) omits the slope term, and Table I has no row for it.
- **B13 — Table I completeness gap:** `batch_size=64` (`pipeline.py:307`) and
  Van Rossum `tau_vr=10.0` (`pipeline.py:306`) are both hardcoded and used
  every fold, but neither appears in Table I.
- **B14 — parameter-count overclaim:** "~8,700 params... several orders of
  magnitude smaller than deep CNN baselines" (`main.tex:438-440`) contradicts
  the paper's own Introduction, which cites EEGNet at "fewer than 3,000
  parameters" (`main.tex:124`). The SNN is ~3× *larger* than EEGNet — true
  against ATCNet/DeepConvNet/TFANet, false against EEGNet, and the sentence
  doesn't distinguish.
- **Worth a note, not a fix:** no gradient clipping anywhere in the optimizer
  step (`training.py:298-306`). Not necessarily broken — it evidently trains
  fine — but a common robustness safeguard for surrogate-gradient SNN
  training that's absent here.
- **Could not verify locally:** whether snnTorch's `Leaky` neuron actually
  defaults to reset-by-subtraction (matching the paper's Eq. `lif` at
  `main.tex:419`, `U_i(t) = βU_i(t-1) + W^{(1)}s(t-1) - S_i(t-1)`) — model.py
  never passes an explicit `reset_mechanism` to `snn.Leaky()`, and `snntorch`
  isn't installed on this local machine to check its source. Worth a quick
  confirmation next time on Puhti.

---

## 9. Plan — pipeline reliability / hardware-realism testing (2026-07-06)

User's own result analysis: the current "hardware compatibility" evidence
(INT8 whole-model quantisation + separate CSP-bit sweep) is messy, redundant,
and only partial-truth — doesn't test real physical scenarios. Read
`quantization.py` in full to confirm this precisely rather than just agree
with the impression. Tracked as **B15** in the paper's `TODO.md` — the single
most consequential open item, since it's the actual experimental work needed
to back up the paper's central hardware-compatibility claim (B8).

**Known scope gap (2026-07-07):** Tier B (implemented, §10 below) only
noise-tests CSP weights, SNN weights, and LIF beta — not the filter bank's
Butterworth/Bessel coefficients (the most important gap, since §2a's
hardware argument specifically leans on the bandpass filter being
Gm-C-realizable), the spike encoder's `adapt_inc`/`decay`, Z-normalisation
mean/std, the Euclidean Alignment whitener, or SNN biases. Full priority
order in `RESULTS_LOG.md`'s "Known scope gap" section (code repo) and
`TODO.md`'s B15 entry (paper folder). Not yet implemented.

### 9a. What's actually wrong with the current quantisation tests

Read `quantization.py` end to end (`quantize_tensor_symmetric`,
`quantize_array_symmetric`, `quantize_model`, `quantize_csp_filters`,
`quantization_report` — the whole file, 218 lines):

1. **Never applied jointly.** `_run_single_fold` calls `quantize_model()`
   (SNN Linear weights → INT8, `pipeline.py:318`) and, completely separately,
   loops `quantize_csp_filters()` over `[8,6,4]` bits (`pipeline.py:325-344`).
   No code path ever quantises both simultaneously. A real chip would have
   both stages lossy at once — the current tables represent two disconnected
   partial scenarios, neither of which is the actual deployment condition.
2. **Purely deterministic bit-rounding — zero analog noise model.**
   `quantize_tensor_symmetric()`/`quantize_array_symmetric()`
   (`quantization.py:48-106`) do exactly: `scale = max(|W|)/q_max`,
   `Wq = round(W/scale).clamp(-q_max,q_max)`, `Wd = Wq * scale`. That's it —
   no thermal noise, no device mismatch, no comparator offset, no
   temperature drift. The module's own docstring is honest about the
   limitation: *"This is simulated quantisation for evaluation purposes
   only... cannot be run on neuromorphic hardware directly"*
   (`quantization.py:18-23`) — but a paper claiming hardware realizability
   needs an actual noise-robustness test, not just an honest caveat.
3. **Biases and LIF parameters are never touched.** `quantize_model()` only
   touches `nn.Linear.weight` (`quantization.py:141-145`) —
   *"Biases and LIF-neuron parameters (beta, threshold, …) are left at full
   precision (standard practice — INT8 biases add hardware complexity for
   minimal benefit)"* (`quantization.py:14-16`). Reasonable for biases, but
   `beta` (an RC time constant) and the firing threshold (a comparator
   reference voltage) are exactly the components that vary between
   fabricated analog units — leaving them untouched in every experiment
   assumes a source of real-world non-ideality doesn't exist.
4. **No event/spike counting exists anywhere in the codebase**, which is
   precisely why B8's energy formula had to guess: `SNNClassifier.forward()`
   (`model.py:108-146`) computes `spk1` (hidden-layer spikes) every timestep
   inside the loop but never appends/returns/logs it — only `spk2`/`mem2`
   (output layer) are stacked and returned. The energy equation
   (`main.tex:809`) ended up using the *input encoder's* firing rate as a
   stand-in for the whole network's activity because the hidden layer's real
   rate was structurally inaccessible.

### 9b. Plan, tiered by cost — cheapest first

- **Tier A — merge the two quantisation experiments (pure re-plumbing, no
  new modeling).** Quantise CSP filters and SNN weights together across a
  small paired grid (e.g. CSP∈{8,6,4}bit × SNN∈{8,6,4}bit) inside
  `_run_single_fold`, replacing the two separate blocks at
  `pipeline.py:317-344` with one joint sweep. Report one coherent
  "accuracy vs. combined hardware precision" grid instead of two
  disconnected tables.
- **Tier B — add real analog noise injection to `quantization.py`.** Three
  new, physically-motivated noise functions, each swept over a severity
  range:
  - **Weight noise** — additive Gaussian noise on CSP/SNN weights (σ as a %
    of weight range), modeling analog crossbar conductance variation.
  - **Threshold/comparator offset** — random per-neuron offset on the LIF
    firing threshold and the spike-encoder's adaptive threshold
    (`base_thresh` in `encoding.py`), modeling comparator mismatch.
  - **β (leak-rate) variation** — random per-neuron perturbation of the LIF
    membrane decay constant, modeling RC time-constant mismatch across
    fabricated leaky integrators. Would need `SNNClassifier` to accept a
    per-neuron `beta` tensor instead of the current shared scalar
    (`model.py:73,89,94` — `beta: float` passed identically to both
    `snn.Leaky()` calls).
- **Tier C — Monte Carlo repetition.** Every noisy condition needs N repeats
  (20-50) with fresh random draws, reporting mean±std accuracy — not today's
  single deterministic number per bit-depth. Cheap: pure inference on
  already-trained `best_model.pt` checkpoints (`evaluate_model()`,
  `training.py:134-177`), no retraining required, so N× cost stays trivial.
- **Tier D — one unified degradation curve.** "Accuracy vs. hardware-realism
  level" (combining Tiers A-C), replacing the scattered INT8 column in
  Tables III/IV and the separate CSP-bit columns in Table VI with a single,
  stronger piece of evidence.

### 9c. Event-counting instrumentation

- Modify `SNNClassifier.forward()` (`model.py:108-146`) to also accumulate
  `spk1` (currently computed at line 136, discarded after feeding `fc2` at
  line 138) — small change, e.g. return a third stacked tensor or a dict of
  `{input, hidden, output}` spike trains instead of just `(spk_out, mem_out)`.
- Define **total events per trial** = input-encoder spikes (already have via
  `encode_csp_projections`'s logged firing rate, `encoding.py:269-276`) +
  hidden-layer spikes (newly exposed `spk1`) + output-layer spikes (`spk2`,
  already available) — summed over all timesteps and neurons, for one
  single-trial classification.
- Compute inside `evaluate_model()`/`_run_single_fold`, log as a new per-fold
  metric (`mean_events_per_trial`), averaged over test trials. Measure under
  both clean and Tier-B noisy/quantised conditions, since firing rates could
  shift under noise in either direction.
- **Directly fixes B8 problem 2:** once measured, this replaces the flawed
  "0.15 spikes/neuron/timestep" assumption (`main.tex:804-805`, which was
  actually the input encoder's rate, not a measured hidden-layer rate) with a
  real, end-to-end measured number — the energy equation becomes evidence
  instead of a guess.

### 9d. Feasibility summary

| Piece | New code | Reused |
|---|---|---|
| Tier A (joint sweep) | Loop restructuring in `_run_single_fold` | `quantize_model`, `quantize_csp_filters` |
| Tier B (noise injection) | 3 new functions in `quantization.py`; `SNNClassifier` needs per-neuron `beta` support | `evaluate_model` |
| Tier C (Monte Carlo) | Repeat-and-aggregate wrapper | `evaluate_model`, saved checkpoints |
| Tier D (unified curve) | Plotting/reporting only | `visualization.py` patterns |
| Event counting | `forward()` return signature change; aggregation logic | `encoding.py`'s existing firing-rate logging |

No new training runs needed anywhere in this plan — everything operates on
already-trained `best_model.pt` checkpoints via repeated inference, so the
whole reliability-test suite is cheap relative to the original training cost.

---

## 10. Implementation record (2026-07-06) — B2, B11, B15, baseline Tier 0

Everything below was actually written and verified (`py_compile` on every
file, standalone argparse dry-runs, pure-numpy sanity checks of new math) —
not just planned. **Cannot verify `torch`/`snntorch`/`sklearn`-dependent
behavior locally** (none installed on this Windows machine) — flagged
per-item below; confirm on the first Puhti run before trusting for
published results. Nothing existing was broken: checked every test file's
imports and call sites (`tests/test_band_csp.py`, `test_cv_pipeline.py`,
`test_spike_snn.py`) against every changed signature — all new parameters
have defaults matching prior behavior, so old call sites are unaffected.

### 10a. B2 — Cho2017 submit script

New `submit_cho2017.sh`, mirroring `submit_bnci2015.sh`'s pattern (Cho2017
already registered in `datasets.py`'s `DATASET_REGISTRY`; only
`submit_cho2017.sh` was missing). No code changes needed beyond the script.

### 10b. B11 — Ablation flags

- **Single-end CSP** (`preprocessing.py`): `PairwiseCSP.__init__` and
  `_solve_csp()` gained a `dual_end: bool = True` parameter. When `False`,
  returns the last `2m` eigenvectors (largest-eigenvalue end only) instead
  of `m` from each end — **same total filter count as dual-end** (`2m` in
  both modes), so the ablation isolates *which* eigenvectors are used, not
  *how many*. Verified with a standalone numpy reimplementation of the
  branching logic (both modes produce identical output shape). CLI:
  `--csp-single-end` / `--csp-dual-end` (default).
- **Fixed-threshold encoder** (`encoding.py`): new `_fixed_threshold_encode_jit`
  kernel — same delta rule as the adaptive encoder, threshold held constant
  at `base_thresh` (no `adapt_inc`/`decay`). Mirrors the already-working
  adaptive kernel's TorchScript-compatible patterns exactly, so it should
  JIT-compile cleanly — **not verified against an installed `torch` locally**.
  CLI: `--encoder-type fixed` (alongside existing `delta`/`adm`).
- **Cross-entropy loss** (`losses.py`, `training.py`): new
  `cross_entropy_spike_loss()` — sums output spikes over time and within
  each class population (identical reshape to `SNNClassifier.decode`), then
  standard softmax cross-entropy against the true label; no target spike
  trains needed. `train_fold()` gained a `loss_type: str = "van_rossum"`
  parameter, branching between this and the existing `van_rossum_loss`. CLI:
  `--loss-type cross_entropy` (alongside default `van_rossum`).
- **Also promoted two previously-hardcoded values to `Config` fields**
  while touching this call site (same "hidden hyperparameter" pattern as
  B12/B13): `tau_vr` (was hardcoded `10.0` in `pipeline.py`) and
  `train_batch_size` (was hardcoded `64`) — both now configurable via
  `--tau-vr`/`--train-batch-size` and recorded in `pipeline_params.json`.
- **Submit scripts:** `submit_ablation_single_end_csp.sh`,
  `submit_ablation_fixed_encoder.sh`, `submit_ablation_cross_entropy.sh` —
  each a thin wrapper around `submit_puhti.sh` passing the corresponding
  flag, writing to its own `Results_ablation_*` directory.

### 10c. B15 — Reliability / hardware-realism testing

- **Tier A (joint CSP+SNN quantisation sweep):** `pipeline.py`'s existing
  CSP-only PTQ loop now also quantises the SNN model at each of
  `[8,6,4]` bits inside the same loop (reusing the CSP-quantised spikes
  already computed per `csp_bits` value, so only the SNN needs re-quantising
  per pair) — 9 `(csp_bits, snn_bits)` combinations per fold, saved as flat
  `test_acc_joint_csp{X}_snn{Y}` fields in `pipeline_params.json` and
  `summary.csv`. **Kept the original isolated `test_acc_int8` and
  `test_acc_csp_{8,6,4}bit` fields completely unchanged** — confirmed
  `analyze_results.py` reads those exact keys, so this was additive, not a
  replacement.
- **Tier B (analog noise injection, `quantization.py`):**
  `inject_weight_noise_tensor`/`inject_weight_noise_array` (Gaussian noise
  sized as a fraction of peak magnitude — conductance variation),
  `inject_model_weight_noise`/`inject_csp_filter_noise` (apply the above to
  `SNNClassifier` Linear weights / CSP filter dicts respectively),
  `inject_beta_noise` (per-neuron LIF membrane-decay perturbation — RC
  time-constant mismatch). Verified the core noise math via pure-numpy
  sanity check (zero-noise passthrough, seed reproducibility, different
  seeds diverge). **`inject_beta_noise`'s assumption that snnTorch's
  `Leaky.beta` accepts a per-neuron tensor is documented snnTorch behaviour
  but not verified against an installed snnTorch locally** — confirm on
  first Puhti run. Threshold/comparator-offset noise needed a new JIT kernel
  variant, `_adaptive_threshold_encode_noisy_jit` (`encoding.py`), accepting
  a per-feature threshold tensor instead of a scalar — existing production
  kernel untouched, exposed via `encode_tensor_with_threshold_noise()`.
- **Tier C (Monte Carlo) + orchestration:** new module
  `fbcsp_snn/reliability.py` — `monte_carlo_eval()` (generic repeat-and-
  aggregate helper) and `run_reliability(cfg)`, a new pipeline entry point
  that loads a trained fold's saved artifacts (same loading logic as
  `run_infer`) and sweeps CSP weight noise, SNN weight noise, and beta noise
  each across `cfg.reliability_severities` with `cfg.reliability_n_repeats`
  Monte Carlo draws, saving `fold_dir/reliability_results.json`. Wired as a
  new `main.py reliability --subject-id N --fold K` CLI mode
  (`config.py`'s `reliability` subparser, `main.py`'s dispatch). One-
  directional import (`reliability.py` imports pipeline.py's private
  loading helpers; pipeline.py does not import back) — no circular-import
  risk. Pure inference on saved checkpoints, no retraining.
- **Event counting:** `SNNClassifier.forward()` gained an opt-in
  `return_hidden: bool = False` parameter — default preserves the exact
  2-tuple `(spk_out, mem_out)` return every existing caller expects (checked
  `training.py`, `pipeline.py` ×2, and `tests/test_spike_snn.py`'s
  `spk_out, mem_out = model(batch_spikes)` call — all unaffected). New
  `evaluate_model_with_events()` in `training.py` (alongside, not replacing,
  `evaluate_model`) sums input-encoder + hidden-layer + output-layer spikes
  per trial and returns `mean_events_per_trial` alongside accuracy. Wired
  into `_run_single_fold`'s FP32 test evaluation — **this is now a measured
  number, saved every fold**, replacing the B8-flagged unmeasured proxy.
  Added to `pipeline_params.json`, `summary.csv` (`run_aggregate`'s
  fieldnames + mean-row computation).
- **Submit script:** `run_puhti_reliability.sh` — array job, one task per
  already-trained fold (same subject/fold task-ID mapping as
  `run_puhti_array.sh`), must run after the corresponding training task has
  completed (checks for `pipeline_params.json`/`best_model.pt` and exits
  cleanly if missing, inherited from `run_reliability`'s own guard).

### 10d. Baseline Tier 0 — SVM/LDA tuning (`baseline.py`)

- SVM now grid-searches `C∈{0.1,1,10} × gamma∈{'scale',0.01,0.1}` (9
  combinations), scored on the val split already computed every fold —
  mirrors how val already selects the SNN's best checkpoint. Test is scored
  exactly once, after the best combination is chosen (no leakage).
- LDA switched from unregularised `solver='svd'` to
  `solver='lsqr', shrinkage='auto'` (Ledoit-Wolf) — matters more as
  `n_features` grows relative to `n_trials` (Schirrmeister2017's 128
  channels vs. BNCI2014-001's 22).
- Selected `svm_best_c`/`svm_best_gamma` now saved to `pipeline_params.json`
  for reproducibility (same "no hidden hyperparameters" reasoning as
  B12/B13). Return dict schema otherwise unchanged (`val_acc_lda`,
  `test_acc_lda`, `val_acc_svm`, `test_acc_svm`) — `pipeline.py`'s existing
  consumers unaffected.
- **`sklearn` isn't installed locally** — `LinearDiscriminantAnalysis`'s
  `lsqr`/`shrinkage='auto'` combination and `SVC(gamma=<float>)` are both
  long-stable, non-experimental sklearn APIs, so confidence is high, but
  this should be confirmed on the first Puhti run alongside the other
  library-dependent pieces above.

### 10e. What's next

All of the above is **code, verified to compile and argue-parse correctly,
not yet run**. The actual "submit → wait → copy results" loop (B2's two
datasets, B11's three ablations, B15's reliability sweep, and re-validating
Tier 0's baseline tuning) hasn't started yet — that's the next phase.

---

## 11. Closing the reliability sweep's scope gap (2026-07-07)

User's request: fully back the "analog-circuit-realizable" claim, so the
noise-injection scope gap flagged after §10 (CSP + SNN weights + beta only)
needed closing. All five gaps closed, plus one new experiment added on top.

### 11a. Filter bank coefficient noise (the most important gap)

New `bandpass_filter_noisy()`/`apply_filter_bank_noisy()` in
`preprocessing.py`, added as new functions alongside (not modifying)
`bandpass_filter()`/`apply_filter_bank()` — the production kernels used by
every training run stay untouched. Perturbs the `scipy.signal` `sos`
(second-order-sections) filter design coefficients with noise from
`quantization.py`'s `inject_weight_noise_array`.

**Real bug caught and fixed during implementation, not just planned:**
naively adding Gaussian noise to the entire `sos` array breaks scipy's
requirement that column 3 (the `a0` coefficient) stay exactly `1.0` per
section — `sosfilt` raises `ValueError: sos[:, 3] should be all ones` and
rejects the array outright. Caught by actually running a numpy/scipy sanity
check before considering this done (not just compiling). Fixed by
perturbing only the other 5 columns per section:
```python
noisy_cols = [0, 1, 2, 4, 5]
sos[:, noisy_cols] = inject_weight_noise_array(sos[:, noisy_cols], sigma_frac, seed=seed)
```
This is also the physically correct choice, not just a workaround: `a0=1`
is scipy's normalisation convention, not itself a real component value — the
true circuit coefficients are the *ratios* of the physical parameters to
`a0`, so leaving it fixed and perturbing the rest matches what "component
variation" actually means here. Verified against both Butterworth and
Bessel filters (both use the same `a0=1` convention) before moving on.

### 11b. Spike encoder: adapt_inc/decay noise (same reasoning as beta)

Generalized `_adaptive_threshold_encode_noisy_jit` (`encoding.py`) from
accepting a per-feature `init_threshold` tensor with scalar `adapt_inc`/
`decay`, to accepting all three as per-feature tensors. The Python wrapper
`encode_tensor_with_threshold_noise()` gained `adapt_inc_sigma_frac`/
`decay_sigma_frac` parameters alongside the existing `sigma_frac`, via a new
small helper `_noisy_per_feature()` shared across all three parameters
(threshold, adapt_inc, decay — each independently perturbable, with `decay`
clamped to `[1e-3, 0.999]` to stay a valid multiplicative factor).

**Also caught while doing this work:** the *original* Tier B threshold-noise
function (`encode_tensor_with_threshold_noise`, built in the §9/§10 pass)
had never actually been wired into a runnable sweep in `reliability.py` —
built but unused. Fixed alongside the new adaptation-noise work; both are
now separate sweeps (`encoder_threshold_noise`, `encoder_adaptation_noise`).

### 11c. Z-normalisation and Euclidean Alignment whitener noise

Two new functions in `quantization.py`, both thin wrappers around the
already-verified `inject_weight_noise_array`:
- `inject_znorm_noise(mean, std, sigma_frac, seed)` — perturbs
  `ZNormaliser.mean_`/`std_`, std floored at `1e-8` (matching `ZNormaliser.
  fit`'s own epsilon).
- `inject_ea_whitener_noise(ea_whiteners, sigma_frac, seed)` — mirrors
  `inject_csp_filter_noise`'s exact structure (same `Dict[int, np.ndarray]`
  shape as `PairwiseCSP.ea_whiteners_`). Degrades gracefully to a no-op if a
  fold's CSP was fit with `euclidean_alignment=False` (empty dict, nothing
  to perturb, not an error).

### 11d. SNN biases

`inject_model_weight_noise` (`quantization.py`) gained an `include_bias:
bool = True` parameter — biases are now perturbed by default alongside
weights, reasoning being that `quantize_model`'s bias-exclusion rationale
("INT8 biases add hardware complexity for minimal benefit") is specific to
*bit-precision* quantisation and doesn't carry over to *noise injection*,
where biases are just as physically-realised (offset currents/voltages) as
weights. `include_bias=False` still available to restrict to the old scope.
Purely additive to the function signature — the existing call in
`reliability.py` (`inject_model_weight_noise(model, sigma, seed=seed)`)
needed no changes, new parameter defaults transparently.

### 11e. Wiring — `_encode_test()` generalised to a single noise-composable function

`run_reliability()`'s `_encode_test()` helper was rewritten from taking one
positional `csp_filters_to_use` argument to a fully keyword-only function
with one optional override per noise source (`csp_filters`, `ea_whiteners`,
`znorm_mean`, `znorm_std`, `filter_sigma_frac`/`filter_seed`,
`encoder_thresh_sigma`/`encoder_adapt_sigma`/`encoder_decay_sigma`/
`encoder_seed`). Every parameter defaults to the clean/saved value, so
`_encode_test()` with no arguments reproduces the noise-free pipeline
exactly — verified this is what the "clean baseline" computation at the top
of `run_reliability()` actually calls. Every sweep function passes only the
override(s) relevant to what it's testing; nothing leaks between calls
since each call resets all four mutable pieces (`csp.filters_`,
`csp.ea_whiteners_`, `znorm.mean_`, `znorm.std_`) before applying overrides,
never carrying over state from a previous call.

`run_reliability()` now runs **9 sweeps**: `csp_weight_noise`,
`snn_weight_noise`, `beta_noise`, `filter_bank_noise`, `ea_whitener_noise`,
`znorm_noise`, `encoder_threshold_noise`, `encoder_adaptation_noise`, and
(§11f) `joint_noise_all_sources`.

### 11f. New experiment: joint noise across all sources simultaneously

Not part of the original scope-gap list — proposed as an additional
experiment, same justification already established for `pipeline.py`'s
joint CSP+SNN quantisation grid: a real chip has every stage imperfect at
once, so testing sources in isolation (as every sweep in §11a-d does
individually) doesn't represent the actual deployment scenario. At each
severity, `_eval_joint_noise()` perturbs all 7 sources simultaneously
(CSP weights, EA whitener, Z-norm, filter bank, encoder threshold+adapt+
decay, SNN weights+biases, LIF beta) with distinct seed offsets per source
(`seed`, `seed+1000`, `seed+2000`, ... `seed+6000`) so they don't share
identical noise draws, then evaluates once. This is the single most
representative number for "does this pipeline survive being built as an
actual analog chip" — more so than any individual sweep.

### 11g. Verification

`py_compile` across every changed file, plus targeted runtime checks where
possible without `torch`/`scipy` dependencies installed for the full stack:
- Pure numpy/scipy check of the filter-noise fix (§11a) — confirmed
  `sosfilt` accepts the noisy `sos`, output is finite, differs from clean,
  and `sigma_frac=0` reproduces the original exactly.
- Full argparse dry-run regression check across all 4 CLI modes — unchanged
  from §10g, still passes (this session's changes don't touch `config.py`).
- **Still not verified locally** (same standing limitation as §10): the
  `torch.jit.script` compilation of the generalized encoder kernel, and the
  actual end-to-end sweep execution against a real trained checkpoint —
  both need the first real Puhti run.

---

## 12. Energy computation — extending an existing script, not replacing it (2026-07-08)

Mid-session discovery: Puhti's project directory had ~40 `Results_*`
directories and several loose top-level scripts never visible to this repo
(untracked by git). Among them, a working `compute_energy.py` +
`run_lava_infer.py` pair — considerably more developed than assumed at
first. Decision: extend, don't replace.

### 12a. What the existing script already did well

- **Real measured Lava SynOps**, not an assumed firing rate — reads
  `Results_lava/lava_summary.csv` (produced by `run_lava_infer.py`, which
  runs actual Lava-framework inference and reports the framework's own
  synaptic-operation accounting).
- **Real cited references** for an analog front-end: Gm-C filter bank
  (Qian 2017, Verhoeven 2007), ADM encoder (Sharifshazileh 2021), ReRAM CSP
  crossbar (Burr 2017) — both a "modern" and "conservative" per-stage
  estimate.
- **Real cited GPU/CPU/EEGNet-M4 comparisons** — V100 TDP-based GPU
  baseline, ARM Cortex-A72 edge-CPU baseline, and a genuinely measured
  external figure (Burrello et al. 2020, EEGNet on Cortex-M4F, 4.28 mJ).
- **Already states its own honesty boundary**: only the Loihi SNN figure is
  measured; front-end figures are extrapolated from cited silicon, reported
  as estimates, not measured results.

This already resolves all three problems flagged in B8 (paper folder
`TODO.md`) — the front-end scope, the firing-rate justification, and the
missing GPU baseline — none of which needed to be built from scratch.

### 12b. Important clarification that reshapes the framing

User: **Loihi 2 is a digital, asynchronous, event-driven chip, not
analog.** The existing script's Gm-C/ADM/ReRAM front-end estimates describe
a *separate*, more speculative hardware story — a hypothetical all-analog
front-end that could pair with a digital Loihi backend — not a claim that
Loihi itself is analog. This distinction wasn't explicit in the original
script's language ("Full Analog-Neuromorphic Pipeline"); now stated
explicitly in the module docstring and the `_print_frontend_breakdown()`
output ("SNN on Loihi 2 (digital) ← measured... Loihi 2 itself is not
analog").

This also retroactively clarifies §9-§11's reliability/noise-injection work:
that sweep models a hypothetical all-analog realization (Gm-C filters,
comparator thresholds, RC time-constant mismatch, crossbar conductance
variation) — a legitimate, separate research question from "does this run
correctly on real Loihi 2 hardware," which is what the Lava-based energy
measurement actually answers. Both are valid, but they're different claims
and shouldn't be conflated when the paper describes them.

### 12c. What got added — a cross-check, not a replacement

New `load_pytorch_side_synops()` and `_print_cross_check()` in
`compute_energy.py`: computes an independent SynOps estimate from this
codebase's own B15 instrumentation —

```
SynOps_pytorch = mean_input_events_per_trial * n_hidden
                 + mean_hidden_events_per_trial * n_output
```

— the same fan-out-weighted logic as the reliability sweep's energy
reasoning (§9), reading directly from `pipeline_params.json` (averaged
across a subject's folds). Reported side-by-side against Lava's measured
`synops_total_mean` per subject, with a ratio column.

**Rationale for keeping Lava primary, not switching to this:** Lava's
number is the real target framework's own accounting for real,
currently-deployable digital hardware — a stronger claim than anything
derived from PyTorch-side instrumentation in a different framework. The
cross-check exists to *validate* that number, not compete with it: rough
agreement is corroborating evidence before either number goes in the paper;
a large disagreement would mean investigating which measurement (or which
assumption feeding it) is off, before trusting either.

Verified the cross-check's arithmetic and its graceful skip-on-missing-data
behavior (folds trained before the B15 event-breakdown existed are logged
and skipped, not silently zero-filled or crashed on) with a mock
`pipeline_params.json` fixture — confirmed both before considering this
done.

### 12d. Also needed: a new field, `n_timesteps`

While building this, checked how many timesteps the spike tensor actually
has — the paper states `T_s=50` (`main.tex:386`), but
`tests/test_spike_snn.py:123`'s own assertion shows the real spike tensor
shape is `(1001, 288, total_features)` for BNCI2014-001 — **T is actually
the raw EEG sample count (~1001 at 250 Hz over ~4s), not a fixed 50.** This
is a previously-undiscovered paper-vs-code mismatch, and it directly affects
the paper's own `eq:energy` (which multplies by `T_s=50`, off by ~20x from
the real value if so). `n_timesteps` wasn't saved anywhere in
`pipeline_params.json` before now — added it (`pipeline.py`, extracted from
`spikes_tr.shape[0]`) so `compute_energy.py`'s dense-equivalent comparison
doesn't have to guess or hardcode it. **Not yet logged as a numbered
barrier in the paper's `TODO.md`** — should be, next time that file is
touched (candidate: B22, or folded into B8's existing entry since it's the
same equation).

### 12e. Cleanup decided alongside this

User: archive (not delete) all pre-existing `Results_*` directories and
loose scripts on Puhti before the next pull — including the *old*,
untracked `compute_energy.py`, since the new tracked version would
otherwise conflict with it on `git pull` (git won't silently overwrite an
untracked file that collides with an incoming tracked one).
