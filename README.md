# FBCSP-SNN Motor Imagery EEG Classifier

A GPU-accelerated pipeline that classifies motor imagery (MI) from EEG signals using
**Filter-Bank Common Spatial Patterns (FBCSP)** for feature extraction and a
**Spiking Neural Network (SNN)** for classification.
The pipeline targets neuromorphic hardware deployment — every operation maps to
hardware-friendly primitives (bandpass filters, matrix multiplies, comparators,
leaky integrators).

---

## Pipeline architecture

```
Raw EEG  (n_trials, n_channels, n_samples)
  │
  ├─ Per-fold adaptive band selection  [band_selection.py]
  │    Fisher ERD/ERS on training data only
  │    → K best bands from dense candidates (4–40 Hz, 4 Hz wide, 2 Hz step)
  │
  ├─ Bandpass filter bank  [preprocessing.py]
  │    Zero-phase Butterworth (order 4)
  │    → (n_trials, n_channels × n_bands, n_samples)
  │
  ├─ Pairwise CSP — dual-end extraction  [preprocessing.py]
  │    For every class pair: solve  Σ_A W = λ (Σ_A + Σ_B) W
  │    Take first m AND last m eigenvectors per pair
  │    → dict { pair → (n_trials, 2m × n_bands, n_samples) }
  │
  ├─ Z-normalisation  [preprocessing.py]
  │    Fit on training fold, apply to val/test
  │
  ├─ Adaptive-threshold spike encoding  [encoding.py]
  │    Delta-based, JIT-compiled inner loop
  │    → binary spike tensor  (n_timesteps, n_trials, n_features)
  │
  ├─ MIBIF feature selection  [mibif.py]
  │    mutual_info_classif on spike counts
  │    → prune to top-K% features
  │
  ├─ 2-layer LIF SNN  [model.py]
  │    Input → Linear → Dropout → LIF
  │          → Linear → Dropout → LIF → Output
  │    Output: population coding (N neurons per class)
  │
  ├─ Van Rossum loss  [losses.py]
  │    FFT convolution with causal exponential kernel
  │    MSE between filtered output spikes and filtered target spikes
  │
  └─ Winner-take-all decoding
       Sum spikes per class population → argmax
```

**4-class, 6 bands, m=2 (dual-end):**
- 6 pairs × 4 filters × 6 bands = **144 CSP features**
- MIBIF at 50 % → **72 features** fed to the SNN
- SNN output: 4 classes × 20 neurons = **80 output neurons**

---

## Datasets

| Dataset | Classes | Subjects | Channels | Sampling rate | Source |
|---|---|---|---|---|---|
| **BNCI2014_001** (primary) | 4 (LH, RH, feet, tongue) | 9 | 22 | 250 Hz | MOABB |
| PhysionetMI | 4 | 109 | 64 | 160 Hz | MOABB |
| Cho2017 | 2 | 52 | 64 | 512 Hz | MOABB |
| BNCI2015_001 | 2 | 12 | 13 | 512 Hz | MOABB |

BNCI2014_001: session 1 → train, session 2 → held-out test (~288 trials per session).
All other datasets: stratified 80/20 split with `random_state=42`.

---

## Install

**Requirements:** Python 3.10+, CUDA-capable GPU (optional but strongly recommended).

```bash
# Clone
git clone <repo-url>
cd fbcsp-snn-sgd-mi-classifier

# Create environment (conda)
conda create -n fbcsp python=3.10
conda activate fbcsp

# Or with venv
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
.\.venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt

# Verify GPU is detected
python -c "from fbcsp_snn import DEVICE; print('Device:', DEVICE)"
```

`requirements.txt`:
```
torch>=2.0
snntorch>=0.9
scipy>=1.10
scikit-learn>=1.3
moabb>=1.1
mne>=1.5
h5py>=3.9
matplotlib>=3.7
seaborn>=0.12
numpy>=1.24
```

> **First run** downloads BNCI2014_001 via MOABB (~400 MB total for all 9 subjects).
> Data is cached in `~/mne_data/` by default.

---

## Usage

### Train — one subject, all folds

```bash
python main.py train \
    --source moabb \
    --moabb-dataset BNCI2014_001 \
    --subject-id 1 \
    --adaptive-bands \
    --n-adaptive-bands 6 \
    --csp-components-per-band 4 \
    --epochs 1000 \
    --n-folds 10
```

### Train — static bands, single fold

```bash
python main.py train \
    --source moabb \
    --moabb-dataset BNCI2014_001 \
    --subject-id 1 \
    --no-adaptive-bands \
    --freq-bands "[(4,10),(10,14),(14,30)]" \
    --fold 0
```

### Train — all options

```
python main.py train --help
```

| Flag | Default | Description |
|---|---|---|
| `--source` | `moabb` | `moabb` or `hdf5` |
| `--moabb-dataset` | `BNCI2014_001` | Dataset name from registry |
| `--subject-id` | `1` | Subject index (1-indexed) |
| `--n-folds` | `10` | Number of CV folds |
| `--fold` | _(all)_ | Run only this fold (0-indexed) |
| `--adaptive-bands` / `--no-adaptive-bands` | adaptive | Band selection mode |
| `--n-adaptive-bands` | `6` | Bands to select |
| `--freq-bands` | `"[(4,8),(8,14),(14,30)]"` | Static bands (when `--no-adaptive-bands`) |
| `--csp-components-per-band` | `4` | Total CSP filters per band (2 from each end) |
| `--lambda-r` | `0.0001` | CSP covariance regularisation |
| `--base-thresh` | `0.001` | Spike encoding base threshold |
| `--adapt-inc` | `0.6` | Spike threshold increment per spike |
| `--decay` | `0.95` | Spike threshold decay per step |
| `--hidden-neurons` | `64` | SNN hidden layer width |
| `--population-per-class` | `20` | Output population neurons per class |
| `--beta` | `0.95` | LIF membrane decay factor |
| `--dropout-prob` | `0.5` | Dropout probability |
| `--lr` | `1e-3` | AdamW learning rate |
| `--weight-decay` | `0.1` | AdamW weight decay |
| `--epochs` | `1000` | Maximum training epochs |
| `--early-stopping-patience` | `100` | Epochs without val-acc improvement before stop |
| `--early-stopping-warmup` | `100` | Minimum epochs before early stopping activates |
| `--spiking-prob` | `0.7` | Target spike probability for Van Rossum targets |
| `--feature-selection-method` | `mibif` | `mibif` or `none` |
| `--feature-percentile` | `50.0` | Percentage of features to keep |
| `--results-dir` | `Results` | Root output directory |

### Inference — saved fold

```bash
python main.py infer \
    --source moabb \
    --moabb-dataset BNCI2014_001 \
    --subject-id 1 \
    --fold 0
```

Loads `Results/Subject_1/fold_0/` artifacts (model, CSP, z-norm, MIBIF),
runs the full preprocessing chain on the held-out test set, logs FP32 and
INT8-simulated accuracy, and saves confusion matrix PNGs.

### Aggregate — collect fold results

```bash
python main.py aggregate \
    --source moabb \
    --moabb-dataset BNCI2014_001 \
    --subject-id 1 \
    --n-folds 10
```

Reads each `fold_K/pipeline_params.json`, writes `Results/Subject_1/summary.csv`
(per-fold metrics + mean row), and saves aggregated confusion matrices.

### Cross-subject analysis

```bash
python analyze_results.py \
    --results-dir Results \
    --subjects 1 2 3 4 5 6 7 8 9
```

Reads every `summary.csv`, prints a cross-subject accuracy table to stdout,
and saves `Results/cross_subject_accuracy.png`.

```
Cross-Subject Accuracy Summary — FBCSP-SNN (BNCI2014_001)
--------------------------------------------------------------
Subject    Test FP32 (%)   Test INT8 (%)   Val FP32 (%)   Folds
--------------------------------------------------------------
S1         85.3 ± 2.1      85.8 ± 2.0      84.1 ± 2.4        10
...
--------------------------------------------------------------
GRAND      70.5 ± 14.2     70.3 ± 14.1                       90
--------------------------------------------------------------
  Baseline target : 64.8% (FP32, static bands, 22 CSP comps)
  Target          : 70.0%+
  vs. baseline    : +5.7 pp
```

---

## HPC — CSC Puhti (SLURM array job)

Train all 9 subjects in parallel, one V100 per subject:

```bash
# Edit run_puhti_array.sh: set --account=<YOUR_PROJECT>
mkdir -p logs
sbatch run_puhti_array.sh       # submits array job 1-9
```

Each task runs for up to 4 hours and saves artifacts to `Results/Subject_N/`.
After all tasks finish, run aggregation as a dependent job or manually:

```bash
for S in 1 2 3 4 5 6 7 8 9; do
    python main.py aggregate --subject-id $S --n-folds 10
done
python analyze_results.py --results-dir Results
```

---

## Output artifacts

```
Results/
  Subject_N/
    fold_K/
      best_model.pt          PyTorch state dict (best val accuracy)
      csp_filters.pkl        PairwiseCSP instance
      znorm.pkl              ZNormaliser instance
      mibif.pkl              MIBIFSelector instance
      pipeline_params.json   Per-fold bands, metrics, hyperparameters
      band_selection.png     Fisher ratio curve + selected band highlights
      spike_propagation.png  Feature × time spike raster (4 training trials)
      neuron_traces.png      Output LIF membrane potential + spike overlay
      weight_histograms.png  FP32 vs INT8-sim weight distributions
      confusion_fp32.png     Test-set confusion matrix (FP32)
      confusion_int8.png     Test-set confusion matrix (INT8-sim)
    summary.csv              Per-fold metrics + mean row
    confusion_aggregate_fp32.png
    confusion_aggregate_int8.png
  cross_subject_accuracy.png
```

---

## Baseline results (previous pipeline)

Static 3-band FBCSP, 22 CSP components, std-based feature selection:

| Subject | FP32 Acc | INT8 Acc |
|---|---|---|
| S1 | 85.3% | 86.0% |
| S2 | 45.4% | 45.8% |
| S3 | 73.8% | 73.6% |
| S4 | 54.3% | 53.3% |
| S5 | 44.4% | 44.8% |
| S6 | 51.7% | 51.5% |
| S7 | 76.6% | 75.4% |
| S8 | 79.6% | 79.5% |
| S9 | 72.4% | 70.9% |
| **Mean** | **64.8%** | **64.5%** |

**Target:** 70%+ mean FP32 accuracy with the adaptive pipeline.
Main opportunity: weak subjects S2, S4, S5, S6 (near 25% chance level).

---

## Design constraints

- **No data leakage.** Band selection, CSP fitting, z-norm statistics, and MIBIF
  are all computed exclusively on the training split of each CV fold.
  Val/test data are transformed using training-derived parameters only.
- **Labels 1-indexed** throughout the data pipeline; converted to 0-indexed only
  when constructing PyTorch tensors for the SNN.
- **Deterministic splits.** `StratifiedKFold(random_state=42)` for multi-session
  datasets; `StratifiedShuffleSplit(random_state=42)` for single-session.
- **`torch.compile` guarded** behind a Triton import check (Triton is Linux-only;
  CUDA Graphs conflict with snnTorch's `init_leaky()`).

---

## Project structure

```
fbcsp-snn-sgd-mi-classifier/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── main.py                    CLI entry point (train / infer / aggregate)
├── analyze_results.py         Cross-subject summary analysis
├── run_puhti_array.sh         SLURM array job for CSC Puhti
├── fbcsp_snn/
│   ├── __init__.py            DEVICE detection, logger, CUDA config
│   ├── config.py              Config dataclass + argparse
│   ├── data.py                HDF5 .mat loader
│   ├── datasets.py            MOABB registry + loader
│   ├── band_selection.py      Fisher ERD/ERS adaptive band selection
│   ├── preprocessing.py       Butterworth filter bank, PairwiseCSP, ZNormaliser
│   ├── encoding.py            Adaptive-threshold spike encoding (JIT)
│   ├── mibif.py               Mutual information feature selection
│   ├── model.py               SNNClassifier (2-layer LIF, population coding)
│   ├── losses.py              Van Rossum loss (FFT convolution)
│   ├── training.py            Training loop, AdamW, AMP, early stopping
│   ├── evaluation.py          Accuracy, confusion matrix
│   ├── quantization.py        Simulated INT8 (symmetric per-tensor)
│   ├── visualization.py       All plot functions
│   └── pipeline.py            run_train, run_infer, run_aggregate
└── tests/
    ├── test_band_csp.py       Band selection + filter bank + CSP shapes
    ├── test_spike_snn.py      Encoding + SNN forward/backward pass
    └── test_cv_pipeline.py    3-fold CV integration test
```

---

## Running tests

```bash
# Full test suite
pytest tests/ -v

# Individual integration tests (each downloads data on first run)
python tests/test_band_csp.py
python tests/test_spike_snn.py
python tests/test_cv_pipeline.py    # ~8 min on V100 (T_MAX=100)
```

---

## Training pipeline — step by step

This section describes exactly what happens inside `run_train` for a single CV fold.

### 0. Data loading

```
load_moabb(dataset, subject_id)
  → X_train (n_trials, n_channels, n_samples)   — session 1
  → X_test  (n_trials, n_channels, n_samples)   — session 2 (held-out, never touched during training)
  → y_train, y_test  — class labels, 1-indexed (1 … C)
```

- BNCI2014_001: session 1 → training pool, session 2 → held-out test (~288 trials each).
- Data is downloaded via MOABB on first run and cached in `~/mne_data/`.

### 1. Cross-validation split

```
StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
  → training indices  (X_f_tr, y_f_tr)
  → validation indices (X_f_val, y_f_val)
```

- Only session-1 data is split. Session-2 (test) is always held out.
- Labels are converted to 0-indexed (`y - 1`) for PyTorch tensors at this point.

### 2. Adaptive band selection  *(fit on training split only)*

```
select_bands(X_f_tr, y_f_tr, sfreq, n_bands, bandwidth, step, band_range)
  → bands: List[(lo, hi)]   e.g. [(8,12), (12,16), (18,22), ...]
  → fisher_freqs, fisher_curve  (for plotting)
```

- Welch PSD computed per trial, averaged across channels.
- Fisher discriminant ratio calculated at each frequency bin.
- Dense candidates: 4 Hz wide, 2 Hz step, over the configured range (default 4–40 Hz).
- Greedy selection of top-K candidates with a max 50% overlap constraint.
- Saves `band_selection.png` showing the Fisher curve and selected bands.

### 3. Bandpass filter bank

```
apply_filter_bank(X, bands, sfreq, order=4)
  → X_bands: (n_trials, n_channels, n_bands, n_samples)
```

Applied independently to training, validation, and test splits using the same bands selected in step 2.

### 4. Pairwise CSP with Euclidean Alignment  *(fit on training split only)*

```
PairwiseCSP(m=2, lambda_r=0.0001, euclidean_alignment=True, riemannian_mean=True)
  .fit(X_bands_tr, y_f_tr)
  .transform(X_bands_*)
  → proj: dict { (class_a, class_b) → (n_trials, 2m × n_bands, n_samples) }
```

- **Euclidean Alignment (EA):** per-band whitening `X̃ = R^{-1/2} X` where R is the
  Riemannian (Fréchet) mean covariance of all training trials. Reduces cross-trial
  non-stationarity before CSP.
- **Dual-end extraction:** first m AND last m eigenvectors per class pair.
  Default m=2 → 4 filters per pair.
- 4 classes → 6 pairs × 4 filters × 6 bands = **144 features total**.
- EA whitener stored in the `PairwiseCSP` pickle for reproducible inference.

### 5. Z-normalisation  *(fit on training split only)*

```
ZNormaliser().fit_transform(X_concat_tr) → X_norm_tr
ZNormaliser().transform(X_concat_val)    → X_norm_val
ZNormaliser().transform(X_concat_te)     → X_norm_te
```

Mean and std computed per feature on the training split. Same statistics applied to val and test.

### 6. Adaptive-threshold spike encoding

```
encode_tensor(X_norm, base_thresh, adapt_inc, decay)
  → spikes: (T, n_trials, n_features)   binary tensor
```

- Delta-based encoder: spike fires when `|signal[t] - signal[t-1]| > threshold`.
- After a spike: `threshold += adapt_inc`. Every timestep: `threshold *= decay`.
- Inner loop JIT-compiled with `@torch.jit.script`.

### 7. MIBIF feature selection  *(fit on training spike counts only)*

```
MIBIFSelector(feature_percentile=50.0).fit_transform(spikes_tr, y_tr)
  → spikes_tr  (T, n_trials, n_selected)   top-K% features
```

- `mutual_info_classif` on per-feature spike counts (sum over time).
- Default: keep top 50% → 72 features from 144.
- Same feature mask applied to val and test spikes.

### 8. SNN construction and training

```
SNNClassifier(n_input, n_hidden=64, n_classes, population_per_class=20,
              beta=0.95, dropout_prob=0.5)
  → 2-layer LIF network
  → n_output = n_classes × population_per_class  (80 neurons for 4-class)
```

Training loop (`train_fold`):

| Stage | Detail |
|---|---|
| Optimiser | AdamW, lr=1e-3, weight_decay=0.1 |
| Loss | Van Rossum (FFT convolution, τ=10 steps) |
| Target spikes | Correct-class population fires at `spiking_prob=0.7`; wrong classes fire at 0 |
| AMP | `torch.autocast` + `GradScaler` on CUDA (float16 stability) |
| Early stopping | Monitor val accuracy; patience=100, warmup=100 epochs |
| Checkpoint | Best model state saved to `best_model.pt` on val-acc improvement |

### 9. Evaluation

```
evaluate_model(model, spikes_test, y_test_0, device)
  → test_acc_fp32, test_preds_fp32
```

- Winner-take-all decoding: sum output spikes per class population → argmax.
- Repeated with `quantize_model(model, bits=8)` for INT8-simulated accuracy.

### 10. Artifact saving

All outputs written to `Results/Subject_N/fold_K/`:

| File | Contents |
|---|---|
| `best_model.pt` | PyTorch state dict at best validation accuracy |
| `csp_filters.pkl` | Fitted `PairwiseCSP` (EA whiteners + eigenvectors) |
| `znorm.pkl` | Fitted `ZNormaliser` (mean + std per feature) |
| `mibif.pkl` | Fitted `MIBIFSelector` (selected feature indices) |
| `pipeline_params.json` | Bands, metrics, hyperparameters |
| `band_selection.png` | Fisher curve + selected band highlights |
| `spike_propagation.png` | Spike raster for 4 training trials |
| `neuron_traces.png` | Output LIF membrane potential + spike overlay |
| `weight_histograms.png` | FP32 vs INT8-sim weight distributions |
| `confusion_fp32.png` | Test-set confusion matrix (FP32) |
| `confusion_int8.png` | Test-set confusion matrix (INT8-sim) |

---

## Inference pipeline — step by step

`run_infer` re-applies the full trained pipeline to the held-out test set using only
saved artifacts. No retraining, no re-fitting.

### Prerequisites

A completed training run for the target fold:

```
Results/Subject_N/fold_K/
  ├── pipeline_params.json   ← bands + n_input_features read from here
  ├── csp_filters.pkl
  ├── znorm.pkl
  ├── mibif.pkl              (if feature selection was used)
  └── best_model.pt
```

### Steps

**1. Load saved parameters**
```python
params = json.load("pipeline_params.json")
bands   = params["bands"]            # selected frequency bands
n_input = params["n_input_features"] # SNN input size
```

**2. Load test data**
```
load_moabb(dataset, subject_id) → X_test, y_test
```
Only the test split is used; training data is not reloaded.

**3. Load preprocessing objects**
```python
csp   = pickle.load("csp_filters.pkl")   # PairwiseCSP with fitted EA whiteners
znorm = pickle.load("znorm.pkl")          # ZNormaliser with fitted mean/std
mibif = pickle.load("mibif.pkl")          # MIBIFSelector with fitted feature mask
```

**4. Preprocessing chain** *(same transforms as training, using training-derived parameters)*
```
apply_filter_bank(X_test, bands, sfreq, order=4)
  → csp.transform(X_bands)       # EA whitening + CSP projection (training whitener/filters)
  → ZNormaliser.transform(...)   # z-norm with training mean/std
  → encode_tensor(...)           # adaptive-threshold spike encoding
  → MIBIFSelector.transform(...) # keep training-selected features only
  → spikes: (T, n_trials, n_features)
```

**5. Load model**
```python
model = SNNClassifier(n_input, n_hidden, n_classes, ...)
model.load_state_dict(torch.load("best_model.pt"))
model.eval()                        # dropout_prob=0.0 at inference
```

**6. FP32 inference**
```
evaluate_model(model, spikes, y_test_0, device)
  → test_acc_fp32, test_preds_fp32
```

**7. INT8-simulated inference**
```
quantize_model(model, bits=8)       # symmetric per-tensor weight quantisation
evaluate_model(model_int8, ...)
  → test_acc_int8, test_preds_int8
```

**8. Output**
- Logs `FP32 X.X%  INT8 X.X%` to stdout.
- Saves `infer_confusion_fp32.png` and `infer_confusion_int8.png` to `fold_K/`.
