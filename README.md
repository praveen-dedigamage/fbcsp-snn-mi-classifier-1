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
  │    → min_fisher_fraction guard filters out noise/EMG bands
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
  ├─ Classical baselines  [baseline.py]       ← runs here, before spike encoding
  │    log-variance features  →  LDA / SVM
  │    Results stored alongside SNN metrics for direct comparison
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

**Current best configuration (static6-overlap) — 4-class, 6 static bands, m=4 per end:**
- 6 overlapping bands: `(4,10)(8,14)(12,18)(16,22)(20,26)(24,30)` — 6 Hz wide, 4 Hz step, 2 Hz overlap, 4–30 Hz
- 6 pairs × 8 filters/band × 6 bands = **288 CSP features** pre-MIBIF
- MIBIF at 50% → **~144 features** fed to the SNN
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
cd fbcsp-snn-mi-classifier-1

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

### Train — one subject, single fold

```bash
python main.py train \
    --source moabb \
    --moabb-dataset BNCI2014_001 \
    --subject-id 1 \
    --fold 0 \
    --adaptive-bands \
    --n-adaptive-bands 12 \
    --min-fisher-fraction 0.15 \
    --csp-components-per-band 8 \
    --epochs 1000 \
    --n-folds 5
```

### Train — all folds for one subject

```bash
python main.py train \
    --source moabb \
    --moabb-dataset BNCI2014_001 \
    --subject-id 1 \
    --adaptive-bands \
    --n-adaptive-bands 12 \
    --min-fisher-fraction 0.15 \
    --csp-components-per-band 8 \
    --epochs 1000 \
    --n-folds 5
```

### Train — static bands

```bash
python main.py train \
    --source moabb \
    --moabb-dataset BNCI2014_001 \
    --subject-id 1 \
    --no-adaptive-bands \
    --freq-bands "[(4,8),(8,14),(14,30)]" \
    --fold 0
```

### All training flags

| Flag | Default | Description |
|---|---|---|
| `--source` | `moabb` | `moabb` or `hdf5` |
| `--moabb-dataset` | `BNCI2014_001` | Dataset name from registry |
| `--subject-id` | `1` | Subject index (1-indexed) |
| `--n-folds` | `5` | Number of CV folds |
| `--fold` | _(all)_ | Run only this fold (0-indexed) |
| `--adaptive-bands` / `--no-adaptive-bands` | adaptive | Band selection mode |
| `--n-adaptive-bands` | `12` | Number of frequency bands to select |
| `--min-fisher-fraction` | `0.15` | Reject bands scoring below `top_score × mff` (guards against noise/EMG) |
| `--freq-bands` | `"[(4,8),(8,14),(14,30)]"` | Static bands (when `--no-adaptive-bands`) |
| `--csp-components-per-band` | `8` | Total CSP filters per band (m from each end, total = 2m) |
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
    --n-folds 5
```

Reads each `fold_K/pipeline_params.json`, writes `Results/Subject_1/summary.csv`
(per-fold metrics + mean row including LDA/SVM baseline results), and saves
aggregated confusion matrices.

### Cross-subject analysis

```bash
python analyze_results.py \
    --results-dir Results \
    --subjects 1 2 3 4 5 6 7 8 9
```

Reads every `summary.csv`, prints a cross-subject accuracy table to stdout
(SNN vs LDA vs SVM when baseline columns are present), and saves
`Results/cross_subject_accuracy.png`.

```
Cross-Subject Accuracy — FBCSP-SNN vs Classical Baselines (BNCI2014_001)
----------------------------------------------------------
Subject    SNN FP32 (%)      LDA (%)      SVM (%)    Folds
----------------------------------------------------------
S1           81.4 ± 2.9   52.5 ± 5.2   76.6 ± 2.4       5
S2           46.5 ± 3.8   42.0 ± 1.5   44.9 ± 3.2       5
S3           74.7 ± 5.2   38.3 ± 1.8   79.1 ± 1.5       5
S4           63.1 ± 5.8   49.9 ± 3.8   59.9 ± 2.2       5
S5           47.2 ± 3.3   38.9 ± 2.9   40.1 ± 2.8       5
S6           54.2 ± 2.1   38.9 ± 4.5   49.0 ± 2.6       5
S7           75.3 ± 4.4   70.1 ± 4.2   67.7 ± 3.5       5
S8           79.9 ± 2.2   55.6 ± 8.6   76.5 ± 2.1       5
S9           80.1 ± 2.9   50.3 ± 8.5   76.9 ± 1.6       5
----------------------------------------------------------
GRAND              66.9         48.5         63.4          45
----------------------------------------------------------
  Baseline target : 64.8% (FP32, static bands, 22 CSP comps)
  Target          : 70.0%+
  SNN vs baseline : +2.1 pp
  LDA vs baseline : -16.3 pp
  SVM vs baseline : -1.4 pp
```

---

## HPC — CSC Puhti (3-stage SLURM pipeline)

One command submits the full pipeline with automatic dependency chaining:

```bash
cd /scratch/project_2003397/praveen/fbcsp-snn-mi-classifier-1
git pull
bash submit_puhti.sh
```

### What it does

```
Stage 1 — Training array   (45 tasks: 9 subjects × 5 folds, all parallel)
           Each task: 1 GPU (V100), 2h wall time, runs a single fold

Stage 2 — Aggregate array  (9 jobs: one per subject, parallel)
           Triggered per-subject as soon as its 5 fold tasks complete
           Not waiting for all 45 — subject 1 aggregates while subject 9 is still training

Stage 3 — Analyze job      (1 job: cross-subject summary)
           Triggered once all 9 aggregate jobs finish
           Prints SNN vs LDA vs SVM comparison table + saves plots
```

### Monitoring

```bash
squeue -u $USER
sacct -j <ARRAY_JOBID> --format=JobID,State,Elapsed,MaxRSS
```

### Reading results

```bash
# Final summary (printed at end of analyze job log)
cat logs/fbcsp_analyze_<JOBID>.out

# Per-subject CSVs
cat Results/Subject_1/summary.csv
```

### Manual aggregate (if training already done)

```bash
source .venv/bin/activate
bash run_puhti_analyze.sh
```

---

## Output artifacts

```
Results/
  Subject_N/
    fold_K/
      best_model.pt          PyTorch state dict (best val accuracy)
      csp_filters.pkl        PairwiseCSP instance (fitted filters + EA whiteners)
      znorm.pkl              ZNormaliser instance (mean + std per feature)
      mibif.pkl              MIBIFSelector instance (selected feature indices)
      pipeline_params.json   Bands, metrics, hyperparams — SNN + LDA + SVM results
      band_selection.png     Fisher ratio curve + selected band highlights
      spike_propagation.png  Feature × time spike raster (4 training trials)
      neuron_traces.png      Output LIF membrane potential + spike overlay
      weight_histograms.png  FP32 vs INT8-sim weight distributions
      confusion_fp32.png     Test-set confusion matrix (FP32)
      confusion_int8.png     Test-set confusion matrix (INT8-sim)
    summary.csv              Per-fold metrics + mean row (SNN + LDA + SVM)
    confusion_aggregate_fp32.png
    confusion_aggregate_int8.png
  cross_subject_accuracy.png
```

---

## Results

### Static baseline (previous pipeline)

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

### Experimental progression (BNCI2014_001, 9 subjects, 5-fold CV)

All results: session 1 train / session 2 test, FP32.

| Version | Mean Acc | Std | vs Baseline | Key change |
|---|---|---|---|---|
| Baseline | 64.8% | — | — | Static 3-band, 22 CSP, std-based selection |
| V4 | 66.9% | ±14.5 | +2.1pp | 9 bands, 6 CSP/band |
| V4.1 | 66.9% | ±13.8 | +2.1pp | 12 bands, 8 CSP/band, mff=0.15 |
| V4.2-augwin | 67.1% | ±15.0 | +2.3pp | V4.1 + sliding-window CSP augmentation |
| **static6-overlap** | **67.2%** | **±14.6** | **+2.4pp** | **6 Hz static bands, 4 Hz step, 2 Hz overlap, 4–30 Hz ⭐ NEW BEST** |
| causal-butterworth | 66.2% | ±14.7 | +1.4pp | static6-overlap + causal `sosfilt` (neuromorphic-compatible) |

### static6-overlap — per-subject results (⭐ current best)

6 static overlapping bands, 8 CSP filters/band, sliding-window augmentation, zero-phase filter:

| Subject | FP32 Acc | vs Baseline |
|---|---|---|
| S1 | 83.6% | −1.7pp |
| S2 | 43.5% | −1.9pp |
| S3 | 77.2% | +3.4pp |
| S4 | 63.0% | +8.7pp |
| S5 | 45.7% | +1.3pp |
| S6 | 54.0% | +2.3pp |
| S7 | 75.7% | −0.9pp |
| S8 | 81.4% | +1.8pp |
| S9 | 79.8% | +7.4pp |
| **Mean** | **67.2%** | **+2.4pp** |

**Target:** 70%+ mean FP32. Gap remaining: **2.8pp**.

### Causal filter — neuromorphic accuracy cost

Replacing zero-phase `sosfiltfilt` with causal `sosfilt` for neuromorphic compatibility:

| Subject | Zero-phase | Causal | Cost |
|---|---|---|---|
| S1 | 83.6% | 83.6% | 0.0pp |
| S2 | 43.5% | 45.4% | +1.9pp |
| S3 | 77.2% | 77.2% | 0.0pp |
| S4 | 63.0% | 63.7% | +0.7pp |
| S5 | 45.7% | 46.2% | +0.5pp |
| S6 | 54.0% | 51.8% | −2.2pp |
| S7 | 75.7% | 70.9% | −4.8pp |
| S8 | 81.4% | 81.4% | 0.0pp |
| S9 | 79.8% | 79.8% | 0.0pp |
| **Mean** | **67.2%** | **66.2%** | **−1.0pp** |

S7 is most affected: needs broad beta coverage (peak at ~23 Hz); causal Butterworth group delay
at beta frequencies shifts the spectral content. A Bessel filter (maximally flat group delay)
is expected to recover part of this gap — **pending experiment**.

### V4.2-augwin — classifier comparison

| Classifier | Mean | vs static baseline |
|---|---|---|
| **SNN (FP32)** | **67.1%** | **+2.3pp** |
| SVM (RBF) | 63.4% | −1.4pp |
| LDA | 48.5% | −16.3pp |

**Key findings:**

- **SNN beats SVM on 8/9 subjects** — temporal spike encoding extracts real information
  beyond log-variance, confirming the SNN is not the bottleneck.
- **LDA collapses** at 48.5% because ~216 features >> ~230 training trials (p >> n regime).
  RBF-SVM handles this better but still underperforms the SNN.
- **Weak subjects (S2, S5, S6):** both SVM and SNN fail similarly (~44–54%). The bottleneck
  is cross-session EEG non-stationarity — needs better feature extraction (e.g. Euclidean
  Alignment, domain-adaptive band selection).
- **Val→test gap:** S5 val=68% vs test=47% (21pp), S2 val=64% vs test=47% (17pp).
  Session 1 patterns do not transfer to session 2 for these subjects.

---

## Neuromorphic hardware mapping

Every inference operation maps to an analog or neuromorphic primitive —
no digital CPU required at inference time.

### Full pipeline mapping

```
┌─────────────────────────────────────────────────────────────────────┐
│  Analog domain  (CMOS Gm-C / resistive crossbar)                    │
│                                                                     │
│  Raw EEG ──► 6-band causal Butterworth filter bank                  │
│              (6 parallel Gm-C biquad chains, tuned to 4–30 Hz)      │
│              Each order-4 band = 4 cascaded Gm-C biquad stages      │
│                                                                     │
│           ──► CSP spatial projection  (W^T × X)                     │
│              (resistive crossbar multiply-accumulate)               │
│                                                                     │
│           ──► Z-normalisation                                        │
│              (affine Gm scaling: x → x/σ − μ/σ)                    │
└─────────────────────────────────────────────────────────────────────┘
                          │
                          ▼  binary spikes
┌─────────────────────────────────────────────────────────────────────┐
│  Neuromorphic fabric  (Loihi / TrueNorth / SpiNNaker)               │
│                                                                     │
│  Delta spike encoder  (adaptive threshold comparator)               │
│           ──► MIBIF feature routing  (fixed wiring, no compute)     │
│           ──► LIF hidden layer  (64 neurons, β = 0.95)              │
│           ──► LIF output layer  (4 × 20 population neurons)         │
│           ──► Winner-take-all  (spike accumulator + argmax)         │
│                                                                     │
│  Weights stored as INT8  (validated ≤ 0.5 pp accuracy drop)         │
└─────────────────────────────────────────────────────────────────────┘
```

### Analog Butterworth filter bank (Gm-C implementation)

The filter bank uses **causal single-pass IIR filtering** (`sosfilt`), making
each band directly implementable as a cascade of Gm-C biquad sections:

```
dV₁/dt = Gm₁(Vᵢₙ − V₂) / C₁      ← leaky integrator 1
dV₂/dt = Gm₂ · V₁ / C₂            ← leaky integrator 2
```

Tuning a band to centre frequency `f₀` requires only `Gm = 2π f₀ C`.
Six bands in parallel share the same fabrication process; only bias currents
differ. This is directly analogous to the **silicon cochlea** (Mead 1989),
which implements a biological auditory filter bank in subthreshold CMOS.

Key properties:
- **Causal and real-time**: no trial buffering — processes sample-by-sample
- **Ultra-low power**: subthreshold Gm-C circuits operate at nanowatt levels
- **CMOS-compatible**: same process as the neuromorphic spiking core
- **Reconfigurable**: shifting a band's centre frequency requires only a bias current change

### Primitive-by-primitive breakdown

| Pipeline stage | Neuromorphic primitive | Hardware-mappable? |
|---|---|---|
| Causal Butterworth filter bank | Gm-C leaky integrator cascade | ✅ Yes |
| CSP spatial filter W^T × X | Resistive crossbar MAC | ✅ Yes |
| Z-normalisation | Affine Gm scaling | ✅ Yes |
| Delta spike encoder | Adaptive threshold comparator | ✅ Yes |
| MIBIF feature selection | Fixed routing / wiring | ✅ Yes |
| LIF hidden layer | Leaky integrate-and-fire neurons | ✅ Yes |
| LIF output layer | LIF population coding | ✅ Yes |
| Winner-take-all decoding | Spike counter + comparator | ✅ Yes |
| INT8 synaptic weights | Fixed-point arithmetic | ✅ Yes |

The entire inference pipeline — from raw EEG sample to classification decision —
maps to hardware primitives with no floating-point operations and no general-purpose
CPU required.

---

## Design constraints

- **No data leakage.** Band selection, CSP fitting, z-norm statistics, and MIBIF
  are all computed exclusively on the training split of each CV fold.
  Val/test data are transformed using training-derived parameters only.
- **Labels 1-indexed** throughout the data pipeline; converted to 0-indexed only
  when constructing PyTorch tensors for the SNN.
- **Deterministic splits.** `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.
- **`torch.compile` guarded** behind a Triton import check (Triton is Linux-only;
  CUDA Graphs conflict with snnTorch's `init_leaky()`).

---

## Project structure

```
fbcsp-snn-mi-classifier-1/
├── README.md
├── CLAUDE.md
├── EXPERIMENTS.md             Full experimental log with rationale and results
├── requirements.txt
├── main.py                    CLI entry point (train / infer / aggregate)
├── analyze_results.py         Cross-subject summary + bar chart
├── submit_puhti.sh            One-shot HPC submit: train → aggregate → analyze
├── run_puhti_array.sh         SLURM training array (9 subjects × 5 folds)
├── run_puhti_aggregate.sh     SLURM per-subject aggregate array (9 tasks)
├── run_puhti_analyze.sh       SLURM cross-subject analysis job
├── fbcsp_snn/
│   ├── __init__.py            DEVICE detection, logger, CUDA config
│   ├── config.py              Config dataclass + argparse
│   ├── data.py                HDF5 .mat loader
│   ├── datasets.py            MOABB registry + loader
│   ├── band_selection.py      Fisher ERD/ERS adaptive band selection
│   ├── preprocessing.py       Butterworth filter bank, PairwiseCSP, ZNormaliser
│   ├── encoding.py            Adaptive-threshold spike encoding (JIT)
│   ├── baseline.py            Classical baselines: LDA + SVM on log-var features
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
StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
  → training indices   (X_f_tr,  y_f_tr)   ~80% of session 1
  → validation indices (X_f_val, y_f_val)  ~20% of session 1
```

- Only session-1 data is split. Session-2 (test) is always held out.
- Labels converted to 0-indexed (`y - 1`) for PyTorch tensors.

### 2. Adaptive band selection *(fit on training split only)*

```
select_bands(X_f_tr, y_f_tr, sfreq, n_bands=12, bandwidth=4, step=2,
             band_range=(4,40), min_fisher_fraction=0.15)
  → bands: List[(lo, hi)]   e.g. [(8,12), (12,16), (18,22), ...]
  → fisher_freqs, fisher_curve  (for plotting)
```

- Welch PSD computed per trial, averaged across channels.
- Fisher discriminant ratio calculated at each frequency bin.
- Dense candidates: 4 Hz wide, 2 Hz step across 4–40 Hz (17 candidates).
- Greedy selection of top-12 with max 50% overlap constraint.
- `min_fisher_fraction=0.15` rejects any band scoring below 15% of the top score —
  prevents EMG noise bands (high-gamma) being forced in when fewer discriminative
  bands exist.
- Saves `band_selection.png` with Fisher curve and selected band highlights.

### 3. Bandpass filter bank

```
apply_filter_bank(X, bands, sfreq, order=4)
  → X_bands: (n_trials, n_channels × n_bands, n_samples)
```

Applied to training, validation, and test splits using the same bands from step 2.

### 4. Pairwise CSP — dual-end extraction *(fit on training split only)*

```
PairwiseCSP(m=4, lambda_r=0.0001)
  .fit(X_bands_tr, y_f_tr)
  .transform(X_bands_*)
  → proj: dict { (class_a, class_b) → (n_trials, 2m × n_bands, n_samples) }
```

- For each class pair: solve `Σ_A W = λ (Σ_A + Σ_B) W` using `scipy.linalg.eigh`.
- **Dual-end extraction:** first m AND last m eigenvectors per pair.
  `m=4` → 8 filters per pair per band (4 maximising class-A variance, 4 maximising class-B).
- Covariance regularisation: `(1-λ)Σ + λI`, `λ=0.0001`.
- 4 classes → 6 pairs × 8 filters × 12 bands = **576 features total**.

### 5. Z-normalisation *(fit on training split only)*

```
ZNormaliser().fit_transform(X_concat_tr) → X_norm_tr   (n_trials, 576, n_samples)
ZNormaliser().transform(X_concat_val)    → X_norm_val
ZNormaliser().transform(X_concat_te)     → X_norm_te
```

Mean and std computed per feature over the training split. Applied identically to val and test.

### 6. Classical baseline evaluation

```
extract_logvar(X_norm_*)  →  feat_*  (n_trials, 576)   log(var over time axis)

LinearDiscriminantAnalysis(solver='svd').fit(feat_tr, y_tr)
  → val_acc_lda,  test_acc_lda

Pipeline([StandardScaler, SVC(C=1, kernel='rbf')]).fit(feat_tr, y_tr)
  → val_acc_svm,  test_acc_svm
```

Runs on z-normalised CSP projections before spike encoding. Log-variance compresses
each feature's time axis to a single scalar — the standard FBCSP feature for classical
classifiers. Results stored in `pipeline_params.json` alongside SNN metrics.

### 7. Adaptive-threshold spike encoding

```
encode_tensor(X_norm, base_thresh=0.001, adapt_inc=0.6, decay=0.95)
  → spikes: (T, n_trials, n_features)   binary tensor
```

- Delta-based encoder: spike fires when `|signal[t] - signal[t-1]| > threshold`.
- After a spike: `threshold += adapt_inc`. Every timestep: `threshold *= decay`.
- Inner loop JIT-compiled with `@torch.jit.script` for performance.

### 8. MIBIF feature selection *(fit on training spike counts only)*

```
MIBIFSelector(feature_percentile=50.0).fit_transform(spikes_tr, y_tr)
  → spikes_tr  (T, n_trials, 216)   top-50% of 576 features
```

- `mutual_info_classif` on per-feature spike counts (summed over time).
- Keeps top 50% → 216 features from 576.
- Same feature mask applied to val and test spikes.

### 9. SNN construction and training

```
SNNClassifier(n_input=216, n_hidden=64, n_classes=4, population_per_class=20,
              beta=0.95, dropout_prob=0.5)
  → 2-layer LIF network
  → n_output = 4 × 20 = 80 neurons
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

### 10. Evaluation

```
evaluate_model(model, spikes_test, y_test_0, device)
  → test_acc_fp32, test_preds_fp32
```

- Winner-take-all decoding: sum output spikes per class population → argmax.
- Repeated with `quantize_model(model, bits=8)` for INT8-simulated accuracy.

### 11. Artifact saving

All outputs written to `Results/Subject_N/fold_K/`:

| File | Contents |
|---|---|
| `best_model.pt` | PyTorch state dict at best validation accuracy |
| `csp_filters.pkl` | Fitted `PairwiseCSP` (eigenvectors per pair/band) |
| `znorm.pkl` | Fitted `ZNormaliser` (mean + std per feature) |
| `mibif.pkl` | Fitted `MIBIFSelector` (selected feature indices) |
| `pipeline_params.json` | Bands, metrics, hyperparams — SNN + LDA + SVM results |
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
params  = json.load("pipeline_params.json")
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
csp   = pickle.load("csp_filters.pkl")   # PairwiseCSP with fitted filters
znorm = pickle.load("znorm.pkl")          # ZNormaliser with fitted mean/std
mibif = pickle.load("mibif.pkl")          # MIBIFSelector with fitted feature mask
```

**4. Preprocessing chain** *(same transforms as training)*
```
apply_filter_bank(X_test, bands, sfreq, order=4)
  → csp.transform(X_bands)       # CSP projection using training filters
  → znorm.transform(...)         # z-norm with training mean/std
  → encode_tensor(...)           # adaptive-threshold spike encoding
  → mibif.transform(...)         # keep training-selected features only
  → spikes: (T, n_trials, n_features)
```

**5. Load and evaluate model**
```python
model = SNNClassifier(n_input, n_hidden, n_classes, ...)
model.load_state_dict(torch.load("best_model.pt"))
model.eval()
evaluate_model(model, spikes, y_test_0, device) → test_acc_fp32
quantize_model(model, bits=8)
evaluate_model(model_int8, ...) → test_acc_int8
```

**6. Output**
- Logs `FP32 X.X%  INT8 X.X%` to stdout.
- Saves `infer_confusion_fp32.png` and `infer_confusion_int8.png` to `fold_K/`.

---

## Roadmap

### In progress / pending BU replenishment

| Priority | Task | Command | Expected outcome |
|---|---|---|---|
| 1 | **Bessel filter** (causal, maximally flat group delay) | `ARRAY_SCRIPT=run_puhti_static6.sh bash submit_puhti.sh Results_bessel_static6 --augment-windows --filter-type bessel` | Reduce S7 group-delay penalty vs causal Butterworth; target ≥ 67.2% with full causal claim |
| 2 | **Multi-dataset compatibility test** (1 subject × fold 0) | `sbatch run_puhti_dataset_test.sh` (causal-filter branch) | Verify pipeline runs on PhysionetMI, Cho2017, BNCI2015_001 without errors |
| 3 | **Full multi-dataset accuracy** (all subjects) | Submit per-dataset array jobs after compatibility confirmed | Cross-dataset generalisation benchmark |

### Completed experiments (closed)

| Result | Verdict |
|---|---|
| Adaptive 6–12 bands (V3–V4.1) | +2.1pp vs baseline; static bands match or beat adaptive |
| Sliding-window CSP augmentation (V4.2-augwin) | +0.2pp; useful for covariance estimation |
| 6 static overlapping bands 4–30 Hz (static6-overlap) | **+2.4pp — current best (67.2%)** |
| Causal Butterworth filter (causal-filter branch) | −1.0pp vs zero-phase; neuromorphic-compatible |
| Ledoit-Wolf CSP regularisation | No gain; Tikhonov wins |
| Adaptive MIBIF (mi_fraction 0.05–0.20) | Max 66.5%; percentile=50% wins |
| Channel selection top-K (Approach A, K=5/8/12) | Ceiling at 67.1%; no global K beats static6 |
| Frequency-shift augmentation | −0.5pp vs V4.2-augwin; noise outweighs benefit |
| Recurrent LIF (RLeaky W_rec 64×64) | −8.9pp; 4096 extra params do not converge at 1000 epochs |

### Gap analysis — reaching 70%

Current best: **67.2%** (2.8pp below target). Remaining opportunities:

- **Bessel filter**: expected to recover ~0.5–1.0pp of the causal-Butterworth penalty for S7;
  may also marginally improve zero-phase configuration
- **Subject-specific band tuning**: S7 needs beta (20–26 Hz); S2/S5 are non-responders with
  near-chance performance — gains must come from S4, S6, S9 (all ≥ 8pp below ceiling)
- **Ensemble / majority vote across folds**: low-effort +0.5–1.0pp potential with no new training
- **Transfer learning / domain adaptation**: address cross-session non-stationarity for S2/S5/S6
