# CLAUDE.md — FBCSP-SNN Motor Imagery EEG Classifier

## What this project is

A GPU-accelerated pipeline that classifies motor imagery (MI) from EEG signals using
Filter-Bank Common Spatial Patterns (FBCSP) for feature extraction and a Spiking Neural
Network (SNN) for classification. The pipeline targets neuromorphic hardware deployment —
every operation must map to hardware-friendly primitives (bandpass filters, matrix
multiplies, comparators, leaky integrators).

## Pipeline architecture

```
Raw EEG (n_trials, n_channels, n_samples)
  │
  ├─ Fixed six-band overlapping filter bank (4–8, 8–14, 12–18, 16–24, 20–30, 26–40 Hz)
  │
  ├─ Bandpass filter bank (Butterworth, causal single forward-pass — sosfilt,
  │    not filtfilt — so it maps to a real-time analog Gm-C circuit)
  │    → (n_trials, n_channels × n_bands, n_samples)
  │
  ├─ Pairwise CSP (dual-end: m filters from each eigenspectrum extreme per class pair)
  │    → dict of (pair → projected trials)
  │
  ├─ Z-normalisation (fit on training fold, apply to val/test)
  │
  ├─ Adaptive-threshold spike encoding (delta-based, JIT-compiled)
  │    → binary spike tensor (n_timesteps, n_trials, n_features)
  │
  ├─ Feature selection (MIBIF — mutual information best individual feature)
  │    → adaptive mode: keep features scoring ≥ mi_fraction × max MI score
  │      (variable count per fold/subject — this is what every real training
  │      run actually uses; percentile mode is a dormant fallback, see below)
  │
  ├─ 2-layer LIF SNN (snnTorch)
  │    Input → Linear → Dropout → LIF → Linear → Dropout → LIF → Output
  │    Output uses population coding (N neurons per class)
  │
  ├─ Van Rossum loss (MSE of exponentially-filtered spike trains, FFT-based)
  │
  └─ Winner-take-all decoding (sum output spikes per class population → argmax)
```

## Critical design constraints

- **No data leakage.** Band selection, CSP fitting, z-normalisation statistics, and
  feature selection are computed ONLY on the training split within each CV fold. Validation
  and test data are transformed using training-derived parameters.
- **Everything is per-fold.** Each fold gets its own selected bands, CSP filters, z-norm
  stats, feature indices, and trained SNN. All are saved in fold artifacts for
  reproducible inference.
- **Labels are 1-indexed** in the data pipeline (classes 1..C), converted to 0-indexed
  (0..C-1) only when creating PyTorch tensors for the SNN.
- **Deterministic splits.** StratifiedKFold with `random_state=42`. MOABB single-session
  datasets use StratifiedShuffleSplit with `random_state=42`.

## Tech stack

- Python 3.10+
- PyTorch >= 2.0 (CUDA or CPU)
- snnTorch >= 0.9 (LIF neurons, surrogate gradients)
- scipy (Butterworth filters, Welch PSD, generalised eigenvalue decomposition)
- scikit-learn (StratifiedKFold, mutual_info_classif, accuracy_score, confusion_matrix)
- MOABB + MNE (dataset download and epoching)
- h5py (reading legacy .mat files)
- matplotlib + seaborn (visualisation)

## Target platforms

- **Development:** Local machine (CPU or CUDA)
- **Training:** CSC Puhti HPC (NVIDIA V100, SLURM array jobs)
- **Inference target:** Neuromorphic hardware (motivates INT8 quantisation evaluation)

Must work on both Linux (Puhti) and Windows (local dev). `torch.compile` should be
guarded behind a Triton availability check (Triton is Linux-only). Use `mode="default"`
not `"reduce-overhead"` because CUDA Graphs conflict with snnTorch's `init_leaky()`.

## Dataset

Primary: **BCI Competition IV 2a** (BNCI2014_001 via MOABB)
- 9 subjects, 4 classes (left hand, right hand, feet, tongue)
- 22 EEG channels, 250 Hz
- 2 sessions: session 1 → train, session 2 → test
- ~288 trials per subject per session

Also support (via MOABB registry): PhysionetMI (4-class, 109 subjects),
Cho2017 (2-class, 52 subjects), BNCI2015_001 (2-class, 12 subjects).
Auto-detect `n_classes` from the dataset registry when not specified.

## Project structure

```
fbcsp-snn-sgd-mi-classifier/
├── CLAUDE.md                      ← this file
├── README.md
├── requirements.txt
├── main.py                        ← CLI entry point (train / infer / aggregate)
├── fbcsp_snn/
│   ├── __init__.py                ← DEVICE detection, logger setup, CUDA config
│   ├── config.py                  ← Config dataclass + argparse CLI
│   ├── data.py                    ← HDF5 .mat file loader (legacy file source)
│   ├── datasets.py                ← MOABB dataset registry + loader
│   ├── preprocessing.py           ← Bandpass filter, PairwiseCSP (dual-end)
│   ├── encoding.py                ← Adaptive-threshold spike encoding (JIT)
│   ├── mibif.py                   ← Mutual information feature selection
│   ├── model.py                   ← SNNClassifier (2-layer LIF, population coded)
│   ├── losses.py                  ← Van Rossum loss (FFT-based)
│   ├── training.py                ← Training loop, early stopping, AMP
│   ├── evaluation.py              ← Accuracy, confusion matrix
│   ├── quantization.py            ← Simulated INT8 (symmetric per-tensor)
│   ├── visualization.py           ← All plotting functions
│   └── pipeline.py                ← run_train, run_infer, run_aggregate
├── tests/
│   └── test_*.py                  ← Unit tests per module
├── Results/                       ← Output artifacts (gitignored except JSON/CSV/PNG)
├── analyze_results.py             ← Cross-subject summary analysis
└── run_puhti_array.sh             ← SLURM array job template
```

## Key algorithms — implementation notes

### Fixed filter bank (`preprocessing.py`)
- Six fixed overlapping bands: 4–8, 8–14, 12–18, 16–24, 20–30, 26–40 Hz
- Same six bands for every subject/fold — no per-trial or per-fold band search.
  (An earlier design used per-fold adaptive Fisher-discriminant band selection;
  that module was removed in favour of this fixed bank — see git history around
  `a9ceddf` if the rationale needs revisiting.)

### Pairwise CSP dual-end extraction (`preprocessing.py`)
- For each class pair: solve generalised eigenvalue problem `cov_A W = λ (cov_A + cov_B) W`
- Take first m AND last m eigenvectors (not just first m) — captures filters maximising
  variance for both class A and class B
- Default `csp_components_per_band=8` → m=4 filters/end, 8 filters per pair.
  With 6 pairs (4-class) and 6 bands = 288 total features (before MIBIF selection)
- Regularise covariance: `(1-λ)Σ + λI` with λ=0.0001

### Spike encoding (`encoding.py`)
- Adaptive threshold: if `|signal[t] - signal[t-1]| > threshold` → spike
- On spike: threshold += adapt_inc. Every step: threshold *= decay
- Inner loop must be JIT-compiled (`@torch.jit.script`) for performance
- Input: dict of CSP projections. Output: binary tensor (T, batch, features)

### Van Rossum loss (`losses.py`)
- Convolve spike trains with causal exponential kernel `h[k] = α(1-α)^k` where `α = dt/τ`
- Use FFT convolution (O(T log T)) not sequential IIR (O(T) but not parallelisable)
- Loss = MSE between filtered output spikes and filtered target spikes
- Target spikes: population neurons for correct class fire at `spike_prob` per timestep

### Training (`training.py`)
- Optimiser: AdamW (lr=1e-3, weight_decay=0.1)
- AMP: torch.autocast on CUDA, GradScaler for float16 stability
- Early stopping: monitor val accuracy, patience=100, warmup=100 epochs
- Save best model state by val accuracy

## Default hyperparameters

```
# Filter bank (fixed, not searched)
freq_bands: [(4,8),(8,14),(12,18),(16,24),(20,30),(26,40)]  # Hz

# CSP
csp_components_per_band: 8   (4 from each end)
lambda_r: 0.0001

# Encoding
base_thresh: 0.001, adapt_inc: 0.6, decay: 0.95

# Model
hidden_neurons: 64
population_per_class: 20
beta: 0.95 (membrane decay)
dropout_prob: 0.5

# Training
lr: 1e-3, weight_decay: 0.1, epochs: 1000
n_folds: 5, early_stopping_patience: 100, early_stopping_warmup: 100
spiking_prob: 0.7 (target spike generation)

# Feature selection (mi_fraction and feature_percentile are MUTUALLY
# EXCLUSIVE — mi_fraction takes priority whenever it's set, which is
# always true in practice; feature_percentile is a dormant fallback
# mode, never actually exercised by any real training run)
feature_selection_method: mibif
mi_fraction: 0.1        # ACTIVE: keep features scoring >= 10% of max MI score
feature_percentile: 50.0   # dormant — only used when mi_fraction=None
```

Note: `n_folds` defaults to 10 in `Config`'s dataclass, but every actual Puhti
run (`run_puhti_array.sh`) hardcodes `N_FOLDS=5` and passes `--n-folds 5`
explicitly — 5 is what every subject in `Results_verify` and the paper's
tables actually reflects, not the dataclass default.

## Baseline results to beat

These are from the current pipeline (3 static bands, 22 CSP comps, std-based selection):

| Subject | FP32 Acc | INT8 Acc |
|---------|----------|----------|
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

Target: improve mean accuracy to 70%+. Main opportunity is the weak subjects
(S2, S4, S5, S6) who are near chance level (25% for 4-class).

## Code style

- Type hints on all function signatures
- Docstrings (NumPy style) on all public functions with Parameters/Returns sections
- Use `from fbcsp_snn import setup_logger` for logging, not print statements
- Vectorise operations — avoid Python loops over trials/channels/timesteps where possible
- Use `torch.no_grad()` for all evaluation paths
- Save all artifacts to `Results/Subject_N/` with consistent naming
- Close matplotlib figures after saving (`plt.close(fig)`) to prevent memory leaks

## Commands

```bash
# Train subject 1 (defaults to the fixed six-band filter bank)
python main.py train --source moabb --moabb-dataset BNCI2014_001 \
    --subject-id 1

# Train with a custom band set (overrides the default)
python main.py train --source moabb --moabb-dataset BNCI2014_001 \
    --subject-id 1 --freq-bands "[(4,10),(10,14),(14,30)]"

# Inference
python main.py infer --source moabb --moabb-dataset BNCI2014_001 \
    --subject-id 1 --fold 1

# Aggregate results after parallel fold jobs
python main.py aggregate --subject-id 1 --n-folds 10

# Cross-subject analysis
python analyze_results.py --results-dir Results --subjects 1 2 3 4 5 6 7 8 9

# Run tests
pytest tests/ -v
```