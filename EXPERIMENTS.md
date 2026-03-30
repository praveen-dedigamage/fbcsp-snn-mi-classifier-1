# FBCSP-SNN — Experimental Log

This document records every algorithmic modification attempted, its rationale,
and its measured effect on cross-subject classification accuracy (BNCI2014_001,
9 subjects, 4 classes, 5-fold CV, session 1 train / session 2 test).

---

## Baseline (pre-adaptive pipeline)

**Configuration:** Static 3-band filter bank, 22 CSP components (standard single-end),
std-based feature selection, no Euclidean Alignment.

| Subject | Test FP32 (%) | Test INT8 (%) |
|---------|--------------|--------------|
| S1 | 85.3 | 86.0 |
| S2 | 45.4 | 45.8 |
| S3 | 73.8 | 73.6 |
| S4 | 54.3 | 53.3 |
| S5 | 44.4 | 44.8 |
| S6 | 51.7 | 51.5 |
| S7 | 76.6 | 75.4 |
| S8 | 79.6 | 79.5 |
| S9 | 72.4 | 70.9 |
| **Mean** | **64.8** | **64.5** |

---

## V3 — Adaptive bands + Euclidean Alignment + Riemannian mean covariance

**Tag:** `v3`

### Changes introduced

| Component | Change | Rationale |
|-----------|--------|-----------|
| Band selection | Fisher ERD/ERS adaptive selection (6 bands, 4–30 Hz) replacing static 3-band | Selects physiologically relevant bands per subject per fold from training data only |
| CSP covariance | Riemannian (Fréchet) mean replacing arithmetic mean | More geometrically correct mean on the SPD manifold; reduces sensitivity to outlier trials |
| Euclidean Alignment (EA) | Per-band whitening `X̃ = R^{-1/2} X` before CSP fitting | Reduces cross-trial covariance non-stationarity within session 1 |
| CV splitter | StratifiedKFold(5-fold, 80/20) on session 1 | Deterministic, stratified train/val splits |
| Feature selection | MIBIF top 50% (mutual information best individual feature) | Prunes session-1-specific noise features |
| SNN | 2-layer LIF, hidden=64, population_per_class=20 | Baseline SNN architecture |

### Results

| Subject | Baseline | V3 | Δ vs Baseline |
|---------|----------|-----|--------------|
| S1 | 85.3% | 81.4% | -3.9pp |
| S2 | 45.4% | 41.1% | -4.3pp |
| S3 | 73.8% | 75.6% | +1.8pp |
| S4 | 54.3% | 61.2% | **+6.9pp** |
| S5 | 44.4% | 40.6% | -3.8pp |
| S6 | 51.7% | 52.8% | +1.1pp |
| S7 | 76.6% | 67.7% | -8.9pp |
| S8 | 79.6% | 77.5% | -2.1pp |
| S9 | 72.4% | 77.3% | **+4.9pp** |
| **Mean** | **64.8%** | **63.9%** | **-0.9pp** |

### Analysis

- EA + Riemannian mean helped weak subjects (S4, S9) but hurt strong subjects (S1, S7, S8).
- Large cross-session val-test gaps persisted (S7: 85.8% val vs 67.7% test = 18.1pp).
- Net effect slightly negative. The adaptive bands helped S4 and S9 but EA introduced
  regressions in subjects where session-1 covariance was already discriminative.

---

## Hyperparameter sweep experiments (all relative to V3)

These experiments tested regularisation and split ratio adjustments on a 4-subject
proxy set (S1, S2, S5, S8) before committing to full 9-subject runs.

### Split ratio experiments

| Split | Train trials | Val trials | Grand mean (4 subjects) | Outcome |
|-------|-------------|-----------|------------------------|---------|
| 80/20 KFold (V3) | ~230 | ~58 | 63.9% | **Best** |
| 90/10 ShuffleSplit | ~259 | ~29 | 59.4% | Worse — 29 val trials too noisy for early stopping (1 error = 3.4% swing) |
| 70/30 ShuffleSplit | ~202 | ~86 | mixed | Worse for most — less training data outweighed the benefit of more reliable val signal |

**Conclusion:** 80/20 StratifiedKFold is the optimal split. Early stopping requires
at least ~58 val trials (≈7 per class) for reliable checkpoint selection.

### Feature selection percentile

| Percentile | Features kept | Grand mean vs V3 | Outcome |
|-----------|--------------|-----------------|---------|
| 50% (V3) | 72 features | — | Baseline |
| 30% | 43 features | -6.2pp | Removes useful discriminative features along with noise |

**Conclusion:** 50% is the right threshold. Pruning to 30% removes signal, not just noise.

### Model capacity

| Hidden neurons | Parameters (fc1) | Grand mean vs V3 | Outcome |
|---------------|-----------------|-----------------|---------|
| 64 (V3) | 72×64 = 4,608 | — | Baseline |
| 32 | 72×32 = 2,304 | ≈0pp (flat) | No effect — overfitting is cross-session, not model-size driven |

**Conclusion:** The cross-session val-test gap is not caused by model overfitting in the
classical sense. Reducing model capacity does not improve test accuracy.

### Key insight from sweep

All regularisation-based interventions (smaller model, fewer features, larger val set)
failed to close the cross-session gap. The gap is caused by genuine EEG non-stationarity
between session 1 (train) and session 2 (test), not by conventional overfitting.
The solution requires richer feature representations that capture session-stable patterns,
not stronger regularisation.

---

## V4 — 9 adaptive bands + 6 CSP components per band

**Tag:** `v4`

### Changes vs V3

| Component | V3 | V4 | Rationale |
|-----------|----|----|-----------|
| Adaptive bands | 6 bands | **9 bands** | More frequency resolution; captures more of the 4–40 Hz MI-relevant spectrum |
| CSP components per band | 4 (2 per end) | **6 (3 per end)** | More spatial filters per band; more diverse spatial patterns per frequency region |
| Total features (pre-MIBIF) | 6 pairs × 4 × 6 bands = 144 | 6 pairs × 6 × 9 bands = **324** | 2.25× richer feature space |
| Features after MIBIF 50% | 72 | **162** | SNN input size more than doubled |

### Results

| Subject | Baseline | V3 | V4 | Δ vs V3 | Δ vs Baseline |
|---------|----------|-----|-----|---------|--------------|
| S1 | 85.3% | 81.4% | 82.6% | +1.2pp | -2.7pp |
| S2 | 45.4% | 41.1% | 44.7% | +3.6pp | -0.7pp |
| S3 | 73.8% | 75.6% | 77.8% | +2.2pp | **+4.0pp** |
| S4 | 54.3% | 61.2% | 63.5% | +2.3pp | **+9.2pp** |
| S5 | 44.4% | 40.6% | 45.3% | +4.7pp | +0.9pp |
| S6 | 51.7% | 52.8% | 54.3% | +1.5pp | +2.6pp |
| S7 | 76.6% | 67.7% | 72.5% | +4.8pp | -4.1pp |
| S8 | 79.6% | 77.5% | 80.3% | +2.8pp | +0.7pp |
| S9 | 72.4% | 77.3% | 80.7% | +3.4pp | **+8.3pp** |
| **Mean** | **64.8%** | **63.9%** | **66.9%** | **+3.0pp** | **+2.1pp** |

### Analysis

- **Every subject improved vs V3** — the first experiment with universal improvement.
- **First result to beat the baseline** (66.9% vs 64.8%, +2.1pp).
- The richer feature set (324 → 162 features after MIBIF) exposes the SNN to a more
  diverse set of spatial-frequency combinations. MIBIF then selects the subset most
  discriminative for each fold, which tends to favour session-stable patterns.
- Largest gains: S4 (+9.2pp vs baseline), S9 (+8.3pp vs baseline) — subjects where
  cross-session covariance shift is the dominant challenge.
- Remaining gap to 70% target: **3.1pp**.

---

## Ongoing experiments (results pending)

### Band count scaling (12 bands, 6 CSP)

**Hypothesis:** Extending from 9 to 12 adaptive bands further increases frequency
resolution and may capture additional session-stable MI patterns.

**4-subject pilot (S1, S2, S5, S9):**

| Subject | V4 (9b6c) | 12b6c | Δ |
|---------|-----------|-------|---|
| S1 | 82.6% | 81.9% | -0.7pp |
| S9 | 80.7% | 80.8% | +0.1pp |
| S2 | 44.7% | 46.1% | +1.4pp |
| S5 | 45.3% | 45.7% | +0.4pp |

**Observation:** Marginal and subject-specific. 9 bands appears to be the sweet spot
for band count. Going 6→9 gave +3pp grand mean; going 9→12 is flat.

### CSP component scaling (9 bands, 8 CSP)

**Hypothesis:** More spatial filters per band captures greater diversity of spatial
patterns, some of which may be more session-stable.

**4-subject pilot (S1, S2, S5, S9):**

| Subject | V4 (9b6c) | 9b8c | Δ |
|---------|-----------|------|---|
| S1 | 82.6% | 81.5% | -1.1pp |
| S9 | 80.7% | 82.6% | +1.9pp |
| S2 | 44.7% | 44.2% | -0.5pp |
| S5 | 45.3% | 46.9% | +1.6pp |

**Observation:** Subject-specific gains. Neither axis (more bands, more CSP) consistently
outperforms V4 on all subjects.

### Combined scaling (12 bands, 8 CSP) — all 9 subjects

**Hypothesis:** Combining both axes (432 features → 216 after MIBIF) may capture
additive benefits seen partially in each individual experiment.
Total features: 6 pairs × 8 filters × 12 bands = 432 → 216 after MIBIF 50%.
**Status:** Running on Puhti.

---

## Summary table

| Version | Bands | CSP/band | Features (post-MIBIF) | Grand Mean | vs Baseline |
|---------|-------|----------|-----------------------|-----------|-------------|
| Baseline | 3 (static) | ~7 | — | 64.8% | — |
| V3 | 6 (adaptive) | 4 | 72 | 63.9% | -0.9pp |
| V4 | **9 (adaptive)** | **6** | **162** | **66.9%** | **+2.1pp** |
| 12b6c (pilot) | 12 | 6 | 216 | ~flat vs V4 | — |
| 9b8c (pilot) | 9 | 8 | 216 | ~flat vs V4 | — |
| 12b8c | 12 | 8 | 216 | pending | — |

**Target:** 70.0% grand mean FP32.
**Current best:** V4 at 66.9% (+2.1pp vs baseline).
