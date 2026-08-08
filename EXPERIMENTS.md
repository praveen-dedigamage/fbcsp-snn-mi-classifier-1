# Experiment runbook — CNA 2026 (binary-only paper)

Living checklist. Update the **Status** column as things finish; the commands
below are the authoritative ones to copy.

Paper scope: binary motor imagery only, BNCI2015-001 (primary) and Cho2017
(secondary). Deadline **2 September 2026**.

Working dir on Roihu: `/scratch/project_2003397/praveen/fbcsp`

---

## Status board

| # | Experiment | Status | Cost | Blocks |
|---|---|---|---|---|
| 1 | BNCI2015-001 training, 12 subj x 5 folds | **DONE** | ~1,800 BU | — |
| 2 | Uniform (whole-pipeline) quantisation sweep, B15 | **DONE** | in job 525714 | — |
| 3 | Per-group quantisation sweep, B15 | **RUNNING** (525714) | — | 4, 7 |
| 4 | Collect evidence bundle | TODO | free | needs 3 |
| 5 | Binary energy estimate | TODO | free | — |
| 6 | Cross-subject summary table | TODO | free | — |
| 7 | Paired significance tests | TODO (needs code) | free | needs 4 |
| 8 | EA/CSP fusion experiment | TODO (needs code) | ~1 short job | — |
| 9 | Fair-baseline control, B15 | TODO (needs Roihu wrapper) | moderate | — |
| 10 | Move MOABB cache to scratch | TODO | free | needs 3 |
| 11 | Cho2017 data prep | TODO | ~1 job | needs 10 |
| 12 | Cho2017 pilot (3 subjects) | TODO | ~800 BU | needs 11 |
| 13 | Cho2017 full training (52 subjects) | TODO | ~13,000 BU (est.) | needs 12 |
| 14 | Cho2017 quantisation sweep | TODO | moderate | needs 13 |

---

## Results so far

BNCI2015-001, 12 subjects, cross-session (session 1 -> session 2), mean of
per-subject fold means:

| Method | Accuracy |
|---|---|
| FBCSP + LDA | 76.7 +/- 14.6 |
| **FBCSP-SNN** | **74.7 +/- 16.0** |
| FBCSP + SVM | 72.0 +/- 14.1 |

Reproduces the earlier Puhti run: LDA and SVM match the published 76.7 / 72.0
exactly, confirming the front end and split are unchanged. The SNN is +2.6 on
the previously reported 72.1, consistent with training stochasticity on
different hardware (V100 -> GH200; earlier runs predate global seeding).

Whole-pipeline quantisation (all five parameter groups at once):

| Bits | Accuracy | Delta |
|---|---|---|
| 32 | 74.7 +/- 16.0 | 0.0 (exact — harness check) |
| 16 | 74.8 | +0.1 |
| 8 | 74.8 | +0.1 |
| 6 | 73.1 | -1.6 |
| **4** | **53.9 +/- 4.0** | **-20.8** |

At 4 bits the SD collapses from 16.0 to 4.0: every subject converges on chance
regardless of prior accuracy. Damage is proportional to available signal
(S1 loses 48 points; S6, already at 56%, loses none).

---

## Commands, in execution order

### 4. Collect evidence bundle  *(after 525714 exits)*

```bash
python collect_results.py --dataset BNCI2015_001 --n-folds 5 --expect-subjects 12
```

Writes `results_bundle_BNCI2015_001.json` (canonical, with provenance and
integrity checks) and `results_summary_BNCI2015_001.md`. Light enough for the
login node: reads JSON/CSV only, no torch import.

### 5. Binary energy estimate

```bash
python compute_energy.py --results-dir Results_bnci2015 --subjects 1 2 3 4 5 6 7 8 9 10 11 12 --n-folds 5 --output-dir Results_energy
```

Replaces the 13.9 uJ figure, which was measured on the **4-class** network
(288 features, 80 output neurons). Reads the measured per-layer spike counts
from each fold's `pipeline_params.json`.

### 6. Cross-subject summary

```bash
python analyze_results.py --results-dir Results_bnci2015 --moabb-dataset BNCI2015_001 --subjects 1 2 3 4 5 6 7 8 9 10 11 12
```

### 7. Paired significance tests — **needs code**

No general tool exists: `analyze_results.py` has no significance testing and
`sig_test_ablation.py` is hard-coded with no CLI. Paired t / Wilcoxon belongs
in `collect_results.py`, which already loads every per-subject mean.

Required before any comparative accuracy claim: the 72.1 -> 74.7 shift moves
every paired comparison, and the SNN-vs-LDA gap narrowed from 4.6 to 2.0
points, which may no longer be significant.

### 8. EA/CSP fusion — **needs code**

EA and CSP are consecutive linear maps: `Y = W^T (R^-1/2 X)`, and `R^-1/2` is
symmetric, so `W_eff = R^-1/2 W` can be precomputed once.

Two predictions, both testable on saved artifacts with no retraining:

- storage per band falls from `n_ch^2 + n_ch*2m` to `n_ch*2m` — at Cho2017's
  64 channels, 4,608 -> 512 values, a **9x** front-end reduction
- the 4-bit collapse disappears, since the wide-dynamic-range whitener
  (built from `eigenvalue^-0.5`) is no longer quantised separately

Highest value per unit of compute of anything on this list: it converts
"EA is fragile" into a design fix.

### 9. Fair-baseline control (B15) — **needs Roihu wrapper**

`run_fair_baseline.py` and `run_fair_baseline_array.sh` exist but are
Puhti-shaped. This underwrites the paper's strongest claim: LDA/SVM given the
same full time series the SNN receives collapse to chance, so the SNN wins
4/4 comparisons.

### 10. Move the MOABB cache to scratch  *(only after 525714 exits)*

```bash
git pull && mv ~/mne_data /scratch/project_2003397/praveen/mne_data
```

`roihu/env.sh` derives `MNE_DATA` from the submit directory, so this needs no
editing. Moving rather than re-downloading preserves the BNCI2015-001 cache.

Do **not** run this while 525714 is alive — it calls `_load_raw()` per subject
and would fail on the remainder.

### 11. Cho2017 data prep

```bash
csc-workspaces
```

```bash
DATASET=Cho2017 sbatch --time=03:00:00 roihu/00b_prepare_data.sh
```

Cho2017 is 52 subjects at 64 channels against BNCI2015-001's 12 at 13, so
roughly 20x the volume. The job prints `df` before and `du -sh` after —
that is the one quantity that has not been measured.

### 12. Cho2017 pilot

```bash
DATASET=Cho2017 sbatch --array=1-3 --time=12:00:00 roihu/01_train_array.sh
```

```bash
seff <jobid>_1
```

The ~13,000 BU estimate for the full run is reasoning, not measurement. The
SNN cost should barely change (features stay at `2m*K = 48` regardless of
channel count); what grows is the Riemannian mean, 13x13 -> 64x64, once per
fold. Three subjects turn the estimate into a number.

### 13. Cho2017 full training

```bash
DATASET=Cho2017 sbatch --array=1-52 --time=12:00:00 roihu/01_train_array.sh
```

### 14. Cho2017 quantisation sweep

```bash
DATASET=Cho2017 sbatch --time=12:00:00 roihu/02_quant_sweep.sh
```

Scientifically the most important reason to run Cho2017: EA scales as
`n_ch^2` while CSP scales as `n_ch`, so at 64 channels EA is ~8x the CSP
block rather than ~1.6x. If the 4-bit collapse is caused by whitener dynamic
range, it should be sharper there — and 52 subjects make the claim far harder
to dismiss than 12.

---

## Deliberately not running

- **Hardware-noise Monte Carlo** — 4-class, expensive, and largely superseded
  by the quantisation sweep.
- **ANN twin** — honest but costly in space, and in a binary-only paper
  ANN+CE already beats the SNN on both datasets.
- **Ablation studies** — excluded by decision; the per-group sweep already
  serves that role.

## Open questions for the paper

- Cho2017's numbers come from the earlier V100 runs while BNCI2015-001 is from
  GH200. Defensible (LDA/SVM reproduce exactly) but must be stated in the
  setup section rather than left implicit.
- The SVM hyperparameter grid is not stated; the artifacts record
  `svm_best_c` and `svm_best_gamma`.
- The introduction claims an analog front end with no ADC, while the bit sweep
  is a **digital** implementation study. The honest framing: the pipeline's
  structure is analog-mappable, and the sweep bounds what a digital
  realisation costs. This needs saying explicitly in the intro.
