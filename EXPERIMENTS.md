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
| 3 | Per-group quantisation sweep, B15 | **DONE** | in job 525714 | — |
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

Per-group at 4 bits, all 12 subjects (delta vs fp32):

| Group | Delta |
|---|---|
| **EA whiteners** | **-19.1** |
| **SNN biases** | **-8.3** |
| z-norm | -3.2 |
| CSP filters | -3.2 |
| SNN weights | -2.9 |

Two findings. The biases are the second most fragile group, not negligible as
first assumed -- they are few, so a per-tensor scale is coarse relative to
their spread, and a bias error displaces membrane potential directly against
a fixed threshold. And the losses **do not accumulate**: individually they sum
to -36.7, but quantising all groups together costs only -20.8, close to EA's
own -19.1. The collapse is dominated by one stage rather than distributed,
which is why fusing EA away recovers most of it.

All 7 integrity checks pass over 1920 sweep rows, including
`all@32bit == fp32_reference` exactly.

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
python compute_energy.py --from-artifacts --n-channels 13 --results-dir Results_bnci2015 --subjects 1 2 3 4 5 6 7 8 9 10 11 12 --n-folds 5 --output-dir Results_energy
```

Writes `Results_energy/energy_summary.csv`. numpy/stdlib only, so login-node
safe.

`--from-artifacts` is required: without it the tool demands
`Results_lava/lava_summary.csv`, and no Loihi/Lava run exists for the binary
datasets. SynOps then come from this codebase's own per-fold instrumentation
(`mean_input_events_per_trial * n_hidden + mean_hidden_events_per_trial *
n_output`), measured during training.

Lava is confirmed unavailable on Roihu (checked 9 Aug 2026): no `lava`
module, and the `.venv_lava` referenced in `fbcsp_snn/lava_model.py` is a
Puhti x86 environment that does not exist there -- and could not be imported
on aarch64 GPU nodes if it did.

This costs less than it sounds. `run_lava_infer.py` computes SynOps as
`input_spikes * n_hidden + hidden_spikes * n_output`, which is the **same
formula** `load_pytorch_side_synops` uses, with the same energy constants.
Lava never measured SynOps independently; it counted spikes in a
`lava.lib.dl.slayer` port and applied identical arithmetic. The artifacts
path is if anything the more faithful spike source, since it comes from the
model whose accuracy the paper actually reports.

What is genuinely lost is the **port validation**: `lava_mean` recorded
accuracy after rebuilding the network in a Loihi-targeted toolchain, which
answers "can this be expressed for Loihi and still classify correctly" --
a different question from energy. The binary paper will not have that.
State it in the methods; do not describe the energy figure as
cross-validated.

`--n-channels` matters: the front-end stages scale with recording geometry,
and the defaults describe BNCI2014-001 (22 ch). Everything else -- samples,
window length, band count, class pairs -- is read from
`pipeline_params.json`, so only the channel count has to be supplied. For
Cho2017 use `--n-channels 64`.

Replaces the 13.9 uJ figure, which was measured on the **4-class** network
(288 features, 80 output neurons, 22 channels, 4 s window).

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

### 8. EA/CSP fusion — **pilot done (S1-S3, 15 folds), full run pending**

Result on the three strongest subjects (FP32 mean 93.8):

| Bits | Split | Fused | Delta |
|---|---|---|---|
| 4 | 57.1 | **75.2** | **+18.2** |
| 6 | 92.3 | 93.4 | +1.0 |
| 8 | 93.5 | 93.7 | +0.2 |
| 16 | 93.8 | 93.8 | +0.0 |
| 32 / FP32 | 93.8 | 93.8 | 0.0 |

`identity_holds: true` (max rel err 4.2e-07), so FP32 parity is exactness,
not coincidence. Storage 1638 -> 624 values per fold (**2.625x**, matching
`1 + n_ch/(P*2m)`).

Mechanism, from the measured dynamic ranges (max / 1st percentile):
EA **48,625x**, CSP 6,504x, fused **3,232x**. Four bits give 15 positive
levels; a 48,625x spread cannot be represented, and everything below
`max/15` rounds to zero -- which is why `ea@4bit` alone went to chance. The
fused matrix is 15x narrower than the whitener and narrower even than the
CSP filters, because the whitener's amplification of low-variance directions
is partly cancelled by filters selecting high-variance ones.

Two qualifications: fusion converts a *collapse* into a *degradation* (75.2
vs 93.8 FP32), it does not make 4 bits free; and S1-S3 are the strongest
subjects, so the 12-subject mean delta will be smaller -- subjects near
chance have nothing to recover.

```bash
python fusion_experiment.py --results-dir Results_bnci2015 --subjects 1 2 3 4 5 6 7 8 9 10 11 12 --n-folds 5 --output-dir Results_fusion
```

Writes `Results_fusion/fusion_BNCI2015_001.{csv,json}`. Needs a GPU (it
re-runs the encoder and classifier), so submit it rather than running on the
login node. Start with `--subjects 1 2 3` to check the FP32 identity holds
before committing to all 12.

Reports four things: whether the fusion is exact in FP32 (it must be --
`max_projection_rel_err` below 1e-6, else the assumed transform order is
wrong), the front-end storage reduction, the dynamic range of each matrix
set, and split-vs-fused accuracy at every bit-width.

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
