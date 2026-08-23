# Experiment runbook — CNA 2026 (binary-only paper)

Living checklist. Update the **Status** column as things finish; the commands
below are the authoritative ones to copy.

Paper scope: binary motor imagery only, BNCI2015-001 (primary) and
BNCI2014-002 (secondary). Deadline **2 September 2026**.

**DECISION 2026-08-23: the two-band bank (6-15, 12-32 Hz) is the pipeline for
the new manuscript.** The six-band results stay as the record of what the
earlier draft reported; they are not being rewritten yet. Every new experiment
runs two bands unless it explicitly says otherwise.

Cho2017 is **out of scope**: there is a problem with the dataset that has to be
dealt with separately. It is not reported and not released. Items 11-14 below
are retired, kept only so their numbering does not shift.

Working dir on Roihu: `/scratch/project_2003397/praveen/fbcsp`

---

## Status board

| # | Experiment | Status | Cost | Blocks |
|---|---|---|---|---|
| 1 | BNCI2015-001 training, 12 subj x 5 folds | **DONE** | ~1,800 BU | — |
| 2 | Uniform (whole-pipeline) quantisation sweep, B15 | **DONE** | in job 525714 | — |
| 3 | Per-group quantisation sweep, B15 | **DONE** | in job 525714 | — |
| 4 | Collect evidence bundle | **DONE** | free | — |
| 5 | Binary energy estimate | **DONE** | free | — |
| 6 | Cross-subject summary table | superseded by 4 | free | — |
| 7 | Paired significance tests | TODO (needs code) | free | — |
| 8 | EA/CSP fusion experiment | **DONE** | ~290 BU (2 failed runs) | — |
| 9 | Fair-baseline control, B15 | script ready, not run | moderate | — |
| 10 | Move MOABB cache to scratch | **DONE** | free | — |
| 11-14 | Cho2017 (prep, pilot, training, sweep) | **RETIRED — out of scope** | — | — |
| 15 | BNCI2014-002 training, 14 subj x 5 folds | **DONE** | ~1,700 BU | — |
| 16 | BNCI2014-002 sweeps, fusion, fair baseline, energy | **DONE** | — | — |
| 17 | Two-band bank (6-15, 12-32 Hz), B15 | RUNNING (job 788894) | ~1,800 BU | — |
| 18 | Two-band bank, BNCI2014-002 | TODO | ~1,700 BU | — |
| 19 | Two-band + m=12 control (48 features) | TODO, optional | ~3,500 BU | needs 17 |
| 20 | Bit-width sweep on the two-band bank | TODO | moderate | needs 17, 18 |
| 21 | Fairness control on the two-band bank | TODO | moderate | needs 17, 18 |
| 22 | Two-band results complete; manuscript rewrite queued | **DONE / on hold** | — | — |

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
own -19.1. The collapse is dominated by one stage rather than distributed.

That made fusion look like it should recover nearly all of the loss. It does
not (see 8): removing EA as a separately quantised stage recovers only a
third. So the whitener's *stored representation* is not the whole of its
contribution -- the conditioning it imposes on the signal reaching CSP
matters too, and that survives fusion.

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
`pipeline_params.json`, so only the channel count has to be supplied:
`--n-channels 13` for BNCI2015-001, `15` for BNCI2014-002.

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

### 8. EA/CSP fusion — **DONE (12 subjects, 60 folds)**

| Bits | Split | Fused | Delta |
|---|---|---|---|
| 4 | 54.2 | 61.0 | **+6.8** |
| 6 | 73.5 | 74.0 | +0.5 |
| 8 | 74.9 | 74.6 | -0.3 |
| 16 | 74.7 | 74.8 | +0.1 |
| 32 / FP32 | 74.7 | 74.7 | 0.0 |

`identity_holds: true` (4.2e-07) across all 60 folds. Storage 1638 -> 624
values per fold (**2.625x**). Dynamic range: EA 48,625x, CSP 6,504x,
fused 3,232x.

**The pilot overstated the accuracy effect.** S1-S3 gave +18.2 at 4 bits; all
12 subjects give **+6.8**, because subjects near chance have nothing to
recover. Fusion mitigates the four-bit failure rather than removing it:
61.0 vs 74.7 FP32 is still a 13.7-point loss, about two thirds of the
original 20.5. At 8 bits it is marginally negative (-0.3, within noise), so
it helps only where precision is genuinely scarce.

What is unconditional is the **storage reduction**: exact, independent of
bit-width and subject, and growing with electrode count
(`1 + n_ch/(P*2m)`: measured 2.6x at 13 channels, 2.9x at 15).

NOTE: this experiment quantises the **front end only** (EA + CSP), so its
split column (54.2 at 4 bits) is not the same measurement as the uniform
sweep (53.9, all five groups). Do not put them in one table.

```bash
python fusion_experiment.py --results-dir Results_bnci2015 --subjects 1 2 3 4 5 6 7 8 9 10 11 12 --n-folds 5 --output-dir Results_fusion
```

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

### 11-14. Cho2017 — RETIRED, out of scope

The data prep, pilot, full-training and sweep steps for Cho2017 were removed.
There is a problem with the dataset that has to be dealt with separately, and
it is neither reported nor released with this paper. Numbering is preserved so
references to items 15+ elsewhere stay valid.

---

### 17. Two-band filter bank — BNCI2015-001

Replaces the six-band bank with two wide bands, (6,15) and (12,32) Hz. The
bank is the largest fixed term in the energy budget (one always-on Gm-C filter
per band per channel), so two bands cut it to a third.

FREQ_BANDS carries the bank; it defaults to the published six, so omitting it
reproduces the paper exactly. Always pair it with RESULTS_DIR or the run
overwrites the fold artifacts behind the published 74.7 %.

```bash
FREQ_BANDS="[(6,15),(12,32)]" RESULTS_DIR=Results_bnci2015_2band     sbatch --array=1-12 roihu/01_train_array.sh
```

Check the job header echoes `FREQ_BANDS : [(6,15),(12,32)]` before letting it
run to completion. If it echoes the six-band list the override did not reach
env.sh, and the run will produce plausible numbers that are just the published
config in a new directory.

Read the accuracy out. Point --quant-dir at a directory that does not exist:
the default Results_quant holds the SIX-band sweeps, and collect_results would
merge them into a two-band bundle, breaking the `all@32bit == fp32_reference`
integrity check.

```bash
module load python-pytorch/2.10
python collect_results.py     --results-dir Results_bnci2015_2band     --quant-dir   Results_quant_2band     --dataset     BNCI2015_001     --n-folds 5 --expect-subjects 12     --output   bundle_2band_BNCI2015_001.json     --markdown summary_2band_BNCI2015_001.md
```

Baseline to beat, paired by subject across 12: **74.7 +/- 16.0**.

Observed at the pilot: MIBIF kept 9-11 of 16 features (56-69 %), against 24 of
48 (50 %) for six bands. Not starved, but the feature count is 2.4x smaller, so
a drop cannot be attributed to the bands alone without item 19.

### 18. Two-band filter bank — BNCI2014-002

14 subjects, so --array must be overridden. Both datasets are already cached
under mne_data/MNE-bnci-data/~bci/database/ (001-2015 and 002-2014), so no
prep job is needed. Note 00b_prepare_data.sh defaults N_SUBJECTS to 12 for any
non-Cho2017 dataset — pass N_SUBJECTS=14 if the cache ever has to be rebuilt.

```bash
DATASET=BNCI2014_002 FREQ_BANDS="[(6,15),(12,32)]"     RESULTS_DIR=Results_bnci2014_002_2band     sbatch --array=1-14 roihu/01_train_array.sh
```

```bash
python collect_results.py     --results-dir Results_bnci2014_002_2band     --quant-dir   Results_quant_2band     --dataset     BNCI2014_002     --n-folds 5 --expect-subjects 14     --output   bundle_2band_BNCI2014_002.json     --markdown summary_2band_BNCI2014_002.md
```

Baseline to beat, paired by subject across 14: **69.2 +/- 16.1**.

### 19. Two-band + m=12 control — optional, disambiguates 17 and 18

Items 17 and 18 change the bank AND the feature count at once (48 -> 16 before
MIBIF). If accuracy moves, this arm says which caused it: m=12 gives
2*12*2 = 48 pre-selection features, matching the six-band bank exactly while
keeping the two bands.

```bash
FREQ_BANDS="[(6,15),(12,32)]" RESULTS_DIR=Results_bnci2015_2band_m12     sbatch --array=1-12 roihu/01_train_array.sh --csp-components-per-band 12
```

Only worth the BU if the two-band result needs attributing rather than just
answering "does a two-band bank work at all".

---

### 20. Bit-width sweep on the two-band bank

Repeats the paper's quantisation study against the two-band front end. The
sweep quantises the composed spatial filters with a separate scale PER BAND,
so the front-end group is band-count dependent by construction: none of
Table 2 or Figure 2 carries over from the six-band bank.

Needs items 17 and 18 finished — the sweep reads saved fold artifacts.

**OUT_DIR is not optional.** Output is named
`quant_sweep_<dataset>_<mode>[_fused].csv` with NO band count in the filename,
so the default OUT_DIR=Results_quant would overwrite the six-band CSVs behind
the published Table 2. There is no undo.

**FUSE=1 is what the paper reports.** Table 2 and Figure 2 come from the
`_fused` sweeps, where the EA whitener is composed into the spatial filters and
the front end is a single group. A split run is a different measurement.

```bash
DATASET=BNCI2015_001 RESULTS_DIR=Results_bnci2015_2band     OUT_DIR=Results_quant_2band FUSE=1 sbatch roihu/02_quant_sweep.sh

DATASET=BNCI2014_002 RESULTS_DIR=Results_bnci2014_002_2band     OUT_DIR=Results_quant_2band FUSE=1 sbatch roihu/02_quant_sweep.sh
```

One job runs both uniform and per-group modes. Subject counts come from the
DATASET case block (12 and 14), so SUBJECTS does not need setting.

To chain straight off training instead of waiting:

```bash
TRAIN=$(FREQ_BANDS="[(6,15),(12,32)]" RESULTS_DIR=Results_bnci2015_2band     sbatch --parsable --array=1-12 roihu/01_train_array.sh)
DATASET=BNCI2015_001 RESULTS_DIR=Results_bnci2015_2band     OUT_DIR=Results_quant_2band FUSE=1     sbatch --dependency=afterok:${TRAIN} roihu/02_quant_sweep.sh
```

Then rebuild the bundles with --quant-dir pointing at the two-band sweeps, so
the accuracy and sweep rows in one bundle describe the same pipeline:

```bash
python collect_results.py     --results-dir Results_bnci2015_2band     --quant-dir   Results_quant_2band     --dataset     BNCI2015_001     --n-folds 5 --expect-subjects 12     --output   bundle_2band_BNCI2015_001.json     --markdown summary_2band_BNCI2015_001.md
```

First thing to check in the result: the 32-bit row must equal that sweep's own
fp32_reference exactly, on every subject and fold. collect_results asserts it.
If it fails the harness is wrong and no other bit-width means anything.

Six-band baselines, uniform fused, to compare against:

| Dataset | 32 | 16 | 8 | 6 | 4 |
|---|---|---|---|---|---|
| BNCI2015-001 | 74.7 | 74.7 | 74.5 | 73.6 | 57.2 |
| BNCI2014-002 | 69.2 | 69.5 | 69.5 | 67.9 | 56.9 |

The interesting question is whether the four-bit collapse softens. The paper
attributes it to the front end spanning the widest dynamic range; two wider
bands mean fewer, differently-conditioned matrices, so the collapse could move.

---

### 21. Fairness control on the two-band bank

Fills the `LDA full-ts` / `SVM full-ts` columns, which come from
fair_baseline_results.json in each fold directory. Needs items 17 and 18.

**This job WRITES into RESULTS_DIR** rather than reading it, so an unset
RESULTS_DIR injects two-band baselines into the artifacts behind the published
numbers. 04_fair_baseline.sh also hardcodes --array=1-12, so BNCI2014-002 must
override it or subjects 13 and 14 silently keep empty columns.

```bash
DATASET=BNCI2015_001 RESULTS_DIR=Results_bnci2015_2band     sbatch --array=1-12 roihu/04_fair_baseline.sh

DATASET=BNCI2014_002 RESULTS_DIR=Results_bnci2014_002_2band     sbatch --array=1-14 roihu/04_fair_baseline.sh
```

Then re-run the item 17/18 collect_results commands; the columns populate and
two snn_vs_*_fullts rows appear in the paired table. aggregate_fair_baseline.py
is NOT needed -- collect_results reads the per-fold JSON directly, and that
script only prints a table. Its one use is listing folds whose fair-baseline
file is missing.

Six-band baselines: 53.9 / 53.3 (B15) and 52.5 / 53.2 (B14), all at chance.
If the two-band controls also sit at chance the paper's central claim survives
the new bank. If they rise, the fairness argument weakens, and that matters
more than the accuracy parity below.

Accuracy result already in hand (item 17/18), paired by subject:

| Dataset | 6-band | 2-band | paired delta | wins | Wilcoxon |
|---|---|---|---|---|---|
| BNCI2015-001 | 74.7 +/- 16.0 | 74.6 +/- 17.1 | -0.06 | 6/12 | 0.87 |
| BNCI2014-002 | 69.2 +/- 16.1 | 70.5 +/- 16.5 | +1.32 | 6/14 | 0.54 |

Parity on both. The +1.32 is carried by four subjects while eight lost
slightly, which is why the rank test sees 6/14. Same accuracy from a third of
the filter bank.

---

### 22. Two-band results — COMPLETE, and what the manuscript needs

All four jobs done for both datasets: training, fairness control, and the
uniform + per-group sweeps, fused. 7/7 integrity checks pass on both bundles,
including all@32bit == fp32_reference exact. Bundles and summaries pulled off
Roihu to a local directory outside both repos.

Paired by subject, two-band against six-band:

| | B15 6-band | B15 2-band | p | B14 6-band | B14 2-band | p |
|---|---|---|---|---|---|---|
| SNN | 74.7 | 74.6 | 0.87 | 69.2 | 70.5 | 0.38 |
| LDA | 76.7 | 77.3 | 0.23 | 71.7 | 74.4 | 0.09 |
| SVM | 72.0 | 75.8 | **0.034** | 70.4 | 72.0 | 0.35 |
| LDA full-ts | 53.9 | 52.7 | 0.23 | 52.5 | 51.8 | 0.49 |
| SVM full-ts | 53.3 | 52.3 | 0.23 | 53.2 | 51.4 | 0.11 |

Accuracy is at parity. The fairness control HOLDS: within the two-band run the
SNN beats the full-ts controls by +21.9/+22.4 (11/12 subjects, p=0.00098) and
+18.8/+19.2 (12/14, p<=0.003). The paper's central claim survives the bank
change.

Two things changed and both need writing up:

1. Quantisation is less forgiving. Uniform fused, delta vs fp32:

   | | 8 bit | 6 bit | 4 bit |
   |---|---|---|---|
   | B15 6-band | -0.2 | -1.1 | -17.5 |
   | B15 2-band | -0.6 | -3.0 | -18.3 |
   | B14 6-band | +0.2 | -1.4 | -12.4 |
   | B14 2-band | -1.0 | -3.0 | -15.6 |

   "Eight bits are free" still holds. "Six bits cost about one point" does not:
   it is 3.0 on both.

2. The per-group ordering INVERTED. Six bands: the composed front end dominates
   on both datasets, which is the whole argument of section 4.3. Two bands:
   snn_b is worst on both (14.5 vs csp 10.0 on B15; 11.4 vs csp 8.3 on B14).
   Shrinking the front end moved the fragility into the classifier biases.

Manuscript changes queued (NOT started, manuscript is on hold):

- 2.2: the band list, the fractional-bandwidth-0.4 rationale (2 bands are 0.86
  and 0.91, not constant-Q), and "six bands rather than FBCSP's nine".
- 2.3: 2mK = 48 becomes 16; F averages ~9.8 not 24; parameter count ~3,300 not
  4,200.
- Table 1 and 4.1: all accuracy and full-ts numbers.
- Table 2 and 4.2: the whole sweep, and the six-bit claim.
- 4.3 and Figure 2: the argument reverses -- front end no longer dominates.
- 4.4 and Table 3: the Gm-C bank is n_bands x n_channels, so the largest fixed
  energy term drops to a third. Recompute with --from-artifacts, which reads
  n_bands from the fold records. compute_energy.py:192's N_CSP_MACS constant
  still hardcodes 6 and is wrong off that path.
- Abstract, intro and discussion: any figure quoted from the above.

Open: whether FREQ_BANDS' default in env.sh should become the two-band bank.
Leaving it at six keeps the published results reproducible by default; changing
it makes the new pipeline the default and the old one the override. Not decided.

---

## Deliberately not running

- **Hardware-noise Monte Carlo** — 4-class, expensive, and largely superseded
  by the quantisation sweep.
- **ANN twin** — honest but costly in space, and in a binary-only paper
  ANN+CE already beats the SNN on both datasets.
- **Ablation studies** — excluded by decision; the per-group sweep already
  serves that role.

## Open questions for the paper

- Two bands change the bank AND the feature count (48 -> 16 before MIBIF).
  Item 19 is the control that separates them; decide whether the attribution
  is worth the BU.
- The SVM hyperparameter grid is not stated; the artifacts record
  `svm_best_c` and `svm_best_gamma`.
- The introduction claims an analog front end with no ADC, while the bit sweep
  is a **digital** implementation study. The honest framing: the pipeline's
  structure is analog-mappable, and the sweep bounds what a digital
  realisation costs. This needs saying explicitly in the intro.
