# B6 significance test on Schirrmeister2017, fixed six-band pipeline — 2026-07-11

Job `35426603` (25 training tasks, 3rd attempt after two wall-time-related
failures — see `RESULTS_LOG.md`), aggregates re-run after bumping
`run_puhti_aggregate.sh`'s budget (`35431545`-`35431548`, all `COMPLETED`),
analyze job `35431810` (`COMPLETED`) — full 14-subject cross-subject table
in `fbcsp_analyze_35431810.out` in this folder.

**Two cosmetic mislabels in the raw output, noted so they aren't mistaken
for data errors**: the header prints `(BNCI2014_001)` as the dataset name
and `Baseline target: 64.8%` — both are display defaults from
`run_puhti_analyze.sh`/`analyze_results.py` (`--moabb-dataset`/`--baseline`
weren't overridden in the manual `sbatch run_puhti_analyze.sh` invocation).
Neither affects the underlying per-subject FP32/LDA/SVM numbers, which come
directly from each fold's `summary.csv`, not from these display arguments.

**Also note**: this analyze run used a stale (pre-`git pull`) copy of
`run_puhti_analyze.sh`/`analyze_results.py` on Puhti — hence the CSP-8b/6b/
4b PTQ columns still appearing, even though that code was retired from this
repo earlier the same day (see `RESULTS_LOG.md`'s PTQ-retirement entry).
The PTQ columns are real, valid data (computed during training, which ran
under the pre-retirement code) — just not something the paper reports
going forward. Only FP32/LDA/SVM are used below.

## Per-subject values used (n=14)

| Subj | SNN FP32 | LDA | SVM |
|---|---|---|---|
| S1  | 65.0 | 64.0 | 62.7 |
| S2  | 53.6 | 68.0 | 61.5 |
| S3  | 81.9 | 91.2 | 90.8 |
| S4  | 77.5 | 81.5 | 82.5 |
| S5  | 64.0 | 76.7 | 75.5 |
| S6  | 71.5 | 72.9 | 75.0 |
| S7  | 52.2 | 62.2 | 60.6 |
| S8  | 63.7 | 76.3 | 74.5 |
| S9  | 48.9 | 55.2 | 58.2 |
| S10 | 68.9 | 78.8 | 78.7 |
| S11 | 59.3 | 59.2 | 57.3 |
| S12 | 70.8 | 82.6 | 80.7 |
| S13 | 60.2 | 71.5 | 66.5 |
| S14 | 43.6 | 71.9 | 70.6 |
| **Mean** | **62.9±10.9** | **72.3±9.9** | **71.1±10.2** |

Same methodology as the BNCI2014-001 re-run (`puhti_logs/Results_verify/
SIGNIFICANCE_TEST_RERUN.md`): paired t-test + Wilcoxon signed-rank on
per-subject means, computed locally with `scipy.stats` (`ttest_rel`,
`wilcoxon`).

## Result — this is a reversal from the stale adaptive-band numbers, not a confirmation

| Comparison | n | mean diff | wins/losses | paired t-test p | Wilcoxon p |
|---|---|---|---|---|---|
| **Old** (adaptive bands, stale): SNN vs SVM | 14 | −6.0 pp | 0/14 | <0.001 (reported) | — |
| **New** (fixed bands): SNN vs SVM | 14 | **−8.14 pp** | 2/12 | **0.00074** | **0.00061** |
| **Old** (adaptive bands, stale): SNN vs LDA | 14 | **+11.25 pp** (SNN ahead) | — | 0.0001 | 0.0002 |
| **New** (fixed bands): SNN vs LDA | 14 | **−9.35 pp** (SNN behind) | 2/12 | **0.00040** | **0.00061** |

This is not a mild correction — **SNN vs LDA sign-flipped entirely.** The
stale adaptive-band numbers had the SNN significantly *beating* LDA
(+11.25pp). On the real fixed-band retrain, the SNN significantly *loses*
to LDA (−9.35pp). Combined with SNN vs SVM staying significant but getting
worse (−6.0pp → −8.14pp), the honest picture on Schirrmeister2017 is: **the
SNN loses significantly to both classical baselines**, not just SVM.

The SNN wins on only 2 of 14 subjects against either baseline — **S1**
(+2.3pp vs SVM, +1.0pp vs LDA) and **S11** (+2.0pp vs SVM, +0.1pp vs LDA,
essentially a tie). The largest deficit by far is **S14** (−27.0pp vs SVM,
−28.3pp vs LDA) — SNN 43.6% vs LDA 71.9%/SVM 70.6%, the SNN's worst subject
on this dataset and the biggest single driver of the mean gap. Next
largest: S5 (−11.5pp vs SVM), S2 (−14.4pp vs LDA).

Per-subject diffs (SNN − baseline), for reference:
- vs SVM: `+2.3, -7.9, -8.9, -5.0, -11.5, -3.5, -8.4, -10.8, -9.3, -9.8, +2.0, -9.9, -6.3, -27.0`
- vs LDA: `+1.0, -14.4, -9.3, -4.0, -12.7, -1.4, -10.0, -12.6, -6.3, -9.9, +0.1, -11.8, -11.3, -28.3`

## What this means for the paper

- Table IV (`tab:per_subject_sch`), `sec:sch`'s narrative, the abstract's
  and Conclusion's Schirrmeister2017 sentences, and B6's significance-test
  table (`TODO.md`) all need updating with these real numbers — the old
  ones described a dataset the pipeline no longer produces (the adaptive
  12-band variant, removed from the codebase; see B23 in `TODO.md`).
- Honest new framing: **"On Schirrmeister2017, the SNN loses significantly
  to both FBCSP+LDA (p<0.001) and FBCSP+SVM (p<0.001), winning on only 2 of
  14 subjects against either baseline."** This is a materially worse result
  than the paper's old text implied (which had the SNN beating LDA on 12 of
  14 subjects) — report it as such, not softened.
- This does **not** change the BNCI2014-001 headline result (SNN
  significantly beats SVM, ties LDA) — the two datasets now tell a more
  clearly *divergent* story: the SNN's relative competitiveness with
  classical baselines seems specific to the lower-channel-count,
  smaller-dataset regime (BNCI2014-001, 22 channels, 9 subjects) and does
  not transfer to the higher-density, larger Schirrmeister2017 setting (128
  channels, 14 subjects) — worth discussing explicitly in the Discussion
  section rather than leaving it as an unexplained anomaly.
- Not yet re-applied: the multiple-comparisons-correction question already
  flagged as open for BNCI2014-001 (4-6 comparisons across 2 datasets × 2
  baselines) — same open item, not blocking.
