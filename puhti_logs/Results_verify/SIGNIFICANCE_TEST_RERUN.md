# B6 significance test re-run on Results_verify (fixed six-band pipeline) — 2026-07-08

Same methodology as the original B6 test (`TODO.md`): paired t-test +
Wilcoxon signed-rank on per-subject means, n=9, BNCI2014-001. Script:
`sig_test_verify.py` (scratchpad), per-subject inputs cross-checked against
the re-pasted `logs/fbcsp_analyze_35390609.out` (see `ANALYSIS.md`) —
FP32/LDA/SVM values match exactly except S8 FP32 (78.7 vs 78.8, a 0.1pp
internal rounding inconsistency between the tool's own two printed tables,
not something introduced during logging).

## Result — this flips the paper's headline claim

| Comparison | n | mean diff | wins/losses | paired t-test p | Wilcoxon p |
|---|---|---|---|---|---|
| **Old** (`Results/`, adaptive bands): SNN vs SVM | 9 | +2.90 pp | — | 0.0638 | 0.0547 |
| **New** (`Results_verify`, fixed bands): SNN vs SVM | 9 | **+6.71 pp** | 8/1 | **0.0070** | **0.0078** |
| **Old**: SNN vs LDA | 9 | +16.33 pp | — | 0.0027 | 0.0039 |
| **New**: SNN vs LDA | 9 | +5.39 pp | 8/1 | 0.2112 | 0.0977 |

Two changes, both real, both worth stating plainly in the paper:

1. **SNN vs. SVM flipped from "not significant" to significant (p<0.01).**
   Old: p≈0.05–0.06, right at the conventional threshold, honestly reported
   as "competitive, not significantly different." New: p=0.007–0.008,
   clearly significant, SNN wins 8 of 9 subjects. This is a *stronger*,
   cleaner headline claim than what the paper currently states — and SVM is
   the more relevant classical baseline in MI-BCI literature, so this is the
   comparison that matters most.

2. **SNN vs. LDA flipped from decisively significant to *not* significant.**
   Old: p=0.003–0.004, SNN crushed LDA by +16.3pp. New: p=0.10–0.21, gap
   shrunk to +5.4pp. This is the direct, expected consequence of Tier 0's
   `shrinkage='auto'` regularisation making LDA a much stronger baseline
   (LDA jumped from 49.8%→60.6% mean) — the old "SNN beats LDA easily" claim
   no longer holds once LDA is tuned properly. This is *good*, not bad: it
   means the old comparison was flattering the SNN by comparing it against
   an artificially weak, unregularised LDA.

Both subjects where the SNN loses to a baseline are the same: **S7** (loses
to both SVM and LDA — the SNN's weakest fold this run, FP32 67.5% vs. 74.8%
in the old adaptive-band run, already flagged in `ANALYSIS.md` as the one
per-subject move "worth a raised eyebrow").

## What this means for the paper

- The abstract/Conclusion's current framing ("competitive with SVM, not
  significant; significantly ahead of LDA") is now backwards on both counts
  once real numbers replace the `\TODO{}` markers. New honest framing:
  **"significantly outperforms FBCSP+SVM (p<0.01); not significantly
  different from a properly-regularised FBCSP+LDA."**
- This also changes B7's framing (the Schirrmeister2017 cherry-picking
  concern) — the BNCI2014-001 headline is now a *stronger* result to lead
  with, which makes the honest cross-dataset framing (losing to SVM on
  Schirrmeister2017) easier to present without feeling like it undercuts the
  paper, since the primary-dataset claim is now solidly significant rather
  than borderline.
- **UPDATE 2026-07-08, later same day: DONE.** The abstract, Table III,
  `tab:multiclass_bnci`, `sec:bnci` prose, the Discussion, and the Conclusion
  were all hand-edited with these p-values and the real per-subject numbers
  (`fbcsp_analyze_35390609.out`'s full re-paste, cross-checked). The
  multiple-comparisons-correction question noted below is still open and
  NOT yet applied to the numbers in `main.tex` — worth revisiting before
  submission, not blocking for now.
- Still open: reconsider whether a multiple-comparisons correction is
  warranted (4 comparisons across 2 datasets × 2 baselines, as already noted
  as an open item in B6 in `TODO.md`).

## INT8 status — RESOLVED, this section's premise turned out wrong

The per-subject INT8 pull below was requested before the user pointed out
(next turn) that this codebase intentionally dropped isolated INT8/CSP-bit
testing in favour of the joint CSP+SNN quantisation sweep (already decided
in `RESULTS_LOG.md`, "What verification actually checks" section, before
this significance-test work even started). **Never pulled, not needed.**
Table III's INT8 column was replaced with a pointer to the new joint sweep
table (`tab:joint_quant`) instead — see
`puhti_logs/Results_verify/joint_quantization_sweep.md` for that real data.
