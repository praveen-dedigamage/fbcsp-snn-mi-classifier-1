> **CORRECTION (2026-07-08, later same day):** Everything below this line was
> written before the real root cause was found. The SVM-tuning "optimizer's
> curse" diagnosis was a false lead — it compared numbers across two different
> feature-extraction pipelines without realizing it. See
> [**BAND_SELECTION_FINDING.md**](BAND_SELECTION_FINDING.md) for the actual
> cause (original `Results/` used a since-removed 12-band adaptive
> Fisher-selection front end; `Results_verify` uses the fixed six-band front
> end — confirmed intentional, keep fixed six-band) and the corrected
> conclusion: **treat `Results_verify` as the new authoritative numbers once
> complete; do not compare them against the old `Results/` as a regression
> check.** The event-count sanity check and the `n_timesteps`/`svm_best_c`/
> `svm_best_gamma` CSV-field fix below are still accurate and stand.
>
> **SECOND CORRECTION (same day, after re-pasting the full analyze log):**
> `fbcsp_analyze_35390609.out` originally logged here included a fabricated
> "INT8" column (83.1%, 49.7%, ...) that does not exist in the real script
> output — it was reconstructed inaccurately from a compacted conversation
> summary and never verified against the actual file before being written
> down as fact. Fixed by re-pasting the real file content. The FP32/LDA/SVM
> point estimates used below and in the significance-test re-run were
> independently checked against the real re-paste and are correct. See
> **SIGNIFICANCE_TEST_RERUN.md** for the new B6 result on the verified
> numbers — it substantially changes the paper's headline claim (SNN now
> significantly beats SVM; no longer significantly beats LDA).

# Results_verify vs. original Results/ (paper Table III) — 2026-07-08

Source: `logs/fbcsp_analyze_35390609.out` + `Results_verify/Subject_1/summary.csv`,
pasted by user after `bash submit_puhti.sh Results_verify`. Raw data archived in
[fbcsp_analyze_35390609.out](fbcsp_analyze_35390609.out). Original numbers are
`main.tex` Table `tab:per_subject_bnci` (lines 686–696).

## FP32 — per subject, original → verify — ⚠️ framing below is SUPERSEDED, see banner at top of file

The numbers in this table are correct, but "no regression" is the wrong
frame — `Original` and `Verify` are two different feature-extraction
pipelines (12 adaptive bands vs. 6 fixed bands), not the same pipeline run
twice. The fact that the means land close together is a coincidence, not a
verification result. See `BAND_SELECTION_FINDING.md`.

| Subj | Original | Verify | Δ (pp) |
|------|----------|--------|--------|
| S1   | 82.0     | 83.2   | +1.2   |
| S2   | 42.8     | 49.9   | +7.1   |
| S3   | 73.5     | 75.3   | +1.8   |
| S4   | 63.8     | 62.6   | −1.2   |
| S5   | 49.0     | 47.1   | −1.9   |
| S6   | 50.2     | 49.4   | −0.8   |
| S7   | 74.8     | 67.5   | **−7.3** |
| S8   | 81.7     | 78.7   | −3.0   |
| S9   | 77.6     | 80.2   | +2.6   |
| **Mean** | **66.2** | **66.0** | **−0.2** |

**Verdict: no regression.** Mean is flat within the ±14pp cross-subject SD. S7's
−7.3pp is the only per-subject move worth a raised eyebrow — everything else is
consistent with normal SGD/dropout training-run variance (fixed CV split via
`random_state=42`, but SNN weight init / dropout masks / batch order are not
independently reseeded per fold, so re-running the exact same fold is not
bit-identical). Not investigating further unless it recurs on the S7 ablation
or reliability runs.

## LDA(tuned) vs LDA(original) — the Tier 0 win

| Subj | Original | Tuned | Δ (pp) |
|------|----------|-------|--------|
| S1   | 55.1     | 79.9  | **+24.8** |
| S2   | 41.9     | 44.1  | +2.2   |
| S3   | 40.5     | 72.6  | **+32.1** |
| S4   | 51.2     | 58.5  | +7.3   |
| S5   | 40.5     | 41.0  | +0.5   |
| S6   | 40.7     | 46.9  | +6.2   |
| S7   | 69.9     | 80.0  | +10.1  |
| S8   | 55.6     | 75.6  | **+20.0** |
| S9   | 53.0     | 46.8  | −6.2   |
| **Mean** | **49.8** | **60.6** | **+10.8** |

8/9 subjects improve, three by 20+pp. This is exactly what `solver='lsqr',
shrinkage='auto'` should do: the original LDA used the default SVD solver with
no regularisation against a near-singular within-class covariance (288 CSP
features, ~230 training trials/fold) — classic small-sample instability.
**Credible, expected, good.** S9's small drop is noise-level and not a concern
on its own — but see below, S9 also has the largest SVM drop, which is a
separate, real problem.

## SVM(tuned) vs SVM(original) — a real problem, not noise — ⚠️ SUPERSEDED, this theory is WRONG, see banner at top of file

The "optimizer's curse" theory below was debunked the same day: `Original`
and `Tuned` here come from two different feature-extraction pipelines
(12 adaptive bands vs. 6 fixed bands), not the same pipeline with/without
SVM hyperparameter tuning. The tables and reasoning are kept for the record
of how the investigation proceeded, **not as a conclusion to act on.**

| Subj | Original | Tuned | Δ (pp) |
|------|----------|-------|--------|
| S1   | 76.1     | 76.0  | −0.1   |
| S2   | 43.6     | 39.6  | −4.0   |
| S3   | 74.9     | 73.9  | −1.0   |
| S4   | 62.2     | 60.0  | −2.2   |
| S5   | 38.3     | 36.8  | −1.5   |
| S6   | 50.9     | 46.5  | −4.4   |
| S7   | 68.3     | 67.6  | −0.7   |
| S8   | 79.2     | 70.4  | **−8.8** |
| S9   | 75.8     | 62.7  | **−13.1** |
| **Mean** | **63.3** | **59.3** | **−4.0** |

**9 of 9 subjects got worse after "tuning."** That's the tell: the grid
(`C ∈ {0.1,1,10} × γ ∈ {'scale',0.01,0.1}`) includes the original untuned
point (`C=1, γ='scale'`) as one of its 9 candidates, so a grid search that
actually generalised should never make things uniformly worse — at worst it
should match the default on subjects where nothing beats it. A **uniform**
drop across every subject, worst on S8/S9, is the signature of **overfitting
the hyperparameter choice to a small validation split**: 5-fold CV on ~288
session-1 trials leaves roughly 55–60 trials per fold for validation, split
across 4 classes (~14/class). Picking the single best of 9 noisy val-accuracy
estimates on that few points reliably overstates val performance and
understates test performance (the "optimizer's curse") — worse than just
keeping the untuned default.

Confirmed in code, not just inferred: `fbcsp_snn/baseline.py`'s SVM block
picks `argmax` over 9 `(C, γ)` combinations by validation accuracy alone, no
inner cross-validation — so there's nothing protecting it from exactly this
failure mode.

**Recommendation (not yet implemented):** replace the single-val-split grid
search with a small inner k-fold CV *within the training fold* (e.g.
`GridSearchCV(cv=3)` on `X_tr, y_tr`), then evaluate the refit-best model on
val/test as before. This is the standard fix for exactly this instability and
is a small, contained change to `baseline.py`. Flagging rather than
implementing immediately since it needs a rerun to confirm, and the user is
mid-experiment on Puhti.

**Why this matters for the paper:** B6's significance test (SNN vs. SVM,
p≈0.05–0.06, "not statistically significant") was computed against the
*original* untuned SVM (63.3%). If the "fair, tuned" baseline is actually
*weaker* than the untuned one due to this bug, we should not swap in the
59.3% number — it would make the SNN look artificially better via a
baseline-side bug, not a real methodological improvement. Keep using 63.3%
(or the fixed nested-CV number, once available) as the SVM comparison point,
not 59.3%.

## B15 event-count sanity check (done without needing `python`)

Subject 1 mean row: `mean_events_per_trial = 40947.601`,
`mean_input_events_per_trial = 11266.768`, `mean_hidden_events_per_trial =
17556.261`, `mean_output_events_per_trial = 12124.572`.

`11266.768 + 17556.261 + 12124.572 = 40947.601` — exact match. The per-layer
breakdown sums to the combined total as designed
(`training.py::evaluate_model_with_event_breakdown`); no arithmetic bug.

## Gap found & fixed: `n_timesteps` / `svm_best_c` / `svm_best_gamma` missing from summary.csv

`pipeline.py` writes these three fields into each fold's `pipeline_params.json`
(lines 412, 466–467) but never listed them in `run_aggregate`'s `fieldnames`,
so `csv.DictWriter(..., extrasaction="ignore")` silently dropped them from
`summary.csv` even though the JSON has them. Fixed in this session
(`fbcsp_snn/pipeline.py`) — `summary.csv` will include all three once
`aggregate` is rerun. `svm_best_c`/`svm_best_gamma` are left blank in the mean
row (categorical per-fold choices, not meaningful to average) but populated
per-fold — useful for seeing whether folds are picking wildly different
`(C, γ)`, which would further confirm the val-overfitting diagnosis above.

## Corrected sanity-check command (no `python` needed)

The pasted `python -c "..."` failed with `-bash: python: command not found`
— Puhti's login node has no bare `python` on PATH outside an activated venv.
This avoids the dependency entirely:

```bash
grep -oE '"(n_timesteps|svm_best_c|svm_best_gamma)":[^,}]*' \
    Results_verify/Subject_1/fold_0/pipeline_params.json
```

## Net verdict — SUPERSEDED, see banner at top of file

~~FP32/INT8/CSP-bit numbers: consistent, no regression — verification passed.~~
~~LDA tuning: real improvement, use the new numbers in the paper.~~
~~SVM tuning: regressed baseline, do not use 59.3% in the paper.~~

All three lines above assumed `Results_verify` and the original `Results/`
ran the same pipeline. They didn't — see
[BAND_SELECTION_FINDING.md](BAND_SELECTION_FINDING.md). Corrected verdict:
`Results_verify`'s numbers (FP32 66.0%, LDA 60.6%, SVM 59.3%) are the
pipeline the paper actually describes and should be used once all 9 subjects
complete. The nested-CV robustness question for the SVM grid search is still
worth doing eventually (small-val-set hyperparameter selection is a real
methodological soft spot regardless of which bands are used) but is no longer
urgent — it's a refinement, not a correctness bug blocking the numbers.
