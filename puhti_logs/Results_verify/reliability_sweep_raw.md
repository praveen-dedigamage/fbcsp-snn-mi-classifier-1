# Reliability sweep — raw data, 2026-07-09/10

Two things captured here that only otherwise exist baked into
`plot_reliability_sweep.py`'s hardcoded arrays: the per-fold sanity-check
JSON (Subject_1/fold_0) and the full cross-subject aggregation output.
Job `35412677` (final, validated run — see `BAND_SELECTION_FINDING.md` and
`RESULTS_LOG.md` for the two bugs found and fixed before this run).

## Cross-subject aggregation (`aggregate_reliability.py --results-dir Results_verify`)

```
clean_acc,mean=0.659799,std=0.135601,n=9

sweep,severity,acc_mean,acc_std,n_subjects
csp_weight_noise,0.0,0.659799,0.135601,9
csp_weight_noise,0.05,0.651073,0.133405,9
csp_weight_noise,0.1,0.621200,0.125892,9
csp_weight_noise,0.2,0.455093,0.107988,9
csp_weight_noise,0.3,0.329171,0.056945,9
snn_weight_noise,0.0,0.659799,0.135601,9
snn_weight_noise,0.05,0.648966,0.133075,9
snn_weight_noise,0.1,0.624660,0.127759,9
snn_weight_noise,0.2,0.560390,0.113909,9
snn_weight_noise,0.3,0.502319,0.097862,9
beta_noise,0.0,0.659799,0.135601,9
beta_noise,0.05,0.640444,0.134792,9
beta_noise,0.1,0.621130,0.134361,9
beta_noise,0.2,0.586292,0.130963,9
beta_noise,0.3,0.559174,0.126147,9
filter_bank_noise,0.0,0.659799,0.135601,9
filter_bank_noise,0.05,0.624850,0.125628,9
filter_bank_noise,0.1,0.560475,0.108638,9
filter_bank_noise,0.2,0.489340,0.093748,9
filter_bank_noise,0.3,0.440988,0.086919,9
ea_whitener_noise,0.0,0.659799,0.135601,9
ea_whitener_noise,0.05,0.657712,0.135689,9
ea_whitener_noise,0.1,0.654128,0.135033,9
ea_whitener_noise,0.2,0.642311,0.131820,9
ea_whitener_noise,0.3,0.621458,0.126107,9
znorm_noise,0.0,0.659799,0.135601,9
znorm_noise,0.05,0.654471,0.133419,9
znorm_noise,0.1,0.638156,0.129170,9
znorm_noise,0.2,0.548781,0.112912,9
znorm_noise,0.3,0.387118,0.074806,9
encoder_threshold_noise,0.0,0.659799,0.135601,9
encoder_threshold_noise,0.05,0.659749,0.135768,9
encoder_threshold_noise,0.1,0.659896,0.135678,9
encoder_threshold_noise,0.2,0.659958,0.135722,9
encoder_threshold_noise,0.3,0.660046,0.135822,9
encoder_adaptation_noise,0.0,0.659799,0.135601,9
encoder_adaptation_noise,0.05,0.450532,0.087970,9
encoder_adaptation_noise,0.1,0.391130,0.058129,9
encoder_adaptation_noise,0.2,0.350652,0.038534,9
encoder_adaptation_noise,0.3,0.330795,0.031018,9
joint_noise_all_sources,0.0,0.659799,0.135601,9
joint_noise_all_sources,0.05,0.417670,0.072458,9
joint_noise_all_sources,0.1,0.345733,0.035930,9
joint_noise_all_sources,0.2,0.289317,0.014089,9
joint_noise_all_sources,0.3,0.264008,0.005739,9
```

This is exactly the data hardcoded into `plot_reliability_sweep.py`'s
`DATA` dict — kept here too as the actual terminal output, independent of
the plotting script, in case the script's copy is ever edited/reordered
and drifts from the source.

## Per-fold sanity check (Subject_1/fold_0/reliability_results.json)

Confirmed both bug fixes (filter_bank_noise, ea_whitener_noise) hold at
the single-fold level too, not just in the cross-subject aggregate — see
full JSON pasted into the conversation 2026-07-09, job `35412677`. Key
values: clean acc 0.8229, `filter_bank_noise` 0.823→0.574 (graceful),
`ea_whitener_noise` 0.823→0.746 (graceful, was 0.823→0.250 before the fix).
Not re-transcribed in full here since the cross-subject aggregate above is
the one actually used in the paper; this file's JSON is still on Puhti at
`Results_verify/Subject_1/fold_0/reliability_results.json` if needed again.
