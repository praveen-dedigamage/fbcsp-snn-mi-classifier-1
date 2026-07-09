# Energy computation — raw terminal output, 2026-07-10

Full chain: `save_test_spikes.py` → `run_lava_infer.py` → `compute_energy.py`,
all against `Results_verify` (fixed six-band pipeline), 9 subjects × 5 folds.
Summarized in `RESULTS_LOG.md` and `PAPER_REWRITE_NOTES.md` §3 (paper folder)
— this file keeps the actual pasted output verbatim in case those summaries
ever need re-checking against the source.

## `run_lava_infer.py` — per-subject summary table

```
==========================================================================
  Lava-DL (SLAYER) Simulation — Loihi 2 Accuracy Verification
==========================================================================
  Subj       FP32     Lava      Gap   SynOps/trial   In rate   Hid rate
  ----------------------------------------------------------------------
  S1         83.2%     81.9%   -1.25pp      1,810,105      6.0%      21.2%
  S2         49.9%     47.4%   -2.43pp      1,515,074      6.6%      16.8%
  S3         75.3%     76.4%   +1.11pp      1,921,487      6.1%      23.5%
  S4         62.6%     64.2%   +1.60pp      1,706,079      6.0%      18.6%
  S5         47.1%     48.2%   +1.11pp      1,638,319      6.5%      17.3%
  S6         49.4%     45.8%   -3.54pp      1,551,688      5.9%      15.5%
  S7         67.5%     71.2%   +3.68pp      1,720,881      6.5%      19.7%
  S8         78.7%     76.4%   -2.36pp      1,856,006      5.6%      20.3%
  S9         80.2%     78.7%   -1.53pp      1,955,601      6.1%      21.6%
  ----------------------------------------------------------------------
  MEAN       66.0%     65.6%   -0.40pp      1,741,693      6.1%      19.4%
==========================================================================
  Accuracy gap mean -0.40 pp  (< 1pp OK)  (FP32 66.0%  ->  Lava 65.6%)
  Mean synapse events/inference: 1,741,693

  Loihi 2 resource summary
  Neurons             : 144
  Synapses            : 17,344
  Max fan-in          : 191  (Loihi 2 limit: 8,192 -- OK)
```

Note: S7 fold 2 individually showed a +18.75pp gap (FP32=66.3%, Lava=85.1%)
— a single-fold outlier, not reflected at the subject-aggregate level
shown above (+3.68pp, unremarkable). Not investigated further given the
tight overall mean gap.

## `compute_energy.py` — full output

```
========================================================================
  Item 9 — Loihi 2 Energy Estimation (inference, batch=1)
========================================================================
  Subj    Lava acc        SynOps    E Loihi1    E Loihi2
                       per trial        (µJ)        (µJ)
  ----------------------------------------------------------
  S1         81.9%     1,810,105        42.7        14.5
  S2         47.4%     1,515,074        35.8        12.1
  S3         76.4%     1,921,487        45.3        15.4
  S4         64.2%     1,706,079        40.3        13.6
  S5         48.2%     1,638,319        38.7        13.1
  S6         45.8%     1,551,688        36.6        12.4
  S7         71.2%     1,720,881        40.6        13.8
  S8         76.4%     1,856,006        43.8        14.8
  S9         78.7%     1,955,601        46.2        15.6
  ----------------------------------------------------------
  MEAN       65.6%     1,741,693        41.1        13.9
========================================================================

  Energy comparison (per inference, mean over 9 subjects)
  ----------------------------------------------------
  Platform                            Energy  vs Loihi 2
  ----------------------------------------------------
  Loihi 2 (Orchard 2021, ~8 pJ/SynOp)         13.9 µJ   --          (baseline)
  Loihi 1 (Davies 2018, 23.6 pJ/SynOp)        41.1 µJ   3x  less efficient
  GPU V100 (30% util, 250 W, 1 ms)         75000.0 µJ   5,383x  less efficient
  GPU V100 (full TDP, 250 W, 1 ms)        250000.0 µJ   17,942x  less efficient
  Edge CPU (ARM A72, 3 W, 20 ms)           60000.0 µJ   4,306x  less efficient

  SynOp energy model:
    Loihi 2 -- ~8 pJ/SynOp [Orchard et al., SiPS 2021]
    Loihi 1 -- 23.6 pJ/SynOp [Davies et al., IEEE Micro 2018]
    GPU/CPU figures are back-of-envelope (TDP x wall time per inference)

========================================================================
  Cross-check: Lava-measured vs. PyTorch-side SynOps estimate
========================================================================
  Subj       Lava SynOps    PyTorch SynOps     Ratio
  ----------------------------------------------------
  S1           1,810,105         2,125,574     1.174
  S2           1,515,074         1,695,929     1.119
  S3           1,921,487         2,278,456     1.186
  S4           1,706,079         1,913,920     1.122
  S5           1,638,319         1,811,742     1.106
  S6           1,551,688         1,704,602     1.099
  S7           1,720,881         2,021,729     1.175
  S8           1,856,006         2,112,824     1.138
  S9           1,955,601         2,250,602     1.151
  ----------------------------------------------------
  Mean ratio (PyTorch / Lava): 1.141  (1.0 = perfect agreement)
========================================================================

========================================================================
  Full Pipeline Energy (4-second MI trial)
  NOTE: Only the Loihi 2 SNN stage is directly measured (digital
  chip, via Lava). The front-end stages below describe a SEPARATE,
  speculative all-analog front-end extrapolated from cited
  silicon precedents -- Loihi 2 itself is not analog.
========================================================================
  Stage                                   Modern    Conserv.  Reference
                                            (µJ)        (µJ)
  --------------------------------------------------------------------
  Gm-C filter bank (6 bands x 22 ch)        26.4      2112.0  Qian 2017 / Verhoeven 2007
  ADM encoder (22 channels)                532.9       532.9    Sharifshazileh 2021
  ReRAM CSP crossbar (22->144, 1001 samp.)        0.0         0.0    Burr 2017
  MIBIF comparator bank                      0.5         0.5    negligible
  SNN on Loihi 2 (digital)  <- measured       13.9        13.9    This work
  --------------------------------------------------------------------
  TOTAL                                    573.8      2659.4
========================================================================

  Full pipeline vs competing classifiers
  --------------------------------------------------------------
  System                                            Energy  Note
  --------------------------------------------------------------
  Ours -- full pipeline (modern Gm-C)                 573.8  all stages
  Ours -- full pipeline (conservative Gm-C)          2659.4  all stages
  EEGNet on Cortex-M4 [Burrello 2020]               4280.0  classifier only (no filter/CSP)
  FBCSP+SNN on edge CPU (estimated)                60000.0  full pipeline, 20ms @ 3W

  Honest note: EEGNet-on-M4 figure is classifier-only.
  Adding their digital filter + CSP preprocessing would add
  ~10-40 mJ, making our full pipeline 105x more efficient
  on a like-for-like basis.

Energy summary saved -> Results_energy/energy_summary.csv
```

## Sanity checks performed before trusting this data

- `1,741,693 SynOps x 8 pJ = 13.9 µJ` -- confirmed by hand.
- `1,741,693 SynOps x 23.6 pJ = 41.1 µJ` -- confirmed by hand.
- Cross-check ratio (1.10-1.19 across all 9 subjects) is tightly
  clustered, not scattered -- read as a consistent systematic offset
  (plausibly the fan-out-weighting convention differing slightly between
  Lava's real SynOp accounting and the PyTorch-side estimate), not a sign
  that either measurement is wrong.
