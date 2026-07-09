# Joint CSP+SNN quantization sweep — raw per-subject data, 2026-07-08

Pulled via a one-shot awk command against each Subject's `summary.csv` mean
row (all 9 `test_acc_joint_csp{X}_snn{Y}` combinations + `test_acc_fp32` in
one pass). Used to build `main.tex`'s `tab:joint_quant` table and the
"Joint CSP+SNN Precision Sensitivity" subsection (`sec:quant`). Aggregation
script: `joint_grid.py` (scratchpad — ephemeral, reproduced here so the raw
data survives in the repo).

## Raw per-subject values (fractions, as returned by awk)

```
=== S1 === fp32=0.831944  c8s8=0.832639  c8s6=0.831945  c8s4=0.825     c6s8=0.826389  c6s6=0.825     c6s4=0.823611  c4s8=0.815278  c4s6=0.811111  c4s4=0.80625
=== S2 === fp32=0.498611  c8s8=0.495833  c8s6=0.493056  c8s4=0.491667  c6s8=0.497222  c6s6=0.494445  c6s4=0.498611  c4s8=0.490972  c4s6=0.490278  c4s4=0.484028
=== S3 === fp32=0.752778  c8s8=0.754167  c8s6=0.752778  c8s4=0.739583  c6s8=0.759722  c6s6=0.752083  c6s4=0.741667  c4s8=0.747222  c4s6=0.741667  c4s4=0.738194
=== S4 === fp32=0.625694  c8s8=0.627083  c8s6=0.624306  c8s4=0.632639  c6s8=0.627083  c6s6=0.622222  c6s4=0.6375    c4s8=0.630556  c4s6=0.624306  c4s4=0.633333
=== S5 === fp32=0.470833  c8s8=0.470833  c8s6=0.471528  c8s4=0.460416  c6s8=0.472917  c6s6=0.475695  c6s4=0.463889  c4s8=0.465972  c4s6=0.470139  c4s4=0.452778
=== S6 === fp32=0.49375   c8s8=0.49375   c8s6=0.495139  c8s4=0.497222  c6s8=0.49375   c6s6=0.49375   c6s4=0.5      c4s8=0.504861  c4s6=0.504167  c4s4=0.498611
=== S7 === fp32=0.675     c8s8=0.673611  c8s6=0.68125   c8s4=0.648611  c6s8=0.668055  c6s6=0.686806  c6s4=0.651389  c4s8=0.638194  c4s6=0.645139  c4s4=0.625
=== S8 === fp32=0.7875    c8s8=0.788194  c8s6=0.788194  c8s4=0.769444  c6s8=0.788194  c6s6=0.788889  c6s4=0.770139  c4s8=0.78125   c4s6=0.788195  c4s4=0.775695
=== S9 === fp32=0.802083  c8s8=0.802083  c8s6=0.801389  c8s4=0.797222  c6s8=0.798611  c6s6=0.795833  c6s4=0.800694  c4s8=0.8       c4s6=0.797917  c4s4=0.79375
```

(`c{X}s{Y}` = `test_acc_joint_csp{X}bit_snn{Y}bit`.)

## Aggregated (mean ± population SD across 9 subjects, %)

```
fp32   mean=65.98  std=13.56  delta_vs_fp32=+0.00
c8s8   mean=65.98  std=13.62  delta_vs_fp32=+0.00
c8s6   mean=66.00  std=13.62  delta_vs_fp32=+0.02
c8s4   mean=65.13  std=13.30  delta_vs_fp32=-0.85
c6s8   mean=65.91  std=13.48  delta_vs_fp32=-0.07
c6s6   mean=65.94  std=13.41  delta_vs_fp32=-0.04
c6s4   mean=65.42  std=13.15  delta_vs_fp32=-0.56
c4s8   mean=65.27  std=13.21  delta_vs_fp32=-0.71
c4s6   mean=65.25  std=13.13  delta_vs_fp32=-0.73
c4s4   mean=64.53  std=13.28  delta_vs_fp32=-1.45
```

Worst case (CSP=4-bit, SNN=4-bit, the most aggressive joint precision
reduction tested): mean accuracy 64.53% vs. FP32's 65.98%, a 1.45pp
degradation — this is the number used in the abstract, Introduction
contribution bullet, `sec:quant`, and the Conclusion.

This table in `main.tex` (`tab:joint_quant`) rounds `c8s4`→8-bit-CSP/SNN=6-bit
row etc. to one decimal place; see the table itself for the exact rounded
values used in print.
